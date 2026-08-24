#!/usr/bin/env python3
"""Label selected normalized decision states with an AP4Fed single-agent teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Sequence

from archive_extractor import (
    FEATURE_COLUMNS,
    LABEL_RECORD_COLUMNS,
    STATE_ID_COLUMNS,
    ExperimentSource,
    extract_experiment,
)
from build_state_bank import ExtractionJob, load_source_list


SELECTION_SCHEMA_VERSION = 1
ATTEMPT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
LABEL_KIND = "offline_teacher_query"
DEFAULT_TEACHER_POLICY = "Single AI-Agent (Few-Shot)"
DEFAULT_TEACHER_MODEL = "deepseek-r1:8b"
DEFAULT_OPTIONS = {"temperature": 1.0, "top_p": 0.9, "num_ctx": 8192}
DECISION_KEYS = (
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
)
SIGNATURE_RE = re.compile(
    r"\bCS\s*=\s*(ON|OFF)\s*;\s*MC\s*=\s*(ON|OFF)\s*;\s*HDH\s*=\s*(ON|OFF)\b",
    flags=re.IGNORECASE,
)
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MODEL_DIGEST_RE = re.compile(r"(?:sha256:)?[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class SelectionRecord:
    record_id: str
    attempts: int
    purpose: str


@dataclass(frozen=True)
class SelectionPlan:
    selection_id: str
    selection_rule: str
    records: tuple[SelectionRecord, ...]
    state_bank_sha256: str | None
    sha256: str

    @property
    def query_budget(self) -> int:
        return sum(record.attempts for record in self.records)


@dataclass(frozen=True)
class SelectedState:
    selection: SelectionRecord
    job: ExtractionJob
    state: dict[str, str]
    audit: dict[str, object]
    config: dict[str, object]
    run_csv: Path
    decision_idx: int

    @property
    def record_id(self) -> str:
        return self.selection.record_id

    @property
    def run_id(self) -> str:
        return self.state["run_id"]


@dataclass(frozen=True)
class TeacherModules:
    prompting: ModuleType
    metrics: ModuleType
    client: ModuleType
    prompting_path: Path
    metrics_path: Path
    client_path: Path


TeacherCall = Callable[[str, str, list[str], dict[str, object]], str]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _stringify_row(row: dict[str, object], columns: Sequence[str]) -> dict[str, str]:
    return {
        column: "" if row.get(column) is None else str(row.get(column, ""))
        for column in columns
    }


def load_selection(path: Path) -> SelectionPlan:
    """Read the predeclared query plan and reject ambiguous or unsafe entries.

    Repeated calls for one state are permitted only when that state is explicitly
    marked as part of the calibration subset.
    """
    document = _read_json(path)
    if document.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"{path}: expected schema_version {SELECTION_SCHEMA_VERSION}")
    selection_id = str(document.get("selection_id", "")).strip()
    if not SAFE_ID_RE.fullmatch(selection_id):
        raise ValueError(
            f"{path}: selection_id must match {SAFE_ID_RE.pattern!r}"
        )
    selection_rule = str(document.get("selection_rule", "")).strip()
    if not selection_rule:
        raise ValueError(f"{path}: selection_rule must be a non-empty predeclared rule")
    state_bank_sha256 = document.get("state_bank_sha256")
    if state_bank_sha256 is not None:
        state_bank_sha256 = str(state_bank_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", state_bank_sha256):
            raise ValueError(f"{path}: state_bank_sha256 must be a full SHA-256 digest")
    raw_records = document.get("records", [])
    raw_groups = document.get("groups", [])
    if not isinstance(raw_records, list):
        raise ValueError(f"{path}: records must be a list")
    if not isinstance(raw_groups, list):
        raise ValueError(f"{path}: groups must be a list")
    if not raw_records and not raw_groups:
        raise ValueError(f"{path}: records or groups must be non-empty")

    records: list[SelectionRecord] = []
    seen: set[str] = set()

    def add_record(
        record_id: str,
        attempts: object,
        purpose: str,
        location: str,
    ) -> None:
        if not record_id or "::d" not in record_id:
            raise ValueError(f"{path}: {location} has an invalid record_id")
        if record_id in seen:
            raise ValueError(f"{path}: duplicate record_id {record_id!r}")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError(f"{path}: {location} attempts must be an integer >= 1")
        if purpose not in {"primary", "calibration"}:
            raise ValueError(
                f"{path}: {location} purpose must be primary or calibration"
            )
        if attempts > 1 and purpose != "calibration":
            raise ValueError(
                f"{path}: {location} repeats queries without purpose=calibration"
            )
        seen.add(record_id)
        records.append(SelectionRecord(
            record_id=record_id,
            attempts=attempts,
            purpose=purpose,
        ))

    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        record_id = str(raw.get("record_id", "")).strip()
        attempts = raw.get("attempts", 1)
        purpose = str(raw.get("purpose", "primary")).strip().lower()
        add_record(record_id, attempts, purpose, f"record {index}")

    def inclusive_range(raw: dict[str, object], key: str, location: str) -> range:
        value = raw.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: {location} {key} must be an object")
        start = value.get("start")
        end = value.get("end")
        invalid = (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
        )
        if invalid:
            raise ValueError(
                f"{path}: {location} {key} needs integer 1 <= start <= end"
            )
        return range(start, end + 1)

    for index, raw in enumerate(raw_groups, start=1):
        location = f"group {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: {location} must be an object")
        source_id = str(raw.get("source_id", "")).strip().rstrip("/")
        if not source_id or "::" in source_id:
            raise ValueError(f"{path}: {location} has an invalid source_id")
        runs = inclusive_range(raw, "run_range", location)
        decisions = inclusive_range(raw, "decision_range", location)
        attempts = raw.get("attempts", 1)
        purpose = str(raw.get("purpose", "primary")).strip().lower()
        raw_exclusions = raw.get("exclude", [])
        if not isinstance(raw_exclusions, list):
            raise ValueError(f"{path}: {location} exclude must be a list")
        exclusions: set[tuple[int, int]] = set()
        for exclusion_index, exclusion in enumerate(raw_exclusions, start=1):
            if not isinstance(exclusion, dict):
                raise ValueError(
                    f"{path}: {location} exclusion {exclusion_index} must be an object"
                )
            run_number = exclusion.get("run")
            decision_idx = exclusion.get("decision")
            pair = (run_number, decision_idx)
            if (
                isinstance(run_number, bool)
                or isinstance(decision_idx, bool)
                or not isinstance(run_number, int)
                or not isinstance(decision_idx, int)
                or run_number not in runs
                or decision_idx not in decisions
                or pair in exclusions
            ):
                raise ValueError(
                    f"{path}: {location} exclusion {exclusion_index} is invalid"
                )
            exclusions.add(pair)
        for run_number in runs:
            for decision_idx in decisions:
                if (run_number, decision_idx) in exclusions:
                    continue
                add_record(
                    f"{source_id}/r{run_number}::d{decision_idx}",
                    attempts,
                    purpose,
                    location,
                )
    return SelectionPlan(
        selection_id=selection_id,
        selection_rule=selection_rule,
        records=tuple(records),
        state_bank_sha256=state_bank_sha256,
        sha256=_sha256(path),
    )


def _read_states(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = list(STATE_ID_COLUMNS + FEATURE_COLUMNS)
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: state columns do not match normalized schema v5")
        return list(reader)


def _validate_recreated_dataset(
    job: ExtractionJob,
    output: Path,
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
    """Re-extract one configuration and compare it with the retained state bank.

    This catches changed source CSVs, configuration drift, and stale normalized
    states before any teacher-query budget is spent.
    """
    states_path = output / "decision_states.csv"
    audit_path = output / "audit.json"
    if not states_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"{output}: normalized state or audit artifact is missing")
    states = _read_states(states_path)
    audit = _read_json(audit_path)
    if audit.get("schema_version") != 5:
        raise ValueError(f"{audit_path}: expected normalized schema version 5")
    if audit.get("source_id") != job.experiment.source_id:
        raise ValueError(f"{audit_path}: source_id does not match the source list")

    recreated = extract_experiment(job.experiment)
    expected_states = [
        _stringify_row(row, STATE_ID_COLUMNS + FEATURE_COLUMNS)
        for row in recreated.states
    ]
    if states != expected_states:
        raise ValueError(f"{states_path}: states differ from current source files")
    if audit.get("config_sha256") != recreated.audit.get("config_sha256"):
        raise ValueError(f"{audit_path}: configuration hash differs from source")
    if audit.get("run_registry") != recreated.audit.get("run_registry"):
        raise ValueError(f"{audit_path}: run inventory or source hashes differ")
    config = _read_json(job.experiment.source.resolve() / "config.json")
    return states, audit, config


def validate_selected_states(
    sources_file: Path,
    state_bank_root: Path,
    selection_file: Path,
) -> tuple[SelectionPlan, list[SelectedState], dict[str, object]]:
    """Resolve selected record IDs only after validating the complete state bank.

    The returned summary makes the planned coverage and query budget visible
    without contacting the teacher.
    """
    plan = load_selection(selection_file)
    jobs = load_source_list(sources_file, state_bank_root)
    wanted = {record.record_id: record for record in plan.records}
    found: dict[str, SelectedState] = {}
    state_bank_artifacts = []
    state_count = 0

    for job in jobs:
        states, audit, config = _validate_recreated_dataset(job, job.output)
        state_bank_artifacts.append({
            "output_name": job.output.name,
            "decision_states_sha256": _sha256(job.output / "decision_states.csv"),
            "audit_sha256": _sha256(job.output / "audit.json"),
        })
        state_count += len(states)
        source_id = job.experiment.source_id
        run_registry = audit.get("run_registry")
        if not isinstance(run_registry, dict):
            raise ValueError(f"{job.output / 'audit.json'}: run_registry must be an object")
        for state in states:
            record_id = state["record_id"]
            if record_id not in wanted:
                continue
            if record_id in found:
                raise ValueError(f"duplicate selected state {record_id!r}")
            match = re.fullmatch(r"(.+/r(\d+))::d(\d+)", record_id)
            if not match:
                raise ValueError(f"{record_id!r}: unsupported normalized record ID")
            run_id, run_number, decision_value = match.groups()
            if not run_id.startswith(source_id + "/") or state["run_id"] != run_id:
                raise ValueError(f"{record_id!r}: run identity does not match its audit")
            if run_id not in run_registry:
                raise ValueError(f"{record_id!r}: run is absent from its audit")
            run_csv = job.experiment.source.resolve() / f"r{run_number}.csv"
            if not run_csv.is_file():
                raise FileNotFoundError(run_csv)
            found[record_id] = SelectedState(
                selection=wanted[record_id],
                job=job,
                state=state,
                audit=audit,
                config=config,
                run_csv=run_csv,
                decision_idx=int(decision_value),
            )

    missing = [record.record_id for record in plan.records if record.record_id not in found]
    if missing:
        raise ValueError(f"selection contains states absent from the bank: {missing}")
    selected = [found[record.record_id] for record in plan.records]
    state_bank_sha256 = _sha256_bytes(json.dumps(
        state_bank_artifacts,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))
    if plan.state_bank_sha256 and plan.state_bank_sha256 != state_bank_sha256:
        raise ValueError(
            "selection state_bank_sha256 differs from the validated state bank"
        )
    summary = {
        "source_manifest_sha256": _sha256(sources_file),
        "selection_sha256": plan.sha256,
        "state_bank_sha256": state_bank_sha256,
        "source_configurations": len(jobs),
        "state_bank_rows": state_count,
        "selected_states": len(selected),
        "query_budget": plan.query_budget,
        "workloads": dict(Counter(
            str(state.audit.get("state_source", {}).get("workload", ""))
            for state in selected
        )),
        "source_policies": dict(Counter(
            str(state.audit.get("state_source", {}).get("policy", ""))
            for state in selected
        )),
        "configurations": dict(Counter(state.job.output.name for state in selected)),
        "start_status": dict(Counter(
            "cold" if state.decision_idx == 1 else "warm" for state in selected
        )),
        "previous_ap_contexts": dict(Counter(
            "/".join((
                state.state["prev_cs"],
                state.state["prev_mc"],
                state.state["prev_hdh"],
            ))
            for state in selected
        )),
        "selected_state_purposes": dict(Counter(
            state.selection.purpose for state in selected
        )),
        "query_budget_by_purpose": {
            purpose: sum(
                state.selection.attempts
                for state in selected
                if state.selection.purpose == purpose
            )
            for purpose in sorted({
                state.selection.purpose for state in selected
            })
        },
    }
    return plan, selected, summary


def _read_run_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not rows or "FL Round" not in fields:
        raise ValueError(f"{path}: invalid archived run CSV")
    return fields, rows


def _write_rows_through(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, str]],
    round_idx: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if int(row["FL Round"]) <= round_idx:
                writer.writerow(row)


def materialize_teacher_view(state: SelectedState, root: Path) -> tuple[int | None, Path]:
    """Recreate the metric files that the live teacher could see at this decision.

    The first decision sees the growing round-one CSV. Later decisions see only
    completed cumulative snapshots from rounds that precede the decision.
    """
    fields, rows = _read_run_rows(state.run_csv)
    performance = root / "performance"
    if state.decision_idx == 1:
        _write_rows_through(
            performance / "FLwithAP_performance_metrics.csv",
            fields,
            rows,
            1,
        )
        return None, root
    for round_idx in range(1, state.decision_idx):
        _write_rows_through(
            performance / f"FLwithAP_performance_metrics_round{round_idx}.csv",
            fields,
            rows,
            round_idx,
        )
    return state.decision_idx - 1, root


def load_teacher_modules() -> TeacherModules:
    """Load the Docker teacher implementation used by substantive experiments.

    Refusing modules already imported from another backend prevents an offline
    run from silently mixing the Local and Docker prompt implementations.
    """
    docker_root = Path(__file__).resolve().parents[1] / "Docker"
    docker_text = str(docker_root)
    inserted = docker_text not in sys.path
    if inserted:
        sys.path.insert(0, docker_text)
    try:
        for module_name in ("agent_prompting", "adaptation_metrics", "ollama_client"):
            existing = sys.modules.get(module_name)
            if existing is None:
                continue
            module_path = Path(existing.__file__).resolve()
            if module_path.parent != docker_root:
                raise RuntimeError(
                    f"{module_name} is already loaded from non-Docker path {module_path}"
                )
        prompting = importlib.import_module("agent_prompting")
        metrics = importlib.import_module("adaptation_metrics")
        client = importlib.import_module("ollama_client")
    finally:
        if inserted:
            sys.path.remove(docker_text)
    return TeacherModules(
        prompting=prompting,
        metrics=metrics,
        client=client,
        prompting_path=docker_root / "agent_prompting.py",
        metrics_path=docker_root / "adaptation_metrics.py",
        client_path=docker_root / "ollama_client.py",
    )


@contextmanager
def _offline_prompt_runtime(prompting: ModuleType, metrics_root: Path):
    """Point prompt construction at reconstructed files and the supplied config."""
    original = prompting._runtime_attr

    def offline_attr(name, default):
        if name == "config_file":
            return str(metrics_root / "__no_runtime_config__.json")
        if name == "USE_RAG":
            return True
        return default

    prompting._runtime_attr = offline_attr
    try:
        yield
    finally:
        prompting._runtime_attr = original


def build_teacher_prompt(
    modules: TeacherModules,
    state: SelectedState,
    teacher_policy: str,
    teacher_model: str,
    metrics_root: Path,
    last_round: int | None,
) -> tuple[str, dict[str, object]]:
    """Build the normal AP4Fed single-agent prompt for one archived state.

    Only the teacher policy, model identity, previous decisions, and reconstructed
    metric location are substituted; the production prompt builder remains in use.
    """
    config = json.loads(json.dumps(state.config))
    config["adaptation"] = teacher_policy
    config["LLM"] = teacher_model
    ap_previous = {
        "client_selector": state.state["prev_cs"],
        "message_compressor": state.state["prev_mc"],
        "heterogeneous_data_handler": state.state["prev_hdh"],
    }
    aggregate: dict[str, object] = {}
    if last_round is not None:
        _, latest = modules.metrics._sa_latest_round_csv(metrics_root=metrics_root)
        if latest is None:
            raise ValueError(f"{state.record_id}: reconstructed snapshot is missing")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import pandas as pd

        aggregate = modules.metrics._sa_aggregate_round(pd.read_csv(latest))
    mode = modules.prompting._sa_mode_from_policy(teacher_policy)
    with _offline_prompt_runtime(modules.prompting, metrics_root):
        prompt = modules.prompting._sa_build_prompt(
            mode,
            config,
            last_round,
            aggregate,
            ap_previous,
            metrics_root=metrics_root,
        )
    return prompt, aggregate


def _response_object(raw: str) -> dict[str, object] | None:
    text = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()
    candidates = [match.group(0) for match in re.finditer(r"\{.*?\}", text, re.DOTALL)]
    for candidate in reversed(candidates + [text]):
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _recognised_decision(value: object) -> bool:
    if isinstance(value, bool):
        return True
    return str(value).strip().upper() in {
        "ON", "OFF", "ENABLED", "DISABLED", "TRUE", "FALSE", "1", "0"
    }


def parse_teacher_response(
    modules: TeacherModules,
    raw: str,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Parse a response with AP4Fed while auditing its usable evidence.

    A response becomes a label only when it contains recognizable JSON decisions
    or the required decision signature; arbitrary prose is retained as invalid.
    """
    obj = _response_object(raw)
    signature = SIGNATURE_RE.search(raw)
    json_decisions_valid = bool(
        obj is not None
        and all(key in obj and _recognised_decision(obj[key]) for key in DECISION_KEYS)
    )
    if not json_decisions_valid and signature is None:
        return None, {
            "parser_status": "invalid",
            "json_decisions_valid": False,
            "signature_status": "missing",
        }

    decisions, _, parser_result = modules.prompting._sa_parse_output(raw)
    parsed = {key: decisions.get(key, "OFF") for key in DECISION_KEYS}
    if "selection_value" in decisions:
        parsed["selection_value"] = decisions["selection_value"]

    signature_status = "missing"
    if signature is not None:
        signature_values = tuple(value.upper() for value in signature.groups())
        json_values = None
        if json_decisions_valid and obj is not None:
            json_values = tuple(str(obj[key]).strip().upper() for key in DECISION_KEYS)
        signature_status = (
            "present_matching"
            if json_values is None or json_values == signature_values
            else "present_overrode_json"
        )
    return parsed, {
        "parser_status": "json" if json_decisions_valid else "signature",
        "parser_result": bool(parser_result),
        "json_decisions_valid": json_decisions_valid,
        "signature_status": signature_status,
    }


def _valid_selection_values(config: dict[str, object]) -> list[int]:
    """Return CPU thresholds that exclude someone while retaining two clients."""
    cpus = []
    for client in config.get("client_details", []):
        if not isinstance(client, dict):
            continue
        try:
            cpu = int(client.get("cpu", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cpu > 0:
            cpus.append(cpu)
    if len(cpus) < 2:
        return []
    return [
        threshold
        for threshold in range(max(cpus))
        if sum(cpu > threshold for cpu in cpus) >= 2
        and sum(cpu <= threshold for cpu in cpus) >= 1
    ]


def apply_selector_guardrail(
    decisions: dict[str, object],
    config: dict[str, object],
) -> tuple[dict[str, object], str]:
    """Apply the live Client Selector safety rule to a parsed teacher decision.

    An invalid threshold is replaced with the smallest safe value. Client
    selection is disabled when the configuration has no safe threshold.
    """
    applied = dict(decisions)
    if applied.get("client_selector") != "ON":
        applied.pop("selection_value", None)
        return applied, "unchanged"
    valid = _valid_selection_values(config)
    if not valid:
        applied["client_selector"] = "OFF"
        applied.pop("selection_value", None)
        return applied, "client_selector_disabled"
    try:
        candidate = int(applied.get("selection_value"))
    except (TypeError, ValueError):
        candidate = None
    if candidate in valid:
        applied["selection_value"] = candidate
        return applied, "unchanged"
    applied["selection_value"] = valid[0]
    return applied, "selection_value_adjusted"


def _attempt_id(run_id: str, record_id: str, sample_index: int) -> str:
    state_digest = _sha256_bytes(record_id.encode("utf-8"))[:16]
    return f"{run_id}-{state_digest}-s{sample_index:03d}"


def _load_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    attempts = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            attempt_id = value.get("attempt_id") if isinstance(value, dict) else None
            if not attempt_id or attempt_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate attempt_id")
            seen.add(attempt_id)
            attempts.append(value)
    return attempts


def _validate_prior_attempts(
    output: Path,
    attempts: Sequence[dict[str, object]],
    planned_ids: set[str],
) -> None:
    """Verify resumed attempts still belong to the plan and retain intact evidence."""
    for attempt in attempts:
        attempt_id = str(attempt["attempt_id"])
        if attempt_id not in planned_ids:
            raise ValueError(f"{attempt_id}: attempt is absent from the current plan")
        status = attempt.get("status")
        if status not in {"success", "invalid_response", "error"}:
            raise ValueError(f"{attempt_id}: unsupported attempt status {status!r}")
        raw_name = attempt.get("raw_response_file")
        if status in {"success", "invalid_response"} and not raw_name:
            raise ValueError(f"{attempt_id}: completed response has no raw artifact")
        if not raw_name:
            continue
        raw_path = (output / str(raw_name)).resolve()
        if output not in raw_path.parents:
            raise ValueError(f"{attempt_id}: raw response path leaves the run directory")
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if attempt.get("raw_response_sha256") != _sha256(raw_path):
            raise ValueError(f"{attempt_id}: raw response hash differs from its audit")
        if status == "success" and not isinstance(attempt.get("label"), dict):
            raise ValueError(f"{attempt_id}: successful attempt has no label")


def _append_attempt(path: Path, attempt: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(attempt, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_atomic(path: Path, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("teacher response must be text")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_labels(path: Path, attempts: Sequence[dict[str, object]]) -> None:
    labels = [attempt["label"] for attempt in attempts if attempt.get("status") == "success"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_RECORD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(labels)
    temporary.replace(path)


def _git_identity(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True,
        ).strip()

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(status)}


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_model_identity(
    base_urls: Sequence[str],
    model: str,
    explicit_digest: str | None = None,
) -> dict[str, object]:
    """Resolve the exact served model digest before starting a labelling run.

    The local server inventory is preferred. A caller may instead provide a full
    digest when an equivalent serving environment cannot expose that inventory.
    """
    def server_version(base_url: str) -> str | None:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/version", timeout=15,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
            return str(value.get("version") or "") or None
        except Exception:
            return None

    if explicit_digest:
        if not MODEL_DIGEST_RE.fullmatch(explicit_digest):
            raise ValueError("--model-digest must be a full SHA-256 digest")
        return {
            "name": model,
            "digest": explicit_digest,
            "source": "command-line",
            "server_version": server_version(base_urls[0]) if base_urls else None,
        }
    errors = []
    for base_url in base_urls:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/tags", timeout=15,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload.get("models", []):
                if item.get("name") == model or item.get("model") == model:
                    digest = item.get("digest")
                    if not isinstance(digest, str) or not MODEL_DIGEST_RE.fullmatch(digest):
                        raise ValueError(f"{base_url}: model has no full SHA-256 digest")
                    return {
                        "name": model,
                        "digest": digest,
                        "size": item.get("size"),
                        "modified_at": item.get("modified_at"),
                        "details": item.get("details"),
                        "server_version": server_version(base_url),
                        "source": f"{base_url.rstrip('/')}/api/tags",
                    }
            errors.append(f"{base_url}: model not found")
        except Exception as error:
            errors.append(f"{base_url}: {type(error).__name__}: {error}")
    raise RuntimeError("teacher model identity unavailable: " + "; ".join(errors))


def _default_teacher_call(modules: TeacherModules) -> TeacherCall:
    def call(model: str, prompt: str, base_urls: list[str], options: dict[str, object]) -> str:
        return modules.client._sa_call_ollama(
            model, prompt, base_urls, force_json=True, options=options,
        )

    return call


def label_selected_states(
    *,
    sources_file: Path,
    state_bank_root: Path,
    selection_file: Path,
    output: Path,
    run_id: str,
    teacher_policy: str = DEFAULT_TEACHER_POLICY,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    base_urls: Sequence[str] = ("http://localhost:11434",),
    options: dict[str, object] | None = None,
    model_identity: dict[str, object] | None = None,
    teacher_call: TeacherCall | None = None,
    resume: bool = False,
    allow_dirty: bool = False,
) -> dict[str, int]:
    """Run the frozen query plan and write an auditable, resumable label archive.

    Each model call is surrounded by durable provenance: an in-flight marker is
    written before the call, the raw response is retained, and only valid parsed
    decisions are projected into the normalized label table.
    """
    if not SAFE_ID_RE.fullmatch(run_id):
        raise ValueError(f"run_id must match {SAFE_ID_RE.pattern!r}")
    plan, states, validation = validate_selected_states(
        sources_file, state_bank_root, selection_file,
    )
    modules = load_teacher_modules()
    repo = Path(__file__).resolve().parents[1]
    git = _git_identity(repo)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("refusing a canonical label run from a dirty AP4Fed checkout")
    actual_options = dict(DEFAULT_OPTIONS if options is None else options)
    identity = model_identity or resolve_model_identity(base_urls, teacher_model)
    digest = identity.get("digest")
    if not isinstance(digest, str) or not MODEL_DIGEST_RE.fullmatch(digest):
        raise ValueError("teacher model identity must contain a full SHA-256 digest")

    output = output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("label-run output must be outside the AP4Fed repository")
    attempts_path = output / "attempts.jsonl"
    labels_path = output / "labels.csv"
    manifest_path = output / "run_manifest.json"
    raw_root = output / "raw_responses"
    inflight_root = output / "inflight"
    if resume and not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: cannot resume without a run manifest")
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"{output}: use --resume to continue an existing run")
    output.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    inflight_root.mkdir(exist_ok=True)

    prior_attempts = _load_attempts(attempts_path)
    planned_ids = {
        _attempt_id(run_id, state.record_id, sample_index)
        for state in states
        for sample_index in range(1, state.selection.attempts + 1)
    }
    _validate_prior_attempts(output, prior_attempts, planned_ids)
    completed_ids = {str(attempt["attempt_id"]) for attempt in prior_attempts}
    prompt_version = {
        "agent_prompting_sha256": _sha256(modules.prompting_path),
        "adaptation_metrics_sha256": _sha256(modules.metrics_path),
        "ollama_client_sha256": _sha256(modules.client_path),
    }
    current_command = " ".join(shlex.quote(argument) for argument in sys.argv)
    current_environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "exact_command": current_command,
        "resume_commands": [],
        "producing_code": git,
        "inputs": {
            **validation,
            "selection_id": plan.selection_id,
            "selection_rule": plan.selection_rule,
        },
        "teacher": {
            "policy": teacher_policy,
            "mode": modules.prompting._sa_mode_from_policy(teacher_policy),
            "model": teacher_model,
            "model_identity": identity,
            "base_urls": list(base_urls),
            "sampling_options": actual_options,
            "prompt_version": prompt_version,
        },
        "environment": current_environment,
        "outputs": {
            "attempts": "attempts.jsonl",
            "labels": "labels.csv",
            "raw_responses": "raw_responses/",
            "inflight_queries": "inflight/",
        },
        "result": None,
    }
    if resume and manifest_path.exists():
        existing = _read_json(manifest_path)
        comparable = (
            existing.get("run_id") == run_id
            and existing.get("inputs", {}).get("selection_sha256") == plan.sha256
            and existing.get("inputs", {}).get("source_manifest_sha256")
            == validation["source_manifest_sha256"]
            and existing.get("producing_code") == git
            and existing.get("teacher", {}).get("policy") == teacher_policy
            and existing.get("teacher", {}).get("model") == teacher_model
            and existing.get("teacher", {}).get("model_identity") == identity
            and existing.get("teacher", {}).get("base_urls") == list(base_urls)
            and existing.get("teacher", {}).get("sampling_options") == actual_options
            and existing.get("teacher", {}).get("prompt_version") == prompt_version
        )
        if not comparable:
            raise ValueError(f"{manifest_path}: resume settings differ from the existing run")
        manifest["started_at"] = existing.get("started_at")
        manifest["exact_command"] = existing.get("exact_command")
        manifest["environment"] = existing.get("environment")
        resume_commands = existing.get("resume_commands", [])
        if not isinstance(resume_commands, list):
            raise ValueError(f"{manifest_path}: resume_commands must be a list")
        manifest["resume_commands"] = [
            *resume_commands,
            {
                "at": _utc_now(),
                "command": current_command,
                "environment": current_environment,
            },
        ]

    for marker in sorted(inflight_root.glob("*.json")):
        marker_value = _read_json(marker)
        marker_id = str(marker_value.get("attempt_id", ""))
        if marker_id in completed_ids:
            marker.unlink()
            continue
        raise RuntimeError(
            f"{marker}: unresolved in-flight query; refusing an automatic repeat"
        )
    _write_manifest(manifest_path, manifest)

    call_teacher = teacher_call or _default_teacher_call(modules)
    new_attempts: list[dict[str, object]] = []
    for state in states:
        for sample_index in range(1, state.selection.attempts + 1):
            attempt_id = _attempt_id(run_id, state.record_id, sample_index)
            if attempt_id in completed_ids:
                continue
            raw_path = raw_root / f"{attempt_id}.txt"
            if raw_path.exists():
                raise FileExistsError(
                    f"{raw_path}: raw response exists without a completed attempt record"
                )
            started_at = _utc_now()
            started_clock = time.perf_counter()
            prompt: str | None = None
            marker_path: Path | None = None
            source_run = state.audit["run_registry"][state.run_id]
            common = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "selection_id": plan.selection_id,
                "record_id": state.record_id,
                "run_id": state.run_id,
                "sample_index": sample_index,
                "attempt_purpose": state.selection.purpose,
                "query_started_at": started_at,
                "teacher_policy": teacher_policy,
                "teacher_model": teacher_model,
                "teacher_model_digest": identity["digest"],
                "sampling_options": actual_options,
                "source_id": state.audit["source_id"],
                "source_policy": state.audit["state_source"]["policy"],
                "workload": state.audit["state_source"]["workload"],
                "source_config_sha256": state.audit["config_sha256"],
                "source_csv_sha256": source_run["source_sha256"],
                "counterfactual_outcome_warning": (
                    "archived downstream outcomes are not attributable to this queried action"
                ),
            }
            try:
                with tempfile.TemporaryDirectory(prefix="ap4fed-teacher-view-") as temporary:
                    metrics_root = Path(temporary)
                    last_round, _ = materialize_teacher_view(state, metrics_root)
                    prompt, _ = build_teacher_prompt(
                        modules,
                        state,
                        teacher_policy,
                        teacher_model,
                        metrics_root,
                        last_round,
                    )
                    marker_candidate = inflight_root / f"{attempt_id}.json"
                    _write_json_exclusive(marker_candidate, {
                        **common,
                        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                        "prompt_version": prompt_version,
                    })
                    marker_path = marker_candidate
                    raw = call_teacher(
                        teacher_model, prompt, list(base_urls), actual_options,
                    )
                _write_text_atomic(raw_path, raw)
                parsed, parser_audit = parse_teacher_response(modules, raw)
                completed_at = _utc_now()
                attempt = {
                    **common,
                    "query_completed_at": completed_at,
                    "latency_seconds": time.perf_counter() - started_clock,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "prompt_version": prompt_version,
                    "raw_response_file": str(raw_path.relative_to(output)),
                    "raw_response_sha256": _sha256(raw_path),
                    **parser_audit,
                }
                if parsed is None:
                    attempt["status"] = "invalid_response"
                else:
                    applied, guardrail_result = apply_selector_guardrail(parsed, state.config)
                    label = {
                        "record_id": state.record_id,
                        "attempt_id": attempt_id,
                        "label_kind": LABEL_KIND,
                        "teacher_policy": teacher_policy,
                        "teacher_model": teacher_model,
                        "y_cs_applied": applied["client_selector"],
                        "selection_value": (
                            applied.get("selection_value", "")
                            if applied["client_selector"] == "ON" else ""
                        ),
                        "y_mc_applied": applied["message_compressor"],
                        "y_hdh_applied": applied["heterogeneous_data_handler"],
                    }
                    attempt.update({
                        "status": "success",
                        "parsed_decisions": parsed,
                        "applied_decisions": applied,
                        "guardrail_result": guardrail_result,
                        "label": label,
                    })
            except Exception as error:
                attempt = {
                    **common,
                    "status": "error",
                    "query_completed_at": _utc_now(),
                    "latency_seconds": time.perf_counter() - started_clock,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                if prompt is not None:
                    attempt["prompt_sha256"] = _sha256_bytes(prompt.encode("utf-8"))
                    attempt["prompt_version"] = prompt_version
                if raw_path.exists():
                    attempt["raw_response_file"] = str(raw_path.relative_to(output))
                    attempt["raw_response_sha256"] = _sha256(raw_path)
            _append_attempt(attempts_path, attempt)
            if marker_path is not None:
                marker_path.unlink(missing_ok=True)
            new_attempts.append(attempt)
            completed_ids.add(attempt_id)

    attempts = _load_attempts(attempts_path)
    _write_labels(labels_path, attempts)
    counts = dict(Counter(str(attempt.get("status")) for attempt in attempts))
    result = {
        "planned_attempts": plan.query_budget,
        "recorded_attempts": len(attempts),
        "new_attempts": len(new_attempts),
        "successful_labels": counts.get("success", 0),
        "invalid_responses": counts.get("invalid_response", 0),
        "errors": counts.get("error", 0),
    }
    manifest["status"] = (
        "complete"
        if result["recorded_attempts"] == result["planned_attempts"]
        and not result["invalid_responses"] and not result["errors"]
        else "partial"
    )
    manifest["finished_at"] = _utc_now()
    manifest["result"] = result
    _write_manifest(manifest_path, manifest)
    return result


def _options_from_args(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
    }
    if args.seed is not None:
        options["seed"] = args.seed
    return options


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sources-file", type=Path, required=True)
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="validate source hashes, normalized states, and selection"
    )
    _add_input_arguments(validate_parser)

    label_parser = subparsers.add_parser(
        "label", help="query the teacher for every declared selection attempt"
    )
    _add_input_arguments(label_parser)
    label_parser.add_argument("--output", type=Path, required=True)
    label_parser.add_argument("--run-id", required=True)
    label_parser.add_argument("--teacher-policy", default=DEFAULT_TEACHER_POLICY)
    label_parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    label_parser.add_argument(
        "--ollama-base-url", action="append", default=None,
        help="repeat for fallback endpoints; defaults to http://localhost:11434",
    )
    label_parser.add_argument("--model-digest")
    label_parser.add_argument("--temperature", type=float, default=1.0)
    label_parser.add_argument("--top-p", type=float, default=0.9)
    label_parser.add_argument("--num-ctx", type=int, default=8192)
    label_parser.add_argument("--seed", type=int)
    label_parser.add_argument("--resume", action="store_true")
    label_parser.add_argument(
        "--allow-dirty", action="store_true",
        help="diagnostic only; canonical runs require a clean checkout",
    )
    args = parser.parse_args()

    if args.command == "validate":
        plan, _, summary = validate_selected_states(
            args.sources_file, args.state_bank_root, args.selection,
        )
        print(json.dumps({"selection_id": plan.selection_id, **summary}, indent=2))
        return

    base_urls = args.ollama_base_url or ["http://localhost:11434"]
    identity = resolve_model_identity(base_urls, args.teacher_model, args.model_digest)
    result = label_selected_states(
        sources_file=args.sources_file,
        state_bank_root=args.state_bank_root,
        selection_file=args.selection,
        output=args.output,
        run_id=args.run_id,
        teacher_policy=args.teacher_policy,
        teacher_model=args.teacher_model,
        base_urls=base_urls,
        options=_options_from_args(args),
        model_identity=identity,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(result, indent=2))
    if result["invalid_responses"] or result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
