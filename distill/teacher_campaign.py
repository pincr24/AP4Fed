"""Prepare predeclared states for an auditable teacher-query campaign.

This module validates the selection plan, recreates each normalized state bank
from its archived source, and resolves the chosen decision points. It performs
those checks before any teacher is contacted, so the query budget and source
states cannot change silently during elicitation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from archive_extractor import STATE_ID_COLUMNS, extract_experiment
from decision_state import FEATURE_COLUMNS
from build_state_bank import ExtractionJob, load_source_list


SELECTION_SCHEMA_VERSION = 1
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class SelectionRecord:
    """One selected decision point and its permitted number of query attempts."""

    record_id: str
    attempts: int
    purpose: str


@dataclass(frozen=True)
class SelectionPlan:
    """A validated, predeclared set of teacher queries."""

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
    """One selected state resolved back to its configuration and archived run."""

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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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
