"""Adapt archived AP4Fed states to the Docker single-agent teacher interface.

Offline teacher elicitation replays decision points that were originally taken
inside a live federated run, so the teacher has to be asked the same question the
live single agent would have been asked. This module supplies that bridge.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

from teacher_campaign import SelectedState


DECISION_KEYS = (
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
)
SIGNATURE_RE = re.compile(
    r"\bCS\s*=\s*(ON|OFF)\s*;\s*MC\s*=\s*(ON|OFF)\s*;\s*HDH\s*=\s*(ON|OFF)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TeacherModules:
    """Docker teacher modules and the source paths used to load them."""

    prompting: ModuleType
    metrics: ModuleType
    client: ModuleType
    prompting_path: Path
    metrics_path: Path
    client_path: Path


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
