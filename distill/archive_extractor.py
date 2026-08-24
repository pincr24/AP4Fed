"""Reusable extraction of one normalized dataset from an AP4Fed archived experiment configuration.

Callers supply the experiment source, either directly or through a higher-level source
list, and decide where each extraction is written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from decision_state import (
    AP_SLOTS,
    FEATURE_COLUMNS,
    build_decision_state,
    configuration_identity,
    parse_ap_vector,
)


LabelMode = Literal["states-only", "source-behavior"]
LABEL_MODES = ("states-only", "source-behavior")

LABEL_COLUMNS = ("y_cs_applied", "y_mc_applied", "y_hdh_applied")
STATE_ID_COLUMNS = ("record_id", "run_id")
LABEL_RECORD_COLUMNS = (
    "record_id", "attempt_id", "label_kind", "teacher_policy",
    "teacher_model", "y_cs_applied", "selection_value",
    "y_mc_applied", "y_hdh_applied",
)


@dataclass(frozen=True)
class ExperimentSource:
    """Identify one experiment configuration without embedding it in extraction code.

    ``source_id`` is a stable logical identifier used in record IDs and audit
    data. It should remain the same if the files move to another machine.
    """

    source: Path
    source_id: str
    label_mode: LabelMode = "states-only"


@dataclass(frozen=True)
class ExtractionResult:
    """Keep immutable states, optional labels, and provenance together."""

    states: list[dict[str, str | int | float]]
    labels: list[dict[str, str]]
    audit: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_run(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    ap_column = next((field for field in rows[0] if field.startswith("AP List")), None)
    if ap_column is None:
        raise ValueError(f"{path}: AP List column not found")
    if "FL Round" not in rows[0]:
        raise ValueError(f"{path}: FL Round column not found")
    return rows, ap_column


def _load_vectors(
    rows: list[dict[str, str]], ap_column: str, source: str, rounds: int,
) -> dict[int, tuple[str, ...]]:
    """Recover one unambiguous AP vector for every configured round."""
    by_round: dict[int, set[tuple[str, ...]]] = {}
    for row in rows:
        if value := row[ap_column].strip():
            round_idx = int(row["FL Round"])
            by_round.setdefault(round_idx, set()).add(
                parse_ap_vector(value, f"{source}, round {round_idx}")
            )
    expected = set(range(1, rounds + 1))
    if set(by_round) != expected:
        raise ValueError(
            f"{source}: AP vectors present for {sorted(by_round)}, "
            f"expected 1..{rounds}"
        )
    conflicts = {index: states for index, states in by_round.items() if len(states) != 1}
    if conflicts:
        raise ValueError(f"{source}: conflicting populated AP vectors: {conflicts}")
    return {index: next(iter(states)) for index, states in by_round.items()}


def _load_config(source: Path) -> tuple[dict[str, object], str]:
    """Validate only fields required by the extraction and feature contracts."""
    config_path = source / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("adaptation", "rounds", "clients", "client_details")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path}: missing required fields {missing}")
    rounds = config["rounds"]
    if not isinstance(rounds, int) or rounds < 2:
        raise ValueError(f"{config_path}: rounds must be an integer >= 2")
    clients = config["client_details"]
    if not isinstance(clients, list) or not clients:
        raise ValueError(f"{config_path}: client_details must be a non-empty list")
    if config["clients"] != len(clients):
        raise ValueError(
            f"{config_path}: clients={config['clients']!r}, but client_details "
            f"contains {len(clients)} entries"
        )

    # Feature Specification v1 represents workload as one categorical value,
    # so one configuration cannot mix datasets or models between its clients.
    datasets = {str(client.get("dataset", "")).strip() for client in clients}
    models = {str(client.get("model", "")).strip() for client in clients}
    if "" in datasets or "" in models or len(datasets) != 1 or len(models) != 1:
        raise ValueError(
            f"{config_path}: expected one workload per configuration; "
            f"found datasets={sorted(datasets)}, models={sorted(models)}"
        )
    return config, _sha256(config_path)


def _discover_run_files(source: Path) -> list[Path]:
    """Return exact r<N>.csv runs while excluding rationale and summary files."""
    numbered: list[tuple[int, Path]] = []
    for path in source.iterdir():
        if match := re.fullmatch(r"r(\d+)\.csv", path.name):
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    numbers = [number for number, _ in numbered]
    if not numbers:
        raise ValueError(f"{source}: no exact r<N>.csv run files found")
    expected = list(range(1, numbers[-1] + 1))
    if numbers != expected:
        raise ValueError(f"{source}: run files are {numbers}, expected contiguous {expected}")
    return [path for _, path in numbered]


def extract_experiment(spec: ExperimentSource) -> ExtractionResult:
    """Extract one experiment configuration into audited decision states.

    In ``states-only`` mode, the archived next action remains factual source
    provenance and label fields stay empty for later teacher relabelling.
    ``source-behavior`` instead treats the source policy's
    logged action as the label.
    """
    source = spec.source.resolve()
    source_id = spec.source_id.strip().rstrip("/")
    if not source_id:
        raise ValueError("source_id must be a non-empty stable identifier")
    if spec.label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")

    config, config_sha = _load_config(source)
    identity = configuration_identity(config)
    rounds = int(config["rounds"])
    run_files = _discover_run_files(source)
    has_labels = spec.label_mode == "source-behavior"

    states: list[dict[str, str | int | float]] = []
    labels: list[dict[str, str]] = []
    run_registry: dict[str, dict[str, object]] = {}
    factual_actions: dict[str, list[str]] = {head: [] for head in LABEL_COLUMNS}
    for csv_path in run_files:
        rows, ap_column = _read_run(csv_path)
        vectors = _load_vectors(rows, ap_column, csv_path.name, rounds)
        source_hash = _sha256(csv_path)
        run_id = f"{source_id}/{csv_path.stem}"
        run_registry[run_id] = {
            "source_csv": f"{source_id}/{csv_path.name}",
            "source_sha256": source_hash,
            "factual_source_actions": {},
        }
        for decision_idx in range(1, rounds):
            source_action = vectors[decision_idx + 1]
            record_id = f"{run_id}::d{decision_idx}"
            raw_source_action = "{" + ",".join(source_action) + "}"
            run_registry[run_id]["factual_source_actions"][str(decision_idx)] = (
                raw_source_action
            )
            actions = {
                "y_cs_applied": source_action[AP_SLOTS["cs"]],
                "y_mc_applied": source_action[AP_SLOTS["mc"]],
                "y_hdh_applied": source_action[AP_SLOTS["hdh"]],
            }
            for head, action in actions.items():
                factual_actions[head].append(action)
            state: dict[str, str | int | float] = {
                "record_id": record_id,
                "run_id": run_id,
            }
            state.update(build_decision_state(config, rows, vectors, decision_idx))
            states.append(state)
            if has_labels:
                labels.append({
                    "record_id": record_id,
                    "attempt_id": "archived-source-behavior",
                    "label_kind": "direct_archived_behavior",
                    "teacher_policy": identity["policy"],
                    "teacher_model": identity["configured_model"],
                    "y_cs_applied": actions["y_cs_applied"],
                    "selection_value": "",
                    "y_mc_applied": actions["y_mc_applied"],
                    "y_hdh_applied": actions["y_hdh_applied"],
                })

    source_action_counts = {
        head: dict(sorted(Counter(actions).items()))
        for head, actions in factual_actions.items()
    }
    class_counts = (
        source_action_counts if has_labels else {head: {} for head in LABEL_COLUMNS}
    )
    audit: dict[str, object] = {
        "schema_version": 5,
        "source_id": source_id,
        "state_source": {
            "policy": identity["policy"],
            "configured_model": identity["configured_model"],
            "dataset": identity["dataset"],
            "model": identity["model"],
            "workload": identity["workload"],
        },
        "label_mode": spec.label_mode,
        "label_contract": (
            "labels are direct behavior of the archived source policy"
            if has_labels else
            "labels.csv has no rows; factual source behavior remains in run_registry"
        ),
        "artifacts": {
            "states": "decision_states.csv",
            "labels": "labels.csv",
            "provenance": "audit.json",
        },
        "feature_schema": (
            "Feature_Spec v1: 48 inputs; rounds-since features are "
            "student-history extensions"
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "state_columns": STATE_ID_COLUMNS + FEATURE_COLUMNS,
        "label_record_columns": LABEL_RECORD_COLUMNS,
        "decision_alignment": (
            "situation at decision d; factual source action is AP List at round d+1"
        ),
        "snapshot_lag": "snapshot metrics use rounds through d-1; d=1 is blank",
        "digest_lag": "digest uses rows through max(1, d-1)",
        "feature_recreation": (
            "AP4Fed single-agent aggregation and digest calculations with the "
            "declared case-insensitive Val F1 correction"
        ),
        "selection_threshold": (
            "not recoverable from archived AP List; client selector remains binary"
        ),
        "run_count": len(run_files),
        "run_registry": run_registry,
        "configured_rounds": rounds,
        "decisions_per_run": rounds - 1,
        "decision_rows": len(states),
        "label_rows": len(labels),
        "positive_counts": (
            {head: class_counts[head].get("ON", 0) for head in LABEL_COLUMNS}
            if has_labels else {}
        ),
        "class_counts": class_counts,
        "factual_source_action_counts": source_action_counts,
        "config_sha256": config_sha,
    }
    return ExtractionResult(states=states, labels=labels, audit=audit)


def write_extraction(
    output: Path, result: ExtractionResult, overwrite: bool = False,
) -> None:
    """Write normalized state, label, and provenance artifacts."""
    output = output.resolve()
    legacy_output = output / "decision_dataset.csv"
    if legacy_output.exists():
        raise FileExistsError(
            f"{legacy_output} uses the previous joined layout; select a new output directory"
        )
    targets = (
        output / "decision_states.csv",
        output / "labels.csv",
        output / "audit.json",
    )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing extraction outputs without "
            f"overwrite=True: {existing}"
        )
    output.mkdir(parents=True, exist_ok=True)
    with targets[0].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=STATE_ID_COLUMNS + FEATURE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.states)
    with targets[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=LABEL_RECORD_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.labels)
    with targets[2].open("w", encoding="utf-8") as stream:
        json.dump(result.audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
