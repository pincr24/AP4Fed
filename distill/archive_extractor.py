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


LabelMode = Literal["states-only", "source-behavior"]
LABEL_MODES = ("states-only", "source-behavior")

# These positions belong to AP4Fed's six-slot metrics format. They are a log
# schema contract, not properties of a particular experiment.
AP_SLOTS = {"cs": 0, "mc": 2, "hdh": 5}
DIGEST_METRICS = ("f1", "traintime", "commtime", "totaltime")

STATIC_FEATURES = (
    "n_clients", "max_cpu", "second_highest_cpu", "cpu_spread",
    "frac_non_iid", "frac_new_data", "has_delay_clients", "workload",
)
ACTION_FEATURES = (
    "prev_cs", "prev_mc", "prev_hdh", "rs_cs_on", "rs_mc_on", "rs_hdh_on",
)
CLOCK_FEATURES = ("round_idx", "rounds_remaining")
SNAPSHOT_FEATURES = (
    "val_f1", "total_round_time", "train_mean", "train_min", "train_max",
    "comm_mean", "comm_min", "comm_max", "jsd", "f1_over_time", "comm_frac",
)
DELTA_FEATURES = ("d_f1", "d_total_time", "d_train_mean", "d_comm_mean", "d_jsd")
DIGEST_FEATURES = tuple(
    f"{metric}_{stat}"
    for metric in DIGEST_METRICS
    for stat in ("mean", "last3", "last5", "slope")
)
FEATURE_COLUMNS = (
    STATIC_FEATURES + ACTION_FEATURES + CLOCK_FEATURES + SNAPSHOT_FEATURES
    + DELTA_FEATURES + DIGEST_FEATURES
)
LABEL_COLUMNS = ("y_cs_applied", "y_mc_applied", "y_hdh_applied")
STATE_ID_COLUMNS = ("record_id", "run_id")
LABEL_RECORD_COLUMNS = (
    "record_id", "attempt_id", "label_kind", "teacher_policy",
    "teacher_model", "y_cs_applied", "selection_value",
    "y_mc_applied", "y_hdh_applied",
)


@dataclass(frozen=True)
class ExperimentSource:
    """Identify one experiment arm without embedding it in extraction code.

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


def _parse_ap_vector(value: str, source: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.strip().strip("{}").split(","))
    if len(values) != 6 or any(item not in {"ON", "OFF"} for item in values):
        raise ValueError(f"{source}: invalid AP vector {value!r}")
    return values


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
                _parse_ap_vector(value, f"{source}, round {round_idx}")
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
    # so a single arm cannot mix datasets or models between its clients.
    datasets = {str(client.get("dataset", "")).strip() for client in clients}
    models = {str(client.get("model", "")).strip() for client in clients}
    if "" in datasets or "" in models or len(datasets) != 1 or len(models) != 1:
        raise ValueError(
            f"{config_path}: expected one workload per arm; "
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


def _identity(config: dict[str, object]) -> dict[str, str]:
    """Read source provenance without inferring experiment-specific behavior."""
    clients = list(config["client_details"])
    dataset = str(clients[0]["dataset"])
    model = str(clients[0]["model"])
    return {
        "dataset": dataset,
        "model": model,
        "workload": f"{dataset}__{model}".replace(" ", "_"),
        "policy": str(config["adaptation"]),
        # This is the configured value. The audit does not claim that every
        # policy actually invokes it.
        "configured_model": str(config.get("LLM", "")),
    }


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_column(rows: list[dict[str, str]], candidates: tuple[object, ...]) -> str | None:
    """Mirror the flexible column matching used by AP4Fed's prompt builder."""
    if not rows:
        return None
    for column in rows[0]:
        normalized = column.strip().lower()
        for pattern in candidates:
            if callable(pattern):
                if pattern(normalized):
                    return column
            elif pattern in normalized:
                return column
    return None


def _values(rows: list[dict[str, str]], column: str | None) -> list[float]:
    if not column:
        return []
    return [parsed for row in rows if (parsed := _number(row.get(column, ""))) is not None]


def _rows_through(rows: list[dict[str, str]], round_idx: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row["FL Round"]) <= round_idx]


def _aggregate_snapshot(rows: list[dict[str, str]]) -> dict[str, float | str]:
    """Recreate the latest teacher-visible AP4Fed round summary."""
    blank = {
        "val_f1": "", "total_round_time": "", "train_mean": "", "train_min": "",
        "train_max": "", "comm_mean": "", "comm_min": "", "comm_max": "", "jsd": "",
    }
    if not rows:
        return blank
    col_round = _find_column(rows, (lambda value: "round" in value,))
    col_client = _find_column(
        rows, (lambda value: ("client" in value and "id" in value) or value == "client id",)
    )
    col_train = _find_column(
        rows,
        (lambda value: "training" in value and "time" in value,
         "training (s)", "training time (s)", "training time"),
    )
    col_comm = _find_column(
        rows, (lambda value: ("comm" in value and "time" in value) or "communication" in value,)
    )
    col_total = _find_column(
        rows, (lambda value: "total time of fl round" in value or ("total" in value and "round" in value),)
    )
    col_f1 = _find_column(rows, (lambda value: "val f1" in value or value == "f1",))
    col_jsd = _find_column(rows, (lambda value: value == "jsd",))

    latest = rows
    if col_round:
        latest_round = max(int(row[col_round]) for row in rows if row.get(col_round, "").strip())
        latest = [row for row in rows if int(row[col_round]) == latest_round]
    per_client = latest
    if col_client:
        per_client = [row for row in latest if row.get(col_client, "").strip()]
    train = _values(per_client, col_train)
    comm = _values(per_client, col_comm)

    def last(column: str | None) -> float | str:
        series = _values(latest, column)
        return series[-1] if series else ""

    return {
        "val_f1": last(col_f1),
        "total_round_time": last(col_total),
        "train_mean": sum(train) / len(train) if train else "",
        "train_min": min(train) if train else "",
        "train_max": max(train) if train else "",
        "comm_mean": sum(comm) / len(comm) if comm else "",
        "comm_min": min(comm) if comm else "",
        "comm_max": max(comm) if comm else "",
        "jsd": last(col_jsd),
    }


def _digest_features(rows: list[dict[str, str]]) -> dict[str, float | str]:
    """Recreate the history digest, including the declared F1 lookup correction."""
    columns = {
        "f1": _find_column(rows, (lambda value: "val f1" in value,)),
        "traintime": _find_column(
            rows,
            (lambda value: "training" in value and "time" in value,
             "training (s)", "training time (s)"),
        ),
        "commtime": _find_column(
            rows, (lambda value: ("comm" in value and "time" in value) or "communication" in value,)
        ),
        "totaltime": _find_column(
            rows,
            (lambda value: ("total" in value and "time" in value)
             or "total time of fl round" in value or "round time" in value,),
        ),
    }
    output: dict[str, float | str] = {}
    for metric, column in columns.items():
        series = _values(rows, column)
        if not series:
            output.update({f"{metric}_{stat}": "" for stat in ("mean", "last3", "last5", "slope")})
            continue
        count = len(series)
        output[f"{metric}_mean"] = sum(series) / count
        output[f"{metric}_last3"] = sum(series[-3:]) / min(3, count)
        output[f"{metric}_last5"] = sum(series[-5:]) / min(5, count)
        output[f"{metric}_slope"] = (
            (series[-1] - series[0]) / (count - 1) if count >= 2 else 0.0
        )
    return output


def _static_features(config: dict[str, object]) -> dict[str, float | int | str]:
    clients = list(config["client_details"])
    cpus = [int(client.get("cpu", 0) or 0) for client in clients]
    ordered_cpus = sorted(cpus, reverse=True)
    n_clients = int(config["clients"])
    non_iid = [
        client for client in clients
        if str(client.get("data_distribution_type", "")).strip().upper() != "IID"
    ]
    new_data = [
        client for client in clients
        if str(client.get("data_persistence_type", "")).strip().lower() == "new data"
    ]
    delayed = any(
        str(client.get("delay_combobox", "")).strip().lower() == "yes"
        for client in clients
    )
    identity = _identity(config)
    return {
        "n_clients": n_clients,
        "max_cpu": max(cpus) if cpus else 0,
        "second_highest_cpu": (
            ordered_cpus[1] if len(ordered_cpus) >= 2
            else (ordered_cpus[0] if ordered_cpus else 0)
        ),
        "cpu_spread": max(cpus) - min(cpus) if cpus else 0,
        "frac_non_iid": len(non_iid) / n_clients,
        "frac_new_data": len(new_data) / n_clients,
        "has_delay_clients": "ON" if delayed else "OFF",
        "workload": identity["workload"],
    }


def _rounds_since(
    vectors: dict[int, tuple[str, ...]], decision_idx: int, slot: int,
) -> int | str:
    previous_on = [
        index for index in vectors
        if index <= decision_idx and vectors[index][slot] == "ON"
    ]
    return decision_idx - max(previous_on) if previous_on else ""


def _situation_features(
    config: dict[str, object],
    rows: list[dict[str, str]],
    vectors: dict[int, tuple[str, ...]],
    decision_idx: int,
) -> dict[str, float | int | str]:
    """Construct a decision state without reading its outcome or future metrics."""
    prior = vectors[decision_idx]
    features: dict[str, float | int | str] = _static_features(config)
    features.update({
        "prev_cs": prior[AP_SLOTS["cs"]],
        "prev_mc": prior[AP_SLOTS["mc"]],
        "prev_hdh": prior[AP_SLOTS["hdh"]],
        "rs_cs_on": _rounds_since(vectors, decision_idx, AP_SLOTS["cs"]),
        "rs_mc_on": _rounds_since(vectors, decision_idx, AP_SLOTS["mc"]),
        "rs_hdh_on": _rounds_since(vectors, decision_idx, AP_SLOTS["hdh"]),
        "round_idx": decision_idx,
        "rounds_remaining": int(config["rounds"]) - decision_idx,
    })

    snapshot = (
        _aggregate_snapshot(_rows_through(rows, decision_idx - 1))
        if decision_idx >= 2 else _aggregate_snapshot([])
    )
    features.update(snapshot)
    f1 = snapshot["val_f1"]
    total_time = snapshot["total_round_time"]
    comm = snapshot["comm_mean"]
    features["f1_over_time"] = (
        f1 / total_time
        if isinstance(f1, float) and isinstance(total_time, float) and total_time else ""
    )
    features["comm_frac"] = (
        comm / total_time
        if isinstance(comm, float) and isinstance(total_time, float) and total_time else ""
    )

    if decision_idx >= 3:
        earlier = _aggregate_snapshot(_rows_through(rows, decision_idx - 2))
        for output, source in (
            ("d_f1", "val_f1"),
            ("d_total_time", "total_round_time"),
            ("d_train_mean", "train_mean"),
            ("d_comm_mean", "comm_mean"),
            ("d_jsd", "jsd"),
        ):
            current = snapshot[source]
            previous = earlier[source]
            features[output] = (
                current - previous
                if isinstance(current, float) and isinstance(previous, float) else ""
            )
    else:
        features.update({feature: "" for feature in DELTA_FEATURES})

    features.update(_digest_features(_rows_through(rows, max(1, decision_idx - 1))))
    if tuple(features) != FEATURE_COLUMNS:
        raise AssertionError("feature schema drift")
    return features


def extract_experiment(spec: ExperimentSource) -> ExtractionResult:
    """Extract one experiment arm into audited decision states.

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
    identity = _identity(config)
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
            state.update(_situation_features(config, rows, vectors, decision_idx))
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
