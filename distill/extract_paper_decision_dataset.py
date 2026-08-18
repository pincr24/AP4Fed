#!/usr/bin/env python3
"""Extract dataset of features->decision rows from archived experiment logs.

The input is an archived AP4Fed experiment directory containing ``config.json``
and ``r1.csv`` through ``r10.csv``. The output has one row per decision, a 10-round
run has nine decisions, and decision d is applied in the AP vector logged for round d+1.

The 48 input features follow ``Feature_Spec.md`` v1. They directly recreate
the single-agent prompt's feature calculations from the archived inputs,
including AP4Fed's one-round snapshot lag. The distillation-side recreation
also corrects the prompt's case-sensitive F1-digest column lookup. The three
``y_*_applied`` columns are the direct logged AP decisions applied in the
following round.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


AP4FED_ROOT = Path(__file__).resolve().parents[1]
DISTILL_ROOT = AP4FED_ROOT / "distill"
DEFAULT_OUTPUT = DISTILL_ROOT / "data" / "paper_archive_fashionmnist_cnn16k_fewshot_deepseek"
DEFAULT_SOURCE_ID = "agentic-paper-archive/fashionmnist__cnn16k/few-shot-deepseek"

# AP List order as emitted by AP4Fed's metrics logger.
AP_SLOTS = {"cs": 0, "mc": 2, "hdh": 5}
EXPECTED_CONFIG = {
    "adaptation": "Single AI-Agent (Few-Shot)",
    "LLM": "deepseek-r1:8b",
    "rounds": 10,
    "clients": 5,
}
# Keep the feature names stable and carry over the prompt's aggregation and
# digest calculations.  The only intentional correction is the F1-digest
# matcher: the runtime prompt compares ``"Val F1"`` to a lower-cased column
# name, so it never discovers F1 history.  Distillation uses the equivalent
# case-insensitive lookup instead.
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
FEATURE_COLUMNS = STATIC_FEATURES + ACTION_FEATURES + CLOCK_FEATURES + SNAPSHOT_FEATURES + DELTA_FEATURES + DIGEST_FEATURES
LABEL_COLUMNS = ("y_cs_applied", "y_mc_applied", "y_hdh_applied")
RECORD_COLUMNS = (
    "record_id", "run_id", "action_ap_round", "raw_ap_at_action_round",
    "teacher_policy", "teacher_model", "label_kind", "source_csv", "source_sha256",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ap_vector(value: str, source: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.strip().strip("{}").split(","))
    if len(values) != 6 or any(item not in {"ON", "OFF"} for item in values):
        raise ValueError(f"{source}: invalid AP vector {value!r}")
    return values


def read_run(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    ap_column = next((field for field in rows[0] if field.startswith("AP List")), None)
    if ap_column is None:
        raise ValueError(f"{path}: AP List column not found")
    return rows, ap_column


def load_vectors(rows: list[dict[str, str]], ap_column: str, source: str) -> dict[int, tuple[str, ...]]:
    """Return one populated AP vector for every round 1-10, reject if ambiguous."""
    by_round: dict[int, set[tuple[str, ...]]] = {}
    for row in rows:
        if value := row[ap_column].strip():
            round_idx = int(row["FL Round"])
            by_round.setdefault(round_idx, set()).add(
                parse_ap_vector(value, f"{source}, round {round_idx}")
            )
    if set(by_round) != set(range(1, 11)):
        raise ValueError(f"{source}: AP vectors present for {sorted(by_round)}, expected 1..10")
    conflicts = {round_idx: states for round_idx, states in by_round.items() if len(states) != 1}
    if conflicts:
        raise ValueError(f"{source}: conflicting populated AP vectors: {conflicts}")
    return {round_idx: next(iter(states)) for round_idx, states in by_round.items()}


def check_config(source: Path) -> tuple[dict[str, object], str]:
    """Prevent mixing data from another experiment"""
    config_path = source / "config.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{config_path}: expected Few-Shot DeepSeek arm: {mismatches}")
    clients = config.get("client_details", [])
    if {client.get("dataset") for client in clients} != {"FashionMNIST"} or \
       {client.get("model") for client in clients} != {"CNN 16k"}:
        raise ValueError(f"{config_path}: expected FashionMNIST / CNN 16k clients")
    return config, sha256(config_path)


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prompt_find_column(rows: list[dict[str, str]], candidates: tuple[object, ...]) -> str | None:
    """Port of the column matcher used in AP4Fed's prompt code."""
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


def values(rows: list[dict[str, str]], column: str | None) -> list[float]:
    if not column:
        return []
    return [parsed for row in rows if (parsed := number(row.get(column, ""))) is not None]


def rows_through(rows: list[dict[str, str]], round_idx: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row["FL Round"]) <= round_idx]


def aggregate_snapshot(rows: list[dict[str, str]]) -> dict[str, float | str]:
    """Port ``adaptation_metrics._sa_aggregate_round`` for archived CSV rows."""
    blank = {
        "val_f1": "", "total_round_time": "", "train_mean": "", "train_min": "",
        "train_max": "", "comm_mean": "", "comm_min": "", "comm_max": "", "jsd": "",
    }
    if not rows:
        return blank
    col_round = prompt_find_column(rows, (lambda value: "round" in value,))
    col_client = prompt_find_column(rows, (lambda value: ("client" in value and "id" in value) or value == "client id",))
    col_train = prompt_find_column(rows, (lambda value: "training" in value and "time" in value, "training (s)", "training time (s)", "training time"))
    col_comm = prompt_find_column(rows, (lambda value: ("comm" in value and "time" in value) or "communication" in value,))
    col_total = prompt_find_column(rows, (lambda value: "total time of fl round" in value or ("total" in value and "round" in value),))
    col_f1 = prompt_find_column(rows, (lambda value: "val f1" in value or value == "f1",))
    col_jsd = prompt_find_column(rows, (lambda value: value == "jsd",))

    latest = rows
    if col_round:
        latest_round = max(int(row[col_round]) for row in rows if row.get(col_round, "").strip())
        latest = [row for row in rows if int(row[col_round]) == latest_round]
    per_client = latest
    if col_client:
        per_client = [row for row in latest if row.get(col_client, "").strip()]
    train, comm = values(per_client, col_train), values(per_client, col_comm)

    def last(column: str | None) -> float | str:
        series = values(latest, column)
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


def digest_features(rows: list[dict[str, str]]) -> dict[str, float | str]:
    """Recreate ``_sa_build_prompt``'s digest, with a corrected F1 lookup."""
    out: dict[str, float | str] = {}
    columns = {
        "f1": prompt_find_column(rows, (lambda value: "val f1" in value,)),
        "traintime": prompt_find_column(rows, (lambda value: "training" in value and "time" in value, "training (s)", "training time (s)")),
        "commtime": prompt_find_column(rows, (lambda value: ("comm" in value and "time" in value) or "communication" in value,)),
        "totaltime": prompt_find_column(rows, (lambda value: ("total" in value and "time" in value) or "total time of fl round" in value or "round time" in value,)),
    }
    for metric, column in columns.items():
        series = values(rows, column)
        if not series:
            out.update({f"{metric}_{stat}": "" for stat in ("mean", "last3", "last5", "slope")})
            continue
        n = len(series)
        out[f"{metric}_mean"] = sum(series) / n
        out[f"{metric}_last3"] = sum(series[-3:]) / min(3, n)
        out[f"{metric}_last5"] = sum(series[-5:]) / min(5, n)
        out[f"{metric}_slope"] = (series[-1] - series[0]) / (n - 1) if n >= 2 else 0.0
    return out


def static_features(config: dict[str, object]) -> dict[str, float | int | str]:
    clients = list(config.get("client_details") or [])
    cpus = [int(client.get("cpu", 0) or 0) for client in clients]
    ordered_cpus = sorted(cpus, reverse=True)
    n_clients = int(config.get("clients", len(clients)) or len(clients))
    non_iid = [client for client in clients if str(client.get("data_distribution_type", "")).upper() != "IID"]
    new_data = [client for client in clients if str(client.get("data_persistence_type", "")).strip() == "New Data"]
    delayed = any(str(client.get("delay_combobox", "")).strip().lower() == "yes" for client in clients)
    dataset = str(clients[0].get("dataset", "") if clients else "")
    model = str(clients[0].get("model", "") if clients else "")
    return {
        "n_clients": n_clients,
        "max_cpu": max(cpus) if cpus else 0,
        "second_highest_cpu": ordered_cpus[1] if len(ordered_cpus) >= 2 else (ordered_cpus[0] if ordered_cpus else 0),
        "cpu_spread": max(cpus) - min(cpus) if cpus else 0,
        "frac_non_iid": len(non_iid) / n_clients if n_clients else "",
        "frac_new_data": len(new_data) / n_clients if n_clients else "",
        "has_delay_clients": "ON" if delayed else "OFF",
        "workload": f"{dataset}__{model}".replace(" ", "_"),
    }


def rounds_since(vectors: dict[int, tuple[str, ...]], decision_idx: int, slot: int) -> int | str:
    on_rounds = [round_idx for round_idx in vectors if round_idx <= decision_idx and vectors[round_idx][slot] == "ON"]
    return decision_idx - max(on_rounds) if on_rounds else ""


def situation_features(
    config: dict[str, object], rows: list[dict[str, str]], vectors: dict[int, tuple[str, ...]], decision_idx: int,
) -> dict[str, float | int | str]:
    """Construct x_d without using the action outcome or later-round metrics."""
    prior = vectors[decision_idx]
    features: dict[str, float | int | str] = static_features(config)
    features.update({
        "prev_cs": prior[AP_SLOTS["cs"]],
        "prev_mc": prior[AP_SLOTS["mc"]],
        "prev_hdh": prior[AP_SLOTS["hdh"]],
        "rs_cs_on": rounds_since(vectors, decision_idx, AP_SLOTS["cs"]),
        "rs_mc_on": rounds_since(vectors, decision_idx, AP_SLOTS["mc"]),
        "rs_hdh_on": rounds_since(vectors, decision_idx, AP_SLOTS["hdh"]),
        "round_idx": decision_idx,
        "rounds_remaining": int(config["rounds"]) - decision_idx,
    })

    # The decision at d occurs before snapshot d is copied. At d=1 there is no
    # snapshot, but the prompt digest falls back to the growing round-1 CSV.
    snapshot = aggregate_snapshot(rows_through(rows, decision_idx - 1)) if decision_idx >= 2 else aggregate_snapshot([])
    features.update(snapshot)
    f1, total_time, comm = snapshot["val_f1"], snapshot["total_round_time"], snapshot["comm_mean"]
    features["f1_over_time"] = f1 / total_time if isinstance(f1, float) and isinstance(total_time, float) and total_time else ""
    features["comm_frac"] = comm / total_time if isinstance(comm, float) and isinstance(total_time, float) and total_time else ""

    if decision_idx >= 3:
        earlier = aggregate_snapshot(rows_through(rows, decision_idx - 2))
        for output, source in (("d_f1", "val_f1"), ("d_total_time", "total_round_time"),
                               ("d_train_mean", "train_mean"), ("d_comm_mean", "comm_mean"), ("d_jsd", "jsd")):
            current, previous = snapshot[source], earlier[source]
            features[output] = current - previous if isinstance(current, float) and isinstance(previous, float) else ""
    else:
        features.update({feature: "" for feature in DELTA_FEATURES})

    features.update(digest_features(rows_through(rows, max(1, decision_idx - 1))))
    assert tuple(features) == FEATURE_COLUMNS, "feature schema drift"
    return features


def extract(source: Path, source_id: str) -> tuple[list[dict[str, str | int | float]], dict[str, object]]:
    """Label the situation features with the AP vector at d+1.
    Each record consists of {teacher-visible state at decision d → action applied in round d+1}
    The attached audit contains additional experiment data
    """
    config, config_sha = check_config(source)
    run_files = sorted(source.glob("r*.csv"), key=lambda path: int(re.fullmatch(r"r(\d+)", path.stem).group(1)))
    if [path.stem for path in run_files] != [f"r{i}" for i in range(1, 11)]:
        raise ValueError(f"{source}: expected exactly r1.csv through r10.csv")

    records: list[dict[str, str | int | float]] = []
    source_hashes: dict[str, str] = {}
    for csv_path in run_files:
        rows, ap_column = read_run(csv_path)
        vectors = load_vectors(rows, ap_column, csv_path.name)
        source_hash = sha256(csv_path)
        source_hashes[csv_path.name] = source_hash
        run_id = f"fashionmnist__cnn16k/few-shot deepseek/{csv_path.stem}"
        for decision_idx in range(1, 10):
            applied = vectors[decision_idx + 1]
            record: dict[str, str | int | float] = {
                "record_id": f"{csv_path.stem}_d{decision_idx}",
                "run_id": run_id,
                "action_ap_round": decision_idx + 1,
                "raw_ap_at_action_round": "{" + ",".join(applied) + "}",
                "teacher_policy": "Single AI-Agent (Few-Shot)",
                "teacher_model": "deepseek-r1:8b",
                "label_kind": "direct_archived_behavior",
                "source_csv": f"{source_id.rstrip('/')}/{csv_path.name}",
                "source_sha256": source_hash,
            }
            record.update(situation_features(config, rows, vectors, decision_idx))
            record.update({
                "y_cs_applied": applied[AP_SLOTS["cs"]],
                "y_mc_applied": applied[AP_SLOTS["mc"]],
                "y_hdh_applied": applied[AP_SLOTS["hdh"]],
            })
            records.append(record)

    class_counts = {head: dict(sorted(Counter(str(record[head]) for record in records).items())) for head in LABEL_COLUMNS}
    audit = {
        "schema_version": 4,
        "scope": "FashionMNIST / CNN 16k / Single-Agent Few-Shot DeepSeek",
        "source_id": source_id,
        "record_kind": "prompt-calculation recreation plus direct archived behavior decision",
        "feature_schema": "Feature_Spec v1: 48 inputs; rounds-since features are student-history extensions",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "decision_alignment": "situation at decision d is labelled from AP List recorded at round d+1",
        "snapshot_lag": "snapshot metrics use rounds through d-1; d=1 snapshot is blank",
        "digest_lag": "digest uses rows through max(1, d-1), matching the prompt fallback at d=1",
        "feature_recreation": "ports AP4Fed prompt aggregation/digest calculations; the canonical feature reconstruction corrects the historical prompt's case-sensitive Val F1 digest lookup (four F1-history fields only; labels unchanged)",
        "selection_threshold": "not recoverable from archived AP List; client selector remains binary",
        "runs": len(run_files), "decisions_per_run": 9, "decision_rows": len(records),
        "positive_counts": {head: class_counts[head].get("ON", 0) for head in class_counts},
        "class_counts": class_counts,
        "config_sha256": config_sha,
        "source_csv_sha256": source_hashes,
    }
    return records, audit


def write_outputs(output: Path, records: list[dict[str, str | int | float]], audit: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decision_dataset.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with (output / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory holding config.json and r1.csv through r10.csv")
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID, help="stable provenance identifier written to output (not a local path)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="defaults to distill/data/... below the AP4Fed root")
    args = parser.parse_args()
    records, audit = extract(args.source.resolve(), args.source_id)
    write_outputs(args.output.resolve(), records, audit)
    print(f"extracted {audit['decision_rows']} situation-decision rows -> {args.output}")
    print(f"features: {audit['feature_count']} | positive counts: {audit['positive_counts']}")


if __name__ == "__main__":
    main()
