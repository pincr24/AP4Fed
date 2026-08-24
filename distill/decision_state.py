"""Build Feature Specification v1 states for one AP4Fed adaptation decision.

This module is the shared implementation boundary for offline archive
reconstruction and the live policy path.  It constructs state only: policy
selection, teacher calls, trace writing, and archive discovery belong elsewhere.
"""

from __future__ import annotations


# These positions belong to AP4Fed's six-slot metrics format. 
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


def parse_ap_vector(value: str, source: str) -> tuple[str, ...]:
    """Validate the six-slot AP vector written by AP4Fed metrics."""
    values = tuple(item.strip() for item in value.strip().strip("{}").split(","))
    if len(values) != 6 or any(item not in {"ON", "OFF"} for item in values):
        raise ValueError(f"{source}: invalid AP vector {value!r}")
    return values


def configuration_identity(config: dict[str, object]) -> dict[str, str]:
    """Read configuration provenance without inferring runtime behavior."""
    clients = list(config["client_details"])
    dataset = str(clients[0]["dataset"])
    model = str(clients[0]["model"])
    return {
        "dataset": dataset,
        "model": model,
        "workload": f"{dataset}__{model}".replace(" ", "_"),
        "policy": str(config["adaptation"]),
        "configured_model": str(config.get("LLM", "")),
    }


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_column(rows: list[dict[str, str]], candidates: tuple[object, ...]) -> str | None:
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
    identity = configuration_identity(config)
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


def build_decision_state(
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


def build_live_state(
    config: dict[str, object],
    rows: list[dict[str, str]],
    decision_idx: int,
) -> dict[str, str]:
    """Build the feature vector (state) that a policy is allowed to see at one live adaptation decision.
    point during a live FL run"""
    rounds = config.get("rounds")
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or decision_idx < 1
        or decision_idx >= rounds
    ):
        raise ValueError("live decision index must satisfy 1 <= decision < rounds")
    if not rows or "FL Round" not in rows[0]:
        raise ValueError("live metrics rows must contain FL Round")
    ap_column = next((field for field in rows[0] if field.startswith("AP List")), None)
    if ap_column is None:
        raise ValueError("live metrics rows must contain the AP List column")

    prefix = _rows_through(rows, decision_idx)
    vectors: dict[int, tuple[str, ...]] = {}
    for round_idx in range(1, decision_idx + 1):
        values = {
            parse_ap_vector(row[ap_column], f"live decision {decision_idx}, round {round_idx}")
            for row in prefix
            if int(row["FL Round"]) == round_idx and row.get(ap_column, "").strip()
        }
        if len(values) != 1:
            raise ValueError(
                f"live decision {decision_idx}: expected one AP vector for round "
                f"{round_idx}, found {len(values)}"
            )
        vectors[round_idx] = next(iter(values))

    features = build_decision_state(config, prefix, vectors, decision_idx)
    return {
        feature: "" if features[feature] == "" else str(features[feature])
        for feature in FEATURE_COLUMNS
    }
