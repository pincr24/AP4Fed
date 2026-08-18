#!/usr/bin/env python3
"""Run the first CON-FOLD decision-policy baseline.

Prepare AP4Fed data, write a derived training view for each binary AP head,
fit one CONFOLD model per head, and evaluate it by holding out one complete
FL trajectory at a time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from confold_adapter import make_classifier


DISTILL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    DISTILL_ROOT
    / "data/paper_archive_fashionmnist_cnn16k_fewshot_deepseek/decision_dataset.csv"
)
DEFAULT_OUTPUT = DEFAULT_DATASET.parent / "confold_baseline"

# This is the small, frozen feature set used for the first baseline.
# The source CSV has more fields, but this is part of the experiment
# definition. 
COMPACT_FEATURES = (
    "prev_cs",
    "prev_mc",
    "prev_hdh",
    "round_idx",
    "rounds_remaining",
    "val_f1",
    "total_round_time",
    "train_mean",
    "train_max",
    "comm_mean",
    "comm_max",
    "jsd",
    "f1_over_time",
    "comm_frac",
)
NUMERIC_FEATURES = frozenset(COMPACT_FEATURES[3:])
HEADS = {"cs": "y_cs_applied", "mc": "y_mc_applied", "hdh": "y_hdh_applied"}
SNAPSHOT_FEATURES = COMPACT_FEATURES[5:]
COLD_START_PREDICTION = "OFF"


def _read_dataset(path: Path) -> list[dict[str, str]]:
    """Read the canonical dataset."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    required = {"record_id", "run_id", *COMPACT_FEATURES, *HEADS.values()}
    absent = required.difference(rows[0])
    if absent:
        raise ValueError(f"Dataset is missing required columns: {sorted(absent)}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: str) -> bool:
    return value.strip() == ""


def _is_cold_start(row: dict[str, str]) -> bool:
    """Return whether a row predates the first completed metric snapshot.

    Snapshot-derived compact features must be jointly present or absent. A
    partial snapshot would be a data-quality error, not a cold-start state.
    """
    missing = [feature for feature in SNAPSHOT_FEATURES if _is_missing(row[feature])]
    if missing and len(missing) != len(SNAPSHOT_FEATURES):
        present = sorted(set(SNAPSHOT_FEATURES).difference(missing))
        raise ValueError(
            f"Partial metric snapshot in {row['record_id']}: "
            f"missing={missing}, present={present}"
        )
    return bool(missing)


def _feature_schema(rows: Sequence[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """Build the learner's schema from warm-start data only.

    Cold-start records have no completed metric snapshot, so feeding them to
    CONFOLD would turn missing context into a learned pattern.  Constant
    columns cannot help split a rule and are removed once for the exploratory
    all-data model; validation folds repeat that check using training data only.
    """
    if any(_is_cold_start(row) for row in rows):
        raise ValueError("Cold-start rows must not enter the CONFOLD feature schema")
    features = list(COMPACT_FEATURES)
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    values = [_transform_row(row, features) for row in rows]
    constants = [
        feature
        for index, feature in enumerate(features)
        if len({row[index] for row in values}) == 1
    ]
    retained = [feature for feature in features if feature not in constants]
    return retained, [item for item in numeric if item in retained], constants


def _transform_row(row: dict[str, str], features: Sequence[str]) -> list[str | float]:
    """Convert numeric cells to floats while keeping categorical states as strings."""
    transformed: list[str | float] = []
    for feature in features:
        if feature in NUMERIC_FEATURES:
            if _is_missing(row[feature]):
                raise ValueError(f"Blank numeric feature {feature!r} in {row['record_id']}")
            transformed.append(float(row[feature]))
        else:
            transformed.append(row[feature])
    return transformed


def _constant_features(rows: Sequence[list[Any]], features: Sequence[str]) -> list[str]:
    return [
        feature
        for index, feature in enumerate(features)
        if len({row[index] for row in rows}) == 1
    ]


def _drop_features(
    rows: Sequence[list[Any]], features: Sequence[str], drops: Iterable[str]
) -> tuple[list[list[Any]], list[str]]:
    """Remove the same feature positions from a group of rows and its schema."""
    dropped = set(drops)
    keep_indices = [index for index, feature in enumerate(features) if feature not in dropped]
    return [[row[index] for index in keep_indices] for row in rows], [features[index] for index in keep_indices]


def _fit(features: Sequence[str], numeric: Sequence[str], label: str, data: list[list[Any]]):
    """Train one binary CONFOLD model for one AP decision head.

    ``ratio=0.5`` is the vendor learner's purity setting: it stops adding main
    conditions once the remaining negative examples are no more than half the
    positives, then lets CONFOLD learn exceptions if needed.
    """
    model = make_classifier(features, numeric, label)
    # CONFOLD's ordinary fit already attaches Wilson-style rule confidence.
    model.fit(data, ratio=0.5)
    return model


def _predict(model: Any, feature_rows: Sequence[list[Any]]) -> list[tuple[str | None, float | None]]:
    return [(label, confidence) for label, confidence in model.predict(feature_rows)]


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(
    labels: Sequence[str], predictions: Sequence[tuple[str | None, float | None]]
) -> dict[str, Any]:
    """Summarise correctness, abstention, ON-class quality, and confidence.

    A ``None`` prediction means that no rule fired.  It deliberately remains
    distinct from an OFF prediction: it is incorrect for all-row accuracy and
    receives the vendor metric's neutral 0.25 Brier loss.
    """
    total = len(labels)
    fired = [(label, confidence) for label, confidence in predictions if label is not None]
    correct_all = sum(label == prediction for label, (prediction, _) in zip(labels, predictions))
    correct_fired = sum(
        label == prediction
        for label, (prediction, _) in zip(labels, predictions)
        if prediction is not None
    )
    tp = sum(label == "ON" and prediction == "ON" for label, (prediction, _) in zip(labels, predictions))
    fp = sum(label == "OFF" and prediction == "ON" for label, (prediction, _) in zip(labels, predictions))
    fn = sum(label == "ON" and prediction != "ON" for label, (prediction, _) in zip(labels, predictions))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    # Keep the vendor convention for an abstention so score comparisons remain
    # meaningful: a no-fire incurs Brier loss 0.25.
    brier = sum(
        0.25
        if prediction is None
        else (1 - confidence) ** 2
        if prediction == label
        else confidence**2
        for label, (prediction, confidence) in zip(labels, predictions)
    ) / total
    return {
        "n": total,
        "label_counts": dict(sorted(Counter(labels).items())),
        "fired": len(fired),
        "no_fire": total - len(fired),
        "coverage": _safe_divide(len(fired), total),
        "correct_all": correct_all,
        "accuracy_all_no_fire_incorrect": _safe_divide(correct_all, total),
        "accuracy_when_fired": _safe_divide(correct_fired, len(fired)),
        "on_true_positive": tp,
        "on_false_positive": fp,
        "on_false_negative": fn,
        "on_precision": precision,
        "on_recall": recall,
        "on_f1": f1,
        "mean_fired_confidence": _safe_divide(sum(confidence for _, confidence in fired), len(fired)),
        "inverse_brier_score": 1 - brier,
        "always_off_accuracy": _safe_divide(sum(label == "OFF" for label in labels), total),
    }


def _condition_dict(condition: tuple[Any, Any, Any], features: Sequence[str]) -> dict[str, Any]:
    """Turn CONFOLD's positional condition encoding into named report data."""
    index, relation, value = condition
    if index == -1:
        return {"kind": "class", "relation": relation, "value": value}
    if index < 0:
        index = -index - 2
        relation = {"<=": ">", ">": "<=", "==": "!=", "!=": "=="}[relation]
    return {"feature": features[index], "relation": relation, "value": value}


def _rule_dict(rule: tuple[Any, Any, Any, Any], features: Sequence[str]) -> dict[str, Any]:
    """Recursively turn one vendor rule, including exceptions, into JSON data."""
    target, conditions, exceptions, confidence = rule
    return {
        # Only a top-level rule predicts a class.  Exception rules carry the
        # vendor's placeholder in this position because they only block a rule.
        "predict": _condition_dict(target, features)["value"] if isinstance(target, tuple) else None,
        "conditions": [_condition_dict(condition, features) for condition in conditions],
        "exceptions": [_rule_dict(exception, features) for exception in exceptions],
        "confidence": confidence,
    }


def _condition_matches(condition: tuple[Any, Any, Any], values: Sequence[Any]) -> bool:
    index, relation, threshold = condition
    if index < 0:
        return False
    value = values[index]
    if isinstance(threshold, str):
        return {"==": value == threshold, "!=": value != threshold}.get(relation, False)
    value = float(value)
    return {
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }[relation]


def _rule_applies(rule: tuple[Any, Any, Any, Any], values: Sequence[Any]) -> bool:
    """Mirror CONFOLD's rule semantics for reporting coverage.

    A rule applies when every main condition matches and none of its exception
    rules applies.  The outer caller handles the separate first-match ordering
    between top-level rules.
    """
    _, conditions, exceptions, _ = rule
    return all(_condition_matches(condition, values) for condition in conditions) and not any(
        _rule_applies(exception, values) for exception in exceptions
    )


def _rule_summaries(model: Any, data: Sequence[list[Any]], features: Sequence[str]) -> list[dict[str, Any]]:
    """Describe both potential and actual training support for each rule.

    Raw coverage is every row whose conditions match.  Effective coverage is
    the smaller set that reaches this rule after earlier rules have had their
    first-match priority.  Showing both prevents a late, broad rule from
    looking more influential than it really is.
    """
    summaries = []
    for number, rule in enumerate(model.rules or [], start=1):
        raw_covered = [row for row in data if _rule_applies(rule, row[:-1])]
        effective_covered = [
            row
            for row in data
            if _rule_applies(rule, row[:-1])
            and not any(_rule_applies(previous, row[:-1]) for previous in (model.rules or [])[: number - 1])
        ]
        item = _rule_dict(rule, features)
        item.update(
            {
                "rule_number": number,
                "raw_condition_coverage": len(raw_covered),
                "raw_condition_coverage_fraction": _safe_divide(len(raw_covered), len(data)),
                "effective_first_match_coverage": len(effective_covered),
                "effective_first_match_coverage_fraction": _safe_divide(len(effective_covered), len(data)),
                "effective_first_match_precision_for_predicted_label": _safe_divide(
                    sum(row[-1] == item["predict"] for row in effective_covered), len(effective_covered)
                ),
            }
        )
        summaries.append(item)
    return summaries


def _vendor_asp_rules(model: Any, features: Sequence[str], label: str) -> list[str]:
    """Render optional vendor ASP with the label slot its decoder expects.

    AP4Fed trains from in-memory rows rather than the vendor CSV loader.  This
    avoids its naive comma parser, but also bypasses the loader's habit of
    adding the label to ``attrs``.  Learning itself only needs positions; the
    vendor renderer needs that final label name to print the rule head.
    """
    original_attrs = model.attrs
    model.attrs = [*features, label]
    model.asp_rules = None
    try:
        return model.asp(simple=True)
    finally:
        model.attrs = original_attrs


def _write_csv(path: Path, features: Sequence[str], rows: Sequence[list[Any]], label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "run_id", *features, label])
        writer.writerows(rows)


def _round_report_metrics(value: Any) -> Any:
    """Round floating-point metrics for the human-readable report only."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_report_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_report_metrics(item) for item in value]
    return value


def _format_rule_decimal(value: float) -> str:
    """Render a rule boundary consistently without changing stored precision."""
    return f"{value:.5f}"


def _write_text_report(path: Path, result: dict[str, Any]) -> None:
    """Write a compact, human-readable companion to the machine-readable JSON."""
    lines = [
        "AP4Fed CONFOLD baseline",
        "",
        "Cold-start policy: decisions without a completed metric snapshot use a fixed OFF action",
        "and are excluded from CONFOLD fitting and rule evaluation. The canonical decision_dataset.csv is unchanged.",
        f"Cold-start / warm-start rows: {result['cold_start_policy']['n_rows']} / {result['n_warm_start_rows']}",
        "",
        f"Dataset: {result['dataset']}",
        f"Total rows / runs: {result['n_rows']} / {result['n_runs']}",
        f"Compact core: {', '.join(COMPACT_FEATURES)}",
        f"Globally dropped constant features: {', '.join(result['global_constant_features']) or '(none)'}",
        "",
    ]
    for head, payload in result["heads"].items():
        fit = payload["exploratory_fit"]
        loro = payload["leave_one_run_out"]
        lines.extend(
            [
                f"## {head.upper()}",
                "Training-set metrics: "
                f"{json.dumps(_round_report_metrics(fit['metrics']), sort_keys=True)}",
                "Learned rules:",
                *(
                    _render_rule(rule)
                    for rule in fit["rules"]
                ),
                *([] if fit["rules"] else ["(no rules learned; every prediction is a no-fire)"]),
                "Rule confidence and training coverage:",
                *(
                    f"  rule {rule['rule_number']}: predict {rule['predict']}; "
                    f"confidence={_format_rule_decimal(rule['confidence'])}; effective first-match coverage="
                    f"{rule['effective_first_match_coverage']}/{fit['metrics']['n']} "
                    f"(raw condition matches={rule['raw_condition_coverage']})"
                    for rule in fit["rules"]
                ),
                "Leave-one-run-out aggregate metrics: "
                f"{json.dumps(_round_report_metrics(loro['aggregate_metrics']), sort_keys=True)}",
                "Cold-start fixed-OFF accuracy: "
                f"{_format_rule_decimal(loro['cold_start_policy']['accuracy'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_condition(condition: dict[str, Any]) -> str:
    value = condition["value"]
    rendered = _format_rule_decimal(value) if isinstance(value, float) else repr(value)
    return f"{condition['feature']} {condition['relation']} {rendered}"


def _render_rule(rule: dict[str, Any]) -> str:
    conditions = " AND ".join(_render_condition(item) for item in rule["conditions"]) or "TRUE"
    exceptions = "; ".join(
        " AND ".join(_render_condition(item) for item in exception["conditions"]) or "TRUE"
        for exception in rule["exceptions"]
    )
    exception_text = f" EXCEPT IF ({exceptions})" if exceptions else ""
    return (
        f"  rule {rule['rule_number']}: predict {rule['predict']} IF {conditions}{exception_text} "
        f"[confidence={_format_rule_decimal(rule['confidence'])}, "
        f"effective coverage={rule['effective_first_match_coverage']}, "
        f"raw matches={rule['raw_condition_coverage']}]"
    )


def run(
    dataset: Path,
    output: Path,
) -> dict[str, Any]:
    """Run the complete, leakage-safe baseline and write its audit trail.

    The all-warm-data fit is solely for reading rules.  The reported evaluation
    is leave-one-run-out: each full federated trajectory is held back together,
    so nearby rounds from the same trajectory cannot leak from training to test.
    Cold starts use the declared fixed-OFF policy rather than a model prediction.
    """
    rows = _read_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    views_dir = output / "training_views"
    views_dir.mkdir(exist_ok=True)

    cold_start_indices = [index for index, row in enumerate(rows) if _is_cold_start(row)]
    cold_start_index_set = set(cold_start_indices)
    warm_start_indices = [index for index in range(len(rows)) if index not in cold_start_index_set]
    if not warm_start_indices:
        raise ValueError("Dataset has no warm-start rows to train")
    warm_rows = [rows[index] for index in warm_start_indices]
    retained, numeric, global_constants = _feature_schema(warm_rows)
    transformed = {index: _transform_row(rows[index], retained) for index in warm_start_indices}

    run_ids = sorted({row["run_id"] for row in rows})
    result: dict[str, Any] = {
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "n_rows": len(rows),
        "n_warm_start_rows": len(warm_start_indices),
        "n_runs": len(run_ids),
        "run_ids": run_ids,
        "compact_features": list(COMPACT_FEATURES),
        "cold_start_policy": {
            "n_rows": len(cold_start_indices),
            "record_ids": [rows[index]["record_id"] for index in cold_start_indices],
            "snapshot_features": list(SNAPSHOT_FEATURES),
            "fixed_prediction": COLD_START_PREDICTION,
            "canonical_dataset_modified": False,
        },
        "global_constant_features": global_constants,
        "retained_features": retained,
        "heads": {},
    }

    for head, label_name in HEADS.items():
        view_rows = [
            [rows[index]["record_id"], rows[index]["run_id"], *transformed[index], rows[index][label_name]]
            for index in warm_start_indices
        ]
        _write_csv(views_dir / f"{head}_training_view.csv", retained, view_rows, label_name)

        all_data = [transformed[index] + [rows[index][label_name]] for index in warm_start_indices]
        # This fit gives us a readable model trained on all available warm data.
        # It is exploratory only; performance is measured by the folds below.
        exploratory = _fit(retained, numeric, label_name, all_data)
        exploratory_predictions = _predict(exploratory, [transformed[index] for index in warm_start_indices])
        exploratory_payload = {
            "metrics": _metrics([rows[index][label_name] for index in warm_start_indices], exploratory_predictions),
            "asp_rules": _vendor_asp_rules(exploratory, retained, label_name),
            "rules": _rule_summaries(exploratory, all_data, retained),
        }

        heldout_predictions: list[tuple[str | None, float | None] | None] = [None] * len(rows)
        folds: list[dict[str, Any]] = []
        for run_id in run_ids:
            # Keep an entire FL trajectory on one side of the split.  A random
            # row split would let adjacent rounds from the same run leak across.
            test_indices = [index for index in warm_start_indices if rows[index]["run_id"] == run_id]
            train_indices = [index for index in warm_start_indices if rows[index]["run_id"] != run_id]
            cold_test_indices = [index for index in cold_start_indices if rows[index]["run_id"] == run_id]
            train_features = [transformed[index] for index in train_indices]
            test_features = [transformed[index] for index in test_indices]
            # Derive fold-specific constants from training data alone.  The test
            # run must not influence even this small schema decision.
            fold_constants = _constant_features(train_features, retained)
            fold_features_data, fold_features = _drop_features(train_features, retained, fold_constants)
            fold_test_data, _ = _drop_features(test_features, retained, fold_constants)
            fold_numeric = [feature for feature in numeric if feature in fold_features]
            fold_train = [features + [rows[index][label_name]] for index, features in zip(train_indices, fold_features_data)]
            fold_model = _fit(fold_features, fold_numeric, label_name, fold_train)
            predictions = _predict(fold_model, fold_test_data)
            for index, prediction in zip(test_indices, predictions):
                heldout_predictions[index] = prediction
            for index in cold_test_indices:
                # There is no completed snapshot to evaluate here, so this is
                # policy output rather than a CONFOLD prediction.
                heldout_predictions[index] = (COLD_START_PREDICTION, None)
            folds.append(
                {
                    "held_out_run_id": run_id,
                    "n_train": len(train_indices),
                    "n_test_warm_start": len(test_indices),
                    "n_test_cold_start": len(cold_test_indices),
                    "dropped_constant_features_from_training_fold": fold_constants,
                    "metrics": _metrics([rows[index][label_name] for index in test_indices], predictions),
                    "cold_start_accuracy": _safe_divide(
                        sum(rows[index][label_name] == COLD_START_PREDICTION for index in cold_test_indices),
                        len(cold_test_indices),
                    ),
                    "rules": _rule_summaries(fold_model, fold_train, fold_features),
                }
            )

        if any(prediction is None for prediction in heldout_predictions):
            raise AssertionError("A leave-one-run-out row was not evaluated")
        complete_predictions = [prediction for prediction in heldout_predictions if prediction is not None]
        warm_predictions = [heldout_predictions[index] for index in warm_start_indices]
        if any(prediction is None for prediction in warm_predictions):
            raise AssertionError("A warm-start leave-one-run-out row was not evaluated")
        complete_warm_predictions = [prediction for prediction in warm_predictions if prediction is not None]
        prediction_rows = [
            {
                "record_id": row["record_id"],
                "run_id": row["run_id"],
                "actual": row[label_name],
                "predicted": prediction[0],
                "confidence": prediction[1],
                "source": "cold_start_policy" if index in cold_start_index_set else "confold",
                "fired": prediction[0] is not None and index not in cold_start_index_set,
            }
            for index, (row, prediction) in enumerate(zip(rows, complete_predictions))
        ]
        loro_payload = {
            "validation": "10 folds; each fold holds out one complete run_id. Metrics evaluate warm-start rows only.",
            "aggregate_metrics": _metrics(
                [rows[index][label_name] for index in warm_start_indices], complete_warm_predictions
            ),
            "cold_start_policy": {
                "prediction": COLD_START_PREDICTION,
                "n_rows": len(cold_start_indices),
                "accuracy": _safe_divide(
                    sum(rows[index][label_name] == COLD_START_PREDICTION for index in cold_start_indices),
                    len(cold_start_indices),
                ),
            },
            "folds": folds,
            "predictions": prediction_rows,
        }
        result["heads"][head] = {
            "label": label_name,
            "exploratory_fit": exploratory_payload,
            "leave_one_run_out": loro_payload,
        }

    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rules_dir = output / "rules"
    rules_dir.mkdir(exist_ok=True)
    for head, payload in result["heads"].items():
        (rules_dir / f"{head}_exploratory_rules.json").write_text(
            json.dumps(payload["exploratory_fit"], indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    _write_text_report(output / "report.txt", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    if args.dataset.resolve().is_relative_to(args.output.resolve()):
        parser.error("Output directory must not contain the canonical input dataset.")
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            parser.error(f"Output exists: {args.output}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output)
    result = run(args.dataset, args.output)
    print(f"Wrote {args.output}")
    for head, payload in result["heads"].items():
        metrics = payload["leave_one_run_out"]["aggregate_metrics"]
        print(
            f"{head.upper()}: LORO accuracy={metrics['accuracy_all_no_fire_incorrect']:.3f}, "
            f"coverage={metrics['coverage']:.3f}, ON F1={metrics['on_f1']}"
        )


if __name__ == "__main__":
    main()
