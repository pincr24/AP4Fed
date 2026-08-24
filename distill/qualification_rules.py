"""Build and load the provisional Sprint 01 qualification rule set.

The Sprint 01 CONFOLD output is an ordered decision list with nested
exceptions. The runtime dispatcher uses independent unordered rules, so the
source reports cannot be loaded directly. This module performs one pinned,
mechanical translation into disjoint rules and labels the result explicitly as
qualification-only evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from policy_interfaces import (
    FEATURE_SCHEMA_ID,
    HEADS,
    validate_rule_logic,
)


QUALIFICATION_ARTIFACT_SCHEMA_VERSION = 1
QUALIFICATION_ARTIFACT_KIND = "qualification_only"
QUALIFICATION_INTENDED_USE = "closed_loop_qualification"
QUALIFICATION_RULE_SET_ID = "qualification/sprint01-confold-disjoint-v1"
SOURCE_LABEL_SCHEMA_ID = "ap4fed-sprint01-decision-dataset-v4"
TRANSFORMATION_ID = "confold-ordered-to-disjoint-v1"
CODE_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

DATASET_SHA256 = "de3833a0f2236048e2d4c802ffbd8044ae645412fb028ce034084c9e16a64233"
SOURCE_RULE_SHA256 = {
    "client_selector": "698e43697525effd09d6e1fdc38528d576f81f11093f024ff2566850260d9501",
    "message_compressor": "4afc0b8a23dc56b443e0125b3cd722c6d70d36b08ea26c1f6bf095ec83e3e4d8",
    "heterogeneous_data_handler": "1e6332a396b73ed7eda182e96320ce37ef81a834a718660fb9fb409a574cd57f",
}
SOURCE_FILENAMES = {
    "client_selector": "cs_exploratory_rules.json",
    "message_compressor": "mc_exploratory_rules.json",
    "heterogeneous_data_handler": "hdh_exploratory_rules.json",
}
LABEL_COLUMNS = {
    "client_selector": "y_cs_applied",
    "message_compressor": "y_mc_applied",
    "heterogeneous_data_handler": "y_hdh_applied",
}

EXPECTED_CONFIGURATION_SCOPE = {
    "workload": "AG_NEWS__MLP",
    "rounds": 10,
    "clients": 5,
    "partition_seed": 2764335072,
    "client_selector_value": 3,
    "client_layout": [
        {
            "client_id": 1,
            "cpu": 5,
            "ram": 2,
            "dataset": "AG_NEWS",
            "model": "MLP",
            "data_distribution_type": "non-IID",
            "non_iid_alpha": 0.9,
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "delay_min_seconds": 0,
            "delay_max_seconds": 0,
            "epochs": 1,
        },
        {
            "client_id": 2,
            "cpu": 5,
            "ram": 2,
            "dataset": "AG_NEWS",
            "model": "MLP",
            "data_distribution_type": "non-IID",
            "non_iid_alpha": 0.9,
            "data_persistence_type": "New Data",
            "delay_combobox": "No",
            "delay_min_seconds": 0,
            "delay_max_seconds": 0,
            "epochs": 1,
        },
        {
            "client_id": 3,
            "cpu": 5,
            "ram": 2,
            "dataset": "AG_NEWS",
            "model": "MLP",
            "data_distribution_type": "IID",
            "non_iid_alpha": 0.5,
            "data_persistence_type": "New Data",
            "delay_combobox": "Yes",
            "delay_min_seconds": 20,
            "delay_max_seconds": 50,
            "epochs": 1,
        },
        {
            "client_id": 4,
            "cpu": 3,
            "ram": 2,
            "dataset": "AG_NEWS",
            "model": "MLP",
            "data_distribution_type": "non-IID",
            "non_iid_alpha": 0.9,
            "data_persistence_type": "New Data",
            "delay_combobox": "Yes",
            "delay_min_seconds": 20,
            "delay_max_seconds": 50,
            "epochs": 1,
        },
        {
            "client_id": 5,
            "cpu": 3,
            "ram": 2,
            "dataset": "AG_NEWS",
            "model": "MLP",
            "data_distribution_type": "IID",
            "non_iid_alpha": 0.5,
            "data_persistence_type": "Same Data",
            "delay_combobox": "No",
            "delay_min_seconds": 0,
            "delay_max_seconds": 0,
            "epochs": 1,
        },
    ],
}
CLIENT_SCOPE_FIELDS = tuple(EXPECTED_CONFIGURATION_SCOPE["client_layout"][0])


@dataclass(frozen=True)
class LoadedQualificationArtifact:
    """A checked provisional artifact together with its exact file hash."""

    document: dict[str, object]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_conditions(value: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(condition) for condition in value]


def _negate_condition(condition: Mapping[str, object]) -> dict[str, object]:
    inverse = {
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "==": "!=",
        "!=": "==",
    }
    relation = str(condition["relation"])
    if relation not in inverse:
        raise ValueError(f"cannot negate unsupported relation {relation!r}")
    return {
        "feature": condition["feature"],
        "relation": inverse[relation],
        "value": condition["value"],
    }


def _flatten_exception(exception: Mapping[str, object]) -> list[list[dict[str, object]]]:
    """Translate one source exception into flat blocking conjunctions."""
    conditions = _copy_conditions(exception.get("conditions", []))
    nested = exception.get("exceptions")
    if not isinstance(nested, list):
        raise ValueError("source exception must contain an exceptions list")
    if not nested:
        return [conditions]
    if len(nested) != 1 or nested[0].get("exceptions"):
        raise ValueError("the pinned Sprint 01 translator supports one nested exception level")
    nested_conditions = nested[0].get("conditions")
    if not isinstance(nested_conditions, list) or not nested_conditions:
        raise ValueError("nested exception must contain conditions")
    # base AND NOT(n1 AND n2) == (base AND NOT n1) OR (base AND NOT n2)
    return [
        [*conditions, _negate_condition(nested_condition)]
        for nested_condition in nested_conditions
    ]


def _flat_exceptions(rule: Mapping[str, object]) -> list[list[dict[str, object]]]:
    source = rule.get("exceptions")
    if not isinstance(source, list):
        raise ValueError("source rule must contain an exceptions list")
    return [group for exception in source for group in _flatten_exception(exception)]


def _assert_source_rule(
    rule: Mapping[str, object],
    predicted: str,
    condition_features: Sequence[str],
    exception_count: int,
) -> None:
    conditions = rule.get("conditions")
    exceptions = rule.get("exceptions")
    if (
        rule.get("predict") != predicted
        or not isinstance(conditions, list)
        or [item.get("feature") for item in conditions] != list(condition_features)
        or not isinstance(exceptions, list)
        or len(exceptions) != exception_count
    ):
        raise ValueError("pinned Sprint 01 rule structure changed; refusing translation")


def _action(head: str, predicted: str) -> dict[str, object]:
    result: dict[str, object] = {"enabled": predicted}
    if head == "client_selector" and predicted == "ON":
        result["selection_value"] = EXPECTED_CONFIGURATION_SCOPE[
            "client_selector_value"
        ]
    return result


def _runtime_rule(
    head: str,
    source_rule: Mapping[str, object],
    source_number: int,
    suffix: str,
    conditions: Sequence[Mapping[str, object]],
    exceptions: Sequence[Sequence[Mapping[str, object]]] = (),
) -> dict[str, object]:
    predicted = str(source_rule["predict"])
    return {
        "rule_id": f"qualification.sprint01.{head}.r{source_number}.{suffix}",
        "action": _action(head, predicted),
        "conditions": _copy_conditions(conditions),
        "exceptions": [_copy_conditions(group) for group in exceptions],
        "training_evidence": {
            "coverage": 0,
            "support": 0,
            "precision": None,
        },
        "provenance": {
            "source_head": head,
            "source_rule_number": source_number,
            "source_confidence": float(source_rule["confidence"]),
            "source_artifact_sha256": SOURCE_RULE_SHA256[head],
        },
    }


def _compile_client_selector(source: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(source) != 3:
        raise ValueError("expected three pinned Client Selector rules")
    off_high, off_low, on_default = source
    _assert_source_rule(off_high, "OFF", ("comm_frac",), 3)
    _assert_source_rule(off_low, "OFF", ("comm_frac",), 0)
    _assert_source_rule(on_default, "ON", ("prev_cs",), 0)
    exceptions = _flat_exceptions(off_high)
    rules = [
        _runtime_rule(
            "client_selector", off_high, 1, "off-high",
            off_high["conditions"], exceptions,
        ),
        _runtime_rule(
            "client_selector", off_low, 2, "off-low",
            off_low["conditions"],
        ),
    ]
    for index, exception in enumerate(exceptions, start=1):
        rules.append(_runtime_rule(
            "client_selector", on_default, 3, f"on-hole-{index}",
            [*on_default["conditions"], *off_high["conditions"], *exception],
        ))
    return rules


def _compile_message_compressor(source: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(source) != 3:
        raise ValueError("expected three pinned Message Compressor rules")
    off_fast, off_previous, on_default = source
    _assert_source_rule(off_fast, "OFF", ("train_mean",), 0)
    _assert_source_rule(off_previous, "OFF", ("prev_hdh",), 3)
    _assert_source_rule(on_default, "ON", ("prev_cs",), 0)
    not_off_fast = _negate_condition(off_fast["conditions"][0])
    previous_conditions = [not_off_fast, *off_previous["conditions"]]
    exceptions = _flat_exceptions(off_previous)
    rules = [
        _runtime_rule(
            "message_compressor", off_fast, 1, "off-fast",
            off_fast["conditions"],
        ),
        _runtime_rule(
            "message_compressor", off_previous, 2, "off-previous",
            previous_conditions, exceptions,
        ),
        _runtime_rule(
            "message_compressor", on_default, 3, "on-previous-not-off",
            [
                *on_default["conditions"],
                not_off_fast,
                _negate_condition(off_previous["conditions"][0]),
            ],
        ),
    ]
    for index, exception in enumerate(exceptions, start=1):
        rules.append(_runtime_rule(
            "message_compressor", on_default, 3, f"on-hole-{index}",
            [*on_default["conditions"], *previous_conditions, *exception],
        ))
    return rules


def _compile_hdh(source: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if len(source) != 2:
        raise ValueError("expected two pinned HDH rules")
    off_normal, on_default = source
    _assert_source_rule(off_normal, "OFF", ("total_round_time",), 3)
    _assert_source_rule(on_default, "ON", ("prev_cs",), 0)
    exceptions = _flat_exceptions(off_normal)
    rules = [
        _runtime_rule(
            "heterogeneous_data_handler", off_normal, 1, "off-normal",
            off_normal["conditions"], exceptions,
        ),
        _runtime_rule(
            "heterogeneous_data_handler", on_default, 2, "on-slow",
            [
                *on_default["conditions"],
                _negate_condition(off_normal["conditions"][0]),
            ],
        ),
    ]
    for index, exception in enumerate(exceptions, start=1):
        rules.append(_runtime_rule(
            "heterogeneous_data_handler", on_default, 2, f"on-hole-{index}",
            [*on_default["conditions"], *off_normal["conditions"], *exception],
        ))
    return rules


def _condition_matches(condition: Mapping[str, object], state: Mapping[str, str]) -> bool:
    raw = state.get(str(condition["feature"]), "")
    if raw == "":
        return False
    target = condition["value"]
    if isinstance(target, str):
        value: object = raw
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False
    relation = condition["relation"]
    return {
        "<": value < target,
        "<=": value <= target,
        ">": value > target,
        ">=": value >= target,
        "==": value == target,
        "!=": value != target,
    }[relation]


def rule_matches(rule: Mapping[str, object], state: Mapping[str, str]) -> bool:
    """Return whether one normalized rule fires for a normalized state."""
    conditions = rule["conditions"]
    exceptions = rule["exceptions"]
    return all(_condition_matches(item, state) for item in conditions) and not any(
        all(_condition_matches(item, state) for item in group)
        for group in exceptions
    )


def _source_rule_matches(
    rule: Mapping[str, object], state: Mapping[str, str],
) -> bool:
    conditions = rule.get("conditions")
    exceptions = rule.get("exceptions")
    if not isinstance(conditions, list) or not isinstance(exceptions, list):
        raise ValueError("source rule has an invalid condition structure")
    return all(_condition_matches(item, state) for item in conditions) and not any(
        _source_rule_matches(exception, state) for exception in exceptions
    )


def _assert_semantic_equivalence(
    source: Sequence[Mapping[str, object]],
    compiled: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
    head: str,
) -> None:
    for row in rows:
        expected = next(
            (rule["predict"] for rule in source if _source_rule_matches(rule, row)),
            None,
        )
        fired = {
            rule["action"]["enabled"]
            for rule in compiled if rule_matches(rule, row)
        }
        if expected is None or fired != {expected}:
            raise ValueError(
                f"{head}: translated rules differ from the ordered source at "
                f"{row.get('record_id', '<unknown>')}"
            )


def _add_training_evidence(
    rules: Sequence[dict[str, object]],
    rows: Sequence[Mapping[str, str]],
    label_column: str,
) -> None:
    for rule in rules:
        covered = [row for row in rows if rule_matches(rule, row)]
        expected = rule["action"]["enabled"]
        support = sum(row[label_column] == expected for row in covered)
        rule["training_evidence"] = {
            "coverage": len(covered),
            "support": support,
            "precision": support / len(covered) if covered else None,
        }


def build_qualification_artifact(
    source_root: Path,
    producing_code_sha: str,
) -> dict[str, object]:
    """Translate the pinned Sprint 01 reports into one provisional artifact."""
    if not CODE_SHA_RE.fullmatch(producing_code_sha):
        raise ValueError("producing_code_sha must be a full Git SHA")
    dataset = source_root / "decision_dataset.csv"
    rules_root = source_root / "confold_baseline" / "rules"
    if _sha256(dataset) != DATASET_SHA256:
        raise ValueError("Sprint 01 decision dataset hash differs from the pinned input")

    source_documents: dict[str, dict[str, object]] = {}
    source_records: dict[str, dict[str, str]] = {}
    for head in HEADS:
        path = rules_root / SOURCE_FILENAMES[head]
        digest = _sha256(path)
        if digest != SOURCE_RULE_SHA256[head]:
            raise ValueError(f"{head}: exploratory rule hash differs from the pinned input")
        source_documents[head] = json.loads(path.read_text(encoding="utf-8"))
        source_records[head] = {
            "path": f"confold_baseline/rules/{path.name}",
            "sha256": digest,
        }

    compiled = {
        "client_selector": _compile_client_selector(
            source_documents["client_selector"]["rules"]
        ),
        "message_compressor": _compile_message_compressor(
            source_documents["message_compressor"]["rules"]
        ),
        "heterogeneous_data_handler": _compile_hdh(
            source_documents["heterogeneous_data_handler"]["rules"]
        ),
    }
    with dataset.open(newline="", encoding="utf-8") as stream:
        warm_rows = [row for row in csv.DictReader(stream) if row["val_f1"] != ""]
    if len(warm_rows) != 80:
        raise ValueError("expected the frozen eighty-row Sprint 01 warm-start sample")
    for head in HEADS:
        _add_training_evidence(compiled[head], warm_rows, LABEL_COLUMNS[head])
        _assert_semantic_equivalence(
            source_documents[head]["rules"], compiled[head], warm_rows, head,
        )

    artifact: dict[str, object] = {
        "schema_version": QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": QUALIFICATION_ARTIFACT_KIND,
        "intended_use": QUALIFICATION_INTENDED_USE,
        "rule_set_id": QUALIFICATION_RULE_SET_ID,
        "feature_schema": FEATURE_SCHEMA_ID,
        "source_label_schema": SOURCE_LABEL_SCHEMA_ID,
        "created_from": {
            "dataset_sha256": DATASET_SHA256,
            "source_rules": source_records,
            "miner": "CONFOLD",
            "transformation": TRANSFORMATION_ID,
            "producing_code_sha": producing_code_sha,
        },
        "source_teacher": {
            "policy": "Single AI-Agent (Few-Shot)",
            "model": "deepseek-r1:8b",
            "model_digest": None,
            "environment_identity": "not-retained-in-paper-archive",
        },
        "configuration_scope": json.loads(json.dumps(EXPECTED_CONFIGURATION_SCOPE)),
        "heads": {head: {"rules": compiled[head]} for head in HEADS},
    }
    validate_qualification_artifact(artifact)
    return artifact


def _exact_keys(value: object, expected: Sequence[str], location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if missing or unknown:
        raise ValueError(
            f"{location}: fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def validate_qualification_artifact(value: object) -> None:
    """Reject a provisional artifact whose logic or provenance is incomplete."""
    artifact = _exact_keys(value, (
        "schema_version", "artifact_kind", "intended_use", "rule_set_id",
        "feature_schema", "source_label_schema", "created_from",
        "source_teacher", "configuration_scope", "heads",
    ), "qualification_artifact")
    expected_scalars = {
        "schema_version": QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": QUALIFICATION_ARTIFACT_KIND,
        "intended_use": QUALIFICATION_INTENDED_USE,
        "rule_set_id": QUALIFICATION_RULE_SET_ID,
        "feature_schema": FEATURE_SCHEMA_ID,
        "source_label_schema": SOURCE_LABEL_SCHEMA_ID,
    }
    for field, expected in expected_scalars.items():
        if artifact[field] != expected:
            raise ValueError(f"qualification_artifact.{field}: unexpected value")

    created = _exact_keys(artifact["created_from"], (
        "dataset_sha256", "source_rules", "miner", "transformation",
        "producing_code_sha",
    ), "qualification_artifact.created_from")
    if created["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("qualification artifact uses the wrong dataset")
    if created["miner"] != "CONFOLD" or created["transformation"] != TRANSFORMATION_ID:
        raise ValueError("qualification artifact uses an unsupported producer")
    if not isinstance(created["producing_code_sha"], str) or not CODE_SHA_RE.fullmatch(
        created["producing_code_sha"]
    ):
        raise ValueError("qualification artifact requires a producing Git SHA")
    source_rules = _exact_keys(created["source_rules"], HEADS, "source_rules")
    for head in HEADS:
        record = _exact_keys(source_rules[head], ("path", "sha256"), f"source_rules.{head}")
        if record["path"] != f"confold_baseline/rules/{SOURCE_FILENAMES[head]}":
            raise ValueError(f"source_rules.{head}: path differs from pinned source")
        if record["sha256"] != SOURCE_RULE_SHA256[head]:
            raise ValueError(f"source_rules.{head}: hash differs from pinned source")

    teacher = _exact_keys(artifact["source_teacher"], (
        "policy", "model", "model_digest", "environment_identity",
    ), "qualification_artifact.source_teacher")
    if (
        teacher["policy"] != "Single AI-Agent (Few-Shot)"
        or teacher["model"] != "deepseek-r1:8b"
        or teacher["model_digest"] is not None
        or teacher["environment_identity"] != "not-retained-in-paper-archive"
    ):
        raise ValueError("qualification artifact source-teacher provenance changed")
    if artifact["configuration_scope"] != EXPECTED_CONFIGURATION_SCOPE:
        raise ValueError("qualification artifact has the wrong configuration scope")

    heads = _exact_keys(artifact["heads"], HEADS, "qualification_artifact.heads")
    seen: set[str] = set()
    for head in HEADS:
        head_value = _exact_keys(heads[head], ("rules",), f"heads.{head}")
        rules = head_value["rules"]
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"heads.{head}.rules: expected a non-empty list")
        for index, item in enumerate(rules):
            location = f"heads.{head}.rules[{index}]"
            rule = _exact_keys(item, (
                "rule_id", "action", "conditions", "exceptions",
                "training_evidence", "provenance",
            ), location)
            validate_rule_logic(rule, head, location)
            rule_id = str(rule["rule_id"])
            if rule_id in seen:
                raise ValueError(f"duplicate qualification rule ID {rule_id!r}")
            seen.add(rule_id)
            evidence = _exact_keys(rule["training_evidence"], (
                "coverage", "support", "precision",
            ), f"{location}.training_evidence")
            coverage = evidence["coverage"]
            support = evidence["support"]
            precision = evidence["precision"]
            if (
                isinstance(coverage, bool) or not isinstance(coverage, int) or coverage < 0
                or isinstance(support, bool) or not isinstance(support, int)
                or support < 0 or support > coverage
            ):
                raise ValueError(f"{location}: invalid training evidence counts")
            expected_precision = support / coverage if coverage else None
            if expected_precision is None:
                if precision is not None:
                    raise ValueError(f"{location}: zero coverage requires null precision")
            elif (
                isinstance(precision, bool)
                or not isinstance(precision, (int, float))
                or not math.isclose(float(precision), expected_precision)
            ):
                raise ValueError(f"{location}: training precision differs from support/coverage")
            provenance = _exact_keys(rule["provenance"], (
                "source_head", "source_rule_number", "source_confidence",
                "source_artifact_sha256",
            ), f"{location}.provenance")
            if (
                provenance["source_head"] != head
                or not isinstance(provenance["source_rule_number"], int)
                or provenance["source_rule_number"] < 1
                or not isinstance(provenance["source_confidence"], (int, float))
                or provenance["source_artifact_sha256"] != SOURCE_RULE_SHA256[head]
            ):
                raise ValueError(f"{location}: invalid source-rule provenance")


def load_qualification_artifact(path: Path) -> LoadedQualificationArtifact:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except Exception as error:
        raise ValueError(f"{path}: invalid qualification artifact JSON") from error
    validate_qualification_artifact(document)
    return LoadedQualificationArtifact(
        document=document,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _project_client(client: Mapping[str, object]) -> dict[str, object]:
    return {field: client.get(field) for field in CLIENT_SCOPE_FIELDS}


def validate_qualification_configuration(config: Mapping[str, object]) -> None:
    """Require the exact predeclared AG News qualification configuration."""
    for field in ("rounds", "clients", "partition_seed"):
        if config.get(field) != EXPECTED_CONFIGURATION_SCOPE[field]:
            raise ValueError(f"qualification configuration has unexpected {field}")
    if config.get("clients_per_round") != EXPECTED_CONFIGURATION_SCOPE["clients"]:
        raise ValueError("qualification configuration has unexpected clients_per_round")
    if config.get("dataset") != "AG_NEWS":
        raise ValueError("qualification configuration must use AG_NEWS")
    clients = config.get("client_details")
    if not isinstance(clients, list):
        raise ValueError("qualification configuration requires client_details")
    projected = [
        _project_client(client) if isinstance(client, dict) else {}
        for client in clients
    ]
    if projected != EXPECTED_CONFIGURATION_SCOPE["client_layout"]:
        raise ValueError("qualification configuration differs from the AG News client layout")
    patterns = config.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError("qualification configuration requires patterns")
    for head in HEADS:
        pattern = patterns.get(head)
        if not isinstance(pattern, dict) or pattern.get("enabled") is not False:
            raise ValueError(f"qualification configuration must start with {head} OFF")
    selector_params = patterns["client_selector"].get("params")
    if (
        not isinstance(selector_params, dict)
        or selector_params.get("selection_value")
        != EXPECTED_CONFIGURATION_SCOPE["client_selector_value"]
    ):
        raise ValueError("qualification configuration must use Client Selector threshold 3")
