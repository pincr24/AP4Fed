"""Define and validate the four documents of the Sprint 03 decide-or-defer path.

This module is the shared interface between rule mining, live dispatch, and the
retained evidence: the rule artifact, the dispatch request, the dispatch result,
and the decision trace. Each validator checks an exact key set, so a missing or
unexpected field fails at the boundary instead of reaching runtime, and each
format carries its own schema version constant. The checks use the standard
library only and never evaluate a rule or select a policy; `canonical_sha256`
supplies the stable encoding that hashes artifacts and traces for identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from decision_state import FEATURE_COLUMNS


RULE_ARTIFACT_SCHEMA_VERSION = 1
DISPATCH_SCHEMA_VERSION = 1
DECISION_TRACE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_ID = "ap4fed-feature-spec-v1"
LABEL_SCHEMA_ID = "ap4fed-normalized-labels-v5"
HEADS = (
    "client_selector",
    "message_compressor",
    "heterogeneous_data_handler",
)
MODES = ("always_defer", "shadow", "active")
RELATIONS = ("==", "!=", "<", "<=", ">", ">=")
STATE_CATEGORICAL_FEATURES = {
    "workload",
    "has_delay_clients",
    "prev_cs",
    "prev_mc",
    "prev_hdh",
}
STATE_INTEGER_FEATURES = {
    "n_clients",
    "max_cpu",
    "second_highest_cpu",
    "cpu_spread",
    "rs_cs_on",
    "rs_mc_on",
    "rs_hdh_on",
    "round_idx",
    "rounds_remaining",
}
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CODE_SHA_RE = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class LoadedRuleArtifact:
    """A validated rule-set document together with the hash of its source file."""

    document: dict[str, object]
    sha256: str


def canonical_sha256(value: object) -> str:
    """Hash a JSON value using the stable encoding shared by decision traces."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: Sequence[str], location: str,
) -> None:
    expected_set = set(expected)
    missing = expected_set - set(value)
    unknown = set(value) - expected_set
    if missing or unknown:
        raise ValueError(
            f"{location}: fields differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{location}: expected a stable identifier")
    return value


def _sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{location}: expected a lowercase SHA-256 digest")
    return value


def _number(value: object, location: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: expected a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{location}: expected a finite number >= {minimum}")
    return result


def _integer(value: object, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{location}: expected an integer >= {minimum}")
    return value


def _string_list(value: object, location: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{location}: expected a non-empty list")
    result = []
    for index, item in enumerate(value):
        result.append(_identifier(item, f"{location}[{index}]"))
    if len(result) != len(set(result)):
        raise ValueError(f"{location}: duplicate identifiers are not allowed")
    return result


def _validate_action(value: object, head: str, location: str) -> None:
    action = _object(value, location)
    allowed = ("enabled", "selection_value") if head == "client_selector" else ("enabled",)
    if set(action) not in ({"enabled"}, set(allowed)):
        raise ValueError(f"{location}: invalid fields for {head}")
    enabled = action.get("enabled")
    if enabled not in {"ON", "OFF"}:
        raise ValueError(f"{location}.enabled: expected ON or OFF")
    has_selection = "selection_value" in action
    if head != "client_selector" and has_selection:
        raise ValueError(f"{location}: selection_value is only valid for client_selector")
    if head == "client_selector" and enabled == "ON":
        if not has_selection:
            raise ValueError(f"{location}: Client Selector ON requires selection_value")
        _integer(action["selection_value"], f"{location}.selection_value")
    elif has_selection:
        raise ValueError(f"{location}: selection_value is invalid when Client Selector is OFF")


def _validate_joint_action(value: object, location: str) -> None:
    action = _object(value, location)
    _exact_keys(action, HEADS, location)
    for head in HEADS:
        _validate_action(action[head], head, f"{location}.{head}")


def _validate_condition(value: object, location: str) -> None:
    condition = _object(value, location)
    _exact_keys(condition, ("feature", "relation", "value"), location)
    feature = condition["feature"]
    relation = condition["relation"]
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"{location}.feature: unknown Feature Specification v1 field")
    if relation not in RELATIONS:
        raise ValueError(f"{location}.relation: unsupported relation")
    comparison = condition["value"]
    if feature in STATE_CATEGORICAL_FEATURES:
        if relation not in {"==", "!="} or not isinstance(comparison, str) or not comparison:
            raise ValueError(f"{location}: categorical conditions require == or != and text")
        if feature != "workload" and comparison not in {"ON", "OFF"}:
            raise ValueError(f"{location}.value: expected ON or OFF")
    else:
        _number(comparison, f"{location}.value")


def _validate_condition_group(value: object, location: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: expected at least one condition")
    for index, condition in enumerate(value):
        _validate_condition(condition, f"{location}[{index}]")


def validate_rule_logic(
    value: object,
    head: str,
    location: str = "rule",
) -> None:
    """Validate the runtime-relevant part of one normalized rule.

    Qualification artifacts and independently validated artifacts carry
    different evidence metadata, but they share exactly the same action and
    condition representation. Keeping this check common prevents their runtime
    semantics from drifting apart.
    """
    if head not in HEADS:
        raise ValueError(f"{location}: unsupported decision head")
    rule = _object(value, location)
    required = {"rule_id", "action", "conditions", "exceptions"}
    missing = required - set(rule)
    if missing:
        raise ValueError(f"{location}: missing rule-logic fields {sorted(missing)}")
    _identifier(rule["rule_id"], f"{location}.rule_id")
    _validate_action(rule["action"], head, f"{location}.action")
    _validate_condition_group(rule["conditions"], f"{location}.conditions")
    exceptions = rule["exceptions"]
    if not isinstance(exceptions, list):
        raise ValueError(f"{location}.exceptions: expected a list of condition groups")
    for index, group in enumerate(exceptions):
        _validate_condition_group(group, f"{location}.exceptions[{index}]")


def _validate_rule(
    value: object,
    head: str,
    minimum_support: int,
    threshold: float,
    location: str,
) -> str:
    rule = _object(value, location)
    _exact_keys(
        rule,
        ("rule_id", "action", "conditions", "exceptions", "validation", "provenance"),
        location,
    )
    validate_rule_logic(rule, head, location)
    rule_id = str(rule["rule_id"])

    validation = _object(rule["validation"], f"{location}.validation")
    _exact_keys(
        validation,
        ("coverage", "support", "precision", "wilson_lower_bound"),
        f"{location}.validation",
    )
    coverage = _integer(validation["coverage"], f"{location}.validation.coverage", 1)
    support = _integer(validation["support"], f"{location}.validation.support", 0)
    if support > coverage:
        raise ValueError(f"{location}.validation: support cannot exceed coverage")
    if support < minimum_support:
        raise ValueError(f"{location}: rule is below its head minimum support")
    precision = _number(validation["precision"], f"{location}.validation.precision", 0.0)
    wilson = _number(
        validation["wilson_lower_bound"],
        f"{location}.validation.wilson_lower_bound",
        0.0,
    )
    if precision > 1 or wilson > precision:
        raise ValueError(f"{location}.validation: invalid precision or Wilson bound")
    if not math.isclose(precision, support / coverage, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{location}.validation: precision must equal support/coverage")
    if wilson < threshold:
        raise ValueError(f"{location}: rule is below its head activation threshold")

    provenance = _object(rule["provenance"], f"{location}.provenance")
    _exact_keys(
        provenance,
        ("source_rule", "validation_split_sha256"),
        f"{location}.provenance",
    )
    _identifier(provenance["source_rule"], f"{location}.provenance.source_rule")
    _sha256(
        provenance["validation_split_sha256"],
        f"{location}.provenance.validation_split_sha256",
    )
    return rule_id


def validate_rule_artifact(value: object) -> None:
    """Check that a mined rule set is safe and complete enough to be loaded.

    Besides checking its shape, this verifies provenance, per-head evidence
    thresholds, allowed feature names, and actions that the runtime can apply.
    """
    artifact = _object(value, "rule_artifact")
    _exact_keys(
        artifact,
        ("schema_version", "rule_set_id", "feature_schema", "label_schema", "created_from", "teacher", "heads"),
        "rule_artifact",
    )
    if artifact["schema_version"] != RULE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("rule_artifact.schema_version: unsupported version")
    _identifier(artifact["rule_set_id"], "rule_artifact.rule_set_id")
    if artifact["feature_schema"] != FEATURE_SCHEMA_ID:
        raise ValueError("rule_artifact.feature_schema: expected Feature Specification v1")
    if artifact["label_schema"] != LABEL_SCHEMA_ID:
        raise ValueError("rule_artifact.label_schema: expected normalized labels v5")

    created = _object(artifact["created_from"], "rule_artifact.created_from")
    _exact_keys(
        created,
        ("dataset_sha256", "folds_sha256", "miner", "miner_version", "producing_code_sha"),
        "rule_artifact.created_from",
    )
    _sha256(created["dataset_sha256"], "rule_artifact.created_from.dataset_sha256")
    _sha256(created["folds_sha256"], "rule_artifact.created_from.folds_sha256")
    _identifier(created["miner"], "rule_artifact.created_from.miner")
    if not isinstance(created["miner_version"], str) or not created["miner_version"]:
        raise ValueError("rule_artifact.created_from.miner_version: expected text")
    if not isinstance(created["producing_code_sha"], str) or not CODE_SHA_RE.fullmatch(created["producing_code_sha"]):
        raise ValueError("rule_artifact.created_from.producing_code_sha: expected a Git SHA")

    teacher = _object(artifact["teacher"], "rule_artifact.teacher")
    _exact_keys(teacher, ("policy", "model", "model_digest"), "rule_artifact.teacher")
    for field in ("policy", "model"):
        if not isinstance(teacher[field], str) or not teacher[field]:
            raise ValueError(f"rule_artifact.teacher.{field}: expected text")
    _sha256(teacher["model_digest"], "rule_artifact.teacher.model_digest")

    heads = _object(artifact["heads"], "rule_artifact.heads")
    _exact_keys(heads, HEADS, "rule_artifact.heads")
    seen: set[str] = set()
    for head in HEADS:
        head_value = _object(heads[head], f"rule_artifact.heads.{head}")
        _exact_keys(
            head_value,
            ("minimum_support", "minimum_wilson_lower_bound", "rules"),
            f"rule_artifact.heads.{head}",
        )
        minimum_support = _integer(
            head_value["minimum_support"],
            f"rule_artifact.heads.{head}.minimum_support",
            1,
        )
        threshold = _number(
            head_value["minimum_wilson_lower_bound"],
            f"rule_artifact.heads.{head}.minimum_wilson_lower_bound",
            0.0,
        )
        if threshold > 1:
            raise ValueError(f"rule_artifact.heads.{head}: threshold cannot exceed one")
        rules = head_value["rules"]
        if not isinstance(rules, list):
            raise ValueError(f"rule_artifact.heads.{head}.rules: expected a list")
        for index, rule in enumerate(rules):
            rule_id = _validate_rule(
                rule,
                head,
                minimum_support,
                threshold,
                f"rule_artifact.heads.{head}.rules[{index}]",
            )
            if rule["provenance"]["validation_split_sha256"] != created["folds_sha256"]:
                raise ValueError(
                    f"rule_artifact.heads.{head}.rules[{index}]: "
                    "validation split differs from the artifact folds"
                )
            if rule_id in seen:
                raise ValueError(f"rule_artifact: duplicate rule_id {rule_id!r}")
            seen.add(rule_id)


def load_rule_artifact(path: Path) -> LoadedRuleArtifact:
    """Read a rule-set file, validate it, and retain its exact file hash."""
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except Exception as error:
        raise ValueError(f"{path}: invalid JSON") from error
    validate_rule_artifact(value)
    return LoadedRuleArtifact(document=value, sha256=hashlib.sha256(raw).hexdigest())


def _validate_state(value: object, location: str) -> None:
    state = _object(value, location)
    _exact_keys(state, FEATURE_COLUMNS, location)
    for feature in FEATURE_COLUMNS:
        item = state[feature]
        if not isinstance(item, str):
            raise ValueError(f"{location}.{feature}: normalized state values must be text")
        if item == "":
            continue
        if feature == "workload":
            continue
        if feature in {"has_delay_clients", "prev_cs", "prev_mc", "prev_hdh"}:
            if item not in {"ON", "OFF"}:
                raise ValueError(f"{location}.{feature}: expected ON, OFF, or blank")
            continue
        try:
            numeric = float(item)
        except ValueError as error:
            raise ValueError(f"{location}.{feature}: expected numeric text or blank") from error
        if not math.isfinite(numeric):
            raise ValueError(f"{location}.{feature}: expected finite numeric text")
        if feature in STATE_INTEGER_FEATURES and not numeric.is_integer():
            raise ValueError(f"{location}.{feature}: expected integer text")


def validate_dispatch_request(value: object) -> None:
    """Check a live state request and the provenance required by its mode."""
    request = _object(value, "dispatch_request")
    _exact_keys(
        request,
        ("schema_version", "mode", "record_id", "run_id", "rule_set_id", "rule_set_sha256", "state"),
        "dispatch_request",
    )
    if request["schema_version"] != DISPATCH_SCHEMA_VERSION:
        raise ValueError("dispatch_request.schema_version: unsupported version")
    if request["mode"] not in MODES:
        raise ValueError("dispatch_request.mode: unsupported mode")
    _identifier(request["record_id"], "dispatch_request.record_id")
    _identifier(request["run_id"], "dispatch_request.run_id")
    if request["mode"] == "always_defer":
        if request["rule_set_id"] is not None or request["rule_set_sha256"] is not None:
            raise ValueError(
                "dispatch_request: always-defer mode must not claim a rule set"
            )
    else:
        _identifier(request["rule_set_id"], "dispatch_request.rule_set_id")
        _sha256(request["rule_set_sha256"], "dispatch_request.rule_set_sha256")
    _validate_state(request["state"], "dispatch_request.state")
    match = re.fullmatch(r"(.+)::d(\d+)", str(request["record_id"]))
    if (
        match is None
        or match.group(1) != request["run_id"]
        or request["state"]["round_idx"] != match.group(2)
    ):
        raise ValueError(
            "dispatch_request: record_id, run_id, and state round_idx are inconsistent"
        )


def _validate_head_evaluation(value: object, head: str, location: str) -> None:
    evaluation = _object(value, location)
    _exact_keys(
        evaluation,
        ("outcome", "fired_rule_ids", "candidate_actions", "selected_action", "trigger"),
        location,
    )
    outcome = evaluation["outcome"]
    if outcome not in {"decide", "defer", "not_evaluated"}:
        raise ValueError(f"{location}.outcome: unsupported value")
    fired = _string_list(evaluation["fired_rule_ids"], f"{location}.fired_rule_ids", allow_empty=True)
    candidates = evaluation["candidate_actions"]
    if not isinstance(candidates, list):
        raise ValueError(f"{location}.candidate_actions: expected a list")
    for index, action in enumerate(candidates):
        _validate_action(action, head, f"{location}.candidate_actions[{index}]")
    if len(candidates) != len({canonical_sha256(item) for item in candidates}):
        raise ValueError(f"{location}.candidate_actions: duplicate actions are not allowed")
    selected = evaluation["selected_action"]
    if selected is not None:
        _validate_action(selected, head, f"{location}.selected_action")
    trigger = evaluation["trigger"]
    if trigger not in {None, "always_defer", "no_rule", "conflict", "insufficient_evidence"}:
        raise ValueError(f"{location}.trigger: unsupported value")

    if outcome == "decide" and (not fired or not candidates or selected is None or trigger is not None):
        raise ValueError(f"{location}: decide requires matches, candidates, and a selected action")
    if outcome == "decide" and selected not in candidates:
        raise ValueError(f"{location}: selected action is absent from candidate actions")
    if outcome == "not_evaluated" and (fired or candidates or selected is not None or trigger != "always_defer"):
        raise ValueError(f"{location}: not_evaluated is reserved for always-defer mode")
    if outcome == "defer" and (selected is not None or trigger not in {"no_rule", "conflict", "insufficient_evidence"}):
        raise ValueError(f"{location}: deferred heads need a rule deferral trigger")
    if outcome == "defer" and trigger == "no_rule" and (fired or candidates):
        raise ValueError(f"{location}: no_rule cannot have fired rules or candidates")
    if outcome == "defer" and trigger == "insufficient_evidence" and not fired:
        raise ValueError(f"{location}: insufficient_evidence requires a fired rule")
    if outcome == "defer" and trigger == "conflict":
        unique_actions = {canonical_sha256(item) for item in candidates}
        if len(fired) < 2 or len(unique_actions) < 2:
            raise ValueError(f"{location}: conflict requires different actions from multiple rules")


def validate_dispatch_result(value: object) -> None:
    """Check that per-head outcomes form one consistent dispatch decision."""
    result = _object(value, "dispatch_result")
    _exact_keys(
        result,
        ("schema_version", "mode", "record_id", "rule_set_id", "head_evaluations", "requires_teacher", "deferral_trigger", "proposed_action", "dispatch_time_us"),
        "dispatch_result",
    )
    if result["schema_version"] != DISPATCH_SCHEMA_VERSION:
        raise ValueError("dispatch_result.schema_version: unsupported version")
    mode = result["mode"]
    if mode not in MODES:
        raise ValueError("dispatch_result.mode: unsupported mode")
    _identifier(result["record_id"], "dispatch_result.record_id")
    if mode == "always_defer":
        if result["rule_set_id"] is not None:
            raise ValueError("dispatch_result: always-defer mode must not claim a rule set")
    else:
        _identifier(result["rule_set_id"], "dispatch_result.rule_set_id")
    evaluations = _object(result["head_evaluations"], "dispatch_result.head_evaluations")
    _exact_keys(evaluations, HEADS, "dispatch_result.head_evaluations")
    for head in HEADS:
        _validate_head_evaluation(evaluations[head], head, f"dispatch_result.head_evaluations.{head}")
    if not isinstance(result["requires_teacher"], bool):
        raise ValueError("dispatch_result.requires_teacher: expected a boolean")
    if result["deferral_trigger"] not in {None, "always_defer", "shadow_mode", "rule_deferral"}:
        raise ValueError("dispatch_result.deferral_trigger: unsupported value")
    if result["proposed_action"] is not None:
        _validate_joint_action(result["proposed_action"], "dispatch_result.proposed_action")
    _integer(result["dispatch_time_us"], "dispatch_result.dispatch_time_us")

    all_decide = all(evaluations[head]["outcome"] == "decide" for head in HEADS)
    if (result["proposed_action"] is not None) != all_decide:
        raise ValueError("dispatch_result: proposed_action must exist exactly when all heads decide")
    if all_decide:
        expected_action = {
            head: evaluations[head]["selected_action"] for head in HEADS
        }
        if result["proposed_action"] != expected_action:
            raise ValueError("dispatch_result: proposed action differs from head selections")
    if mode == "always_defer" and any(
        evaluations[head]["outcome"] != "not_evaluated" for head in HEADS
    ):
        raise ValueError("dispatch_result: always-defer mode must not evaluate rules")
    if mode != "always_defer" and any(
        evaluations[head]["outcome"] == "not_evaluated" for head in HEADS
    ):
        raise ValueError("dispatch_result: only always-defer mode may skip rule evaluation")
    expected = {
        "always_defer": (True, "always_defer"),
        "shadow": (True, "shadow_mode"),
        "active": (not all_decide, "rule_deferral" if not all_decide else None),
    }[mode]
    if (result["requires_teacher"], result["deferral_trigger"]) != expected:
        raise ValueError("dispatch_result: mode and deferral fields are inconsistent")


def _validate_teacher_resolution(value: object, location: str) -> str:
    resolution = _object(value, location)
    status = resolution.get("status")
    if status == "not_queried":
        _exact_keys(resolution, ("status",), location)
    elif status == "success":
        _exact_keys(
            resolution,
            ("status", "policy", "model", "model_digest", "prompt_sha256", "response_sha256", "latency_ms", "action"),
            location,
        )
        for field in ("policy", "model"):
            if not isinstance(resolution[field], str) or not resolution[field]:
                raise ValueError(f"{location}.{field}: expected text")
        for field in ("model_digest", "prompt_sha256", "response_sha256"):
            _sha256(resolution[field], f"{location}.{field}")
        _number(resolution["latency_ms"], f"{location}.latency_ms", 0.0)
        _validate_joint_action(resolution["action"], f"{location}.action")
    elif status == "error":
        _exact_keys(resolution, ("status", "error_type", "error_message", "latency_ms"), location)
        for field in ("error_type", "error_message"):
            if not isinstance(resolution[field], str) or not resolution[field]:
                raise ValueError(f"{location}.{field}: expected text")
        _number(resolution["latency_ms"], f"{location}.latency_ms", 0.0)
    else:
        raise ValueError(f"{location}.status: unsupported value")
    return str(status)


def validate_decision_trace(value: object) -> None:
    """Check a retained trace from state capture through applied action.

    The cross-field checks ensure the request and dispatch describe the same
    decision and that teacher use, fallback behavior, and the applied source
    agree with the selected runtime mode.
    """
    trace = _object(value, "decision_trace")
    _exact_keys(
        trace,
        ("schema_version", "trace_id", "recorded_at_utc", "request", "request_sha256", "dispatch", "teacher_resolution", "application", "controller_time_us"),
        "decision_trace",
    )
    if trace["schema_version"] != DECISION_TRACE_SCHEMA_VERSION:
        raise ValueError("decision_trace.schema_version: unsupported version")
    _identifier(trace["trace_id"], "decision_trace.trace_id")
    try:
        recorded_at = datetime.fromisoformat(str(trace["recorded_at_utc"]))
    except ValueError as error:
        raise ValueError(
            "decision_trace.recorded_at_utc: expected an explicit UTC timestamp"
        ) from error
    if recorded_at.utcoffset() != timezone.utc.utcoffset(recorded_at):
        raise ValueError("decision_trace.recorded_at_utc: expected an explicit UTC timestamp")
    validate_dispatch_request(trace["request"])
    _sha256(trace["request_sha256"], "decision_trace.request_sha256")
    if trace["request_sha256"] != canonical_sha256(trace["request"]):
        raise ValueError("decision_trace.request_sha256: differs from the embedded request")
    validate_dispatch_result(trace["dispatch"])
    request = trace["request"]
    dispatch = trace["dispatch"]
    for field in ("mode", "record_id", "rule_set_id"):
        if request[field] != dispatch[field]:
            raise ValueError(f"decision_trace: request and dispatch {field} differ")
    teacher_status = _validate_teacher_resolution(
        trace["teacher_resolution"], "decision_trace.teacher_resolution",
    )
    application = _object(trace["application"], "decision_trace.application")
    _exact_keys(
        application,
        ("source", "action_before_guardrail", "applied_action", "guardrail_result"),
        "decision_trace.application",
    )
    if application["source"] not in {"rules", "teacher", "safe_fallback"}:
        raise ValueError("decision_trace.application.source: unsupported value")
    _validate_joint_action(
        application["action_before_guardrail"],
        "decision_trace.application.action_before_guardrail",
    )
    _validate_joint_action(application["applied_action"], "decision_trace.application.applied_action")
    if application["guardrail_result"] not in {
        "unchanged", "selection_value_adjusted", "client_selector_disabled", "safe_fallback",
    }:
        raise ValueError("decision_trace.application.guardrail_result: unsupported value")
    controller_time = _integer(
        trace["controller_time_us"], "decision_trace.controller_time_us",
    )
    if controller_time < dispatch["dispatch_time_us"]:
        raise ValueError("decision_trace.controller_time_us: cannot be below dispatch time")

    requires_teacher = dispatch["requires_teacher"]
    expected_source = {
        "not_queried": "rules",
        "success": "teacher",
        "error": "safe_fallback",
    }[teacher_status]
    if application["source"] != expected_source:
        raise ValueError("decision_trace: teacher status and applied-action source differ")
    if requires_teacher != (teacher_status != "not_queried"):
        raise ValueError("decision_trace: teacher resolution does not match dispatch requirement")
    if teacher_status == "not_queried" and application["action_before_guardrail"] != dispatch["proposed_action"]:
        raise ValueError("decision_trace: rule action differs from the dispatch proposal")
    if teacher_status == "success" and application["action_before_guardrail"] != trace["teacher_resolution"]["action"]:
        raise ValueError("decision_trace: teacher action differs from the applied source action")
