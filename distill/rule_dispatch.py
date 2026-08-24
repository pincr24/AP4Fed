"""Deterministic decide-or-defer evaluation for normalized rule artifacts."""

from __future__ import annotations

import time
from typing import Mapping

from policy_interfaces import (
    DISPATCH_SCHEMA_VERSION,
    HEADS,
    canonical_sha256,
    validate_dispatch_request,
    validate_dispatch_result,
)
from qualification_rules import rule_matches


def dispatch_rules(
    request: dict[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate all rules and return a validated whole-decision result.

    Multiple firing rules that support the same action are unambiguous. Rules
    supporting different actions produce a conflict and therefore a teacher
    deferral. The first decision always defers because the Sprint 01 learner was
    fitted only on warm-start states.
    """
    validate_dispatch_request(request)
    mode = request["mode"]
    if mode not in {"shadow", "active"}:
        raise ValueError("rule dispatch requires shadow or active mode")
    if artifact.get("rule_set_id") != request["rule_set_id"]:
        raise ValueError("dispatch request and artifact rule-set IDs differ")
    state = request["state"]
    if not isinstance(state, dict):
        raise ValueError("dispatch request state must be an object")
    heads = artifact.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("rule artifact has no heads object")

    started = time.perf_counter_ns()
    evaluations: dict[str, dict[str, object]] = {}
    cold_start = state.get("round_idx") == "1"
    for head in HEADS:
        head_value = heads.get(head)
        if not isinstance(head_value, dict) or not isinstance(head_value.get("rules"), list):
            raise ValueError(f"rule artifact has no rules for {head}")
        fired = [] if cold_start else [
            rule for rule in head_value["rules"] if rule_matches(rule, state)
        ]
        actions: list[dict[str, object]] = []
        action_hashes: set[str] = set()
        for rule in fired:
            action = rule["action"]
            digest = canonical_sha256(action)
            if digest not in action_hashes:
                actions.append(action)
                action_hashes.add(digest)
        if not fired:
            evaluation = {
                "outcome": "defer",
                "fired_rule_ids": [],
                "candidate_actions": [],
                "selected_action": None,
                "trigger": "no_rule",
            }
        elif len(actions) > 1:
            evaluation = {
                "outcome": "defer",
                "fired_rule_ids": [rule["rule_id"] for rule in fired],
                "candidate_actions": actions,
                "selected_action": None,
                "trigger": "conflict",
            }
        else:
            evaluation = {
                "outcome": "decide",
                "fired_rule_ids": [rule["rule_id"] for rule in fired],
                "candidate_actions": actions,
                "selected_action": actions[0],
                "trigger": None,
            }
        evaluations[head] = evaluation

    all_decide = all(
        evaluations[head]["outcome"] == "decide" for head in HEADS
    )
    proposal = (
        {head: evaluations[head]["selected_action"] for head in HEADS}
        if all_decide else None
    )
    result = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "mode": mode,
        "record_id": request["record_id"],
        "rule_set_id": request["rule_set_id"],
        "head_evaluations": evaluations,
        "requires_teacher": mode == "shadow" or not all_decide,
        "deferral_trigger": (
            "shadow_mode" if mode == "shadow"
            else "rule_deferral" if not all_decide
            else None
        ),
        "proposed_action": proposal,
        "dispatch_time_us": max(0, (time.perf_counter_ns() - started) // 1000),
    }
    validate_dispatch_result(result)
    return result

