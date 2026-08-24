"""Check strict validation of rule sets, dispatch messages, and traces."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


DISTILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DISTILL_ROOT))

from decision_state import FEATURE_COLUMNS  # noqa: E402
from policy_interfaces import (  # noqa: E402
    canonical_sha256,
    load_rule_artifact,
    validate_decision_trace,
    validate_dispatch_request,
    validate_dispatch_result,
    validate_rule_artifact,
)


DIGEST = "a" * 64
FOLDS_DIGEST = "b" * 64
MODEL_DIGEST = "c" * 64
CODE_SHA = "d" * 40


def action(head: str, enabled: str = "OFF") -> dict[str, object]:
    value: dict[str, object] = {"enabled": enabled}
    if head == "client_selector" and enabled == "ON":
        value["selection_value"] = 3
    return value


def joint_action() -> dict[str, object]:
    return {
        "client_selector": action("client_selector", "ON"),
        "message_compressor": action("message_compressor", "OFF"),
        "heterogeneous_data_handler": action("heterogeneous_data_handler", "ON"),
    }


def condition(head: str) -> dict[str, object]:
    if head == "client_selector":
        return {"feature": "prev_cs", "relation": "==", "value": "OFF"}
    if head == "message_compressor":
        return {"feature": "comm_frac", "relation": ">", "value": 0.2}
    return {"feature": "frac_non_iid", "relation": ">", "value": 0.5}


def rule(head: str) -> dict[str, object]:
    enabled = "OFF" if head == "message_compressor" else "ON"
    return {
        "rule_id": f"{head}-rule-1",
        "action": action(head, enabled),
        "conditions": [condition(head)],
        "exceptions": [],
        "validation": {
            "coverage": 10,
            "support": 9,
            "precision": 0.9,
            "wilson_lower_bound": 0.8,
        },
        "provenance": {
            "source_rule": f"synthetic/{head}/rule-1",
            "validation_split_sha256": FOLDS_DIGEST,
        },
    }


def artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "rule_set_id": "synthetic-rules-v1",
        "feature_schema": "ap4fed-feature-spec-v1",
        "label_schema": "ap4fed-normalized-labels-v5",
        "created_from": {
            "dataset_sha256": DIGEST,
            "folds_sha256": FOLDS_DIGEST,
            "miner": "synthetic-miner",
            "miner_version": "test-v1",
            "producing_code_sha": CODE_SHA,
        },
        "teacher": {
            "policy": "Single AI-Agent (Few-Shot)",
            "model": "deepseek-r1:8b",
            "model_digest": MODEL_DIGEST,
        },
        "heads": {
            head: {
                "minimum_support": 5,
                "minimum_wilson_lower_bound": 0.75,
                "rules": [rule(head)],
            }
            for head in (
                "client_selector",
                "message_compressor",
                "heterogeneous_data_handler",
            )
        },
    }


def state() -> dict[str, str]:
    value = {feature: "1" for feature in FEATURE_COLUMNS}
    value.update({
        "workload": "FashionMNIST__CNN_16k",
        "has_delay_clients": "ON",
        "prev_cs": "OFF",
        "prev_mc": "OFF",
        "prev_hdh": "OFF",
        "round_idx": "2",
    })
    return value


def request(mode: str = "active") -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": mode,
        "record_id": "synthetic/r1::d2",
        "run_id": "synthetic/r1",
        "rule_set_id": None if mode == "always_defer" else "synthetic-rules-v1",
        "rule_set_sha256": None if mode == "always_defer" else DIGEST,
        "state": state(),
    }


def evaluation(head: str) -> dict[str, object]:
    enabled = "OFF" if head == "message_compressor" else "ON"
    selected = action(head, enabled)
    return {
        "outcome": "decide",
        "fired_rule_ids": [f"{head}-rule-1"],
        "candidate_actions": [selected],
        "selected_action": selected,
        "trigger": None,
    }


def result(mode: str = "active") -> dict[str, object]:
    if mode == "always_defer":
        evaluations = {
            head: {
                "outcome": "not_evaluated",
                "fired_rule_ids": [],
                "candidate_actions": [],
                "selected_action": None,
                "trigger": "always_defer",
            }
            for head in (
                "client_selector",
                "message_compressor",
                "heterogeneous_data_handler",
            )
        }
    else:
        evaluations = {
            head: evaluation(head)
            for head in (
                "client_selector",
                "message_compressor",
                "heterogeneous_data_handler",
            )
        }
    return {
        "schema_version": 1,
        "mode": mode,
        "record_id": "synthetic/r1::d2",
        "rule_set_id": None if mode == "always_defer" else "synthetic-rules-v1",
        "head_evaluations": evaluations,
        "requires_teacher": mode != "active",
        "deferral_trigger": {
            "active": None,
            "shadow": "shadow_mode",
            "always_defer": "always_defer",
        }[mode],
        "proposed_action": None if mode == "always_defer" else joint_action(),
        "dispatch_time_us": 17,
    }


class PolicyInterfaceTest(unittest.TestCase):
    """Cover valid examples and the inconsistencies each interface must reject."""

    def test_valid_rule_artifact_loads_with_exact_file_hash(self) -> None:
        document = artifact()
        validate_rule_artifact(document)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rules.json"
            raw = json.dumps(document, indent=2) + "\n"
            path.write_text(raw, encoding="utf-8")
            loaded = load_rule_artifact(path)

        self.assertEqual(document, loaded.document)
        self.assertEqual(64, len(loaded.sha256))
        self.assertNotEqual(canonical_sha256(document), loaded.sha256)

    def test_rule_artifact_rejects_unvalidated_or_malformed_actions(self) -> None:
        below_threshold = artifact()
        below_threshold["heads"]["client_selector"]["rules"][0]["validation"][
            "wilson_lower_bound"
        ] = 0.7
        with self.assertRaisesRegex(ValueError, "below its head activation threshold"):
            validate_rule_artifact(below_threshold)

        below_support = artifact()
        below_support["heads"]["client_selector"]["minimum_support"] = 10
        with self.assertRaisesRegex(ValueError, "below its head minimum support"):
            validate_rule_artifact(below_support)

        missing_threshold = artifact()
        del missing_threshold["heads"]["client_selector"]["rules"][0]["action"][
            "selection_value"
        ]
        with self.assertRaisesRegex(ValueError, "requires selection_value"):
            validate_rule_artifact(missing_threshold)

    def test_rule_artifact_rejects_unknown_features(self) -> None:
        document = artifact()
        document["heads"]["message_compressor"]["rules"][0]["conditions"][0][
            "feature"
        ] = "source_policy"
        with self.assertRaisesRegex(ValueError, "unknown Feature Specification v1"):
            validate_rule_artifact(document)

    def test_dispatch_request_requires_the_exact_normalized_state_schema(self) -> None:
        document = request()
        validate_dispatch_request(document)
        del document["state"]["comm_frac"]
        with self.assertRaisesRegex(ValueError, "fields differ"):
            validate_dispatch_request(document)

        wrong_identity = request()
        wrong_identity["state"]["round_idx"] = "3"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_dispatch_request(wrong_identity)

    def test_modes_keep_rule_evaluation_and_deferral_separate(self) -> None:
        active = result("active")
        shadow = result("shadow")
        always = result("always_defer")
        validate_dispatch_result(active)
        validate_dispatch_result(shadow)
        validate_dispatch_result(always)
        self.assertFalse(active["requires_teacher"])
        self.assertTrue(shadow["requires_teacher"])
        self.assertEqual(active["proposed_action"], shadow["proposed_action"])
        self.assertIsNone(always["proposed_action"])
        self.assertIsNone(always["rule_set_id"])

    def test_decision_trace_accepts_rule_and_teacher_resolution_paths(self) -> None:
        active_request = request("active")
        active_result = result("active")
        rule_trace = {
            "schema_version": 1,
            "trace_id": "synthetic-r1-d2-rules",
            "recorded_at_utc": "2026-08-24T12:00:00+00:00",
            "request": active_request,
            "request_sha256": canonical_sha256(active_request),
            "dispatch": active_result,
            "teacher_resolution": {"status": "not_queried"},
            "application": {
                "source": "rules",
                "action_before_guardrail": joint_action(),
                "applied_action": joint_action(),
                "guardrail_result": "unchanged",
            },
            "controller_time_us": 29,
        }
        validate_decision_trace(rule_trace)

        shadow_request = request("shadow")
        shadow_result = result("shadow")
        teacher_trace = copy.deepcopy(rule_trace)
        teacher_trace.update({
            "trace_id": "synthetic-r1-d2-shadow",
            "request": shadow_request,
            "request_sha256": canonical_sha256(shadow_request),
            "dispatch": shadow_result,
            "teacher_resolution": {
                "status": "success",
                "policy": "Single AI-Agent (Few-Shot)",
                "model": "deepseek-r1:8b",
                "model_digest": MODEL_DIGEST,
                "prompt_sha256": "e" * 64,
                "response_sha256": "f" * 64,
                "latency_ms": 1200.0,
                "action": joint_action(),
            },
        })
        teacher_trace["application"]["source"] = "teacher"
        validate_decision_trace(teacher_trace)

        fallback_trace = copy.deepcopy(teacher_trace)
        fallback_trace["trace_id"] = "synthetic-r1-d2-fallback"
        fallback_trace["teacher_resolution"] = {
            "status": "error",
            "error_type": "TimeoutError",
            "error_message": "teacher deadline exceeded",
            "latency_ms": 5000.0,
        }
        fallback_trace["application"] = {
            "source": "safe_fallback",
            "action_before_guardrail": joint_action(),
            "applied_action": joint_action(),
            "guardrail_result": "safe_fallback",
        }
        validate_decision_trace(fallback_trace)

    def test_trace_rejects_a_missing_teacher_resolution(self) -> None:
        shadow_request = request("shadow")
        trace = {
            "schema_version": 1,
            "trace_id": "synthetic-r1-d2-invalid",
            "recorded_at_utc": "2026-08-24T12:00:00+00:00",
            "request": shadow_request,
            "request_sha256": canonical_sha256(shadow_request),
            "dispatch": result("shadow"),
            "teacher_resolution": {"status": "not_queried"},
            "application": {
                "source": "rules",
                "action_before_guardrail": joint_action(),
                "applied_action": joint_action(),
                "guardrail_result": "unchanged",
            },
            "controller_time_us": 29,
        }
        with self.assertRaisesRegex(ValueError, "does not match dispatch requirement"):
            validate_decision_trace(trace)


if __name__ == "__main__":
    unittest.main()
