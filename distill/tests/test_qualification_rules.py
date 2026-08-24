"""Check the provisional Sprint 01 translation and qualification dispatcher."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


DISTILL_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    DISTILL_ROOT
    / "data"
    / "paper_archive_fashionmnist_cnn16k_fewshot_deepseek"
)
sys.path.insert(0, str(DISTILL_ROOT))

from live_policy import (  # noqa: E402
    apply_rule_action,
    load_live_policy_settings,
    write_policy_trace,
)
from policy_interfaces import (  # noqa: E402
    DISPATCH_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    HEADS,
    validate_decision_trace,
    validate_rule_artifact,
)
from qualification_rules import (  # noqa: E402
    DATASET_SHA256,
    EXPECTED_CONFIGURATION_SCOPE,
    QUALIFICATION_ARTIFACT_KIND,
    QUALIFICATION_RULE_SET_ID,
    LoadedQualificationArtifact,
    build_qualification_artifact,
    rule_matches,
    validate_qualification_artifact,
    validate_qualification_configuration,
)
from prepare_sprint03_run import prepare_run_config  # noqa: E402
from rule_dispatch import dispatch_rules  # noqa: E402


def qualification_config(mode: str = "active") -> dict[str, object]:
    scope = copy.deepcopy(EXPECTED_CONFIGURATION_SCOPE)
    return {
        "adaptation": "Distilled Policy",
        "LLM": "deepseek-r1:8b",
        "rounds": scope["rounds"],
        "clients": scope["clients"],
        "clients_per_round": scope["clients"],
        "dataset": "AG_NEWS",
        "partition_seed": scope["partition_seed"],
        "client_details": scope["client_layout"],
        "patterns": {
            "client_selector": {
                "enabled": False,
                "params": {"selection_value": 3},
            },
            "message_compressor": {"enabled": False, "params": {}},
            "heterogeneous_data_handler": {"enabled": False, "params": {}},
        },
        "distill_policy": {
            "mode": mode,
            "run_id": "sprint03/agnews/qualification/r1",
            "teacher_policy": "Single AI-Agent (Few-Shot)",
            "teacher_model_digest": "a" * 64,
            "trace_dir": "performance/distill_decisions",
            "artifact_kind": QUALIFICATION_ARTIFACT_KIND,
            "rule_artifact": "artifacts/sprint01-qualification.json",
            "rule_artifact_sha256": "b" * 64,
        },
    }


def source_rule_matches(rule: dict[str, object], state: dict[str, str]) -> bool:
    condition_only = {
        "conditions": rule["conditions"],
        "exceptions": [],
    }
    return rule_matches(condition_only, state) and not any(
        source_rule_matches(exception, state)
        for exception in rule["exceptions"]
    )


class QualificationRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = build_qualification_artifact(SOURCE_ROOT, "c" * 40)
        with (SOURCE_ROOT / "decision_dataset.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            cls.rows = list(csv.DictReader(stream))
        cls.warm_rows = [row for row in cls.rows if row["val_f1"] != ""]

    def request(self, row: dict[str, str], mode: str = "active") -> dict[str, object]:
        return {
            "schema_version": DISPATCH_SCHEMA_VERSION,
            "mode": mode,
            "record_id": f"qualification-test::d{row['round_idx']}",
            "run_id": "qualification-test",
            "rule_set_id": QUALIFICATION_RULE_SET_ID,
            "rule_set_sha256": "b" * 64,
            "state": {feature: row[feature] for feature in FEATURE_COLUMNS},
        }

    def test_translation_is_pinned_and_explicitly_qualification_only(self) -> None:
        validate_qualification_artifact(self.artifact)
        with self.assertRaises(ValueError):
            validate_rule_artifact(self.artifact)
        self.assertEqual(QUALIFICATION_ARTIFACT_KIND, self.artifact["artifact_kind"])
        self.assertEqual(DATASET_SHA256, self.artifact["created_from"]["dataset_sha256"])
        self.assertEqual(
            {"client_selector": 6, "message_compressor": 6,
             "heterogeneous_data_handler": 5},
            {head: len(self.artifact["heads"][head]["rules"]) for head in HEADS},
        )
        for head in HEADS:
            for rule in self.artifact["heads"][head]["rules"]:
                self.assertIn("training_evidence", rule)
                self.assertNotIn("validation", rule)

    def test_disjoint_translation_matches_ordered_source_on_all_warm_rows(self) -> None:
        rules_root = SOURCE_ROOT / "confold_baseline" / "rules"
        names = {
            "client_selector": "cs_exploratory_rules.json",
            "message_compressor": "mc_exploratory_rules.json",
            "heterogeneous_data_handler": "hdh_exploratory_rules.json",
        }
        for head in HEADS:
            source = json.loads((rules_root / names[head]).read_text(encoding="utf-8"))
            compiled = self.artifact["heads"][head]["rules"]
            for row in self.warm_rows:
                expected = next(
                    rule["predict"] for rule in source["rules"]
                    if source_rule_matches(rule, row)
                )
                fired_actions = {
                    rule["action"]["enabled"]
                    for rule in compiled if rule_matches(rule, row)
                }
                self.assertEqual({expected}, fired_actions, (head, row["record_id"]))

    def test_configuration_scope_is_exact(self) -> None:
        value = qualification_config()
        validate_qualification_configuration(value)
        frozen_base = json.loads(
            (DISTILL_ROOT / "configs" / "sprint03_agnews_base.json").read_text(
                encoding="utf-8"
            )
        )
        validate_qualification_configuration(frozen_base)
        changed = copy.deepcopy(value)
        changed["client_details"][3]["cpu"] = 1
        with self.assertRaisesRegex(ValueError, "client layout"):
            validate_qualification_configuration(changed)

    def test_settings_verify_artifact_kind_hash_and_configuration(self) -> None:
        loaded = LoadedQualificationArtifact(self.artifact, "b" * 64)
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "live_policy.load_qualification_artifact", return_value=loaded,
        ):
            settings = load_live_policy_settings(
                qualification_config(), Path(temporary),
            )
        self.assertEqual("active", settings.mode)
        self.assertEqual(QUALIFICATION_RULE_SET_ID, settings.rule_artifact["rule_set_id"])

    def test_run_preparation_keeps_control_and_active_inputs_paired(self) -> None:
        base = qualification_config()
        base.pop("distill_policy")
        control = prepare_run_config(
            base, "always_defer", "sprint03/agnews/control/r1", "a" * 64,
        )
        self.assertNotIn("rule_artifact", control["distill_policy"])
        with tempfile.TemporaryDirectory(dir=DISTILL_ROOT) as temporary:
            artifact_path = Path(temporary) / "qualification.json"
            artifact_path.write_text(
                json.dumps(self.artifact, sort_keys=True) + "\n", encoding="utf-8",
            )
            artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            active = prepare_run_config(
                base, "active", "sprint03/agnews/active/r1", "a" * 64,
                artifact_path,
            )
        self.assertEqual(artifact_hash, active["distill_policy"]["rule_artifact_sha256"])
        control_without_policy = copy.deepcopy(control)
        active_without_policy = copy.deepcopy(active)
        control_without_policy.pop("distill_policy")
        active_without_policy.pop("distill_policy")
        self.assertEqual(control_without_policy, active_without_policy)

    def test_cold_start_defers_and_warm_states_are_unambiguous(self) -> None:
        cold = dispatch_rules(self.request(self.rows[0]), self.artifact)
        self.assertTrue(cold["requires_teacher"])
        self.assertTrue(all(
            cold["head_evaluations"][head]["trigger"] == "no_rule"
            for head in HEADS
        ))
        for row in self.warm_rows:
            result = dispatch_rules(self.request(row), self.artifact)
            self.assertFalse(result["requires_teacher"], row["record_id"])
            self.assertIsNotNone(result["proposed_action"])

    def test_shadow_evaluates_rules_but_still_uses_teacher(self) -> None:
        result = dispatch_rules(self.request(self.warm_rows[0], "shadow"), self.artifact)
        self.assertTrue(result["requires_teacher"])
        self.assertEqual("shadow_mode", result["deferral_trigger"])
        self.assertIsNotNone(result["proposed_action"])

    def test_active_rule_trace_records_no_teacher_query(self) -> None:
        request = self.request(self.warm_rows[0])
        dispatch = dispatch_rules(request, self.artifact)
        base = qualification_config()
        base["distill_policy"]["run_id"] = "qualification-test"
        next_config, audit = apply_rule_action(base, dispatch["proposed_action"],
                                                lambda value, runtime: value)
        loaded = LoadedQualificationArtifact(self.artifact, "b" * 64)
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "live_policy.load_qualification_artifact", return_value=loaded,
        ):
            settings = load_live_policy_settings(base, Path(temporary))
            trace_path = write_policy_trace(
                settings, request, dispatch, None, audit, base, 100,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        validate_decision_trace(trace)
        self.assertEqual("not_queried", trace["teacher_resolution"]["status"])
        self.assertEqual("rules", trace["application"]["source"])
        self.assertEqual(audit["applied_action"], trace["application"]["applied_action"])
        self.assertIsInstance(next_config, dict)


if __name__ == "__main__":
    unittest.main()
