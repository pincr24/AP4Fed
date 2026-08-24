"""Check live state capture, teacher auditing, and immutable v0 traces."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


DISTILL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = DISTILL_ROOT.parent
sys.path.insert(0, str(DISTILL_ROOT))

from live_policy import (  # noqa: E402
    action_from_config,
    apply_rule_action,
    always_defer_dispatch,
    build_always_defer_request,
    load_live_policy_settings,
    write_always_defer_trace,
)
from policy_interfaces import validate_decision_trace  # noqa: E402
from teacher_adapter import load_teacher_modules  # noqa: E402


AP_COLUMN = (
    "AP List (client_selector,client_cluster,message_compressor,"
    "model_co-versioning_registry,multi-task_model_trainer,"
    "heterogeneous_data_handler)"
)


def config(run_id: str = "sprint03/smoke/r1") -> dict[str, object]:
    return {
        "adaptation": "Distilled Policy",
        "LLM": "deepseek-r1:8b",
        "rounds": 3,
        "clients": 3,
        "distill_policy": {
            "mode": "always_defer",
            "run_id": run_id,
            "teacher_policy": "Single AI-Agent (Few-Shot)",
            "teacher_model_digest": "a" * 64,
            "trace_dir": "performance/distill_decisions",
        },
        "client_details": [
            {
                "client_id": 1,
                "dataset": "FashionMNIST",
                "model": "CNN 16k",
                "cpu": 5,
                "data_distribution_type": "IID",
                "data_persistence_type": "Same Data",
                "delay_combobox": "No",
            },
            {
                "client_id": 2,
                "dataset": "FashionMNIST",
                "model": "CNN 16k",
                "cpu": 5,
                "data_distribution_type": "IID",
                "data_persistence_type": "New Data",
                "delay_combobox": "No",
            },
            {
                "client_id": 3,
                "dataset": "FashionMNIST",
                "model": "CNN 16k",
                "cpu": 3,
                "data_distribution_type": "non-IID",
                "data_persistence_type": "New Data",
                "delay_combobox": "Yes",
            },
        ],
        "patterns": {
            "client_selector": {"enabled": False, "params": {}},
            "message_compressor": {"enabled": False, "params": {}},
            "heterogeneous_data_handler": {"enabled": False, "params": {}},
        },
    }


def write_metrics(path: Path) -> None:
    fields = [
        "Client ID", "FL Round", "Training Time", "JSD",
        "Communication Time", "Total Time of FL Round", "Val F1", AP_COLUMN,
    ]
    vectors = {
        1: "{OFF,OFF,OFF,OFF,OFF,OFF}",
        2: "{OFF,OFF,ON,OFF,OFF,OFF}",
    }
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for round_idx in range(1, 3):
            for client_idx in range(1, 4):
                writer.writerow({
                    "Client ID": client_idx,
                    "FL Round": round_idx,
                    "Training Time": 1.0 + client_idx,
                    "JSD": 0.1 * round_idx,
                    "Communication Time": 0.5 * client_idx,
                    "Total Time of FL Round": 5.0 + round_idx,
                    "Val F1": 0.6 + 0.1 * round_idx,
                    AP_COLUMN: vectors[round_idx] if client_idx == 1 else "",
                })


def guarded_config() -> dict[str, object]:
    value = config()
    value["patterns"] = {
        "client_selector": {
            "enabled": True,
            "params": {"selection_value": 3},
        },
        "message_compressor": {"enabled": True, "params": {}},
        "heterogeneous_data_handler": {"enabled": False, "params": {}},
    }
    return value


class LivePolicyTest(unittest.TestCase):
    """Exercise the opt-in path without running a full federated experiment."""

    def test_prompt_helper_retains_exact_teacher_io_only_when_audit_is_requested(self) -> None:
        prompting = load_teacher_modules().prompting
        original_build = prompting._sa_build_prompt
        original_call = prompting._sa_call_ollama
        original_parse = prompting._sa_parse_output
        prompting._sa_build_prompt = lambda *args, **kwargs: "exact live prompt"
        prompting._sa_call_ollama = (
            lambda *args, **kwargs: '{"client_selector":"OFF"}'
        )
        prompting._sa_parse_output = lambda raw: (
            {
                "client_selector": "OFF",
                "message_compressor": "OFF",
                "heterogeneous_data_handler": "OFF",
            },
            "rationale",
            True,
        )
        try:
            audit = {}
            decisions, _ = prompting._sa_generate_with_retry(
                "deepseek-r1:8b", "few", {}, None, {}, {}, ["http://unused"],
                audit=audit,
            )
            without_audit, _ = prompting._sa_generate_with_retry(
                "deepseek-r1:8b", "few", {}, None, {}, {}, ["http://unused"],
            )
        finally:
            prompting._sa_build_prompt = original_build
            prompting._sa_call_ollama = original_call
            prompting._sa_parse_output = original_parse

        self.assertEqual(decisions, without_audit)
        self.assertEqual("exact live prompt", audit["prompt"])
        self.assertEqual('{"client_selector":"OFF"}', audit["raw_response"])
        self.assertEqual("success", audit["status"])

    def test_existing_single_agent_path_exposes_audit_without_changing_action(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "test_agent_policy_single",
            REPOSITORY / "Docker" / "agent_policy_single.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        adaptation = types.ModuleType("adaptation")
        adaptation.PATTERNS = [
            "client_selector", "message_compressor", "heterogeneous_data_handler",
        ]
        adaptation._append_agent_log = lambda *args, **kwargs: None
        adaptation._sa_aggregate_round = lambda *args, **kwargs: {}
        adaptation._sa_build_prompt = lambda *args, **kwargs: "unused"
        adaptation._sa_call_ollama = lambda *args, **kwargs: "unused"
        adaptation._sa_latest_round_csv = lambda *args, **kwargs: (None, None)
        adaptation._sa_mode_from_policy = lambda value: (
            "few" if value == "Single AI-Agent (Few-Shot)" else "wrong"
        )
        adaptation._sa_parse_output = lambda *args, **kwargs: ({}, "", True)

        def fake_generate(*args, audit=None, **kwargs):
            audit.update({
                "status": "success",
                "prompt": "prompt",
                "raw_response": "response",
                "latency_ms": 1.0,
            })
            self.assertEqual("few", args[1])
            return {
                "client_selector": "ON",
                "selection_value": 99,
                "message_compressor": "ON",
                "heterogeneous_data_handler": "OFF",
            }, "rationale"

        adaptation._sa_generate_with_retry = fake_generate

        class DummyManager(module.SingleAgentPolicyMixin):
            policy = "Distilled Policy"
            sa_model = "deepseek-r1:8b"
            sa_ollama_urls = ["http://unused"]

            def _read_previous_ap_state(self):
                return {
                    "client_selector": "OFF",
                    "message_compressor": "OFF",
                    "heterogeneous_data_handler": "OFF",
                }

            def _fix_selection_value(self, value, base_config):
                return 3

        previous = sys.modules.get("adaptation")
        sys.modules["adaptation"] = adaptation
        try:
            audit = {}
            next_config, _ = DummyManager()._decide_single_agent(
                config(),
                2,
                teacher_policy="Single AI-Agent (Few-Shot)",
                audit=audit,
            )
        finally:
            if previous is None:
                del sys.modules["adaptation"]
            else:
                sys.modules["adaptation"] = previous

        self.assertTrue(next_config["patterns"]["client_selector"]["enabled"])
        self.assertEqual(
            3,
            next_config["patterns"]["client_selector"]["params"]["selection_value"],
        )
        self.assertEqual(
            99, audit["action_before_guardrail"]["client_selector"]["selection_value"]
        )
        self.assertEqual("selection_value_adjusted", audit["guardrail_result"])

    def test_v0_settings_are_explicit_and_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = load_live_policy_settings(config(), root)
            self.assertEqual("always_defer", settings.mode)
            self.assertEqual(
                (root / "performance" / "distill_decisions").resolve(),
                settings.trace_dir,
            )

            invalid = config()
            invalid["distill_policy"]["mode"] = "active"
            with self.assertRaisesRegex(ValueError, "missing=.*artifact_kind"):
                load_live_policy_settings(invalid, root)

            escaping = config()
            escaping["distill_policy"]["trace_dir"] = "../outside"
            with self.assertRaisesRegex(ValueError, "working directory"):
                load_live_policy_settings(escaping, root)

    def test_always_defer_captures_state_without_claiming_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "performance" / "FLwithAP_performance_metrics.csv"
            write_metrics(metrics)
            settings = load_live_policy_settings(config(), root)
            request = build_always_defer_request(settings, config(), metrics, 2)
            dispatch = always_defer_dispatch(request)

        self.assertEqual("sprint03/smoke/r1::d2", request["record_id"])
        self.assertEqual("2", request["state"]["round_idx"])
        self.assertIsNone(request["rule_set_id"])
        self.assertTrue(dispatch["requires_teacher"])
        self.assertIsNone(dispatch["proposed_action"])

    def test_success_trace_separates_teacher_candidate_and_guarded_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "performance" / "FLwithAP_performance_metrics.csv"
            write_metrics(metrics)
            settings = load_live_policy_settings(config(), root)
            request = build_always_defer_request(settings, config(), metrics, 2)
            dispatch = always_defer_dispatch(request)
            teacher_audit = {
                "status": "success",
                "prompt": "exact prompt",
                "raw_response": "exact response",
                "latency_ms": 1250.0,
                "action_before_guardrail": {
                    "client_selector": {"enabled": "ON", "selection_value": 99},
                    "message_compressor": {"enabled": "ON"},
                    "heterogeneous_data_handler": {"enabled": "OFF"},
                },
                "applied_action": action_from_config(guarded_config()),
                "guardrail_result": "selection_value_adjusted",
            }
            trace_path = write_always_defer_trace(
                settings,
                request,
                dispatch,
                teacher_audit,
                config(),
                1300,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

            validate_decision_trace(trace)
            self.assertEqual(99, trace["teacher_resolution"]["action"][
                "client_selector"
            ]["selection_value"])
            self.assertEqual(3, trace["application"]["applied_action"][
                "client_selector"
            ]["selection_value"])
            self.assertEqual(
                "exact prompt",
                (settings.trace_dir / "teacher_io" / "d2-prompt.txt").read_text(
                    encoding="utf-8"
                ),
            )
            with self.assertRaises(FileExistsError):
                write_always_defer_trace(
                    settings, request, dispatch, teacher_audit, config(), 1300,
                )

    def test_teacher_error_records_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = root / "performance" / "FLwithAP_performance_metrics.csv"
            write_metrics(metrics)
            settings = load_live_policy_settings(config(), root)
            request = build_always_defer_request(settings, config(), metrics, 2)
            dispatch = always_defer_dispatch(request)
            trace_path = write_always_defer_trace(
                settings,
                request,
                dispatch,
                {
                    "status": "error",
                    "error_type": "TimeoutError",
                    "error_message": "teacher deadline exceeded",
                    "latency_ms": 5000.0,
                },
                config(),
                5100,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

        validate_decision_trace(trace)
        self.assertEqual("safe_fallback", trace["application"]["source"])
        self.assertEqual("OFF", trace["application"]["applied_action"][
            "client_selector"
        ]["enabled"])

    def test_rule_action_uses_the_existing_selector_guardrail(self) -> None:
        proposal = {
            "client_selector": {"enabled": "ON", "selection_value": 3},
            "message_compressor": {"enabled": "ON"},
            "heterogeneous_data_handler": {"enabled": "OFF"},
        }
        next_config, audit = apply_rule_action(
            config(), proposal, lambda value, runtime: 2,
        )
        self.assertEqual("selection_value_adjusted", audit["guardrail_result"])
        self.assertEqual(3, audit["action_before_guardrail"][
            "client_selector"
        ]["selection_value"])
        self.assertEqual(2, audit["applied_action"][
            "client_selector"
        ]["selection_value"])
        self.assertTrue(next_config["patterns"]["message_compressor"]["enabled"])

    def test_mirrored_runtime_files_remain_identical(self) -> None:
        for filename in (
            "adaptation.py", "agent_policy_single.py", "agent_prompting.py",
        ):
            self.assertEqual(
                (REPOSITORY / "Docker" / filename).read_bytes(),
                (REPOSITORY / "Local" / filename).read_bytes(),
                filename,
            )


if __name__ == "__main__":
    unittest.main()
