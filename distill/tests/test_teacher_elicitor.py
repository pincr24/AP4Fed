from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


DISTILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DISTILL_ROOT))

from archive_extractor import (  # noqa: E402
    ExperimentSource,
    extract_experiment,
    write_extraction,
)
from teacher_elicitor import (  # noqa: E402
    _offline_prompt_runtime,
    build_teacher_prompt,
    label_selected_states,
    load_selection,
    load_teacher_modules,
    materialize_teacher_view,
    validate_selected_states,
)


AP_COLUMN = (
    "AP List (client_selector,client_cluster,message_compressor,"
    "model_co-versioning_registry,multi-task_model_trainer,"
    "heterogeneous_data_handler)"
)


class TeacherElicitorTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path | str]:
        source = root / "archive" / "fashion-configuration"
        source.mkdir(parents=True)
        config = {
            "adaptation": "Random",
            "LLM": "unused-model",
            "rounds": 3,
            "clients": 3,
            "client_details": [
                {
                    "client_id": 1,
                    "dataset": "FashionMNIST",
                    "model": "CNN 16k",
                    "cpu": 5,
                    "ram": 2,
                    "data_distribution_type": "IID",
                    "data_persistence_type": "Same Data",
                    "delay_combobox": "No",
                },
                {
                    "client_id": 2,
                    "dataset": "FashionMNIST",
                    "model": "CNN 16k",
                    "cpu": 5,
                    "ram": 2,
                    "data_distribution_type": "IID",
                    "data_persistence_type": "New Data",
                    "delay_combobox": "No",
                },
                {
                    "client_id": 3,
                    "dataset": "FashionMNIST",
                    "model": "CNN 16k",
                    "cpu": 3,
                    "ram": 2,
                    "data_distribution_type": "non-IID",
                    "data_persistence_type": "New Data",
                    "delay_combobox": "Yes",
                },
            ],
        }
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        fields = [
            "Client ID", "FL Round", "Training Time", "JSD",
            "Communication Time", "Total Time of FL Round", "Val F1", AP_COLUMN,
        ]
        vectors = {
            1: "{OFF,OFF,OFF,OFF,OFF,OFF}",
            2: "{ON,OFF,OFF,OFF,OFF,OFF}",
            3: "{ON,OFF,ON,OFF,OFF,ON}",
        }
        with (source / "r1.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for round_idx in range(1, 4):
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

        source_id = "test-archive/fashion/random"
        output_name = "fashion-random"
        state_bank_root = root / "state-bank"
        output = state_bank_root / output_name
        result = extract_experiment(ExperimentSource(source, source_id))
        write_extraction(output, result)

        sources_file = root / "sources.json"
        sources_file.write_text(json.dumps({
            "schema_version": 1,
            "experiments": [{
                "source": str(source.relative_to(root)),
                "source_id": source_id,
                "output_name": output_name,
                "label_mode": "states-only",
            }],
        }), encoding="utf-8")
        record_id = f"{source_id}/r1::d2"
        selection = root / "selection.json"
        selection.write_text(json.dumps({
            "schema_version": 1,
            "selection_id": "teacher-screen-test-v1",
            "selection_rule": "One fixed warm-start fixture for parser calibration.",
            "records": [{
                "record_id": record_id,
                "attempts": 2,
                "purpose": "calibration",
            }],
        }), encoding="utf-8")
        return {
            "source": source,
            "sources_file": sources_file,
            "state_bank_root": state_bank_root,
            "selection": selection,
            "record_id": record_id,
        }

    def test_prompt_metrics_root_matches_runtime_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            _, states, summary = validate_selected_states(
                fixture["sources_file"],
                fixture["state_bank_root"],
                fixture["selection"],
            )
            state = states[0]
            modules = load_teacher_modules()
            view_root = Path(temporary) / "teacher-view"
            last_round, _ = materialize_teacher_view(state, view_root)
            hooked_prompt, aggregate = build_teacher_prompt(
                modules,
                state,
                "Single AI-Agent (Few-Shot)",
                "deepseek-r1:8b",
                view_root,
                last_round,
            )

            config = json.loads(json.dumps(state.config))
            config["adaptation"] = "Single AI-Agent (Few-Shot)"
            config["LLM"] = "deepseek-r1:8b"
            previous = {
                "client_selector": state.state["prev_cs"],
                "message_compressor": state.state["prev_mc"],
                "heterogeneous_data_handler": state.state["prev_hdh"],
            }
            original_cwd = Path.cwd()
            try:
                os.chdir(view_root)
                with _offline_prompt_runtime(modules.prompting, view_root):
                    runtime_prompt = modules.prompting._sa_build_prompt(
                        "few", config, last_round, aggregate, previous,
                    )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(runtime_prompt, hooked_prompt)
        self.assertEqual({"calibration": 1}, summary["selected_state_purposes"])
        self.assertEqual({"calibration": 2}, summary["query_budget_by_purpose"])
        self.assertIn(
            '"file": "performance/FLwithAP_performance_metrics_round1.csv"',
            hooked_prompt,
        )

    def test_local_and_docker_prompt_helpers_remain_mirrored(self) -> None:
        repository = DISTILL_ROOT.parent
        for filename in ("agent_prompting.py", "adaptation_metrics.py"):
            docker = (repository / "Docker" / filename).read_bytes()
            local = (repository / "Local" / filename).read_bytes()
            self.assertEqual(docker, local, filename)

    def test_campaign_screen_is_a_subset_of_the_full_state_bank(self) -> None:
        campaigns = DISTILL_ROOT / "campaigns"
        screen = load_selection(
            campaigns / "agentic_paper_teacher_screen_v1.json"
        )
        full = load_selection(
            campaigns / "agentic_paper_teacher_full_v1.json"
        )
        screen_ids = {record.record_id for record in screen.records}
        full_ids = {record.record_id for record in full.records}
        calibration = [
            record for record in screen.records
            if record.purpose == "calibration"
        ]

        self.assertEqual(39, len(screen_ids))
        self.assertEqual(1170, len(full_ids))
        self.assertTrue(screen_ids < full_ids)
        self.assertEqual(79, screen.query_budget)
        self.assertEqual(1170, full.query_budget)
        self.assertEqual(10, len(calibration))
        self.assertTrue(all(record.attempts == 5 for record in calibration))
        self.assertEqual(screen.state_bank_sha256, full.state_bank_sha256)

    def test_selection_can_pin_the_state_bank_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.make_fixture(Path(temporary))
            selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
            selection["state_bank_sha256"] = "0" * 64
            fixture["selection"].write_text(json.dumps(selection), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "state_bank_sha256 differs"):
                validate_selected_states(
                    fixture["sources_file"],
                    fixture["state_bank_root"],
                    fixture["selection"],
                )

    def test_fake_queries_write_resumable_provenance_and_guardrailed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            output = root / "label-run"
            responses = iter([
                json.dumps({
                    "client_selector": "ON",
                    "selection_value": 99,
                    "message_compressor": "ON",
                    "heterogeneous_data_handler": "OFF",
                    "rationale": "Calibration response. CS=ON; MC=ON; HDH=OFF",
                }),
                json.dumps({
                    "client_selector": "OFF",
                    "message_compressor": "OFF",
                    "heterogeneous_data_handler": "ON",
                    "rationale": "Second response. CS=OFF; MC=OFF; HDH=ON",
                }),
            ])
            prompts = []

            def fake_teacher(model, prompt, base_urls, options):
                prompts.append((model, prompt, base_urls, options))
                return next(responses)

            result = label_selected_states(
                sources_file=fixture["sources_file"],
                state_bank_root=fixture["state_bank_root"],
                selection_file=fixture["selection"],
                output=output,
                run_id="fake-teacher-run-v1",
                model_identity={"name": "deepseek-r1:8b", "digest": "a" * 64},
                teacher_call=fake_teacher,
                allow_dirty=True,
            )
            with (output / "labels.csv").open(newline="", encoding="utf-8") as stream:
                labels = list(csv.DictReader(stream))
            attempts = [
                json.loads(line)
                for line in (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            raw_response_count = len(list((output / "raw_responses").glob("*.txt")))

            def unexpected_query(*args):
                raise AssertionError("resume repeated an already recorded query")

            resumed = label_selected_states(
                sources_file=fixture["sources_file"],
                state_bank_root=fixture["state_bank_root"],
                selection_file=fixture["selection"],
                output=output,
                run_id="fake-teacher-run-v1",
                model_identity={"name": "deepseek-r1:8b", "digest": "a" * 64},
                teacher_call=unexpected_query,
                resume=True,
                allow_dirty=True,
            )
            next((output / "raw_responses").glob("*.txt")).write_text(
                "tampered response", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "raw response hash differs"):
                label_selected_states(
                    sources_file=fixture["sources_file"],
                    state_bank_root=fixture["state_bank_root"],
                    selection_file=fixture["selection"],
                    output=output,
                    run_id="fake-teacher-run-v1",
                    model_identity={"name": "deepseek-r1:8b", "digest": "a" * 64},
                    teacher_call=unexpected_query,
                    resume=True,
                    allow_dirty=True,
                )

        self.assertEqual(2, len(prompts))
        self.assertEqual(2, result["successful_labels"])
        self.assertEqual(0, resumed["new_attempts"])
        self.assertEqual(2, len(labels))
        self.assertEqual("offline_teacher_query", labels[0]["label_kind"])
        self.assertEqual("ON", labels[0]["y_cs_applied"])
        self.assertEqual("3", labels[0]["selection_value"])
        self.assertEqual("selection_value_adjusted", attempts[0]["guardrail_result"])
        self.assertEqual("success", attempts[0]["status"])
        self.assertEqual("complete", manifest["status"])
        self.assertEqual(2, manifest["inputs"]["query_budget"])
        self.assertEqual("a" * 64, attempts[0]["teacher_model_digest"])
        self.assertEqual(2, raw_response_count)
        self.assertNotIn("rationale", labels[0])

    def test_invalid_response_is_audited_without_becoming_a_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
            selection["records"][0] = {
                "record_id": fixture["record_id"],
                "attempts": 1,
                "purpose": "primary",
            }
            fixture["selection"].write_text(json.dumps(selection), encoding="utf-8")
            output = root / "invalid-label-run"

            result = label_selected_states(
                sources_file=fixture["sources_file"],
                state_bank_root=fixture["state_bank_root"],
                selection_file=fixture["selection"],
                output=output,
                run_id="invalid-teacher-run-v1",
                model_identity={"name": "deepseek-r1:8b", "digest": "b" * 64},
                teacher_call=lambda *args: "not a parseable teacher decision",
                allow_dirty=True,
            )
            with (output / "labels.csv").open(newline="", encoding="utf-8") as stream:
                labels = list(csv.DictReader(stream))
            attempt = json.loads(
                (output / "attempts.jsonl").read_text(encoding="utf-8").strip()
            )
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(1, result["invalid_responses"])
        self.assertEqual([], labels)
        self.assertEqual("invalid_response", attempt["status"])
        self.assertEqual("invalid", attempt["parser_status"])
        self.assertEqual("partial", manifest["status"])

    def test_resume_refuses_to_repeat_an_unresolved_inflight_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.make_fixture(root)
            output = root / "interrupted-label-run"

            def interrupted_teacher(*args):
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                label_selected_states(
                    sources_file=fixture["sources_file"],
                    state_bank_root=fixture["state_bank_root"],
                    selection_file=fixture["selection"],
                    output=output,
                    run_id="interrupted-teacher-run-v1",
                    model_identity={"name": "deepseek-r1:8b", "digest": "c" * 64},
                    teacher_call=interrupted_teacher,
                    allow_dirty=True,
                )

            markers = list((output / "inflight").glob("*.json"))
            with self.assertRaisesRegex(RuntimeError, "unresolved in-flight query"):
                label_selected_states(
                    sources_file=fixture["sources_file"],
                    state_bank_root=fixture["state_bank_root"],
                    selection_file=fixture["selection"],
                    output=output,
                    run_id="interrupted-teacher-run-v1",
                    model_identity={"name": "deepseek-r1:8b", "digest": "c" * 64},
                    teacher_call=lambda *args: self.fail("repeated interrupted query"),
                    resume=True,
                    allow_dirty=True,
                )

        self.assertEqual(1, len(markers))


if __name__ == "__main__":
    unittest.main()
