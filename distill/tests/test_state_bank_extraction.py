from __future__ import annotations

import csv
import json
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
from build_state_bank import load_source_list  # noqa: E402
from decision_state import build_live_state  # noqa: E402


AP_COLUMN = (
    "AP List (client_selector,client_cluster,message_compressor,"
    "model_co-versioning_registry,multi-task_model_trainer,"
    "heterogeneous_data_handler)"
)


class ConfiguredExtractorTest(unittest.TestCase):
    def make_configuration(self, root: Path) -> Path:
        source = root / "configuration"
        source.mkdir()
        config = {
            "adaptation": "Random",
            "LLM": "unused-model",
            "rounds": 3,
            "clients": 2,
            "client_details": [
                {
                    "dataset": "AG_NEWS",
                    "model": "MLP",
                    "cpu": 4,
                    "data_distribution_type": "IID",
                    "data_persistence_type": "Same Data",
                    "delay_combobox": "No",
                },
                {
                    "dataset": "AG_NEWS",
                    "model": "MLP",
                    "cpu": 2,
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
            3: "{ON,OFF,ON,OFF,OFF,OFF}",
        }
        with (source / "r1.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for round_idx in range(1, 4):
                for client_idx in range(1, 3):
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
        (source / "r1_rationales.csv").write_text("not,a,run\n", encoding="utf-8")
        return source

    def test_states_only_separates_source_actions_from_teacher_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_configuration(Path(tmp))
            result = extract_experiment(ExperimentSource(source, "archive/ag-news/random"))

        self.assertEqual(2, len(result.states))
        self.assertEqual(0, len(result.labels))
        self.assertEqual("AG_NEWS__MLP", result.states[0]["workload"])
        self.assertNotIn("state_source_policy", result.states[0])
        self.assertNotIn("teacher_policy", result.states[0])
        self.assertNotIn("label_kind", result.states[0])
        self.assertEqual("Random", result.audit["state_source"]["policy"])
        self.assertEqual("unused-model", result.audit["state_source"]["configured_model"])
        run_id = "archive/ag-news/random/r1"
        self.assertEqual(
            "{ON,OFF,OFF,OFF,OFF,OFF}",
            result.audit["run_registry"][run_id]["factual_source_actions"]["1"],
        )
        self.assertEqual("states-only", result.audit["label_mode"])
        self.assertEqual(1, result.audit["run_count"])
        self.assertEqual(2, result.audit["decisions_per_run"])
        self.assertEqual(
            {"OFF": 1, "ON": 1},
            result.audit["factual_source_action_counts"]["y_mc_applied"],
        )

    def test_source_behavior_is_an_explicit_label_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_configuration(Path(tmp))
            result = extract_experiment(
                ExperimentSource(
                    source, "archive/ag-news/random", label_mode="source-behavior"
                )
            )

        self.assertEqual(2, len(result.states))
        self.assertEqual(2, len(result.labels))
        self.assertEqual("Random", result.labels[0]["teacher_policy"])
        self.assertEqual("direct_archived_behavior", result.labels[0]["label_kind"])
        self.assertEqual("archived-source-behavior", result.labels[0]["attempt_id"])
        self.assertEqual("ON", result.labels[0]["y_cs_applied"])
        self.assertEqual("", result.labels[0]["selection_value"])
        self.assertEqual("OFF", result.labels[0]["y_mc_applied"])
        self.assertEqual("source-behavior", result.audit["label_mode"])

    def test_writer_separates_states_labels_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_configuration(root)
            result = extract_experiment(ExperimentSource(source, "archive/ag-news/random"))
            output = root / "output"
            write_extraction(output, result)
            with (output / "decision_states.csv").open(newline="", encoding="utf-8") as stream:
                state_reader = csv.DictReader(stream)
                state = next(state_reader)
                state_columns = state_reader.fieldnames
            with (output / "labels.csv").open(newline="", encoding="utf-8") as stream:
                label_reader = csv.DictReader(stream)
                label_rows = list(label_reader)
                label_columns = label_reader.fieldnames
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))

        self.assertEqual(["record_id", "run_id"], state_columns[:2])
        self.assertNotIn("label_kind", state)
        self.assertNotIn("source_sha256", state)
        self.assertIn("label_kind", label_columns)
        self.assertEqual([], label_rows)
        self.assertEqual("Random", audit["state_source"]["policy"])
        self.assertIn("archive/ag-news/random/r1", audit["run_registry"])

    def test_source_list_supplies_experiments_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_configuration(root)
            source_list = root / "sources.json"
            source_list.write_text(json.dumps({
                "schema_version": 1,
                "experiments": [{
                    "source": source.name,
                    "source_id": "archive/ag-news/random",
                    "output_name": "ag-news-random",
                }],
            }), encoding="utf-8")

            jobs = load_source_list(source_list, root / "outputs")
            expected_source = source.resolve()
            expected_output = (root / "outputs" / "ag-news-random").resolve()

        self.assertEqual(1, len(jobs))
        self.assertEqual(expected_source, jobs[0].experiment.source)
        self.assertEqual("archive/ag-news/random", jobs[0].experiment.source_id)
        self.assertEqual(expected_output, jobs[0].output)

    def test_live_prefix_state_matches_archive_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_configuration(Path(tmp))
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            with (source / "r1.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            extracted = extract_experiment(
                ExperimentSource(source, "archive/ag-news/random")
            ).states

        for decision_idx in (1, 2):
            live_rows = [
                row for row in rows if int(row["FL Round"]) <= decision_idx
            ]
            live_state = build_live_state(config, live_rows, decision_idx)
            archived_state = {
                feature: (
                    "" if extracted[decision_idx - 1][feature] == ""
                    else str(extracted[decision_idx - 1][feature])
                )
                for feature in live_state
            }
            self.assertEqual(archived_state, live_state)


if __name__ == "__main__":
    unittest.main()
