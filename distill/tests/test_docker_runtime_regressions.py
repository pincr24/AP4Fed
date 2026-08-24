"""Regression checks for Docker-backend failures found by Sprint 03 control runs."""

import ast
import csv
import io
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def load_function(path: Path, name: str, namespace: dict[str, object]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class DockerRuntimeRegressionTests(unittest.TestCase):
    def test_agnews_readers_accept_a_field_larger_than_the_csv_default(self):
        oversized_article = "x" * 131073
        for relative_path in ("Docker/taskA.py", "Local/taskA.py"):
            path = REPOSITORY / relative_path
            configure = load_function(path, "_configure_csv_field_size_limit", {
                "csv": csv,
                "sys": sys,
            })
            configure()
            rows = list(csv.reader(io.StringIO(f"1,{oversized_article}\n")))
            self.assertEqual([["1", oversized_article]], rows, relative_path)

    def test_docker_aggregation_defaults_ssim_overhead_to_none(self):
        path = REPOSITORY / "Docker/server.py"
        logged = []
        aggregate = load_function(path, "weighted_average_global", {
            "global_metrics": {},
            "currentRnd": 1,
            "log_round_time": lambda *args: logged.append(args),
        })
        aggregate(
            [(1, {"client_id": "Client1"})],
            "FedAvg",
            0.0,
            0.0,
            0.0,
        )
        self.assertEqual(1, len(logged))
        self.assertIsNone(logged[0][-2])


if __name__ == "__main__":
    unittest.main()
