#!/usr/bin/env python3
"""Extract normalized datasets from one or more caller-selected AP4Fed experiments.

Single-source mode is useful for inspection. Source-list mode are used for an extraction
campaign, where experiment paths and identifiers reside in a JSON file. 
Both modes call the reusable ``archive_extractor`` module.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from archive_extractor import (
    LABEL_MODES,
    ExperimentSource,
    extract_experiment,
    write_extraction,
)


@dataclass(frozen=True)
class ExtractionJob:
    """Pair a caller-supplied experiment source with its output directory."""

    experiment: ExperimentSource
    output: Path


def _resolve_relative(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def load_source_list(path: Path, output_root: Path) -> list[ExtractionJob]:
    """Load a portable list of experiments chosen by a campaign orchestrator.

    Source paths are resolved relative to the list file. Each ``output_name``
    is a simple directory name below ``output_root``.
    """
    path = path.resolve()
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"{path}: expected a schema_version 1 object")
    entries = document.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: experiments must be a non-empty list")

    jobs: list[ExtractionJob] = []
    source_ids: set[str] = set()
    output_names: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: experiment {index} must be an object")
        required = ("source", "source_id", "output_name")
        missing = [key for key in required if not str(entry.get(key, "")).strip()]
        if missing:
            raise ValueError(f"{path}: experiment {index} is missing {missing}")
        source_id = str(entry["source_id"]).strip().rstrip("/")
        output_name = str(entry["output_name"]).strip()
        label_mode = str(entry.get("label_mode", "states-only"))
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"{path}: experiment {index} label_mode must be one of {LABEL_MODES}"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name):
            raise ValueError(
                f"{path}: experiment {index} output_name must be one safe directory name"
            )
        if source_id in source_ids:
            raise ValueError(f"{path}: duplicate source_id {source_id!r}")
        if output_name in output_names:
            raise ValueError(f"{path}: duplicate output_name {output_name!r}")
        source_ids.add(source_id)
        output_names.add(output_name)
        jobs.append(ExtractionJob(
            experiment=ExperimentSource(
                source=_resolve_relative(str(entry["source"]), path.parent),
                source_id=source_id,
                label_mode=label_mode,
            ),
            output=output_root.resolve() / output_name,
        ))
    return jobs


def _single_job(args: argparse.Namespace) -> ExtractionJob:
    missing = [name for name in ("source_id", "output") if getattr(args, name) is None]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"single-source mode also requires {options}")
    return ExtractionJob(
        experiment=ExperimentSource(
            source=args.source,
            source_id=args.source_id,
            label_mode=args.label_mode,
        ),
        output=args.output,
    )


def _jobs_from_args(args: argparse.Namespace) -> list[ExtractionJob]:
    if args.sources_file:
        if args.output_root is None:
            raise ValueError("source-list mode requires --output-root")
        if args.source_id is not None or args.output is not None:
            raise ValueError("--source-id and --output belong to single-source mode")
        return load_source_list(args.sources_file, args.output_root)
    if args.output_root is not None:
        raise ValueError("--output-root belongs to source-list mode")
    return [_single_job(args)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--source", type=Path,
        help="one directory containing config.json and exact r<N>.csv files",
    )
    selection.add_argument(
        "--sources-file", type=Path,
        help="schema-versioned JSON list of experiments to extract",
    )
    parser.add_argument(
        "--source-id",
        help="stable provenance identifier for single-source mode",
    )
    parser.add_argument(
        "--output", type=Path,
        help="dataset directory for single-source mode",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="parent directory for source-list outputs",
    )
    parser.add_argument(
        "--label-mode",
        choices=("states-only", "source-behavior"),
        default="states-only",
        help="single-source label mode; list entries carry their own mode",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing normalized state, label, and audit outputs",
    )
    args = parser.parse_args()

    try:
        jobs = _jobs_from_args(args)
    except ValueError as error:
        parser.error(str(error))

    for job in jobs:
        result = extract_experiment(job.experiment)
        write_extraction(job.output, result, overwrite=args.overwrite)
        source = result.audit["state_source"]
        print(
            f"{result.audit['source_id']}: {result.audit['decision_rows']} rows, "
            f"{result.audit['run_count']} runs, {source['workload']} -> {job.output}"
        )


if __name__ == "__main__":
    main()
