#!/usr/bin/env python3
"""Build the pinned provisional rule artifact used by Sprint 03."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from qualification_rules import build_qualification_artifact


DISTILL_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    DISTILL_ROOT
    / "data"
    / "paper_archive_fashionmnist_cnn16k_fewshot_deepseek"
)


def _require_clean_producing_checkout(producing_code_sha: str) -> None:
    repository = DISTILL_ROOT.parent
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != producing_code_sha:
        raise ValueError("producing_code_sha must equal the checked-out AP4Fed HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("qualification artifacts require a clean AP4Fed checkout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate the frozen Sprint 01 CONFOLD reports for qualification use."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Frozen Sprint 01 dataset directory.",
    )
    parser.add_argument(
        "--producing-code-sha",
        required=True,
        help="Reviewed AP4Fed Git SHA that produces the artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output JSON path; existing files are not replaced.",
    )
    args = parser.parse_args()
    _require_clean_producing_checkout(args.producing_code_sha)
    artifact = build_qualification_artifact(
        args.source.resolve(), args.producing_code_sha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
