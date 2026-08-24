#!/usr/bin/env python3
"""Prepare one immutable Sprint 03 Docker run configuration."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from qualification_rules import (
    QUALIFICATION_ARTIFACT_KIND,
    load_qualification_artifact,
    validate_qualification_configuration,
)


DISTILL_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = DISTILL_ROOT / "configs" / "sprint03_agnews_base.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")


def prepare_run_config(
    base: dict[str, object],
    mode: str,
    run_id: str,
    teacher_model_digest: str,
    artifact_path: Path | None = None,
) -> dict[str, object]:
    """Return a checked control, shadow, or qualification-active config."""
    if mode not in {"always_defer", "shadow", "active"}:
        raise ValueError("mode must be always_defer, shadow, or active")
    if not SAFE_RUN_ID_RE.fullmatch(run_id) or "::" in run_id:
        raise ValueError("run_id must be a stable identifier without '::'")
    if not SHA256_RE.fullmatch(teacher_model_digest):
        raise ValueError("teacher_model_digest must be a full lowercase SHA-256")

    prepared = copy.deepcopy(base)
    prepared["adaptation"] = "Distilled Policy"
    validate_qualification_configuration(prepared)
    policy: dict[str, object] = {
        "mode": mode,
        "run_id": run_id,
        "teacher_policy": "Single AI-Agent (Few-Shot)",
        "teacher_model_digest": teacher_model_digest,
        "trace_dir": f"performance/distill_decisions/{run_id.replace('/', '-')}",
    }
    if mode == "always_defer":
        if artifact_path is not None:
            raise ValueError("always_defer mode must not receive a rule artifact")
    else:
        if artifact_path is None:
            raise ValueError("shadow and active modes require a rule artifact")
        resolved = artifact_path.resolve()
        try:
            relative = resolved.relative_to(DISTILL_ROOT)
        except ValueError as error:
            raise ValueError("rule artifact must be stored below distill/") from error
        loaded = load_qualification_artifact(resolved)
        policy.update({
            "artifact_kind": QUALIFICATION_ARTIFACT_KIND,
            "rule_artifact": relative.as_posix(),
            "rule_artifact_sha256": loaded.sha256,
        })
    prepared["distill_policy"] = policy
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned Sprint 03 AG News Docker configuration."
    )
    parser.add_argument("--mode", choices=("always_defer", "shadow", "active"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--teacher-model-digest", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    prepared = prepare_run_config(
        base,
        args.mode,
        args.run_id,
        args.teacher_model_digest,
        args.artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(prepared, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
