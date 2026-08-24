#!/usr/bin/env python3
"""Label selected normalized decision states with an AP4Fed single-agent teacher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from archive_extractor import LABEL_RECORD_COLUMNS
from teacher_adapter import (
    DECISION_KEYS,
    SIGNATURE_RE,
    TeacherModules,
    _offline_prompt_runtime,
    _read_run_rows,
    _recognised_decision,
    _response_object,
    _valid_selection_values,
    _write_rows_through,
    apply_selector_guardrail,
    build_teacher_prompt,
    load_teacher_modules,
    materialize_teacher_view,
    parse_teacher_response,
)
from teacher_campaign import (
    SAFE_ID_RE,
    SELECTION_SCHEMA_VERSION,
    SelectedState,
    SelectionPlan,
    SelectionRecord,
    _read_json,
    _read_states,
    _sha256,
    _sha256_bytes,
    _stringify_row,
    _validate_recreated_dataset,
    load_selection,
    validate_selected_states,
)


ATTEMPT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
LABEL_KIND = "offline_teacher_query"
DEFAULT_TEACHER_POLICY = "Single AI-Agent (Few-Shot)"
DEFAULT_TEACHER_MODEL = "deepseek-r1:8b"
DEFAULT_OPTIONS = {"temperature": 1.0, "top_p": 0.9, "num_ctx": 8192}
MODEL_DIGEST_RE = re.compile(r"(?:sha256:)?[0-9a-fA-F]{64}")

TeacherCall = Callable[[str, str, list[str], dict[str, object]], str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id(run_id: str, record_id: str, sample_index: int) -> str:
    state_digest = _sha256_bytes(record_id.encode("utf-8"))[:16]
    return f"{run_id}-{state_digest}-s{sample_index:03d}"


def _load_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    attempts = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            attempt_id = value.get("attempt_id") if isinstance(value, dict) else None
            if not attempt_id or attempt_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate attempt_id")
            seen.add(attempt_id)
            attempts.append(value)
    return attempts


def _validate_prior_attempts(
    output: Path,
    attempts: Sequence[dict[str, object]],
    planned_ids: set[str],
) -> None:
    """Verify resumed attempts still belong to the plan and retain intact evidence."""
    for attempt in attempts:
        attempt_id = str(attempt["attempt_id"])
        if attempt_id not in planned_ids:
            raise ValueError(f"{attempt_id}: attempt is absent from the current plan")
        status = attempt.get("status")
        if status not in {"success", "invalid_response", "error"}:
            raise ValueError(f"{attempt_id}: unsupported attempt status {status!r}")
        raw_name = attempt.get("raw_response_file")
        if status in {"success", "invalid_response"} and not raw_name:
            raise ValueError(f"{attempt_id}: completed response has no raw artifact")
        if not raw_name:
            continue
        raw_path = (output / str(raw_name)).resolve()
        if output not in raw_path.parents:
            raise ValueError(f"{attempt_id}: raw response path leaves the run directory")
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if attempt.get("raw_response_sha256") != _sha256(raw_path):
            raise ValueError(f"{attempt_id}: raw response hash differs from its audit")
        if status == "success" and not isinstance(attempt.get("label"), dict):
            raise ValueError(f"{attempt_id}: successful attempt has no label")


def _append_attempt(path: Path, attempt: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(attempt, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_atomic(path: Path, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("teacher response must be text")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_labels(path: Path, attempts: Sequence[dict[str, object]]) -> None:
    labels = [attempt["label"] for attempt in attempts if attempt.get("status") == "success"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_RECORD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(labels)
    temporary.replace(path)


def _git_identity(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True,
        ).strip()

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(status)}


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_model_identity(
    base_urls: Sequence[str],
    model: str,
    explicit_digest: str | None = None,
) -> dict[str, object]:
    """Resolve the exact served model digest before starting a labelling run."""
    def server_version(base_url: str) -> str | None:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/version", timeout=15,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
            return str(value.get("version") or "") or None
        except Exception:
            return None

    if explicit_digest:
        if not MODEL_DIGEST_RE.fullmatch(explicit_digest):
            raise ValueError("--model-digest must be a full SHA-256 digest")
        return {
            "name": model,
            "digest": explicit_digest,
            "source": "command-line",
            "server_version": server_version(base_urls[0]) if base_urls else None,
        }
    errors = []
    for base_url in base_urls:
        try:
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/tags", timeout=15,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload.get("models", []):
                if item.get("name") == model or item.get("model") == model:
                    digest = item.get("digest")
                    if not isinstance(digest, str) or not MODEL_DIGEST_RE.fullmatch(digest):
                        raise ValueError(f"{base_url}: model has no full SHA-256 digest")
                    return {
                        "name": model,
                        "digest": digest,
                        "size": item.get("size"),
                        "modified_at": item.get("modified_at"),
                        "details": item.get("details"),
                        "server_version": server_version(base_url),
                        "source": f"{base_url.rstrip('/')}/api/tags",
                    }
            errors.append(f"{base_url}: model not found")
        except Exception as error:
            errors.append(f"{base_url}: {type(error).__name__}: {error}")
    raise RuntimeError("teacher model identity unavailable: " + "; ".join(errors))


def _default_teacher_call(modules: TeacherModules) -> TeacherCall:
    def call(model: str, prompt: str, base_urls: list[str], options: dict[str, object]) -> str:
        return modules.client._sa_call_ollama(
            model, prompt, base_urls, force_json=True, options=options,
        )

    return call


def label_selected_states(
    *,
    sources_file: Path,
    state_bank_root: Path,
    selection_file: Path,
    output: Path,
    run_id: str,
    teacher_policy: str = DEFAULT_TEACHER_POLICY,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    base_urls: Sequence[str] = ("http://localhost:11434",),
    options: dict[str, object] | None = None,
    model_identity: dict[str, object] | None = None,
    teacher_call: TeacherCall | None = None,
    resume: bool = False,
    allow_dirty: bool = False,
) -> dict[str, int]:
    """Run the frozen query plan and write an auditable, resumable label archive."""
    if not SAFE_ID_RE.fullmatch(run_id):
        raise ValueError(f"run_id must match {SAFE_ID_RE.pattern!r}")
    plan, states, validation = validate_selected_states(
        sources_file, state_bank_root, selection_file,
    )
    modules = load_teacher_modules()
    repo = Path(__file__).resolve().parents[1]
    git = _git_identity(repo)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("refusing a canonical label run from a dirty AP4Fed checkout")
    actual_options = dict(DEFAULT_OPTIONS if options is None else options)
    identity = model_identity or resolve_model_identity(base_urls, teacher_model)
    digest = identity.get("digest")
    if not isinstance(digest, str) or not MODEL_DIGEST_RE.fullmatch(digest):
        raise ValueError("teacher model identity must contain a full SHA-256 digest")

    output = output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("label-run output must be outside the AP4Fed repository")
    attempts_path = output / "attempts.jsonl"
    labels_path = output / "labels.csv"
    manifest_path = output / "run_manifest.json"
    raw_root = output / "raw_responses"
    inflight_root = output / "inflight"
    if resume and not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: cannot resume without a run manifest")
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"{output}: use --resume to continue an existing run")
    output.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    inflight_root.mkdir(exist_ok=True)

    prior_attempts = _load_attempts(attempts_path)
    planned_ids = {
        _attempt_id(run_id, state.record_id, sample_index)
        for state in states
        for sample_index in range(1, state.selection.attempts + 1)
    }
    _validate_prior_attempts(output, prior_attempts, planned_ids)
    completed_ids = {str(attempt["attempt_id"]) for attempt in prior_attempts}
    prompt_version = {
        "agent_prompting_sha256": _sha256(modules.prompting_path),
        "adaptation_metrics_sha256": _sha256(modules.metrics_path),
        "ollama_client_sha256": _sha256(modules.client_path),
    }
    current_command = " ".join(shlex.quote(argument) for argument in sys.argv)
    current_environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "exact_command": current_command,
        "resume_commands": [],
        "producing_code": git,
        "inputs": {
            **validation,
            "selection_id": plan.selection_id,
            "selection_rule": plan.selection_rule,
        },
        "teacher": {
            "policy": teacher_policy,
            "mode": modules.prompting._sa_mode_from_policy(teacher_policy),
            "model": teacher_model,
            "model_identity": identity,
            "base_urls": list(base_urls),
            "sampling_options": actual_options,
            "prompt_version": prompt_version,
        },
        "environment": current_environment,
        "outputs": {
            "attempts": "attempts.jsonl",
            "labels": "labels.csv",
            "raw_responses": "raw_responses/",
            "inflight_queries": "inflight/",
        },
        "result": None,
    }
    if resume and manifest_path.exists():
        existing = _read_json(manifest_path)
        comparable = (
            existing.get("run_id") == run_id
            and existing.get("inputs", {}).get("selection_sha256") == plan.sha256
            and existing.get("inputs", {}).get("source_manifest_sha256")
            == validation["source_manifest_sha256"]
            and existing.get("producing_code") == git
            and existing.get("teacher", {}).get("policy") == teacher_policy
            and existing.get("teacher", {}).get("model") == teacher_model
            and existing.get("teacher", {}).get("model_identity") == identity
            and existing.get("teacher", {}).get("base_urls") == list(base_urls)
            and existing.get("teacher", {}).get("sampling_options") == actual_options
            and existing.get("teacher", {}).get("prompt_version") == prompt_version
        )
        if not comparable:
            raise ValueError(f"{manifest_path}: resume settings differ from the existing run")
        manifest["started_at"] = existing.get("started_at")
        manifest["exact_command"] = existing.get("exact_command")
        manifest["environment"] = existing.get("environment")
        resume_commands = existing.get("resume_commands", [])
        if not isinstance(resume_commands, list):
            raise ValueError(f"{manifest_path}: resume_commands must be a list")
        manifest["resume_commands"] = [
            *resume_commands,
            {"at": _utc_now(), "command": current_command, "environment": current_environment},
        ]

    for marker in sorted(inflight_root.glob("*.json")):
        marker_value = _read_json(marker)
        marker_id = str(marker_value.get("attempt_id", ""))
        if marker_id in completed_ids:
            marker.unlink()
            continue
        raise RuntimeError(
            f"{marker}: unresolved in-flight query; refusing an automatic repeat"
        )
    _write_manifest(manifest_path, manifest)

    call_teacher = teacher_call or _default_teacher_call(modules)
    new_attempts: list[dict[str, object]] = []
    for state in states:
        for sample_index in range(1, state.selection.attempts + 1):
            attempt_id = _attempt_id(run_id, state.record_id, sample_index)
            if attempt_id in completed_ids:
                continue
            raw_path = raw_root / f"{attempt_id}.txt"
            if raw_path.exists():
                raise FileExistsError(
                    f"{raw_path}: raw response exists without a completed attempt record"
                )
            started_at = _utc_now()
            started_clock = time.perf_counter()
            prompt: str | None = None
            marker_path: Path | None = None
            source_run = state.audit["run_registry"][state.run_id]
            common = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "selection_id": plan.selection_id,
                "record_id": state.record_id,
                "run_id": state.run_id,
                "sample_index": sample_index,
                "attempt_purpose": state.selection.purpose,
                "query_started_at": started_at,
                "teacher_policy": teacher_policy,
                "teacher_model": teacher_model,
                "teacher_model_digest": identity["digest"],
                "sampling_options": actual_options,
                "source_id": state.audit["source_id"],
                "source_policy": state.audit["state_source"]["policy"],
                "workload": state.audit["state_source"]["workload"],
                "source_config_sha256": state.audit["config_sha256"],
                "source_csv_sha256": source_run["source_sha256"],
                "counterfactual_outcome_warning": (
                    "archived downstream outcomes are not attributable to this queried action"
                ),
            }
            try:
                with tempfile.TemporaryDirectory(prefix="ap4fed-teacher-view-") as temporary:
                    metrics_root = Path(temporary)
                    last_round, _ = materialize_teacher_view(state, metrics_root)
                    prompt, _ = build_teacher_prompt(
                        modules, state, teacher_policy, teacher_model, metrics_root, last_round,
                    )
                    marker_candidate = inflight_root / f"{attempt_id}.json"
                    _write_json_exclusive(marker_candidate, {
                        **common,
                        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                        "prompt_version": prompt_version,
                    })
                    marker_path = marker_candidate
                    raw = call_teacher(
                        teacher_model, prompt, list(base_urls), actual_options,
                    )
                _write_text_atomic(raw_path, raw)
                parsed, parser_audit = parse_teacher_response(modules, raw)
                attempt = {
                    **common,
                    "query_completed_at": _utc_now(),
                    "latency_seconds": time.perf_counter() - started_clock,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "prompt_version": prompt_version,
                    "raw_response_file": str(raw_path.relative_to(output)),
                    "raw_response_sha256": _sha256(raw_path),
                    **parser_audit,
                }
                if parsed is None:
                    attempt["status"] = "invalid_response"
                else:
                    applied, guardrail_result = apply_selector_guardrail(parsed, state.config)
                    label = {
                        "record_id": state.record_id,
                        "attempt_id": attempt_id,
                        "label_kind": LABEL_KIND,
                        "teacher_policy": teacher_policy,
                        "teacher_model": teacher_model,
                        "y_cs_applied": applied["client_selector"],
                        "selection_value": (
                            applied.get("selection_value", "")
                            if applied["client_selector"] == "ON" else ""
                        ),
                        "y_mc_applied": applied["message_compressor"],
                        "y_hdh_applied": applied["heterogeneous_data_handler"],
                    }
                    attempt.update({
                        "status": "success",
                        "parsed_decisions": parsed,
                        "applied_decisions": applied,
                        "guardrail_result": guardrail_result,
                        "label": label,
                    })
            except Exception as error:
                attempt = {
                    **common,
                    "status": "error",
                    "query_completed_at": _utc_now(),
                    "latency_seconds": time.perf_counter() - started_clock,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                if prompt is not None:
                    attempt["prompt_sha256"] = _sha256_bytes(prompt.encode("utf-8"))
                    attempt["prompt_version"] = prompt_version
                if raw_path.exists():
                    attempt["raw_response_file"] = str(raw_path.relative_to(output))
                    attempt["raw_response_sha256"] = _sha256(raw_path)
            _append_attempt(attempts_path, attempt)
            if marker_path is not None:
                marker_path.unlink(missing_ok=True)
            new_attempts.append(attempt)
            completed_ids.add(attempt_id)

    attempts = _load_attempts(attempts_path)
    _write_labels(labels_path, attempts)
    counts = dict(Counter(str(attempt.get("status")) for attempt in attempts))
    result = {
        "planned_attempts": plan.query_budget,
        "recorded_attempts": len(attempts),
        "new_attempts": len(new_attempts),
        "successful_labels": counts.get("success", 0),
        "invalid_responses": counts.get("invalid_response", 0),
        "errors": counts.get("error", 0),
    }
    manifest["status"] = (
        "complete"
        if result["recorded_attempts"] == result["planned_attempts"]
        and not result["invalid_responses"] and not result["errors"]
        else "partial"
    )
    manifest["finished_at"] = _utc_now()
    manifest["result"] = result
    _write_manifest(manifest_path, manifest)
    return result


def _options_from_args(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
    }
    if args.seed is not None:
        options["seed"] = args.seed
    return options


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sources-file", type=Path, required=True)
    parser.add_argument("--state-bank-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="validate source hashes, normalized states, and selection"
    )
    _add_input_arguments(validate_parser)

    label_parser = subparsers.add_parser(
        "label", help="query the teacher for every declared selection attempt"
    )
    _add_input_arguments(label_parser)
    label_parser.add_argument("--output", type=Path, required=True)
    label_parser.add_argument("--run-id", required=True)
    label_parser.add_argument("--teacher-policy", default=DEFAULT_TEACHER_POLICY)
    label_parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    label_parser.add_argument(
        "--ollama-base-url", action="append", default=None,
        help="repeat for fallback endpoints; defaults to http://localhost:11434",
    )
    label_parser.add_argument("--model-digest")
    label_parser.add_argument("--temperature", type=float, default=1.0)
    label_parser.add_argument("--top-p", type=float, default=0.9)
    label_parser.add_argument("--num-ctx", type=int, default=8192)
    label_parser.add_argument("--seed", type=int)
    label_parser.add_argument("--resume", action="store_true")
    label_parser.add_argument(
        "--allow-dirty", action="store_true",
        help="diagnostic only; canonical runs require a clean checkout",
    )
    args = parser.parse_args()

    if args.command == "validate":
        plan, _, summary = validate_selected_states(
            args.sources_file, args.state_bank_root, args.selection,
        )
        print(json.dumps({"selection_id": plan.selection_id, **summary}, indent=2))
        return

    base_urls = args.ollama_base_url or ["http://localhost:11434"]
    identity = resolve_model_identity(base_urls, args.teacher_model, args.model_digest)
    result = label_selected_states(
        sources_file=args.sources_file,
        state_bank_root=args.state_bank_root,
        selection_file=args.selection,
        output=args.output,
        run_id=args.run_id,
        teacher_policy=args.teacher_policy,
        teacher_model=args.teacher_model,
        base_urls=base_urls,
        options=_options_from_args(args),
        model_identity=identity,
        resume=args.resume,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(result, indent=2))
    if result["invalid_responses"] or result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
