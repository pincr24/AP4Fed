"""Run one live AP4Fed adaptation decision through the distilled policy.

This module is the runtime side of the Sprint 03 decide-or-defer path, which a
run enables through the optional `distill_policy` configuration block. It
validates that block, builds the Feature Specification v1 state from the live
metrics CSV, and dispatches it in the configured mode: `always_defer` is the
control and sends every decision to the teacher agent; `shadow` evaluates the
qualification rules and records their proposal but still defers; `active`
applies the proposal through the existing Client Selector guardrail when every
head decides, and defers the whole decision otherwise. Each decision writes an
immutable trace of the state, the rule set identity, the dispatch outcome, and
the configuration that was applied.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from decision_state import build_live_state
from policy_interfaces import (
    DECISION_TRACE_SCHEMA_VERSION,
    DISPATCH_SCHEMA_VERSION,
    HEADS,
    canonical_sha256,
    validate_decision_trace,
    validate_dispatch_request,
    validate_dispatch_result,
)
from qualification_rules import (
    QUALIFICATION_ARTIFACT_KIND,
    LoadedQualificationArtifact,
    load_qualification_artifact,
    validate_qualification_configuration,
)
from rule_dispatch import dispatch_rules


SUPPORTED_MODES = {"always_defer", "shadow", "active"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*")


@dataclass(frozen=True)
class LivePolicySettings:
    """Validated settings needed for one live policy run."""

    mode: str
    run_id: str
    teacher_policy: str
    teacher_model: str
    teacher_model_digest: str
    trace_dir: Path
    rule_artifact: dict[str, object] | None
    rule_artifact_sha256: str | None


def load_live_policy_settings(
    config: dict[str, object],
    working_directory: Path,
) -> LivePolicySettings:
    """Read and validate the settings for the opt-in live policy.

    Paths are resolved below the supplied working directory so a configuration
    cannot write its evidence elsewhere. Other AP4Fed policies never call this
    function and keep their existing behavior.
    """
    raw = config.get("distill_policy")
    if not isinstance(raw, dict):
        raise ValueError("Distilled Policy requires a distill_policy object")
    common = {
        "mode", "run_id", "teacher_policy", "teacher_model_digest", "trace_dir",
    }
    mode = raw.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "distill_policy.mode must be always_defer, shadow, or active"
        )
    expected = common if mode == "always_defer" else common | {
        "artifact_kind", "rule_artifact", "rule_artifact_sha256",
    }
    if set(raw) != expected:
        raise ValueError(
            "distill_policy fields differ; "
            f"missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}"
        )
    run_id = raw["run_id"]
    if (
        not isinstance(run_id, str)
        or not SAFE_RUN_ID_RE.fullmatch(run_id)
        or "::" in run_id
    ):
        raise ValueError("distill_policy.run_id must be a stable identifier without '::'")
    teacher_policy = raw["teacher_policy"]
    if not isinstance(teacher_policy, str) or "single" not in teacher_policy.lower():
        raise ValueError("distill_policy.teacher_policy must identify a single-agent policy")
    teacher_model = config.get("LLM")
    if not isinstance(teacher_model, str) or not teacher_model:
        raise ValueError("Distilled Policy requires a non-empty LLM model name")
    digest = raw["teacher_model_digest"]
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError("distill_policy.teacher_model_digest must be a full SHA-256")
    trace_value = raw["trace_dir"]
    if not isinstance(trace_value, str) or not trace_value.strip():
        raise ValueError("distill_policy.trace_dir must be a relative path")
    trace_relative = Path(trace_value)
    if trace_relative.is_absolute() or ".." in trace_relative.parts:
        raise ValueError("distill_policy.trace_dir must stay below the working directory")
    trace_dir = (working_directory / trace_relative).resolve()
    if working_directory.resolve() not in trace_dir.parents:
        raise ValueError("distill_policy.trace_dir must stay below the working directory")

    loaded: LoadedQualificationArtifact | None = None
    if mode != "always_defer":
        if raw["artifact_kind"] != QUALIFICATION_ARTIFACT_KIND:
            raise ValueError(
                "Sprint 03 shadow/active mode requires a qualification_only artifact"
            )
        expected_hash = raw["rule_artifact_sha256"]
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise ValueError("distill_policy.rule_artifact_sha256 must be a full SHA-256")
        artifact_value = raw["rule_artifact"]
        if not isinstance(artifact_value, str) or not artifact_value.strip():
            raise ValueError("distill_policy.rule_artifact must be a relative path")
        artifact_relative = Path(artifact_value)
        if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
            raise ValueError("distill_policy.rule_artifact must stay below distill/")
        distill_root = Path(__file__).resolve().parent
        artifact_path = (distill_root / artifact_relative).resolve()
        if distill_root not in artifact_path.parents:
            raise ValueError("distill_policy.rule_artifact must stay below distill/")
        loaded = load_qualification_artifact(artifact_path)
        if loaded.sha256 != expected_hash:
            raise ValueError("distill_policy.rule_artifact_sha256 differs from the file")
        validate_qualification_configuration(config)
    return LivePolicySettings(
        mode=mode,
        run_id=run_id,
        teacher_policy=teacher_policy,
        teacher_model=teacher_model,
        teacher_model_digest=digest,
        trace_dir=trace_dir,
        rule_artifact=loaded.document if loaded is not None else None,
        rule_artifact_sha256=loaded.sha256 if loaded is not None else None,
    )


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: live metrics CSV is empty")
    return rows


def build_always_defer_request(
    settings: LivePolicySettings,
    config: dict[str, object],
    metrics_csv: Path,
    decision_idx: int,
) -> dict[str, object]:
    """Build a validated request from the metrics visible at one decision.

    Always-defer requests deliberately contain no rule-set identity because no
    mined rules have been loaded or evaluated at this milestone.
    """
    if settings.mode != "always_defer":
        raise ValueError("always-defer request requires always_defer settings")
    request = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "mode": "always_defer",
        "record_id": f"{settings.run_id}::d{decision_idx}",
        "run_id": settings.run_id,
        "rule_set_id": None,
        "rule_set_sha256": None,
        "state": build_live_state(config, _read_metrics(metrics_csv), decision_idx),
    }
    validate_dispatch_request(request)
    return request


def build_policy_request(
    settings: LivePolicySettings,
    config: dict[str, object],
    metrics_csv: Path,
    decision_idx: int,
) -> dict[str, object]:
    """Build a request for either the control or qualification policy."""
    if settings.mode == "always_defer":
        return build_always_defer_request(
            settings, config, metrics_csv, decision_idx,
        )
    if settings.rule_artifact is None or settings.rule_artifact_sha256 is None:
        raise RuntimeError("qualification policy settings have no loaded rule artifact")
    request = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "mode": settings.mode,
        "record_id": f"{settings.run_id}::d{decision_idx}",
        "run_id": settings.run_id,
        "rule_set_id": settings.rule_artifact["rule_set_id"],
        "rule_set_sha256": settings.rule_artifact_sha256,
        "state": build_live_state(config, _read_metrics(metrics_csv), decision_idx),
    }
    validate_dispatch_request(request)
    return request


def always_defer_dispatch(request: dict[str, object]) -> dict[str, object]:
    """Return the v0 dispatch result, which sends the whole decision to the teacher."""
    validate_dispatch_request(request)
    if request["mode"] != "always_defer":
        raise ValueError("always-defer dispatch requires an always_defer request")
    started = time.perf_counter_ns()
    evaluations = {
        head: {
            "outcome": "not_evaluated",
            "fired_rule_ids": [],
            "candidate_actions": [],
            "selected_action": None,
            "trigger": "always_defer",
        }
        for head in HEADS
    }
    elapsed_us = max(0, (time.perf_counter_ns() - started) // 1000)
    result = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "mode": "always_defer",
        "record_id": request["record_id"],
        "rule_set_id": None,
        "head_evaluations": evaluations,
        "requires_teacher": True,
        "deferral_trigger": "always_defer",
        "proposed_action": None,
        "dispatch_time_us": elapsed_us,
    }
    validate_dispatch_result(result)
    return result


def dispatch_policy(
    settings: LivePolicySettings,
    request: dict[str, object],
) -> dict[str, object]:
    """Dispatch through the selected, already-validated policy mode."""
    if request.get("mode") != settings.mode:
        raise ValueError("dispatch request mode differs from loaded settings")
    if settings.mode == "always_defer":
        return always_defer_dispatch(request)
    if settings.rule_artifact is None or settings.rule_artifact_sha256 is None:
        raise RuntimeError("qualification policy settings have no loaded rule artifact")
    if request.get("rule_set_sha256") != settings.rule_artifact_sha256:
        raise ValueError("dispatch request hash differs from loaded rule artifact")
    return dispatch_rules(request, settings.rule_artifact)


def action_from_config(config: dict[str, object]) -> dict[str, object]:
    """Express the three adaptation-pattern settings as a traceable joint action."""
    patterns = config.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError("runtime configuration has no patterns object")
    action: dict[str, object] = {}
    for head in HEADS:
        value = patterns.get(head)
        if not isinstance(value, dict):
            raise ValueError(f"runtime configuration has no {head} pattern")
        enabled = "ON" if bool(value.get("enabled")) else "OFF"
        head_action: dict[str, object] = {"enabled": enabled}
        if head == "client_selector" and enabled == "ON":
            params = value.get("params")
            selection_value = params.get("selection_value") if isinstance(params, dict) else None
            if isinstance(selection_value, bool):
                raise ValueError("Client Selector selection_value must be an integer")
            try:
                head_action["selection_value"] = int(selection_value)
            except (TypeError, ValueError) as error:
                raise ValueError("Client Selector ON requires selection_value") from error
        action[head] = head_action
    return action


def apply_rule_action(
    base_config: dict[str, object],
    proposal: dict[str, object],
    fix_selection_value,
) -> tuple[dict[str, object], dict[str, object]]:
    """Apply one joint rule action through the existing selector guardrail."""
    next_config = copy.deepcopy(base_config)
    before = copy.deepcopy(proposal)
    guardrail_result = "unchanged"
    patterns = next_config.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError("runtime configuration has no patterns object")
    for head in HEADS:
        action = proposal.get(head)
        pattern = patterns.get(head)
        if not isinstance(action, dict) or not isinstance(pattern, dict):
            raise ValueError(f"rule proposal or runtime pattern is missing {head}")
        pattern["enabled"] = action.get("enabled") == "ON"

    client_selector = proposal["client_selector"]
    if client_selector["enabled"] == "ON":
        requested = client_selector["selection_value"]
        selected = fix_selection_value(requested, base_config)
        if selected is None:
            patterns["client_selector"]["enabled"] = False
            guardrail_result = "client_selector_disabled"
        else:
            if selected != requested:
                guardrail_result = "selection_value_adjusted"
            patterns["client_selector"]["params"] = {
                "selection_strategy": "Resource-Based",
                "selection_criteria": "CPU",
                "selection_value": selected,
            }
    audit = {
        "action_before_guardrail": before,
        "applied_action": action_from_config(next_config),
        "guardrail_result": guardrail_result,
    }
    return next_config, audit


def _write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def write_policy_trace(
    settings: LivePolicySettings,
    request: dict[str, object],
    dispatch: dict[str, object],
    teacher_audit: dict[str, object] | None,
    rule_application: dict[str, object] | None,
    fallback_config: dict[str, object],
    controller_time_us: int,
) -> Path:
    """Write the evidence for one decision without replacing existing files.

    A successful teacher call retains the exact prompt and response beside the
    validated JSON trace. If the call failed, the trace instead records the
    unchanged configuration as the safe fallback action.
    """
    if request.get("mode") != settings.mode or request.get("run_id") != settings.run_id:
        raise ValueError("trace request differs from loaded policy settings")
    if (
        settings.mode != "always_defer"
        and request.get("rule_set_sha256") != settings.rule_artifact_sha256
    ):
        raise ValueError("trace request hash differs from loaded rule artifact")
    decision_idx = request["state"]["round_idx"]
    trace_id = f"{settings.run_id.replace('/', '-')}-d{decision_idx}"
    teacher_status = teacher_audit.get("status") if teacher_audit is not None else None
    teacher_io: tuple[Path, str, Path, str] | None = None
    if not dispatch["requires_teacher"]:
        if teacher_audit is not None or rule_application is None:
            raise ValueError("rule decisions require only a rule application audit")
        teacher_resolution = {"status": "not_queried"}
        application = {
            "source": "rules",
            "action_before_guardrail": rule_application["action_before_guardrail"],
            "applied_action": rule_application["applied_action"],
            "guardrail_result": rule_application["guardrail_result"],
        }
    elif teacher_status == "success":
        prompt = teacher_audit.get("prompt")
        response = teacher_audit.get("raw_response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("successful teacher audit must retain prompt and response text")
        prompt_relative = Path("teacher_io") / f"d{decision_idx}-prompt.txt"
        response_relative = Path("teacher_io") / f"d{decision_idx}-response.txt"
        teacher_io = (prompt_relative, prompt, response_relative, response)
        teacher_resolution = {
            "status": "success",
            "policy": settings.teacher_policy,
            "model": settings.teacher_model,
            "model_digest": settings.teacher_model_digest,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "latency_ms": float(teacher_audit["latency_ms"]),
            "action": teacher_audit["action_before_guardrail"],
        }
        application = {
            "source": "teacher",
            "action_before_guardrail": teacher_audit["action_before_guardrail"],
            "applied_action": teacher_audit["applied_action"],
            "guardrail_result": teacher_audit["guardrail_result"],
        }
    elif teacher_status == "error":
        fallback_action = action_from_config(fallback_config)
        teacher_resolution = {
            "status": "error",
            "error_type": str(teacher_audit.get("error_type") or "TeacherError"),
            "error_message": str(teacher_audit.get("error_message") or "teacher call failed"),
            "latency_ms": float(teacher_audit.get("latency_ms") or 0.0),
        }
        application = {
            "source": "safe_fallback",
            "action_before_guardrail": fallback_action,
            "applied_action": fallback_action,
            "guardrail_result": "safe_fallback",
        }
    else:
        raise ValueError("a deferred decision needs a completed teacher audit")

    trace = {
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "trace_id": trace_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "request_sha256": canonical_sha256(request),
        "dispatch": dispatch,
        "teacher_resolution": teacher_resolution,
        "application": application,
        "controller_time_us": max(controller_time_us, dispatch["dispatch_time_us"]),
    }
    validate_decision_trace(trace)
    trace_path = settings.trace_dir / f"decision-d{decision_idx}.json"
    targets = [trace_path]
    if teacher_io is not None:
        targets.extend((
            settings.trace_dir / teacher_io[0],
            settings.trace_dir / teacher_io[2],
        ))
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite decision evidence: "
            + ", ".join(str(path) for path in existing)
        )
    if teacher_io is not None:
        prompt_relative, prompt, response_relative, response = teacher_io
        _write_exclusive(settings.trace_dir / prompt_relative, prompt)
        _write_exclusive(settings.trace_dir / response_relative, response)
    _write_exclusive(
        trace_path,
        json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return trace_path


def write_always_defer_trace(
    settings: LivePolicySettings,
    request: dict[str, object],
    dispatch: dict[str, object],
    teacher_audit: dict[str, object],
    fallback_config: dict[str, object],
    controller_time_us: int,
) -> Path:
    """Compatibility wrapper for the Sprint 03 always-defer milestone."""
    return write_policy_trace(
        settings,
        request,
        dispatch,
        teacher_audit,
        None,
        fallback_config,
        controller_time_us,
    )
