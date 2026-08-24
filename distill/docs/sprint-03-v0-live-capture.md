# Sprint 03 v0 live capture and always-defer policy

This milestone adds the first opt-in AP4Fed runtime path for the proposed
policy. Version v0 captures the exact student state but deliberately makes no
student decision: every non-final decision defers to the existing single-agent
teacher.

The implementation is review-ready but has not yet completed a real Local or
Docker FL smoke run. Its status is therefore **partial** rather than complete.

## Opt-in configuration

Existing policies ignore this path. It is selected only by the exact
`adaptation` value `Distilled Policy` and requires:

```json
{
  "adaptation": "Distilled Policy",
  "LLM": "deepseek-r1:8b",
  "distill_policy": {
    "mode": "always_defer",
    "run_id": "sprint03/<configuration>/r1",
    "teacher_policy": "Single AI-Agent (Few-Shot)",
    "teacher_model_digest": "<full-model-sha256>",
    "trace_dir": "performance/distill_decisions"
  }
}
```

`run_id` is the stable identity of one immutable FL trajectory and must change
for a rerun. `trace_dir` must be relative and remain below the AP4Fed backend's
working directory. Shadow and active modes additionally require an exact,
hash-bound qualification artifact and the frozen AG News configuration; see
[sprint-03-qualification-policy.md](sprint-03-qualification-policy.md).

Always-defer requests contain null `rule_set_id` and `rule_set_sha256` fields
because v0 loads no rules. Claiming a rule set that was not evaluated would be
false provenance.

## Decision sequence

At each non-final adaptation decision, the shared Local/Docker path performs:

```text
cumulative metrics + runtime configuration
    -> Feature Specification v1 live state
    -> validated always-defer dispatch
    -> existing single-agent teacher call
    -> existing Client Selector safety handling
    -> immutable validated decision trace
    -> next-round AP4Fed configuration
```

The call still goes through `_decide_single_agent`. Its default behavior is
unchanged; optional audit inputs expose the exact prompt, raw response, latency,
pre-guardrail action, and guardrail result to the v0 trace writer. The configured
`teacher_policy` selects the same zero/few/fine-tuned prompt mode that would be
used when that single-agent policy runs directly.

If the teacher call fails, the existing pre-decision pattern configuration is
retained as the non-learned safe fallback and the trace records the error. A
capture, validation, or trace-write failure is not swallowed: AP4Fed must not
apply a decision that the declared v0 evidence path failed to record.

## Live feature parity

`decision_state.build_live_state` reuses the same Feature Specification v1
implementation as archive extraction. It accepts the cumulative metrics prefix
through decision `d`, uses only snapshots through `d-1` (with the declared
decision-1 digest fallback), and reads the AP vector applied in round `d` as the
previous action.

The runtime passes the configuration actually shown to the teacher, including
the participating client set selected by AP4Fed for the preceding round. A
deterministic prefix test confirms that live construction and archive
reconstruction produce identical normalized feature text. The required real-run
gate remains: rebuild every state from a completed v0 smoke trajectory and
compare all 48 values byte-for-byte.

## Output layout

The configured trace directory is append-only:

```text
performance/distill_decisions/
├── decision-d1.json
├── decision-d2.json
└── teacher_io/
    ├── d1-prompt.txt
    ├── d1-response.txt
    ├── d2-prompt.txt
    └── d2-response.txt
```

Each decision JSON conforms to decision-trace schema v1 and embeds the complete
dispatch request and result. It records the logical run/record identity,
normalized state, teacher/model identity, prompt and response hashes, timing,
teacher candidate action, guarded applied action, and attribution source. Files
are created exclusively; resuming into a directory containing the same decision
refuses to overwrite it.

Prompt and response files are retained only for successful teacher calls. A
failed call records its error and latency in the decision trace.

## Implementation boundary

- `distill/live_policy.py` owns v0 configuration validation, live state capture,
  always-defer dispatch, action normalization, and immutable trace writing.
- `distill/decision_state.py` owns the shared Feature Specification v1 decision
  state builder.
- mirrored `Local/` and `Docker/` adaptation and single-agent files contain the
  smallest opt-in hook and audit exposure.
- `distill/policy_interfaces.py` remains the versioned validation boundary.

The Local and Docker files remain byte-identical. Default, Random,
Expert-Driven, single-agent, and multi-agent policy selection does not enter the
new branch.

## Checks completed

From the AP4Fed repository root:

```sh
python3 -m unittest distill.tests.test_live_policy -v
python3 -m unittest discover -s distill/tests -v
```

The eight focused v0 tests cover explicit configuration, path containment, live
state capture, null rule provenance, immutable successful traces, teacher-error
fallback, Client Selector adjustment, unchanged single-agent action behavior,
exact opt-in prompt/response auditing, and Local/Docker parity. The archive test
suite separately covers live-prefix
versus archived-state equality.

## Remaining gate

Before the active qualification run:

1. run a small opt-in Local configuration with a fake or approved teacher;
2. confirm one trace per non-final decision and unchanged teacher actions;
3. reconstruct every captured state offline and require byte-identical 48-field
   parity;
4. repeat the smoke in Docker, which is the substantive experiment backend;
5. record the producing clean Git SHA and backend identity.

No canonical FL outcome claim follows from the deterministic tests alone.
