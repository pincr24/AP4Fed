# Sprint 03 runtime interfaces

This document freezes the first version of the data exchanged between rule
validation, decide-or-defer dispatch, and live AP4Fed decision logging. It does
not activate a student policy or claim that any current exploratory rule is
safe to deploy.

The implementation is `distill/policy_interfaces.py`. Its dependency-free
validators reject missing and unknown fields, invalid cross-references, unsafe
Client Selector actions, rules below their declared evidence gates, and traces
that blur rule, teacher, or fallback attribution.

## Why miner output is not a runtime artifact

The Sprint 01 `*_exploratory_rules.json` files are reports from an all-warm-row
training fit. They contain CONFOLD-specific nested exceptions and the Sprint 01
evaluation found that none of those rules is suitable for autonomous
activation. Loading those files directly would turn an explicitly negative
held-out result into a policy.

A validated runtime artifact is therefore a separate normalized file. The
Sprint 03 qualification uses a second, explicitly `qualification_only` artifact
whose training evidence cannot satisfy or bypass this interface. Its pinned
mechanical translation and limited use are documented in
[sprint-03-qualification-policy.md](sprint-03-qualification-policy.md).

## Rule artifact schema v1

The top-level object records:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. |
| `rule_set_id` | Stable logical identifier for the complete validated rule set. |
| `feature_schema` | Exactly `ap4fed-feature-spec-v1`. |
| `label_schema` | Exactly `ap4fed-normalized-labels-v5`. |
| `created_from` | Dataset and grouped-fold hashes, miner name/version, and producing Git SHA. |
| `teacher` | Teacher policy, model, and full model digest copied from the labelled evidence. |
| `heads` | Exact entries for Client Selector, Message Compressor, and HDH. |

Each head declares `minimum_support`, `minimum_wilson_lower_bound`, and a list
of independently validated rules. An empty list is valid and means that the
head has no autonomous coverage; dispatch must defer it.

Each rule contains:

- a globally unique `rule_id`;
- an action, including `selection_value` whenever Client Selector is ON;
- one non-empty conjunction of Feature Specification v1 conditions;
- zero or more exception conjunctions;
- held-out `coverage`, correct `support`, `precision`, and Wilson lower bound;
- the source-rule identifier and the hash of the shared validation split.

`precision` must equal `support / coverage`. The loader rejects a rule below
its head's minimum support or Wilson threshold. Ordered comparisons are numeric;
workload, delay presence, and previous-action conditions use equality or
inequality. Source policy and provenance fields cannot appear as learner
conditions because they are absent from Feature Specification v1.

The runtime representation uses independent unordered rules. Miner-specific
ordering or nested exception structures must be resolved by the producer before
validation. One exception entry is a conjunction that prevents its parent rule
from firing.

## Dispatch request and result schema v1

A dispatch request fixes:

- mode: `always_defer`, `shadow`, or `active`;
- normalized `record_id` and `run_id`;
- logical `rule_set_id` and exact `rule_set_sha256` of its file in shadow or
  active mode; both are null in always-defer mode because no rules are loaded;
- all 48 Feature Specification v1 values in canonical CSV text form.

The validator requires the exact feature set. Blank text is normalized
missingness; other numeric fields must contain finite numeric text. The
`record_id`, `run_id`, state `round_idx`, and `::d<N>` suffix must agree.

The dispatch result retains one evaluation per head:

| Field | Meaning |
|---|---|
| `outcome` | `decide`, `defer`, or `not_evaluated` in always-defer mode. |
| `fired_rule_ids` | Every validated rule that fired. |
| `candidate_actions` | Distinct actions supported by the firing rules. |
| `selected_action` | The head action only when the evidence is unambiguous. |
| `trigger` | `always_defer`, `no_rule`, `conflict`, `insufficient_evidence`, or null. |

The complete result separately records `proposed_action`, `requires_teacher`,
the whole-decision deferral trigger, and dispatch time. A joint proposal exists
only when all three heads decide and must equal their selected actions.

Mode semantics are fixed:

- `always_defer` does not evaluate rules and always requires the teacher;
- `shadow` evaluates rules and may retain a full proposal, but always requires
  the teacher and never applies that proposal;
- `active` uses the full proposal only when every head decides, otherwise it
  defers the complete decision to the teacher.

## Live decision trace schema v1

One trace embeds the exact request and dispatch result. It also records the
request's canonical hash, a UTC timestamp, teacher resolution, action
application, and total controller time.

Teacher resolution has three explicit states:

- `not_queried` — valid only when active dispatch does not require the teacher;
- `success` — records policy/model identity, prompt and response hashes,
  latency, and the teacher action;
- `error` — records the error and latency, after which only the declared safe
  fallback may supply the applied action.

Application keeps `action_before_guardrail` separate from `applied_action` and
records whether the selector threshold was adjusted or Client Selector was
disabled. Cross-field checks enforce the attribution chain:

```text
rule proposal -> rules source
teacher action -> teacher source
teacher error  -> safe fallback source
```

This trace deliberately excludes FL outcomes. Outcomes are linked later by
`record_id` and round alignment so the controller decision is not rewritten
after observing its consequence.

## Focused check

From the AP4Fed repository root:

```sh
python3 -m unittest distill.tests.test_policy_interfaces -v
```

The seven deterministic tests cover exact rule-file hashing, evidence gates,
Client Selector thresholds, Feature Specification v1 field restrictions,
state identity, active versus shadow separation, rule and teacher attribution,
and refusal of a missing teacher resolution.

## Integration status

The always-defer, shadow, and qualification-active paths now use these
interfaces. Their deterministic checks pass. A real AP4Fed Docker run and full
online-versus-offline state parity remain mandatory before recording the Sprint
03 engineering qualification.
