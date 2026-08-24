# Offline teacher elicitation

`teacher_elicitor.py` labels a predeclared subset of normalized decision states
with AP4Fed's single-agent teacher. It does not choose states, mutate the state
bank, or assign archived outcomes to newly queried actions.

The implementation is separated by responsibility while retaining the same
user-facing command:

- `teacher_campaign.py` validates selection manifests and resolves them against
  a recreated, hash-checked state bank;
- `teacher_adapter.py` reconstructs the Docker teacher view, builds and parses
  the normal AP4Fed prompt, and applies the selector safety rule;
- `teacher_elicitor.py` owns durable query attempts, provenance, resume
  behavior, model identity, output artifacts, and the command-line interface.

This boundary keeps offline campaign data handling separate from the runtime
teacher interface that the later decide-or-defer pipeline will reuse.

## Inputs

The command requires three separately versioned inputs:

1. the source list used to build the state bank;
2. the root containing its normalized per-configuration outputs; and
3. a selection manifest fixing the record IDs and number of attempts before
   any new teacher labels are observed.

Selection schema version 1 is:

```json
{
  "schema_version": 1,
  "selection_id": "sprint02-teacher-screen-v1",
  "selection_rule": "Fixed workload, source-policy, decision-index, previous-AP, and cold/warm quotas.",
  "state_bank_sha256": "<full-state-bank-fingerprint>",
  "records": [
    {
      "record_id": "agentic-paper-archive/fashionmnist__cnn16k/random/r1::d2",
      "attempts": 1,
      "purpose": "primary"
    }
  ]
}
```

`selection_id` is a stable identifier containing letters, numbers, `.`, `_`,
or `-`. `selection_rule` records the predeclared stratification rule. Each
state appears once. `attempts` defaults to one and must be a positive integer.
Repeated attempts require `purpose=calibration`; all other records use
`purpose=primary`. `state_bank_sha256` is optional for ad hoc diagnostics; a
canonical manifest should include it so validation refuses a different state
bank even when selected record IDs still exist.

Contiguous selections may use `groups` instead of listing every state:

```json
{
  "groups": [
    {
      "source_id": "agentic-paper-archive/ag_news__mlp/random",
      "run_range": {"start": 1, "end": 10},
      "decision_range": {"start": 1, "end": 9},
      "exclude": [{"run": 1, "decision": 1}],
      "attempts": 1,
      "purpose": "primary"
    }
  ]
}
```

Ranges are inclusive. The elicitor deterministically expands each group into
the same normalized `record_id` values used by explicit records and rejects
duplicate or out-of-range exclusions. Explicit `records` and grouped entries
may coexist when a plan needs both sparse and contiguous selections.

Before querying, the elicitor recreates every normalized state from the raw
configuration and run CSVs. It refuses to continue if a selected ID is absent,
the state table differs, or a recorded configuration/run hash no longer
matches its source.

## Teacher-view reconstruction

For a decision at index `d`, the elicitor reconstructs the metric files that
were visible to the live teacher:

- at `d = 1`, the growing cumulative CSV contains round 1 while no completed
  snapshot exists;
- at `d >= 2`, cumulative snapshot files contain rounds `1` through `d - 1`.

The previous Client Selector, Message Compressor, and HDH states come from the
immutable normalized state. The Docker single-agent prompt builder and parser
are then called directly. An optional `metrics_root` input was added to the
mirrored Local and Docker prompt helpers so offline code can point them at the
reconstructed files; omitting it preserves normal AP4Fed behaviour.

The labeller uses the Docker prompt path because substantive Sprint 02
experiments use the Docker backend. Local and Docker prompt files remain
byte-identical.

## Validate before querying

From the AP4Fed repository root:

```sh
python3 distill/teacher_elicitor.py validate \
  --sources-file distill/campaigns/agentic_paper_state_bank_sources.json \
  --state-bank-root <state-bank-root> \
  --selection <selection.json>
```

This performs source/state/hash checks and reports the selected workloads and
fixed query budget without contacting the teacher. It reports both the number
of selected states and the query budget by purpose, so repeated calibration
attempts remain explicit.

## Frozen Sprint 02 query plans

The committed campaign contains two complementary manifests:

| Manifest | Unique states | Query budget | Use |
|---|---:|---:|---|
| `agentic_paper_teacher_screen_v1.json` | 39 | 79 per teacher | Shared Few-Shot/Zero-Shot screen; ten states receive five calibration attempts. |
| `agentic_paper_teacher_full_v1.json` | 1,170 | 1,170 | Thirteen compact configuration groups covering the full bank for the primary teacher, after the screening gate passes. |

The screening selection takes three states from every retained configuration,
covers both workloads, all source policies, all eight previous-pattern
contexts, and cold/mid/late decisions. The full manifest independently covers
all 1,170 archived states once. Screening outputs are not merged into the
training labels; they remain separate evidence for the teacher-selection gate.

Run the screening manifest once with each candidate teacher. Do not run the
full manifest until the teacher-screening gate has been reviewed. Retain the
screening run manifest, aggregate screening result, and raw-response hashes in
the external results archive even though its labels are excluded from the full
training dataset.

## Run labels

Canonical runs require a clean AP4Fed checkout and a new output directory:

```sh
python3 distill/teacher_elicitor.py label \
  --sources-file distill/campaigns/agentic_paper_state_bank_sources.json \
  --state-bank-root <state-bank-root> \
  --selection <selection.json> \
  --output <new-immutable-run-directory> \
  --run-id <stable-run-id> \
  --teacher-policy "Single AI-Agent (Few-Shot)" \
  --teacher-model deepseek-r1:8b \
  --ollama-base-url http://localhost:11434
```

The default sampling options match live DeepSeek use: temperature `1.0`,
top-p `0.9`, and context length `8192`. A seed is absent by default because the
runtime policy does not set one. The model digest is discovered through
Ollama's local `/api/tags` endpoint; `--model-digest` is available when an
equivalent serving environment cannot expose that endpoint.

Use `--resume` only with the same run directory. Resume requires the original
selection, state-bank fingerprint, code revision, prompt files, complete model
identity, endpoints, and sampling options. Every recorded attempt is skipped.
A resume also rechecks every retained raw-response hash before deriving labels.
A durable in-flight marker is written immediately before each model call; if a
process interruption leaves the outcome uncertain, resume refuses to repeat
that call automatically. Failed attempts are retained rather than retried.

## Run artifacts

Each run directory contains:

| Artifact | Purpose |
|---|---|
| `run_manifest.json` | Producing code state, exact command, source/selection hashes, teacher identity, prompt hashes, environment, timestamps, and result counts. |
| `attempts.jsonl` | Append-only record of every successful, invalid, or failed query. |
| `labels.csv` | Deterministic normalized view of successful attempts only. |
| `raw_responses/` | Exact teacher responses, linked by `attempt_id` and SHA-256. |
| `inflight/` | Normally empty; a retained marker prevents automatic repetition of a query interrupted at an uncertain point. |

Successful labels use `label_kind=offline_teacher_query`. The live selector
safety rule is applied after AP4Fed parsing: invalid thresholds are replaced
with the smallest safe threshold, or Client Selector is turned off when no
safe threshold exists. The attempt record retains both parsed and applied
decisions and states whether the guardrail changed the result.

Raw responses and machine-specific run manifests belong in the external
results archive. Only a small curated evidence bundle should later be added to
Git after review.

## Interpretation constraint

Every offline query is counterfactual with respect to its archived trajectory.
The label is valid imitation evidence for the declared teacher, but the
trajectory's later F1, timing, or communication outcome cannot be attributed
to that queried action. Only fresh closed-loop runs establish action effects.
