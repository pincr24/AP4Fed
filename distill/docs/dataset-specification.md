# Dataset specification

This document defines the interface between archived AP4Fed logs, reusable
decision states, teacher labels, and later student-policy experiments.

## Artifact layout

The frozen Sprint 01 dataset remains a joined schema-v4 CSV. Reusable Sprint 02
extraction uses normalized schema v5 with three artifacts:

| Artifact | Purpose |
|---|---|
| `decision_states.csv` | Immutable decision IDs and the 48 Feature Specification v1 inputs. |
| `labels.csv` | Zero or more label attempts joined to states by `record_id`. |
| `audit.json` | Experiment/run provenance, source hashes, factual archived actions, schema, timing, and counts. |

This separation prevents repeated source and teacher metadata from polluting
the learner table. It also allows one state to receive several teacher queries
without duplicating its features.

## Unit of state

One row in `decision_states.csv` represents the state visible when AP4Fed makes
decision `d`. Under the archive timing contract, the resulting action appears
in the AP vector applied in round `d + 1`. A ten-round run therefore provides
nine decision states.

### `decision_states.csv` columns

| Column | Meaning |
|---|---|
| `record_id` | Stable identifier for one decision state, formed from the logical source ID, run, and decision index. It is the join key for labels and future query logs. |
| `run_id` | Stable identifier for the complete FL trajectory. It supports run-grouped fitting and evaluation. |
| Feature Specification v1 columns | The 48 learner inputs defined in [../Feature_Spec.md](../Feature_Spec.md), beginning with `n_clients` and ending with `totaltime_slope`. |

The state table contains no source-policy, teacher, label-status, file-path, or
hash columns.

## Label records

`labels.csv` contains one row per label attempt. A states-only extraction writes
the header but no rows. The pair (`record_id`, `attempt_id`) identifies a label
attempt; later offline querying may therefore append repeated samples for the
same state.

| Column | Meaning | Blank when |
|---|---|---|
| `record_id` | State receiving the label. Must match `decision_states.csv`. | Never. |
| `attempt_id` | Identifier for this label attempt within the state. Direct archived labels use `archived-source-behavior`; offline queries will use stable query-attempt IDs. | Never. |
| `label_kind` | How the label was obtained. Current values are defined below. | Never. |
| `teacher_policy` | Policy that produced this label. For direct archived behavior it is the source `adaptation` value. | Never. |
| `teacher_model` | Model identifier associated with label production. For direct source behavior this is the configured `LLM` value and may be blank or operationally irrelevant for non-LLM policies. Offline queries must record the exact queried model identity. | When no model applies. |
| `y_cs_applied` | Client Selector decision. | Never for a successful label. |
| `selection_value` | Client Selector threshold associated with an ON decision. The old archive cannot recover it. | For CS OFF, unspecified thresholds, and old archived labels. |
| `y_mc_applied` | Message Compressor decision. | Never for a successful label. |
| `y_hdh_applied` | Heterogeneous Data Handler decision. | Never for a successful label. |

### `label_kind` values

| Value | Meaning |
|---|---|
| `direct_archived_behavior` | The archived source policy is explicitly treated as the teacher; its factual CS, MC, and HDH actions are copied into the label record. |

An unlabelled state is represented by having no row in `labels.csv`; it does
not need a sentinel `label_kind`. The offline relabelling stage must introduce
and document a distinct value before writing counterfactual teacher-query
labels. It must not reuse `direct_archived_behavior`.

## Provenance in `audit.json`

Source metadata is recorded once per extraction rather than once per state:

| Field | Meaning |
|---|---|
| `source_id` | Stable logical identifier supplied by the caller. |
| `state_source` | Source policy, configured model, dataset, learner model, and workload derived from `config.json`. The configured model is provenance; it does not prove that every policy invoked it. |
| `config_sha256` | SHA-256 of the exact source configuration. |
| `run_registry` | Mapping from each `run_id` to its portable CSV reference, SHA-256, and factual six-slot AP vector for every decision. |
| `label_mode` | `states-only` or the explicit `source-behavior` opt-in. |
| `artifacts` | Filenames belonging to this normalized extraction. |
| Schema/timing fields | Feature order, label order, decision alignment, snapshot lag, digest lag, and F1 correction. |
| Count fields | Runs, decisions, label rows, source-action balance, and—when present—label balance. |

The factual source action is kept under the relevant run and decision in
`run_registry.factual_source_actions`. It is not a teacher label in
`states-only` mode.

## State-source versus teacher-label provenance

States from Random, Expert-Driven, zero-shot, or multi-agent trajectories are
useful realistic inputs, but their factual actions are not labels from the
chosen teacher. Generic extraction therefore defaults to `states-only` and
writes an empty `labels.csv`.

`source-behavior` is an explicit alternative for a within-policy dataset. It
creates one `direct_archived_behavior` label per state. Offline relabelling will
add separate label attempts while retaining the immutable state and factual
source action.

## Timing and history

The state uses only information available before the action is applied.
Snapshot metrics use logs through round `d - 1`; decision `d = 1` has no
completed snapshot. The history digest uses rows through `max(1, d - 1)`,
matching the target teacher prompt's first-round fallback.

Fields without sufficient history remain blank. The initial CONFOLD baseline
treats states without a completed snapshot as cold starts: it uses a fixed
all-OFF action and excludes them from fitting and rule evaluation.

## Reconstruction and training views

`archive_extractor.py` reconstructs states from `config.json` and exact
`r<N>.csv` run files, ignoring rationale exports and campaign summaries. It
does not contain a named experiment or output destination.

Experiment selection is external: callers pass `ExperimentSource` values
directly or use the JSON-list orchestrator in `build_state_bank.py`.
Later mining code will create an explicit joined training view from
`decision_states.csv` and a declared selection of label attempts; the canonical
state table itself remains unchanged.
