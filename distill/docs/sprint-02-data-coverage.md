# Sprint 02: reusable extraction and broader data coverage

## Goal

Expand the rule-mining evidence beyond the single 90-decision archive used in Sprint 01.
The first step is to make dataset extraction reusable across AP4Fed archived experiment
configurations.

Data coverage will then be expanded in two ways:

1. Reconstruct realistic states from other archived experiment configurations
   and label selected states offline with the chosen teacher.
2. Run AP4Fed with targeted client and workload configurations where the archive
   provides insufficient evidence.

## Reusable extractor

`archive_extractor.py` contains the reusable single-configuration extraction interface.
It derives the workload, source policy, configured model, client count, round count,
and run inventory from each experiment's `config.json`, while retaining the 48 inputs
from Feature Specification v1.

`build_state_bank.py` is the user-facing orchestrator. It accepts either one explicit
experiment or a schema-versioned JSON source list supplied by the campaign. No
experiment names, model defaults, client counts, round counts, or output locations are
embedded in the reusable modules.

The extractor accepts only exact `r<N>.csv` run files.

The frozen Sprint 01 command remains available through
`extract_paper_decision_dataset.py`. That script is kept as the independent Sprint 01
compatibility artifact and is not imported by the reusable path.

## Provenance rule

States from Random, Expert-Driven, zero-shot, or multi-agent trajectories are useful
because they represent realistic FL conditions. Their archived actions are not labels
from the chosen few-shot DeepSeek teacher.

The generic extractor therefore defaults to `states-only` mode:

- `decision_states.csv` contains only stable IDs and the 48 features;
- source policy, model, hashes, and factual actions remain in `audit.json`;
- `labels.csv` has no rows until the chosen teacher is queried.

An explicit `source-behavior` mode is available only when the archived source policy
itself is the declared teacher. It writes a separate label row rather than adding
provenance columns to the state table. This separation prevents actions from different
policies from being accidentally pooled as teacher labels.

## Prepared archive campaign

`campaigns/agentic_paper_state_bank_sources.json` freezes the archived experiment
configurations to extract before new labels are observed. It contains 13 compatible
experiments: 8 FashionMNIST/CNN-16k configurations and 5 AG_NEWS/MLP configurations,
totalling 130 runs and 1,170 decision states. The few-shot DeepSeek baseline contributes
its 90 direct archived labels; the other 1,080 states remain unlabelled.

Five archived experiment configurations are explicitly excluded: FashionMNIST
debate-based has an incomplete run, FashionMNIST few-shot GPT has an empty run,
FashionMNIST expert-driven and never lack their configuration files, and AG_NEWS
voting-based lacks `r8.csv`. A full extraction of the list reproduced the declared
experiment, run, state, and label counts.

## Offline teacher elicitor

`teacher_elicitor.py` now implements prompt-faithful offline labelling for a
predeclared selection of normalized records. It validates the complete state bank
against the frozen raw sources, reconstructs the decision-time metric-file view, calls
the Docker single-agent prompt/parser path, applies the live Client Selector safety
rule, and writes resumable attempt provenance plus a derived normalized label table.
An in-flight query marker prevents resume from silently repeating a model call whose
completion is uncertain after an abrupt interruption.

The frozen screening manifest contains 39 states, three from every retained
configuration, and covers all eight previous-pattern contexts. Ten balanced
calibration states receive five attempts, for 79 calls per candidate teacher.
The separate full-bank manifest groups the 13 configurations and covers all
1,170 states once. Screening outputs remain gate evidence rather than being
merged into the full training labels.

No teacher labels have been queried yet. The remaining evidence work is to prepare a
clean model-serving environment, run and review the teacher screen, conditionally run
the primary-teacher full bank, and audit coverage. The implementation and run
interface are documented in [teacher-elicitation.md](teacher-elicitation.md).

## Next sprint boundary

Sprint 02 ends with a broader, provenance-safe labelled dataset and its coverage report.
Sprint 03 will complete the distillation pipeline, calibrate the decide-or-defer
mechanism on held-out evidence, and evaluate it in fresh closed-loop AP4Fed runs.
