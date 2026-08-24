# Sprint 02: reusable extraction and broader data coverage

## Goal

Expand the rule-mining evidence beyond the single 90-decision archive used in Sprint 01.
The first step is to make dataset extraction reusable across AP4Fed experiment arms and
configurations.

Data coverage will then be expanded in two ways:

1. Reconstruct realistic states from other archived experiment arms and label selected
   states offline with the chosen teacher.
2. Run AP4Fed with targeted client and workload configurations where the archive
   provides insufficient evidence.

## Reusable extractor

`archive_extractor.py` contains the reusable one-arm extraction contract. It derives the
workload, source policy, configured model, client count, round count, and run inventory
from each experiment's `config.json`, while retaining the 48 inputs from Feature
Specification v1.

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

`campaigns/agentic_paper_state_bank_sources.json` freezes the archive arms to extract
before new labels are observed. It contains 14 compatible experiments: 8
FashionMNIST/CNN-16k arms and 6 AG_NEWS/MLP arms, totalling 140 runs and 1,260 decision
states. The few-shot DeepSeek baseline contributes its 90 direct archived labels; the
other 1,170 states remain unlabelled.

Four FashionMNIST arms are explicitly excluded: debate-based has an incomplete run,
few-shot GPT has an empty run, and expert-driven and never lack their configuration
files. A full dry run of the list reproduced the declared experiment, run, state, and
label counts.

## Next sprint boundary

Sprint 02 ends with a broader, provenance-safe labelled dataset and its coverage report.
Sprint 03 will complete the distillation pipeline, calibrate the decide-or-defer
mechanism on held-out evidence, and evaluate it in fresh closed-loop AP4Fed runs.
