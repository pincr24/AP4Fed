# AP4Fed Policy Distillation

This directory contains AP4Fed's policy-distillation workflow. Its goal is to learn an
inexpensive, interpretable student policy for architectural-pattern (AP) toggling. The
student acts when it is sufficiently confident and defers the remaining decisions to the
runtime LLM controller.

The workflow converts AP4Fed trajectories into versioned decision states, keeps the
origin of each state separate from the origin of its teacher label, learns candidate
symbolic rules, and checks whether those rules are reliable enough to act. The
implemented components live under `distill/`, leaving the core AP4Fed framework
unchanged.

## Pipeline and implementation boundary

The intended pipeline is:

```text
AP4Fed trajectories
    -> decision-state extraction and provenance audit
    -> selective teacher labelling
    -> symbolic rule learning
    -> held-out validation and safety checks
    -> decide-or-defer runtime policy
    -> fresh closed-loop AP4Fed evaluation
```

The repository currently provides state extraction, campaign assembly, selective
offline teacher labelling, a frozen CONFOLD baseline, and the versioned interfaces for
validated rules, dispatch, and live decision traces. Sprint 02 still owns the broader
labelled state bank and coverage audit. Sprint 03 now includes the opt-in always-defer,
shadow, and qualification-active runtime paths. Deterministic checks pass, but the paired
Docker trajectories and live/offline state-parity gate remain to be run after review.
The implementation milestones are recorded in
[the Sprint 03 runtime-interface report](docs/sprint-03-runtime-interfaces.md) and
[the v0 live-capture report](docs/sprint-03-v0-live-capture.md), with the provisional
rule path in [the qualification-policy report](docs/sprint-03-qualification-policy.md), keeping this overview
focused on the stable workflow.

## Stable data interfaces

Feature Specification v1 defines the 48 inputs available to the policy. Their meanings
and timing remain fixed while the evidence base grows:

- [Feature_Spec.md](Feature_Spec.md) defines the feature inventory;
- [dataset-specification.md](docs/dataset-specification.md) defines record identity,
  alignment, provenance, and versioning;
- `decision_states.csv` contains stable record IDs and policy inputs;
- `labels.csv` contains label attempts separately from the states;
- `audit.json` records source identity, hashes, extraction settings, factual archived
  actions, and label status.

These three files belong to one archived experiment configuration (not to the campaign
as a whole or an individual run). Each `decision_states.csv` combines the states from all
`r<N>.csv` runs found for that experiment. Its `labels.csv` and `audit.json` cover the
same experiment and run inventory.

In source-list mode, every experiment's `output_name` becomes a separate directory below
`--output-root`:

```text
<output-root>/
├── <experiment-a>/
│   ├── decision_states.csv
│   ├── labels.csv
│   └── audit.json
└── <experiment-b>/
    ├── decision_states.csv
    ├── labels.csv
    └── audit.json
```

The extractor does not currently produce a merged campaign-level CSV. The versioned
source list defines the campaign and links its separate experiment outputs.

An archived trajectory can provide a realistic state even when its action did not come
from the declared teacher. The generic extractor therefore defaults to `states-only`. It
records a source action as a direct label only when `--label-mode source-behavior` is
selected explicitly and the source policy is the declared teacher for that dataset.

An offline teacher label for a state produced under another policy is counterfactual. It
can be used to train or evaluate an imitation policy, but the trajectory's archived
outcome cannot be presented as the effect of the newly elicited action.

## Available components

- `decision_state.py` — builds the normalized Feature Specification v1 state shared by archive
  reconstruction and the live policy path.
- `archive_extractor.py` — validates one AP4Fed experiment configuration, reads its archived
  metrics, and writes the resulting states with their provenance.
- `build_state_bank.py` — extracts either one source or a versioned JSON source list; it
  defaults to unlabelled states for offline teacher labelling.
- `teacher_elicitor.py` — validates a frozen state selection, reconstructs the
  teacher-visible history, and writes resumable offline label attempts.
- `policy_interfaces.py` — validates versioned runtime rule artifacts, dispatch
  requests/results, and live decision traces without activating a policy.
- `qualification_rules.py` — mechanically translates the pinned Sprint 01 ordered
  CONFOLD reports into a separately labelled qualification-only artifact.
- `rule_dispatch.py` — evaluates all rules per head and implements deterministic
  whole-decision deferral.
- `live_policy.py` — implements opt-in Feature Specification v1 capture, policy
  dispatch, guarded application, and immutable decision traces.
- `build_sprint01_qualification_rules.py` — produces the provisional artifact from a
  clean reviewed checkout.
- `prepare_sprint03_run.py` and `configs/sprint03_agnews_base.json` — prepare paired
  always-defer and active Docker configurations from the frozen AG News setting.
- `extract_paper_decision_dataset.py` — preserves the independent Sprint 01 baseline
  command and reproduces its frozen output.
- `run_confold_baseline.py` — runs the initial symbolic CONFOLD baseline.
- `confold_adapter.py` — maps AP4Fed decision data to the vendored learner.
- `confold/` — minimal vendored CONFOLD runtime with its licence and pinned revision.
- `campaigns/` — versioned campaign selections, separate from reusable extraction code.
- `docs/` — dataset specification, implementation reports, evidence, and limitations.

## Reproduce the frozen baseline

From the AP4Fed repository root, reconstruct the frozen baseline dataset:

```sh
python3 distill/extract_paper_decision_dataset.py \
  --source ../paper_campaigns/fashionmnist__cnn16k/experiments/few-shot\ deepseek
```

Install CONFOLD's runtime dependency and run the baseline:

```sh
python3 -m pip install -r distill/requirements-confold.txt
python3 distill/run_confold_baseline.py --overwrite
```

The extractor writes `decision_dataset.csv` and `audit.json`. The CONFOLD run writes its
results to the dataset's `confold_baseline/` directory. Its result and limitations are
documented in [sprint-01-baseline.md](docs/sprint-01-baseline.md).

## Extract one archived experiment configuration

Assign the experiment a stable logical source ID rather than using its local filesystem
path as its identity:

```sh
python3 distill/build_state_bank.py \
  --source <archived-experiment-directory> \
  --source-id <archive/workload/policy> \
  --output <output-directory>
```

This command writes `decision_states.csv`, `labels.csv`, and `audit.json`. An empty
label file is the expected default when the states will be sent to the declared teacher
later.

## Extract a source list

Define a campaign in a versioned JSON source list. Relative source paths are resolved from the list
file, and each output name becomes a directory below the selected output root:

```json
{
  "schema_version": 1,
  "experiments": [
    {
      "source": "<relative-or-absolute-experiment-directory>",
      "source_id": "<stable-provenance-id>",
      "output_name": "<safe-directory-name>",
      "label_mode": "states-only"
    }
  ]
}
```

```sh
python3 distill/build_state_bank.py \
  --sources-file <sources.json> \
  --output-root <campaign-output-directory>
```

Higher-level Python code can construct `ExperimentSource` values and call
`extract_experiment()` directly. Neither reusable layer fixes a particular experiment,
model, client count, round count, run count, or output location.

The prepared agentic-paper source inventory is
[`campaigns/agentic_paper_state_bank_sources.json`](campaigns/agentic_paper_state_bank_sources.json).
The frozen query plans are
[`campaigns/agentic_paper_teacher_screen_v1.json`](campaigns/agentic_paper_teacher_screen_v1.json)
and
[`campaigns/agentic_paper_teacher_full_v1.json`](campaigns/agentic_paper_teacher_full_v1.json).
Their inventory, exclusions, dry-run results, and unfinished evidence work are reported in
[the Sprint 02 implementation report](docs/sprint-02-data-coverage.md).

## Label selected states offline

Freeze the selected record IDs and attempt counts before observing new teacher labels.
Then validate the state bank without contacting the model:

```sh
python3 distill/teacher_elicitor.py validate \
  --sources-file distill/campaigns/agentic_paper_state_bank_sources.json \
  --state-bank-root <campaign-output-directory> \
  --selection <selection.json>
```

The separate `label` operation requires a clean producing checkout, records the exact
teacher model digest and prompt hashes, and writes raw responses outside the learner
table. See [teacher-elicitation.md](docs/teacher-elicitation.md) for the selection
schema, run command, resume rules, artifacts, and interpretation constraint.

## Documentation map

- [Sprint 01 baseline](docs/sprint-01-baseline.md) — frozen dataset, initial CONFOLD
  result, and interpretation limits.
- [Sprint 02 data coverage](docs/sprint-02-data-coverage.md) — reusable extraction
  milestone, campaign inventory, checks, and remaining work.
- [Sprint 03 runtime interfaces](docs/sprint-03-runtime-interfaces.md) — validated
  rule, dispatcher, and live-trace schemas plus their attribution constraints.
- [Sprint 03 v0 live capture](docs/sprint-03-v0-live-capture.md) — opt-in
  always-defer configuration, trace layout, checks, and the remaining live gate.
- [Sprint 03 qualification policy](docs/sprint-03-qualification-policy.md) — pinned
  rule translation, exact AG News configuration, shadow/active behavior, and run
  preparation.
- [Offline teacher elicitation](docs/teacher-elicitation.md) — selection input,
  prompt reconstruction, run artifacts, and resume behaviour.
- [Dataset specification](docs/dataset-specification.md) — schemas, alignment,
  provenance, and versioning rules.
