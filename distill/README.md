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

The repository currently provides state extraction, campaign assembly, and a frozen
CONFOLD baseline. Sprint 02 is expanding this into a broader labelled state bank and
auditing its coverage. Calibrated dispatch and closed-loop evaluation belong to later
sprints. The active milestone and its actual check results are recorded in
[the Sprint 02 implementation report](docs/sprint-02-data-coverage.md), keeping this
overview focused on the stable workflow.

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

These three files belong to one experiment arm (not to the campaign as a whole or an
individual run). Each `decision_states.csv` combines the decision states from all
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

- `archive_extractor.py` — validates one AP4Fed experiment arm, reconstructs Feature
  Specification v1 states, and writes the states with their provenance.
- `build_state_bank.py` — extracts either one source or a versioned JSON source list; it
  defaults to unlabelled states for offline teacher labelling.
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

## Extract one experiment arm

Assign the experiment a stable logical source ID rather than using its local filesystem
path as its identity:

```sh
python3 distill/build_state_bank.py \
  --source <archive-arm-directory> \
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

The prepared agentic-paper selection is
[`campaigns/agentic_paper_state_bank_sources.json`](campaigns/agentic_paper_state_bank_sources.json).
Its inventory, exclusions, dry-run result, and unfinished evidence work are reported in
[the Sprint 02 implementation report](docs/sprint-02-data-coverage.md).

## Documentation map

- [Sprint 01 baseline](docs/sprint-01-baseline.md) — frozen dataset, initial CONFOLD
  result, and interpretation limits.
- [Sprint 02 data coverage](docs/sprint-02-data-coverage.md) — reusable extraction
  milestone, campaign inventory, checks, and remaining work.
- [Dataset specification](docs/dataset-specification.md) — schemas, alignment,
  provenance, and versioning rules.
