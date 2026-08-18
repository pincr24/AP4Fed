# Distillation baseline

This directory contains the reproducible distillation pipeline for AP4Fed.
It converts archived teacher decisions into a versioned situation-to-decision
dataset and evaluates the initial symbolic student baseline without modifying
the core AP4Fed runtime.

The current supported baseline is the paper archive's FashionMNIST / CNN 16k /
Single-Agent Few-Shot DeepSeek experiment. Its scope, assumptions, and
first-sprint findings are recorded in [docs/sprint-01-baseline.md](docs/sprint-01-baseline.md).

The dataset interface is frozen as Feature Specification v1. See
[Feature_Spec.md](Feature_Spec.md) for the feature inventory and
[docs/dataset-specification.md](docs/dataset-specification.md) for provenance, alignment, and
versioning rules.

## Directory map

- `extract_paper_decision_dataset.py` — recreates the teacher-visible state
  from archived AP4Fed logs and writes the canonical dataset.
- `run_confold_baseline.py` — runs the symbolic CONFOLD baseline.
- `confold_adapter.py` — AP4Fed-specific integration code.
- `confold/` — minimal vendored CONFOLD runtime; its licence and pinned
  revision are retained in that directory.
- `docs/` — committed technical documentation for reproduction and review.

## Reproduce the baseline

From the AP4Fed repository root, extract the archive:

```sh
python3 distill/extract_paper_decision_dataset.py \
  --source ../paper_campaigns/fashionmnist__cnn16k/experiments/few-shot\ deepseek
```

Then install CONFOLD's runtime dependency and run the baseline:

```sh
python3 -m pip install -r distill/requirements-confold.txt
python3 distill/run_confold_baseline.py --overwrite
```

The extractor writes `decision_dataset.csv` and `audit.json`; the baseline
writes results below the selected dataset's `confold_baseline/` directory.