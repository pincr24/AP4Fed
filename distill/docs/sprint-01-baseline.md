# Sprint 01: baseline from paper archive

## Goal

Establish a reproducible rule-mining baseline from one
archived teacher arm:

- Workload: FashionMNIST with CNN 16k clients (from agentic FL paper).
- Teacher: Single AI-Agent (Few-Shot), `deepseek-r1:8b`.
- Archive: ten runs with ten federated-learning rounds each.
- Dataset: 90 state-to-decision rows (nine decisions per run).
- Learner sample: 80 warm-start rows; the ten first decisions use the fixed-OFF
  cold-start policy because no completed metric snapshot is available.

We assume the baseline is still too narrow to generalize across workloads,
teacher strategies, or longer runs.

## Feature->label extraction

The extractor builds a feature->label dataset from `config.json` and the archived
metrics CSVs. It fills the 48 inputs from Feature Specification v1 and labels each
feature state with the next round's archived AP decision.

We preserve AP4Fed's one-round snapshot lag, and correct a
prompt-construction bug: the runtime lower-cased CSV headers but
searched for mixed-case `Val F1`, leaving F1-digest fields blank. The corrected
digest fields were therefore not visible to the archived teacher. 

## First student baseline

The baseline uses the vendored CONFOLD runtime through
`run_confold_baseline.py` and learns separate decision views for CS, MC, and
HDH. It begins with the interpretable 14-feature compact core from Feature
Specification v1 and removes constant columns.

Evaluation is leave-one-run-out: a fold holds out a complete `run_id`.
The exploratory model is fitted on all 80 warm-start rows.
Cold-start row are recorded separately as fixed-policy decisions.

The runner records exploratory (training-set) and held-out (test-set) metrics,
learned rules, confidence, raw condition matches, effective first-match
coverage, and no-fire counts.

The held-out result is negative: no decision head beats the trivial always-OFF
baseline.

| Head | LORO accuracy | LORO coverage | LORO ON precision / recall / F1 | Always-OFF accuracy |
|---|---:|---:|---:|---:|
| CS | 85.0% | 100.0% | 0.00 / 0.00 / — | 93.75% |
| MC | 81.25% | 98.75% | 0.00 / 0.00 / — | 92.5% |
| HDH | 87.5% | 100.0% | 0.222 / 0.40 / 0.286 | 93.75% |

The exploratory rules fit the 80 available rows perfectly, but that training
fit does not transfer to held-out runs. None of the learned rules is suitable
for autonomous activation from this evidence.

## Interpretation limits

The archive is small and imbalanced: CS and HDH each have five ON labels, and
MC has six. For every head, those positive labels occur in only three of the
ten runs. Its value is just preliminary evaluation of the framework.
The recorded client-selector label is binary because the archive does not retain its selected threshold.
Early cold-start decisions are excluded from rule fitting and evaluation,
leaving eight learner-eligible decisions in each of the ten runs.
