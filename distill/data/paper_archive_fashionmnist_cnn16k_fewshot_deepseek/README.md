# Archived teacher-label baseline

`decision_dataset.csv` contains 90 situation-to-decision pairs: 10
paper-archive repetitions times 9 agent decisions. Every row has the frozen
48-feature situation vector, then the direct ``y_*_applied`` labels produced
by `Single AI-Agent (Few-Shot)` using `deepseek-r1:8b` on FashionMNIST / CNN
16k. Each label is the AP state applied in the following round.

For decision `d`, the situation uses only metrics visible to the teacher: its
last snapshot is through round `d-1` and its history digest is through
`max(1, d-1)`. The archive's AP List vector at round `d+1` is the action label;
the `prev_*` columns are the vector at round `d`. The labels are binary for
client selector (CS), message compressor (MC), and heterogeneous-data handler
(HDH). The archive does not retain the client-selector threshold, so CS remains
binary for this baseline.

This is a baseline for the initial CONFOLD run, not a broad training set.

## Corrected F1-history extraction

The prompt (as generated in AP4Fed) contains a case-sensitivity bug in its F1 lookup:
it lower-cases CSV headers but searches for `"Val F1"`.  The canonical dataset
uses the intended case-insensitive lookup, so its four F1-history fields
(`f1_mean`, `f1_last3`, `f1_last5`, and `f1_slope`) are populated from the
archived rows.  This affects features only, not the archived action labels;
the dataset is therefore a corrected, not exact, reconstruction.

`audit.json` records the source hashes, the 48 input columns, and class
balance.

## Derived CONFOLD baseline

`confold_baseline/` contains reproducible outputs derived from this exact
`decision_dataset.csv`. Regenerate it from the AP4Fed repository root with:

```sh
python3 distill/run_confold_baseline.py --overwrite
```

[report.txt](confold_baseline/report.txt) is the human-readable execution
summary and [results.json](confold_baseline/results.json) is the structured
record. `training_views/` contains the learner inputs for CS, MC, and HDH;
`rules/` contains their exploratory rules. Decisions without a completed
metric snapshot use the fixed-OFF cold-start policy; only the 80 warm-start
rows are fitted and evaluated with leave-one-run-out validation.
