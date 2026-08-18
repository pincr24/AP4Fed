# Dataset specification

This document defines the  interface between the archived AP4Fed logs,
the decision dataset, and student-policy experiments.

## Unit of data

One row represents the state visible when AP4Fed makes decision `d`. Its label
is the AP vector logged as applied in round `d + 1`. A ten-round run therefore
provides nine decision rows.

The dataset CSV contains four ordered sections:

1. Record and provenance fields.
2. The 48 input features defined in [../Feature_Spec.md](../Feature_Spec.md).
3. The labels `y_cs_applied`, `y_mc_applied`, and `y_hdh_applied`.

`audit.json` records the schema version, feature order, source IDs and hashes,
decision alignment, row counts, and class counts.

## Timing and history

The state uses only information available before the labelled action is
applied. Snapshot metrics use logs through round `d - 1`; decision `d = 1`
has no completed snapshot. The history digest uses logs through
`max(1, d - 1)`, matching the teacher prompt's first-round fallback.

Fields that lack sufficient history are blank. The initial CONFOLD baseline
treats decisions without a completed snapshot as cold starts: it uses a fixed
all-OFF action and excludes them from fitting and rule evaluation.

## Data reconstruction

The extractor recreates the teacher's structured feature calculations from
`config.json` and the per-run CSV logs; it never uses a previously logged
prompt as input. It also records SHA-256 hashes for the configuration and each
source CSV in `audit.json`.