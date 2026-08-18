# Feature specification (v1)

Full feature specification of the distillation dataset.  It describes one
state-to-decision row: features available when action decision `d` is made,
labelled with the AP settings applied in round `d + 1`.

## Inputs

The frozen extraction schema contains 48 inputs, in this order:

- **Configuration (8):** `n_clients`, `max_cpu`, `second_highest_cpu`,
  `cpu_spread`, `frac_non_iid`, `frac_new_data`, `has_delay_clients`,
  `workload`.
- **Action history (6):** `prev_cs`, `prev_mc`, `prev_hdh`, `rs_cs_on`,
  `rs_mc_on`, `rs_hdh_on`.
- **Decision clock (2):** `round_idx`, `rounds_remaining`.
- **Lagged snapshot and ratios (11):** `val_f1`, `total_round_time`,
  `train_mean`, `train_min`, `train_max`, `comm_mean`, `comm_min`,
  `comm_max`, `jsd`, `f1_over_time`, `comm_frac`.
- **Snapshot deltas (5):** `d_f1`, `d_total_time`, `d_train_mean`,
  `d_comm_mean`, `d_jsd`.
- **History digest (16):** for each of `f1`, `traintime`, `commtime`, and
  `totaltime`: `_mean`, `_last3`, `_last5`, and `_slope`.

The snapshot is lagged, representing the latest metrics file
available to the teacher at decision time.  Fields without sufficient history
are blank in the canonical CSV. Decisions without a completed snapshot are
**cold starts**: the first CONFOLD baseline applies the fixed all-OFF action
and learns rules only from warm-start decisions, without adding per-feature
missing-value flags to the rule schema.

`prev_*` fields reproduce the teacher's previous-action input.  `rs_*_on`
fields are student-side action-history extensions and must be treated as an
ablation, not strict teacher-prompt parity.

## Labels

The three binary labels are `y_cs_applied`, `y_mc_applied`, and
`y_hdh_applied`.  Client selection is binary in the current paper archive;
future logs may additionally retain its selected threshold.

## First baseline

The initial CONFOLD model uses the compact 14-field core:

`prev_cs`, `prev_mc`, `prev_hdh`, `round_idx`, `rounds_remaining`, `val_f1`,
`total_round_time`, `train_mean`, `train_max`, `comm_mean`, `comm_max`, `jsd`,
`f1_over_time`, `comm_frac`.

Drop constant fields in the selected training slice.  Keep this schema stable;
any added or changed feature requires a new specification version and a newly
extracted dataset.

## Implementation

`extract_paper_decision_dataset.py` is the implementation of this version.
It recreates the prompt's structured aggregations and history
digest rather than parsing prompt text.
