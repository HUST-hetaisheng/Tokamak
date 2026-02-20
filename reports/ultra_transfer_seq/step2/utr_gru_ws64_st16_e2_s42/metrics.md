# Advanced Model Metrics

- generated_at_utc: `2026-02-20T11:28:19.530139+00:00`
- run_name: `utr_gru_ws64_st16_e2_s42`
- model_name: `gru`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.991247 |
| roc_auc | 0.907319 |
| pr_auc | 0.483099 |
| tpr | 0.400000 |
| fpr | 0.000739 |
| ece_15_bins | 0.012949 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.966667 |
| shot_tpr | 0.833333 |
| shot_fpr | 0.000000 |
| lead_time_ms_median | 2.011 |

## Threshold

- objective: `shot_fpr_constrained_stable`
- max_shot_fpr: `0.0200`
- theta: `0.687329`
- sustain_ms: `3.000`

