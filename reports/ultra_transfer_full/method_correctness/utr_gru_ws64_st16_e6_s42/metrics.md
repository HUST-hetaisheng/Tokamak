# Advanced Model Metrics

- generated_at_utc: `2026-02-20T11:48:15.408365+00:00`
- run_name: `utr_gru_ws64_st16_e6_s42`
- model_name: `gru`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.990412 |
| roc_auc | 0.925367 |
| pr_auc | 0.744433 |
| tpr | 0.452261 |
| fpr | 0.000176 |
| ece_15_bins | 0.003691 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.965318 |
| shot_tpr | 0.868421 |
| shot_fpr | 0.007407 |
| lead_time_ms_median | 4.059 |

## Threshold

- objective: `shot_fpr_constrained_stable`
- max_shot_fpr: `0.0200`
- theta: `0.978548`
- sustain_ms: `3.000`

