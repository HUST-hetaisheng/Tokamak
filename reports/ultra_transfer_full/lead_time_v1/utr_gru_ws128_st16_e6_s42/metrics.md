# Advanced Model Metrics

- generated_at_utc: `2026-02-20T13:07:15.339092+00:00`
- run_name: `utr_gru_ws128_st16_e6_s42`
- model_name: `gru`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.993898 |
| roc_auc | 0.985527 |
| pr_auc | 0.866340 |
| tpr | 0.712079 |
| fpr | 0.000398 |
| ece_15_bins | 0.002414 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.976879 |
| shot_tpr | 0.894737 |
| shot_fpr | 0.000000 |
| lead_time_ms_median | 6.584 |

## Threshold

- objective: `shot_fpr_constrained_stable`
- max_shot_fpr: `0.0500`
- theta: `0.531137`
- sustain_ms: `3.000`

