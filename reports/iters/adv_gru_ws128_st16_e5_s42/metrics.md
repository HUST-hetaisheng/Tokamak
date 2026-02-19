# Advanced Model Metrics

- generated_at_utc: `2026-02-19T16:02:34.301109+00:00`
- run_name: `adv_gru_ws128_st16_e5_s42`
- model_name: `gru`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.984509 |
| roc_auc | 0.986405 |
| pr_auc | 0.760785 |
| tpr | 0.526316 |
| fpr | 0.000841 |
| ece_15_bins | 0.003577 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.924855 |
| shot_tpr | 0.684211 |
| shot_fpr | 0.007407 |
| lead_time_ms_median | 0.000 |

## Threshold

- objective: `shot_fpr_constrained`
- max_shot_fpr: `0.0200`
- theta: `0.858896`
- sustain_ms: `3.000`

