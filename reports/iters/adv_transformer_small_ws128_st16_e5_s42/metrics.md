# Advanced Model Metrics

- generated_at_utc: `2026-02-19T16:00:07.120953+00:00`
- run_name: `adv_transformer_small_ws128_st16_e5_s42`
- model_name: `transformer_small`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.986955 |
| roc_auc | 0.977404 |
| pr_auc | 0.879365 |
| tpr | 0.592105 |
| fpr | 0.000421 |
| ece_15_bins | 0.006398 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.959538 |
| shot_tpr | 0.842105 |
| shot_fpr | 0.007407 |
| lead_time_ms_median | 0.000 |

## Threshold

- objective: `shot_fpr_constrained`
- max_shot_fpr: `0.0200`
- theta: `0.826120`
- sustain_ms: `3.000`

