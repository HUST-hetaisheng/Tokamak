# Advanced Model Metrics

- generated_at_utc: `2026-02-20T07:03:09.312801+00:00`
- run_name: `adv_mamba_lite_ws128_st16_e2_s42`
- model_name: `mamba_lite`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.976083 |
| roc_auc | 0.861444 |
| pr_auc | 0.042635 |
| tpr | 0.162162 |
| fpr | 0.016264 |
| ece_15_bins | 0.017936 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.700000 |
| shot_tpr | 0.250000 |
| shot_fpr | 0.187500 |
| lead_time_ms_median | 49.879 |

## Threshold

- objective: `shot_fpr_constrained`
- max_shot_fpr: `0.0200`
- theta: `0.039498`
- sustain_ms: `3.000`

