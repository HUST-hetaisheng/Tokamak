# Advanced Model Metrics

- generated_at_utc: `2026-02-19T16:01:52.799145+00:00`
- run_name: `adv_mamba_lite_ws128_st16_e5_s42`
- model_name: `mamba_lite`

## Test Timepoint (Calibrated)

| metric | value |
|---|---:|
| accuracy | 0.988993 |
| roc_auc | 0.990111 |
| pr_auc | 0.913488 |
| tpr | 0.644737 |
| fpr | 0.000000 |
| ece_15_bins | 0.003919 |

## Test Shot Policy

| metric | value |
|---|---:|
| shot_accuracy | 0.976879 |
| shot_tpr | 0.894737 |
| shot_fpr | 0.000000 |
| lead_time_ms_median | 0.000 |

## Threshold

- objective: `shot_fpr_constrained`
- max_shot_fpr: `0.0200`
- theta: `0.767124`
- sustain_ms: `3.000`

