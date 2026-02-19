# Agent-3 Metrics Summary

- Generated at (UTC): `2026-02-19T12:55:04.145946+00:00`
- Selected baseline: `xgb_dart`
- Accuracy>=0.98: `ACHIEVED`
- Feature policy: `use_all_required_features` (23/23)

## Timepoint Metrics (Test, calibrated)

| Metric | Value |
|---|---:|
| accuracy | 0.992293 |
| roc_auc | 0.985884 |
| pr_auc | 0.754584 |
| tpr | 0.654182 |
| fpr | 0.002874 |
| brier | 0.006045 |
| ece_15_bins | 0.003922 |

## Shot Policy Metrics (Test)

| Metric | Value |
|---|---:|
| shot_accuracy | 0.936416 |
| shot_tpr | 0.736842 |
| shot_fpr | 0.007407 |
| lead_time_ms_median | 8.076 |

## Threshold Policy

- objective: `accuracy`
- theta: `0.468863`
- sustain: `5.000 ms`

## Generated File Counts

| Artifact Type | Count |
|---|---:|
| probability_timeline_png | 173 |
| report_plot_png_total | 175 |

## Plotting Controls Used

- plot_all_test_shots: `True`
- plot_shot_limit: `3`
- test_shot_count: `173`

## Baseline Comparison (Test raw P, threshold=0.5)

| Model | accuracy | roc_auc | pr_auc | tpr | fpr |
|---|---:|---:|---:|---:|---:|
| logreg | 0.868098 | 0.895885 | 0.236606 | 0.730337 | 0.129933 |
| xgb_gbtree | 0.981436 | 0.982816 | 0.759271 | 0.813983 | 0.016170 |
| xgb_dart | 0.973500 | 0.986158 | 0.768612 | 0.848939 | 0.024719 |

## SHAP Top Features

| feature | mean_abs_shap | direction_hint | mechanism_tags |
|---|---:|---|---|
| Mir_avg_fre | 0.146257 | not_available_without_shap | locked_mode |
| v_loop | 0.098692 | not_available_without_shap | density_limit,vde_control_loss,impurity_radiation_collapse |
| mode_number_n | 0.078820 | not_available_without_shap | locked_mode,low_q_current_limit |
| sxr_var | 0.067145 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| sxr_kurt | 0.065826 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| xuv_ratio | 0.057054 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| Z_proxy | 0.055537 | not_available_without_shap | vde_control_loss |
| qa_proxy | 0.051082 | not_available_without_shap | low_q_current_limit |
| dy_a | 0.039816 | not_available_without_shap | vde_control_loss |
| MNM | 0.037035 | not_available_without_shap | locked_mode |
| sxr_mean | 0.032719 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| dx_a | 0.027488 | not_available_without_shap | vde_control_loss |
