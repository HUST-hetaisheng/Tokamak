# Agent-3 Metrics Summary

- Generated at (UTC): `2026-02-19T13:22:50.450343+00:00`
- Selected baseline: `xgb_dart`
- Accuracy>=0.98: `ACHIEVED`
- Feature policy: `use_all_required_features` (23/23)

## Timepoint Metrics (Test, calibrated)

| Metric | Value |
|---|---:|
| accuracy | 0.991008 |
| roc_auc | 0.978493 |
| pr_auc | 0.637393 |
| tpr | 0.548065 |
| fpr | 0.002659 |
| brier | 0.007586 |
| ece_15_bins | 0.003659 |

## Shot Policy Metrics (Test)

| Metric | Value |
|---|---:|
| shot_accuracy | 0.953757 |
| shot_tpr | 0.842105 |
| shot_fpr | 0.014815 |
| lead_time_ms_median | 7.041 |

## Threshold Policy

- objective: `accuracy`
- theta: `0.426254`
- sustain: `3.000 ms`

## Generated File Counts

| Artifact Type | Count |
|---|---:|
| probability_timeline_png | 0 |
| report_plot_png_total | 2 |

## Plotting Controls Used

- plot_all_test_shots: `False`
- plot_shot_limit: `0`
- test_shot_count: `173`

## Baseline Comparison (Test raw P, threshold=0.5)

| Model | accuracy | roc_auc | pr_auc | tpr | fpr |
|---|---:|---:|---:|---:|---:|
| logreg | 0.868098 | 0.895885 | 0.236606 | 0.730337 | 0.129933 |
| xgb_gbtree | 0.964948 | 0.983333 | 0.704003 | 0.841448 | 0.033286 |
| xgb_dart | 0.951997 | 0.979128 | 0.665416 | 0.851436 | 0.046565 |

## SHAP Top Features

| feature | mean_abs_shap | direction_hint | mechanism_tags |
|---|---:|---|---|
| Mir_avg_fre | 0.102619 | not_available_without_shap | locked_mode |
| v_loop | 0.083668 | not_available_without_shap | density_limit,vde_control_loss,impurity_radiation_collapse |
| sxr_var | 0.082842 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| xuv_ratio | 0.072112 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| mode_number_n | 0.070825 | not_available_without_shap | locked_mode,low_q_current_limit |
| sxr_kurt | 0.064200 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| Z_proxy | 0.059160 | not_available_without_shap | vde_control_loss |
| dy_a | 0.048083 | not_available_without_shap | vde_control_loss |
| sxr_mean | 0.047984 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| qa_proxy | 0.047866 | not_available_without_shap | low_q_current_limit |
| MNM | 0.045483 | not_available_without_shap | locked_mode |
| dx_a | 0.036530 | not_available_without_shap | vde_control_loss |
