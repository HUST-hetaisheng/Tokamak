# Agent-3 Metrics Summary

- Generated at (UTC): `2026-02-19T13:13:13.974920+00:00`
- Selected baseline: `xgb_dart`
- Accuracy>=0.98: `ACHIEVED`
- Feature policy: `use_all_required_features` (23/23)

## Timepoint Metrics (Test, calibrated)

| Metric | Value |
|---|---:|
| accuracy | 0.992152 |
| roc_auc | 0.981306 |
| pr_auc | 0.750693 |
| tpr | 0.666667 |
| fpr | 0.003195 |
| brier | 0.006248 |
| ece_15_bins | 0.004469 |

## Shot Policy Metrics (Test)

| Metric | Value |
|---|---:|
| shot_accuracy | 0.936416 |
| shot_tpr | 0.736842 |
| shot_fpr | 0.007407 |
| lead_time_ms_median | 7.040 |

## Threshold Policy

- objective: `accuracy`
- theta: `0.426254`
- sustain: `5.000 ms`

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
| xgb_gbtree | 0.991202 | 0.982055 | 0.786695 | 0.746567 | 0.005301 |
| xgb_dart | 0.988369 | 0.981520 | 0.770534 | 0.782772 | 0.008692 |

## SHAP Top Features

| feature | mean_abs_shap | direction_hint | mechanism_tags |
|---|---:|---|---|
| Mir_avg_fre | 0.154138 | not_available_without_shap | locked_mode |
| v_loop | 0.110055 | not_available_without_shap | density_limit,vde_control_loss,impurity_radiation_collapse |
| mode_number_n | 0.082149 | not_available_without_shap | locked_mode,low_q_current_limit |
| sxr_var | 0.077154 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| sxr_kurt | 0.059907 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| qa_proxy | 0.058282 | not_available_without_shap | low_q_current_limit |
| Z_proxy | 0.050406 | not_available_without_shap | vde_control_loss |
| MNM | 0.047725 | not_available_without_shap | locked_mode |
| xuv_ratio | 0.046054 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| dy_a | 0.032378 | not_available_without_shap | vde_control_loss |
| xuv_var | 0.028253 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| ne_nG | 0.027868 | not_available_without_shap | density_limit |
