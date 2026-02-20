# Agent-3 Metrics Summary

- Generated at (UTC): `2026-02-20T06:50:49.857468+00:00`
- Selected baseline: `xgb_dart`
- Accuracy>=0.98: `NOT ACHIEVED`
- Feature policy: `use_all_required_features` (23/23)

## Timepoint Metrics (Test, calibrated)

| Metric | Value |
|---|---:|
| accuracy | 0.961181 |
| roc_auc | 0.792851 |
| pr_auc | 0.043360 |
| tpr | 0.027027 |
| fpr | 0.033147 |
| brier | 0.021894 |
| ece_15_bins | 0.024358 |

## Shot Policy Metrics (Test)

| Metric | Value |
|---|---:|
| shot_accuracy | 0.750000 |
| shot_tpr | 0.000000 |
| shot_fpr | 0.062500 |
| lead_time_ms_median | nan |

## Threshold Policy

- objective: `shot_fpr_constrained`
- max_shot_fpr: `0.0200`
- theta: `0.026388`
- sustain: `5.000 ms`

## Calibration / Threshold Split

| Metric | Value |
|---|---:|
| val_total_shots | 20 |
| val_calibration_shots | 10 |
| val_threshold_shots | 10 |

## Generated File Counts

| Artifact Type | Count |
|---|---:|
| probability_timeline_png | 3 |
| report_plot_png_total | 5 |

## Plotting Controls Used

- plot_all_test_shots: `False`
- plot_shot_limit: `3`
- test_shot_count: `20`

## Disruption Reason Coverage

| Metric | Value |
|---|---:|
| disruptive_shots_test | 4 |
| reason_rows | 4 |
| reason_top_k | 3 |

## Baseline Comparison (Test raw P, threshold=0.5)

| Model | accuracy | roc_auc | pr_auc | tpr | fpr |
|---|---:|---:|---:|---:|---:|
| logreg | 0.822215 | 0.872770 | 0.134961 | 0.675676 | 0.176895 |
| xgb_gbtree | 0.862665 | 0.832079 | 0.031667 | 0.405405 | 0.134559 |
| xgb_dart | 0.841135 | 0.795761 | 0.040906 | 0.162162 | 0.154742 |

## SHAP Top Features

| feature | mean_abs_shap | direction_hint | mechanism_tags |
|---|---:|---|---|
| Mir_avg_amp | 0.387672 | not_available_without_shap | density_limit,locked_mode,low_q_current_limit |
| Mir_VV | 0.177061 | not_available_without_shap | density_limit,locked_mode |
| sxr_skew | 0.121672 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| mode_number_n | 0.105974 | not_available_without_shap | locked_mode,low_q_current_limit |
| dx_a | 0.028792 | not_available_without_shap | vde_control_loss |
| xuv_ratio | 0.025324 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| xuv_var | 0.022919 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| ne_nG | 0.021053 | not_available_without_shap | density_limit |
| n=1 amplitude | 0.017811 | not_available_without_shap | locked_mode |
| xuv_kurt | 0.015226 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| CIII | 0.014757 | not_available_without_shap | density_limit,impurity_radiation_collapse |
| qa_proxy | 0.013344 | not_available_without_shap | low_q_current_limit |
