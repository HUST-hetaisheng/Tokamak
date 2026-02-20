# Agent-3 指标摘要

- 生成时间 (UTC)：`2026-02-19T12:55:04.145946+00:00`
- 选定基线：`xgb_dart`
- 准确率>=0.98：`已达到`
- 特征策略：`use_all_required_features`（23/23）

## 时间点级指标（测试集，校准后）

| 指标 | 值 |
|---|---:|
| accuracy（准确率） | 0.992293 |
| roc_auc（ROC 曲线下面积） | 0.985884 |
| pr_auc（精确率-召回率曲线下面积） | 0.754584 |
| tpr（真阳性率） | 0.654182 |
| fpr（假阳性率） | 0.002874 |
| brier（布里尔评分） | 0.006045 |
| ece_15_bins（期望校准误差，15区间） | 0.003922 |

## 炮级策略指标（测试集）

| 指标 | 值 |
|---|---:|
| shot_accuracy（炮级准确率） | 0.936416 |
| shot_tpr（炮级真阳性率） | 0.736842 |
| shot_fpr（炮级假阳性率） | 0.007407 |
| lead_time_ms_median（中位提前量，毫秒） | 8.076 |

## 阈值策略

- 目标：`accuracy`
- theta（阈值）：`0.468863`
- sustain（持续时间）：`5.000 ms`

## 生成的文件计数

| 产物类型 | 数量 |
|---|---:|
| probability_timeline_png（概率时间线图） | 173 |
| report_plot_png_total（报告图总数） | 175 |

## 使用的绘图控制

- plot_all_test_shots（绘制所有测试炮）：`True`
- plot_shot_limit（绘图炮数限制）：`3`
- test_shot_count（测试炮总数）：`173`

## 基线对比（测试集原始概率，threshold=0.5）

| 模型 | accuracy | roc_auc | pr_auc | tpr | fpr |
|---|---:|---:|---:|---:|---:|
| logreg（逻辑回归） | 0.868098 | 0.895885 | 0.236606 | 0.730337 | 0.129933 |
| xgb_gbtree | 0.981436 | 0.982816 | 0.759271 | 0.813983 | 0.016170 |
| xgb_dart | 0.973500 | 0.986158 | 0.768612 | 0.848939 | 0.024719 |

## SHAP 重要特征排序

| 特征 | mean_abs_shap（平均绝对SHAP值） | direction_hint（方向提示） | mechanism_tags（机制标签） |
|---|---:|---|---|
| Mir_avg_fre | 0.146257 | not_available_without_shap | locked_mode（锁模） |
| v_loop | 0.098692 | not_available_without_shap | density_limit（密度极限）,vde_control_loss（VDE控制丧失）,impurity_radiation_collapse（杂质辐射坍塌） |
| mode_number_n | 0.078820 | not_available_without_shap | locked_mode（锁模）,low_q_current_limit（低q电流极限） |
| sxr_var | 0.067145 | not_available_without_shap | density_limit（密度极限）,impurity_radiation_collapse（杂质辐射坍塌） |
| sxr_kurt | 0.065826 | not_available_without_shap | density_limit（密度极限）,impurity_radiation_collapse（杂质辐射坍塌） |
| xuv_ratio | 0.057054 | not_available_without_shap | density_limit（密度极限）,impurity_radiation_collapse（杂质辐射坍塌） |
| Z_proxy | 0.055537 | not_available_without_shap | vde_control_loss（VDE控制丧失） |
| qa_proxy | 0.051082 | not_available_without_shap | low_q_current_limit（低q电流极限） |
| dy_a | 0.039816 | not_available_without_shap | vde_control_loss（VDE控制丧失） |
| MNM | 0.037035 | not_available_without_shap | locked_mode（锁模） |
| sxr_mean | 0.032719 | not_available_without_shap | density_limit（密度极限）,impurity_radiation_collapse（杂质辐射坍塌） |
| dx_a | 0.027488 | not_available_without_shap | vde_control_loss（VDE控制丧失） |
