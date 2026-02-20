# 单炮破裂原因分析报告 (sfpr002)

- 原因行数: `38`
- 测试准确率: `0.990885`
- 测试 ROC-AUC: `0.978437`
- 炮级准确率: `0.953757`
- 炮级 TPR（真阳性率）/ FPR（假阳性率）: `0.842105` / `0.014815`
- 阈值策略/theta: `shot_fpr_constrained` / `0.668796`

## 破裂机制分布

| 机制 | 计数 |
|---|---:|
| density_limit | 18 |
| vde_control_loss | 10 |
| low_q_current_limit | 8 |
| locked_mode | 2 |

## 单炮详细分析

### 炮号 1051510

- 主要机制: `low_q_current_limit`
- 机制评分: `1.517893`
- 是否触发警报: `是`
- 提前量 (ms): `120.486`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `44`

证据:
1. `qa_proxy` (贡献度 `0.779343`), 标签: `low_q_current_limit`
2. `mode_number_n` (贡献度 `0.738550`), 标签: `locked_mode,low_q_current_limit`
3. `v_loop` (贡献度 `0.412575`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`

### 炮号 1051684

- 主要机制: `density_limit`
- 机制评分: `1.268418`
- 是否触发警报: `是`
- 提前量 (ms): `46.675`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `15`

证据:
1. `sxr_kurt` (贡献度 `0.792833`), 标签: `density_limit,impurity_radiation_collapse`
2. `sxr_var` (贡献度 `0.475585`), 标签: `density_limit,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.463835`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1051701

- 主要机制: `vde_control_loss`
- 机制评分: `1.588321`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `5`

证据:
1. `Z_proxy` (贡献度 `1.086058`), 标签: `vde_control_loss`
2. `v_loop` (贡献度 `0.502264`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.428745`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1052062

- 主要机制: `density_limit`
- 机制评分: `1.099757`
- 是否触发警报: `是`
- 提前量 (ms): `1.010`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `16`

证据:
1. `xuv_ratio` (贡献度 `0.657856`), 标签: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (贡献度 `0.555681`), 标签: `vde_control_loss`
3. `v_loop` (贡献度 `0.441902`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`

### 炮号 1052127

- 主要机制: `vde_control_loss`
- 机制评分: `1.105375`
- 是否触发警报: `是`
- 提前量 (ms): `3.048`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `12`

证据:
1. `v_loop` (贡献度 `0.711419`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `Bt` (贡献度 `0.439520`), 标签: `low_q_current_limit`
3. `Z_proxy` (贡献度 `0.393957`), 标签: `vde_control_loss`

### 炮号 1052920

- 主要机制: `vde_control_loss`
- 机制评分: `1.233884`
- 是否触发警报: `是`
- 提前量 (ms): `1.006`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `7`

证据:
1. `Z_proxy` (贡献度 `0.700322`), 标签: `vde_control_loss`
2. `v_loop` (贡献度 `0.533562`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.391761`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1052948

- 主要机制: `low_q_current_limit`
- 机制评分: `0.677389`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `7`

证据:
1. `qa_proxy` (贡献度 `0.677389`), 标签: `low_q_current_limit`
2. `Mir_avg_fre` (贡献度 `0.536732`), 标签: `locked_mode`
3. `dx_a` (贡献度 `0.303911`), 标签: `vde_control_loss`

### 炮号 1052954

- 主要机制: `low_q_current_limit`
- 机制评分: `1.405668`
- 是否触发警报: `是`
- 提前量 (ms): `1.012`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `6`

证据:
1. `mode_number_n` (贡献度 `0.722922`), 标签: `locked_mode,low_q_current_limit`
2. `qa_proxy` (贡献度 `0.682746`), 标签: `low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.476223`), 标签: `locked_mode`

### 炮号 1052960

- 主要机制: `low_q_current_limit`
- 机制评分: `0.665117`
- 是否触发警报: `是`
- 提前量 (ms): `7.030`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `16`

证据:
1. `qa_proxy` (贡献度 `0.665117`), 标签: `low_q_current_limit`
2. `sxr_mean` (贡献度 `0.381901`), 标签: `density_limit,impurity_radiation_collapse`
3. `dx_a` (贡献度 `0.359801`), 标签: `vde_control_loss`

### 炮号 1053058

- 主要机制: `vde_control_loss`
- 机制评分: `1.758013`
- 是否触发警报: `是`
- 提前量 (ms): `1.007`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `6`

证据:
1. `v_loop` (贡献度 `1.016109`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `mode_number_n` (贡献度 `0.859155`), 标签: `locked_mode,low_q_current_limit`
3. `Z_proxy` (贡献度 `0.741903`), 标签: `vde_control_loss`

### 炮号 1053313

- 主要机制: `vde_control_loss`
- 机制评分: `1.200469`
- 是否触发警报: `是`
- 提前量 (ms): `2.007`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `7`

证据:
1. `mode_number_n` (贡献度 `1.176652`), 标签: `locked_mode,low_q_current_limit`
2. `v_loop` (贡献度 `0.631085`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `Z_proxy` (贡献度 `0.569383`), 标签: `vde_control_loss`

### 炮号 1053552

- 主要机制: `low_q_current_limit`
- 机制评分: `1.588849`
- 是否触发警报: `是`
- 提前量 (ms): `4.019`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `10`

证据:
1. `qa_proxy` (贡献度 `0.944837`), 标签: `low_q_current_limit`
2. `mode_number_n` (贡献度 `0.644012`), 标签: `locked_mode,low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.335896`), 标签: `locked_mode`

### 炮号 1053932

- 主要机制: `vde_control_loss`
- 机制评分: `2.142276`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `1`

证据:
1. `v_loop` (贡献度 `1.294714`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `Z_proxy` (贡献度 `0.847562`), 标签: `vde_control_loss`
3. `Bt` (贡献度 `0.745447`), 标签: `low_q_current_limit`

### 炮号 1054153

- 主要机制: `locked_mode`
- 机制评分: `0.979493`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `8`

证据:
1. `mode_number_n` (贡献度 `0.504911`), 标签: `locked_mode,low_q_current_limit`
2. `Mir_avg_fre` (贡献度 `0.474583`), 标签: `locked_mode`
3. `CIII` (贡献度 `0.428903`), 标签: `density_limit,impurity_radiation_collapse`

### 炮号 1054154

- 主要机制: `density_limit`
- 机制评分: `1.024138`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `10`

证据:
1. `CIII` (贡献度 `0.564860`), 标签: `density_limit,impurity_radiation_collapse`
2. `v_loop` (贡献度 `0.459278`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `MNM` (贡献度 `0.258689`), 标签: `locked_mode`

### 炮号 1054161

- 主要机制: `density_limit`
- 机制评分: `0.829674`
- 是否触发警报: `是`
- 提前量 (ms): `-0.000`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `10`

证据:
1. `Mir_avg_fre` (贡献度 `0.462799`), 标签: `locked_mode`
2. `CIII` (贡献度 `0.454228`), 标签: `density_limit,impurity_radiation_collapse`
3. `v_loop` (贡献度 `0.375446`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`

### 炮号 1054188

- 主要机制: `low_q_current_limit`
- 机制评分: `0.982058`
- 是否触发警报: `是`
- 提前量 (ms): `12.052`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `32`

证据:
1. `Bt` (贡献度 `0.759899`), 标签: `low_q_current_limit`
2. `xuv_skew` (贡献度 `0.248054`), 标签: `density_limit,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.222160`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1054337

- 主要机制: `density_limit`
- 机制评分: `1.278655`
- 是否触发警报: `否`
- 提前量 (ms): `N/A`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `12`

证据:
1. `xuv_ratio` (贡献度 `0.893637`), 标签: `density_limit,impurity_radiation_collapse`
2. `sxr_mean` (贡献度 `0.385018`), 标签: `density_limit,impurity_radiation_collapse`
3. `Z_proxy` (贡献度 `0.256352`), 标签: `vde_control_loss`

### 炮号 1054838

- 主要机制: `density_limit`
- 机制评分: `1.990761`
- 是否触发警报: `是`
- 提前量 (ms): `-0.000`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `6`

证据:
1. `sxr_kurt` (贡献度 `1.636415`), 标签: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (贡献度 `0.688372`), 标签: `locked_mode,low_q_current_limit`
3. `CIII` (贡献度 `0.354347`), 标签: `density_limit,impurity_radiation_collapse`

### 炮号 1055242

- 主要机制: `density_limit`
- 机制评分: `1.496075`
- 是否触发警报: `是`
- 提前量 (ms): `14.050`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `17`

证据:
1. `sxr_kurt` (贡献度 `1.496075`), 标签: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (贡献度 `0.511880`), 标签: `low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.507289`), 标签: `locked_mode`

### 炮号 1055244

- 主要机制: `density_limit`
- 机制评分: `2.679164`
- 是否触发警报: `是`
- 提前量 (ms): `20.055`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `23`

证据:
1. `sxr_kurt` (贡献度 `2.154273`), 标签: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (贡献度 `0.632882`), 标签: `vde_control_loss`
3. `v_loop` (贡献度 `0.524890`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`

### 炮号 1055353

- 主要机制: `density_limit`
- 机制评分: `2.250139`
- 是否触发警报: `是`
- 提前量 (ms): `5.014`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `8`

证据:
1. `sxr_kurt` (贡献度 `2.250139`), 标签: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (贡献度 `0.398731`), 标签: `low_q_current_limit`
3. `Z_proxy` (贡献度 `0.337077`), 标签: `vde_control_loss`

### 炮号 1055357

- 主要机制: `density_limit`
- 机制评分: `0.690792`
- 是否触发警报: `是`
- 提前量 (ms): `9.032`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `12`

证据:
1. `sxr_kurt` (贡献度 `0.690792`), 标签: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (贡献度 `0.678523`), 标签: `vde_control_loss`
3. `qa_proxy` (贡献度 `0.650684`), 标签: `low_q_current_limit`

### 炮号 1056187

- 主要机制: `locked_mode`
- 机制评分: `0.689935`
- 是否触发警报: `是`
- 提前量 (ms): `57.283`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `8`

证据:
1. `Mir_avg_fre` (贡献度 `0.689935`), 标签: `locked_mode`
2. `qa_proxy` (贡献度 `0.651779`), 标签: `low_q_current_limit`
3. `v_loop` (贡献度 `0.359056`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`

### 炮号 1056189

- 主要机制: `low_q_current_limit`
- 机制评分: `0.552552`
- 是否触发警报: `是`
- 提前量 (ms): `53.260`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `8`

证据:
1. `qa_proxy` (贡献度 `0.552552`), 标签: `low_q_current_limit`
2. `v_loop` (贡献度 `0.466907`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `MNM` (贡献度 `0.228567`), 标签: `locked_mode`

### 炮号 1056204

- 主要机制: `low_q_current_limit`
- 机制评分: `0.745279`
- 是否触发警报: `是`
- 提前量 (ms): `10.042`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `19`

证据:
1. `qa_proxy` (贡献度 `0.745279`), 标签: `low_q_current_limit`
2. `Z_proxy` (贡献度 `0.653257`), 标签: `vde_control_loss`
3. `Mir_avg_fre` (贡献度 `0.599674`), 标签: `locked_mode`

### 炮号 1056828

- 主要机制: `density_limit`
- 机制评分: `2.320270`
- 是否触发警报: `是`
- 提前量 (ms): `9.084`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `20`

证据:
1. `sxr_kurt` (贡献度 `1.696847`), 标签: `density_limit,impurity_radiation_collapse`
2. `v_loop` (贡献度 `0.327064`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `xuv_ratio` (贡献度 `0.296359`), 标签: `density_limit,impurity_radiation_collapse`

### 炮号 1057200

- 主要机制: `density_limit`
- 机制评分: `1.069599`
- 是否触发警报: `是`
- 提前量 (ms): `3.025`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `24`

证据:
1. `xuv_ratio` (贡献度 `1.069599`), 标签: `density_limit,impurity_radiation_collapse`
2. `Bt` (贡献度 `0.409778`), 标签: `low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.338444`), 标签: `locked_mode`

### 炮号 1057254

- 主要机制: `density_limit`
- 机制评分: `1.007067`
- 是否触发警报: `是`
- 提前量 (ms): `66.197`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `82`

证据:
1. `xuv_ratio` (贡献度 `1.007067`), 标签: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (贡献度 `0.726177`), 标签: `locked_mode,low_q_current_limit`
3. `Z_proxy` (贡献度 `0.674415`), 标签: `vde_control_loss`

### 炮号 1057448

- 主要机制: `vde_control_loss`
- 机制评分: `0.832846`
- 是否触发警报: `是`
- 提前量 (ms): `-0.000`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `165`

证据:
1. `dx_a` (贡献度 `0.832846`), 标签: `vde_control_loss`
2. `sxr_mean` (贡献度 `0.298542`), 标签: `density_limit,impurity_radiation_collapse`
3. `sxr_var` (贡献度 `0.218366`), 标签: `density_limit,impurity_radiation_collapse`

### 炮号 1057472

- 主要机制: `density_limit`
- 机制评分: `1.802226`
- 是否触发警报: `是`
- 提前量 (ms): `7.136`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `13`

证据:
1. `sxr_kurt` (贡献度 `1.479521`), 标签: `density_limit,impurity_radiation_collapse`
2. `v_loop` (贡献度 `0.322705`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `Z_proxy` (贡献度 `0.260236`), 标签: `vde_control_loss`

### 炮号 1057504

- 主要机制: `density_limit`
- 机制评分: `1.771994`
- 是否触发警报: `是`
- 提前量 (ms): `1.010`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `7`

证据:
1. `sxr_kurt` (贡献度 `1.771994`), 标签: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (贡献度 `0.577026`), 标签: `locked_mode,low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.499690`), 标签: `locked_mode`

### 炮号 1057573

- 主要机制: `density_limit`
- 机制评分: `1.416704`
- 是否触发警报: `是`
- 提前量 (ms): `6.023`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `9`

证据:
1. `sxr_kurt` (贡献度 `1.416704`), 标签: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (贡献度 `0.540852`), 标签: `locked_mode,low_q_current_limit`
3. `qa_proxy` (贡献度 `0.524148`), 标签: `low_q_current_limit`

### 炮号 1057607

- 主要机制: `density_limit`
- 机制评分: `1.325927`
- 是否触发警报: `是`
- 提前量 (ms): `7.051`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `10`

证据:
1. `sxr_kurt` (贡献度 `1.325927`), 标签: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (贡献度 `0.612826`), 标签: `low_q_current_limit`
3. `Mir_avg_fre` (贡献度 `0.371895`), 标签: `locked_mode`

### 炮号 1057650

- 主要机制: `vde_control_loss`
- 机制评分: `1.527121`
- 是否触发警报: `是`
- 提前量 (ms): `50.897`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `10`

证据:
1. `Z_proxy` (贡献度 `0.909077`), 标签: `vde_control_loss`
2. `v_loop` (贡献度 `0.618045`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.579770`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1057652

- 主要机制: `vde_control_loss`
- 机制评分: `3.057058`
- 是否触发警报: `是`
- 提前量 (ms): `158.839`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `100`

证据:
1. `dy_a` (贡献度 `2.189639`), 标签: `vde_control_loss`
2. `Z_proxy` (贡献度 `0.867419`), 标签: `vde_control_loss`
3. `mode_number_n` (贡献度 `0.261871`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1057917

- 主要机制: `vde_control_loss`
- 机制评分: `1.354571`
- 是否触发警报: `是`
- 提前量 (ms): `59.251`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `16`

证据:
1. `Z_proxy` (贡献度 `0.709848`), 标签: `vde_control_loss`
2. `v_loop` (贡献度 `0.644724`), 标签: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (贡献度 `0.570671`), 标签: `locked_mode,low_q_current_limit`

### 炮号 1058008

- 主要机制: `density_limit`
- 机制评分: `1.420626`
- 是否触发警报: `是`
- 提前量 (ms): `16.125`
- 原因窗口规则: `y_true==1`
- 原因窗口数据点数: `20`

证据:
1. `sxr_kurt` (贡献度 `1.420626`), 标签: `density_limit,impurity_radiation_collapse`
2. `n=1 amplitude` (贡献度 `0.412052`), 标签: `locked_mode`
3. `mode_number_n` (贡献度 `0.362717`), 标签: `locked_mode,low_q_current_limit`

