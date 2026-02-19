# Per-Shot Disruption Reason Report (sfpr002)

- Reason rows: `38`
- Test accuracy: `0.990885`
- Test ROC-AUC: `0.978437`
- Shot accuracy: `0.953757`
- Shot TPR/FPR: `0.842105` / `0.014815`
- Threshold objective/theta: `shot_fpr_constrained` / `0.668796`

## Mechanism Distribution

| mechanism | count |
|---|---:|
| density_limit | 18 |
| vde_control_loss | 10 |
| low_q_current_limit | 8 |
| locked_mode | 2 |

## Shot Explanations

### Shot 1051510

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `1.517893`
- Warning triggered: `Yes`
- Lead time (ms): `120.486`
- Reason window rule: `y_true==1`
- Reason window points: `44`

Evidence:
1. `qa_proxy` (contribution `0.779343`), tags: `low_q_current_limit`
2. `mode_number_n` (contribution `0.738550`), tags: `locked_mode,low_q_current_limit`
3. `v_loop` (contribution `0.412575`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`

### Shot 1051684

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.268418`
- Warning triggered: `Yes`
- Lead time (ms): `46.675`
- Reason window rule: `y_true==1`
- Reason window points: `15`

Evidence:
1. `sxr_kurt` (contribution `0.792833`), tags: `density_limit,impurity_radiation_collapse`
2. `sxr_var` (contribution `0.475585`), tags: `density_limit,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.463835`), tags: `locked_mode,low_q_current_limit`

### Shot 1051701

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.588321`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `5`

Evidence:
1. `Z_proxy` (contribution `1.086058`), tags: `vde_control_loss`
2. `v_loop` (contribution `0.502264`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.428745`), tags: `locked_mode,low_q_current_limit`

### Shot 1052062

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.099757`
- Warning triggered: `Yes`
- Lead time (ms): `1.010`
- Reason window rule: `y_true==1`
- Reason window points: `16`

Evidence:
1. `xuv_ratio` (contribution `0.657856`), tags: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (contribution `0.555681`), tags: `vde_control_loss`
3. `v_loop` (contribution `0.441902`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`

### Shot 1052127

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.105375`
- Warning triggered: `Yes`
- Lead time (ms): `3.048`
- Reason window rule: `y_true==1`
- Reason window points: `12`

Evidence:
1. `v_loop` (contribution `0.711419`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `Bt` (contribution `0.439520`), tags: `low_q_current_limit`
3. `Z_proxy` (contribution `0.393957`), tags: `vde_control_loss`

### Shot 1052920

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.233884`
- Warning triggered: `Yes`
- Lead time (ms): `1.006`
- Reason window rule: `y_true==1`
- Reason window points: `7`

Evidence:
1. `Z_proxy` (contribution `0.700322`), tags: `vde_control_loss`
2. `v_loop` (contribution `0.533562`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.391761`), tags: `locked_mode,low_q_current_limit`

### Shot 1052948

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `0.677389`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `7`

Evidence:
1. `qa_proxy` (contribution `0.677389`), tags: `low_q_current_limit`
2. `Mir_avg_fre` (contribution `0.536732`), tags: `locked_mode`
3. `dx_a` (contribution `0.303911`), tags: `vde_control_loss`

### Shot 1052954

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `1.405668`
- Warning triggered: `Yes`
- Lead time (ms): `1.012`
- Reason window rule: `y_true==1`
- Reason window points: `6`

Evidence:
1. `mode_number_n` (contribution `0.722922`), tags: `locked_mode,low_q_current_limit`
2. `qa_proxy` (contribution `0.682746`), tags: `low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.476223`), tags: `locked_mode`

### Shot 1052960

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `0.665117`
- Warning triggered: `Yes`
- Lead time (ms): `7.030`
- Reason window rule: `y_true==1`
- Reason window points: `16`

Evidence:
1. `qa_proxy` (contribution `0.665117`), tags: `low_q_current_limit`
2. `sxr_mean` (contribution `0.381901`), tags: `density_limit,impurity_radiation_collapse`
3. `dx_a` (contribution `0.359801`), tags: `vde_control_loss`

### Shot 1053058

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.758013`
- Warning triggered: `Yes`
- Lead time (ms): `1.007`
- Reason window rule: `y_true==1`
- Reason window points: `6`

Evidence:
1. `v_loop` (contribution `1.016109`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `mode_number_n` (contribution `0.859155`), tags: `locked_mode,low_q_current_limit`
3. `Z_proxy` (contribution `0.741903`), tags: `vde_control_loss`

### Shot 1053313

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.200469`
- Warning triggered: `Yes`
- Lead time (ms): `2.007`
- Reason window rule: `y_true==1`
- Reason window points: `7`

Evidence:
1. `mode_number_n` (contribution `1.176652`), tags: `locked_mode,low_q_current_limit`
2. `v_loop` (contribution `0.631085`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `Z_proxy` (contribution `0.569383`), tags: `vde_control_loss`

### Shot 1053552

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `1.588849`
- Warning triggered: `Yes`
- Lead time (ms): `4.019`
- Reason window rule: `y_true==1`
- Reason window points: `10`

Evidence:
1. `qa_proxy` (contribution `0.944837`), tags: `low_q_current_limit`
2. `mode_number_n` (contribution `0.644012`), tags: `locked_mode,low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.335896`), tags: `locked_mode`

### Shot 1053932

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `2.142276`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `1`

Evidence:
1. `v_loop` (contribution `1.294714`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
2. `Z_proxy` (contribution `0.847562`), tags: `vde_control_loss`
3. `Bt` (contribution `0.745447`), tags: `low_q_current_limit`

### Shot 1054153

- Primary mechanism: `locked_mode`
- Primary mechanism score: `0.979493`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `8`

Evidence:
1. `mode_number_n` (contribution `0.504911`), tags: `locked_mode,low_q_current_limit`
2. `Mir_avg_fre` (contribution `0.474583`), tags: `locked_mode`
3. `CIII` (contribution `0.428903`), tags: `density_limit,impurity_radiation_collapse`

### Shot 1054154

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.024138`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `10`

Evidence:
1. `CIII` (contribution `0.564860`), tags: `density_limit,impurity_radiation_collapse`
2. `v_loop` (contribution `0.459278`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `MNM` (contribution `0.258689`), tags: `locked_mode`

### Shot 1054161

- Primary mechanism: `density_limit`
- Primary mechanism score: `0.829674`
- Warning triggered: `Yes`
- Lead time (ms): `-0.000`
- Reason window rule: `y_true==1`
- Reason window points: `10`

Evidence:
1. `Mir_avg_fre` (contribution `0.462799`), tags: `locked_mode`
2. `CIII` (contribution `0.454228`), tags: `density_limit,impurity_radiation_collapse`
3. `v_loop` (contribution `0.375446`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`

### Shot 1054188

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `0.982058`
- Warning triggered: `Yes`
- Lead time (ms): `12.052`
- Reason window rule: `y_true==1`
- Reason window points: `32`

Evidence:
1. `Bt` (contribution `0.759899`), tags: `low_q_current_limit`
2. `xuv_skew` (contribution `0.248054`), tags: `density_limit,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.222160`), tags: `locked_mode,low_q_current_limit`

### Shot 1054337

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.278655`
- Warning triggered: `No`
- Lead time (ms): `N/A`
- Reason window rule: `y_true==1`
- Reason window points: `12`

Evidence:
1. `xuv_ratio` (contribution `0.893637`), tags: `density_limit,impurity_radiation_collapse`
2. `sxr_mean` (contribution `0.385018`), tags: `density_limit,impurity_radiation_collapse`
3. `Z_proxy` (contribution `0.256352`), tags: `vde_control_loss`

### Shot 1054838

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.990761`
- Warning triggered: `Yes`
- Lead time (ms): `-0.000`
- Reason window rule: `y_true==1`
- Reason window points: `6`

Evidence:
1. `sxr_kurt` (contribution `1.636415`), tags: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (contribution `0.688372`), tags: `locked_mode,low_q_current_limit`
3. `CIII` (contribution `0.354347`), tags: `density_limit,impurity_radiation_collapse`

### Shot 1055242

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.496075`
- Warning triggered: `Yes`
- Lead time (ms): `14.050`
- Reason window rule: `y_true==1`
- Reason window points: `17`

Evidence:
1. `sxr_kurt` (contribution `1.496075`), tags: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (contribution `0.511880`), tags: `low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.507289`), tags: `locked_mode`

### Shot 1055244

- Primary mechanism: `density_limit`
- Primary mechanism score: `2.679164`
- Warning triggered: `Yes`
- Lead time (ms): `20.055`
- Reason window rule: `y_true==1`
- Reason window points: `23`

Evidence:
1. `sxr_kurt` (contribution `2.154273`), tags: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (contribution `0.632882`), tags: `vde_control_loss`
3. `v_loop` (contribution `0.524890`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`

### Shot 1055353

- Primary mechanism: `density_limit`
- Primary mechanism score: `2.250139`
- Warning triggered: `Yes`
- Lead time (ms): `5.014`
- Reason window rule: `y_true==1`
- Reason window points: `8`

Evidence:
1. `sxr_kurt` (contribution `2.250139`), tags: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (contribution `0.398731`), tags: `low_q_current_limit`
3. `Z_proxy` (contribution `0.337077`), tags: `vde_control_loss`

### Shot 1055357

- Primary mechanism: `density_limit`
- Primary mechanism score: `0.690792`
- Warning triggered: `Yes`
- Lead time (ms): `9.032`
- Reason window rule: `y_true==1`
- Reason window points: `12`

Evidence:
1. `sxr_kurt` (contribution `0.690792`), tags: `density_limit,impurity_radiation_collapse`
2. `Z_proxy` (contribution `0.678523`), tags: `vde_control_loss`
3. `qa_proxy` (contribution `0.650684`), tags: `low_q_current_limit`

### Shot 1056187

- Primary mechanism: `locked_mode`
- Primary mechanism score: `0.689935`
- Warning triggered: `Yes`
- Lead time (ms): `57.283`
- Reason window rule: `y_true==1`
- Reason window points: `8`

Evidence:
1. `Mir_avg_fre` (contribution `0.689935`), tags: `locked_mode`
2. `qa_proxy` (contribution `0.651779`), tags: `low_q_current_limit`
3. `v_loop` (contribution `0.359056`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`

### Shot 1056189

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `0.552552`
- Warning triggered: `Yes`
- Lead time (ms): `53.260`
- Reason window rule: `y_true==1`
- Reason window points: `8`

Evidence:
1. `qa_proxy` (contribution `0.552552`), tags: `low_q_current_limit`
2. `v_loop` (contribution `0.466907`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `MNM` (contribution `0.228567`), tags: `locked_mode`

### Shot 1056204

- Primary mechanism: `low_q_current_limit`
- Primary mechanism score: `0.745279`
- Warning triggered: `Yes`
- Lead time (ms): `10.042`
- Reason window rule: `y_true==1`
- Reason window points: `19`

Evidence:
1. `qa_proxy` (contribution `0.745279`), tags: `low_q_current_limit`
2. `Z_proxy` (contribution `0.653257`), tags: `vde_control_loss`
3. `Mir_avg_fre` (contribution `0.599674`), tags: `locked_mode`

### Shot 1056828

- Primary mechanism: `density_limit`
- Primary mechanism score: `2.320270`
- Warning triggered: `Yes`
- Lead time (ms): `9.084`
- Reason window rule: `y_true==1`
- Reason window points: `20`

Evidence:
1. `sxr_kurt` (contribution `1.696847`), tags: `density_limit,impurity_radiation_collapse`
2. `v_loop` (contribution `0.327064`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `xuv_ratio` (contribution `0.296359`), tags: `density_limit,impurity_radiation_collapse`

### Shot 1057200

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.069599`
- Warning triggered: `Yes`
- Lead time (ms): `3.025`
- Reason window rule: `y_true==1`
- Reason window points: `24`

Evidence:
1. `xuv_ratio` (contribution `1.069599`), tags: `density_limit,impurity_radiation_collapse`
2. `Bt` (contribution `0.409778`), tags: `low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.338444`), tags: `locked_mode`

### Shot 1057254

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.007067`
- Warning triggered: `Yes`
- Lead time (ms): `66.197`
- Reason window rule: `y_true==1`
- Reason window points: `82`

Evidence:
1. `xuv_ratio` (contribution `1.007067`), tags: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (contribution `0.726177`), tags: `locked_mode,low_q_current_limit`
3. `Z_proxy` (contribution `0.674415`), tags: `vde_control_loss`

### Shot 1057448

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `0.832846`
- Warning triggered: `Yes`
- Lead time (ms): `-0.000`
- Reason window rule: `y_true==1`
- Reason window points: `165`

Evidence:
1. `dx_a` (contribution `0.832846`), tags: `vde_control_loss`
2. `sxr_mean` (contribution `0.298542`), tags: `density_limit,impurity_radiation_collapse`
3. `sxr_var` (contribution `0.218366`), tags: `density_limit,impurity_radiation_collapse`

### Shot 1057472

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.802226`
- Warning triggered: `Yes`
- Lead time (ms): `7.136`
- Reason window rule: `y_true==1`
- Reason window points: `13`

Evidence:
1. `sxr_kurt` (contribution `1.479521`), tags: `density_limit,impurity_radiation_collapse`
2. `v_loop` (contribution `0.322705`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `Z_proxy` (contribution `0.260236`), tags: `vde_control_loss`

### Shot 1057504

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.771994`
- Warning triggered: `Yes`
- Lead time (ms): `1.010`
- Reason window rule: `y_true==1`
- Reason window points: `7`

Evidence:
1. `sxr_kurt` (contribution `1.771994`), tags: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (contribution `0.577026`), tags: `locked_mode,low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.499690`), tags: `locked_mode`

### Shot 1057573

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.416704`
- Warning triggered: `Yes`
- Lead time (ms): `6.023`
- Reason window rule: `y_true==1`
- Reason window points: `9`

Evidence:
1. `sxr_kurt` (contribution `1.416704`), tags: `density_limit,impurity_radiation_collapse`
2. `mode_number_n` (contribution `0.540852`), tags: `locked_mode,low_q_current_limit`
3. `qa_proxy` (contribution `0.524148`), tags: `low_q_current_limit`

### Shot 1057607

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.325927`
- Warning triggered: `Yes`
- Lead time (ms): `7.051`
- Reason window rule: `y_true==1`
- Reason window points: `10`

Evidence:
1. `sxr_kurt` (contribution `1.325927`), tags: `density_limit,impurity_radiation_collapse`
2. `qa_proxy` (contribution `0.612826`), tags: `low_q_current_limit`
3. `Mir_avg_fre` (contribution `0.371895`), tags: `locked_mode`

### Shot 1057650

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.527121`
- Warning triggered: `Yes`
- Lead time (ms): `50.897`
- Reason window rule: `y_true==1`
- Reason window points: `10`

Evidence:
1. `Z_proxy` (contribution `0.909077`), tags: `vde_control_loss`
2. `v_loop` (contribution `0.618045`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.579770`), tags: `locked_mode,low_q_current_limit`

### Shot 1057652

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `3.057058`
- Warning triggered: `Yes`
- Lead time (ms): `158.839`
- Reason window rule: `y_true==1`
- Reason window points: `100`

Evidence:
1. `dy_a` (contribution `2.189639`), tags: `vde_control_loss`
2. `Z_proxy` (contribution `0.867419`), tags: `vde_control_loss`
3. `mode_number_n` (contribution `0.261871`), tags: `locked_mode,low_q_current_limit`

### Shot 1057917

- Primary mechanism: `vde_control_loss`
- Primary mechanism score: `1.354571`
- Warning triggered: `Yes`
- Lead time (ms): `59.251`
- Reason window rule: `y_true==1`
- Reason window points: `16`

Evidence:
1. `Z_proxy` (contribution `0.709848`), tags: `vde_control_loss`
2. `v_loop` (contribution `0.644724`), tags: `density_limit,vde_control_loss,impurity_radiation_collapse`
3. `mode_number_n` (contribution `0.570671`), tags: `locked_mode,low_q_current_limit`

### Shot 1058008

- Primary mechanism: `density_limit`
- Primary mechanism score: `1.420626`
- Warning triggered: `Yes`
- Lead time (ms): `16.125`
- Reason window rule: `y_true==1`
- Reason window points: `20`

Evidence:
1. `sxr_kurt` (contribution `1.420626`), tags: `density_limit,impurity_radiation_collapse`
2. `n=1 amplitude` (contribution `0.412052`), tags: `locked_mode`
3. `mode_number_n` (contribution `0.362717`), tags: `locked_mode,low_q_current_limit`

