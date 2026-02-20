# 特征-物理映射 (MVP，基于文献)

## 范围
- 本映射将破裂机制类别与 `paper_131` PGFE/PGFE-U 设计中可用的可观测特征关联。
- 主要证据来源：`ref/paper_131.txt`。

## 机制-特征映射

| 机制类别 | 物理链（论文术语） | 可观测特征关系 | 候选特征 (PGFE / PGFE-U) | 证据 |
|---|---|---|---|---|
| 密度极限破裂 | 逼近/超过格林瓦尔德密度极限 -> 边缘辐射不对称/电流通道收缩 -> MHD（磁流体动力学）升级 -> 破裂 | `ne0/nG` 趋近极限；密度/辐射剖面统计量变化；后期 MHD 代理量通常上升 | `ne0`、`ne0/nG`、`DENkurt/skew/var`、`SXRkurt/skew/var`、`AXUVkurt/skew/var`、`CIII`、`v_loop`、`Mir_*` | `ref/paper_131.txt:448`、`ref/paper_131.txt:449`、`ref/paper_131.txt:2694`、`ref/paper_131.txt:1661` |
| 锁模 / 撕裂模路径 | 撕裂模增长与模式减速 -> 模式锁定 -> 破裂 | 主 Mirnov 频率下降而幅值增大；`n=1` 锁模幅值成为强前兆信号 | `Mir_fre`、`Mir_abs`、`Mir_Vpp`、`mode_number_m`、`mode_number_n`、`n=1 amplitude`、`n=1 phase` | `ref/paper_131.txt:1457`、`ref/paper_131.txt:1461`、`ref/paper_131.txt:1523`、`ref/paper_131.txt:1533` |
| 低 q（电流极限）路径 | `q95 < 2` 区域增加不稳定性风险（内部扭曲模行为） | 电流相关代理量指示低 q 趋势；结合 MHD 增长特征可增加风险 | `Bt/Ip`（或 `qa_proxy`）、`Ip`、`Ip_diff`、`Mir_*`、模式数 | `ref/paper_131.txt:450`、`ref/paper_131.txt:451`、`ref/paper_131.txt:2680`、`ref/paper_131.txt:2681` |
| VDE（垂直位移事件）/ 控制丧失路径 | 垂直/水平控制恶化或 VDE 链 -> 快速终端破裂 | 位置偏移和快速轨迹变化先于终端阶段出现；常与控制/辐射变化耦合 | `dz`、`dr`、`dz/a`、`dr/a`、`Ip_diff`、`v_loop`、`Z_proxy` | `ref/paper_131.txt:439`、`ref/paper_131.txt:2195`、`ref/paper_131.txt:2251`、`ref/paper_131.txt:2696` |
| 杂质/辐射坍塌路径 | 杂质上升/辐射坍塌 -> 剖面退化 -> MHD 去稳定 -> 破裂 | 杂质和环电压代理量上升；辐射剖面矩量偏移；后期 MHD 特征增强 | `CIII`、`v_loop`、`SXRcore`/`SXR_mean`、`AXUVkurt/skew/var`、`SXRkurt/skew/var`、`Mir_*` | `ref/paper_131.txt:439`、`ref/paper_131.txt:2195`、`ref/paper_131.txt:2220`、`ref/paper_131.txt:3552` |
| Beta 极限 / 压力驱动 MHD（粗略代理） | 高归一化 beta 可激发理想 MHD 不稳定性 | MVP 特征集中无直接 `beta_N` 特征；通过耦合的辐射/控制/MHD 演化推断 | MVP 中仅有代理：`SXR*`、`AXUV*`、`Ip`、`Bt`、`Mir_*` | `ref/paper_131.txt:455`（直接极限声明）；代理映射为假设 |

## 跨装置特征协调说明
- J-TEXT（90 通道）和 EAST（65 通道）被映射到共同的 25 特征空间，采样率 1 kHz。
- EAST 的非圆截面几何结构使直接提取 `mode_number_m` 更加困难；PGFE-U 引入基于 SVD（奇异值分解）的模式提取以提高鲁棒性。
- PGFE-U 还引入了归一化代理量（`ne0/nG`、`qa_proxy`、`Z_proxy`、`dr/a`、`dz/a`）以实现跨装置可比性。
- 证据：`ref/paper_131.txt:2221`、`ref/paper_131.txt:2222`、`ref/paper_131.txt:2605`、`ref/paper_131.txt:2613`、`ref/paper_131.txt:2694`。

## 标注/可解释性关联 (供下游使用)
- 固定窗口标签（J-TEXT 25 ms；EAST 在先前设置中为 125 ms）提供基线训练目标。
- FLS 改进了逐炮的前兆起始点并引入灰区缓冲（J-TEXT 50 ms；EAST 200 ms）。
- SHAP 是论文中规定的后处理桥梁，在单装置与跨装置研究中均从特征到机制进行解释。
- 证据：`ref/paper_131.txt:2704`、`ref/paper_131.txt:2741`、`ref/paper_131.txt:2743`、`ref/paper_131.txt:3459`。
