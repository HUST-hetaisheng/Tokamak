# Literature Review: Tokamak Disruption Mechanisms and Method Chain (J-TEXT/EAST)

## Scope and Sources
- Primary evidence source: `ref/paper_131.txt`.
- Supporting project context: `analysis/method_survey.md`, `analysis/framework_review.md`, `ref/托卡马克项目 - 2026-02-19.md`.
- If a statement comes from conversational project notes rather than a clearly traceable paper excerpt, it is marked as an assumption.

## 1. Disruption Mechanism Categories

### 1.1 Event-chain perspective
- Disruptions are described as either human-triggered (control/system faults) or naturally evolved plasma events.
- Natural-disruption chains are reported as complex and frequently converge to locked-mode-associated terminal behavior.
- Human-triggered chains include VDE and impurity/radiation-collapse pathways.
- Evidence: `ref/paper_131.txt:433`, `ref/paper_131.txt:439`, `ref/paper_131.txt:443`, `ref/paper_131.txt:445`.

### 1.2 Operational-limit categories (physics scaling viewpoint)
- Density limit (Greenwald): approaching/exceeding `n_GW` is linked to edge radiation asymmetry and current-channel contraction, then MHD destabilization.
- Low-`q`/current limit: `q95 < 2` increases MHD risk (internal kink-like instability chain).
- Beta limit: high normalized beta can trigger ideal MHD instability.
- Evidence: `ref/paper_131.txt:448`, `ref/paper_131.txt:450`, `ref/paper_131.txt:455`.

### 1.3 Device-observed dominant causes (EAST/J-TEXT context)
- EAST major causes are summarized as impurity radiation, density-limit-related behavior, VDE, and MHD instability.
- J-TEXT and EAST differ in diagnostics/configuration; therefore feature extraction must preserve physics equivalence rather than raw-channel equivalence.
- Evidence: `ref/paper_131.txt:2195`, `ref/paper_131.txt:2199`, `ref/paper_131.txt:2201`.

## 2. Observable Feature Relations to Mechanisms

### 2.1 PGFE feature families
- Four feature families are explicitly defined: MHD instability, radiation, density, and plasma-control features.
- Evidence: `ref/paper_131.txt:1416`.

### 2.2 MHD precursor observables
- Tearing/locked-mode precursors are represented by `Mir_Vpp`, `Mir_fre`, `Mir_abs`, mode numbers, and `n=1 amplitude`.
- Typical locked-mode progression: main frequency decreases and amplitude grows rapidly before disruption.
- Evidence: `ref/paper_131.txt:1457`, `ref/paper_131.txt:1461`, `ref/paper_131.txt:1523`, `ref/paper_131.txt:1531`.

### 2.3 Radiation/density profile observables
- Array higher-order statistics (`var`, `skew`, `kurt`) are used to encode profile asymmetry/shape evolution for SXR/AXUV/FIR-like channels.
- Additional scalar proxies include `P_rad`, `CIII`, `SXR_core`, `sum_ne`, `ne0`.
- Evidence: `ref/paper_131.txt:1558`, `ref/paper_131.txt:1593`, `ref/paper_131.txt:1661`, `ref/paper_131.txt:1663`.

### 2.4 Control-state observables
- Balance/control signals include `Ip`, `Bt`, loop voltage, displacement (`dr`, `dz`), and derivatives/proxy terms.
- Evidence: `ref/paper_131.txt:1671`, `ref/paper_131.txt:2250`.

## 3. Method Chain: Purpose, Boundaries, Reproducibility Points

| Method | Purpose | Boundaries | Reproducibility points |
|---|---|---|---|
| PGFE | Encode disruption physics into tabular features (MHD/radiation/density/control), reduce diagnostic heterogeneity, inject inductive bias. | Not one-size-fits-all; feature-extraction parameters must be adapted to each device; some features (e.g., `mode_number_m`) are harder on non-circular EAST geometry. | Keep explicit feature definitions and channel mapping per device; keep sampling frequency and normalization protocol fixed. Evidence: `ref/paper_131.txt:1416`, `ref/paper_131.txt:1445`, `ref/paper_131.txt:2204`, `ref/paper_131.txt:2222`. |
| DART (GBDT with tree dropout) | Baseline classifier balancing performance and interpretability under limited data. | Still statistical, not causal; sensitive to feature quality and label policy; needs class imbalance handling. | Use same split protocol and weighted/undersampled training as reported. Evidence: `ref/paper_131.txt:1836`, `ref/paper_131.txt:1883`, `ref/paper_131.txt:2004`. |
| SHAP | Post-hoc global/local attribution for feature contribution and mechanism analysis. | Attribution is model-relative (not direct causality proof); interpretation depends on baseline/reference data. | Fix model version, test split, and SHAP computation dataset for comparability. Evidence: `ref/paper_131.txt:1843`, `ref/paper_131.txt:3460`, `ref/paper_131.txt:3463`. |
| CORAL / S-CORAL | Align source/target second-order statistics for cross-device transfer; S-CORAL uses label-aware class-wise alignment. | Linear covariance alignment cannot fully remove all domain gaps; U-CORAL wastes available label information in this task. | Report both U-CORAL and S-CORAL under identical features/splits; keep whitening-coloring and covariance computations deterministic. Evidence: `ref/paper_131.txt:2262`, `ref/paper_131.txt:2303`, `ref/paper_131.txt:2312`. |
| FLS | Improve label fidelity by shot-wise precursor onset from anomaly detection; enable reuse of disruptive-shot non-disruptive segments with gray-zone buffering. | Depends on anomaly-detection quality and gray-zone width; if AD is weak on new devices, manual/alternative labeling is needed. | Log AD model choice and onset outputs; keep gray-zone widths fixed (J-TEXT 50 ms, EAST 200 ms in reported setup). Evidence: `ref/paper_131.txt:2713`, `ref/paper_131.txt:2741`, `ref/paper_131.txt:2743`. |
| EFD | Enable zero-shot transfer by estimating target normalization parameters (`mu`, `sigma`) without target-model training. | Performance depends on normalization-parameter estimation quality; thesis implementation still used a small set of EAST disruptive shots for parameter estimation (not model training). | Save estimated `mu/sigma` artifact and sensitivity results; separate "parameter estimation data" from "model training data". Evidence: `ref/paper_131.txt:2834`, `ref/paper_131.txt:2855`, `ref/paper_131.txt:2861`. |

## 4. Quantitative Findings Reported in `paper_131`
- J-TEXT source model (IDP-PGFE + DART): `AUC 0.987`, `TPR(>10ms) 96.36%`, `FPR 2.73%`.
- With 10% training data: `AUC 0.939` (reported as 95.2% of baseline AUC).
- Cross-device (J-TEXT->EAST) with S-CORAL using `10` EAST disruptive + `100` EAST non-disruptive shots: `AUC 0.890`, with reported `17%` gain over mixing-data strategy.
- PGFE-U + FLS cross-device with only `10` EAST disruptive shots: `AUC 0.947` (`+6.4%` over prior S-CORAL setting).
- EFD zero-shot transfer: `AUC 0.892` (`91.6%` of baseline).
- Evidence: `ref/paper_131.txt:1928`, `ref/paper_131.txt:2057`, `ref/paper_131.txt:2577`, `ref/paper_131.txt:2795`, `ref/paper_131.txt:2904`, `ref/paper_131.txt:3002`.

## 5. Reproducibility Notes for This Repository
- Repository currently contains shot lists and references, but no active training script is present in `analysis/` at this snapshot (assumption from workspace inspection).
- Therefore, phase-1 should prioritize a minimal, quickly runnable J-TEXT E2E baseline with explicit artifacts and split manifests before expanding to cross-device transfer.
- Project notes in `ref/托卡马克项目 - 2026-02-19.md` provide useful engineering hints but are conversational; use as assumption unless cross-checked with `paper_131`.
