# Feature-Physics Map (MVP, Source-Grounded)

## Scope
- This map links disruption mechanism categories to observable features available in `paper_131` PGFE/PGFE-U design.
- Primary evidence source: `ref/paper_131.txt`.

## Mechanism-to-Feature Mapping

| Mechanism category | Physical chain (paper terminology) | Observable feature relations | Candidate features (PGFE / PGFE-U) | Evidence |
|---|---|---|---|---|
| Density-limit disruption | Approaching/exceeding Greenwald density limit -> edge radiation asymmetry/current-channel contraction -> MHD escalation -> disruption | `ne0/nG` rises toward limit; density/radiation profile statistics change; MHD proxies often rise in later phase | `ne0`, `ne0/nG`, `DENkurt/skew/var`, `SXRkurt/skew/var`, `AXUVkurt/skew/var`, `CIII`, `v_loop`, `Mir_*` | `ref/paper_131.txt:448`, `ref/paper_131.txt:449`, `ref/paper_131.txt:2694`, `ref/paper_131.txt:1661` |
| Locked-mode / tearing-mode pathway | Tearing-mode growth and mode deceleration -> mode locking -> disruption | Main Mirnov frequency decreases while amplitude increases; `n=1` locked-mode amplitude becomes strong precursor | `Mir_fre`, `Mir_abs`, `Mir_Vpp`, `mode_number_m`, `mode_number_n`, `n=1 amplitude`, `n=1 phase` | `ref/paper_131.txt:1457`, `ref/paper_131.txt:1461`, `ref/paper_131.txt:1523`, `ref/paper_131.txt:1533` |
| Low-q (current-limit) pathway | `q95 < 2` regime increases instability risk (internal kink-like behavior) | Current-related proxies indicate low-q tendency; combined with MHD growth features increases risk | `Bt/Ip` (or `qa_proxy`), `Ip`, `Ip_diff`, `Mir_*`, mode numbers | `ref/paper_131.txt:450`, `ref/paper_131.txt:451`, `ref/paper_131.txt:2680`, `ref/paper_131.txt:2681` |
| VDE/control-loss pathway | Vertical/horizontal control deterioration or VDE chain -> fast terminal disruption | Position excursions and fast trajectory changes precede terminal phase; often coupled with control/radiation changes | `dz`, `dr`, `dz/a`, `dr/a`, `Ip_diff`, `v_loop`, `Z_proxy` | `ref/paper_131.txt:439`, `ref/paper_131.txt:2195`, `ref/paper_131.txt:2251`, `ref/paper_131.txt:2696` |
| Impurity/radiation-collapse pathway | Impurity rise/radiative collapse -> profile degradation -> MHD destabilization -> disruption | Impurity and loop-voltage proxies rise; radiation profile moments shift; later MHD signatures increase | `CIII`, `v_loop`, `SXRcore`/`SXR_mean`, `AXUVkurt/skew/var`, `SXRkurt/skew/var`, `Mir_*` | `ref/paper_131.txt:439`, `ref/paper_131.txt:2195`, `ref/paper_131.txt:2220`, `ref/paper_131.txt:3552` |
| Beta-limit / pressure-driven MHD (coarse proxy) | High normalized beta can excite ideal MHD instability | No direct `beta_N` feature listed in MVP set; infer via coupled radiation/control/MHD evolution | Proxy-only in MVP: `SXR*`, `AXUV*`, `Ip`, `Bt`, `Mir_*` | `ref/paper_131.txt:455` (direct limit statement); proxy mapping is an assumption |

## Cross-Device Feature Harmonization Notes
- J-TEXT (90 channels) and EAST (65 channels) are mapped into a common 25-feature space at 1 kHz.
- Non-circular EAST geometry complicates direct `mode_number_m` extraction; PGFE-U introduces SVD-based mode extraction to improve robustness.
- PGFE-U also introduces normalized proxies (`ne0/nG`, `qa_proxy`, `Z_proxy`, `dr/a`, `dz/a`) for cross-device comparability.
- Evidence: `ref/paper_131.txt:2221`, `ref/paper_131.txt:2222`, `ref/paper_131.txt:2605`, `ref/paper_131.txt:2613`, `ref/paper_131.txt:2694`.

## Labeling/Interpretability Linkage (for downstream use)
- Fixed-window labels (J-TEXT 25 ms; EAST 125 ms in prior setup) provide baseline training targets.
- FLS refines shot-wise precursor onset and introduces gray-zone buffering (J-TEXT 50 ms; EAST 200 ms).
- SHAP is the prescribed post-hoc bridge from features to mechanism interpretation in both single-device and cross-device studies.
- Evidence: `ref/paper_131.txt:2704`, `ref/paper_131.txt:2741`, `ref/paper_131.txt:2743`, `ref/paper_131.txt:3459`.
