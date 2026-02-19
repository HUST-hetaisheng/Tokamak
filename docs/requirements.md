# Requirements: Phase-1 Implementation (J-TEXT E2E Baseline First)

## 0. Objective and Priority
- Immediate objective: deliver a quickly runnable end-to-end J-TEXT disruption prediction baseline in this repository.
- Baseline output must include time-resolved disruption probability and shot-level warning decisions.
- Phase-1 prioritizes reproducibility and short turnaround; optional enhancements are deferred.

## 1. Project-Level Constraints
- Default data root must be: `G:\我的云端硬盘\Fuison\data` (unless explicitly overridden by user).
- `ref/paper_131.txt` is the primary methodology reference.
- `1.ipynb`, `2.ipynb`, `3.ipynb` (lithium-battery project lineage) are out of scope.
- Do not run long training in phase-1.

## 2. Scientific/Modeling Requirements (Phase-1 Mandatory)

### 2.1 Task definition
- Binary disruption prediction (`disruptive` vs `non-disruptive`) with timepoint probability output.
- Evaluation must report both:
- shot-level metrics (`AUC`, `TPR`, `FPR`, lead-time success),
- timepoint-level metrics (`ROC-AUC`, optional `PR-AUC`).

### 2.2 Label protocol
- Baseline label policy (required now): fixed pre-disruption warning window for J-TEXT (paper reports 25 ms in fixed-label setup).
- Keep a config switch for later FLS integration (do not block phase-1 on FLS).
- Evidence: `ref/paper_131.txt:1880`, `ref/paper_131.txt:2704`.

### 2.3 Feature protocol
- Use PGFE-style physics features grouped as MHD/radiation/density/control.
- Keep feature definitions explicit and versioned (feature name, formula, source channels, units/scaling).
- Evidence: `ref/paper_131.txt:1416`, `ref/paper_131.txt:2228`.

### 2.4 Classifier protocol
- Phase-1 baseline classifier: tree-ensemble with DART-style setup (paper baseline family).
- Must include class-imbalance treatment (weighting and/or controlled downsampling).
- Evidence: `ref/paper_131.txt:1836`, `ref/paper_131.txt:1883`.

### 2.5 Normalization and slicing
- Enforce z-score normalization using training-derived statistics only.
- Enforce causal time slicing and record all window lengths in run metadata.
- Evidence: `ref/paper_131.txt:1885`.

## 3. Data and Artifact Requirements

### 3.1 Required inputs
- `shot_list/J-TEXT/Disruption_J-TEXT.json`
- `shot_list/J-TEXT/Non-disruption_J-TEXT.json`
- `shot_list/J-TEXT/AdvancedTime_J-TEXT.json` (for later FLS/advanced labeling compatibility)
- J-TEXT HDF5 data under default data root (exact subpath resolved by Agent-2 pipeline).

### 3.2 Required outputs (phase-1 run artifact set)
- `metrics_summary.json` with `AUC`, `TPR`, `FPR`, lead-time metrics.
- `training_config.json` including seeds, split manifest, feature list, label policy.
- `normalization_stats.json` (`mu`, `sigma` per feature).
- `sequence_predictions/*.csv` (per-shot probability timeline + warning decision).
- `run_manifest.md` (data snapshot, commit hash if available, commands used).
- Output location must remain under `output/` (project rule).

## 4. Acceptance Criteria (Phase-1)
- Pipeline runs end-to-end on J-TEXT without manual edits after config setup.
- Re-run with same seed reproduces metrics within a tight tolerance (document tolerance).
- Baseline must report whether it meets the project discrimination target (`>=95%` classification accuracy target at project level); if not met, gap analysis is required.
- Evidence for benchmark expectation: J-TEXT paper baseline `AUC 0.987`, `TPR 96.36%`, `FPR 2.73%`.
- Evidence: `ref/paper_131.txt:1928`.

## 5. Phase-1 Execution Roadmap (Prioritized)

## P0 (Do now, required for immediate progress)
1. Build J-TEXT dataset manifest and deterministic split manifest.
2. Implement PGFE feature extraction path needed for baseline features.
3. Train/evaluate DART-style baseline with fixed-label protocol.
4. Export required artifacts and concise run report.

## P1 (Do next, after P0 is stable)
1. Add SHAP global/local explainability export for baseline model.
2. Add FLS-compatible labeling pipeline and A/B comparison against fixed labels.
3. Add feature-ablation report by mechanism family (MHD/radiation/density/control).

## P2 (Postpone; optional for phase-1 deadline)
1. Cross-device J-TEXT->EAST adaptation with CORAL/S-CORAL.
2. PGFE-U refinements (SVD mode extraction, normalized proxies).
3. EFD zero-shot normalization estimation workflow.
4. Lightweight sequence-model backend (TCN/SSM) for online deployment study.

## 6. Explicit Non-Goals for Phase-1
- Full multi-device transfer benchmark.
- Zero-shot transfer productionization.
- Long-horizon hyperparameter sweeps or exhaustive model zoo comparisons.

