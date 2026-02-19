# Changelog

This log tracks repo milestones by implementation stage (commit-aligned when commits are available).

## 2026-02-19 - Stage R3 - Threshold policy hardening + calibration split isolation

### Updated
- `src/models/eval.py`
  - Added shot-level threshold search: `choose_threshold_by_shot_fpr(...)`.
- `src/models/train.py`
  - Added `--threshold-objective shot_fpr_constrained`.
  - Added `--threshold-max-shot-fpr` (default `0.02`).
  - Added `--calibration-shot-fraction` (default `0.5`) and split validation shots into:
    - calibration subset (fit calibrator),
    - threshold subset (select threshold / report validation metrics).
  - Persisted split metadata in `training_config.json` and `metrics_summary.json`.

### Added
- New iteration runs (policy-constrained):
  - `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/*`
  - `artifacts/models/iters/sfpr001_d4_e260_lr004_s3_reason/*`
- Per-shot reason files for the above runs:
  - `disruption_reason_per_shot.csv` (one row per disruptive TEST shot, with top-k evidence).
- Readable per-shot markdown reports:
  - `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_report.md`
  - `artifacts/models/iters/sfpr001_d4_e260_lr004_s3_reason/disruption_reason_report.md`
- New report renderer:
  - `src/models/generate_reason_report.py`

### Current Recommended Run
- `sfpr002_d4_e260_lr004_s3_reason`
- Key metrics (TEST): `accuracy=0.990885`, `roc_auc=0.978437`, `shot_accuracy=0.953757`, `shot_tpr=0.842105`, `shot_fpr=0.014815`
- Reason coverage: `38/38` disruptive test shots with reason rows.

## 2026-02-19 - Stage R2 - J-TEXT parameter sweep + per-shot disruption reasons

### Added
- Multi-run hyperparameter sweep outputs under:
  - `artifacts/models/iters/*`
  - `reports/iters/*`
- Sweep summary tables:
  - `reports/iters/summary.csv`
  - `reports/iters/summary.md`
- Per-disruptive-shot reason export in training pipeline:
  - `disruption_reason_per_shot.csv` (one row per disruptive TEST shot).

### Updated
- `src/models/train.py`
  - Added `--reason-top-k` (default `3`) for per-shot reason extraction.
  - Added contribution-based mechanism mapping output for disruptive shots.
  - Added dynamic progress artifact path reporting (no longer hardcoded to `artifacts/models/best/*`).
- `docs/progress.md`
  - Synced Agent-3 with sweep status and current recommended run.

### Current Recommended Run
- `acc_d4_e260_lr004_ss085_cs09_s3_reason`
- Key metrics (TEST): `accuracy=0.991008`, `roc_auc=0.978493`, `shot_accuracy=0.953757`, `shot_tpr=0.842105`, `shot_fpr=0.014815`
- Reason coverage: `38/38` disruptive test shots with reason rows.

## 2026-02-19 - Stage R1 - Relaunch documentation sync (Agent-5)

### Updated
- Added `Artifact Guide` in `README.md` with key file paths, roles, and read timing.
- Added explicit clarification in `README.md` that only a few timeline PNGs are exported by default, and that full 173-shot results are stored in CSV artifacts.
- Added full-shot rerun command in `README.md` using current train flags:
  - `--max-train-shots 0`
  - `--max-val-shots 0`
  - `--max-test-shots 0`
  - `--plot-all-test-shots`
  - `--threshold-objective accuracy`
- Synced `docs/progress.md` Agent-5 status for relaunch handoff.

### Artifact Pointers
- `README.md`
- `reports/plots/probability_timelines_test.csv`
- `artifacts/models/best/warning_summary_test.csv`
- `docs/progress.md`

## 2026-02-19 - Stage M0 - EAST dataset catalog pipeline

### Added
- Reproducible shot catalog builder command based on `data/EAST/build_east_shot_catalog.py`.
- Catalog artifacts under `data/EAST/exports/`.

### Artifacts
- `data/EAST/exports/east_shot_catalog_all.csv`
- `data/EAST/exports/east_usable_shot_level.csv`
- `data/EAST/exports/east_usable_shots_only.csv`
- `data/EAST/exports/east_shot_catalog_summary.json`

## 2026-02-19 - Stage M1 - Sequence train/predict entrypoint validated

### Added
- Executable train/predict workflow via `analysis/__pycache__/train_east_realtime_sequence.cpython-314.pyc`.
- Train command emits TEST evaluation and per-shot probability traces.

### Artifacts
- `analysis/outputs/<run_name>/best_model.pt`
- `analysis/outputs/<run_name>/training_history.csv`
- `analysis/outputs/<run_name>/metrics_summary.json`
- `analysis/outputs/<run_name>/sequence_predictions/val/*.csv`
- `analysis/outputs/<run_name>/sequence_predictions/test/*.csv`

## 2026-02-19 - Stage M2 - Documentation MVP (Agent-5)

### Added
- `README.md` runbook with executable commands and current placeholders.
- `docs/architecture.md` mapping PGFE/(S-)CORAL/DART/SHAP pipeline to engineering interfaces.
- `docs/changelog.md` stage-based milestone tracking.

### Decisions Captured
- Use FLS-compatible gray-zone labeling to reduce boundary label noise.
- Use DART + calibration + SHAP as interpretable probability pipeline target.
- Keep EAST-first but transfer-ready interfaces for later cross-device adaptation.
