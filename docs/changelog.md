# Changelog

This log tracks repo milestones by implementation stage (commit-aligned when commits are available).

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
