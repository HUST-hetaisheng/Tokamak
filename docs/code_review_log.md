# Code Review Log

Last Updated: 2026-02-19

## Agent-4 Review Checklist
- Data leakage: training/validation/test isolation at shot level and by device.
- Label leakage: no direct or proxy leakage from disruption labels or advanced-time labels into features.
- Time-axis integrity: causal windows only, no future samples, aligned clocks across signals.
- Metric validity: event-level and timepoint-level metrics reported with clear threshold protocol.
- Probability quality: calibration checked (for example reliability curve, ECE/Brier) before operational claims.
- Reproducibility: fixed seeds, deterministic settings documented, split artifacts versioned.

## 2026-02-19 Initial Baseline Scan

Scope:
- Reviewed `docs/progress.md` for Agent-2/3/5 artifact status.
- Scanned repository file tree and `analysis/` for executable training/evaluation code.
- Checked for existing Python source files with `rg --files -g "*.py"`.

Findings:
- [INFO] No reviewable outputs from Agent-2, Agent-3, or Agent-5 are present yet. Current `docs/progress.md` marks them queued or dependency-waiting.
- [MEDIUM] Reproducibility risk in current workspace state: no Python source files were detected, so smoke validation and leakage checks cannot be executed yet.

Actionable Suggestions:
- Agent-2: when dataset-build scripts are added, expose split generation and feature assembly as functions with deterministic seed arguments and explicit shot-level split asserts.
- Agent-3: when training/eval scripts are added, separate fit/transform stages by split and include calibration evaluation artifacts (`calibration_curve`, `brier_score`, or ECE).
- Agent-5: document exact commands, seeds, data roots, and artifact paths in changelog/architecture docs to keep runs reproducible.

## 2026-02-19 Review Cycle: Agent-2 Dataset Pipeline

Scope:
- Reviewed `src/data/build_dataset.py`.
- Reviewed `reports/data_audit.md`.
- Reviewed dataset metadata artifacts under `artifacts/datasets/jtext_v1/`.
- Verified split-file disjointness in `splits/train.txt`, `splits/val.txt`, `splits/test.txt`.

Findings:
- [MEDIUM][Reproducibility/Data Source Policy] Hard-coded J-TEXT root selection does not honor repository default data-root policy.
  - Evidence: `src/data/build_dataset.py:58`, `src/data/build_dataset.py:60`, `artifacts/datasets/jtext_v1/summary.json:7`.
  - Risk: runs may silently use machine-local data snapshots, reducing reproducibility across environments.
  - Suggestion: add explicit `--data-root` (required unless user override), default it from the repository policy root, and persist the resolved source in summary metadata.

- [MEDIUM][Data Leakage / Input Drift Risk] Metadata JSON discovery is heuristic over all repo JSON files.
  - Evidence: `src/data/build_dataset.py:76`, `src/data/build_dataset.py:85`, `src/data/build_dataset.py:104`.
  - Risk: if multiple J-TEXT JSON variants exist, the pipeline can bind to an unintended file without failing, causing split/label drift or accidental leakage from derived files.
  - Suggestion: resolve fixed canonical paths under `shot_list/J-TEXT/` first, fail closed on ambiguity, and record file hashes.

- [MEDIUM][Time/Label Validity] Mixed disruptive labeling policy is active (`advanced_time` plus fallback fixed window for missing advanced shots).
  - Evidence: `src/data/build_dataset.py:243`, `src/data/build_dataset.py:247`, `artifacts/datasets/jtext_v1/summary.json:30`, `artifacts/datasets/jtext_v1/summary.json:34`.
  - Risk: training targets for disruptive class are not homogeneous; fallback-labeled shots can shift boundary behavior and distort warning-time metrics.
  - Suggestion: report metrics separately for `fls_source=advanced_time` vs `fallback_25ms`, and prefer excluding fallback-labeled disruptive shots from val/test until advanced labels are complete.

- [MEDIUM][Evaluation Bias Risk] Split strategy is shot-level stratification only, without temporal/session grouping.
  - Evidence: `src/data/build_dataset.py:279`, `src/data/build_dataset.py:500`.
  - Risk: near-neighbor shots from the same campaign may be split across train/val/test, inflating apparent generalization.
  - Suggestion: add a grouped temporal split option (for example by contiguous shot ranges or campaign/session id) and compare against random stratified split.

- [MEDIUM][Reproducibility Gap] Run-defining hyperparameters are not fully captured in persisted summary.
  - Evidence: `src/data/build_dataset.py:673` (summary content), missing `seed`, `gray_ms`, `fallback_fls_ms`, `fallback_dt_ms`, `reconcile_len_tol`.
  - Risk: exact dataset reconstruction is not guaranteed from artifacts alone.
  - Suggestion: persist full CLI/config block and script version/commit in `summary.json`.

Checks Passed:
- [INFO] No shot overlap across splits and no duplicate IDs within each split in current artifact set.
- [INFO] No non-disruptive shots with positive labels and no disruptive shots with zero positive labels in `clean_shots.csv`.
- [INFO] `positive_start_ms` stays within estimated time bounds for all reviewed shots.

Pending Next Review Trigger:
- `src/models/*.py` (not present yet).
- `reports/metrics.md` (not present yet).
