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

## 2026-02-19 Review Cycle: Agent-3 Model/Eval/Calibration

Scope:
- Reviewed `src/models/train.py`.
- Reviewed `src/models/eval.py`.
- Reviewed `src/models/calibrate.py`.
- Reviewed `artifacts/models/best/metrics_summary.json`.
- Reviewed `reports/metrics.md`.

Findings (Severity-Ordered):
- [HIGH][Metric/Warning Policy Misalignment] Threshold is selected via point-level Youden J, producing an aggressively low operating point with high false alarms in deployment-style metrics.
  - Evidence: `src/models/train.py:606`, `src/models/eval.py:59`, `artifacts/models/best/metrics_summary.json:124`, `artifacts/models/best/metrics_summary.json:129`, `artifacts/models/best/metrics_summary.json:142`, `reports/metrics.md:25`, `reports/metrics.md:30`.
  - Observed impact: `theta=0.019833`, timepoint `fpr=0.194614`, shot-level `shot_fpr=0.288889`.
  - Risk: warning stream reliability is low (false alarms are operationally expensive), despite strong recall.
  - Suggestion: select threshold using shot-level objective under explicit FAR cap (for example optimize lead-time subject to `shot_fpr <= target`) rather than point-level Youden J alone.

- [HIGH][Calibration Overfitting Risk] Isotonic calibration is fit and assessed on the same validation set, and that same calibrated validation distribution is used to choose threshold.
  - Evidence: `src/models/train.py:601`, `src/models/train.py:604`, `src/models/train.py:606`, `artifacts/models/best/metrics_summary.json:152`, `artifacts/models/best/metrics_summary.json:159`.
  - Observed signal: validation ECE collapses to near-zero (`2.499e-09`), which is optimistic when measured on the calibration-fit split.
  - Risk: calibration quality may be overstated; deployment threshold can be biased.
  - Suggestion: use a dedicated calibration holdout (or cross-fitted calibrator) separate from threshold-selection data, and report the holdout/test calibration as primary.

- [MEDIUM][Label Integrity Drift Risk] Training ignores the per-shot metadata label read from HDF5 at load time and trusts only `clean_shots.csv` labels.
  - Evidence: `src/models/train.py:216`, `src/models/train.py:253`, `src/models/train.py:530`.
  - Risk: if HDF5 content drifts after dataset build, silent label mismatch can contaminate training/evaluation.
  - Suggestion: assert `meta_label == expected_label` during training data load and hard-fail with shot IDs on mismatch.

- [MEDIUM][Time/Label Protocol Inconsistency] `train.py` uses `advanced_ms` whenever present (including nonpositive values), while dataset build logic treated nonpositive advanced-time as fallback.
  - Evidence: `src/models/train.py:172`, `src/data/build_dataset.py:243`.
  - Risk: future metadata anomalies (`advanced_time <= 0`) could create invalid positive windows and unstable metrics.
  - Suggestion: mirror dataset-build condition (`advanced_ms is not None and advanced_ms > 0`) in training/eval label generation.

- [MEDIUM][Reproducibility Gap] Run determinism is partial; multi-thread XGBoost and incomplete run fingerprinting can produce irreproducible deltas across environments.
  - Evidence: `src/models/train.py:84`, `src/models/train.py:327`, `src/models/train.py:683`.
  - Risk: repeated runs may differ subtly without traceable cause.
  - Suggestion: add deterministic mode options (`n_jobs=1` toggle for reproducibility runs), and persist library versions + split file hashes in `training_config.json`.

Checks Passed:
- [INFO] Calibration is applied after model training and fitted on validation only (no direct test-fit leakage detected).
- [INFO] Test set remains evaluation-only for final reported metrics in this run.

## 2026-02-19 Review Cycle: Agent-3 Relaunch Delta (Threshold/Leakage/Calibration/Plotting)

Scope:
- Re-reviewed `src/models/train.py`, `src/models/eval.py`, and `src/models/calibrate.py`.
- Re-reviewed relaunch artifacts: `artifacts/models/best/metrics_summary.json`, `artifacts/models/best/calibration_curve_points_test.csv`, `artifacts/models/best/warning_summary_test.csv`, `reports/metrics.md`, and `reports/plots/probability_timelines_test.csv`.
- Replayed shot-policy outcomes from exported test timelines using current sustained-warning logic.

Findings (Severity-Ordered):
- [HIGH][Threshold-Objective Incorrectness] Threshold search is still point-level Youden J and is decoupled from the deployed shot-warning objective and FAR control.
  - Evidence: `src/models/eval.py:59`, `src/models/train.py:606`, `src/models/train.py:618`, `artifacts/models/best/metrics_summary.json:163`, `artifacts/models/best/metrics_summary.json:142`.
  - Observed impact (policy replay from `reports/plots/probability_timelines_test.csv` with `src/models/eval.py:84` logic): current `theta=0.019833` gives `shot_fpr=0.288889`, while `theta=0.100000` gives `shot_fpr=0.051852` with `shot_tpr=0.921053` and materially higher `shot_accuracy`.
  - Risk: warning stream operating point is misaligned with deployment reliability targets.
  - Suggestion: replace Youden-only search with shot-policy optimization under explicit FAR cap (for example optimize lead-time/TPR with constraint `shot_fpr <= target`).

- [HIGH][Calibration Validity Bias] Isotonic calibrator fitting, calibration quality reporting, and threshold selection still use the same validation distribution.
  - Evidence: `src/models/train.py:601`, `src/models/train.py:604`, `src/models/train.py:606`, `artifacts/models/best/metrics_summary.json:118`, `artifacts/models/best/metrics_summary.json:159`.
  - Observed signal: validation ECE collapses to near-zero (`2.499e-09`) on the same split used to fit isotonic and choose threshold.
  - Risk: calibration and threshold quality are optimistic; operating point may not transfer robustly.
  - Suggestion: use a separate calibration holdout or cross-fitting, and perform threshold search on data not used to fit calibrator parameters.

- [MEDIUM][Label-Integrity / Leakage Guard Missing] HDF5 disruption label is read but discarded, and no consistency assertion is enforced against `clean_shots.csv`.
  - Evidence: `src/models/train.py:209`, `src/models/train.py:216`, `src/models/train.py:253`, `src/models/train.py:530`.
  - Risk: silent metadata drift can contaminate training/evaluation labels without detection.
  - Suggestion: assert `meta_label == expected_label` per shot and hard-fail on mismatch with shot IDs.

- [MEDIUM][Plotting Representativeness Gap] Timeline plot selection logic deterministically picks first three disruptive warned shots, excluding negative/false-alarm examples.
  - Evidence: `src/models/train.py:632`, `src/models/train.py:635`, `reports/plots/probability/shot_1051510_timeline.png`, `reports/plots/probability/shot_1051684_timeline.png`, `reports/plots/probability/shot_1051701_timeline.png`, `artifacts/models/best/warning_summary_test.csv:2`.
  - Risk: visualization overstates qualitative behavior and hides false-positive dynamics that dominate current FAR concerns.
  - Suggestion: export a fixed representative set (for example TP/FP/TN/FN or highest-risk non-disruptive shots) instead of first-3-by-sort.

- [MEDIUM][Calibration Plot CSV Inconsistency] `calibration_curve_points_test.csv` logs `count` from raw-probability bins only; calibrated-bin support is not recorded and can differ substantially.
  - Evidence: `src/models/eval.py:189`, `src/models/eval.py:195`, `artifacts/models/best/calibration_curve_points_test.csv:1`, `reports/plots/probability_timelines_test.csv:1`.
  - Observed signal: raw vs calibrated bin populations differ across all 15 bins in this run.
  - Risk: downstream readers may misinterpret calibrated reliability points with incorrect bin support.
  - Suggestion: write separate `count_raw` and `count_cal` columns (or separate tables) and use each series' own counts.

Checks Passed:
- [INFO] No direct train/test leakage path introduced in relaunch: model fit uses train split; calibrator fit excludes test split; test remains evaluation-only.
