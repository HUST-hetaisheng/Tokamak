# Multi-Agent Progress Board

Last Updated: 2026-02-19

## Agent-1 (Researcher / Learner)
Status: relaunched (in_progress)
Done:
- Read `ref/paper_131.txt` and project markdowns relevant to mechanism extraction and method chain summary.
- Produced mechanism taxonomy and observable feature relations (density limit, locked mode, VDE/low-q, impurity/radiation pathways).
- Completed method-chain summary for PGFE, FLS, (S-)CORAL, DART, SHAP, and EFD with reproducibility notes.
- Kept existing research docs intact and added an appendix mapping output artifacts to physics-interpretability concepts in `docs/requirements.md`.
Next:
- Maintain the artifact-to-interpretability mapping as Agent-2/Agent-3 outputs evolve.
- Revise terminology/citations if reviewer feedback requires stricter traceability.
Blockers:
- None.
Artifacts:
- `docs/literature_review.md`
- `docs/requirements.md`
- `docs/feature_physics_map.md`

## Agent-2 (Data Engineer)
Status: completed
Done:
- Detected data roots in required order and selected `D:/Fuison/data/J-TEXT/unified_hdf5`.
- Located J-TEXT shot-list and advanced-time metadata files under `shot_list/J-TEXT/`.
- Implemented reproducible audit/clean/split pipeline: `src/data/build_dataset.py`.
- Ran bounded MVP build and generated small artifacts under `artifacts/datasets/jtext_v1/`.
- Built stratified shot splits (8/1/1) into `splits/train.txt`, `splits/val.txt`, `splits/test.txt`.
- Produced data audit report and 3-shot `y(t)` plots in `reports/data_audit.md` and `reports/plots/data_audit/`.
- Clarified in `reports/data_audit.md` that 3-shot `y(t)` plots are sample audit visualizations only; full test timelines are in `reports/plots/probability_timelines_test.csv` (with rendered subset in `reports/plots/probability/`).
- Added explicit train/val/test shot-count references in `reports/data_audit.md` and `artifacts/datasets/jtext_v1/summary.json`, with split-id sources in `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.
Next:
- Hand off split/artifact outputs to Agent-3 for downstream training.
- Extend pipeline options if reviewer/modeling feedback requires alternate ratios or stricter filters.
Blockers:
- None.
Artifacts:
- `src/data/build_dataset.py`
- `artifacts/datasets/jtext_v1/summary.json`
- `artifacts/datasets/jtext_v1/stats.json`
- `artifacts/datasets/jtext_v1/small_sample.npz`
- `artifacts/datasets/jtext_v1/clean_shots.csv`
- `artifacts/datasets/jtext_v1/excluded_shots.csv`
- `artifacts/datasets/jtext_v1/class_weights.json`
- `artifacts/datasets/jtext_v1/required_features.json`
- `artifacts/datasets/jtext_v1/example_shots.csv`
- `artifacts/datasets/jtext_v1/label_examples.csv`
- `splits/train.txt`
- `splits/val.txt`
- `splits/test.txt`
- `reports/data_audit.md`

## Agent-3 (Modeler / Experimenter)
Status: completed
Done:
- Added train CLI plotting controls `--plot-shot-limit` and `--plot-all-test-shots` to resolve limited timeline exports.
- Added validation threshold objective selection via `--threshold-objective {youden,accuracy,shot_fpr_constrained}`.
- Ran continuation training on full split sizes (train=1386, val=174, test=173); validation was split into calibration=87 and threshold=87 shots.
- Kept 23-feature use-all-by-default policy (23/23) and persisted it in `training_config.json`.
- Produced `173` probability timeline PNG files for test shots.
- Produced `38` per-disruptive-shot reason rows (expected disruptive shots in TEST: 38).
- Current test metrics: accuracy=0.990885, roc_auc=0.978437, shot_accuracy=0.953757, threshold=0.668796 (shot_fpr_constrained).
Next:
- Coordinate with reviewer on threshold objective trade-offs and calibration holdout strategy.
Blockers:
- None.
Artifacts:
- `src/models/train.py`
- `src/models/eval.py`
- `src/models/calibrate.py`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/model_xgb_dart.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/calibrator.joblib`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/training_config.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/metrics_summary.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/shap_topk.csv`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/warning_summary_test.csv`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_per_shot.csv`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/metrics.md`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/calibration_curve_test.png`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/probability_timelines_test.csv`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/probability`
## Agent-4 (Reviewer / Maintainer)
Status: in_progress
Done:
- Completed initial repo scan and risk checklist setup.
- Added first dated entry in `docs/code_review_log.md`.
- Reviewed Agent-2 pipeline and metadata artifacts: `src/data/build_dataset.py`, `reports/data_audit.md`, `artifacts/datasets/jtext_v1/*`, and `splits/*.txt`.
- Appended severity-tagged findings with file/function references in `docs/code_review_log.md`.
- Reviewed Agent-3 outputs: `src/models/train.py`, `src/models/eval.py`, `src/models/calibrate.py`, `artifacts/models/best/metrics_summary.json`, `reports/metrics.md`.
- Appended severity-ranked Agent-3 findings (threshold-policy mismatch, calibration overfitting risk, label-integrity checks, and reproducibility gaps) to `docs/code_review_log.md`.
- Reviewed Agent-3 relaunch delta and appended severity-ranked findings on threshold-objective correctness, calibration validity bias, label-integrity leakage guard gaps, and plotting behavior diagnostics in `docs/code_review_log.md`.
Next:
- Verify Agent-3 remediation with rerun artifacts, prioritizing threshold search aligned to shot-level FAR targets and calibration/threshold split isolation.
- Recheck plotting outputs for representative TP/FP/TN/FN timelines and corrected calibration-bin count reporting.
Blockers:
- Waiting on Agent-3 remediation updates for the relaunch findings logged in `docs/code_review_log.md`.
Artifacts:
- `docs/code_review_log.md`

## Agent-5 (Technical Writer)
Status: completed
Done:
- Added `README.md` Artifact Guide (path + role + when to read) for relaunch-critical files.
- Explicitly documented why only a few shot PNGs exist by default and pointed to full 173-shot CSV artifacts.
- Updated full-shot rerun command in `README.md` with current train flags (`--max-train-shots 0`, `--max-val-shots 0`, `--max-test-shots 0`, `--plot-all-test-shots`, `--threshold-objective accuracy`).
- Added relaunch stage notes to `docs/changelog.md`.
Next:
- Keep artifact pointers synchronized with future EAST/J-TEXT reruns and output path changes.
Blockers:
- None.
Artifacts:
- `README.md`
- `docs/changelog.md`
- `docs/progress.md`

## Agent-6 (Advanced Modeler)
Status: completed
Done:
- Added `src/models/sequence_arch.py` with `TemporalTransformerClassifier`, `MambaLiteClassifier`, and `GRUClassifier`.
- Added `src/models/train_advanced.py` with bounded advanced training sweep, calibration, shot-level thresholding, probability-timeline export, and gradient*input reasons.
- Executed 3 fair-window runs (`window_size=128`, `stride=16`): transformer_small / mamba_lite / gru.
- Best run: `adv_mamba_lite_ws128_st16_e5_s42` with test_acc=0.988993, roc_auc=0.990111, shot_acc=0.976879, shot_fpr=0.000000.
Next:
- Scale to larger window/horizon ablations and add cross-device transfer hooks on the same architecture backbone.
Blockers:
- None.
Artifacts:
- `src/models/sequence_arch.py`
- `src/models/train_advanced.py`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/plots/probability`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/plots/probability`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_gru_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_gru_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_gru_ws128_st16_e5_s42/plots/probability`
- `reports/iters/advanced_summary.csv`
- `reports/iters/advanced_summary.md`