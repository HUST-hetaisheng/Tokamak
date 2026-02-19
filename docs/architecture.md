# Architecture: PGFE / (S-)CORAL / DART / SHAP / EAST Transfer

## 1. Engineering Goal

Build a real-time disruption predictor that outputs:
- disruption probability over time,
- trigger-ready warning signal,
- transfer-ready interfaces for EAST-first and later cross-device extension.

This mapping follows paper_131 as the primary prior.

## 2. Research-to-Engineering Mapping

| Research block | Engineering mapping | Current hook in repo | Status |
|---|---|---|---|
| PGFE / PGFE-U | Feature channel contract + temporal derivatives + physics-prior scaling | `train_east_realtime_sequence` supports base keys, `--add-diff`, `--add-abs-diff`, `--use-paper131-prior` | Active |
| FLS + gray zone | Dynamic/advanced labeling with uncertain interval excluded from supervised loss | `--uncertain-ms`; per-point `valid_label` in `sequence_predictions/*.csv` | Active (MVP-compatible) |
| CORAL / S-CORAL | Domain alignment loss on latent features (source-target covariance alignment) | Interface hook after encoder features, before classifier head | Placeholder (to be implemented) |
| DART classifier | Tree-based interpretable baseline (`booster='dart'`) for tabular PGFE features | Baseline entry currently `train_fusion_baseline` (RandomForest); DART adapter pending | Placeholder (planned) |
| Probability calibration | Post-hoc calibration on validation probabilities | MVP command in README saves `calibration/isotonic.joblib` | Active (runbook placeholder) |
| SHAP explainability | Global + shot-level feature attribution for calibrated model | Output contract reserved under run dir (`explainability/`) | Placeholder (planned) |

## 3. Dataflow (Current + Planned)

1. HDF5 shot data (`unified_hdf5`) + shot lists (`Disruption`, `Non-disruption`, `AdvancedTime`).
2. Feature assembly (PGFE channel set + optional diffs + prior scaling).
3. Labeling with gray-zone exclusion (`valid_label=0` excluded from loss/metrics).
4. Sequence model training and TEST evaluation.
5. Sequence probability export (`sequence_predictions/val|test/*.csv`).
6. Calibration stage on val predictions.
7. Probability curve plotting and trigger policy checks.
8. SHAP reporting (DART path) as explainability artifact.

## 4. EAST Transfer Hooks

Transfer-ready interfaces are defined as contracts, not hardcoded device logic:
- `shot_list/<device>/Disruption_<device>_TV.json`
- `shot_list/<device>/Non-disruption_<device>_TV.json`
- `shot_list/<device>/AdvancedTime_<device>.json`
- `<data_root>/<device>/unified_hdf5/<bucket>/<shot>.hdf5`

Required hooks for cross-device training:
- separate source and target loaders with identical feature schema,
- domain-adaptation insertion point (CORAL/S-CORAL loss),
- calibration per target domain,
- explainability export with device tag.

## 5. Decision Record (This Cycle)

### Why FLS + gray-zone labeling

- Fixed windows create label noise near pre-disruption transition.
- Gray-zone exclusion keeps labels cleaner for both training and calibration.
- The current `valid_label` pipeline already supports this behavior.

### Why DART + calibration + SHAP

- DART is the paper_131-aligned interpretable tree baseline for tabular PGFE features.
- Calibration is required because trigger policy depends on probability quality, not only ranking.
- SHAP is required for operator-facing physical interpretability and feature audit.

### Why transfer-ready EAST interface

- EAST is the immediate deployment target.
- Cross-device extension (e.g., EAST -> J-TEXT or inverse) should reuse the same contracts.
- Keeping loaders and adaptation hooks device-agnostic avoids rework in later migration stages.
