# AGENTS.md

## Default Data Source (Codex)
- Default data root for this repository is `G:\我的云端硬盘\Fuison\data`.
- Unless user explicitly overrides, always use this path as the primary data source.
- If a script requires a relative data path, resolve from this root first.
- Do not switch to any other data source automatically.

## Project Goal And Scope
- The primary goal of this repository is to build an innovative and feasible real-time tokamak plasma disruption prediction system.
- The system should output real-time disruption probability, disruption type classification, and early warning signals.
- The disruption vs non-disruption classification accuracy target is at least 95%.
- The key priority is visualization of the disruption process and reliable real-time disruption probability output.
- The research direction should support cross-device transfer learning in later stages (for example from EAST to J-TEXT datasets).
- `D:\Fuison\docs\literature\paper_131.txt` is a core prior-knowledge source and industry research reference for this project.
- Project modeling, feature design, and evaluation strategy should be grounded in this paper unless the user explicitly requests a different direction.
- Out-of-scope files like the lithium battery notebooks have been removed.
- `artifacts/` and `reports/` are reserved for generated output artifacts.

## Repository Guidelines

### Project Structure & Module Organization
- `src/models/baseline/train_xgb.py` is the main entry point for the XGBoost baseline.
- `src/models/advanced/train_sequence.py` is the main entry point for Mamba, Transformer, and GRU models.
- `src/evaluation/` contains shared evaluation, calibration, and report generation logic.
- `src/data/build_dataset.py` handles data preprocessing and train/val/test splitting.
- `shot_list/EAST/*.json` and `shot_list/J-TEXT/*.json` define disruptive/non-disruptive shot splits.
- `data/EAST/unified_hdf5/<bucket>/<shot>.hdf5` stores source time-series signals and metadata.
- `artifacts/` contains generated models, calibration joblibs, and `metrics_summary.json`.

### Build, Test, and Development Commands
- Create environment: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- Install dependencies: `pip install numpy pandas h5py scikit-learn torch xgboost shap matplotlib joblib`
- Smoke training run for baseline:
  `python src/models/baseline/train_xgb.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/best`
- Smoke training run for advanced:
  `python src/models/advanced/train_sequence.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/iters/adv_mamba_lite --models mamba_lite`
- Build dataset: `python src/data/build_dataset.py --repo-root . --dataset-name jtext_v1`

### Coding Style & Naming Conventions
- Use Python with 4-space indentation, explicit type hints for new functions, and `snake_case` naming.
- Keep module constants in `UPPER_SNAKE_CASE` (for example `TIME_KEYS`, `PHYSICS_MAP`).
- Prefer `pathlib.Path` for filesystem operations.
- Save scripts in UTF-8 and avoid introducing mixed-encoding comments or literals.
- Class names use `PascalCase` (for example `ProbabilityCalibrator`, `ShotSeries`).
- Dataclass names use `PascalCase`; fields use `snake_case`.
- Private helper functions prefix with underscore: `_as_1d_float`, `_xy`.

### Testing Guidelines
- No formal `pytest` suite exists yet; use reproducible smoke runs as baseline validation.
- A change is minimally validated when training completes and regenerates `metrics_summary.json` and `warning_summary_test.csv`.

### Imports Organization
Organize imports in three blocks, sorted alphabetically within each:
```python
# 1. Standard library
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 2. Third-party libraries
import numpy as np
import pandas as pd
import torch
from torch import nn

# 3. Local imports (absolute from src.*)
from src.evaluation.calibrate import ProbabilityCalibrator
from src.evaluation.eval import compute_binary_metrics
```

### Type Annotations
- Always add type hints to function signatures for new code.
- Use `Optional[T]` for nullable parameters/returns.
- Use `Sequence[T]` for read-only list-like inputs, `List[T]` for mutable.
- Use `Mapping[K, V]` for read-only dict-like inputs, `Dict[K, V]` for mutable.
- Use `np.ndarray` with shape hints in docstrings when helpful.
- Use `float | None` union syntax for Python 3.10+ style optional returns.

### Error Handling
- Raise specific exceptions with informative messages: `ValueError`, `FileNotFoundError`, `RuntimeError`.
- Prefer early validation at function entry (guard clauses).
- Use `try/except` for I/O operations (HDF5, JSON, file system).
- Log or propagate errors with context; avoid silent failures.
- Example: `raise FileNotFoundError(f"HDF5 root not found: {hdf5_root}")`

### Documentation
- Add docstrings for public functions and classes (triple-double-quotes).
- Include parameter descriptions and return types in Google style.
- Document side effects (file writes, global state changes).
- Keep docstrings concise; prefer self-documenting code.
- Example:
  ```python
  def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
      """Compute binary classification metrics at a given threshold."""
  ```

### Function Design Patterns
- Use `@dataclass` for simple data containers (e.g., `ShotSeries`, `WindowPack`, `SplitPack`).
- Return tuples for multiple related values; use named tuples or dataclasses for clarity.
- Prefer pure functions (no side effects) for data transformations.
- Separate I/O from computation: load data, process, then write results.

### Physics-Guided Feature Mapping
- `PHYSICS_MAP` in `train_xgb.py` maps features to disruption mechanisms:
  - `density_limit`: ne_nG, CIII, sxr_*, xuv_*, Mir_avg_amp, Mir_VV, v_loop
  - `locked_mode`: Mir_avg_fre, Mir_avg_amp, Mir_VV, mode_number_n, n=1 amplitude, MNM
  - `low_q_current_limit`: qa_proxy, ip, Bt, mode_number_n, Mir_avg_amp
  - `vde_control_loss`: Z_proxy, dx_a, dy_a, v_loop
  - `impurity_radiation_collapse`: CIII, v_loop, sxr_*, xuv_*
- When adding new features, consider mapping them to relevant mechanisms.

### Labeling Strategy
- Gray zone: configurable via `--gray-ms` (default 30.0 ms before positive start).
- FLS (Flexible Labeling Strategy): uses `AdvancedTime` if available, else fallback.
- Positive label window: from `t_end - fls_ms` to `t_end` for disruptive shots.
- Non-disruptive shots: all labels are 0, no gray zone exclusion.

### Evaluation Metrics
- Primary: ROC-AUC, PR-AUC, accuracy, TPR, FPR at chosen threshold.
- Shot-level: shot_accuracy, shot_tpr, shot_fpr, median lead time.
- Calibration: ECE (Expected Calibration Error), Brier score.
- Threshold selection: Youden J, accuracy maximization, or shot-FPR constrained.

## Output Artifacts
- `metrics_summary.json`: Core metrics for the run.
- `warning_summary_test.csv`: Per-shot warning decisions.
- `disruption_reason_per_shot.csv`: Per-shot mechanism attribution.
- `probability_timelines_test.csv`: Full time-point predictions.
- `calibrator.joblib`: Fitted probability calibrator.
- `training_config.json`: Full training configuration and metadata.

## Commit & Pull Request Guidelines
- This workspace uses Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`).
- PRs should include purpose, exact command(s), key hyperparameters, and before/after metrics.
- Avoid committing large generated artifacts or raw HDF5 data (`artifacts/**`, `data/**`) unless explicitly required.
- Use descriptive commit messages that explain the "why" and "what" of the change.