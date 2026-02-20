# Threshold Stability Report

## Executive Summary

The disruption prediction system achieves strong model quality (ROC-AUC ~0.98-0.99) but previously exhibited severe threshold instability: the operating threshold (theta) varied from 0.0166 to 0.9671 across 11 runs on the same data and similar model configurations. This report documents the root causes, implemented fixes, and the industrial validation framework now embedded in the pipeline.

**Verdict framework**: The system now produces an automated `STABLE / MARGINAL / UNSTABLE` verdict with quantitative backing.

## Problem Diagnosis

### Threshold Instability (11-run analysis)

| Problem | Evidence |
|---|---|
| Theta spans 58x range | 0.0166 (youden) to 0.9671 (sfpr<=0.01) |
| Shot TPR spans 0.13-0.97 | Entirely due to threshold selection, not model quality |
| Val set too small | Only 38 disruptive shots; 50/50 calib/thresh split -> ~19 for search |
| 1 shot flip = 5.26% TPR change | On threshold subset (19 disruptive shots) |
| Calibration overfitting | Isotonic on 87 samples -> ECE = 2.27e-9 (memorization) |
| sfpr<=0.01 has only 8 feasible thresholds | Selection is noise |
| GRU feasible_found = 0 | Falls back to heuristic |
| Advanced models: 0.0 ms lead time | 79% of warned shots fire at disruption onset |
| ROC-AUC stable (0.977-0.990) | Instability is entirely in the threshold layer |

### Lead Time = 0.0 ms (Advanced Models)

Three compounding factors:

1. **Short shots (12/27 zero-lead shots)**: Shots shorter than window_size=128 produce only 1 padded window at `time_to_end=0.0`. No earlier window exists.
2. **Coarse eval stride**: `stride=16` during evaluation skips 15 of every 16 timepoints, losing temporal resolution.
3. **Calibration squashing**: Isotonic on small samples maps raw prob 0.4-0.8 to calibrated prob 0.07-0.20, far below theta=0.767. Only raw >0.96 (disruption physically present) crosses threshold.

## Implemented Fixes

### 1. Cross-Validated Isotonic Calibration (`isotonic_cv`)

**File**: `src/evaluation/calibrate.py`

Plain isotonic regression on 87 calibration samples memorizes the training data (ECE -> 2.27e-9). The new `isotonic_cv` method uses K-fold cross-validation:
- Produces out-of-fold calibrated probabilities
- Computes honest `cv_oof_ece` metric
- Fits final model on all data for inference

Now the default for both `train_xgb.py` and `train_sequence.py`.

### 2. Fine-Grained Eval Stride (`--eval-stride`)

**File**: `src/models/advanced/train_sequence.py`

Training uses `--stride 16` for efficiency. Evaluation now uses `--eval-stride 1` (default) to produce per-timepoint predictions. This gives the model a chance to predict disruption *before* it physically occurs in the window.

**Result**: Lead time improved from 0.0 ms to 49.9 ms in smoke test.

### 3. Multi-Window Short Shots

**File**: `src/models/advanced/train_sequence.py` (`build_window_pack`)

Short shots (< window_size) previously produced only 1 window at the end. Now generate multiple sub-windows at every stride step, each ending at a different timepoint. Provides temporal resolution even for short shots.

### 4. PyTorch Model Checkpoints

**File**: `src/models/advanced/train_sequence.py`

Saves `{model}_best.pt` with model state dict, normalization parameters, and training metadata.

### 5. Threshold Stability Analysis Module

**File**: `src/evaluation/threshold_stability.py` (~530 lines)

Run with `--run-stability` flag. Produces:

#### Bootstrap Confidence Intervals (2000 iterations)
- Stratified shot-level bootstrap resampling
- 95% CI for theta, shot_tpr, shot_fpr, shot_accuracy, lead_time
- Histogram plot of theta distribution

#### Shot-Level ROC Curve
- Full theta sweep from 0.01 to 0.99
- Shot-level TPR vs FPR at each threshold
- Operating point annotation

#### Sensitivity Analysis
- Perturb theta in [-delta, +delta] around chosen operating point
- Compute gradients: d(shot_tpr)/d(theta), d(shot_fpr)/d(theta)
- Identify stable region where FPR constraint remains satisfied

#### Leave-One-Out Cross-Validation
- Drop each shot, re-select theta, measure prediction accuracy
- Quantifies single-shot sensitivity
- Theta range across all folds

#### Operational Envelope
- 2D heatmap: (theta, sustain_ms) -> shot_tpr | shot_fpr
- Shows the full operating space available to operators

### Automated Verdict

The system scores 6 criteria and produces:

| Score | Verdict |
|---|---|
| >= 6 | STABLE |
| >= 2 | MARGINAL |
| < 2 | UNSTABLE |

Criteria:
- Bootstrap 95% CI width (< 0.10: +2, < 0.20: +1, else: -2)
- Theta coefficient of variation (< 0.10: +2, < 0.20: +1, else: -1)
- Shot TPR 95% CI width (< 0.15: +1, > 0.30: -2)
- Stable region width (> 0.15: +2, > 0.05: +1, else: -2)
- LOO-CV accuracy (>= 0.90: +2, >= 0.80: +1, else: -1)
- LOO theta range (< 0.10: +2, < 0.25: +1, else: -1)

## Usage

### XGBoost Baseline with Stability
```bash
python src/models/baseline/train_xgb.py \
    --run-stability \
    --stability-n-boot 2000 \
    --calibration-method isotonic_cv
```

### Advanced Models with Stability
```bash
python src/models/advanced/train_sequence.py \
    --run-stability \
    --stability-n-boot 2000 \
    --eval-stride 1 \
    --calibration-method isotonic_cv \
    --models mamba_lite
```

## Output Artifacts

When `--run-stability` is enabled:

| Artifact | Description |
|---|---|
| `stability/stability_report.json` | Full numerical results + verdict |
| `stability/shot_level_roc.csv` | Shot-level ROC data |
| `stability/sensitivity_sweep.csv` | Sensitivity sweep data |
| `stability/operational_envelope.csv` | Envelope grid data |
| `stability/loocv_details.csv` | Per-fold LOO results |
| `stability/stability_plots/bootstrap_theta_hist.png` | Bootstrap distribution |
| `stability/stability_plots/shot_level_roc.png` | Shot-level ROC curve |
| `stability/stability_plots/sensitivity_sweep.png` | Sensitivity sweep plot |
| `stability/stability_plots/operational_envelope.png` | Envelope heatmap |

The `stability` key is also appended to `metrics_summary.json`.
