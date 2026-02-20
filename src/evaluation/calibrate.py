#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty input array")
    return arr


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = _as_1d_float(y_true)
    p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = float(len(y))
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += float(np.sum(mask)) / n * abs(acc - conf)
    return float(ece)


def calibration_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> List[Dict[str, float]]:
    y = _as_1d_float(y_true)
    p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            rows.append(
                {
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "mean_prob": float("nan"),
                    "frac_positive": float("nan"),
                    "count": 0.0,
                }
            )
            continue
        rows.append(
            {
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "mean_prob": float(np.mean(p[mask])),
                "frac_positive": float(np.mean(y[mask])),
                "count": float(np.sum(mask)),
            }
        )
    return rows


@dataclass
class ProbabilityCalibrator:
    """Probability calibrator with support for isotonic, sigmoid, and
    cross-validated isotonic (``isotonic_cv``) methods.

    ``isotonic_cv`` uses K-fold cross-validation to produce out-of-fold
    calibrated probabilities, then fits a *final* isotonic model on all
    data.  This prevents the in-sample memorization that causes ECE to
    drop to ~1e-9 on small datasets (the key failure mode observed with
    plain isotonic on ~87 calibration samples).
    """

    method: str = "isotonic"
    model: object | None = None
    cv_oof_ece: float | None = None  # out-of-fold ECE (isotonic_cv only)

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_cv_folds: int = 5,
        cv_seed: int = 42,
    ) -> "ProbabilityCalibrator":
        """Fit the calibrator.

        Args:
            y_true: Binary labels (0/1).
            y_prob: Raw predicted probabilities in [0, 1].
            n_cv_folds: Number of CV folds (only used when method='isotonic_cv').
            cv_seed: Random seed for CV splitting.
        """
        y = _as_1d_float(y_true).astype(int)
        p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = iso

        elif self.method == "isotonic_cv":
            # Cross-validated isotonic: produce out-of-fold predictions,
            # measure honest ECE, then fit final model on all data.
            n = len(y)
            oof_pred = np.full(n, np.nan, dtype=np.float64)
            actual_folds = min(n_cv_folds, n)
            if actual_folds < 2:
                # Too few samples for CV; fall back to plain isotonic
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p, y)
                self.model = iso
                self.cv_oof_ece = None
                return self

            kf = KFold(n_splits=actual_folds, shuffle=True, random_state=cv_seed)
            for train_idx, val_idx in kf.split(p):
                iso_fold = IsotonicRegression(out_of_bounds="clip")
                iso_fold.fit(p[train_idx], y[train_idx])
                oof_pred[val_idx] = iso_fold.predict(p[val_idx])

            # Clip OOF predictions
            oof_pred = np.clip(oof_pred, 1e-8, 1 - 1e-8)
            self.cv_oof_ece = float(expected_calibration_error(y, oof_pred, n_bins=15))

            # Final model on all data (for inference on new data)
            iso_final = IsotonicRegression(out_of_bounds="clip")
            iso_final.fit(p, y)
            self.model = iso_final

        elif self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(logits, y)
            self.model = lr

        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
        if self.method in ("isotonic", "isotonic_cv"):
            out = self.model.predict(p)  # type: ignore[union-attr]
        elif self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            out = self.model.predict_proba(logits)[:, 1]  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return np.clip(np.asarray(out, dtype=np.float64), 1e-8, 1 - 1e-8)

    def quality_report(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        y = _as_1d_float(y_true)
        p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
        return {
            "brier": float(brier_score_loss(y, p)),
            "ece_15_bins": float(expected_calibration_error(y, p, n_bins=15)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"method": self.method, "model": self.model}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        payload = joblib.load(path)
        out = cls(method=str(payload["method"]))
        out.model = payload["model"]
        return out


def calibration_quality_delta(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    before = {
        "brier": float(
            brier_score_loss(_as_1d_float(y_true), _as_1d_float(prob_before))
        ),
        "ece_15_bins": float(
            expected_calibration_error(y_true, prob_before, n_bins=15)
        ),
    }
    after = {
        "brier": float(
            brier_score_loss(_as_1d_float(y_true), _as_1d_float(prob_after))
        ),
        "ece_15_bins": float(expected_calibration_error(y_true, prob_after, n_bins=15)),
    }
    return {"before": before, "after": after}
