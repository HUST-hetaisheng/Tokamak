#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


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
    method: str = "isotonic"
    model: object | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        y = _as_1d_float(y_true).astype(int)
        p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = iso
        elif self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(logits, y)
            self.model = lr
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        p = np.clip(_as_1d_float(y_prob), 1e-8, 1 - 1e-8)
        if self.method == "isotonic":
            out = self.model.predict(p)  # type: ignore[union-attr]
        elif self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            out = self.model.predict_proba(logits)[:, 1]  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return np.clip(np.asarray(out, dtype=np.float64), 1e-8, 1 - 1e-8)

    def quality_report(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
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
        "brier": float(brier_score_loss(_as_1d_float(y_true), _as_1d_float(prob_before))),
        "ece_15_bins": float(expected_calibration_error(y_true, prob_before, n_bins=15)),
    }
    after = {
        "brier": float(brier_score_loss(_as_1d_float(y_true), _as_1d_float(prob_after))),
        "ece_15_bins": float(expected_calibration_error(y_true, prob_after, n_bins=15)),
    }
    return {"before": before, "after": after}

