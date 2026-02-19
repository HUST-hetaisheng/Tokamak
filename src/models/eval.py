#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[2]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.models.calibrate import calibration_curve_points, expected_calibration_error


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tpr": tpr,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "ece_15_bins": float(expected_calibration_error(y, p, n_bins=15)),
        "brier": float(np.mean((p - y) ** 2)),
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def choose_threshold_by_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_steps: int = 300,
) -> Tuple[float, Dict[str, float]]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    best_thr = 0.5
    best_score = -1e9
    best_metrics: Dict[str, float] = {}
    for thr in np.linspace(min_threshold, max_threshold, num_steps):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 1.0
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = {"youden_j": float(score), "tpr": tpr, "fpr": fpr}
    return best_thr, best_metrics


def choose_threshold_by_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_steps: int = 300,
) -> Tuple[float, Dict[str, float]]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    best_thr = 0.5
    best_acc = -1.0
    best_tpr = -1.0
    best_fpr = 1.0
    best_metrics: Dict[str, float] = {}
    eps = 1e-12
    for thr in np.linspace(min_threshold, max_threshold, num_steps):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y, pred))
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 1.0
        better = False
        if acc > best_acc + eps:
            better = True
        elif abs(acc - best_acc) <= eps and tpr > best_tpr + eps:
            better = True
        elif abs(acc - best_acc) <= eps and abs(tpr - best_tpr) <= eps and fpr < best_fpr - eps:
            better = True
        if better:
            best_acc = acc
            best_tpr = tpr
            best_fpr = fpr
            best_thr = float(thr)
            best_metrics = {"accuracy": acc, "tpr": tpr, "fpr": fpr}
    return best_thr, best_metrics


def sustained_warning_decision(
    probs: np.ndarray,
    time_ms: np.ndarray,
    threshold: float,
    sustain_ms: float,
) -> Tuple[bool, int]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    t = np.asarray(time_ms, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return False, -1
    if len(p) == 1:
        return bool(p[0] >= threshold), 0 if p[0] >= threshold else -1
    dt = float(np.median(np.diff(t)))
    dt = max(dt, 1e-6)
    need = max(1, int(np.ceil(sustain_ms / dt)))
    hit = (p >= threshold).astype(np.int32)
    run = 0
    for i, v in enumerate(hit):
        run = run + 1 if v == 1 else 0
        if run >= need:
            return True, i
    return False, -1


def apply_shot_warning_policy(
    timeline_df: pd.DataFrame,
    threshold: float,
    sustain_ms: float,
) -> pd.DataFrame:
    required_cols = {"shot_id", "time_ms", "time_to_end_ms", "y_true", "prob_cal"}
    miss = required_cols.difference(timeline_df.columns)
    if miss:
        raise ValueError(f"timeline_df missing columns: {sorted(miss)}")

    rows: List[Dict[str, float | int | str]] = []
    for sid, g in timeline_df.groupby("shot_id"):
        gg = g.sort_values("time_ms")
        probs = gg["prob_cal"].to_numpy(dtype=np.float64)
        time_ms = gg["time_ms"].to_numpy(dtype=np.float64)
        label = int(np.max(gg["y_true"].to_numpy(dtype=int)))
        warn, idx = sustained_warning_decision(probs, time_ms, threshold, sustain_ms)
        warning_time_to_end = float("nan")
        lead_time_ms = float("nan")
        if warn and idx >= 0:
            warning_time_to_end = float(gg.iloc[idx]["time_to_end_ms"])
            if label == 1:
                lead_time_ms = float(-warning_time_to_end)
        rows.append(
            {
                "shot_id": int(sid),
                "shot_label": label,
                "warning": int(warn),
                "warning_index": int(idx),
                "warning_time_to_end_ms": warning_time_to_end,
                "lead_time_ms": lead_time_ms,
                "threshold": float(threshold),
                "sustain_ms": float(sustain_ms),
            }
        )
    return pd.DataFrame(rows).sort_values("shot_id").reset_index(drop=True)


def compute_shot_level_metrics(summary_df: pd.DataFrame) -> Dict[str, float]:
    y = summary_df["shot_label"].to_numpy(dtype=int)
    pred = summary_df["warning"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    disruptive_warned = summary_df.loc[(summary_df["shot_label"] == 1) & (summary_df["warning"] == 1)]
    return {
        "shot_accuracy": float(np.mean(y == pred)),
        "shot_tpr": tpr,
        "shot_fpr": fpr,
        "shot_tp": int(tp),
        "shot_fp": int(fp),
        "shot_tn": int(tn),
        "shot_fn": int(fn),
        "lead_time_ms_median": float(disruptive_warned["lead_time_ms"].median())
        if not disruptive_warned.empty
        else float("nan"),
        "lead_time_ms_p25": float(disruptive_warned["lead_time_ms"].quantile(0.25))
        if not disruptive_warned.empty
        else float("nan"),
        "lead_time_ms_p75": float(disruptive_warned["lead_time_ms"].quantile(0.75))
        if not disruptive_warned.empty
        else float("nan"),
        "n_shots": int(len(summary_df)),
    }


def save_calibration_curve_plot(
    y_true: np.ndarray,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    out_png: Path,
    out_csv: Path,
    n_bins: int = 15,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = calibration_curve_points(y_true, prob_raw, n_bins=n_bins)
    cal_rows = calibration_curve_points(y_true, prob_cal, n_bins=n_bins)

    merged: List[Dict[str, float | str]] = []
    for r, c in zip(raw_rows, cal_rows):
        merged.append(
            {
                "bin_left": r["bin_left"],
                "bin_right": r["bin_right"],
                "count": r["count"],
                "mean_prob_raw": r["mean_prob"],
                "frac_positive_raw": r["frac_positive"],
                "mean_prob_cal": c["mean_prob"],
                "frac_positive_cal": c["frac_positive"],
            }
        )
    pd.DataFrame(merged).to_csv(out_csv, index=False)

    def _xy(rows: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([r["mean_prob"] for r in rows], dtype=np.float64)
        y = np.array([r["frac_positive"] for r in rows], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        return x[valid], y[valid]

    x_raw, y_raw = _xy(raw_rows)
    x_cal, y_cal = _xy(cal_rows)

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.2, label="Perfect")
    if len(x_raw) > 0:
        ax.plot(x_raw, y_raw, marker="o", color="#d62728", label="Raw")
    if len(x_cal) > 0:
        ax.plot(x_cal, y_cal, marker="o", color="#1f77b4", label="Calibrated")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_probability_timeline_plot(
    shot_df: pd.DataFrame,
    threshold: float,
    sustain_ms: float,
    out_png: Path,
    title: str,
) -> Dict[str, float]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g = shot_df.sort_values("time_ms")
    probs = g["prob_cal"].to_numpy(dtype=np.float64)
    time_ms = g["time_to_end_ms"].to_numpy(dtype=np.float64)
    y = g["y_true"].to_numpy(dtype=int)

    warn, idx = sustained_warning_decision(
        probs=probs,
        time_ms=g["time_ms"].to_numpy(dtype=np.float64),
        threshold=threshold,
        sustain_ms=sustain_ms,
    )

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6), dpi=140)
    ax1.plot(time_ms, probs, color="#0b5394", linewidth=1.8, label="P(disruption|t)")
    ax1.axhline(threshold, color="#cc0000", linestyle="--", linewidth=1.2, label=f"theta={threshold:.3f}")
    ax1.set_xlabel("Time to end (ms)")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.2)
    if warn and idx >= 0:
        xw = float(time_ms[idx])
        yw = float(probs[idx])
        ax1.scatter([xw], [yw], color="#cc0000", s=30, zorder=5, label="Warning trigger")
    ax2 = ax1.twinx()
    ax2.plot(time_ms, y, color="#6a329f", alpha=0.45, linewidth=1.2, label="y(t)")
    ax2.set_ylabel("Label")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    out = {
        "warning": int(warn),
        "warning_index": int(idx),
        "warning_time_to_end_ms": float(time_ms[idx]) if warn and idx >= 0 else float("nan"),
    }
    return out


def save_metrics_json(metrics: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
