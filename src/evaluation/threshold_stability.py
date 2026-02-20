#!/usr/bin/env python3
"""Industrial-grade threshold stability analysis for tokamak disruption prediction.

This module provides statistical tools to validate that the chosen operating
threshold is robust enough for deployment.  The key output is a stability
report containing:

- Bootstrap confidence intervals for the threshold and all shot-level metrics
- A full shot-level ROC curve (theta sweep) with operating-point annotation
- Sensitivity analysis:  how much do metrics move per unit theta change?
- Leave-one-out cross-validation at the shot level
- An operational envelope heatmap:  (theta, sustain_ms) -> shot_tpr | shot_fpr

Design principles
-----------------
1. **Prove a stable region, not just a single point.**  If theta +/- 0.05
   still satisfies the FPR constraint and keeps TPR >= X, the deployment
   argument is far stronger than reporting one theta.
2. **Quantify statistical uncertainty.**  With only ~38 disruptive val
   shots, one shot flip = 2.6% TPR change.  Bootstrap CI exposes this.
3. **Separate model quality from threshold selection noise.**  ROC-AUC
   is consistently ~0.98; the instability is entirely in the threshold
   layer operating on small samples.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.eval import (
    apply_shot_warning_policy,
    compute_shot_level_metrics,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""

    theta_mean: float
    theta_std: float
    theta_ci_lower: float
    theta_ci_upper: float
    theta_samples: List[float]
    metric_ci: Dict[str, Dict[str, float]]  # metric -> {mean, std, ci_lo, ci_hi}
    n_boot: int
    n_shots_per_sample: int
    n_disruptive_per_sample_mean: float


@dataclass
class SensitivityResult:
    """Result of threshold sensitivity (perturbation) analysis."""

    theta_center: float
    delta: float
    sweep_df: (
        pd.DataFrame
    )  # columns: theta, shot_tpr, shot_fpr, shot_accuracy, lead_time_ms_median
    gradient_at_center: Dict[str, float]  # d(metric)/d(theta) at center
    stable_region: Dict[str, float]  # theta_lo, theta_hi where constraints hold


@dataclass
class StabilityReport:
    """Aggregated stability report for the operating point."""

    bootstrap: BootstrapResult
    sensitivity: SensitivityResult
    shot_roc: pd.DataFrame
    loocv_metrics: Dict[str, float]
    verdict: str  # "STABLE", "MARGINAL", "UNSTABLE"
    verdict_reasons: List[str]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# 1.  Bootstrap confidence interval for theta and shot-level metrics
# ---------------------------------------------------------------------------


def bootstrap_threshold_ci(
    timeline_df: pd.DataFrame,
    sustain_ms: float,
    max_shot_fpr: float,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap the full threshold-selection pipeline at the shot level.

    On each iteration, resample shots *with replacement* (stratified by
    label), re-run ``choose_threshold_by_shot_fpr`` on the resampled
    timeline, and record the selected theta + resulting metrics.

    Args:
        timeline_df: Full validation timeline with columns
            {shot_id, time_ms, time_to_end_ms, y_true, prob_cal}.
        sustain_ms: Sustained-warning window length in ms.
        max_shot_fpr: Maximum tolerable shot-level FPR.
        n_boot: Number of bootstrap iterations.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        BootstrapResult with theta distribution and metric CIs.
    """
    rng = np.random.RandomState(seed)

    # Identify unique shots and their labels
    shot_info = (
        timeline_df.groupby("shot_id")["y_true"]
        .max()
        .reset_index()
        .rename(columns={"y_true": "shot_label"})
    )
    pos_shots = shot_info.loc[shot_info["shot_label"] == 1, "shot_id"].tolist()
    neg_shots = shot_info.loc[shot_info["shot_label"] == 0, "shot_id"].tolist()

    theta_samples: List[float] = []
    metric_samples: Dict[str, List[float]] = {
        "shot_tpr": [],
        "shot_fpr": [],
        "shot_accuracy": [],
        "lead_time_ms_median": [],
    }
    n_disruptive_counts: List[int] = []

    for _ in range(n_boot):
        # Stratified resample with replacement
        boot_pos = list(rng.choice(pos_shots, size=len(pos_shots), replace=True))
        boot_neg = list(rng.choice(neg_shots, size=len(neg_shots), replace=True))
        boot_shots = boot_pos + boot_neg

        # Build resampled timeline (need unique shot_id to avoid groupby collision)
        frames: List[pd.DataFrame] = []
        for new_idx, original_sid in enumerate(boot_shots):
            chunk = timeline_df[timeline_df["shot_id"] == original_sid].copy()
            chunk["shot_id"] = new_idx  # unique synthetic ID
            frames.append(chunk)
        boot_timeline = pd.concat(frames, ignore_index=True)

        # Run threshold selection on resampled data
        try:
            from src.evaluation.eval import choose_threshold_by_shot_fpr

            theta, diag = choose_threshold_by_shot_fpr(
                timeline_df=boot_timeline,
                sustain_ms=sustain_ms,
                max_shot_fpr=max_shot_fpr,
            )
        except Exception:
            continue

        # Evaluate on the *same* resampled data (in-sample bootstrap metrics)
        summary = apply_shot_warning_policy(
            boot_timeline, threshold=theta, sustain_ms=sustain_ms
        )
        sm = compute_shot_level_metrics(summary)

        theta_samples.append(float(theta))
        n_disruptive_counts.append(len(boot_pos))
        for k in metric_samples:
            v = float(sm.get(k, float("nan")))
            metric_samples[k].append(v if np.isfinite(v) else float("nan"))

    if not theta_samples:
        raise RuntimeError("All bootstrap iterations failed")

    alpha = (1.0 - ci_level) / 2.0
    theta_arr = np.array(theta_samples)

    metric_ci: Dict[str, Dict[str, float]] = {}
    for k, vals in metric_samples.items():
        arr = np.array(vals)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            metric_ci[k] = {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
            continue
        metric_ci[k] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "ci_lower": float(np.percentile(valid, 100 * alpha)),
            "ci_upper": float(np.percentile(valid, 100 * (1 - alpha))),
        }

    return BootstrapResult(
        theta_mean=float(np.mean(theta_arr)),
        theta_std=float(np.std(theta_arr)),
        theta_ci_lower=float(np.percentile(theta_arr, 100 * alpha)),
        theta_ci_upper=float(np.percentile(theta_arr, 100 * (1 - alpha))),
        theta_samples=[float(x) for x in theta_arr],
        metric_ci=metric_ci,
        n_boot=len(theta_samples),
        n_shots_per_sample=len(pos_shots) + len(neg_shots),
        n_disruptive_per_sample_mean=float(np.mean(n_disruptive_counts)),
    )


# ---------------------------------------------------------------------------
# 2.  Shot-level ROC curve (full theta sweep)
# ---------------------------------------------------------------------------


def shot_level_roc_curve(
    timeline_df: pd.DataFrame,
    sustain_ms: float,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_steps: int = 200,
) -> pd.DataFrame:
    """Compute shot-level (TPR, FPR, accuracy, lead_time) at each theta.

    This provides a complete operating-point trade-off surface that an
    operations team can use to pick a deployment point.

    Returns:
        DataFrame with one row per theta value.
    """
    rows: List[Dict[str, float]] = []
    for thr in np.linspace(min_threshold, max_threshold, num_steps):
        summary = apply_shot_warning_policy(
            timeline_df=timeline_df,
            threshold=float(thr),
            sustain_ms=sustain_ms,
        )
        sm = compute_shot_level_metrics(summary)
        row = {"theta": float(thr)}
        for k in (
            "shot_tpr",
            "shot_fpr",
            "shot_accuracy",
            "lead_time_ms_median",
            "lead_time_ms_p25",
            "lead_time_ms_p75",
            "shot_tp",
            "shot_fp",
            "shot_tn",
            "shot_fn",
            "n_shots",
        ):
            row[k] = float(sm.get(k, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3.  Sensitivity analysis (perturbation around the operating point)
# ---------------------------------------------------------------------------


def threshold_sensitivity_analysis(
    timeline_df: pd.DataFrame,
    theta_center: float,
    sustain_ms: float,
    delta: float = 0.10,
    num_steps: int = 41,
    max_shot_fpr: float = 0.02,
) -> SensitivityResult:
    """Quantify how metrics change as theta is perturbed around the center.

    The output includes:
    - A fine-grained sweep around theta +/- delta
    - Numerical gradient d(metric)/d(theta) at the center
    - The maximal stable region [theta_lo, theta_hi] where shot_fpr <= constraint

    Args:
        timeline_df: Validation or test timeline.
        theta_center: The chosen operating threshold.
        sustain_ms: Sustained-warning window.
        delta: Half-width of the perturbation window.
        num_steps: Number of theta values in the sweep.
        max_shot_fpr: FPR constraint for stable-region identification.

    Returns:
        SensitivityResult.
    """
    lo = max(0.01, theta_center - delta)
    hi = min(0.99, theta_center + delta)
    sweep = shot_level_roc_curve(
        timeline_df=timeline_df,
        sustain_ms=sustain_ms,
        min_threshold=lo,
        max_threshold=hi,
        num_steps=num_steps,
    )

    # Numerical gradient at center (central difference)
    eps = max(0.005, (hi - lo) / (num_steps - 1))
    gradient: Dict[str, float] = {}
    for metric in ("shot_tpr", "shot_fpr", "shot_accuracy", "lead_time_ms_median"):
        idx_lo = (sweep["theta"] - (theta_center - eps)).abs().idxmin()
        idx_hi = (sweep["theta"] - (theta_center + eps)).abs().idxmin()
        v_lo = float(sweep.loc[idx_lo, metric])
        v_hi = float(sweep.loc[idx_hi, metric])
        d_theta = float(sweep.loc[idx_hi, "theta"] - sweep.loc[idx_lo, "theta"])
        if abs(d_theta) > 1e-9 and np.isfinite(v_lo) and np.isfinite(v_hi):
            gradient[f"d_{metric}_d_theta"] = float((v_hi - v_lo) / d_theta)
        else:
            gradient[f"d_{metric}_d_theta"] = float("nan")

    # Find stable region where shot_fpr <= constraint
    feasible_mask = sweep["shot_fpr"] <= max_shot_fpr + 1e-9
    if feasible_mask.any():
        feasible_rows = sweep[feasible_mask]
        stable_lo = float(feasible_rows["theta"].min())
        stable_hi = float(feasible_rows["theta"].max())
        stable_width = stable_hi - stable_lo
    else:
        stable_lo = float("nan")
        stable_hi = float("nan")
        stable_width = 0.0

    stable_region = {
        "theta_lo": stable_lo,
        "theta_hi": stable_hi,
        "stable_width": stable_width,
        "max_shot_fpr_constraint": max_shot_fpr,
    }

    return SensitivityResult(
        theta_center=theta_center,
        delta=delta,
        sweep_df=sweep,
        gradient_at_center=gradient,
        stable_region=stable_region,
    )


# ---------------------------------------------------------------------------
# 4.  Leave-one-out cross-validation (shot level)
# ---------------------------------------------------------------------------


def leave_one_out_cv(
    timeline_df: pd.DataFrame,
    sustain_ms: float,
    max_shot_fpr: float,
) -> Dict[str, Any]:
    """Leave-one-shot-out cross-validation for threshold robustness.

    For each shot, remove it from the val set, re-select the optimal
    threshold on the remaining shots, then evaluate whether the
    held-out shot would be correctly classified.

    This directly answers: "If we had one fewer (or different) shot in
    validation, would the threshold change meaningfully?"

    Returns:
        Dict with LOO statistics including theta variance and hit rates.
    """
    from src.evaluation.eval import choose_threshold_by_shot_fpr

    shot_info = (
        timeline_df.groupby("shot_id")["y_true"]
        .max()
        .reset_index()
        .rename(columns={"y_true": "shot_label"})
    )
    all_shots = shot_info["shot_id"].tolist()

    thetas: List[float] = []
    correct_predictions: List[int] = []
    loo_rows: List[Dict[str, Any]] = []

    for held_out_sid in all_shots:
        held_out_label = int(
            shot_info.loc[shot_info["shot_id"] == held_out_sid, "shot_label"].iloc[0]
        )
        # Train on remaining shots
        train_timeline = timeline_df[timeline_df["shot_id"] != held_out_sid].copy()

        # Need at least 1 positive and 1 negative shot to select threshold
        train_labels = train_timeline.groupby("shot_id")["y_true"].max()
        if train_labels.sum() < 1 or (train_labels == 0).sum() < 1:
            continue

        try:
            theta, _ = choose_threshold_by_shot_fpr(
                timeline_df=train_timeline,
                sustain_ms=sustain_ms,
                max_shot_fpr=max_shot_fpr,
            )
        except Exception:
            continue

        # Evaluate on held-out shot
        held_out_timeline = timeline_df[timeline_df["shot_id"] == held_out_sid].copy()
        summary = apply_shot_warning_policy(
            held_out_timeline, threshold=theta, sustain_ms=sustain_ms
        )
        predicted_warning = int(summary["warning"].iloc[0]) if not summary.empty else 0
        correct = int(predicted_warning == held_out_label)

        thetas.append(float(theta))
        correct_predictions.append(correct)
        loo_rows.append(
            {
                "held_out_shot_id": int(held_out_sid),
                "held_out_label": held_out_label,
                "theta_selected": float(theta),
                "predicted_warning": predicted_warning,
                "correct": correct,
            }
        )

    if not thetas:
        return {"error": "LOO-CV failed: not enough data"}

    theta_arr = np.array(thetas)
    correct_arr = np.array(correct_predictions)

    return {
        "n_folds": len(thetas),
        "theta_mean": float(np.mean(theta_arr)),
        "theta_std": float(np.std(theta_arr)),
        "theta_min": float(np.min(theta_arr)),
        "theta_max": float(np.max(theta_arr)),
        "theta_range": float(np.max(theta_arr) - np.min(theta_arr)),
        "theta_cv": float(np.std(theta_arr) / max(np.mean(theta_arr), 1e-9)),
        "loo_accuracy": float(np.mean(correct_arr)),
        "loo_correct": int(np.sum(correct_arr)),
        "loo_total": int(len(correct_arr)),
        "loo_details": loo_rows,
    }


# ---------------------------------------------------------------------------
# 5.  Operational envelope:  (theta, sustain_ms) heatmap
# ---------------------------------------------------------------------------


def operational_envelope(
    timeline_df: pd.DataFrame,
    theta_range: Tuple[float, float] = (0.1, 0.95),
    sustain_range: Tuple[float, float] = (1.0, 10.0),
    n_theta: int = 30,
    n_sustain: int = 20,
) -> pd.DataFrame:
    """2-D sweep over (theta, sustain_ms) producing shot-level metrics.

    The resulting DataFrame can be pivoted into heatmaps of TPR / FPR.
    """
    rows: List[Dict[str, float]] = []
    for thr in np.linspace(theta_range[0], theta_range[1], n_theta):
        for sus in np.linspace(sustain_range[0], sustain_range[1], n_sustain):
            summary = apply_shot_warning_policy(
                timeline_df=timeline_df,
                threshold=float(thr),
                sustain_ms=float(sus),
            )
            sm = compute_shot_level_metrics(summary)
            rows.append(
                {
                    "theta": float(thr),
                    "sustain_ms": float(sus),
                    "shot_tpr": float(sm.get("shot_tpr", float("nan"))),
                    "shot_fpr": float(sm.get("shot_fpr", float("nan"))),
                    "shot_accuracy": float(sm.get("shot_accuracy", float("nan"))),
                    "lead_time_ms_median": float(
                        sm.get("lead_time_ms_median", float("nan"))
                    ),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6.  Plotting utilities
# ---------------------------------------------------------------------------


def save_bootstrap_histogram(
    bootstrap: BootstrapResult,
    theta_chosen: float,
    out_png: Path,
) -> None:
    """Save histogram of bootstrapped theta values with CI lines."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=140)
    ax.hist(
        bootstrap.theta_samples, bins=50, color="#1f77b4", alpha=0.75, edgecolor="white"
    )
    ax.axvline(
        theta_chosen,
        color="#d62728",
        linewidth=2,
        linestyle="-",
        label=f"Selected: {theta_chosen:.4f}",
    )
    ax.axvline(
        bootstrap.theta_ci_lower,
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label=f"95% CI lower: {bootstrap.theta_ci_lower:.4f}",
    )
    ax.axvline(
        bootstrap.theta_ci_upper,
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label=f"95% CI upper: {bootstrap.theta_ci_upper:.4f}",
    )
    ax.set_xlabel("Threshold (theta)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap Theta Distribution (n={bootstrap.n_boot})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_shot_roc_plot(
    roc_df: pd.DataFrame,
    theta_chosen: float,
    out_png: Path,
) -> None:
    """Save shot-level ROC curve with operating point marked."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=140)

    # Panel 1: Shot ROC (FPR vs TPR)
    ax = axes[0]
    ax.plot(roc_df["shot_fpr"], roc_df["shot_tpr"], color="#0b5394", linewidth=1.8)
    # Find operating point
    idx = (roc_df["theta"] - theta_chosen).abs().idxmin()
    ax.scatter(
        [roc_df.loc[idx, "shot_fpr"]],
        [roc_df.loc[idx, "shot_tpr"]],
        color="#d62728",
        s=60,
        zorder=5,
        label=f"theta={theta_chosen:.3f}",
    )
    ax.set_xlabel("Shot FPR")
    ax.set_ylabel("Shot TPR")
    ax.set_title("Shot-Level ROC")
    ax.set_xlim(-0.02, 0.25)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: TPR & FPR vs theta
    ax = axes[1]
    ax.plot(
        roc_df["theta"],
        roc_df["shot_tpr"],
        color="#0b5394",
        linewidth=1.5,
        label="Shot TPR",
    )
    ax.plot(
        roc_df["theta"],
        roc_df["shot_fpr"],
        color="#d62728",
        linewidth=1.5,
        label="Shot FPR",
    )
    ax.axvline(theta_chosen, color="#555555", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Threshold (theta)")
    ax.set_ylabel("Rate")
    ax.set_title("Metrics vs Threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: Lead time vs theta
    ax = axes[2]
    lead = roc_df["lead_time_ms_median"].copy()
    lead = lead.clip(lower=0)
    ax.plot(roc_df["theta"], lead, color="#2ca02c", linewidth=1.5)
    ax.axvline(theta_chosen, color="#555555", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Threshold (theta)")
    ax.set_ylabel("Lead Time (ms, median)")
    ax.set_title("Lead Time vs Threshold")
    ax.grid(alpha=0.25)

    fig.suptitle("Shot-Level Operating Point Trade-offs", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_plot(
    sensitivity: SensitivityResult,
    max_shot_fpr: float,
    out_png: Path,
) -> None:
    """Save sensitivity sweep around the operating point."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df = sensitivity.sweep_df
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)

    ax.plot(df["theta"], df["shot_tpr"], color="#0b5394", linewidth=2, label="Shot TPR")
    ax.plot(df["theta"], df["shot_fpr"], color="#d62728", linewidth=2, label="Shot FPR")
    ax.axhline(
        max_shot_fpr,
        color="#d62728",
        linewidth=1,
        linestyle=":",
        alpha=0.5,
        label=f"FPR constraint: {max_shot_fpr:.2%}",
    )
    ax.axvline(
        sensitivity.theta_center,
        color="#555555",
        linewidth=1.2,
        linestyle="--",
        label=f"theta = {sensitivity.theta_center:.4f}",
    )

    sr = sensitivity.stable_region
    if np.isfinite(sr["theta_lo"]) and np.isfinite(sr["theta_hi"]):
        ax.axvspan(
            sr["theta_lo"],
            sr["theta_hi"],
            alpha=0.08,
            color="#2ca02c",
            label=f"Stable region: [{sr['theta_lo']:.3f}, {sr['theta_hi']:.3f}]",
        )

    ax.set_xlabel("Threshold (theta)")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend(fontsize=7, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_envelope_heatmap(
    envelope_df: pd.DataFrame,
    theta_chosen: float,
    sustain_chosen: float,
    out_png: Path,
) -> None:
    """Save (theta, sustain_ms) operational envelope heatmap."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)

    for ax_idx, (metric, title, cmap) in enumerate(
        [
            ("shot_tpr", "Shot TPR", "RdYlGn"),
            ("shot_fpr", "Shot FPR", "RdYlGn_r"),
        ]
    ):
        ax = axes[ax_idx]
        pivot = envelope_df.pivot_table(
            index="sustain_ms", columns="theta", values=metric, aggfunc="mean"
        )
        im = ax.imshow(
            pivot.values[::-1],
            aspect="auto",
            extent=[
                pivot.columns.min(),
                pivot.columns.max(),
                pivot.index.min(),
                pivot.index.max(),
            ],
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax.scatter(
            [theta_chosen],
            [sustain_chosen],
            color="black",
            s=60,
            marker="x",
            linewidths=2,
            zorder=5,
        )
        ax.set_xlabel("Threshold (theta)")
        ax.set_ylabel("Sustain (ms)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Operational Envelope", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Stability verdict
# ---------------------------------------------------------------------------


def _assess_verdict(
    bootstrap: BootstrapResult,
    sensitivity: SensitivityResult,
    loocv: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Determine overall stability verdict with reasons."""
    reasons: List[str] = []
    score = 0  # higher = more stable

    # Bootstrap theta spread
    ci_width = bootstrap.theta_ci_upper - bootstrap.theta_ci_lower
    if ci_width < 0.10:
        score += 2
    elif ci_width < 0.20:
        score += 1
        reasons.append(f"Bootstrap 95% CI width = {ci_width:.3f} (moderate)")
    else:
        score -= 2
        reasons.append(f"Bootstrap 95% CI width = {ci_width:.3f} (too wide)")

    # Bootstrap theta CV
    theta_cv = bootstrap.theta_std / max(bootstrap.theta_mean, 1e-9)
    if theta_cv < 0.10:
        score += 2
    elif theta_cv < 0.20:
        score += 1
    else:
        score -= 1
        reasons.append(f"Theta CV = {theta_cv:.3f} (high variability)")

    # Shot TPR CI width
    tpr_ci = bootstrap.metric_ci.get("shot_tpr", {})
    tpr_width = tpr_ci.get("ci_upper", 1.0) - tpr_ci.get("ci_lower", 0.0)
    if tpr_width < 0.15:
        score += 1
    elif tpr_width > 0.30:
        score -= 2
        reasons.append(f"Shot TPR 95% CI width = {tpr_width:.3f} (too uncertain)")

    # Stable region width
    sr_width = sensitivity.stable_region.get("stable_width", 0.0)
    if sr_width > 0.15:
        score += 2
    elif sr_width > 0.05:
        score += 1
        reasons.append(f"Stable region width = {sr_width:.3f} (narrow but acceptable)")
    else:
        score -= 2
        reasons.append(f"Stable region width = {sr_width:.3f} (too narrow)")

    # LOO-CV accuracy
    loo_acc = loocv.get("loo_accuracy", 0.0)
    if loo_acc >= 0.90:
        score += 2
    elif loo_acc >= 0.80:
        score += 1
    else:
        score -= 1
        reasons.append(f"LOO-CV accuracy = {loo_acc:.3f} (low)")

    # LOO theta range
    loo_range = loocv.get("theta_range", 1.0)
    if loo_range < 0.10:
        score += 2
    elif loo_range < 0.25:
        score += 1
    else:
        score -= 1
        reasons.append(
            f"LOO theta range = {loo_range:.3f} (single-shot removal causes instability)"
        )

    if score >= 6:
        verdict = "STABLE"
    elif score >= 2:
        verdict = "MARGINAL"
    else:
        verdict = "UNSTABLE"

    if not reasons:
        reasons.append("All stability criteria met")

    return verdict, reasons


# ---------------------------------------------------------------------------
# 8.  Main entry point: run full stability analysis
# ---------------------------------------------------------------------------


def run_stability_analysis(
    timeline_df: pd.DataFrame,
    theta_chosen: float,
    sustain_ms: float,
    max_shot_fpr: float,
    output_dir: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> StabilityReport:
    """Run the full stability analysis suite and save all artifacts.

    Args:
        timeline_df: Validation timeline with columns
            {shot_id, time_ms, time_to_end_ms, y_true, prob_cal}.
        theta_chosen: The threshold selected by the training pipeline.
        sustain_ms: Sustained-warning window.
        max_shot_fpr: Shot-FPR constraint used during threshold selection.
        output_dir: Directory for output artifacts (plots + JSON).
        n_boot: Bootstrap iterations.
        seed: Random seed.

    Returns:
        StabilityReport with all results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "stability_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[stability] Running bootstrap CI analysis ...")
    bootstrap = bootstrap_threshold_ci(
        timeline_df=timeline_df,
        sustain_ms=sustain_ms,
        max_shot_fpr=max_shot_fpr,
        n_boot=n_boot,
        seed=seed,
    )
    save_bootstrap_histogram(
        bootstrap, theta_chosen, plots_dir / "bootstrap_theta_hist.png"
    )

    print("[stability] Computing shot-level ROC curve ...")
    shot_roc = shot_level_roc_curve(
        timeline_df=timeline_df,
        sustain_ms=sustain_ms,
    )
    save_shot_roc_plot(shot_roc, theta_chosen, plots_dir / "shot_level_roc.png")

    print("[stability] Running sensitivity analysis ...")
    sensitivity = threshold_sensitivity_analysis(
        timeline_df=timeline_df,
        theta_center=theta_chosen,
        sustain_ms=sustain_ms,
        max_shot_fpr=max_shot_fpr,
    )
    save_sensitivity_plot(
        sensitivity, max_shot_fpr, plots_dir / "sensitivity_sweep.png"
    )

    print("[stability] Running leave-one-out CV ...")
    loocv = leave_one_out_cv(
        timeline_df=timeline_df,
        sustain_ms=sustain_ms,
        max_shot_fpr=max_shot_fpr,
    )

    print("[stability] Computing operational envelope ...")
    envelope = operational_envelope(timeline_df=timeline_df)
    save_envelope_heatmap(
        envelope, theta_chosen, sustain_ms, plots_dir / "operational_envelope.png"
    )

    verdict, verdict_reasons = _assess_verdict(bootstrap, sensitivity, loocv)

    # Assemble summary
    summary: Dict[str, Any] = {
        "theta_chosen": theta_chosen,
        "sustain_ms": sustain_ms,
        "max_shot_fpr": max_shot_fpr,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "bootstrap": {
            "n_boot": bootstrap.n_boot,
            "theta_mean": bootstrap.theta_mean,
            "theta_std": bootstrap.theta_std,
            "theta_95ci": [bootstrap.theta_ci_lower, bootstrap.theta_ci_upper],
            "theta_95ci_width": bootstrap.theta_ci_upper - bootstrap.theta_ci_lower,
            "metric_ci": bootstrap.metric_ci,
            "n_shots_per_sample": bootstrap.n_shots_per_sample,
            "n_disruptive_per_sample_mean": bootstrap.n_disruptive_per_sample_mean,
        },
        "sensitivity": {
            "theta_center": sensitivity.theta_center,
            "delta": sensitivity.delta,
            "gradient_at_center": sensitivity.gradient_at_center,
            "stable_region": sensitivity.stable_region,
        },
        "loocv": {k: v for k, v in loocv.items() if k != "loo_details"},
        "shot_roc_rows": len(shot_roc),
    }

    # Save artifacts
    shot_roc.to_csv(output_dir / "shot_level_roc.csv", index=False)
    sensitivity.sweep_df.to_csv(output_dir / "sensitivity_sweep.csv", index=False)
    envelope.to_csv(output_dir / "operational_envelope.csv", index=False)
    if "loo_details" in loocv:
        pd.DataFrame(loocv["loo_details"]).to_csv(
            output_dir / "loocv_details.csv", index=False
        )
    (output_dir / "stability_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[stability] Verdict: {verdict}")
    for r in verdict_reasons:
        print(f"  - {r}")

    return StabilityReport(
        bootstrap=bootstrap,
        sensitivity=sensitivity,
        shot_roc=shot_roc,
        loocv_metrics={k: v for k, v in loocv.items() if k != "loo_details"},
        verdict=verdict,
        verdict_reasons=verdict_reasons,
        summary=summary,
    )
