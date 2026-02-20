#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from src.evaluation.calibrate import (
    ProbabilityCalibrator,
    calibration_quality_delta,
    expected_calibration_error,
)
from src.evaluation.eval import (
    apply_shot_warning_policy,
    choose_threshold_by_shot_fpr,
    compute_shot_level_metrics,
    save_probability_timeline_plot,
)
from src.models.advanced import train_sequence as adv_seq


@dataclass
class TransferEvaluationOutput:
    """Calibrated evaluation artifacts for transfer workflow."""

    theta: float
    theta_diag: dict[str, Any]
    calibration_delta: dict[str, dict[str, float]]

    val_metrics_calibrated: dict[str, float]
    test_metrics_calibrated: dict[str, float]

    val_timeline: pd.DataFrame
    test_timeline: pd.DataFrame
    shot_warn_test: pd.DataFrame
    shot_metrics_test: dict[str, float]

    reason_df: pd.DataFrame
    test_prob_calibrated: Any


def _compute_binary_metrics_safe(
    y_true: Any,
    y_prob: Any,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    pred = (p >= float(threshold)).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    has_both = np.unique(y).size >= 2
    has_pos = int(np.sum(y == 1)) > 0
    roc_auc = float(roc_auc_score(y, p)) if has_both else float("nan")
    pr_auc = float(average_precision_score(y, p)) if has_pos else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr": tpr,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "ece_15_bins": float(expected_calibration_error(y, p, n_bins=15)),
        "brier": float(np.mean((p - y) ** 2)),
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)) if y.size > 0 else float("nan"),
    }


def _threshold_sweep(
    timeline_df: pd.DataFrame,
    sustain_ms: float,
    min_threshold: float,
    max_threshold: float,
    num_steps: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for thr in np.linspace(min_threshold, max_threshold, num_steps):
        summary = apply_shot_warning_policy(
            timeline_df=timeline_df,
            threshold=float(thr),
            sustain_ms=float(sustain_ms),
        )
        sm = compute_shot_level_metrics(summary)
        rows.append(
            {
                "theta": float(thr),
                "shot_tpr": float(sm.get("shot_tpr", float("nan"))),
                "shot_fpr": float(sm.get("shot_fpr", float("nan"))),
                "shot_accuracy": float(sm.get("shot_accuracy", float("nan"))),
                "lead_time_ms_median": float(
                    sm.get("lead_time_ms_median", float("nan"))
                ),
            }
        )
    df = pd.DataFrame(rows)
    df["shot_tpr"] = df["shot_tpr"].fillna(0.0)
    df["shot_fpr"] = df["shot_fpr"].fillna(1.0)
    df["shot_accuracy"] = df["shot_accuracy"].fillna(0.0)
    df["lead_time_ms_median"] = df["lead_time_ms_median"].fillna(-1e12)
    return df


def _select_stable_theta(
    sweep_df: pd.DataFrame,
    max_shot_fpr: float,
    robust_delta: float,
    tpr_tolerance: float,
) -> tuple[float, dict[str, float | str]]:
    theta_arr = sweep_df["theta"].to_numpy(dtype=np.float64)
    tpr_arr = sweep_df["shot_tpr"].to_numpy(dtype=np.float64)
    fpr_arr = sweep_df["shot_fpr"].to_numpy(dtype=np.float64)
    acc_arr = sweep_df["shot_accuracy"].to_numpy(dtype=np.float64)
    lead_arr = sweep_df["lead_time_ms_median"].to_numpy(dtype=np.float64)

    feasible_mask = fpr_arr <= float(max_shot_fpr)
    feasible_idx = np.where(feasible_mask)[0]
    if feasible_idx.size == 0:
        order = np.lexsort((-acc_arr, -tpr_arr, fpr_arr))
        i = int(order[0])
        return float(theta_arr[i]), {
            "mode": "fallback_min_fpr",
            "shot_tpr": float(tpr_arr[i]),
            "shot_fpr": float(fpr_arr[i]),
            "shot_accuracy": float(acc_arr[i]),
            "lead_time_ms_median": float(lead_arr[i]),
        }

    best_tpr = float(np.max(tpr_arr[feasible_idx]))
    tpr_floor = best_tpr - float(max(tpr_tolerance, 0.0))
    candidate_mask = feasible_mask & (tpr_arr >= tpr_floor)
    candidate_idx = np.where(candidate_mask)[0]
    if candidate_idx.size == 0:
        candidate_idx = feasible_idx

    rows: list[dict[str, float]] = []
    for i in candidate_idx:
        theta = float(theta_arr[i])
        local_mask = (theta_arr >= theta - float(robust_delta)) & (
            theta_arr <= theta + float(robust_delta)
        )
        robust_fpr = float(np.max(fpr_arr[local_mask]))
        robust_tpr = float(np.min(tpr_arr[local_mask]))
        robust_acc = float(np.min(acc_arr[local_mask]))

        span_mask = (
            local_mask
            & (fpr_arr <= float(max_shot_fpr))
            & (tpr_arr >= float(tpr_floor))
        )
        if np.any(span_mask):
            span_vals = theta_arr[span_mask]
            stable_span = float(np.max(span_vals) - np.min(span_vals))
        else:
            stable_span = 0.0

        rows.append(
            {
                "theta": theta,
                "shot_tpr": float(tpr_arr[i]),
                "shot_fpr": float(fpr_arr[i]),
                "shot_accuracy": float(acc_arr[i]),
                "lead_time_ms_median": float(lead_arr[i]),
                "robust_fpr": robust_fpr,
                "robust_tpr": robust_tpr,
                "robust_accuracy": robust_acc,
                "stable_span": stable_span,
            }
        )

    rows.sort(
        key=lambda r: (
            r["robust_fpr"],
            -r["robust_tpr"],
            -r["stable_span"],
            -r["shot_accuracy"],
        )
    )
    chosen = rows[0]
    return float(chosen["theta"]), {
        "mode": "robust_stable_region",
        "shot_tpr": float(chosen["shot_tpr"]),
        "shot_fpr": float(chosen["shot_fpr"]),
        "shot_accuracy": float(chosen["shot_accuracy"]),
        "lead_time_ms_median": float(chosen["lead_time_ms_median"]),
        "robust_fpr": float(chosen["robust_fpr"]),
        "robust_tpr": float(chosen["robust_tpr"]),
        "robust_accuracy": float(chosen["robust_accuracy"]),
        "stable_span": float(chosen["stable_span"]),
        "max_shot_fpr": float(max_shot_fpr),
        "tpr_floor": float(tpr_floor),
        "robust_delta": float(robust_delta),
    }


def evaluate_transfer_predictions(
    *,
    val_calib_y: Any,
    val_calib_prob_raw: Any,
    val_thresh_y: Any,
    val_thresh_prob_raw: Any,
    test_y: Any,
    test_prob_raw: Any,
    val_timeline_base: pd.DataFrame,
    test_timeline_base: pd.DataFrame,
    sustain_ms: float,
    max_shot_fpr: float,
    calibration_method: str,
    reason_top_k: int,
    features: Sequence[str],
    contrib_by_window: Any,
    threshold_num_steps: int,
    threshold_robust_delta: float,
    threshold_tpr_tolerance: float,
) -> TransferEvaluationOutput:
    """Run calibration + thresholding + shot policy + reason extraction."""
    calibrator = ProbabilityCalibrator(method=str(calibration_method)).fit(
        y_true=val_calib_y,
        y_prob=val_calib_prob_raw,
    )
    val_thresh_prob_cal = calibrator.predict(val_thresh_prob_raw)
    test_prob_cal = calibrator.predict(test_prob_raw)
    cal_delta = calibration_quality_delta(
        y_true=val_thresh_y,
        prob_before=val_thresh_prob_raw,
        prob_after=val_thresh_prob_cal,
    )

    val_timeline = val_timeline_base.copy()
    val_timeline["prob_raw"] = val_thresh_prob_raw
    val_timeline["prob_cal"] = val_thresh_prob_cal
    sweep_df = _threshold_sweep(
        timeline_df=val_timeline,
        sustain_ms=float(sustain_ms),
        min_threshold=0.01,
        max_threshold=0.99,
        num_steps=max(int(threshold_num_steps), 50),
    )
    theta, stable_diag = _select_stable_theta(
        sweep_df=sweep_df,
        max_shot_fpr=float(max_shot_fpr),
        robust_delta=float(max(threshold_robust_delta, 0.0)),
        tpr_tolerance=float(max(threshold_tpr_tolerance, 0.0)),
    )
    theta_diag: dict[str, Any] = dict(stable_diag)
    theta_diag["objective"] = "shot_fpr_constrained_stable"

    # Keep original policy selection diagnostics for traceability.
    base_theta, base_diag = choose_threshold_by_shot_fpr(
        timeline_df=val_timeline,
        sustain_ms=float(sustain_ms),
        max_shot_fpr=float(max_shot_fpr),
    )
    theta_diag["base_theta"] = float(base_theta)
    theta_diag["base_shot_tpr"] = float(base_diag.get("shot_tpr", 0.0))
    theta_diag["base_shot_fpr"] = float(base_diag.get("shot_fpr", 1.0))

    test_timeline = test_timeline_base.copy()
    test_timeline["prob_raw"] = test_prob_raw
    test_timeline["prob_cal"] = test_prob_cal

    val_metrics_cal = _compute_binary_metrics_safe(
        y_true=val_thresh_y,
        y_prob=val_thresh_prob_cal,
        threshold=float(theta),
    )
    test_metrics_cal = _compute_binary_metrics_safe(
        y_true=test_y,
        y_prob=test_prob_cal,
        threshold=float(theta),
    )

    shot_warn_test = apply_shot_warning_policy(
        timeline_df=test_timeline,
        threshold=float(theta),
        sustain_ms=float(sustain_ms),
    )
    shot_metrics_test = compute_shot_level_metrics(shot_warn_test)

    reason_features: list[str] = []
    for feat in features:
        if feat.startswith("d1__") or feat.startswith("dr__"):
            reason_features.append(feat.split("__", 1)[1])
        else:
            reason_features.append(feat)

    reason_df = adv_seq.compute_disruption_reasons_per_shot(
        contrib_by_window=contrib_by_window,
        timeline_df=test_timeline,
        shot_summary=shot_warn_test,
        features=reason_features,
        top_k=int(reason_top_k),
    )

    return TransferEvaluationOutput(
        theta=float(theta),
        theta_diag=theta_diag,
        calibration_delta=cal_delta,
        val_metrics_calibrated=val_metrics_cal,
        test_metrics_calibrated=test_metrics_cal,
        val_timeline=val_timeline,
        test_timeline=test_timeline,
        shot_warn_test=shot_warn_test,
        shot_metrics_test=shot_metrics_test,
        reason_df=reason_df,
        test_prob_calibrated=test_prob_cal,
    )


def persist_transfer_outputs(
    *,
    output_dir: Path,
    report_dir: Path,
    eval_out: TransferEvaluationOutput,
    metrics_summary: dict[str, Any],
    training_config: dict[str, Any],
    sustain_ms: float,
    plot_all_test_shots: bool,
    plot_shot_limit: int,
) -> dict[str, Any]:
    """Persist core transfer artifacts and return plotting metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = (report_dir / "plots").resolve()
    prob_plot_dir = (plots_dir / "probability").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    prob_plot_dir.mkdir(parents=True, exist_ok=True)

    test_shot_ids = sorted(
        eval_out.test_timeline["shot_id"].drop_duplicates().astype(int).tolist()
    )
    if bool(plot_all_test_shots):
        selected_shots = test_shot_ids
    else:
        limit = int(max(plot_shot_limit, 0))
        selected_shots = test_shot_ids[:limit] if limit > 0 else []

    timeline_plot_count = 0
    for sid in selected_shots:
        g = eval_out.test_timeline[eval_out.test_timeline["shot_id"] == sid].copy()
        if not isinstance(g, pd.DataFrame) or g.empty:
            continue
        save_probability_timeline_plot(
            shot_df=g,
            threshold=float(eval_out.theta),
            sustain_ms=float(sustain_ms),
            out_png=prob_plot_dir / f"shot_{sid}_timeline.png",
            title=f"Shot {sid}: P(disruption|t) and y(t)",
        )
        timeline_plot_count += 1

    eval_out.test_timeline.to_csv(
        plots_dir / "probability_timelines_test.csv", index=False
    )
    eval_out.shot_warn_test.to_csv(output_dir / "warning_summary_test.csv", index=False)
    eval_out.reason_df.to_csv(
        output_dir / "disruption_reason_per_shot.csv", index=False, encoding="utf-8"
    )

    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )
    adv_seq.write_metrics_markdown(report_dir / "metrics.md", metrics_summary)

    return {
        "test_shot_count": int(len(test_shot_ids)),
        "generated_timeline_png": int(timeline_plot_count),
        "plots_dir": str(plots_dir),
        "probability_plot_dir": str(prob_plot_dir),
    }


def append_stability_if_requested(
    *,
    run_stability: bool,
    val_timeline: pd.DataFrame,
    theta: float,
    sustain_ms: float,
    max_shot_fpr: float,
    output_dir: Path,
    n_boot: int,
    seed: int,
    metrics_summary: dict[str, Any],
) -> dict[str, Any]:
    """Optionally run threshold stability analysis and update metrics."""
    if not run_stability:
        return metrics_summary

    from src.evaluation.threshold_stability import run_stability_analysis

    stability_report = run_stability_analysis(
        timeline_df=val_timeline,
        theta_chosen=float(theta),
        sustain_ms=float(sustain_ms),
        max_shot_fpr=float(max_shot_fpr),
        output_dir=(output_dir / "stability"),
        n_boot=int(n_boot),
        seed=int(seed),
    )
    out = dict(metrics_summary)
    out["stability"] = stability_report.summary
    return out
