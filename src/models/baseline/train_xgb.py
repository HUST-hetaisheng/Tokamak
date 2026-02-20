#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import shap
except Exception:
    shap = None

if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[3]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.evaluation.calibrate import ProbabilityCalibrator, calibration_quality_delta
from src.evaluation.eval import (
    apply_shot_warning_policy,
    choose_threshold_by_accuracy,
    choose_threshold_by_shot_fpr,
    choose_threshold_by_youden,
    compute_binary_metrics,
    compute_shot_level_metrics,
    save_calibration_curve_plot,
    save_probability_timeline_plot,
)


DEFAULT_DATA_ROOT = Path("G:/我的云端硬盘/Fuison/data")
DEFAULT_REQUIRED_FEATURE_COUNT = 23

PHYSICS_MAP: Dict[str, List[str]] = {
    "density_limit": [
        "ne_nG",
        "CIII",
        "sxr_kurt",
        "sxr_mean",
        "sxr_skew",
        "sxr_var",
        "xuv_kurt",
        "xuv_ratio",
        "xuv_skew",
        "xuv_var",
        "Mir_avg_amp",
        "Mir_VV",
        "v_loop",
    ],
    "locked_mode": [
        "Mir_avg_fre",
        "Mir_avg_amp",
        "Mir_VV",
        "mode_number_n",
        "n=1 amplitude",
        "MNM",
    ],
    "low_q_current_limit": ["qa_proxy", "ip", "Bt", "mode_number_n", "Mir_avg_amp"],
    "vde_control_loss": ["Z_proxy", "dx_a", "dy_a", "v_loop"],
    "impurity_radiation_collapse": [
        "CIII",
        "v_loop",
        "sxr_kurt",
        "sxr_mean",
        "sxr_skew",
        "sxr_var",
        "xuv_kurt",
        "xuv_ratio",
        "xuv_skew",
        "xuv_var",
    ],
}


@dataclass
class SplitPack:
    x: np.ndarray
    y: np.ndarray
    timeline: pd.DataFrame
    missing_shots: List[int]
    n_raw: int
    n_used: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded DART-centered J-TEXT training + calibration + SHAP"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hdf5-subdir", default="J-TEXT/unified_hdf5")
    parser.add_argument(
        "--dataset-artifact-dir", type=Path, default=Path("artifacts/datasets/jtext_v1")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/models/best")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gray-ms", type=float, default=30.0)
    parser.add_argument("--fallback-fls-ms", type=float, default=25.0)
    parser.add_argument("--fallback-dt-ms", type=float, default=1.0)
    parser.add_argument("--reconcile-len-tol", type=int, default=2)
    parser.add_argument("--max-train-shots", type=int, default=600)
    parser.add_argument("--max-val-shots", type=int, default=174)
    parser.add_argument("--max-test-shots", type=int, default=173)
    parser.add_argument("--xgb-estimators", type=int, default=140)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.07)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "isotonic_cv", "sigmoid"],
        default="isotonic_cv",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["youden", "accuracy", "shot_fpr_constrained"],
        default="shot_fpr_constrained",
    )
    parser.add_argument("--threshold-max-shot-fpr", type=float, default=0.02)
    parser.add_argument("--calibration-shot-fraction", type=float, default=0.5)
    parser.add_argument("--sustain-ms", type=float, default=5.0)
    parser.add_argument("--plot-shot-limit", type=int, default=3)
    parser.add_argument("--plot-all-test-shots", action="store_true")
    parser.add_argument("--reason-top-k", type=int, default=3)
    parser.add_argument("--max-shap-samples", type=int, default=5000)
    parser.add_argument("--top-k-shap", type=int, default=12)
    parser.add_argument(
        "--run-stability",
        action="store_true",
        help="Run threshold stability analysis (bootstrap CI, LOO-CV, sensitivity)",
    )
    parser.add_argument(
        "--stability-n-boot",
        type=int,
        default=2000,
        help="Number of bootstrap iterations for stability analysis",
    )
    return parser.parse_args()


def to_repo_rel(path: Path, repo_root: Path) -> str:
    p = path.resolve()
    r = repo_root.resolve()
    try:
        return str(p.relative_to(r)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def read_split_ids(path: Path) -> List[int]:
    out: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t:
            out.append(int(t))
    return out


def take_bounded(ids: Sequence[int], max_count: int, seed: int) -> List[int]:
    arr = list(ids)
    if max_count <= 0 or len(arr) <= max_count:
        return arr
    rng = random.Random(seed)
    rng.shuffle(arr)
    return sorted(arr[:max_count])


def split_val_for_calibration_and_threshold(
    shot_ids: Sequence[int],
    label_map: Mapping[int, int],
    calibration_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    ids = [int(s) for s in shot_ids]
    frac = float(np.clip(calibration_fraction, 0.1, 0.9))
    rng = random.Random(seed + 1009)

    pos = [sid for sid in ids if int(label_map.get(sid, 0)) == 1]
    neg = [sid for sid in ids if int(label_map.get(sid, 0)) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _split_group(arr: List[int]) -> Tuple[List[int], List[int]]:
        if not arr:
            return [], []
        k = int(round(len(arr) * frac))
        if len(arr) >= 2:
            k = max(1, min(len(arr) - 1, k))
        else:
            k = 1
        return arr[:k], arr[k:]

    pos_cal, pos_thr = _split_group(pos)
    neg_cal, neg_thr = _split_group(neg)

    calib = sorted(pos_cal + neg_cal)
    thresh = sorted(pos_thr + neg_thr)

    if not thresh:
        # Extreme fallback: keep at least one shot for threshold subset.
        if calib:
            thresh = [calib.pop()]
    if not calib:
        if thresh:
            calib = [thresh.pop()]

    return calib, thresh


def read_advanced_map(path: Path) -> Dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                continue
    return out


def build_hdf5_index(root: Path) -> Dict[int, Path]:
    idx: Dict[int, Path] = {}
    for p in root.rglob("*.hdf5"):
        if p.stem.isdigit() and int(p.stem) not in idx:
            idx[int(p.stem)] = p
    return idx


def read_scalar(ds: Any) -> Optional[float]:
    try:
        arr = np.asarray(ds)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def infer_time_axis_ms(h5: h5py.File, n: int, fallback_dt_ms: float) -> np.ndarray:
    start = None
    down = None
    for k in ("meta/StartTime", "StartTime", "meta/start_time"):
        if k in h5:
            start = read_scalar(h5[k][()])
            break
    for k in ("meta/DownTime", "DownTime", "meta/EndTime", "EndTime"):
        if k in h5:
            down = read_scalar(h5[k][()])
            break
    if start is not None and down is not None and n > 1 and down > start:
        return np.linspace(start * 1000.0, down * 1000.0, n, dtype=np.float64)
    if down is not None:
        dt = float(fallback_dt_ms)
        return (down * 1000.0) - np.arange(n - 1, -1, -1, dtype=np.float64) * dt
    dt = float(fallback_dt_ms)
    return np.arange(n, dtype=np.float64) * dt


def make_labels(
    t_ms: np.ndarray,
    shot_label: int,
    advanced_ms: Optional[float],
    gray_ms: float,
    fallback_fls_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(t_ms.shape[0])
    y = np.zeros(n, dtype=np.int32)
    keep = np.ones(n, dtype=bool)
    if shot_label == 0:
        return y, keep
    fls_ms = float(advanced_ms) if advanced_ms is not None else float(fallback_fls_ms)
    t_end = float(t_ms[-1])
    pos_start = t_end - fls_ms
    gray_start = pos_start - float(gray_ms)
    y = (t_ms >= pos_start).astype(np.int32)
    keep = (t_ms < gray_start) | (t_ms >= pos_start)
    return y, keep


def load_shot_features(
    h5_path: Path,
    features: Sequence[str],
    fallback_dt_ms: float,
    reconcile_len_tol: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(h5_path, "r") as h5:
        if "data" in h5 and isinstance(h5["data"], h5py.Group):
            g = h5["data"]
        elif "signals" in h5 and isinstance(h5["signals"], h5py.Group):
            g = h5["signals"]
        else:
            raise ValueError(f"No data group in {h5_path}")

        miss = [f for f in features if f not in g]
        if miss:
            raise ValueError(f"Missing required features {miss} in {h5_path}")

        arrs = [np.asarray(g[f], dtype=np.float64).reshape(-1) for f in features]
        lengths = [int(a.shape[0]) for a in arrs]
        n_min = min(lengths)
        n_max = max(lengths)
        if n_max - n_min > reconcile_len_tol:
            raise ValueError(
                f"Length mismatch too large in {h5_path}: {n_min}..{n_max}"
            )

        x = np.stack([a[:n_min] for a in arrs], axis=1).astype(np.float32)
        t_ms = infer_time_axis_ms(h5, n_min, fallback_dt_ms=fallback_dt_ms)

        label = 0
        for k in ("meta/IsDisrupt", "meta/is_disrupt", "IsDisrupt"):
            if k in h5:
                v = read_scalar(h5[k][()])
                if v is not None:
                    label = int(round(v))
                    break
    return x, t_ms, label


def load_label_map(clean_shots_csv: Path) -> Dict[int, int]:
    df = pd.read_csv(clean_shots_csv)
    return {int(r.shot_id): int(r.expected_label) for _, r in df.iterrows()}


def load_split(
    split_name: str,
    shot_ids: Sequence[int],
    hdf5_idx: Mapping[int, Path],
    features: Sequence[str],
    label_map: Mapping[int, int],
    advanced_map: Mapping[int, float],
    gray_ms: float,
    fallback_fls_ms: float,
    fallback_dt_ms: float,
    reconcile_len_tol: int,
) -> SplitPack:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    rows: List[Dict[str, float | int | str]] = []
    missing: List[int] = []
    n_raw = 0
    n_used = 0

    for sid in shot_ids:
        p = hdf5_idx.get(int(sid))
        if p is None:
            missing.append(int(sid))
            continue
        shot_label = int(label_map.get(int(sid), -1))
        if shot_label not in (0, 1):
            missing.append(int(sid))
            continue

        x, t_ms, _ = load_shot_features(
            h5_path=p,
            features=features,
            fallback_dt_ms=fallback_dt_ms,
            reconcile_len_tol=reconcile_len_tol,
        )
        adv = advanced_map.get(int(sid)) if shot_label == 1 else None
        y, keep = make_labels(
            t_ms=t_ms,
            shot_label=shot_label,
            advanced_ms=adv,
            gray_ms=gray_ms,
            fallback_fls_ms=fallback_fls_ms,
        )

        xk = x[keep]
        yk = y[keep]
        tk = t_ms[keep]
        t_rel = tk - float(t_ms[-1])

        xs.append(xk)
        ys.append(yk)
        n_raw += int(x.shape[0])
        n_used += int(xk.shape[0])

        for i in range(xk.shape[0]):
            rows.append(
                {
                    "split": split_name,
                    "shot_id": int(sid),
                    "time_ms": float(tk[i]),
                    "time_to_end_ms": float(t_rel[i]),
                    "y_true": int(yk[i]),
                }
            )

    if not xs:
        raise RuntimeError(f"No valid samples in split {split_name}")

    return SplitPack(
        x=np.concatenate(xs, axis=0).astype(np.float32),
        y=np.concatenate(ys, axis=0).astype(np.int32),
        timeline=pd.DataFrame(rows),
        missing_shots=missing,
        n_raw=n_raw,
        n_used=n_used,
    )


def train_logreg(
    x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        max_iter=600, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    model.fit(x_scaled, y_train)
    return scaler, model


def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    booster: str,
    scale_pos_weight: float,
    args: argparse.Namespace,
) -> xgb.XGBClassifier:
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "booster": booster,
        "n_estimators": int(args.xgb_estimators),
        "learning_rate": float(args.xgb_learning_rate),
        "max_depth": int(args.xgb_max_depth),
        "subsample": float(args.xgb_subsample),
        "colsample_bytree": float(args.xgb_colsample_bytree),
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "tree_method": "hist",
        "scale_pos_weight": float(scale_pos_weight),
    }
    if booster == "dart":
        params.update(
            {
                "sample_type": "uniform",
                "normalize_type": "tree",
                "rate_drop": 0.1,
                "skip_drop": 0.5,
            }
        )
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, verbose=False)
    return model


def metrics_from_prob(
    y_true: np.ndarray, prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    return compute_binary_metrics(y_true=y_true, y_prob=prob, threshold=threshold)


def mechanism_tags(feature: str) -> str:
    tags: List[str] = []
    for name, feats in PHYSICS_MAP.items():
        if feature in feats:
            tags.append(name)
    return ",".join(tags) if tags else "unmapped"


def compute_shap_topk(
    model: xgb.XGBClassifier,
    x_ref: np.ndarray,
    features: Sequence[str],
    max_samples: int,
    top_k: int,
    seed: int,
) -> pd.DataFrame:
    if shap is None:
        raise RuntimeError("SHAP package is unavailable in this environment")
    rng = np.random.default_rng(seed)
    n = min(max_samples, x_ref.shape[0])
    idx = rng.choice(x_ref.shape[0], size=n, replace=False)
    xs = x_ref[idx].astype(np.float32)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(xs)
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[-1], dtype=np.float64)
    else:
        arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, -1]

    rows: List[Dict[str, Any]] = []
    for i, f in enumerate(features):
        vals = arr[:, i]
        feat = xs[:, i].astype(np.float64)
        corr = (
            np.corrcoef(feat, vals)[0, 1]
            if np.std(feat) > 0 and np.std(vals) > 0
            else float("nan")
        )
        rows.append(
            {
                "feature": f,
                "mean_abs_shap": float(np.mean(np.abs(vals))),
                "corr_feature_shap": float(corr),
                "direction_hint": "higher->higher risk"
                if np.isfinite(corr) and corr >= 0
                else "higher->lower risk",
                "mechanism_tags": mechanism_tags(f),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def compute_model_importance_topk(
    model: xgb.XGBClassifier,
    features: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    importances = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    if importances.size != len(features):
        raise RuntimeError(
            f"Importance vector size mismatch: expected {len(features)} features, got {importances.size}"
        )
    rows: List[Dict[str, Any]] = []
    for i, f in enumerate(features):
        rows.append(
            {
                "feature": f,
                "mean_abs_shap": float(importances[i]),
                "corr_feature_shap": float("nan"),
                "direction_hint": "not_available_without_shap",
                "mechanism_tags": mechanism_tags(f),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def compute_disruption_reasons_per_shot(
    model: xgb.XGBClassifier,
    x_split: np.ndarray,
    timeline_df: pd.DataFrame,
    shot_summary: pd.DataFrame,
    features: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    if x_split.shape[0] != int(len(timeline_df)):
        raise RuntimeError(
            f"Timeline/feature length mismatch: x_split={x_split.shape[0]}, timeline={len(timeline_df)}"
        )
    if top_k <= 0:
        top_k = 1

    contrib_all = model.get_booster().predict(
        xgb.DMatrix(x_split, feature_names=list(features)),
        pred_contribs=True,
        validate_features=False,
    )
    contrib_all = np.asarray(contrib_all, dtype=np.float64)
    feat_contrib = contrib_all[:, : len(features)]

    warn_lookup = (
        shot_summary.set_index("shot_id", drop=False)
        if not shot_summary.empty
        else pd.DataFrame()
    )
    rows: List[Dict[str, Any]] = []

    for sid, g in timeline_df.groupby("shot_id", sort=True):
        sid_int = int(sid)
        if int(np.max(g["y_true"].to_numpy(dtype=int))) != 1:
            continue

        idx_all = g.index.to_numpy(dtype=np.int64)
        idx_pos = g.index[g["y_true"].to_numpy(dtype=int) == 1].to_numpy(dtype=np.int64)
        if idx_pos.size > 0:
            idx_use = idx_pos
            window_rule = "y_true==1"
        else:
            keep_n = int(min(200, idx_all.size))
            idx_use = idx_all[-keep_n:]
            window_rule = "tail_fallback"

        mean_contrib = feat_contrib[idx_use].mean(axis=0)
        pos_order = [
            int(i) for i in np.argsort(-mean_contrib) if mean_contrib[int(i)] > 0
        ]
        abs_order = [int(i) for i in np.argsort(-np.abs(mean_contrib))]
        chosen: List[int] = []
        for i in pos_order + abs_order:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= top_k:
                break

        top_entries: List[Dict[str, Any]] = []
        mech_scores: Dict[str, float] = {}
        for rank, i in enumerate(chosen, start=1):
            feat = str(features[i])
            contrib = float(mean_contrib[i])
            tags = mechanism_tags(feat)
            top_entries.append(
                {
                    "rank": rank,
                    "feature": feat,
                    "contribution": contrib,
                    "mechanism_tags": tags,
                }
            )
            for tag in [t for t in tags.split(",") if t and t != "unmapped"]:
                mech_scores[tag] = mech_scores.get(tag, 0.0) + max(contrib, 0.0)

        if mech_scores:
            primary_mechanism = max(mech_scores.items(), key=lambda kv: kv[1])[0]
            primary_score = float(mech_scores[primary_mechanism])
        else:
            primary_mechanism = "unmapped"
            primary_score = 0.0

        row: Dict[str, Any] = {
            "shot_id": sid_int,
            "primary_mechanism": primary_mechanism,
            "primary_mechanism_score": primary_score,
            "reason_window_rule": window_rule,
            "reason_window_points": int(idx_use.size),
            "top_features_json": json.dumps(top_entries, ensure_ascii=False),
        }
        for entry in top_entries:
            r = int(entry["rank"])
            row[f"top{r}_feature"] = entry["feature"]
            row[f"top{r}_contribution"] = float(entry["contribution"])
            row[f"top{r}_mechanism_tags"] = entry["mechanism_tags"]

        if not warn_lookup.empty and sid_int in warn_lookup.index:
            w = warn_lookup.loc[sid_int]
            row["warning"] = int(w["warning"])
            row["lead_time_ms"] = (
                float(w["lead_time_ms"])
                if pd.notna(w["lead_time_ms"])
                else float("nan")
            )
            row["warning_time_to_end_ms"] = (
                float(w["warning_time_to_end_ms"])
                if pd.notna(w["warning_time_to_end_ms"])
                else float("nan")
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "shot_id",
                "primary_mechanism",
                "primary_mechanism_score",
                "reason_window_rule",
                "reason_window_points",
                "top_features_json",
            ]
        )
    return pd.DataFrame(rows).sort_values("shot_id").reset_index(drop=True)


def write_markdown_report(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test = metrics["test_timepoint_calibrated"]
    shot = metrics["test_shot_policy"]
    target_hit = "ACHIEVED" if float(test["accuracy"]) >= 0.98 else "NOT ACHIEVED"
    threshold_policy = metrics["threshold_policy"]
    feature_policy = metrics.get("feature_policy", {})
    plotting = metrics.get("plotting", {})
    file_counts = metrics.get("artifact_file_counts", {})
    reason_summary = metrics.get("reason_summary", {})
    calibration_split = metrics.get("calibration_split", {})

    lines = [
        "# Agent-3 Metrics Summary",
        "",
        f"- Generated at (UTC): `{metrics['generated_at_utc']}`",
        f"- Selected baseline: `{metrics['selected_model']}`",
        f"- Accuracy>=0.98: `{target_hit}`",
        f"- Feature policy: `{feature_policy.get('mode', 'use_all_required_features')}` ({feature_policy.get('selected_feature_count', 0)}/{feature_policy.get('expected_feature_count', 0)})",
        "",
        "## Timepoint Metrics (Test, calibrated)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| accuracy | {test['accuracy']:.6f} |",
        f"| roc_auc | {test['roc_auc']:.6f} |",
        f"| pr_auc | {test['pr_auc']:.6f} |",
        f"| tpr | {test['tpr']:.6f} |",
        f"| fpr | {test['fpr']:.6f} |",
        f"| brier | {test['brier']:.6f} |",
        f"| ece_15_bins | {test['ece_15_bins']:.6f} |",
        "",
        "## Shot Policy Metrics (Test)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| shot_accuracy | {shot['shot_accuracy']:.6f} |",
        f"| shot_tpr | {shot['shot_tpr']:.6f} |",
        f"| shot_fpr | {shot['shot_fpr']:.6f} |",
        f"| lead_time_ms_median | {shot['lead_time_ms_median']:.3f} |",
        "",
        "## Threshold Policy",
        "",
        f"- objective: `{threshold_policy.get('objective', 'youden')}`",
        f"- max_shot_fpr: `{float(threshold_policy.get('max_shot_fpr', 0.0)):.4f}`",
        f"- theta: `{threshold_policy['theta']:.6f}`",
        f"- sustain: `{threshold_policy['sustain_ms']:.3f} ms`",
        "",
        "## Calibration / Threshold Split",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| val_total_shots | {int(calibration_split.get('val_total_shots', 0))} |",
        f"| val_calibration_shots | {int(calibration_split.get('val_calibration_shots', 0))} |",
        f"| val_threshold_shots | {int(calibration_split.get('val_threshold_shots', 0))} |",
        "",
        "## Generated File Counts",
        "",
        "| Artifact Type | Count |",
        "|---|---:|",
        f"| probability_timeline_png | {int(file_counts.get('probability_timeline_png', 0))} |",
        f"| report_plot_png_total | {int(file_counts.get('report_plot_png_total', 0))} |",
        "",
        "## Plotting Controls Used",
        "",
        f"- plot_all_test_shots: `{bool(plotting.get('plot_all_test_shots', False))}`",
        f"- plot_shot_limit: `{int(plotting.get('plot_shot_limit', 0))}`",
        f"- test_shot_count: `{int(plotting.get('test_shot_count', 0))}`",
        "",
        "## Disruption Reason Coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| disruptive_shots_test | {int(reason_summary.get('disruptive_shots_test', 0))} |",
        f"| reason_rows | {int(reason_summary.get('reason_rows', 0))} |",
        f"| reason_top_k | {int(reason_summary.get('reason_top_k', 0))} |",
        "",
        "## Baseline Comparison (Test raw P, threshold=0.5)",
        "",
        "| Model | accuracy | roc_auc | pr_auc | tpr | fpr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics["baselines_test_raw"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.6f} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['tpr']:.6f} | {m['fpr']:.6f} |"
        )

    lines += [
        "",
        "## SHAP Top Features",
        "",
        "| feature | mean_abs_shap | direction_hint | mechanism_tags |",
        "|---|---:|---|---|",
    ]
    for row in metrics["shap_topk"]:
        lines.append(
            f"| {row['feature']} | {row['mean_abs_shap']:.6f} | {row['direction_hint']} | {row['mechanism_tags']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_progress(
    progress_path: Path, metrics: Dict[str, Any], artifacts: List[str]
) -> None:
    if not progress_path.exists():
        return
    text = progress_path.read_text(encoding="utf-8")
    marker = "## Agent-3 (Modeler / Experimenter)"
    if marker not in text:
        return

    head, tail = text.split(marker, 1)
    if "\n## " in tail:
        _, rest = tail.split("\n## ", 1)
        rest = "\n## " + rest
    else:
        rest = ""

    test_acc = float(metrics["test_timepoint_calibrated"]["accuracy"])
    test_auc = float(metrics["test_timepoint_calibrated"]["roc_auc"])
    shot_acc = float(metrics["test_shot_policy"]["shot_accuracy"])
    reason_summary = metrics.get("reason_summary", {})
    reason_rows = int(reason_summary.get("reason_rows", 0))
    disruptive_shots_test = int(reason_summary.get("disruptive_shots_test", 0))
    threshold_policy = metrics["threshold_policy"]
    threshold_objective = str(threshold_policy.get("objective", "youden"))
    split_counts = metrics.get("split_shot_counts", {})
    feature_policy = metrics.get("feature_policy", {})
    file_counts = metrics.get("artifact_file_counts", {})
    timeline_png_count = int(file_counts.get("probability_timeline_png", 0))
    expected_feature_count = int(feature_policy.get("expected_feature_count", 0))
    selected_feature_count = int(feature_policy.get("selected_feature_count", 0))
    train_shots = int(split_counts.get("train", 0))
    val_shots = int(split_counts.get("val_total", split_counts.get("val", 0)))
    val_calib_shots = int(split_counts.get("val_calibration", 0))
    val_thresh_shots = int(split_counts.get("val_threshold", 0))
    test_shots = int(split_counts.get("test", 0))
    lines = [
        marker,
        "Status: completed",
        "Done:",
        "- Added train CLI plotting controls `--plot-shot-limit` and `--plot-all-test-shots` to resolve limited timeline exports.",
        "- Added validation threshold objective selection via `--threshold-objective {youden,accuracy,shot_fpr_constrained}`.",
        f"- Ran continuation training on full split sizes (train={train_shots}, val={val_shots}, test={test_shots}); validation was split into calibration={val_calib_shots} and threshold={val_thresh_shots} shots.",
        f"- Kept 23-feature use-all-by-default policy ({selected_feature_count}/{expected_feature_count}) and persisted it in `training_config.json`.",
        f"- Produced `{timeline_png_count}` probability timeline PNG files for test shots.",
        f"- Produced `{reason_rows}` per-disruptive-shot reason rows (expected disruptive shots in TEST: {disruptive_shots_test}).",
        f"- Current test metrics: accuracy={test_acc:.6f}, roc_auc={test_auc:.6f}, shot_accuracy={shot_acc:.6f}, threshold={float(threshold_policy['theta']):.6f} ({threshold_objective}).",
        "Next:",
        "- Coordinate with reviewer on threshold objective trade-offs and calibration holdout strategy.",
        "Blockers:",
        "- None.",
        "Artifacts:",
    ]
    lines.extend([f"- `{p}`" for p in artifacts])

    progress_path.write_text(head + "\n".join(lines) + rest, encoding="utf-8")


def save_shap_plot(shap_df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ordered = shap_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Top SHAP Features (DART)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    report_dir = (repo_root / args.report_dir).resolve()
    plots_dir = (report_dir / "plots").resolve()
    prob_plot_dir = (plots_dir / "probability").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prob_plot_dir.mkdir(parents=True, exist_ok=True)

    hdf5_root = (args.data_root / args.hdf5_subdir).resolve()
    if not hdf5_root.exists():
        raise FileNotFoundError(f"HDF5 root not found: {hdf5_root}")

    features_path = (
        repo_root / args.dataset_artifact_dir / "required_features.json"
    ).resolve()
    features = list(json.loads(features_path.read_text(encoding="utf-8")))
    if len(features) != DEFAULT_REQUIRED_FEATURE_COUNT:
        raise RuntimeError(
            f"Feature coverage requirement failed. Expected {DEFAULT_REQUIRED_FEATURE_COUNT}, found {len(features)}"
        )
    feature_policy = {
        "mode": "use_all_required_features",
        "use_all_by_default": True,
        "expected_feature_count": int(DEFAULT_REQUIRED_FEATURE_COUNT),
        "selected_feature_count": int(len(features)),
    }

    split_train = take_bounded(
        read_split_ids((repo_root / args.split_dir / "train.txt").resolve()),
        args.max_train_shots,
        args.seed,
    )
    split_val = take_bounded(
        read_split_ids((repo_root / args.split_dir / "val.txt").resolve()),
        args.max_val_shots,
        args.seed,
    )
    split_test = take_bounded(
        read_split_ids((repo_root / args.split_dir / "test.txt").resolve()),
        args.max_test_shots,
        args.seed,
    )

    label_map = load_label_map(
        (repo_root / args.dataset_artifact_dir / "clean_shots.csv").resolve()
    )
    split_val_calib, split_val_thresh = split_val_for_calibration_and_threshold(
        shot_ids=split_val,
        label_map=label_map,
        calibration_fraction=float(args.calibration_shot_fraction),
        seed=int(args.seed),
    )
    advanced_map = read_advanced_map(
        (repo_root / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json").resolve()
    )
    hdf5_idx = build_hdf5_index(hdf5_root)

    train = load_split(
        split_name="train",
        shot_ids=split_train,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=args.gray_ms,
        fallback_fls_ms=args.fallback_fls_ms,
        fallback_dt_ms=args.fallback_dt_ms,
        reconcile_len_tol=args.reconcile_len_tol,
    )
    val_calib = load_split(
        split_name="val_calib",
        shot_ids=split_val_calib,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=args.gray_ms,
        fallback_fls_ms=args.fallback_fls_ms,
        fallback_dt_ms=args.fallback_dt_ms,
        reconcile_len_tol=args.reconcile_len_tol,
    )
    val_thresh = load_split(
        split_name="val_threshold",
        shot_ids=split_val_thresh,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=args.gray_ms,
        fallback_fls_ms=args.fallback_fls_ms,
        fallback_dt_ms=args.fallback_dt_ms,
        reconcile_len_tol=args.reconcile_len_tol,
    )
    test = load_split(
        split_name="test",
        shot_ids=split_test,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=args.gray_ms,
        fallback_fls_ms=args.fallback_fls_ms,
        fallback_dt_ms=args.fallback_dt_ms,
        reconcile_len_tol=args.reconcile_len_tol,
    )

    n_pos = int(np.sum(train.y == 1))
    n_neg = int(np.sum(train.y == 0))
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    # Baseline 1: Logistic Regression
    lr_scaler, lr_model = train_logreg(train.x, train.y, seed=args.seed)
    lr_val_prob = lr_model.predict_proba(lr_scaler.transform(val_thresh.x))[:, 1]
    lr_test_prob = lr_model.predict_proba(lr_scaler.transform(test.x))[:, 1]

    # Baseline 2: XGBoost gbtree
    gbt_model = train_xgb(
        train.x, train.y, booster="gbtree", scale_pos_weight=scale_pos_weight, args=args
    )
    gbt_val_prob = gbt_model.predict_proba(val_thresh.x)[:, 1]
    gbt_test_prob = gbt_model.predict_proba(test.x)[:, 1]

    # Primary baseline: XGBoost DART
    dart_model = train_xgb(
        train.x, train.y, booster="dart", scale_pos_weight=scale_pos_weight, args=args
    )
    dart_val_calib_prob = dart_model.predict_proba(val_calib.x)[:, 1]
    dart_val_prob = dart_model.predict_proba(val_thresh.x)[:, 1]
    dart_test_prob = dart_model.predict_proba(test.x)[:, 1]

    baselines_val_raw = {
        "logreg": metrics_from_prob(val_thresh.y, lr_val_prob, 0.5),
        "xgb_gbtree": metrics_from_prob(val_thresh.y, gbt_val_prob, 0.5),
        "xgb_dart": metrics_from_prob(val_thresh.y, dart_val_prob, 0.5),
    }
    baselines_test_raw = {
        "logreg": metrics_from_prob(test.y, lr_test_prob, 0.5),
        "xgb_gbtree": metrics_from_prob(test.y, gbt_test_prob, 0.5),
        "xgb_dart": metrics_from_prob(test.y, dart_test_prob, 0.5),
    }

    calibrator = ProbabilityCalibrator(method=args.calibration_method).fit(
        val_calib.y, dart_val_calib_prob
    )
    val_prob_cal = calibrator.predict(dart_val_prob)
    test_prob_cal = calibrator.predict(dart_test_prob)
    cal_delta = calibration_quality_delta(val_thresh.y, dart_val_prob, val_prob_cal)

    val_timeline = val_thresh.timeline.copy()
    val_timeline["prob_raw"] = dart_val_prob
    val_timeline["prob_cal"] = val_prob_cal

    if args.threshold_objective == "accuracy":
        theta, theta_diag = choose_threshold_by_accuracy(val_thresh.y, val_prob_cal)
    elif args.threshold_objective == "shot_fpr_constrained":
        theta, theta_diag = choose_threshold_by_shot_fpr(
            timeline_df=val_timeline,
            sustain_ms=float(args.sustain_ms),
            max_shot_fpr=float(args.threshold_max_shot_fpr),
        )
    else:
        theta, theta_diag = choose_threshold_by_youden(val_thresh.y, val_prob_cal)
    theta_diag = dict(theta_diag)
    theta_diag["objective"] = args.threshold_objective
    val_metrics_cal = metrics_from_prob(val_thresh.y, val_prob_cal, theta)
    test_metrics_cal = metrics_from_prob(test.y, test_prob_cal, theta)

    test_timeline = test.timeline.copy()
    test_timeline["prob_raw"] = dart_test_prob
    test_timeline["prob_cal"] = test_prob_cal

    # Real-time warning policy on test timeline.
    shot_warn_test = apply_shot_warning_policy(
        test_timeline, threshold=float(theta), sustain_ms=float(args.sustain_ms)
    )
    shot_metrics_test = compute_shot_level_metrics(shot_warn_test)
    reason_df = compute_disruption_reasons_per_shot(
        model=dart_model,
        x_split=test.x,
        timeline_df=test_timeline,
        shot_summary=shot_warn_test,
        features=features,
        top_k=int(args.reason_top_k),
    )

    # Calibration curve artifact.
    save_calibration_curve_plot(
        y_true=test.y,
        prob_raw=dart_test_prob,
        prob_cal=test_prob_cal,
        out_png=plots_dir / "calibration_curve_test.png",
        out_csv=output_dir / "calibration_curve_points_test.csv",
        n_bins=15,
    )

    test_shot_ids = sorted(
        test_timeline["shot_id"].drop_duplicates().astype(int).tolist()
    )
    plot_limit = max(int(args.plot_shot_limit), 0)
    if args.plot_all_test_shots:
        selected_shots = test_shot_ids
    else:
        ranked_shots = (
            shot_warn_test.sort_values(
                ["shot_label", "shot_id"], ascending=[False, True]
            )["shot_id"]
            .astype(int)
            .tolist()
        )
        if plot_limit > 0 and len(ranked_shots) < plot_limit:
            ranked_shots = test_shot_ids
        selected_shots = ranked_shots[:plot_limit]

    timeline_plot_count = 0
    for sid in selected_shots:
        g = test_timeline[test_timeline["shot_id"] == sid].copy()
        if g.empty:
            continue
        save_probability_timeline_plot(
            shot_df=g,
            threshold=float(theta),
            sustain_ms=float(args.sustain_ms),
            out_png=prob_plot_dir / f"shot_{sid}_timeline.png",
            title=f"Shot {sid}: P(disruption|t) and y(t)",
        )
        timeline_plot_count += 1

    # SHAP on DART (fallback to gbtree, then model gain if SHAP is unavailable).
    shap_source = "xgb_dart"
    if shap is None:
        shap_source = "xgb_dart_gain_fallback_no_shap"
        shap_topk = compute_model_importance_topk(
            model=dart_model,
            features=features,
            top_k=int(args.top_k_shap),
        )
    else:
        try:
            shap_topk = compute_shap_topk(
                model=dart_model,
                x_ref=val_thresh.x,
                features=features,
                max_samples=int(args.max_shap_samples),
                top_k=int(args.top_k_shap),
                seed=int(args.seed),
            )
        except Exception:
            try:
                shap_source = "xgb_gbtree_fallback"
                shap_topk = compute_shap_topk(
                    model=gbt_model,
                    x_ref=val_thresh.x,
                    features=features,
                    max_samples=int(args.max_shap_samples),
                    top_k=int(args.top_k_shap),
                    seed=int(args.seed),
                )
            except Exception:
                shap_source = "xgb_gbtree_gain_fallback"
                shap_topk = compute_model_importance_topk(
                    model=gbt_model,
                    features=features,
                    top_k=int(args.top_k_shap),
                )

    save_shap_plot(shap_topk, plots_dir / "shap_topk_dart.png")

    # Persist artifacts.
    dart_model.save_model(str(output_dir / "model_xgb_dart.json"))
    calibrator.save(output_dir / "calibrator.joblib")
    (output_dir / "feature_list.json").write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )

    feature_stats = pd.DataFrame(
        {
            "feature": features,
            "mean": np.mean(train.x, axis=0),
            "std": np.std(train.x, axis=0),
        }
    )
    (output_dir / "normalization_stats.json").write_text(
        feature_stats.to_json(orient="records", indent=2), encoding="utf-8"
    )
    shap_topk.to_csv(output_dir / "shap_topk.csv", index=False)
    shot_warn_test.to_csv(output_dir / "warning_summary_test.csv", index=False)
    reason_df.to_csv(
        output_dir / "disruption_reason_per_shot.csv", index=False, encoding="utf-8"
    )
    test_timeline.to_csv(plots_dir / "probability_timelines_test.csv", index=False)

    plotting_config = {
        "plot_all_test_shots": bool(args.plot_all_test_shots),
        "plot_shot_limit": int(plot_limit),
        "test_shot_count": int(len(test_shot_ids)),
        "generated_timeline_png": int(timeline_plot_count),
    }
    artifact_file_counts = {
        "probability_timeline_png": int(timeline_plot_count),
        "report_plot_png_total": int(timeline_plot_count + 2),
    }

    training_config: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "hdf5_root": str(hdf5_root),
        "feature_count": len(features),
        "features": features,
        "feature_drop_reason": [],
        "feature_policy": feature_policy,
        "shot_counts": {
            "train": len(split_train),
            "val_total": len(split_val),
            "val_calibration": len(split_val_calib),
            "val_threshold": len(split_val_thresh),
            "test": len(split_test),
        },
        "point_counts": {
            "train_raw": train.n_raw,
            "train_used": train.n_used,
            "val_calibration_raw": val_calib.n_raw,
            "val_calibration_used": val_calib.n_used,
            "val_threshold_raw": val_thresh.n_raw,
            "val_threshold_used": val_thresh.n_used,
            "test_raw": test.n_raw,
            "test_used": test.n_used,
        },
        "missing_shots": {
            "train": train.missing_shots,
            "val_calibration": val_calib.missing_shots,
            "val_threshold": val_thresh.missing_shots,
            "test": test.missing_shots,
        },
        "labeling": {
            "gray_ms": float(args.gray_ms),
            "fallback_fls_ms": float(args.fallback_fls_ms),
            "fallback_dt_ms": float(args.fallback_dt_ms),
        },
        "class_imbalance": {
            "train_pos_points": n_pos,
            "train_neg_points": n_neg,
            "scale_pos_weight": scale_pos_weight,
        },
        "selected_model": "xgb_dart",
        "calibration": args.calibration_method,
        "threshold_policy": {
            "objective": args.threshold_objective,
            "max_shot_fpr": float(args.threshold_max_shot_fpr),
            "theta": float(theta),
            "sustain_ms": float(args.sustain_ms),
            "selection_diag": theta_diag,
        },
        "calibration_split": {
            "calibration_shot_fraction": float(args.calibration_shot_fraction),
            "val_total_shots": int(len(split_val)),
            "val_calibration_shots": int(len(split_val_calib)),
            "val_threshold_shots": int(len(split_val_thresh)),
        },
        "plotting": plotting_config,
        "xgb_params": {
            "n_estimators": args.xgb_estimators,
            "learning_rate": args.xgb_learning_rate,
            "max_depth": args.xgb_max_depth,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
        },
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )

    metrics_summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "xgb_dart",
        "shap_source": shap_source,
        "target_accuracy": 0.98,
        "target_accuracy_achieved": bool(float(test_metrics_cal["accuracy"]) >= 0.98),
        "feature_policy": feature_policy,
        "split_shot_counts": {
            "train": len(split_train),
            "val_total": len(split_val),
            "val_calibration": len(split_val_calib),
            "val_threshold": len(split_val_thresh),
            "test": len(split_test),
        },
        "calibration_split": {
            "calibration_shot_fraction": float(args.calibration_shot_fraction),
            "val_total_shots": int(len(split_val)),
            "val_calibration_shots": int(len(split_val_calib)),
            "val_threshold_shots": int(len(split_val_thresh)),
        },
        "plotting": plotting_config,
        "artifact_file_counts": artifact_file_counts,
        "reason_summary": {
            "disruptive_shots_test": int((shot_warn_test["shot_label"] == 1).sum()),
            "reason_rows": int(len(reason_df)),
            "reason_top_k": int(args.reason_top_k),
        },
        "baselines_val_raw": baselines_val_raw,
        "baselines_test_raw": baselines_test_raw,
        "val_timepoint_calibrated": val_metrics_cal,
        "test_timepoint_calibrated": test_metrics_cal,
        "test_shot_policy": shot_metrics_test,
        "calibration_threshold_delta": cal_delta,
        "calibration_val_delta": cal_delta,
        "threshold_policy": {
            "objective": args.threshold_objective,
            "max_shot_fpr": float(args.threshold_max_shot_fpr),
            "theta": float(theta),
            "sustain_ms": float(args.sustain_ms),
            "selection_diag": theta_diag,
        },
        "shap_topk": shap_topk.to_dict(orient="records"),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    write_markdown_report(report_dir / "metrics.md", metrics_summary)

    artifacts_for_progress = [
        "src/models/train.py",
        "src.evaluation.eval.py",
        "src.evaluation.calibrate.py",
        to_repo_rel(output_dir / "model_xgb_dart.json", repo_root),
        to_repo_rel(output_dir / "calibrator.joblib", repo_root),
        to_repo_rel(output_dir / "training_config.json", repo_root),
        to_repo_rel(output_dir / "metrics_summary.json", repo_root),
        to_repo_rel(output_dir / "shap_topk.csv", repo_root),
        to_repo_rel(output_dir / "warning_summary_test.csv", repo_root),
        to_repo_rel(output_dir / "disruption_reason_per_shot.csv", repo_root),
        to_repo_rel(report_dir / "metrics.md", repo_root),
        to_repo_rel(plots_dir / "calibration_curve_test.png", repo_root),
        to_repo_rel(plots_dir / "probability_timelines_test.csv", repo_root),
        to_repo_rel(prob_plot_dir, repo_root),
    ]
    update_progress(
        repo_root / "docs/progress.md", metrics_summary, artifacts_for_progress
    )

    # ---- Threshold stability analysis (industrial validation) ----
    if args.run_stability and args.threshold_objective == "shot_fpr_constrained":
        from src.evaluation.threshold_stability import run_stability_analysis

        stability_dir = output_dir / "stability"
        print("\n=== Running Threshold Stability Analysis ===")
        stability_report = run_stability_analysis(
            timeline_df=val_timeline,
            theta_chosen=float(theta),
            sustain_ms=float(args.sustain_ms),
            max_shot_fpr=float(args.threshold_max_shot_fpr),
            output_dir=stability_dir,
            n_boot=int(args.stability_n_boot),
            seed=int(args.seed),
        )
        metrics_summary["stability"] = stability_report.summary
        # Re-save metrics with stability results
        (output_dir / "metrics_summary.json").write_text(
            json.dumps(metrics_summary, indent=2), encoding="utf-8"
        )
        print(f"\n=== Stability Verdict: {stability_report.verdict} ===")
        for reason in stability_report.verdict_reasons:
            print(f"  - {reason}")

    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    main()
