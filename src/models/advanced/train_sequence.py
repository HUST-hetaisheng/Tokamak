#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    repo_root_for_imports = Path(__file__).resolve().parents[3]
    if str(repo_root_for_imports) not in sys.path:
        sys.path.insert(0, str(repo_root_for_imports))

from src.evaluation.calibrate import ProbabilityCalibrator, calibration_quality_delta
from src.evaluation.eval import (
    apply_shot_warning_policy,
    choose_threshold_by_shot_fpr,
    compute_binary_metrics,
    compute_shot_level_metrics,
    save_probability_timeline_plot,
)
from src.models.advanced.sequence_arch import (
    GRUClassifier,
    MambaLiteClassifier,
    TemporalTransformerClassifier,
)
from src.models.baseline import train_xgb as train_base


DEFAULT_DATA_ROOT = Path("G:/我的云端硬盘/Fuison/data")
DEFAULT_REQUIRED_FEATURE_COUNT = 23


@dataclass
class ShotSeries:
    shot_id: int
    x: np.ndarray
    y: np.ndarray
    time_ms: np.ndarray
    time_to_end_ms: np.ndarray


@dataclass
class WindowPack:
    x: np.ndarray
    y: np.ndarray
    timeline: pd.DataFrame
    short_shots: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced sequence models (Transformer/MambaLite/GRU) for J-TEXT disruption prediction"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hdf5-subdir", default="J-TEXT/unified_hdf5")
    parser.add_argument(
        "--dataset-artifact-dir", type=Path, default=Path("artifacts/datasets/jtext_v1")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/models/iters")
    )
    parser.add_argument("--report-root", type=Path, default=Path("reports/iters"))
    parser.add_argument("--models", default="transformer_small,mamba_lite,gru")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gray-ms", type=float, default=30.0)
    parser.add_argument("--fallback-fls-ms", type=float, default=25.0)
    parser.add_argument("--fallback-dt-ms", type=float, default=1.0)
    parser.add_argument("--reconcile-len-tol", type=int, default=2)
    parser.add_argument("--max-train-shots", type=int, default=0)
    parser.add_argument("--max-val-shots", type=int, default=0)
    parser.add_argument("--max-test-shots", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=1,
        help="Stride for val/test windowing (default 1 for maximum temporal resolution). "
        "Training always uses --stride for efficiency.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--calibration-method",
        choices=["isotonic", "isotonic_cv", "sigmoid"],
        default="isotonic_cv",
    )
    parser.add_argument("--calibration-shot-fraction", type=float, default=0.5)
    parser.add_argument("--threshold-max-shot-fpr", type=float, default=0.02)
    parser.add_argument("--sustain-ms", type=float, default=3.0)
    parser.add_argument("--reason-top-k", type=int, default=3)
    parser.add_argument("--pad-short-shots", action="store_true", default=True)
    parser.add_argument("--short-pad-mode", choices=["edge", "zero"], default="edge")
    parser.add_argument("--plot-all-test-shots", action="store_true", default=True)
    parser.add_argument("--plot-shot-limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mechanism_tags(feature: str) -> str:
    tags: List[str] = []
    for name, feats in train_base.PHYSICS_MAP.items():
        if feature in feats:
            tags.append(name)
    return ",".join(tags) if tags else "unmapped"


def load_features(path: Path) -> List[str]:
    feats = list(json.loads(path.read_text(encoding="utf-8")))
    if len(feats) != DEFAULT_REQUIRED_FEATURE_COUNT:
        raise RuntimeError(
            f"Feature count mismatch: expected={DEFAULT_REQUIRED_FEATURE_COUNT}, got={len(feats)}"
        )
    return feats


def load_split_data(
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
) -> Tuple[Dict[int, ShotSeries], Dict[str, Any]]:
    out: Dict[int, ShotSeries] = {}
    missing_shots: List[int] = []
    label_missing: List[int] = []
    feature_errors: List[Dict[str, Any]] = []
    n_raw_total = 0
    n_used_total = 0

    for sid in shot_ids:
        sid_int = int(sid)
        h5_path = hdf5_idx.get(sid_int)
        if h5_path is None:
            missing_shots.append(sid_int)
            continue
        shot_label = int(label_map.get(sid_int, -1))
        if shot_label not in (0, 1):
            label_missing.append(sid_int)
            continue
        try:
            x, t_ms, _ = train_base.load_shot_features(
                h5_path=h5_path,
                features=features,
                fallback_dt_ms=fallback_dt_ms,
                reconcile_len_tol=reconcile_len_tol,
            )
            adv = advanced_map.get(sid_int) if shot_label == 1 else None
            y, keep = train_base.make_labels(
                t_ms=t_ms,
                shot_label=shot_label,
                advanced_ms=adv,
                gray_ms=gray_ms,
                fallback_fls_ms=fallback_fls_ms,
            )
            xk = x[keep]
            yk = y[keep]
            tk = t_ms[keep]
            if xk.shape[0] < 4:
                feature_errors.append(
                    {
                        "shot_id": sid_int,
                        "error": "too_few_points_after_gray_zone",
                    }
                )
                continue
            n_raw_total += int(x.shape[0])
            n_used_total += int(xk.shape[0])
            out[sid_int] = ShotSeries(
                shot_id=sid_int,
                x=xk.astype(np.float32),
                y=yk.astype(np.int32),
                time_ms=tk.astype(np.float64),
                time_to_end_ms=(tk - float(t_ms[-1])).astype(np.float64),
            )
        except Exception as exc:
            feature_errors.append({"shot_id": sid_int, "error": str(exc)})
            continue

    meta = {
        "split_name": split_name,
        "requested_shots": int(len(shot_ids)),
        "loaded_shots": int(len(out)),
        "missing_shots": missing_shots,
        "label_missing_shots": label_missing,
        "feature_errors": feature_errors,
        "raw_points_total": int(n_raw_total),
        "used_points_total": int(n_used_total),
    }
    return out, meta


def build_window_pack(
    split_name: str,
    shots: Mapping[int, ShotSeries],
    window_size: int,
    stride: int,
    pad_short_shots: bool,
    short_pad_mode: str,
) -> WindowPack:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    timeline_rows: List[Dict[str, Any]] = []
    short_shots: List[int] = []

    for sid in sorted(shots):
        shot = shots[sid]
        n = int(shot.x.shape[0])
        if n < window_size:
            short_shots.append(int(sid))
            if not pad_short_shots or n <= 0:
                continue
            # Generate sub-windows at every stride step for temporal resolution.
            # Each sub-window ends at a different data index, with appropriate
            # left-padding to fill the window to window_size.
            end_idx = list(range(0, n, stride))
            if end_idx[-1] != n - 1:
                end_idx.append(n - 1)
            for end in end_idx:
                x_raw = shot.x[: end + 1].astype(np.float32)
                pad_len = int(window_size - x_raw.shape[0])
                if pad_len > 0:
                    if short_pad_mode == "edge" and x_raw.shape[0] > 0:
                        pad = np.repeat(x_raw[:1], repeats=pad_len, axis=0)
                    else:
                        pad = np.zeros((pad_len, x_raw.shape[1]), dtype=np.float32)
                    xw = np.concatenate([pad, x_raw], axis=0).astype(np.float32)
                else:
                    xw = x_raw[-window_size:]
                    pad_len = 0
                y_end = int(shot.y[end])
                x_rows.append(xw)
                y_rows.append(y_end)
                timeline_rows.append(
                    {
                        "split": split_name,
                        "shot_id": int(sid),
                        "time_ms": float(shot.time_ms[end]),
                        "time_to_end_ms": float(shot.time_to_end_ms[end]),
                        "y_true": y_end,
                        "window_start_idx": 0,
                        "window_end_idx": int(end),
                        "pad_left_len": int(max(pad_len, 0)),
                    }
                )
            continue
        end_idx = list(range(window_size - 1, n, stride))
        if end_idx[-1] != n - 1:
            end_idx.append(n - 1)
        for end in end_idx:
            start = end - window_size + 1
            xw = shot.x[start : end + 1]
            y_end = int(shot.y[end])
            x_rows.append(xw)
            y_rows.append(y_end)
            timeline_rows.append(
                {
                    "split": split_name,
                    "shot_id": int(sid),
                    "time_ms": float(shot.time_ms[end]),
                    "time_to_end_ms": float(shot.time_to_end_ms[end]),
                    "y_true": y_end,
                    "window_start_idx": int(start),
                    "window_end_idx": int(end),
                    "pad_left_len": 0,
                }
            )

    if not x_rows:
        raise RuntimeError(
            f"No windows generated for split={split_name}. window_size={window_size}, stride={stride}"
        )

    return WindowPack(
        x=np.stack(x_rows).astype(np.float32),
        y=np.asarray(y_rows, dtype=np.int32),
        timeline=pd.DataFrame(timeline_rows).reset_index(drop=True),
        short_shots=short_shots,
    )


def fit_normalizer(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64)
    mu = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mu.astype(np.float32), std.astype(np.float32)


def apply_normalizer(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mu[None, None, :]) / std[None, None, :]).astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def build_model(model_name: str, input_dim: int, dropout: float) -> nn.Module:
    if model_name == "transformer_small":
        return TemporalTransformerClassifier(
            input_dim=input_dim,
            d_model=96,
            n_heads=4,
            n_layers=2,
            dropout=dropout,
            pooling="last",
        )
    if model_name == "mamba_lite":
        return MambaLiteClassifier(
            input_dim=input_dim,
            d_model=96,
            state_dim=64,
            n_layers=2,
            dropout=dropout,
            pooling="last",
        )
    if model_name == "gru":
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            n_layers=2,
            dropout=dropout,
            bidirectional=False,
            pooling="last",
        )
    raise ValueError(f"Unsupported model_name={model_name}")


def binary_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        y_true,
        reduction="none",
        pos_weight=pos_weight,
    )
    if focal_gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * y_true + (1.0 - p) * (1.0 - y_true)
        mod = torch.pow(torch.clamp(1.0 - p_t, min=1e-8), focal_gamma)
        return torch.mean(mod * bce)
    return torch.mean(bce)


def predict_logits(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = np.zeros((x.shape[0],), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            j = min(i + batch_size, x.shape[0])
            xb = torch.from_numpy(x[i:j]).to(device)
            logits = model(xb)
            out[i:j] = logits.detach().cpu().numpy().astype(np.float64)
    return out


def compute_gradient_input_attribution(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    prev_mode = model.training
    model.train()
    n, _, f = x.shape
    out = np.zeros((n, f), dtype=np.float64)
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            xb = torch.from_numpy(x[i:j]).to(device)
            xb.requires_grad_(True)
            logits = model(xb)
            grads = torch.autograd.grad(
                outputs=logits.sum(), inputs=xb, create_graph=False, retain_graph=False
            )[0]
            contrib = (grads * xb).mean(dim=1)
            out[i:j] = contrib.detach().cpu().numpy().astype(np.float64)
    if not prev_mode:
        model.eval()
    return out


def train_one_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    focal_gamma: float,
    max_grad_norm: float,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, Any]], float, int]:
    ds = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    n_pos = int(np.sum(train_y == 1))
    n_neg = int(np.sum(train_y == 0))
    pos_weight_value = float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = -1.0
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = binary_loss(
                logits=logits,
                y_true=yb,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
            )
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
            optimizer.step()
            running_loss += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])

        val_logits = predict_logits(
            model=model, x=val_x, batch_size=batch_size, device=device
        )
        val_prob = sigmoid_np(val_logits)
        try:
            val_auc = float(
                compute_binary_metrics(val_y, val_prob, threshold=0.5)["roc_auc"]
            )
        except Exception:
            val_auc = float("nan")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(running_loss / max(seen, 1)),
                "val_roc_auc": float(val_auc) if np.isfinite(val_auc) else None,
            }
        )

        if best_state is not None and (epoch - best_epoch) >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, float(best_auc), int(best_epoch)


def compute_disruption_reasons_per_shot(
    contrib_by_window: np.ndarray,
    timeline_df: pd.DataFrame,
    shot_summary: pd.DataFrame,
    features: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    if contrib_by_window.shape[0] != int(len(timeline_df)):
        raise RuntimeError(
            f"Attribution/timeline mismatch: attr={contrib_by_window.shape[0]}, timeline={len(timeline_df)}"
        )
    warn_lookup = (
        shot_summary.set_index("shot_id", drop=False)
        if not shot_summary.empty
        else pd.DataFrame()
    )
    rows: List[Dict[str, Any]] = []
    kk = max(int(top_k), 1)

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
            idx_use = idx_all[-min(20, idx_all.size) :]
            window_rule = "tail_fallback"

        mean_contrib = contrib_by_window[idx_use].mean(axis=0)
        pos_order = [
            int(i) for i in np.argsort(-mean_contrib) if mean_contrib[int(i)] > 0
        ]
        abs_order = [int(i) for i in np.argsort(-np.abs(mean_contrib))]
        chosen: List[int] = []
        for i in pos_order + abs_order:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= kk:
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


def write_metrics_markdown(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test = metrics["test_timepoint_calibrated"]
    shot = metrics["test_shot_policy"]
    th = metrics["threshold_policy"]
    lines = [
        "# Advanced Model Metrics",
        "",
        f"- generated_at_utc: `{metrics['generated_at_utc']}`",
        f"- run_name: `{metrics['run_name']}`",
        f"- model_name: `{metrics['model_name']}`",
        "",
        "## Test Timepoint (Calibrated)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| accuracy | {float(test['accuracy']):.6f} |",
        f"| roc_auc | {float(test['roc_auc']):.6f} |",
        f"| pr_auc | {float(test['pr_auc']):.6f} |",
        f"| tpr | {float(test['tpr']):.6f} |",
        f"| fpr | {float(test['fpr']):.6f} |",
        f"| ece_15_bins | {float(test['ece_15_bins']):.6f} |",
        "",
        "## Test Shot Policy",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| shot_accuracy | {float(shot['shot_accuracy']):.6f} |",
        f"| shot_tpr | {float(shot['shot_tpr']):.6f} |",
        f"| shot_fpr | {float(shot['shot_fpr']):.6f} |",
        f"| lead_time_ms_median | {float(shot['lead_time_ms_median']):.3f} |",
        "",
        "## Threshold",
        "",
        f"- objective: `{th.get('objective', 'shot_fpr_constrained')}`",
        f"- max_shot_fpr: `{float(th.get('max_shot_fpr', 0.02)):.4f}`",
        f"- theta: `{float(th.get('theta', 0.5)):.6f}`",
        f"- sustain_ms: `{float(th.get('sustain_ms', 3.0)):.3f}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_advanced_summary(
    summary_rows: List[Dict[str, Any]], summary_csv: Path, summary_md: Path
) -> None:
    df = pd.DataFrame(summary_rows)
    df = df.sort_values(
        ["shot_accuracy", "test_roc_auc", "test_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)

    lines = [
        "# Advanced Model Summary",
        "",
        f"- total_runs: `{len(df)}`",
        f"- recommended: `{df.iloc[0]['run_name'] if len(df) > 0 else 'N/A'}`",
        "",
        "| run_name | model_name | test_acc | test_auc | test_pr_auc | shot_acc | shot_tpr | shot_fpr | theta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {run} | {model} | {acc:.6f} | {auc:.6f} | {prauc:.6f} | {sacc:.6f} | {stpr:.6f} | {sfpr:.6f} | {theta:.6f} |".format(
                run=r["run_name"],
                model=r["model_name"],
                acc=float(r["test_accuracy"]),
                auc=float(r["test_roc_auc"]),
                prauc=float(r["test_pr_auc"]),
                sacc=float(r["shot_accuracy"]),
                stpr=float(r["shot_tpr"]),
                sfpr=float(r["shot_fpr"]),
                theta=float(r["theta"]),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_progress_agent6(
    progress_path: Path,
    best_row: Dict[str, Any],
    artifacts: List[str],
) -> None:
    if not progress_path.exists():
        return
    marker = "## Agent-6 (Advanced Modeler)"
    lines = [
        marker,
        "Status: completed",
        "Done:",
        "- Added `src.models.advanced.sequence_arch.py` with `TemporalTransformerClassifier`, `MambaLiteClassifier`, and `GRUClassifier`.",
        "- Added `src/models/train_advanced.py` with bounded advanced training sweep, calibration, shot-level thresholding, probability-timeline export, and gradient*input reasons.",
        f"- Executed 3 fair-window runs (`window_size={int(best_row['window_size'])}`, `stride={int(best_row['stride'])}`): transformer_small / mamba_lite / gru.",
        f"- Best run: `{best_row['run_name']}` with test_acc={float(best_row['test_accuracy']):.6f}, roc_auc={float(best_row['test_roc_auc']):.6f}, shot_acc={float(best_row['shot_accuracy']):.6f}, shot_fpr={float(best_row['shot_fpr']):.6f}.",
        "Next:",
        "- Scale to larger window/horizon ablations and add cross-device transfer hooks on the same architecture backbone.",
        "Blockers:",
        "- None.",
        "Artifacts:",
    ]
    lines.extend([f"- `{p}`" for p in artifacts])

    text = progress_path.read_text(encoding="utf-8")
    new_block = "\n".join(lines)
    if marker in text:
        head, tail = text.split(marker, 1)
        if "\n## " in tail:
            _, rest = tail.split("\n## ", 1)
            rest = "\n## " + rest
        else:
            rest = ""
        out = head.rstrip() + "\n\n" + new_block + rest
    else:
        out = text.rstrip() + "\n\n" + new_block + "\n"
    progress_path.write_text(out, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = choose_device(args.device)

    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve()
    hdf5_root = (data_root / args.hdf5_subdir).resolve()
    if not hdf5_root.exists():
        raise FileNotFoundError(f"HDF5 root not found: {hdf5_root}")

    features_path = (
        repo_root / args.dataset_artifact_dir / "required_features.json"
    ).resolve()
    features = load_features(features_path)
    input_dim = int(len(features))

    train_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / args.split_dir / "train.txt").resolve()),
        int(args.max_train_shots),
        int(args.seed),
    )
    val_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / args.split_dir / "val.txt").resolve()),
        int(args.max_val_shots),
        int(args.seed),
    )
    test_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / args.split_dir / "test.txt").resolve()),
        int(args.max_test_shots),
        int(args.seed),
    )

    label_map = train_base.load_label_map(
        (repo_root / args.dataset_artifact_dir / "clean_shots.csv").resolve()
    )
    val_calib_ids, val_thresh_ids = train_base.split_val_for_calibration_and_threshold(
        shot_ids=val_ids,
        label_map=label_map,
        calibration_fraction=float(args.calibration_shot_fraction),
        seed=int(args.seed),
    )
    advanced_map = train_base.read_advanced_map(
        (repo_root / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json").resolve()
    )
    hdf5_idx = train_base.build_hdf5_index(hdf5_root)

    train_shots, train_meta = load_split_data(
        split_name="train",
        shot_ids=train_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(args.gray_ms),
        fallback_fls_ms=float(args.fallback_fls_ms),
        fallback_dt_ms=float(args.fallback_dt_ms),
        reconcile_len_tol=int(args.reconcile_len_tol),
    )
    val_calib_shots, val_calib_meta = load_split_data(
        split_name="val_calib",
        shot_ids=val_calib_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(args.gray_ms),
        fallback_fls_ms=float(args.fallback_fls_ms),
        fallback_dt_ms=float(args.fallback_dt_ms),
        reconcile_len_tol=int(args.reconcile_len_tol),
    )
    val_thresh_shots, val_thresh_meta = load_split_data(
        split_name="val_thresh",
        shot_ids=val_thresh_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(args.gray_ms),
        fallback_fls_ms=float(args.fallback_fls_ms),
        fallback_dt_ms=float(args.fallback_dt_ms),
        reconcile_len_tol=int(args.reconcile_len_tol),
    )
    test_shots, test_meta = load_split_data(
        split_name="test",
        shot_ids=test_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(args.gray_ms),
        fallback_fls_ms=float(args.fallback_fls_ms),
        fallback_dt_ms=float(args.fallback_dt_ms),
        reconcile_len_tol=int(args.reconcile_len_tol),
    )

    train_pack = build_window_pack(
        split_name="train",
        shots=train_shots,
        window_size=int(args.window_size),
        stride=int(args.stride),
        pad_short_shots=bool(args.pad_short_shots),
        short_pad_mode=str(args.short_pad_mode),
    )
    eval_stride = int(args.eval_stride)
    val_calib_pack = build_window_pack(
        split_name="val_calib",
        shots=val_calib_shots,
        window_size=int(args.window_size),
        stride=eval_stride,
        pad_short_shots=bool(args.pad_short_shots),
        short_pad_mode=str(args.short_pad_mode),
    )
    val_thresh_pack = build_window_pack(
        split_name="val_thresh",
        shots=val_thresh_shots,
        window_size=int(args.window_size),
        stride=eval_stride,
        pad_short_shots=bool(args.pad_short_shots),
        short_pad_mode=str(args.short_pad_mode),
    )
    test_pack = build_window_pack(
        split_name="test",
        shots=test_shots,
        window_size=int(args.window_size),
        stride=eval_stride,
        pad_short_shots=bool(args.pad_short_shots),
        short_pad_mode=str(args.short_pad_mode),
    )

    norm_mu, norm_std = fit_normalizer(train_pack.x)
    train_x = apply_normalizer(train_pack.x, norm_mu, norm_std)
    val_calib_x = apply_normalizer(val_calib_pack.x, norm_mu, norm_std)
    val_thresh_x = apply_normalizer(val_thresh_pack.x, norm_mu, norm_std)
    test_x = apply_normalizer(test_pack.x, norm_mu, norm_std)

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    summary_rows: List[Dict[str, Any]] = []
    artifact_paths_for_progress: List[str] = [
        "src.models.advanced.sequence_arch.py",
        "src/models/train_advanced.py",
    ]

    for model_name in models:
        run_name = f"adv_{model_name}_ws{int(args.window_size)}_st{int(args.stride)}_e{int(args.epochs)}_s{int(args.seed)}"
        output_dir = (repo_root / args.output_root / run_name).resolve()
        report_dir = (repo_root / args.report_root / run_name).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = (report_dir / "plots").resolve()
        prob_plot_dir = (plots_dir / "probability").resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        prob_plot_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(
            model_name=model_name, input_dim=input_dim, dropout=float(args.dropout)
        )
        model, history, best_auc, best_epoch = train_one_model(
            model=model,
            train_x=train_x,
            train_y=train_pack.y,
            val_x=val_thresh_x,
            val_y=val_thresh_pack.y,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            patience=int(args.patience),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            focal_gamma=float(args.focal_gamma),
            max_grad_norm=float(args.max_grad_norm),
            device=device,
        )

        val_calib_logit = predict_logits(
            model=model, x=val_calib_x, batch_size=int(args.batch_size), device=device
        )
        val_thresh_logit = predict_logits(
            model=model, x=val_thresh_x, batch_size=int(args.batch_size), device=device
        )
        test_logit = predict_logits(
            model=model, x=test_x, batch_size=int(args.batch_size), device=device
        )
        val_calib_prob_raw = sigmoid_np(val_calib_logit)
        val_thresh_prob_raw = sigmoid_np(val_thresh_logit)
        test_prob_raw = sigmoid_np(test_logit)

        calibrator = ProbabilityCalibrator(method=str(args.calibration_method)).fit(
            y_true=val_calib_pack.y,
            y_prob=val_calib_prob_raw,
        )
        val_thresh_prob_cal = calibrator.predict(val_thresh_prob_raw)
        test_prob_cal = calibrator.predict(test_prob_raw)
        cal_delta = calibration_quality_delta(
            y_true=val_thresh_pack.y,
            prob_before=val_thresh_prob_raw,
            prob_after=val_thresh_prob_cal,
        )

        val_timeline = val_thresh_pack.timeline.copy()
        val_timeline["prob_raw"] = val_thresh_prob_raw
        val_timeline["prob_cal"] = val_thresh_prob_cal
        theta, theta_diag = choose_threshold_by_shot_fpr(
            timeline_df=val_timeline,
            sustain_ms=float(args.sustain_ms),
            max_shot_fpr=float(args.threshold_max_shot_fpr),
        )
        theta_diag = dict(theta_diag)
        theta_diag["objective"] = "shot_fpr_constrained"

        test_timeline = test_pack.timeline.copy()
        test_timeline["prob_raw"] = test_prob_raw
        test_timeline["prob_cal"] = test_prob_cal

        val_metrics_cal = compute_binary_metrics(
            y_true=val_thresh_pack.y,
            y_prob=val_thresh_prob_cal,
            threshold=float(theta),
        )
        test_metrics_cal = compute_binary_metrics(
            y_true=test_pack.y,
            y_prob=test_prob_cal,
            threshold=float(theta),
        )
        shot_warn_test = apply_shot_warning_policy(
            timeline_df=test_timeline,
            threshold=float(theta),
            sustain_ms=float(args.sustain_ms),
        )
        shot_metrics_test = compute_shot_level_metrics(shot_warn_test)

        contrib = compute_gradient_input_attribution(
            model=model,
            x=test_x,
            batch_size=max(1, int(args.batch_size // 2)),
            device=device,
        )
        reason_df = compute_disruption_reasons_per_shot(
            contrib_by_window=contrib,
            timeline_df=test_timeline,
            shot_summary=shot_warn_test,
            features=features,
            top_k=int(args.reason_top_k),
        )

        test_shot_ids = sorted(
            test_timeline["shot_id"].drop_duplicates().astype(int).tolist()
        )
        if bool(args.plot_all_test_shots):
            selected_shots = test_shot_ids
        else:
            limit = int(max(args.plot_shot_limit, 0))
            selected_shots = test_shot_ids[:limit] if limit > 0 else []
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
        test_timeline.to_csv(plots_dir / "probability_timelines_test.csv", index=False)

        training_config = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_name": model_name,
            "device": str(device),
            "data_root": str(data_root),
            "hdf5_root": str(hdf5_root),
            "features": features,
            "feature_count": int(len(features)),
            "window": {
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "eval_stride": eval_stride,
            },
            "optimizer": {
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "focal_gamma": float(args.focal_gamma),
            },
            "labeling": {
                "gray_ms": float(args.gray_ms),
                "fallback_fls_ms": float(args.fallback_fls_ms),
                "fallback_dt_ms": float(args.fallback_dt_ms),
            },
            "calibration": {
                "method": str(args.calibration_method),
                "calibration_shot_fraction": float(args.calibration_shot_fraction),
            },
            "threshold_policy": {
                "objective": "shot_fpr_constrained",
                "max_shot_fpr": float(args.threshold_max_shot_fpr),
                "theta": float(theta),
                "sustain_ms": float(args.sustain_ms),
                "selection_diag": theta_diag,
            },
            "split_counts": {
                "train_shots": int(len(train_shots)),
                "val_calib_shots": int(len(val_calib_shots)),
                "val_thresh_shots": int(len(val_thresh_shots)),
                "test_shots": int(len(test_shots)),
                "train_windows": int(train_x.shape[0]),
                "val_calib_windows": int(val_calib_x.shape[0]),
                "val_thresh_windows": int(val_thresh_x.shape[0]),
                "test_windows": int(test_x.shape[0]),
            },
            "data_loading_meta": {
                "train": train_meta,
                "val_calib": val_calib_meta,
                "val_thresh": val_thresh_meta,
                "test": test_meta,
                "train_short_shots_for_window": train_pack.short_shots,
                "val_calib_short_shots_for_window": val_calib_pack.short_shots,
                "val_thresh_short_shots_for_window": val_thresh_pack.short_shots,
                "test_short_shots_for_window": test_pack.short_shots,
            },
            "normalization": {
                "feature_mean": [float(x) for x in norm_mu],
                "feature_std": [float(x) for x in norm_std],
            },
            "plotting": {
                "plot_all_test_shots": bool(args.plot_all_test_shots),
                "plot_shot_limit": int(args.plot_shot_limit),
                "test_shot_count": int(len(test_shot_ids)),
                "generated_timeline_png": int(timeline_plot_count),
            },
            "training_history": history,
            "best_epoch": int(best_epoch),
            "best_val_roc_auc_raw": float(best_auc) if np.isfinite(best_auc) else None,
        }

        metrics_summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "model_name": model_name,
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "eval_stride": eval_stride,
            "val_timepoint_calibrated": val_metrics_cal,
            "test_timepoint_calibrated": test_metrics_cal,
            "test_shot_policy": shot_metrics_test,
            "threshold_policy": {
                "objective": "shot_fpr_constrained",
                "max_shot_fpr": float(args.threshold_max_shot_fpr),
                "theta": float(theta),
                "sustain_ms": float(args.sustain_ms),
                "selection_diag": theta_diag,
            },
            "calibration_delta_val_threshold": cal_delta,
            "reason_summary": {
                "reason_top_k": int(args.reason_top_k),
                "reason_rows": int(len(reason_df)),
                "disruptive_shots_test": int((shot_warn_test["shot_label"] == 1).sum()),
            },
            "plotting": {
                "plot_all_test_shots": bool(args.plot_all_test_shots),
                "plot_shot_limit": int(args.plot_shot_limit),
                "test_shot_count": int(len(test_shot_ids)),
                "generated_timeline_png": int(timeline_plot_count),
            },
        }

        (output_dir / "training_config.json").write_text(
            json.dumps(training_config, indent=2), encoding="utf-8"
        )
        (output_dir / "metrics_summary.json").write_text(
            json.dumps(metrics_summary, indent=2), encoding="utf-8"
        )

        # Save PyTorch model checkpoint
        checkpoint_path = output_dir / f"{model_name}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "input_dim": input_dim,
                "dropout": float(args.dropout),
                "best_epoch": int(best_epoch),
                "best_val_roc_auc_raw": float(best_auc)
                if np.isfinite(best_auc)
                else None,
                "normalization": {
                    "feature_mean": norm_mu.tolist(),
                    "feature_std": norm_std.tolist(),
                },
            },
            checkpoint_path,
        )

        shot_warn_test.to_csv(output_dir / "warning_summary_test.csv", index=False)
        reason_df.to_csv(
            output_dir / "disruption_reason_per_shot.csv", index=False, encoding="utf-8"
        )
        write_metrics_markdown(report_dir / "metrics.md", metrics_summary)

        # ---- Threshold stability analysis (industrial validation) ----
        if args.run_stability:
            from src.evaluation.threshold_stability import run_stability_analysis

            stability_dir = output_dir / "stability"
            print(f"\n=== [{model_name}] Running Threshold Stability Analysis ===")
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
            print(
                f"\n=== [{model_name}] Stability Verdict: {stability_report.verdict} ==="
            )
            for reason in stability_report.verdict_reasons:
                print(f"  - {reason}")

        artifact_paths_for_progress.extend(
            [
                train_base.to_repo_rel(output_dir / "training_config.json", repo_root),
                train_base.to_repo_rel(output_dir / "metrics_summary.json", repo_root),
                train_base.to_repo_rel(
                    output_dir / "warning_summary_test.csv", repo_root
                ),
                train_base.to_repo_rel(
                    output_dir / "disruption_reason_per_shot.csv", repo_root
                ),
                train_base.to_repo_rel(report_dir / "metrics.md", repo_root),
                train_base.to_repo_rel(
                    plots_dir / "probability_timelines_test.csv", repo_root
                ),
                train_base.to_repo_rel(prob_plot_dir, repo_root),
            ]
        )

        summary_rows.append(
            {
                "run_name": run_name,
                "model_name": model_name,
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "test_accuracy": float(test_metrics_cal["accuracy"]),
                "test_roc_auc": float(test_metrics_cal["roc_auc"]),
                "test_pr_auc": float(test_metrics_cal["pr_auc"]),
                "test_tpr": float(test_metrics_cal["tpr"]),
                "test_fpr": float(test_metrics_cal["fpr"]),
                "test_ece": float(test_metrics_cal["ece_15_bins"]),
                "shot_accuracy": float(shot_metrics_test["shot_accuracy"]),
                "shot_tpr": float(shot_metrics_test["shot_tpr"]),
                "shot_fpr": float(shot_metrics_test["shot_fpr"]),
                "lead_time_ms_median": float(shot_metrics_test["lead_time_ms_median"]),
                "theta": float(theta),
            }
        )

    if not summary_rows:
        raise RuntimeError("No advanced runs completed.")

    summary_csv = (repo_root / args.report_root / "advanced_summary.csv").resolve()
    summary_md = (repo_root / args.report_root / "advanced_summary.md").resolve()
    write_advanced_summary(summary_rows, summary_csv=summary_csv, summary_md=summary_md)

    best = max(
        summary_rows,
        key=lambda r: (
            float(r["shot_accuracy"]),
            float(r["test_roc_auc"]),
            float(r["test_accuracy"]),
        ),
    )
    artifact_paths_for_progress.extend(
        [
            train_base.to_repo_rel(summary_csv, repo_root),
            train_base.to_repo_rel(summary_md, repo_root),
        ]
    )
    update_progress_agent6(
        progress_path=(repo_root / "docs/progress.md").resolve(),
        best_row=best,
        artifacts=artifact_paths_for_progress,
    )

    print(
        json.dumps(
            {
                "best_run": best,
                "summary_csv": train_base.to_repo_rel(summary_csv, repo_root),
                "summary_md": train_base.to_repo_rel(summary_md, repo_root),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
