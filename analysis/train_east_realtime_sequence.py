"""
Realtime EAST disruption probability training + streaming prediction.

Core design:
- Train on EAST TV split (train/val split done by shot).
- Test strictly on original TEST split (as requested).
- Per-time labels:
  - disruptive shot:
    y=1 for [DownTime-AdvancedTime, end_of_sequence]
    y=-1 for previous uncertain window (ignored in training/eval)
    y=0 for earlier timestamps
  - non-disruptive shot: y=0 for all timestamps

This script provides two modes:
1) train: train a realtime sequence model on TV and evaluate on original TEST.
2) predict: load best checkpoint and run streaming prediction on an N x F matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_KEYS: List[str] = [
    "Bt",
    "ip",
    "ne_nG",
    "n=1 amplitude",
    "Mir_avg_amp",
    "Mir_avg_fre",
    "v_loop",
    "dx_a",
    "dy_a",
    "sxr_mean",
    "xuv_ratio",
]


@dataclass
class ShotSequence:
    shot_id: int
    x: np.ndarray  # [T, F], float32
    y: np.ndarray  # [T], int8, in {-1, 0, 1}
    time_s: np.ndarray  # [T], float64
    is_disruptive: bool
    down_time: float


@dataclass
class FeatureScaler:
    mean: np.ndarray  # [F]
    std: np.ndarray  # [F]
    keys: List[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = x.astype(np.float32, copy=True)
        finite = np.isfinite(arr)
        if not finite.all():
            arr = np.where(finite, arr, self.mean[None, :])
        arr = (arr - self.mean[None, :]) / self.std[None, :]
        return arr.astype(np.float32, copy=False)

    def to_json(self) -> Dict[str, object]:
        return {
            "keys": self.keys,
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @staticmethod
    def from_json(obj: Dict[str, object]) -> "FeatureScaler":
        keys = [str(k) for k in obj["keys"]]
        mean = np.asarray(obj["mean"], dtype=np.float32)
        std = np.asarray(obj["std"], dtype=np.float32)
        return FeatureScaler(mean=mean, std=std, keys=keys)


def _safe_float(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    if not math.isfinite(x):
        return float("nan")
    return x


def _load_json_int_list(path: Path) -> List[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [int(x) for x in raw]


def _load_advance_time(path: Path) -> Dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, float] = {}
    for k, v in raw.items():
        shot = int(k)
        vv = _safe_float(v)
        if math.isfinite(vv):
            out[shot] = vv
    return out


def _resolve_hdf5_path(hdf5_root: Path, shot_id: int) -> Path:
    folder = str((shot_id // 100) * 100)
    candidate = hdf5_root / folder / f"{shot_id}.hdf5"
    if candidate.exists():
        return candidate
    fallback = list(hdf5_root.rglob(f"{shot_id}.hdf5"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Missing HDF5 for shot={shot_id} under {hdf5_root}")


def _pick_length(data_group: h5py.Group, keys: Sequence[str]) -> int:
    lengths: List[int] = []
    for key in keys:
        if key not in data_group:
            continue
        ds = data_group[key]
        if hasattr(ds, "shape") and ds.shape:
            lengths.append(int(ds.shape[0]))
        else:
            lengths.append(len(ds))  # pragma: no cover - fallback
    if not lengths:
        raise RuntimeError("No valid feature dataset found in HDF5 data group.")
    return int(min(lengths))


def _infer_dt_seconds(meta_group: h5py.Group, n: int, start_time: float, down_time: float) -> float:
    if "time_interval" in meta_group:
        dt = _safe_float(meta_group["time_interval"][()])
        if math.isfinite(dt) and dt > 0:
            return float(dt)
    if n > 1 and math.isfinite(start_time) and math.isfinite(down_time) and down_time > start_time:
        dt = (down_time - start_time) / float(n - 1)
        if dt > 0:
            return float(dt)
    return 1e-3


def _build_labels(
    time_s: np.ndarray,
    down_time: float,
    is_disruptive: bool,
    advance_ms: float,
    uncertain_ms: float,
) -> np.ndarray:
    y = np.zeros(time_s.shape[0], dtype=np.int8)
    if (not is_disruptive) or (not math.isfinite(down_time)):
        return y

    adv_s = max(float(advance_ms), 0.0) / 1000.0
    uncertain_s = max(float(uncertain_ms), 0.0) / 1000.0
    pos_start = down_time - adv_s
    uncertain_start = pos_start - uncertain_s

    if uncertain_s > 0:
        uncertain_mask = (time_s >= uncertain_start) & (time_s < pos_start)
        y[uncertain_mask] = -1

    # Cumulative risk label:
    # once entering the pre-disruption onset window, keep positive afterwards.
    pos_mask = time_s >= pos_start
    y[pos_mask] = 1
    return y


def _read_shot_sequence(
    file_path: Path,
    shot_id: int,
    is_disruptive: bool,
    advance_ms: float,
    uncertain_ms: float,
    keys: Sequence[str],
) -> ShotSequence:
    with h5py.File(file_path, "r") as f:
        data_group = f["data"]
        meta_group = f["meta"]

        n = _pick_length(data_group, keys)
        start_time = _safe_float(meta_group["StartTime"][()]) if "StartTime" in meta_group else 0.0
        down_time = _safe_float(meta_group["DownTime"][()]) if "DownTime" in meta_group else float("nan")
        dt = _infer_dt_seconds(meta_group=meta_group, n=n, start_time=start_time, down_time=down_time)
        time_s = start_time + np.arange(n, dtype=np.float64) * dt

        cols: List[np.ndarray] = []
        for key in keys:
            if key in data_group:
                arr = np.asarray(data_group[key][:n], dtype=np.float32).reshape(-1)
                if arr.shape[0] < n:
                    pad = np.full((n - arr.shape[0],), np.nan, dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = np.full((n,), np.nan, dtype=np.float32)
            cols.append(arr)
        x = np.stack(cols, axis=1).astype(np.float32, copy=False)
        y = _build_labels(
            time_s=time_s,
            down_time=down_time,
            is_disruptive=is_disruptive,
            advance_ms=advance_ms,
            uncertain_ms=uncertain_ms,
        )
    return ShotSequence(
        shot_id=int(shot_id),
        x=x,
        y=y,
        time_s=time_s,
        is_disruptive=bool(is_disruptive),
        down_time=float(down_time),
    )


def _collect_sequences(
    hdf5_root: Path,
    dis_list: Sequence[int],
    non_list: Sequence[int],
    adv_map: Dict[int, float],
    uncertain_ms: float,
    keys: Sequence[str],
) -> Tuple[Dict[int, ShotSequence], Dict[str, int]]:
    stats = {
        "requested_disruptive": len(dis_list),
        "requested_non_disruptive": len(non_list),
        "loaded_disruptive": 0,
        "loaded_non_disruptive": 0,
        "failed": 0,
    }
    sequences: Dict[int, ShotSequence] = {}
    t0 = time.time()
    for i, shot_id in enumerate(dis_list, start=1):
        try:
            path = _resolve_hdf5_path(hdf5_root=hdf5_root, shot_id=int(shot_id))
            adv_ms = float(adv_map.get(int(shot_id), 50.0))
            seq = _read_shot_sequence(
                file_path=path,
                shot_id=int(shot_id),
                is_disruptive=True,
                advance_ms=adv_ms,
                uncertain_ms=uncertain_ms,
                keys=keys,
            )
            sequences[int(shot_id)] = seq
            stats["loaded_disruptive"] += 1
        except Exception:
            stats["failed"] += 1
        if (i % 50) == 0 or i == len(dis_list):
            print(
                f"[INFO] loading disruptive {i}/{len(dis_list)} "
                f"(ok={stats['loaded_disruptive']} fail={stats['failed']}) "
                f"elapsed={time.time() - t0:.1f}s"
            )

    for j, shot_id in enumerate(non_list, start=1):
        try:
            path = _resolve_hdf5_path(hdf5_root=hdf5_root, shot_id=int(shot_id))
            seq = _read_shot_sequence(
                file_path=path,
                shot_id=int(shot_id),
                is_disruptive=False,
                advance_ms=0.0,
                uncertain_ms=uncertain_ms,
                keys=keys,
            )
            sequences[int(shot_id)] = seq
            stats["loaded_non_disruptive"] += 1
        except Exception:
            stats["failed"] += 1
        if (j % 50) == 0 or j == len(non_list):
            print(
                f"[INFO] loading non-disruptive {j}/{len(non_list)} "
                f"(ok={stats['loaded_non_disruptive']} fail={stats['failed']}) "
                f"elapsed={time.time() - t0:.1f}s"
            )

    return sequences, stats


def _split_shots(
    disruptive_ids: Sequence[int],
    non_ids: Sequence[int],
    val_ratio: float,
    random_seed: int,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(int(random_seed))
    dis_ids = np.asarray(sorted(set(int(x) for x in disruptive_ids)), dtype=np.int64)
    non_ids_arr = np.asarray(sorted(set(int(x) for x in non_ids)), dtype=np.int64)
    rng.shuffle(dis_ids)
    rng.shuffle(non_ids_arr)

    n_dis_val = max(1, int(round(dis_ids.size * float(val_ratio)))) if dis_ids.size > 1 else 0
    n_non_val = max(1, int(round(non_ids_arr.size * float(val_ratio)))) if non_ids_arr.size > 1 else 0

    val_ids = np.concatenate([dis_ids[:n_dis_val], non_ids_arr[:n_non_val]]).tolist()
    train_ids = np.concatenate([dis_ids[n_dis_val:], non_ids_arr[n_non_val:]]).tolist()
    if not train_ids or not val_ids:
        raise RuntimeError("Train/val split failed: empty split.")
    return [int(x) for x in train_ids], [int(x) for x in val_ids]


def _subsample_ids(ids: Sequence[int], max_count: int, seed: int, tag: str) -> List[int]:
    arr = np.asarray([int(x) for x in ids], dtype=np.int64)
    if max_count < 0:
        print(f"[INFO] subsample {tag}: {arr.size} -> 0 (disabled)")
        return []
    if max_count == 0 or arr.size <= max_count:
        return arr.tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(arr)
    out = arr[: int(max_count)].tolist()
    print(f"[INFO] subsample {tag}: {arr.size} -> {len(out)}")
    return [int(x) for x in out]


def _build_augmented_keys(base_keys: Sequence[str], add_diff: bool, add_abs_diff: bool) -> List[str]:
    out = [str(k) for k in base_keys]
    if add_diff:
        out.extend(f"{k}__diff" for k in base_keys)
    if add_abs_diff:
        out.extend(f"{k}__absdiff" for k in base_keys)
    return out


def _paper131_fluctuation_scale(base_keys: Sequence[str]) -> np.ndarray:
    """
    Prior from paper_131 SHAP/physics discussion:
    - Stronger temporal sensitivity: Mir_fre, Mir_abs, ne, ip, v_loop.
    - n=1 amplitude is useful but less time-sensitive.
    """
    weights = np.ones((len(base_keys),), dtype=np.float32)
    for i, key in enumerate(base_keys):
        k = str(key).strip().lower()
        if ("mir_avg_fre" in k) or ("mir_fre" in k):
            weights[i] = 2.0
        elif ("mir_avg_amp" in k) or ("mir_abs" in k):
            weights[i] = 1.8
        elif ("ne_ng" in k) or (k == "ne") or ("density" in k):
            weights[i] = 1.6
        elif (k == "ip") or ("current" in k):
            weights[i] = 1.4
        elif "v_loop" in k:
            weights[i] = 1.5
        elif ("n=1 amplitude" in k) or ("n=1" in k):
            weights[i] = 1.2
        elif ("dx_a" in k) or ("dy_a" in k):
            weights[i] = 1.3
        elif ("sxr" in k) or ("xuv" in k):
            weights[i] = 1.2
    return weights


def _augment_feature_matrix(
    x: np.ndarray,
    base_keys: Sequence[str],
    add_diff: bool,
    add_abs_diff: bool,
    use_paper131_prior: bool,
) -> np.ndarray:
    base = np.asarray(x, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={base.shape}")

    parts: List[np.ndarray] = [base]
    if add_diff or add_abs_diff:
        diff = np.diff(base, axis=0, prepend=base[:1])
        if use_paper131_prior:
            scale = _paper131_fluctuation_scale(base_keys=base_keys)
            diff = diff * scale[None, :]
        if add_diff:
            parts.append(diff.astype(np.float32, copy=False))
        if add_abs_diff:
            parts.append(np.abs(diff).astype(np.float32, copy=False))

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _apply_feature_augmentation_inplace(
    sequences: Dict[int, ShotSequence],
    base_keys: Sequence[str],
    add_diff: bool,
    add_abs_diff: bool,
    use_paper131_prior: bool,
) -> None:
    if (not add_diff) and (not add_abs_diff):
        return
    for sid in list(sequences.keys()):
        seq = sequences[sid]
        seq.x = _augment_feature_matrix(
            seq.x,
            base_keys=base_keys,
            add_diff=add_diff,
            add_abs_diff=add_abs_diff,
            use_paper131_prior=use_paper131_prior,
        )


def _fit_scaler(sequences: Dict[int, ShotSequence], shot_ids: Sequence[int], keys: Sequence[str]) -> FeatureScaler:
    fdim = len(keys)
    sum_ = np.zeros((fdim,), dtype=np.float64)
    sq_ = np.zeros((fdim,), dtype=np.float64)
    cnt = np.zeros((fdim,), dtype=np.float64)

    for sid in shot_ids:
        x = sequences[int(sid)].x
        finite = np.isfinite(x)
        xf = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        sum_ += xf.sum(axis=0)
        sq_ += (xf * xf).sum(axis=0)
        cnt += finite.sum(axis=0)

    safe_cnt = np.maximum(cnt, 1.0)
    mean = sum_ / safe_cnt
    var = sq_ / safe_cnt - mean * mean
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)
    std = np.where(std > 1e-6, std, 1.0)
    return FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32), keys=list(keys))


def _apply_scaler_inplace(sequences: Dict[int, ShotSequence], scaler: FeatureScaler) -> None:
    for sid in list(sequences.keys()):
        sequences[sid].x = scaler.transform(sequences[sid].x)


def _count_labels(sequences: Dict[int, ShotSequence], shot_ids: Sequence[int]) -> Tuple[int, int]:
    n_pos = 0
    n_neg = 0
    for sid in shot_ids:
        y = sequences[int(sid)].y
        n_pos += int((y == 1).sum())
        n_neg += int((y == 0).sum())
    return n_pos, n_neg


class SequenceDataset(Dataset):
    def __init__(self, sequences: Dict[int, ShotSequence], shot_ids: Sequence[int]) -> None:
        self.items: List[ShotSequence] = [sequences[int(s)] for s in shot_ids]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq = self.items[idx]
        return seq.x, seq.y, seq.time_s, seq.shot_id, seq.is_disruptive, seq.down_time


def _collate_sequences(batch):
    lengths = [int(item[0].shape[0]) for item in batch]
    bsz = len(batch)
    max_len = int(max(lengths))
    feat_dim = int(batch[0][0].shape[1])

    x = np.zeros((bsz, max_len, feat_dim), dtype=np.float32)
    y = np.full((bsz, max_len), -1, dtype=np.float32)
    mask = np.zeros((bsz, max_len), dtype=np.float32)

    shot_ids: List[int] = []
    is_disruptive: List[bool] = []
    down_times: List[float] = []
    time_list: List[np.ndarray] = []

    for i, (xi, yi, ti, sid, is_dis, down_t) in enumerate(batch):
        n = xi.shape[0]
        x[i, :n] = xi
        y[i, :n] = yi.astype(np.float32, copy=False)
        mask[i, :n] = (yi >= 0).astype(np.float32, copy=False)
        shot_ids.append(int(sid))
        is_disruptive.append(bool(is_dis))
        down_times.append(float(down_t))
        time_list.append(np.asarray(ti, dtype=np.float64))

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask),
        torch.tensor(lengths, dtype=torch.long),
        shot_ids,
        time_list,
        is_disruptive,
        down_times,
    )


class RealtimeGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, h0: Optional[torch.Tensor] = None):
        # Keep the forward path cuDNN-safe by using padded tensors directly.
        # Loss/eval masking already excludes padded timesteps, so this is stable
        # and avoids packed-sequence non-contiguous issues on some CUDA setups.
        _ = lengths
        out, h = self.gru(x.contiguous(), h0)
        logits = self.classifier(out).squeeze(-1)
        return logits, h

    def step(self, x_t: torch.Tensor, h: Optional[torch.Tensor] = None):
        """
        x_t: [B, F]
        returns logits: [B], new_hidden
        """
        out, h_new = self.gru(x_t.unsqueeze(1), h)
        logits = self.classifier(out.squeeze(1)).squeeze(-1)
        return logits, h_new

    def forward_prob(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0 = state.get("hidden") if isinstance(state, dict) else None
        logits, h = self.forward(x, lengths=lengths, h0=h0)
        prob = torch.sigmoid(logits)
        return prob, {"hidden": h}

    def step_prob(
        self, x_t: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0 = state.get("hidden") if isinstance(state, dict) else None
        logits, h = self.step(x_t, h=h0)
        prob = torch.sigmoid(logits)
        return prob, {"hidden": h}


class RealtimeHazardGRU(nn.Module):
    """
    Causal monotonic risk model.
    It predicts instantaneous risk logits, then applies cumulative max:
    risk_t = max_{i<=t}(sigmoid(logit_i)).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _risk_from_base_prob(
        base_prob: torch.Tensor, prev_max: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = base_prob.clamp(min=1e-6, max=1.0 - 1e-6)
        if prev_max is None:
            risk = torch.cummax(p, dim=1).values
        else:
            prefix = prev_max.clamp(min=1e-6, max=1.0 - 1e-6).unsqueeze(1)
            merged = torch.cat([prefix, p], dim=1)
            risk = torch.cummax(merged, dim=1).values[:, 1:]
        return risk, risk[:, -1]

    def forward_prob(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _ = lengths
        h0 = state.get("hidden") if isinstance(state, dict) else None
        prev_max = state.get("risk_max") if isinstance(state, dict) else None
        z = self.input_layer(x)
        out, h = self.gru(z.contiguous(), h0)
        base_prob = torch.sigmoid(self.hazard_head(out).squeeze(-1))
        risk, risk_max = self._risk_from_base_prob(base_prob=base_prob, prev_max=prev_max)
        new_state = {"hidden": h, "risk_max": risk_max}
        return risk, new_state

    def step_prob(
        self, x_t: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h0 = state.get("hidden") if isinstance(state, dict) else None
        risk_prev = state.get("risk_max") if isinstance(state, dict) else None

        z_t = self.input_layer(x_t)
        out, h = self.gru(z_t.unsqueeze(1), h0)
        base_t = torch.sigmoid(self.hazard_head(out.squeeze(1)).squeeze(-1)).clamp(
            min=1e-6, max=1.0 - 1e-6
        )
        if risk_prev is None:
            risk_t = base_t
        else:
            risk_t = torch.maximum(risk_prev, base_t)
        new_state = {"hidden": h, "risk_max": risk_t}
        return risk_t, new_state


def _metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_i = y_true.astype(np.int64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true_i)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true_i, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true_i, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _trigger_scan(prob: np.ndarray, threshold: float, consecutive: int) -> Tuple[int, np.ndarray, np.ndarray]:
    streak = 0
    streak_arr = np.zeros_like(prob, dtype=np.int32)
    triggered = np.zeros_like(prob, dtype=np.int8)
    trigger_idx = -1
    need = max(1, int(consecutive))

    for i, p in enumerate(prob):
        if p >= threshold:
            streak += 1
        else:
            streak = 0
        streak_arr[i] = streak
        if trigger_idx < 0 and streak >= need:
            trigger_idx = i
        if trigger_idx >= 0:
            triggered[i] = 1
    return trigger_idx, streak_arr, triggered


def _shot_level_metrics(
    shot_results: Dict[int, Dict[str, object]],
    threshold: float,
    consecutive: int,
) -> Dict[str, float]:
    dis_total = 0
    dis_hit = 0
    non_total = 0
    non_false_alarm = 0
    lead_times_ms: List[float] = []

    for sid, row in shot_results.items():
        _ = sid
        prob = np.asarray(row["prob"], dtype=np.float64)
        time_s = np.asarray(row["time_s"], dtype=np.float64)
        is_dis = bool(row["is_disruptive"])
        down_t = _safe_float(row["down_time"])
        trig_idx, _, _ = _trigger_scan(prob=prob, threshold=threshold, consecutive=consecutive)
        trig_t = float(time_s[trig_idx]) if trig_idx >= 0 and trig_idx < time_s.shape[0] else float("nan")

        if is_dis:
            dis_total += 1
            if trig_idx >= 0 and (not math.isfinite(down_t) or trig_t <= down_t):
                dis_hit += 1
                if math.isfinite(down_t):
                    lead_times_ms.append((down_t - trig_t) * 1000.0)
        else:
            non_total += 1
            if trig_idx >= 0:
                non_false_alarm += 1

    out: Dict[str, float] = {
        "disruptive_shot_recall": float(dis_hit / dis_total) if dis_total > 0 else float("nan"),
        "non_disruptive_false_alarm_rate": float(non_false_alarm / non_total) if non_total > 0 else float("nan"),
        "n_disruptive_shots": float(dis_total),
        "n_non_disruptive_shots": float(non_total),
        "mean_lead_time_ms": float(np.mean(lead_times_ms)) if lead_times_ms else float("nan"),
        "median_lead_time_ms": float(np.median(lead_times_ms)) if lead_times_ms else float("nan"),
    }
    return out


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, object]]]:
    model.eval()
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []
    shot_results: Dict[int, Dict[str, object]] = {}

    with torch.no_grad():
        for x, y, mask, lengths, shot_ids, time_list, is_disruptive, down_times in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
            lengths = lengths.to(device=device)

            if not hasattr(model, "forward_prob"):
                raise RuntimeError("Model must implement forward_prob(...)")
            prob, _ = model.forward_prob(x, lengths=lengths, state=None)

            prob_np = prob.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy().astype(bool)
            lengths_np = lengths.detach().cpu().numpy()

            for i, sid in enumerate(shot_ids):
                n = int(lengths_np[i])
                valid = mask_np[i, :n]
                if valid.any():
                    yy = y_np[i, :n][valid].astype(np.int64, copy=False)
                    pp = prob_np[i, :n][valid].astype(np.float64, copy=False)
                    y_true_list.append(yy)
                    y_prob_list.append(pp)

                shot_results[int(sid)] = {
                    "shot_id": int(sid),
                    "time_s": np.asarray(time_list[i][:n], dtype=np.float64),
                    "y_true": np.asarray(y_np[i, :n], dtype=np.int64),
                    "prob": np.asarray(prob_np[i, :n], dtype=np.float64),
                    "is_disruptive": bool(is_disruptive[i]),
                    "down_time": float(down_times[i]),
                }

    if not y_true_list:
        raise RuntimeError("No valid labels found while collecting predictions.")
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    return y_true, y_prob, shot_results


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    pos_weight: float,
    terminal_pos_lambda: float,
    terminal_neg_lambda: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_weight = 0.0

    for x, y, mask, lengths, _, _, is_disruptive, _ in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = lengths.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        valid = mask > 0.5
        targets = torch.where(valid, (y > 0.5).float(), torch.zeros_like(y))

        if not hasattr(model, "forward_prob"):
            raise RuntimeError("Model must implement forward_prob(...)")
        prob, _ = model.forward_prob(x, lengths=lengths, state=None)
        prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)

        weight = torch.where(
            targets > 0.5,
            torch.full_like(targets, float(pos_weight)),
            torch.ones_like(targets),
        )
        weight = weight * valid.float()
        denom = weight.sum().clamp(min=1.0)
        loss_raw = F.binary_cross_entropy(prob, targets, reduction="none")
        loss = (loss_raw * weight).sum() / denom

        if (float(terminal_pos_lambda) > 0) or (float(terminal_neg_lambda) > 0):
            batch_idx = torch.arange(prob.shape[0], device=prob.device)
            last_idx = (lengths - 1).clamp(min=0)
            final_prob = prob[batch_idx, last_idx]
            is_dis = torch.tensor(is_disruptive, dtype=torch.bool, device=prob.device)
            if bool(is_dis.any().item()) and float(terminal_pos_lambda) > 0:
                pos_loss = F.binary_cross_entropy(
                    final_prob[is_dis],
                    torch.ones_like(final_prob[is_dis]),
                    reduction="mean",
                )
                loss = loss + float(terminal_pos_lambda) * pos_loss
            non_mask = ~is_dis
            if bool(non_mask.any().item()) and float(terminal_neg_lambda) > 0:
                non_loss = F.binary_cross_entropy(
                    final_prob[non_mask],
                    torch.zeros_like(final_prob[non_mask]),
                    reduction="mean",
                )
                loss = loss + float(terminal_neg_lambda) * non_loss

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        running_loss += float(loss.detach().item()) * float(denom.detach().item())
        total_weight += float(denom.detach().item())

    return running_loss / max(total_weight, 1.0)


def _export_sequence_predictions(
    shot_results: Dict[int, Dict[str, object]],
    output_dir: Path,
    split_name: str,
    threshold: float,
    consecutive: int,
) -> None:
    seq_dir = output_dir / "sequence_predictions" / split_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for sid in sorted(shot_results.keys()):
        row = shot_results[sid]
        time_s = np.asarray(row["time_s"], dtype=np.float64)
        y_true = np.asarray(row["y_true"], dtype=np.int64)
        prob = np.asarray(row["prob"], dtype=np.float64)
        is_dis = bool(row["is_disruptive"])
        down_t = _safe_float(row["down_time"])

        trig_idx, streak_arr, triggered_arr = _trigger_scan(
            prob=prob, threshold=threshold, consecutive=consecutive
        )
        trig_t = float(time_s[trig_idx]) if trig_idx >= 0 and trig_idx < time_s.shape[0] else float("nan")
        prob_clamped = np.asarray(prob, dtype=np.float64).copy()
        if trig_idx >= 0:
            prob_clamped[trig_idx:] = 1.0

        df = pd.DataFrame(
            {
                "shot_id": int(sid),
                "time_s": time_s,
                "label": y_true,
                "valid_label": (y_true >= 0).astype(np.int8),
                "prob_disrupt": prob,
                "prob_disrupt_clamped": prob_clamped,
                "streak": streak_arr,
                "pred_disruptive": (prob >= threshold).astype(np.int8),
                "triggered": triggered_arr,
            }
        )
        df.to_csv(seq_dir / f"{int(sid)}.csv", index=False)

        summary_rows.append(
            {
                "shot_id": int(sid),
                "is_disruptive": int(is_dis),
                "down_time_s": float(down_t),
                "trigger_index": int(trig_idx),
                "trigger_time_s": float(trig_t),
                "triggered": int(trig_idx >= 0),
                "lead_time_ms": float((down_t - trig_t) * 1000.0)
                if (is_dis and trig_idx >= 0 and math.isfinite(down_t))
                else float("nan"),
            }
        )

    pd.DataFrame(summary_rows).to_csv(seq_dir / "_summary.csv", index=False)


def _build_model_from_checkpoint(ckpt: Dict[str, object]) -> nn.Module:
    model_cfg = ckpt.get("model", {})
    model_type = str(model_cfg.get("model_type", "gru")).strip().lower()
    common_kwargs = {
        "input_dim": int(model_cfg["input_dim"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dropout": float(model_cfg["dropout"]),
    }
    if model_type in {"hazard_gru", "risk_gru"}:
        return RealtimeHazardGRU(**common_kwargs)
    return RealtimeGRU(**common_kwargs)


def _load_matrix_for_predict(path: Path, keys: Sequence[str]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        x = np.load(path)
        if x.ndim != 2:
            raise ValueError(f"Numpy matrix must be 2D, got shape={x.shape}")
        if x.shape[1] != len(keys):
            raise ValueError(
                f"Feature mismatch: matrix has {x.shape[1]} cols but checkpoint expects {len(keys)}"
            )
        return x.astype(np.float32, copy=False)

    df = pd.read_csv(path)
    if all(k in df.columns for k in keys):
        x = df[list(keys)].to_numpy(dtype=np.float32, copy=False)
    else:
        x = df.to_numpy(dtype=np.float32, copy=False)
        if x.ndim != 2 or x.shape[1] != len(keys):
            raise ValueError(
                f"CSV feature mismatch: got shape={x.shape}, expected N x {len(keys)} "
                f"or columns named {list(keys)}"
            )
    return x.astype(np.float32, copy=False)


def _predict_stream_matrix(
    model: nn.Module,
    x_norm: np.ndarray,
    device: torch.device,
    trigger_prob: float,
    trigger_consecutive: int,
    dt_ms: float,
) -> pd.DataFrame:
    model.eval()
    probs: List[float] = []
    state: Optional[Dict[str, torch.Tensor]] = None

    with torch.no_grad():
        for i in range(x_norm.shape[0]):
            x_t = torch.from_numpy(x_norm[i : i + 1]).to(device=device, dtype=torch.float32)
            if not hasattr(model, "step_prob"):
                raise RuntimeError("Model must implement step_prob(...)")
            prob_t, state = model.step_prob(x_t, state)
            p = float(prob_t.item())
            probs.append(float(p))

    prob_np = np.asarray(probs, dtype=np.float64)
    trig_idx, streak_arr, triggered_arr = _trigger_scan(
        prob=prob_np, threshold=trigger_prob, consecutive=trigger_consecutive
    )
    prob_clamped = prob_np.copy()
    if trig_idx >= 0:
        prob_clamped[trig_idx:] = 1.0
    time_s = np.arange(x_norm.shape[0], dtype=np.float64) * (float(dt_ms) / 1000.0)

    out = pd.DataFrame(
        {
            "time_index": np.arange(x_norm.shape[0], dtype=np.int64),
            "time_s": time_s,
            "prob_disrupt": prob_np,
            "prob_disrupt_clamped": prob_clamped,
            "streak": streak_arr,
            "pred_disruptive": (prob_np >= trigger_prob).astype(np.int8),
            "triggered": triggered_arr,
            "trigger_index": int(trig_idx),
            "trigger_time_s": float(time_s[trig_idx]) if trig_idx >= 0 else float("nan"),
        }
    )
    return out

def _train_mode(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    hdf5_root = Path(args.hdf5_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.keys.strip().lower() == "default":
        base_keys = list(DEFAULT_KEYS)
    elif args.keys.strip().lower() == "all":
        # Discover from one file quickly.
        sample = _resolve_hdf5_path(hdf5_root=hdf5_root, shot_id=53825)
        with h5py.File(sample, "r") as f:
            base_keys = sorted(list(f["data"].keys()))
    else:
        base_keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        if not base_keys:
            raise ValueError("--keys parsed empty.")

    add_diff = bool(int(args.add_diff))
    add_abs_diff = bool(int(args.add_abs_diff))
    use_paper131_prior = bool(int(args.use_paper131_prior))
    keys = _build_augmented_keys(
        base_keys=base_keys,
        add_diff=add_diff,
        add_abs_diff=add_abs_diff,
    )

    tv_dis = _load_json_int_list(repo_root / "shot_list" / "EAST" / "Disruption_EAST_TV.json")
    tv_non = _load_json_int_list(repo_root / "shot_list" / "EAST" / "Non-disruption_EAST_TV.json")
    test_dis = _load_json_int_list(repo_root / "shot_list" / "TEST" / "Disruption_EAST_Test.json")
    test_non = _load_json_int_list(repo_root / "shot_list" / "TEST" / "Non-disruption_EAST_Test.json")
    adv_map = _load_advance_time(repo_root / "shot_list" / "EAST" / "AdvancedTime_EAST.json")

    tv_dis = _subsample_ids(tv_dis, int(args.max_disrupt_tv), int(args.random_seed) + 11, "tv_disrupt")
    tv_non = _subsample_ids(tv_non, int(args.max_nondisrupt_tv), int(args.random_seed) + 17, "tv_non_disrupt")
    test_dis = _subsample_ids(test_dis, int(args.max_disrupt_test), int(args.random_seed) + 23, "test_disrupt")
    test_non = _subsample_ids(test_non, int(args.max_nondisrupt_test), int(args.random_seed) + 29, "test_non_disrupt")

    test_set = set(test_dis) | set(test_non)
    tv_dis_set = set(tv_dis)
    tv_non_set = set(tv_non)
    overlap = sorted((tv_dis_set | tv_non_set) & test_set)
    if overlap:
        # Prevent leakage into step-3 original TEST evaluation.
        tv_dis_set = {x for x in tv_dis_set if x not in overlap}
        tv_non_set = {x for x in tv_non_set if x not in overlap}
        print(f"[WARN] Removed overlap TV∩TEST shots to avoid leakage: {overlap}")

    train_ids, val_ids = _split_shots(
        disruptive_ids=sorted(tv_dis_set),
        non_ids=sorted(tv_non_set),
        val_ratio=float(args.val_ratio),
        random_seed=int(args.random_seed),
    )
    print("=== Split Summary ===")
    print(f"TV disruptive: {len(tv_dis_set)}")
    print(f"TV non-disruptive: {len(tv_non_set)}")
    print(f"Train shots: {len(train_ids)}")
    print(f"Val shots:   {len(val_ids)}")
    print(f"TEST shots:  {len(test_set)} (original split)")

    print("\n=== Loading TV Sequences ===")
    tv_sequences, tv_load_stats = _collect_sequences(
        hdf5_root=hdf5_root,
        dis_list=sorted(tv_dis_set),
        non_list=sorted(tv_non_set),
        adv_map=adv_map,
        uncertain_ms=float(args.uncertain_ms),
        keys=base_keys,
    )
    for k, v in tv_load_stats.items():
        print(f"{k}: {v}")

    # Keep split IDs that are successfully loaded.
    train_ids = [sid for sid in train_ids if sid in tv_sequences]
    val_ids = [sid for sid in val_ids if sid in tv_sequences]
    if not train_ids or not val_ids:
        raise RuntimeError("No train/val shots after loading.")

    _apply_feature_augmentation_inplace(
        sequences=tv_sequences,
        base_keys=base_keys,
        add_diff=add_diff,
        add_abs_diff=add_abs_diff,
        use_paper131_prior=use_paper131_prior,
    )

    scaler = _fit_scaler(sequences=tv_sequences, shot_ids=train_ids, keys=keys)
    _apply_scaler_inplace(sequences=tv_sequences, scaler=scaler)

    n_pos, n_neg = _count_labels(tv_sequences, train_ids)
    if n_pos <= 0:
        raise RuntimeError("Training labels contain no positive points.")
    raw_pos_weight = max(float(n_neg) / float(max(n_pos, 1)), 1.0)
    max_pos_weight = float(args.max_pos_weight)
    if max_pos_weight > 0:
        pos_weight = min(raw_pos_weight, max_pos_weight)
    else:
        pos_weight = raw_pos_weight
    print("\n=== Label Stats (train points) ===")
    print(f"n_pos: {n_pos}")
    print(f"n_neg: {n_neg}")
    print(f"raw_pos_weight: {raw_pos_weight:.6f}")
    print(f"pos_weight: {pos_weight:.6f}")
    print(
        f"terminal_loss: pos_lambda={float(args.terminal_pos_lambda):.3f} "
        f"neg_lambda={float(args.terminal_neg_lambda):.3f}"
    )
    print(
        f"feature_aug: add_diff={int(add_diff)} add_abs_diff={int(add_abs_diff)} "
        f"use_paper131_prior={int(use_paper131_prior)} "
        f"(base_dim={len(base_keys)} -> model_dim={len(keys)})"
    )

    train_ds = SequenceDataset(tv_sequences, train_ids)
    val_ds = SequenceDataset(tv_sequences, val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate_sequences,
    )
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate_sequences,
    )

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    print(f"\n=== Device ===\n{device}")

    model_type = str(args.model_type).strip().lower()
    if model_type in {"hazard_gru", "risk_gru"}:
        model = RealtimeHazardGRU(
            input_dim=len(keys),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        ).to(device)
    elif model_type == "gru":
        model = RealtimeGRU(
            input_dim=len(keys),
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        ).to(device)
    else:
        raise ValueError(f"Unsupported --model-type: {args.model_type}")

    print(f"model_type: {model_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_score = -float("inf")
    best_epoch = -1
    best_ckpt = output_dir / "best_model.pt"
    patience_left = int(args.patience)
    history_rows: List[Dict[str, float]] = []

    print("\n=== Training ===")
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(args.grad_clip),
            pos_weight=float(pos_weight),
            terminal_pos_lambda=float(args.terminal_pos_lambda),
            terminal_neg_lambda=float(args.terminal_neg_lambda),
        )

        y_val_true, y_val_prob, val_shot_results = _collect_predictions(
            model=model, loader=val_loader, device=device
        )
        val_point_metrics = _metric_dict(
            y_true=y_val_true, y_prob=y_val_prob, threshold=float(args.trigger_prob)
        )
        val_shot_metrics = _shot_level_metrics(
            shot_results=val_shot_results,
            threshold=float(args.trigger_prob),
            consecutive=int(args.trigger_consecutive),
        )
        val_pr_auc = float(val_point_metrics.get("pr_auc", float("nan")))
        monitor = val_pr_auc if math.isfinite(val_pr_auc) else float(val_point_metrics["f1"])

        history_row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_accuracy": float(val_point_metrics["accuracy"]),
            "val_precision": float(val_point_metrics["precision"]),
            "val_recall": float(val_point_metrics["recall"]),
            "val_f1": float(val_point_metrics["f1"]),
            "val_roc_auc": float(val_point_metrics["roc_auc"]),
            "val_pr_auc": float(val_point_metrics["pr_auc"]),
            "val_shot_recall": float(val_shot_metrics["disruptive_shot_recall"]),
            "val_false_alarm_rate": float(val_shot_metrics["non_disruptive_false_alarm_rate"]),
        }
        history_rows.append(history_row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_f1={val_point_metrics['f1']:.6f} "
            f"val_pr_auc={val_point_metrics['pr_auc']:.6f} "
            f"val_shot_recall={val_shot_metrics['disruptive_shot_recall']:.6f} "
            f"val_far={val_shot_metrics['non_disruptive_false_alarm_rate']:.6f}"
        )

        if monitor > best_score:
            best_score = monitor
            best_epoch = epoch
            patience_left = int(args.patience)
            ckpt = {
                "model_state": model.state_dict(),
                "model": {
                    "model_type": model_type,
                    "input_dim": len(keys),
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                },
                "scaler": scaler.to_json(),
                "keys": keys,
                "base_keys": base_keys,
                "feature_aug": {
                    "add_diff": int(add_diff),
                    "add_abs_diff": int(add_abs_diff),
                    "use_paper131_prior": int(use_paper131_prior),
                },
                "trigger": {
                    "prob": float(args.trigger_prob),
                    "consecutive": int(args.trigger_consecutive),
                },
                "meta": {
                    "best_epoch": int(best_epoch),
                    "best_score": float(best_score),
                    "val_ratio": float(args.val_ratio),
                    "random_seed": int(args.random_seed),
                    "uncertain_ms": float(args.uncertain_ms),
                    "raw_pos_weight": float(raw_pos_weight),
                    "pos_weight": float(pos_weight),
                    "max_pos_weight": float(args.max_pos_weight),
                    "terminal_pos_lambda": float(args.terminal_pos_lambda),
                    "terminal_neg_lambda": float(args.terminal_neg_lambda),
                    "overlap_removed_from_tv": overlap,
                },
            }
            torch.save(ckpt, best_ckpt)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best_epoch={best_epoch}).")
                break

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

    if not best_ckpt.exists():
        raise RuntimeError("Best checkpoint not saved.")

    print(f"\nBest checkpoint: {best_ckpt} (epoch={best_epoch}, score={best_score:.6f})")

    # Reload best for final val/test export.
    ckpt_obj = torch.load(best_ckpt, map_location=device)
    model = _build_model_from_checkpoint(ckpt_obj).to(device)
    model.load_state_dict(ckpt_obj["model_state"])
    scaler = FeatureScaler.from_json(ckpt_obj["scaler"])

    # Final val export with best model.
    y_val_true, y_val_prob, val_shot_results = _collect_predictions(model=model, loader=val_loader, device=device)
    val_point_metrics = _metric_dict(y_true=y_val_true, y_prob=y_val_prob, threshold=float(args.trigger_prob))
    val_shot_metrics = _shot_level_metrics(
        shot_results=val_shot_results,
        threshold=float(args.trigger_prob),
        consecutive=int(args.trigger_consecutive),
    )
    _export_sequence_predictions(
        shot_results=val_shot_results,
        output_dir=output_dir,
        split_name="val",
        threshold=float(args.trigger_prob),
        consecutive=int(args.trigger_consecutive),
    )

    # Step-3 requested by user: evaluate on original TEST split.
    print("\n=== Loading TEST Sequences (original TEST split) ===")
    test_sequences, test_load_stats = _collect_sequences(
        hdf5_root=hdf5_root,
        dis_list=test_dis,
        non_list=test_non,
        adv_map=adv_map,
        uncertain_ms=float(args.uncertain_ms),
        keys=base_keys,
    )
    for k, v in test_load_stats.items():
        print(f"{k}: {v}")

    _apply_feature_augmentation_inplace(
        sequences=test_sequences,
        base_keys=base_keys,
        add_diff=add_diff,
        add_abs_diff=add_abs_diff,
        use_paper131_prior=use_paper131_prior,
    )
    _apply_scaler_inplace(sequences=test_sequences, scaler=scaler)

    test_ids = sorted(list(test_sequences.keys()))
    test_ds = SequenceDataset(test_sequences, test_ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate_sequences,
    )
    y_test_true, y_test_prob, test_shot_results = _collect_predictions(
        model=model, loader=test_loader, device=device
    )
    test_point_metrics = _metric_dict(
        y_true=y_test_true, y_prob=y_test_prob, threshold=float(args.trigger_prob)
    )
    test_shot_metrics = _shot_level_metrics(
        shot_results=test_shot_results,
        threshold=float(args.trigger_prob),
        consecutive=int(args.trigger_consecutive),
    )
    _export_sequence_predictions(
        shot_results=test_shot_results,
        output_dir=output_dir,
        split_name="test",
        threshold=float(args.trigger_prob),
        consecutive=int(args.trigger_consecutive),
    )

    print("\n=== TEST Point Metrics ===")
    for k, v in test_point_metrics.items():
        print(f"{k}: {v:.6f}")
    print("=== TEST Shot Metrics ===")
    for k, v in test_shot_metrics.items():
        print(f"{k}: {v:.6f}")
    print("=== TEST Confusion Matrix (point-level) ===")
    test_pred = (y_test_prob >= float(args.trigger_prob)).astype(np.int64)
    print(confusion_matrix(y_test_true.astype(np.int64), test_pred))

    summary = {
        "best_epoch": int(best_epoch),
        "best_monitor_score": float(best_score),
        "val_point_metrics": val_point_metrics,
        "val_shot_metrics": val_shot_metrics,
        "test_point_metrics": test_point_metrics,
        "test_shot_metrics": test_shot_metrics,
        "tv_load_stats": tv_load_stats,
        "test_load_stats": test_load_stats,
        "n_train_shots": len(train_ids),
        "n_val_shots": len(val_ids),
        "n_test_shots": len(test_ids),
        "raw_pos_weight": float(raw_pos_weight),
        "pos_weight": float(pos_weight),
        "max_pos_weight": float(args.max_pos_weight),
        "terminal_pos_lambda": float(args.terminal_pos_lambda),
        "terminal_neg_lambda": float(args.terminal_neg_lambda),
        "model_type": model_type,
        "feature_keys": keys,
        "base_feature_keys": base_keys,
        "feature_aug": {
            "add_diff": int(add_diff),
            "add_abs_diff": int(add_abs_diff),
            "use_paper131_prior": int(use_paper131_prior),
        },
        "trigger_prob": float(args.trigger_prob),
        "trigger_consecutive": int(args.trigger_consecutive),
        "uncertain_ms": float(args.uncertain_ms),
        "overlap_removed_from_tv": overlap,
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved outputs under: {output_dir}")
    print(f"Val sequence predictions:  {output_dir / 'sequence_predictions' / 'val'}")
    print(f"Test sequence predictions: {output_dir / 'sequence_predictions' / 'test'}")
    print(f"Metrics: {output_dir / 'metrics_summary.json'}")


def _predict_mode(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint).resolve()
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    ckpt = torch.load(ckpt_path, map_location=device)
    scaler = FeatureScaler.from_json(ckpt["scaler"])
    model_keys = scaler.keys
    base_keys_obj = ckpt.get("base_keys", model_keys)
    base_keys = [str(k) for k in base_keys_obj]
    aug_cfg = ckpt.get("feature_aug", {})
    add_diff = bool(int(aug_cfg.get("add_diff", 0)))
    add_abs_diff = bool(int(aug_cfg.get("add_abs_diff", 0)))
    use_paper131_prior = bool(int(aug_cfg.get("use_paper131_prior", 0)))

    model = _build_model_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = _load_matrix_for_predict(path=Path(args.matrix).resolve(), keys=base_keys)
    x = _augment_feature_matrix(
        x,
        base_keys=base_keys,
        add_diff=add_diff,
        add_abs_diff=add_abs_diff,
        use_paper131_prior=use_paper131_prior,
    )
    if x.shape[1] != len(model_keys):
        raise ValueError(
            f"Augmented feature dimension mismatch: got {x.shape[1]} vs checkpoint {len(model_keys)}"
        )
    x_norm = scaler.transform(x)

    trigger_prob = float(args.trigger_prob)
    trigger_consecutive = int(args.trigger_consecutive)
    if trigger_prob <= 0:
        trigger_prob = float(ckpt.get("trigger", {}).get("prob", 0.5))
    if trigger_consecutive <= 0:
        trigger_consecutive = int(ckpt.get("trigger", {}).get("consecutive", 20))

    out_df = _predict_stream_matrix(
        model=model,
        x_norm=x_norm,
        device=device,
        trigger_prob=trigger_prob,
        trigger_consecutive=trigger_consecutive,
        dt_ms=float(args.dt_ms),
    )
    out_df.to_csv(out_path, index=False)

    trigger_idx = int(out_df["trigger_index"].iloc[0])
    trigger_time = float(out_df["trigger_time_s"].iloc[0]) if trigger_idx >= 0 else float("nan")
    print(f"Saved streaming prediction CSV: {out_path}")
    print(f"n_rows: {out_df.shape[0]}")
    print(f"trigger_idx: {trigger_idx}")
    print(f"trigger_time_s: {trigger_time}")
    print(f"used_base_keys: {base_keys}")
    print(f"used_model_keys: {model_keys}")
    print(
        f"feature_aug: add_diff={int(add_diff)} add_abs_diff={int(add_abs_diff)} "
        f"use_paper131_prior={int(use_paper131_prior)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EAST realtime disruption training and streaming prediction.")
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train", help="Train on TV and evaluate on original TEST.")
    tr.add_argument("--repo-root", type=str, default=".")
    tr.add_argument("--hdf5-root", type=str, default="data/EAST/unified_hdf5")
    tr.add_argument("--output-dir", type=str, default="analysis/outputs/realtime_gru")
    tr.add_argument("--keys", type=str, default="default", help="'default', 'all', or comma-separated keys")
    tr.add_argument("--add-diff", type=int, default=1, help="1 enables first-order temporal difference features")
    tr.add_argument(
        "--add-abs-diff", type=int, default=1, help="1 enables absolute first-order difference features"
    )
    tr.add_argument(
        "--use-paper131-prior",
        type=int,
        default=1,
        help="1 scales fluctuation features using paper_131 feature-importance priors",
    )
    tr.add_argument("--uncertain-ms", type=float, default=30.0)
    tr.add_argument("--max-disrupt-tv", type=int, default=0, help="0 means no limit; <0 disables")
    tr.add_argument("--max-nondisrupt-tv", type=int, default=0, help="0 means no limit; <0 disables")
    tr.add_argument("--max-disrupt-test", type=int, default=0, help="0 means no limit; <0 disables")
    tr.add_argument("--max-nondisrupt-test", type=int, default=0, help="0 means no limit; <0 disables")
    tr.add_argument("--val-ratio", type=float, default=0.2)
    tr.add_argument("--random-seed", type=int, default=42)

    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--eval-batch-size", type=int, default=8, help="<=0 means use --batch-size")
    tr.add_argument("--num-workers", type=int, default=0)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--grad-clip", type=float, default=5.0)
    tr.add_argument("--patience", type=int, default=8)
    tr.add_argument("--max-pos-weight", type=float, default=50.0, help="<=0 disables capping")
    tr.add_argument(
        "--terminal-pos-lambda",
        type=float,
        default=2.0,
        help="penalty weight to push final risk high on disruptive shots",
    )
    tr.add_argument(
        "--terminal-neg-lambda",
        "--non-terminal-lambda",
        dest="terminal_neg_lambda",
        type=float,
        default=0.5,
        help="penalty weight to keep final risk low on non-disruptive shots",
    )

    tr.add_argument("--hidden-dim", type=int, default=128)
    tr.add_argument("--num-layers", type=int, default=2)
    tr.add_argument("--dropout", type=float, default=0.2)
    tr.add_argument("--model-type", type=str, default="hazard_gru", help="hazard_gru or gru")

    tr.add_argument("--trigger-prob", type=float, default=0.5)
    tr.add_argument("--trigger-consecutive", type=int, default=20)
    tr.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")

    pr = sub.add_parser("predict", help="Streaming predict on custom N x F matrix.")
    pr.add_argument("--checkpoint", type=str, required=True)
    pr.add_argument("--matrix", type=str, required=True, help="CSV/NPY matrix path")
    pr.add_argument("--output-csv", type=str, default="analysis/outputs/realtime_predict.csv")
    pr.add_argument("--dt-ms", type=float, default=1.0)
    pr.add_argument("--trigger-prob", type=float, default=-1.0, help="<=0 means use checkpoint default")
    pr.add_argument(
        "--trigger-consecutive", type=int, default=-1, help="<=0 means use checkpoint default"
    )
    pr.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    if args.mode == "train":
        _train_mode(args)
    elif args.mode == "predict":
        _predict_mode(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
