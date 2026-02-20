#!/usr/bin/env python3
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.advanced import train_sequence as adv_seq


@dataclass
class TransferTrainOutput:
    """Model training + inference outputs for transfer workflow."""

    model: nn.Module
    history: list[dict[str, Any]]
    best_val_roc_auc_raw: float
    best_epoch: int

    val_calib_logit: np.ndarray
    val_thresh_logit: np.ndarray
    test_logit: np.ndarray

    val_calib_prob_raw: np.ndarray
    val_thresh_prob_raw: np.ndarray
    test_prob_raw: np.ndarray

    used_balanced_sampler: bool
    class_balance: dict[str, int]
    pos_weight_value: float


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_prob, dtype=np.float64).reshape(-1)))


def _build_train_loader(
    train_x: np.ndarray,
    train_y: np.ndarray,
    batch_size: int,
    use_balanced_sampler: bool,
) -> tuple[Any, bool]:
    ds = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.float32)),
    )

    y = np.asarray(train_y, dtype=np.int32).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if use_balanced_sampler and n_pos > 0 and n_neg > 0:
        pos_w = 1.0 / float(n_pos)
        neg_w = 1.0 / float(n_neg)
        sample_weights = np.where(y == 1, pos_w, neg_w).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)
        return loader, True

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, False


def _compute_val_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            j = min(i + batch_size, x.shape[0])
            xb = torch.from_numpy(x[i:j]).to(device)
            yb = torch.from_numpy(y[i:j].astype(np.float32)).to(device)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                yb,
                reduction="none",
                pos_weight=pos_weight,
            )
            if focal_gamma > 0:
                p = torch.sigmoid(logits)
                p_t = p * yb + (1.0 - p) * (1.0 - yb)
                mod = torch.pow(torch.clamp(1.0 - p_t, min=1e-8), float(focal_gamma))
                loss = torch.mean(mod * bce)
            else:
                loss = torch.mean(bce)
            total += float(loss.item()) * int(j - i)
            count += int(j - i)
    return float(total / max(count, 1))


def _train_one_model_balanced(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    focal_gamma: float,
    max_grad_norm: float,
    device: torch.device,
    use_balanced_sampler: bool,
) -> tuple[
    nn.Module,
    list[dict[str, Any]],
    float,
    int,
    bool,
    dict[str, int],
    float,
]:
    y = np.asarray(train_y, dtype=np.int32).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    train_loader, sampler_used = _build_train_loader(
        train_x=train_x,
        train_y=train_y,
        batch_size=int(batch_size),
        use_balanced_sampler=bool(use_balanced_sampler),
    )

    pos_weight_value = 1.0 if sampler_used else float(n_neg / max(n_pos, 1))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    best_epoch = 0
    best_auc = -1.0
    best_score = -1e18
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = adv_seq.binary_loss(
                logits=logits,
                y_true=yb,
                pos_weight=pos_weight,
                focal_gamma=float(focal_gamma),
            )
            loss.backward()
            if float(max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(max_grad_norm)
                )
            optimizer.step()
            running_loss += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])

        val_logits = adv_seq.predict_logits(
            model=model,
            x=val_x,
            batch_size=int(batch_size),
            device=device,
        )
        val_prob = adv_seq.sigmoid_np(val_logits)
        val_auc = _safe_roc_auc(val_y, val_prob)
        val_loss = _compute_val_loss(
            model=model,
            x=val_x,
            y=val_y,
            batch_size=int(batch_size),
            pos_weight=pos_weight,
            focal_gamma=float(focal_gamma),
            device=device,
        )

        score = float(val_auc) if np.isfinite(val_auc) else float(-val_loss)
        if score > best_score:
            best_score = score
            best_epoch = int(epoch)
            best_auc = float(val_auc) if np.isfinite(val_auc) else float("nan")
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(running_loss / max(seen, 1)),
                "val_roc_auc": float(val_auc) if np.isfinite(val_auc) else None,
                "val_loss": float(val_loss),
                "sampler": "balanced" if sampler_used else "shuffle",
            }
        )

        if best_state is not None and (int(epoch) - int(best_epoch)) >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    class_balance = {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }
    return (
        model,
        history,
        float(best_auc),
        int(best_epoch),
        bool(sampler_used),
        class_balance,
        float(pos_weight_value),
    )


def train_and_infer(
    *,
    model_name: str,
    input_dim: int,
    dropout: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_calib_y: np.ndarray,
    val_calib_x: np.ndarray,
    val_thresh_x: np.ndarray,
    test_x: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    focal_gamma: float,
    max_grad_norm: float,
    device: torch.device,
    use_balanced_sampler: bool,
) -> TransferTrainOutput:
    """Train one sequence model and return logits/probabilities for all splits."""
    model = adv_seq.build_model(
        model_name=model_name,
        input_dim=int(input_dim),
        dropout=float(dropout),
    )
    (
        model,
        history,
        best_auc,
        best_epoch,
        sampler_used,
        class_balance,
        pos_weight_value,
    ) = _train_one_model_balanced(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_calib_x,
        val_y=val_calib_y,
        batch_size=int(batch_size),
        epochs=int(epochs),
        patience=int(patience),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        focal_gamma=float(focal_gamma),
        max_grad_norm=float(max_grad_norm),
        device=device,
        use_balanced_sampler=bool(use_balanced_sampler),
    )

    val_calib_logit = adv_seq.predict_logits(
        model=model,
        x=val_calib_x,
        batch_size=int(batch_size),
        device=device,
    )
    val_thresh_logit = adv_seq.predict_logits(
        model=model,
        x=val_thresh_x,
        batch_size=int(batch_size),
        device=device,
    )
    test_logit = adv_seq.predict_logits(
        model=model,
        x=test_x,
        batch_size=int(batch_size),
        device=device,
    )

    return TransferTrainOutput(
        model=model,
        history=history,
        best_val_roc_auc_raw=float(best_auc),
        best_epoch=int(best_epoch),
        val_calib_logit=val_calib_logit,
        val_thresh_logit=val_thresh_logit,
        test_logit=test_logit,
        val_calib_prob_raw=adv_seq.sigmoid_np(val_calib_logit),
        val_thresh_prob_raw=adv_seq.sigmoid_np(val_thresh_logit),
        test_prob_raw=adv_seq.sigmoid_np(test_logit),
        used_balanced_sampler=bool(sampler_used),
        class_balance=class_balance,
        pos_weight_value=float(pos_weight_value),
    )


def compute_reason_attribution(
    *,
    model: nn.Module,
    test_x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute gradient*input attributions on test windows."""
    effective_batch = max(1, int(batch_size // 2))
    return adv_seq.compute_gradient_input_attribution(
        model=model,
        x=test_x,
        batch_size=effective_batch,
        device=device,
    )
