#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


def _sinusoidal_positional_encoding(
    length: int, dim: int, device: torch.device
) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got shape={tuple(x.shape)}")
        h = self.input_proj(x)
        pe = _sinusoidal_positional_encoding(h.shape[1], h.shape[2], h.device)
        h = h + pe.unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class MambaLiteBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.in_proj = nn.Linear(dim, dim)
        self.state_in = nn.Linear(dim, state_dim)
        self.decay_proj = nn.Linear(dim, state_dim)
        self.select_proj = nn.Linear(dim, state_dim)
        self.state_out = nn.Linear(state_dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        mixed = self.mix(h.transpose(1, 2)).transpose(1, 2)
        u = torch.tanh(self.in_proj(mixed))
        x_state = torch.tanh(self.state_in(mixed))
        decay = torch.sigmoid(self.decay_proj(mixed))
        select = torch.sigmoid(self.select_proj(mixed))

        bsz, seq_len, _ = u.shape
        state = torch.zeros(bsz, x_state.shape[-1], device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            state = decay[:, t, :] * state + (1.0 - decay[:, t, :]) * x_state[:, t, :]
            ys.append((select[:, t, :] * state).unsqueeze(1))
        y = torch.cat(ys, dim=1)
        y = self.state_out(y)
        g = torch.sigmoid(self.gate(mixed))
        out = self.out_proj(g * y + (1.0 - g) * u)
        return x + self.drop(out)


class MambaLiteClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 96,
        state_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaLiteBlock(d_model, state_dim=state_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got shape={tuple(x.shape)}")
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        pooled = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=(dropout if n_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F], got shape={tuple(x.shape)}")
        h, _ = self.gru(x)
        pooled = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        pooled = self.norm(pooled)
        return self.head(pooled).squeeze(-1)
