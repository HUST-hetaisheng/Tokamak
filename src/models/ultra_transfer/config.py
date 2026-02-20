#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransferWorkspaceConfig:
    """Configuration for isolated ultra-transfer experiments.

    This config intentionally defaults to dedicated output/report roots so
    artifacts do not mix with previous pipelines.
    """

    repo_root: Path = Path(".")
    data_root: Path = Path("G:/我的云端硬盘/Fuison/data")
    hdf5_subdir: str = "J-TEXT/unified_hdf5"
    dataset_artifact_dir: Path = Path("artifacts/datasets/jtext_v1")
    split_dir: Path = Path("splits")

    output_root: Path = Path("artifacts/models/ultra_transfer")
    report_root: Path = Path("reports/ultra_transfer")

    seed: int = 42
    gray_ms: float = 50.0
    fallback_fls_ms: float = 100.0
    fallback_dt_ms: float = 1.0
    reconcile_len_tol: int = 2

    max_train_shots: int = 0
    max_val_shots: int = 0
    max_test_shots: int = 0

    window_size: int = 128
    stride: int = 16
    eval_stride: int = 1
    pad_short_shots: bool = True
    short_pad_mode: str = "edge"

    strict_method_checks: bool = True

    augment_dynamics: bool = True
    dynamics_eps: float = 1e-6
