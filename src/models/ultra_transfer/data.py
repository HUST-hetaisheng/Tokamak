#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.models.advanced import train_sequence as adv_seq
from src.models.baseline import train_xgb as train_base
from src.models.ultra_transfer.config import TransferWorkspaceConfig


@dataclass
class PreparedTransferData:
    """Prepared datasets/windows for transfer training and evaluation."""

    repo_root: Path
    data_root: Path
    hdf5_root: Path
    features: List[str]
    input_dim: int

    train_pack: adv_seq.WindowPack
    val_calib_pack: adv_seq.WindowPack
    val_thresh_pack: adv_seq.WindowPack
    test_pack: adv_seq.WindowPack

    train_x: np.ndarray
    val_calib_x: np.ndarray
    val_thresh_x: np.ndarray
    test_x: np.ndarray

    norm_mu: np.ndarray
    norm_std: np.ndarray

    train_meta: Dict[str, Any]
    val_calib_meta: Dict[str, Any]
    val_thresh_meta: Dict[str, Any]
    test_meta: Dict[str, Any]


def _augment_physics_dynamics(
    x: np.ndarray,
    feature_names: List[str],
    eps: float,
) -> tuple[np.ndarray, List[str]]:
    """Append temporal-change channels to emphasize physics evolution.

    Channels appended:
    - First-order delta: x_t - x_{t-1}
    - Relative rate: (x_t - x_{t-1}) / (abs(x_{t-1}) + eps)
    """
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    d1 = x - prev
    dr = d1 / (np.abs(prev) + float(max(eps, 1e-12)))
    out = np.concatenate([x, d1, dr], axis=2).astype(np.float32)

    names = list(feature_names)
    names.extend([f"d1__{f}" for f in feature_names])
    names.extend([f"dr__{f}" for f in feature_names])
    return out, names


def _count_labels(
    shot_ids: Sequence[int],
    label_map: Mapping[int, int],
) -> dict[str, int]:
    arr = [int(label_map.get(int(sid), 0)) for sid in shot_ids]
    n_pos = int(sum(1 for v in arr if v == 1))
    n_neg = int(sum(1 for v in arr if v == 0))
    return {"total": int(len(arr)), "n_pos": n_pos, "n_neg": n_neg}


def _move_one_of_class(
    src: list[int],
    dst: list[int],
    label_map: Mapping[int, int],
    target_cls: int,
) -> bool:
    for sid in sorted(src):
        if int(label_map.get(int(sid), 0)) == int(target_cls):
            src.remove(sid)
            dst.append(sid)
            return True
    return False


def _rebalance_val_subsets_for_binary(
    calib_ids: Sequence[int],
    thresh_ids: Sequence[int],
    label_map: Mapping[int, int],
) -> tuple[list[int], list[int], dict[str, int]]:
    calib = [int(x) for x in calib_ids]
    thresh = [int(x) for x in thresh_ids]
    moves = {"to_calib": 0, "to_thresh": 0}

    for cls in (1, 0):
        calib_stats = _count_labels(calib, label_map)
        thresh_stats = _count_labels(thresh, label_map)

        if (
            calib_stats[f"n_{'pos' if cls == 1 else 'neg'}"] == 0
            and thresh_stats[f"n_{'pos' if cls == 1 else 'neg'}"] > 1
        ):
            if _move_one_of_class(thresh, calib, label_map, target_cls=cls):
                moves["to_calib"] += 1

        calib_stats = _count_labels(calib, label_map)
        thresh_stats = _count_labels(thresh, label_map)
        if (
            thresh_stats[f"n_{'pos' if cls == 1 else 'neg'}"] == 0
            and calib_stats[f"n_{'pos' if cls == 1 else 'neg'}"] > 1
        ):
            if _move_one_of_class(calib, thresh, label_map, target_cls=cls):
                moves["to_thresh"] += 1

    return sorted(calib), sorted(thresh), moves


def _require_binary_split(
    *,
    split_name: str,
    shot_ids: Sequence[int],
    label_map: Mapping[int, int],
) -> None:
    stats = _count_labels(shot_ids, label_map)
    if stats["n_pos"] <= 0 or stats["n_neg"] <= 0:
        raise RuntimeError(
            f"{split_name} split must contain both classes for method-correct training/evaluation. "
            f"Got stats={stats}. Consider increasing --max-*-shots or adjusting split files."
        )


def _assert_disjoint_partition(
    *,
    parent_name: str,
    parent_ids: Sequence[int],
    left_name: str,
    left_ids: Sequence[int],
    right_name: str,
    right_ids: Sequence[int],
) -> None:
    parent_list = [int(x) for x in parent_ids]
    left_list = [int(x) for x in left_ids]
    right_list = [int(x) for x in right_ids]

    if len(left_list) != len(set(left_list)):
        raise RuntimeError(f"{left_name} contains duplicated shot IDs")
    if len(right_list) != len(set(right_list)):
        raise RuntimeError(f"{right_name} contains duplicated shot IDs")

    inter = set(left_list).intersection(set(right_list))
    if inter:
        sample = sorted(inter)[:10]
        raise RuntimeError(
            f"{left_name} and {right_name} overlap; sample overlap IDs={sample}"
        )

    parent_set = set(parent_list)
    union_set = set(left_list).union(set(right_list))
    if union_set != parent_set:
        missing = sorted(parent_set.difference(union_set))[:10]
        extra = sorted(union_set.difference(parent_set))[:10]
        raise RuntimeError(
            f"{left_name}+{right_name} is not a full partition of {parent_name}; "
            f"missing_sample={missing}, extra_sample={extra}"
        )


def _assert_splits_disjoint(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
) -> None:
    train_set = set(int(x) for x in train_ids)
    val_set = set(int(x) for x in val_ids)
    test_set = set(int(x) for x in test_ids)
    if train_set.intersection(val_set):
        raise RuntimeError("train and val splits overlap")
    if train_set.intersection(test_set):
        raise RuntimeError("train and test splits overlap")
    if val_set.intersection(test_set):
        raise RuntimeError("val and test splits overlap")


def prepare_transfer_data(cfg: TransferWorkspaceConfig) -> PreparedTransferData:
    """Load splits, build windows, and normalize for isolated transfer runs."""
    repo_root = cfg.repo_root.resolve()
    data_root = cfg.data_root.resolve()
    hdf5_root = (data_root / cfg.hdf5_subdir).resolve()
    if not hdf5_root.exists():
        raise FileNotFoundError(f"HDF5 root not found: {hdf5_root}")

    features_path = (
        repo_root / cfg.dataset_artifact_dir / "required_features.json"
    ).resolve()
    features = adv_seq.load_features(features_path)

    train_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / cfg.split_dir / "train.txt").resolve()),
        int(cfg.max_train_shots),
        int(cfg.seed),
    )
    val_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / cfg.split_dir / "val.txt").resolve()),
        int(cfg.max_val_shots),
        int(cfg.seed),
    )
    test_ids = train_base.take_bounded(
        train_base.read_split_ids((repo_root / cfg.split_dir / "test.txt").resolve()),
        int(cfg.max_test_shots),
        int(cfg.seed),
    )

    label_map = train_base.load_label_map(
        (repo_root / cfg.dataset_artifact_dir / "clean_shots.csv").resolve()
    )
    val_calib_ids, val_thresh_ids = train_base.split_val_for_calibration_and_threshold(
        shot_ids=val_ids,
        label_map=label_map,
        calibration_fraction=0.5,
        seed=int(cfg.seed),
    )
    val_calib_ids, val_thresh_ids, rebalance_meta = _rebalance_val_subsets_for_binary(
        calib_ids=val_calib_ids,
        thresh_ids=val_thresh_ids,
        label_map=label_map,
    )

    if bool(cfg.strict_method_checks):
        _assert_splits_disjoint(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )
        _assert_disjoint_partition(
            parent_name="val",
            parent_ids=val_ids,
            left_name="val_calib",
            left_ids=val_calib_ids,
            right_name="val_thresh",
            right_ids=val_thresh_ids,
        )
        _require_binary_split(
            split_name="train",
            shot_ids=train_ids,
            label_map=label_map,
        )
        _require_binary_split(
            split_name="val_calib",
            shot_ids=val_calib_ids,
            label_map=label_map,
        )
        _require_binary_split(
            split_name="val_thresh",
            shot_ids=val_thresh_ids,
            label_map=label_map,
        )
    advanced_map = train_base.read_advanced_map(
        (repo_root / "shot_list/J-TEXT/AdvancedTime_J-TEXT.json").resolve()
    )
    hdf5_idx = train_base.build_hdf5_index(hdf5_root)

    train_shots, train_meta = adv_seq.load_split_data(
        split_name="train",
        shot_ids=train_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(cfg.gray_ms),
        fallback_fls_ms=float(cfg.fallback_fls_ms),
        fallback_dt_ms=float(cfg.fallback_dt_ms),
        reconcile_len_tol=int(cfg.reconcile_len_tol),
    )
    val_calib_shots, val_calib_meta = adv_seq.load_split_data(
        split_name="val_calib",
        shot_ids=val_calib_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(cfg.gray_ms),
        fallback_fls_ms=float(cfg.fallback_fls_ms),
        fallback_dt_ms=float(cfg.fallback_dt_ms),
        reconcile_len_tol=int(cfg.reconcile_len_tol),
    )
    val_thresh_shots, val_thresh_meta = adv_seq.load_split_data(
        split_name="val_thresh",
        shot_ids=val_thresh_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(cfg.gray_ms),
        fallback_fls_ms=float(cfg.fallback_fls_ms),
        fallback_dt_ms=float(cfg.fallback_dt_ms),
        reconcile_len_tol=int(cfg.reconcile_len_tol),
    )
    test_shots, test_meta = adv_seq.load_split_data(
        split_name="test",
        shot_ids=test_ids,
        hdf5_idx=hdf5_idx,
        features=features,
        label_map=label_map,
        advanced_map=advanced_map,
        gray_ms=float(cfg.gray_ms),
        fallback_fls_ms=float(cfg.fallback_fls_ms),
        fallback_dt_ms=float(cfg.fallback_dt_ms),
        reconcile_len_tol=int(cfg.reconcile_len_tol),
    )

    train_pack = adv_seq.build_window_pack(
        split_name="train",
        shots=train_shots,
        window_size=int(cfg.window_size),
        stride=int(cfg.stride),
        pad_short_shots=bool(cfg.pad_short_shots),
        short_pad_mode=str(cfg.short_pad_mode),
    )
    val_calib_pack = adv_seq.build_window_pack(
        split_name="val_calib",
        shots=val_calib_shots,
        window_size=int(cfg.window_size),
        stride=int(cfg.eval_stride),
        pad_short_shots=bool(cfg.pad_short_shots),
        short_pad_mode=str(cfg.short_pad_mode),
    )
    val_thresh_pack = adv_seq.build_window_pack(
        split_name="val_thresh",
        shots=val_thresh_shots,
        window_size=int(cfg.window_size),
        stride=int(cfg.eval_stride),
        pad_short_shots=bool(cfg.pad_short_shots),
        short_pad_mode=str(cfg.short_pad_mode),
    )
    test_pack = adv_seq.build_window_pack(
        split_name="test",
        shots=test_shots,
        window_size=int(cfg.window_size),
        stride=int(cfg.eval_stride),
        pad_short_shots=bool(cfg.pad_short_shots),
        short_pad_mode=str(cfg.short_pad_mode),
    )

    model_features = list(features)
    train_x_raw = train_pack.x
    val_calib_x_raw = val_calib_pack.x
    val_thresh_x_raw = val_thresh_pack.x
    test_x_raw = test_pack.x

    if bool(cfg.augment_dynamics):
        train_x_raw, model_features = _augment_physics_dynamics(
            train_x_raw, model_features, eps=float(cfg.dynamics_eps)
        )
        val_calib_x_raw, _ = _augment_physics_dynamics(
            val_calib_x_raw, list(features), eps=float(cfg.dynamics_eps)
        )
        val_thresh_x_raw, _ = _augment_physics_dynamics(
            val_thresh_x_raw, list(features), eps=float(cfg.dynamics_eps)
        )
        test_x_raw, _ = _augment_physics_dynamics(
            test_x_raw, list(features), eps=float(cfg.dynamics_eps)
        )

    norm_mu, norm_std = adv_seq.fit_normalizer(train_x_raw)
    train_x = adv_seq.apply_normalizer(train_x_raw, norm_mu, norm_std)
    val_calib_x = adv_seq.apply_normalizer(val_calib_x_raw, norm_mu, norm_std)
    val_thresh_x = adv_seq.apply_normalizer(val_thresh_x_raw, norm_mu, norm_std)
    test_x = adv_seq.apply_normalizer(test_x_raw, norm_mu, norm_std)

    train_meta = dict(train_meta)
    val_calib_meta = dict(val_calib_meta)
    val_thresh_meta = dict(val_thresh_meta)
    train_meta["label_stats"] = _count_labels(train_ids, label_map)
    val_calib_meta["label_stats"] = _count_labels(val_calib_ids, label_map)
    val_thresh_meta["label_stats"] = _count_labels(val_thresh_ids, label_map)
    val_calib_meta["rebalance_moves"] = dict(rebalance_meta)

    return PreparedTransferData(
        repo_root=repo_root,
        data_root=data_root,
        hdf5_root=hdf5_root,
        features=model_features,
        input_dim=int(len(model_features)),
        train_pack=train_pack,
        val_calib_pack=val_calib_pack,
        val_thresh_pack=val_thresh_pack,
        test_pack=test_pack,
        train_x=train_x,
        val_calib_x=val_calib_x,
        val_thresh_x=val_thresh_x,
        test_x=test_x,
        norm_mu=norm_mu,
        norm_std=norm_std,
        train_meta=train_meta,
        val_calib_meta=val_calib_meta,
        val_thresh_meta=val_thresh_meta,
        test_meta=test_meta,
    )
