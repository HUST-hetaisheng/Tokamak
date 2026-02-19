#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


TIME_KEYS = {"time", "Time", "t", "T", "ttd", "TTD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MVP J-TEXT data audit + clean + split pipeline (small artifacts)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", default="jtext_v1")
    parser.add_argument("--split-ratio", default="8,1,1", help="e.g., 8,1,1 or 7,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gray-ms", type=float, default=30.0)
    parser.add_argument("--fallback-fls-ms", type=float, default=25.0)
    parser.add_argument("--fallback-dt-ms", type=float, default=1.0)
    parser.add_argument("--reconcile-len-tol", type=int, default=2)
    return parser.parse_args()


def ratio_tuple(raw: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("split ratio must have exactly three comma-separated integers")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("split ratio values must be positive")
    return vals  # type: ignore[return-value]


def detect_data_root(repo_root: Path) -> Tuple[Path, List[str], List[str]]:
    checked: List[str] = []
    ordered_candidates = [
        Path("D:/Fuison/data/J-TEXT/unified_hdf5"),
        (repo_root / "data/J-TEXT/unified_hdf5").resolve(),
    ]
    for candidate in ordered_candidates:
        c = candidate.resolve()
        checked.append(str(c))
        if c.exists() and c.is_dir():
            return c, checked, []

    hits = [p.resolve() for p in repo_root.rglob("unified_hdf5") if p.is_dir()]
    hits = sorted(set(hits), key=lambda p: str(p).lower())
    if not hits:
        raise FileNotFoundError("No unified_hdf5 directory found")
    preferred = [p for p in hits if "j-text" in str(p).lower()]
    return (preferred[0] if preferred else hits[0]), checked, [str(p) for p in hits]


def find_json(repo_root: Path, matcher) -> Path:
    files = sorted(repo_root.rglob("*.json"), key=lambda p: str(p).lower())
    for p in files:
        n = str(p).replace("\\", "/").lower()
        if matcher(n):
            return p
    raise FileNotFoundError("Required json file not found")


def discover_metadata_paths(repo_root: Path) -> Dict[str, Path]:
    disruption = find_json(
        repo_root,
        lambda n: "j-text" in n
        and "disruption" in n
        and "non-disruption" not in n
        and "nondisruption" not in n
        and "non_disruption" not in n,
    )
    nondisruption = find_json(
        repo_root,
        lambda n: "j-text" in n
        and (
            "non-disruption" in n
            or "nondisruption" in n
            or "non_disruption" in n
            or ("non" in n and "disruption" in n)
        ),
    )
    advanced = find_json(
        repo_root,
        lambda n: "j-text" in n and ("advancedtime" in n or "advanced_time" in n),
    )
    return {
        "disruption": disruption,
        "nondisruption": nondisruption,
        "advanced": advanced,
    }


def load_int_list(path: Path) -> List[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def load_advanced_map(path: Path) -> Dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def duplicates(values: Sequence[int]) -> List[int]:
    c = Counter(values)
    d = [k for k, n in c.items() if n > 1]
    d.sort()
    return d


def hdf5_index(data_root: Path) -> Dict[int, Path]:
    idx: Dict[int, Path] = {}
    for p in data_root.rglob("*.hdf5"):
        if p.stem.isdigit():
            sid = int(p.stem)
            if sid not in idx:
                idx[sid] = p
    return idx


def read_scalar(ds: Any) -> Optional[float]:
    try:
        arr = np.asarray(ds)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def get_meta_label(h5: h5py.File) -> Optional[int]:
    for key in ("meta/IsDisrupt", "meta/is_disrupt", "IsDisrupt"):
        if key in h5:
            val = read_scalar(h5[key][()])
            if val is not None:
                return int(round(val))
    return None


def get_data_group(h5: h5py.File) -> Optional[h5py.Group]:
    if "data" in h5 and isinstance(h5["data"], h5py.Group):
        return h5["data"]  # type: ignore[return-value]
    if "signals" in h5 and isinstance(h5["signals"], h5py.Group):
        return h5["signals"]  # type: ignore[return-value]
    return None


def infer_required_features(idx: Mapping[int, Path]) -> List[str]:
    for sid in sorted(idx):
        with h5py.File(idx[sid], "r") as h5:
            group = get_data_group(h5)
            if group is None:
                continue
            feats = [
                k
                for k in group.keys()
                if isinstance(group[k], h5py.Dataset)
                and group[k].ndim >= 1
                and k not in TIME_KEYS
            ]
            feats.sort()
            if feats:
                return feats
    raise RuntimeError("Could not infer required features")


def infer_time_axis_ms(h5: h5py.File, n: int, fallback_dt_ms: float) -> Tuple[np.ndarray, float, str]:
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
        t = np.linspace(start * 1000.0, down * 1000.0, n)
        dt = float(np.median(np.diff(t)))
        return t, dt, "meta_start_down"

    if down is not None:
        dt = float(fallback_dt_ms)
        t_end = down * 1000.0
        t = t_end - np.arange(n - 1, -1, -1) * dt
        return t, dt, "ttd_fallback_down"

    dt = float(fallback_dt_ms)
    t = np.arange(n) * dt
    return t, dt, "ttd_fallback_index"


def make_labels(
    t_ms: np.ndarray,
    shot_label: int,
    advanced_ms: Optional[float],
    gray_ms: float,
    fallback_fls_ms: float,
) -> Tuple[np.ndarray, np.ndarray, float, str, Optional[float], Optional[float]]:
    y = np.zeros(len(t_ms), dtype=np.int8)
    keep = np.ones(len(t_ms), dtype=bool)
    t_end = float(t_ms[-1]) if len(t_ms) else 0.0
    if shot_label == 0:
        return y, keep, 0.0, "non_disruptive", None, None

    if advanced_ms is not None and advanced_ms > 0:
        fls_ms = float(advanced_ms)
        src = "advanced_time"
    else:
        fls_ms = float(fallback_fls_ms)
        src = "fallback_25ms"
    pos_start = t_end - fls_ms
    gray_start = pos_start - gray_ms

    pos_mask = (t_ms >= pos_start) & (t_ms <= t_end + 1e-9)
    gray_mask = (t_ms >= gray_start) & (t_ms < pos_start)
    y[pos_mask] = 1
    keep[gray_mask] = False
    return y, keep, fls_ms, src, pos_start, gray_start


def allocate(n: int, r: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    total = float(sum(r))
    raw = [n * x / total for x in r]
    cnt = [int(math.floor(x)) for x in raw]
    rem = n - sum(cnt)
    order = sorted(range(3), key=lambda i: raw[i] - cnt[i], reverse=True)
    for i in range(rem):
        cnt[order[i % 3]] += 1
    if n >= 3:
        for i in range(3):
            if cnt[i] == 0:
                donor = max(range(3), key=lambda j: cnt[j])
                if cnt[donor] > 1:
                    cnt[donor] -= 1
                    cnt[i] += 1
    return cnt[0], cnt[1], cnt[2]


def stratified_split(shot_to_label: Mapping[int, int], ratio: Tuple[int, int, int], seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    pos = [sid for sid, lb in shot_to_label.items() if lb == 1]
    neg = [sid for sid, lb in shot_to_label.items() if lb == 0]
    pos.sort()
    neg.sort()
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(ids: List[int]) -> Tuple[List[int], List[int], List[int]]:
        a, b, c = allocate(len(ids), ratio)
        return ids[:a], ids[a : a + b], ids[a + b : a + b + c]

    p_tr, p_va, p_te = split_one(pos)
    n_tr, n_va, n_te = split_one(neg)
    out = {
        "train": sorted(p_tr + n_tr),
        "val": sorted(p_va + n_va),
        "test": sorted(p_te + n_te),
    }
    return out


def write_txt(path: Path, ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(x) for x in ids)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ratio = ratio_tuple(args.split_ratio)

    data_root, checked_roots, searched_roots = detect_data_root(repo_root)
    paths = discover_metadata_paths(repo_root)
    disrupt_raw = load_int_list(paths["disruption"])
    non_raw = load_int_list(paths["nondisruption"])
    adv_map = load_advanced_map(paths["advanced"])

    dup_disrupt = duplicates(disrupt_raw)
    dup_non = duplicates(non_raw)
    set_disrupt = set(disrupt_raw)
    set_non = set(non_raw)
    conflicts = sorted(set_disrupt & set_non)
    set_disrupt -= set(conflicts)
    set_non -= set(conflicts)

    shot_to_label: Dict[int, int] = {sid: 1 for sid in set_disrupt}
    shot_to_label.update({sid: 0 for sid in set_non})

    idx = hdf5_index(data_root)
    required = infer_required_features(idx)

    clean_rows: List[Dict[str, Any]] = []
    excl_rows: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    fls_counts: Counter[str] = Counter()
    time_counts: Counter[str] = Counter()

    for sid in sorted(shot_to_label):
        expected = shot_to_label[sid]
        h5_path = idx.get(sid)
        if h5_path is None:
            reason = "missing_hdf5"
            reason_counts[reason] += 1
            excl_rows.append({"shot_id": sid, "expected_label": expected, "reason": reason, "detail": ""})
            continue

        try:
            with h5py.File(h5_path, "r") as h5:
                meta_label = get_meta_label(h5)
                if meta_label is None:
                    reason = "missing_meta_label"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": "",
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue
                if meta_label != expected:
                    reason = "label_meta_mismatch"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": f"meta={meta_label}",
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue

                group = get_data_group(h5)
                if group is None:
                    reason = "missing_data_group"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": "",
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue

                missing = [f for f in required if f not in group]
                if missing:
                    reason = "missing_required_feature"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": ",".join(missing[:6]),
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue

                lengths = [int(group[f].shape[0]) for f in required]
                n_min = min(lengths)
                n_max = max(lengths)
                if n_min <= 0:
                    reason = "empty_feature_series"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": f"n_min={n_min}",
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue
                if n_max - n_min > args.reconcile_len_tol:
                    reason = "irreconcilable_length_mismatch"
                    reason_counts[reason] += 1
                    excl_rows.append(
                        {
                            "shot_id": sid,
                            "expected_label": expected,
                            "reason": reason,
                            "detail": f"min={n_min},max={n_max}",
                            "hdf5_path": str(h5_path),
                        }
                    )
                    continue

                t_ms, dt_ms, t_src = infer_time_axis_ms(h5, n_min, args.fallback_dt_ms)
                adv = adv_map.get(sid) if expected == 1 else None
                y, keep, fls_ms, fls_src, pos_start_ms, gray_start_ms = make_labels(
                    t_ms=t_ms,
                    shot_label=expected,
                    advanced_ms=adv,
                    gray_ms=args.gray_ms,
                    fallback_fls_ms=args.fallback_fls_ms,
                )

                row = {
                    "shot_id": sid,
                    "expected_label": expected,
                    "meta_label": meta_label,
                    "hdf5_path": str(h5_path),
                    "required_feature_count": len(required),
                    "length_reconciled": int(n_max != n_min),
                    "n_raw": int(n_min),
                    "n_used": int(np.sum(keep)),
                    "n_positive": int(np.sum((y == 1) & keep)),
                    "n_negative": int(np.sum((y == 0) & keep)),
                    "n_gray_dropped": int(np.sum(~keep)),
                    "dt_ms": float(dt_ms),
                    "time_end_ms": float(t_ms[-1]),
                    "time_source": t_src,
                    "has_advanced_time": int(adv is not None),
                    "advanced_time_ms": float(adv) if adv is not None else "",
                    "fls_used_ms": float(fls_ms),
                    "fls_source": fls_src,
                    "positive_start_ms": float(pos_start_ms) if pos_start_ms is not None else "",
                    "gray_start_ms": float(gray_start_ms) if gray_start_ms is not None else "",
                }
                clean_rows.append(row)
                fls_counts[fls_src] += 1
                time_counts[t_src] += 1
        except Exception as e:
            reason = "read_error"
            reason_counts[reason] += 1
            excl_rows.append(
                {
                    "shot_id": sid,
                    "expected_label": expected,
                    "reason": reason,
                    "detail": str(e),
                    "hdf5_path": str(h5_path),
                }
            )

    clean_rows.sort(key=lambda r: int(r["shot_id"]))
    row_by_shot = {int(r["shot_id"]): r for r in clean_rows}
    clean_labels = {sid: int(r["expected_label"]) for sid, r in row_by_shot.items()}
    splits = stratified_split(clean_labels, ratio, args.seed)

    split_tag: Dict[int, str] = {}
    for k in ("train", "val", "test"):
        for sid in splits[k]:
            split_tag[sid] = k
    for row in clean_rows:
        row["split"] = split_tag[int(row["shot_id"])]

    artifact_root = (repo_root / f"artifacts/datasets/{args.dataset_name}").resolve()
    split_dir = (repo_root / "splits").resolve()
    plot_dir = (repo_root / "reports/plots/data_audit").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    write_txt(split_dir / "train.txt", splits["train"])
    write_txt(split_dir / "val.txt", splits["val"])
    write_txt(split_dir / "test.txt", splits["test"])

    write_csv(
        artifact_root / "clean_shots.csv",
        clean_rows,
        [
            "shot_id",
            "split",
            "expected_label",
            "meta_label",
            "hdf5_path",
            "required_feature_count",
            "length_reconciled",
            "n_raw",
            "n_used",
            "n_positive",
            "n_negative",
            "n_gray_dropped",
            "dt_ms",
            "time_end_ms",
            "time_source",
            "has_advanced_time",
            "advanced_time_ms",
            "fls_used_ms",
            "fls_source",
            "positive_start_ms",
            "gray_start_ms",
        ],
    )
    write_csv(
        artifact_root / "excluded_shots.csv",
        excl_rows,
        ["shot_id", "expected_label", "reason", "detail", "hdf5_path"],
    )

    # Imbalance handling: implemented class weighting (train split) + documented alternatives.
    train_rows = [row_by_shot[sid] for sid in splits["train"] if sid in row_by_shot]
    n_pos_shot = sum(1 for r in train_rows if int(r["expected_label"]) == 1)
    n_neg_shot = sum(1 for r in train_rows if int(r["expected_label"]) == 0)
    n_total = n_pos_shot + n_neg_shot
    pos_points = sum(int(r["n_positive"]) for r in train_rows)
    neg_points = sum(int(r["n_negative"]) for r in train_rows)
    class_weights = {
        "implemented_method": "class_weighting",
        "shot_level_weights": {
            "disruptive": (n_total / (2.0 * n_pos_shot)) if n_pos_shot else None,
            "nondisruptive": (n_total / (2.0 * n_neg_shot)) if n_neg_shot else None,
        },
        "point_level": {
            "positive_points": int(pos_points),
            "negative_points": int(neg_points),
            "bce_pos_weight": (neg_points / float(pos_points)) if pos_points else None,
        },
        "considered_alternatives": [
            "weighted_random_sampler",
            "focal_loss_or_hard_negative_mining",
        ],
    }
    (artifact_root / "class_weights.json").write_text(
        json.dumps(class_weights, indent=2), encoding="utf-8"
    )
    (artifact_root / "required_features.json").write_text(
        json.dumps(required, indent=2), encoding="utf-8"
    )

    # 3-shot y(t) examples: disruptive+advanced, disruptive+fallback, non-disruptive.
    examples: List[Dict[str, Any]] = []
    roles: Dict[str, Optional[int]] = {
        "disrupt_with_advanced": None,
        "disrupt_without_advanced": None,
        "non_disruptive": None,
    }
    for r in clean_rows:
        if roles["disrupt_with_advanced"] is None and int(r["expected_label"]) == 1 and int(r["has_advanced_time"]) == 1:
            roles["disrupt_with_advanced"] = int(r["shot_id"])
        if roles["disrupt_without_advanced"] is None and int(r["expected_label"]) == 1 and int(r["has_advanced_time"]) == 0:
            roles["disrupt_without_advanced"] = int(r["shot_id"])
        if roles["non_disruptive"] is None and int(r["expected_label"]) == 0:
            roles["non_disruptive"] = int(r["shot_id"])
    sample_rows: List[Dict[str, Any]] = []
    for role, sid in roles.items():
        if sid is None:
            continue
        row = row_by_shot[sid]
        with h5py.File(Path(str(row["hdf5_path"])), "r") as h5:
            t_ms, _, _ = infer_time_axis_ms(h5, int(row["n_raw"]), args.fallback_dt_ms)
            adv = adv_map.get(sid) if int(row["expected_label"]) == 1 else None
            y, keep, *_ = make_labels(
                t_ms=t_ms,
                shot_label=int(row["expected_label"]),
                advanced_ms=adv,
                gray_ms=args.gray_ms,
                fallback_fls_ms=args.fallback_fls_ms,
            )
        t_end = float(t_ms[-1])
        rel = t_ms - t_end
        for i in range(len(t_ms)):
            sample_rows.append(
                {
                    "role": role,
                    "shot_id": sid,
                    "idx": i,
                    "time_to_end_ms": float(rel[i]),
                    "label": int(y[i]),
                    "keep": int(keep[i]),
                }
            )
        plot_path = ""
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(6, 2.4), dpi=120)
            ax.step(rel, y, where="post", color="#0b5394", linewidth=1.5)
            ax.set_title(f"{role}: shot {sid}")
            ax.set_xlabel("time relative to t_end (ms)")
            ax.set_ylabel("y")
            ax.set_ylim(-0.05, 1.1)
            ax.grid(alpha=0.3)
            out = plot_dir / f"shot_{sid}_{role}.png"
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            plot_path = str(out)
        examples.append(
            {
                "role": role,
                "shot_id": sid,
                "label": int(row["expected_label"]),
                "has_advanced_time": int(row["has_advanced_time"]),
                "fls_source": row["fls_source"],
                "plot_path": plot_path,
            }
        )
    write_csv(
        artifact_root / "label_examples.csv",
        sample_rows,
        ["role", "shot_id", "idx", "time_to_end_ms", "label", "keep"],
    )
    write_csv(
        artifact_root / "example_shots.csv",
        examples,
        ["role", "shot_id", "label", "has_advanced_time", "fls_source", "plot_path"],
    )

    def split_stats(ids: Sequence[int]) -> Dict[str, int]:
        rows = [row_by_shot[sid] for sid in ids if sid in row_by_shot]
        return {
            "shots": len(rows),
            "disruptive_shots": sum(1 for r in rows if int(r["expected_label"]) == 1),
            "nondisruptive_shots": sum(1 for r in rows if int(r["expected_label"]) == 0),
            "raw_points": int(sum(int(r["n_raw"]) for r in rows)),
            "used_points": int(sum(int(r["n_used"]) for r in rows)),
            "positive_points": int(sum(int(r["n_positive"]) for r in rows)),
            "negative_points": int(sum(int(r["n_negative"]) for r in rows)),
            "gray_points": int(sum(int(r["n_gray_dropped"]) for r in rows)),
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovered_paths": {
            "checked_in_order": checked_roots,
            "selected_hdf5_root": str(data_root),
            "repo_search_hits": searched_roots,
            "disruption_shot_list": str(paths["disruption"]),
            "nondisruption_shot_list": str(paths["nondisruption"]),
            "advanced_time_json": str(paths["advanced"]),
        },
        "shot_list_stats": {
            "disruption_raw": len(disrupt_raw),
            "nondisruption_raw": len(non_raw),
            "disruption_duplicates": len(dup_disrupt),
            "nondisruption_duplicates": len(dup_non),
            "cross_conflicts": len(conflicts),
            "usable_after_dedup_conflict": len(shot_to_label),
        },
        "cleaning_stats": {
            "clean_shots": len(clean_rows),
            "excluded_shots": len(excl_rows),
            "excluded_reason_counts": dict(sorted(reason_counts.items())),
            "clean_disruptive": sum(1 for r in clean_rows if int(r["expected_label"]) == 1),
            "clean_nondisruptive": sum(1 for r in clean_rows if int(r["expected_label"]) == 0),
            "disruptive_with_advanced": sum(
                1 for r in clean_rows if int(r["expected_label"]) == 1 and int(r["has_advanced_time"]) == 1
            ),
            "disruptive_without_advanced": sum(
                1 for r in clean_rows if int(r["expected_label"]) == 1 and int(r["has_advanced_time"]) == 0
            ),
            "length_reconciled_count": sum(int(r["length_reconciled"]) for r in clean_rows),
            "fls_source_counts": dict(sorted(fls_counts.items())),
            "time_source_counts": dict(sorted(time_counts.items())),
            "raw_points_total": int(sum(int(r["n_raw"]) for r in clean_rows)),
            "used_points_total": int(sum(int(r["n_used"]) for r in clean_rows)),
            "positive_points_total": int(sum(int(r["n_positive"]) for r in clean_rows)),
            "negative_points_total": int(sum(int(r["n_negative"]) for r in clean_rows)),
            "gray_points_total": int(sum(int(r["n_gray_dropped"]) for r in clean_rows)),
        },
        "split_ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
        "splits": {
            "train": split_stats(splits["train"]),
            "val": split_stats(splits["val"]),
            "test": split_stats(splits["test"]),
        },
        "class_imbalance": class_weights,
        "required_features_count": len(required),
        "examples": examples,
        "artifacts": {
            "clean_shots_csv": str(artifact_root / "clean_shots.csv"),
            "excluded_shots_csv": str(artifact_root / "excluded_shots.csv"),
            "label_examples_csv": str(artifact_root / "label_examples.csv"),
            "example_shots_csv": str(artifact_root / "example_shots.csv"),
            "class_weights_json": str(artifact_root / "class_weights.json"),
            "required_features_json": str(artifact_root / "required_features.json"),
            "summary_json": str(artifact_root / "summary.json"),
            "train_txt": str(split_dir / "train.txt"),
            "val_txt": str(split_dir / "val.txt"),
            "test_txt": str(split_dir / "test.txt"),
            "plot_dir": str(plot_dir),
        },
    }
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

