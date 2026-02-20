from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py


@dataclass
class MetaRecord:
    meta_missing: bool
    read_error: str | None
    is_disrupt: int | None
    start_time: float | None
    down_time: float | None
    length: int | None
    time_interval: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EAST shot-level catalog and usable shot list with data quality flags."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains shot_list/ and data/EAST/unified_hdf5.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers used for HDF5 meta reads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "exports",
        help="Directory for generated CSV/JSON outputs.",
    )
    return parser.parse_args()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _read_scalar(group: h5py.Group, key: str) -> Any:
    if key not in group:
        return None
    value = group[key][()]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def load_json_list(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[int] = []
    for value in raw:
        iv = _safe_int(value)
        if iv is not None:
            out.append(iv)
    return out


def load_advanced_time(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, float] = {}
    for shot, value in raw.items():
        s = _safe_int(shot)
        v = _safe_float(value)
        if s is not None and v is not None:
            out[s] = v
    return out


def read_meta_for_one(shot: int, path: Path) -> tuple[int, MetaRecord]:
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                return (
                    shot,
                    MetaRecord(
                        meta_missing=True,
                        read_error=None,
                        is_disrupt=None,
                        start_time=None,
                        down_time=None,
                        length=None,
                        time_interval=None,
                    ),
                )
            meta = f["meta"]
            return (
                shot,
                MetaRecord(
                    meta_missing=False,
                    read_error=None,
                    is_disrupt=_safe_int(_read_scalar(meta, "IsDisrupt")),
                    start_time=_safe_float(_read_scalar(meta, "StartTime")),
                    down_time=_safe_float(_read_scalar(meta, "DownTime")),
                    length=_safe_int(_read_scalar(meta, "length")),
                    time_interval=_safe_float(_read_scalar(meta, "time_interval")),
                ),
            )
    except Exception as exc:
        return (
            shot,
            MetaRecord(
                meta_missing=False,
                read_error=str(exc),
                is_disrupt=None,
                start_time=None,
                down_time=None,
                length=None,
                time_interval=None,
            ),
        )


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    shot_list_dir = repo_root / "shot_list" / "EAST"
    hdf5_root = repo_root / "data" / "EAST" / "unified_hdf5"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    disrupt_list = load_json_list(shot_list_dir / "Disruption_EAST_TV.json")
    nondisrupt_list = load_json_list(shot_list_dir / "Non-disruption_EAST_TV.json")
    advanced_time = load_advanced_time(shot_list_dir / "AdvancedTime_EAST.json")

    disrupt_counter = Counter(disrupt_list)
    nondisrupt_counter = Counter(nondisrupt_list)
    dup_in_disrupt = {shot for shot, count in disrupt_counter.items() if count > 1}
    dup_in_nondisrupt = {shot for shot, count in nondisrupt_counter.items() if count > 1}

    disrupt_set = set(disrupt_list)
    nondisrupt_set = set(nondisrupt_list)
    conflict_labels = disrupt_set & nondisrupt_set

    hdf5_map = {int(path.stem): path for path in hdf5_root.glob("*/*.hdf5")}
    list_union = disrupt_set | nondisrupt_set
    listed_hdf5_map = {shot: hdf5_map[shot] for shot in list_union if shot in hdf5_map}

    meta_records: dict[int, MetaRecord] = {}
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(read_meta_for_one, shot, path): shot
            for shot, path in listed_hdf5_map.items()
        }
        for future in as_completed(futures):
            shot, record = future.result()
            meta_records[shot] = record

    all_shots = sorted(set(hdf5_map.keys()) | list_union)
    reason_counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    usable_rows: list[dict[str, Any]] = []

    for shot in all_shots:
        in_disrupt = shot in disrupt_set
        in_nondisrupt = shot in nondisrupt_set
        in_hdf5 = shot in hdf5_map
        list_label = 1 if in_disrupt and not in_nondisrupt else 0 if in_nondisrupt and not in_disrupt else None
        adv_time = advanced_time.get(shot)
        meta = meta_records.get(shot)

        reasons: list[str] = []
        if shot in conflict_labels:
            reasons.append("conflict_in_disrupt_and_nondisrupt_lists")
        if shot in dup_in_disrupt:
            reasons.append("duplicate_within_disrupt_list")
        if shot in dup_in_nondisrupt:
            reasons.append("duplicate_within_nondisrupt_list")
        if not in_hdf5:
            reasons.append("missing_hdf5")
        if not in_disrupt and not in_nondisrupt:
            reasons.append("not_in_tv_shotlists")

        if list_label is not None:
            if meta is None and in_hdf5:
                reasons.append("meta_not_scanned")
            if meta is not None:
                if meta.meta_missing:
                    reasons.append("missing_meta_group")
                if meta.read_error:
                    reasons.append("hdf5_read_error")
                if meta.is_disrupt is None:
                    reasons.append("missing_meta_isdisrupt")
                elif meta.is_disrupt != list_label:
                    reasons.append("meta_isdisrupt_mismatch")
            if list_label == 1:
                if adv_time is None:
                    reasons.append("missing_advanced_time_for_disruptive")
                elif adv_time <= 0:
                    reasons.append("nonpositive_advanced_time_for_disruptive")

        usable = (
            list_label in (0, 1)
            and shot not in conflict_labels
            and shot not in dup_in_disrupt
            and shot not in dup_in_nondisrupt
            and in_hdf5
            and meta is not None
            and not meta.meta_missing
            and not meta.read_error
            and meta.is_disrupt is not None
            and meta.is_disrupt == list_label
            and (list_label == 0 or (adv_time is not None and adv_time > 0))
        )

        if not usable:
            for reason in reasons:
                reason_counter[reason] += 1

        row = {
            "shot": shot,
            "in_disrupt_list": int(in_disrupt),
            "in_nondisrupt_list": int(in_nondisrupt),
            "list_label": "" if list_label is None else list_label,
            "advanced_time": "" if adv_time is None else adv_time,
            "has_hdf5": int(in_hdf5),
            "meta_is_disrupt": "" if meta is None or meta.is_disrupt is None else meta.is_disrupt,
            "meta_start_time": "" if meta is None or meta.start_time is None else meta.start_time,
            "meta_down_time": "" if meta is None or meta.down_time is None else meta.down_time,
            "meta_length": "" if meta is None or meta.length is None else meta.length,
            "meta_time_interval": "" if meta is None or meta.time_interval is None else meta.time_interval,
            "hdf5_path": "" if not in_hdf5 else str(hdf5_map[shot].as_posix()),
            "usable": int(usable),
            "exclude_reasons": ";".join(reasons),
        }
        rows.append(row)
        if usable:
            usable_rows.append(row)

    all_csv = output_dir / "east_shot_catalog_all.csv"
    with all_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    usable_csv = output_dir / "east_usable_shot_level.csv"
    with usable_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "shot",
            "list_label",
            "advanced_time",
            "hdf5_path",
            "meta_is_disrupt",
            "meta_start_time",
            "meta_down_time",
            "meta_length",
            "meta_time_interval",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in usable_rows:
            writer.writerow({field: row[field] for field in fields})

    usable_shot_csv = output_dir / "east_usable_shots_only.csv"
    with usable_shot_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shot", "label", "advanced_time"])
        writer.writeheader()
        for row in usable_rows:
            writer.writerow(
                {
                    "shot": row["shot"],
                    "label": row["list_label"],
                    "advanced_time": row["advanced_time"],
                }
            )

    summary = {
        "input_counts": {
            "disrupt_list_raw": len(disrupt_list),
            "nondisrupt_list_raw": len(nondisrupt_list),
            "disrupt_list_unique": len(disrupt_set),
            "nondisrupt_list_unique": len(nondisrupt_set),
            "advanced_time_entries": len(advanced_time),
            "hdf5_total_files": len(hdf5_map),
        },
        "conflicts": {
            "shots_in_both_disrupt_and_nondisrupt": len(conflict_labels),
            "duplicate_within_disrupt_list": len(dup_in_disrupt),
            "duplicate_within_nondisrupt_list": len(dup_in_nondisrupt),
        },
        "output_counts": {
            "all_shot_rows": len(rows),
            "usable_shots": len(usable_rows),
        },
        "exclude_reason_counts": dict(reason_counter),
        "outputs": {
            "all_catalog_csv": str(all_csv.as_posix()),
            "usable_shot_level_csv": str(usable_csv.as_posix()),
            "usable_shots_only_csv": str(usable_shot_csv.as_posix()),
        },
    }

    summary_json = output_dir / "east_shot_catalog_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"all_catalog_csv={all_csv.as_posix()}")
    print(f"usable_shot_level_csv={usable_csv.as_posix()}")
    print(f"usable_shots_only_csv={usable_shot_csv.as_posix()}")
    print(f"summary_json={summary_json.as_posix()}")
    print(f"usable_shots={len(usable_rows)}")


if __name__ == "__main__":
    main()
