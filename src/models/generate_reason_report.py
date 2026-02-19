#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-shot disruption reason markdown report from disruption_reason_per_shot.csv"
    )
    parser.add_argument("--reason-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Per-Shot Disruption Reason Report")
    return parser.parse_args()


def _fmt_float(v: Any, digits: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return "N/A"
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def _fmt_bool01(v: Any) -> str:
    try:
        return "Yes" if int(v) == 1 else "No"
    except Exception:
        return "N/A"


def _parse_top_features_json(v: Any) -> List[Dict[str, Any]]:
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
    except Exception:
        pass
    return []


def build_report(
    reason_df: pd.DataFrame,
    title: str,
    metrics: Dict[str, Any] | None = None,
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")

    rows_total = int(len(reason_df))
    if rows_total == 0:
        lines.append("No disruptive-shot reason rows were found.")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append(f"- Reason rows: `{rows_total}`")
    if metrics:
        test = metrics.get("test_timepoint_calibrated", {})
        shot = metrics.get("test_shot_policy", {})
        th = metrics.get("threshold_policy", {})
        lines.append(f"- Test accuracy: `{_fmt_float(test.get('accuracy'), 6)}`")
        lines.append(f"- Test ROC-AUC: `{_fmt_float(test.get('roc_auc'), 6)}`")
        lines.append(f"- Shot accuracy: `{_fmt_float(shot.get('shot_accuracy'), 6)}`")
        lines.append(f"- Shot TPR/FPR: `{_fmt_float(shot.get('shot_tpr'), 6)}` / `{_fmt_float(shot.get('shot_fpr'), 6)}`")
        lines.append(f"- Threshold objective/theta: `{th.get('objective', 'N/A')}` / `{_fmt_float(th.get('theta'), 6)}`")
    lines.append("")

    dist = reason_df["primary_mechanism"].fillna("unmapped").value_counts()
    lines.append("## Mechanism Distribution")
    lines.append("")
    lines.append("| mechanism | count |")
    lines.append("|---|---:|")
    for mech, cnt in dist.items():
        lines.append(f"| {mech} | {int(cnt)} |")
    lines.append("")

    sort_df = reason_df.copy()
    if "shot_id" in sort_df.columns:
        sort_df = sort_df.sort_values("shot_id")

    lines.append("## Shot Explanations")
    lines.append("")
    for _, row in sort_df.iterrows():
        sid = row.get("shot_id", "N/A")
        lines.append(f"### Shot {sid}")
        lines.append("")
        lines.append(f"- Primary mechanism: `{row.get('primary_mechanism', 'N/A')}`")
        lines.append(f"- Primary mechanism score: `{_fmt_float(row.get('primary_mechanism_score'), 6)}`")
        lines.append(f"- Warning triggered: `{_fmt_bool01(row.get('warning'))}`")
        lines.append(f"- Lead time (ms): `{_fmt_float(row.get('lead_time_ms'), 3)}`")
        lines.append(f"- Reason window rule: `{row.get('reason_window_rule', 'N/A')}`")
        lines.append(f"- Reason window points: `{int(row.get('reason_window_points', 0))}`")
        lines.append("")
        lines.append("Evidence:")
        top_json = _parse_top_features_json(row.get("top_features_json"))
        if top_json:
            for item in top_json:
                rank = item.get("rank", "N/A")
                feat = item.get("feature", "N/A")
                contrib = _fmt_float(item.get("contribution"), 6)
                tags = item.get("mechanism_tags", "unmapped")
                lines.append(f"{rank}. `{feat}` (contribution `{contrib}`), tags: `{tags}`")
        else:
            for rank in [1, 2, 3]:
                feat = row.get(f"top{rank}_feature")
                if pd.isna(feat) or str(feat).strip() == "":
                    continue
                contrib = _fmt_float(row.get(f"top{rank}_contribution"), 6)
                tags = row.get(f"top{rank}_mechanism_tags", "unmapped")
                lines.append(f"{rank}. `{feat}` (contribution `{contrib}`), tags: `{tags}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    reason_csv = args.reason_csv.resolve()
    if not reason_csv.exists():
        raise FileNotFoundError(f"Reason CSV not found: {reason_csv}")

    if args.output_md is None:
        output_md = (reason_csv.parent / "disruption_reason_report.md").resolve()
    else:
        output_md = args.output_md.resolve()

    metrics: Dict[str, Any] | None = None
    metrics_json = args.metrics_json.resolve() if args.metrics_json else (reason_csv.parent / "metrics_summary.json").resolve()
    if metrics_json.exists():
        try:
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    df = pd.read_csv(reason_csv)
    report = build_report(df, title=args.title, metrics=metrics)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(str(output_md))


if __name__ == "__main__":
    main()
