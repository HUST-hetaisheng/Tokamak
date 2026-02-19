# Fuison: Tokamak Disruption Prediction (MVP Runbook)

This repository targets real-time tokamak disruption prediction with paper_131-aligned decisions:
- physics-guided feature pipeline (PGFE family),
- gray-zone-aware labeling (FLS-compatible),
- calibrated disruption probability output,
- transfer-ready interfaces for EAST and later cross-device adaptation.

## Data Path Configuration

Default data root for this repo:

```powershell
$env:FUISON_DATA_ROOT = "G:\我的云端硬盘\Fuison\data"
if (-not (Test-Path $env:FUISON_DATA_ROOT)) { throw "FUISON_DATA_ROOT not found: $env:FUISON_DATA_ROOT" }
```

Current scripts expect shot lists in this repo and HDF5 under `data/EAST/unified_hdf5`.
If you keep HDF5 only in cloud root, link once:

```powershell
if (-not (Test-Path "data/EAST/unified_hdf5")) {
  cmd /c mklink /J data\EAST\unified_hdf5 "$env:FUISON_DATA_ROOT\EAST\unified_hdf5"
}
```

## Environment

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas h5py scikit-learn torch matplotlib joblib
```

## Entry Points (Current Workspace)

Source `.py` entrypoints are not present yet for sequence training; use current compiled entrypoint:

```powershell
if (Test-Path "analysis/train_east_realtime_sequence.py") {
  $TRAIN_ENTRY = "analysis/train_east_realtime_sequence.py"
} else {
  $TRAIN_ENTRY = "analysis/__pycache__/train_east_realtime_sequence.cpython-314.pyc"
}
```

## Reproducible Runbook

### 1) Build dataset catalog

```powershell
py -3.14 data/EAST/build_east_shot_catalog.py `
  --repo-root . `
  --workers 8 `
  --output-dir data/EAST/exports
```

### 2) Train + evaluate (TEST eval is part of `train`)

```powershell
$RUN_DIR = "analysis/outputs/realtime_hazard_gru_mvp"
py -3.14 $TRAIN_ENTRY train `
  --repo-root . `
  --hdf5-root "$env:FUISON_DATA_ROOT\EAST\unified_hdf5" `
  --output-dir $RUN_DIR `
  --model-type hazard_gru `
  --epochs 30 `
  --batch-size 256 `
  --eval-batch-size 512 `
  --num-workers 0
```

### 2b) Full-shot rerun (J-TEXT DART pipeline, no shot cap)

Use the current modeling entrypoint and disable caps with `--max-*-shots 0`.  
This is the command used in the relaunch stage:

```powershell
python -m src.models.train `
  --repo-root . `
  --max-train-shots 0 `
  --max-val-shots 0 `
  --max-test-shots 0 `
  --plot-all-test-shots `
  --threshold-objective accuracy
```

### 2c) Recommended policy run (shot-level FAR constrained + per-shot reasons)

This run separates validation shots into calibration/threshold subsets, then selects threshold under shot-level FAR constraint.

```powershell
python -m src.models.train `
  --repo-root . `
  --max-train-shots 0 `
  --max-val-shots 0 `
  --max-test-shots 0 `
  --plot-shot-limit 0 `
  --threshold-objective shot_fpr_constrained `
  --threshold-max-shot-fpr 0.02 `
  --calibration-shot-fraction 0.5 `
  --sustain-ms 3 `
  --xgb-estimators 260 `
  --xgb-learning-rate 0.04 `
  --xgb-max-depth 4 `
  --xgb-subsample 0.85 `
  --xgb-colsample-bytree 0.9 `
  --reason-top-k 3 `
  --output-dir artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason `
  --report-dir reports/iters/sfpr002_d4_e260_lr004_s3_reason
```

### 2d) Generate per-shot readable reason report (Markdown)

```powershell
python -m src.models.generate_reason_report `
  --reason-csv artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_per_shot.csv `
  --title "Per-Shot Disruption Reason Report (sfpr002)"
```

### 3) Streaming prediction on custom matrix

```powershell
py -3.14 $TRAIN_ENTRY predict `
  --checkpoint "$RUN_DIR/best_model.pt" `
  --matrix analysis/outputs/realtime_gru_run7/demo_matrix_2000x11.csv `
  --output-csv analysis/outputs/realtime_predict.csv
```

### 4) Calibration command (MVP placeholder linked to current artifacts)

This calibrates val-set probabilities from `sequence_predictions/val/*.csv` and saves an isotonic model.

```powershell
@'
import glob, os, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import joblib

run_dir = r"analysis/outputs/realtime_hazard_gru_mvp"
val_dir = os.path.join(run_dir, "sequence_predictions", "val")
frames = []
for p in glob.glob(os.path.join(val_dir, "*.csv")):
    if p.endswith("_summary.csv"):
        continue
    df = pd.read_csv(p)
    frames.append(df[df["valid_label"] == 1][["prob_disrupt", "label"]])

data = pd.concat(frames, ignore_index=True)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(data["prob_disrupt"], data["label"])
cal = iso.predict(data["prob_disrupt"])

out_dir = os.path.join(run_dir, "calibration")
os.makedirs(out_dir, exist_ok=True)
joblib.dump(iso, os.path.join(out_dir, "isotonic.joblib"))
pd.DataFrame({"prob_raw": data["prob_disrupt"], "prob_cal": cal, "label": data["label"]}).to_csv(
    os.path.join(out_dir, "val_calibrated_points.csv"), index=False
)
print("brier_raw=", brier_score_loss(data["label"], data["prob_disrupt"]))
print("brier_cal=", brier_score_loss(data["label"], cal))
print("saved:", out_dir)
'@ | py -3.14 -
```

### 5) Probability-curve plotting command (per-shot)

```powershell
@'
import pandas as pd
import matplotlib.pyplot as plt

shot_csv = r"analysis/outputs/realtime_hazard_gru_mvp/sequence_predictions/test/64448.csv"
df = pd.read_csv(shot_csv)

plt.figure(figsize=(10, 4))
plt.plot(df["time_s"], df["prob_disrupt"], label="prob_disrupt")
plt.plot(df["time_s"], df["prob_disrupt_clamped"], label="prob_disrupt_clamped", alpha=0.8)
plt.ylim(0, 1.02)
plt.xlabel("time_s")
plt.ylabel("probability")
plt.title(f"Disruption probability curve: shot {int(df['shot_id'].iloc[0])}")
plt.legend()
plt.tight_layout()
out_png = r"analysis/outputs/realtime_hazard_gru_mvp/sequence_predictions/test/64448_prob_curve.png"
plt.savefig(out_png, dpi=160)
print("saved:", out_png)
'@ | py -3.14 -
```

## PNG Count And 173-Shot Outputs

By default, only a few shot PNGs are exported because the trainer uses preview mode (`--plot-shot-limit 3`) unless `--plot-all-test-shots` is enabled.

In the latest relaunch full-shot run, `--plot-all-test-shots` was enabled and `173` shot PNGs were generated in:
- `reports/plots/probability/`

The full 173-shot artifacts are in CSV form:
- Full per-timepoint timelines: `reports/plots/probability_timelines_test.csv` (173 unique `shot_id`).
- Full per-shot warning summary: `artifacts/models/best/warning_summary_test.csv` (173 rows).

## Artifact Guide

| Path | Role | When to read |
|---|---|---|
| `ref/paper_131.txt` | Core prior-knowledge reference for mechanism, feature, and labeling choices. | Before changing feature set, label strategy, or evaluation policy. |
| `data/EAST/exports/east_shot_catalog_summary.json` | EAST catalog availability and usability summary. | Before launching training to verify data coverage. |
| `analysis/outputs/<run_name>/metrics_summary.json` | EAST run-level metrics and headline performance. | Immediately after each `train` run. |
| `analysis/outputs/<run_name>/sequence_predictions/test/_summary.csv` | EAST test shot-level summary. | When checking warning hit/miss behavior per shot. |
| `analysis/outputs/<run_name>/sequence_predictions/test/*.csv` | EAST per-shot probability timelines. | When plotting or diagnosing an individual shot timeline. |
| `reports/plots/probability/shot_*_timeline.png` | Small visual sanity-check set of timeline plots (sampled shots). | Quick visual QA only; not full-shot coverage. |
| `reports/plots/probability_timelines_test.csv` | Full timepoint-level timeline table for all 173 TEST shots (relaunch artifacts). | Full-batch analysis, aggregation, and custom plotting. |
| `artifacts/models/best/warning_summary_test.csv` | Full shot-level warning decisions for the same 173 TEST shots. | Shot-level policy analysis and confusion/lead-time checks. |
| `artifacts/models/iters/*/disruption_reason_per_shot.csv` | Per-disruptive-shot mechanism reason table (primary mechanism + top-k feature evidence). | Explain each disruptive shot and trace operator-facing root-cause hypotheses. |
| `artifacts/models/iters/*/disruption_reason_report.md` | Readable per-shot markdown report with mechanism distribution and shot-by-shot evidence. | Human review, experiment logs, and handoff to control/operations teams. |
| `reports/iters/summary.md` | Hyperparameter sweep comparison table and current recommended run. | Parameter iteration tracking and next-run selection. |
| `artifacts/models/best/metrics_summary.json` | Relaunch-stage bounded MVP summary metrics. | Baseline comparison and report updates. |
| `output/` | Reserved location for generated artifacts requested by users. | Use for ad-hoc generated exports and reports. |
