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

## Artifact Locations

- Dataset catalog:
  - `data/EAST/exports/east_shot_catalog_all.csv`
  - `data/EAST/exports/east_usable_shot_level.csv`
  - `data/EAST/exports/east_usable_shots_only.csv`
  - `data/EAST/exports/east_shot_catalog_summary.json`
- Training/eval run:
  - `analysis/outputs/<run_name>/best_model.pt`
  - `analysis/outputs/<run_name>/training_history.csv`
  - `analysis/outputs/<run_name>/metrics_summary.json`
  - `analysis/outputs/<run_name>/sequence_predictions/val/*.csv`
  - `analysis/outputs/<run_name>/sequence_predictions/test/*.csv`
  - `analysis/outputs/<run_name>/sequence_predictions/*/_summary.csv`
- Calibration placeholder output:
  - `analysis/outputs/<run_name>/calibration/isotonic.joblib`
  - `analysis/outputs/<run_name>/calibration/val_calibrated_points.csv`

`output/` remains reserved for generated artifacts.
