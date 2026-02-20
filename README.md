# Fuison: 托卡马克破裂预测 (MVP 运行手册)

本仓库面向实时托卡马克等离子体破裂预测，核心设计对齐 paper_131：
- 物理引导特征管线 (PGFE（物理引导特征工程） 家族),
- 灰区感知标注 (FLS（灵活标注策略） 兼容),
- 校准后的破裂概率输出,
- 面向 EAST 的迁移学习接口，后续可扩展至跨装置适配。

## 数据路径配置

本仓库的默认数据根目录：

```powershell
$env:FUISON_DATA_ROOT = "G:\我的云端硬盘\Fuison\data"
if (-not (Test-Path $env:FUISON_DATA_ROOT)) { throw "FUISON_DATA_ROOT not found: $env:FUISON_DATA_ROOT" }
```

当前脚本要求炮号列表位于本仓库内，HDF5 文件位于 `data/EAST/unified_hdf5`。
如果 HDF5 仅存放在云端根目录，请创建一次链接：

```powershell
if (-not (Test-Path "data/EAST/unified_hdf5")) {
  cmd /c mklink /J data\EAST\unified_hdf5 "$env:FUISON_DATA_ROOT\EAST\unified_hdf5"
}
```

## 环境搭建

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas h5py scikit-learn torch matplotlib joblib
```

## 入口点 (当前工作区)

序列训练的 `.py` 入口尚未就绪；请使用当前已编译的入口：

```powershell
if (Test-Path "analysis/train_east_realtime_sequence.py") {
  $TRAIN_ENTRY = "analysis/train_east_realtime_sequence.py"
} else {
  $TRAIN_ENTRY = "analysis/__pycache__/train_east_realtime_sequence.cpython-314.pyc"
}
```

## 可复现运行手册

### 1) 构建数据集目录

```powershell
py -3.14 data/EAST/build_east_shot_catalog.py `
  --repo-root . `
  --workers 8 `
  --output-dir data/EAST/exports
```

### 2) 训练 + 评估 (TEST 评估已包含在 `train` 中)

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

### 2b) 全炮次重跑 (J-TEXT DART（丢弃正则化树）管线，不设炮数上限)

使用当前建模入口并以 `--max-*-shots 0` 取消上限。
以下为重启阶段使用的命令：

```powershell
python -m src.models.train `
  --repo-root . `
  --max-train-shots 0 `
  --max-val-shots 0 `
  --max-test-shots 0 `
  --plot-all-test-shots `
  --threshold-objective accuracy
```

### 2c) 推荐策略运行 (炮级误报率约束 + 单炮原因分析)

本运行将验证集炮次拆分为校准/阈值子集，然后在炮级误报率约束下选取阈值。

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

### 2d) 生成单炮可读原因报告 (Markdown)

```powershell
python -m src.models.generate_reason_report `
  --reason-csv artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_per_shot.csv `
  --title "单炮破裂原因报告 (sfpr002)"
```

### 3) 对自定义矩阵进行流式预测

```powershell
py -3.14 $TRAIN_ENTRY predict `
  --checkpoint "$RUN_DIR/best_model.pt" `
  --matrix analysis/outputs/realtime_gru_run7/demo_matrix_2000x11.csv `
  --output-csv analysis/outputs/realtime_predict.csv
```

### 4) 校准命令 (MVP 占位符，关联当前产物)

对 `sequence_predictions/val/*.csv` 中的验证集概率进行校准，并保存等保回归模型。

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

### 5) 概率曲线绘图命令 (单炮)

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

## PNG 数量与 173 炮输出说明

默认情况下仅导出少量炮号的 PNG，因为训练器使用预览模式 (`--plot-shot-limit 3`)，除非启用 `--plot-all-test-shots`。

在最新的全炮次重跑中，启用了 `--plot-all-test-shots`，共生成 `173` 张炮号 PNG 于：
- `reports/plots/probability/`

完整的 173 炮产物以 CSV 形式保存：
- 完整逐时间点时间线：`reports/plots/probability_timelines_test.csv`（173 个唯一 `shot_id`）。
- 完整炮级预警汇总：`artifacts/models/best/warning_summary_test.csv`（173 行）。

## 产物指南

| 路径 | 角色 | 何时查阅 |
|---|---|---|
| `ref/paper_131.txt` | 核心先验知识参考，用于机制、特征和标注策略决策。 | 在修改特征集、标注策略或评估策略之前。 |
| `data/EAST/exports/east_shot_catalog_summary.json` | EAST 数据目录可用性及可用性摘要。 | 在启动训练前，验证数据覆盖情况。 |
| `analysis/outputs/<run_name>/metrics_summary.json` | EAST 运行级指标与核心性能。 | 每次 `train` 运行后立即查看。 |
| `analysis/outputs/<run_name>/sequence_predictions/test/_summary.csv` | EAST 测试集炮级汇总。 | 检查单炮级别的预警命中/未命中行为时。 |
| `analysis/outputs/<run_name>/sequence_predictions/test/*.csv` | EAST 单炮概率时间线。 | 绘图或诊断单炮时间线时。 |
| `reports/plots/probability/shot_*_timeline.png` | 时间线图的小规模可视化样本（采样炮号）。 | 仅用于快速视觉 QA，非全炮次覆盖。 |
| `reports/plots/probability_timelines_test.csv` | 所有 173 个 TEST 炮号的完整逐时间点时间线表（重启产物）。 | 全批量分析、聚合和自定义绘图。 |
| `artifacts/models/best/warning_summary_test.csv` | 同一 173 个 TEST 炮号的完整炮级预警决策。 | 炮级策略分析及混淆/提前量检查。 |
| `artifacts/models/iters/*/disruption_reason_per_shot.csv` | 每个破裂炮的机制原因表（主要机制 + top-k 特征证据）。 | 解释每个破裂炮并追溯面向运行人员的根因假设。 |
| `artifacts/models/iters/*/disruption_reason_report.md` | 可读的单炮 Markdown 报告，含机制分布和逐炮证据。 | 人工审阅、实验记录及向控制/运行团队交接。 |
| `reports/iters/summary.md` | 超参数扫描对比表及当前推荐运行。 | 参数迭代跟踪和下一次运行选择。 |
| `artifacts/models/best/metrics_summary.json` | 重启阶段受限 MVP 汇总指标。 | 基线对比与报告更新。 |
| `output/` | 用户请求的生成产物的保留位置。 | 用于临时生成的导出和报告。 |
