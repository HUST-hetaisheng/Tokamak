# 变更日志

本日志按实施阶段追踪仓库里程碑（如有提交记录则与之对齐）。

## 2026-02-19 - 阶段 R3 - 阈值策略加固 + 校准拆分隔离

### 更新
- `src/models/eval.py`
  - 新增炮级阈值搜索：`choose_threshold_by_shot_fpr(...)`。
- `src/models/train.py`
  - 新增 `--threshold-objective shot_fpr_constrained`。
  - 新增 `--threshold-max-shot-fpr`（默认 `0.02`）。
  - 新增 `--calibration-shot-fraction`（默认 `0.5`），将验证集炮次拆分为：
    - 校准子集（拟合校准器），
    - 阈值子集（选择阈值 / 报告验证指标）。
  - 在 `training_config.json` 和 `metrics_summary.json` 中持久化拆分元数据。

### 新增
- 新的迭代运行（策略约束型）：
  - `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/*`
  - `artifacts/models/iters/sfpr001_d4_e260_lr004_s3_reason/*`
- 上述运行的单炮原因文件：
  - `disruption_reason_per_shot.csv`（每个破裂 TEST 炮一行，含 top-k 证据）。
- 可读的单炮 Markdown 报告：
  - `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_report.md`
  - `artifacts/models/iters/sfpr001_d4_e260_lr004_s3_reason/disruption_reason_report.md`
- 新的报告渲染器：
  - `src/models/generate_reason_report.py`

### 当前推荐运行
- `sfpr002_d4_e260_lr004_s3_reason`
- 关键指标 (TEST)：`accuracy=0.990885`、`roc_auc=0.978437`、`shot_accuracy=0.953757`、`shot_tpr=0.842105`、`shot_fpr=0.014815`
- 原因覆盖率：`38/38` 个破裂测试炮均有原因行。

## 2026-02-19 - 阶段 R2 - J-TEXT 参数扫描 + 单炮破裂原因

### 新增
- 多运行超参数扫描输出位于：
  - `artifacts/models/iters/*`
  - `reports/iters/*`
- 扫描汇总表：
  - `reports/iters/summary.csv`
  - `reports/iters/summary.md`
- 训练管线中的单破裂炮原因导出：
  - `disruption_reason_per_shot.csv`（每个破裂 TEST 炮一行）。

### 更新
- `src/models/train.py`
  - 新增 `--reason-top-k`（默认 `3`），用于单炮原因提取。
  - 新增基于贡献度的机制映射输出，面向破裂炮。
  - 新增动态进度产物路径报告（不再硬编码为 `artifacts/models/best/*`）。
- `docs/progress.md`
  - 将 Agent-3 与扫描状态及当前推荐运行同步。

### 当前推荐运行
- `acc_d4_e260_lr004_ss085_cs09_s3_reason`
- 关键指标 (TEST)：`accuracy=0.991008`、`roc_auc=0.978493`、`shot_accuracy=0.953757`、`shot_tpr=0.842105`、`shot_fpr=0.014815`
- 原因覆盖率：`38/38` 个破裂测试炮均有原因行。

## 2026-02-19 - 阶段 R1 - 重启文档同步 (Agent-5)

### 更新
- 在 `README.md` 中添加了产物指南（路径 + 角色 + 查阅时机），涵盖重启关键文件。
- 在 `README.md` 中明确说明默认仅导出少量时间线 PNG，并指向完整的 173 炮 CSV 产物。
- 在 `README.md` 中更新了使用当前训练参数的全炮次重跑命令：
  - `--max-train-shots 0`
  - `--max-val-shots 0`
  - `--max-test-shots 0`
  - `--plot-all-test-shots`
  - `--threshold-objective accuracy`
- 在 `docs/changelog.md` 中同步了重启阶段备注。

### 产物指针
- `README.md`
- `reports/plots/probability_timelines_test.csv`
- `artifacts/models/best/warning_summary_test.csv`
- `docs/progress.md`

## 2026-02-19 - 阶段 M0 - EAST 数据集目录管线

### 新增
- 基于 `data/EAST/build_east_shot_catalog.py` 的可复现炮号目录构建命令。
- 目录产物位于 `data/EAST/exports/`。

### 产物
- `data/EAST/exports/east_shot_catalog_all.csv`
- `data/EAST/exports/east_usable_shot_level.csv`
- `data/EAST/exports/east_usable_shots_only.csv`
- `data/EAST/exports/east_shot_catalog_summary.json`

## 2026-02-19 - 阶段 M1 - 序列训练/预测入口验证

### 新增
- 通过 `analysis/__pycache__/train_east_realtime_sequence.cpython-314.pyc` 的可执行训练/预测工作流。
- 训练命令输出 TEST 评估和单炮概率轨迹。

### 产物
- `analysis/outputs/<run_name>/best_model.pt`
- `analysis/outputs/<run_name>/training_history.csv`
- `analysis/outputs/<run_name>/metrics_summary.json`
- `analysis/outputs/<run_name>/sequence_predictions/val/*.csv`
- `analysis/outputs/<run_name>/sequence_predictions/test/*.csv`

## 2026-02-19 - 阶段 M2 - 文档 MVP (Agent-5)

### 新增
- 含可执行命令和当前占位符的 `README.md` 运行手册。
- `docs/architecture.md` 将 PGFE/(S-)CORAL/DART/SHAP 管线映射到工程接口。
- `docs/changelog.md` 基于阶段的里程碑追踪。

### 记录的决策
- 使用 FLS 兼容的灰区标注以减少边界标签噪声。
- 使用 DART + 校准 + SHAP 作为可解释概率管线目标。
- 保持 EAST 优先但面向迁移的接口，以便后续跨装置适配。
