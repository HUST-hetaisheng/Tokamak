# 架构设计：PGFE（物理引导特征工程） / (S-)CORAL（子空间相关对齐） / DART（丢弃正则化树） / SHAP（可解释性归因） / EAST 迁移

## 1. 工程目标

构建一个实时破裂预测器，输出：
- 随时间变化的破裂概率,
- 可触发的预警信号,
- 面向 EAST 优先、后续可扩展至跨装置的迁移接口。

本映射以 paper_131 为首要先验依据。

## 2. 研究-工程映射

| 研究模块 | 工程映射 | 仓库中的当前挂接点 | 状态 |
|---|---|---|---|
| PGFE / PGFE-U | 特征通道契约 + 时间导数 + 物理先验缩放 | `train_east_realtime_sequence` 支持基础特征键、`--add-diff`、`--add-abs-diff`、`--use-paper131-prior` | 活跃 |
| FLS（灵活标注策略） + 灰区 | 动态/高级标注，排除不确定区间的监督损失 | `--uncertain-ms`；`sequence_predictions/*.csv` 中的逐点 `valid_label` | 活跃（MVP 兼容） |
| CORAL / S-CORAL | 潜在特征上的域对齐损失（源-目标协方差对齐） | 编码器特征之后、分类器头之前的接口挂接点 | 占位（待实现） |
| DART 分类器 | 基于树的可解释基线（`booster='dart'`），用于表格 PGFE 特征 | 当前基线入口为 `train_fusion_baseline`（RandomForest）；DART 适配器待定 | 占位（已规划） |
| 概率校准 | 验证集概率的后处理校准 | README 中的 MVP 命令保存 `calibration/isotonic.joblib` | 活跃（运行手册占位符） |
| SHAP 可解释性 | 全局 + 炮级特征归因，用于校准模型 | 运行目录下预留输出契约（`explainability/`） | 占位（已规划） |

## 3. 数据流 (当前 + 规划)

1. HDF5 炮号数据 (`unified_hdf5`) + 炮号列表 (`Disruption`、`Non-disruption`、`AdvancedTime`)。
2. 特征组装 (PGFE 通道集 + 可选差分 + 先验缩放)。
3. 带灰区排除的标注 (`valid_label=0` 的点排除在损失/指标之外)。
4. 序列模型训练与 TEST 评估。
5. 序列概率导出 (`sequence_predictions/val|test/*.csv`)。
6. 验证集预测的校准阶段。
7. 概率曲线绘图与触发策略检查。
8. SHAP 报告（DART 路径）作为可解释性产物。

## 4. EAST 迁移挂接点

迁移接口定义为契约，而非硬编码的装置逻辑：
- `shot_list/<device>/Disruption_<device>_TV.json`
- `shot_list/<device>/Non-disruption_<device>_TV.json`
- `shot_list/<device>/AdvancedTime_<device>.json`
- `<data_root>/<device>/unified_hdf5/<bucket>/<shot>.hdf5`

跨装置训练所需的挂接点：
- 具有相同特征模式的独立源/目标数据加载器,
- 域适配插入点 (CORAL/S-CORAL 损失),
- 每个目标域的校准,
- 带装置标签的可解释性导出。

## 5. 决策记录 (本周期)

### 为何选择 FLS + 灰区标注

- 固定窗口在破裂前过渡期产生标签噪声。
- 灰区排除使训练和校准的标签更加干净。
- 当前 `valid_label` 管线已支持此行为。

### 为何选择 DART + 校准 + SHAP

- DART 是 paper_131 对齐的可解释树基线，用于表格 PGFE 特征。
- 校准是必须的，因为触发策略依赖概率质量而非仅排序。
- SHAP 是实现面向操作人员的物理可解释性和特征审计所必需的。

### 为何需要面向迁移的 EAST 接口

- EAST 是近期的部署目标。
- 跨装置扩展（如 EAST -> J-TEXT 或反向）应复用相同的契约。
- 保持数据加载器和适配挂接点与装置无关，可避免后续迁移阶段的返工。
