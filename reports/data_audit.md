# J-TEXT 数据审计 (Agent-2 MVP)

## 范围
- 数据集：`J-TEXT` 统一 HDF5。
- 目标：可复现的审计/清洗/拆分/标注构建，仅使用小型产物。
- 构建脚本：`src/data/build_dataset.py`。
- 运行命令：
  - `python src/data/build_dataset.py --repo-root . --dataset-name jtext_v1 --split-ratio 8,1,1 --seed 42`

## 发现的路径
- HDF5（按检测顺序）：
  - `D:/Fuison/data/J-TEXT/unified_hdf5`（已找到，选定）
  - `./data/J-TEXT/unified_hdf5`（在此工作区中为同一实际位置）
- 炮号列表：
  - `shot_list/J-TEXT/Disruption_J-TEXT.json`
  - `shot_list/J-TEXT/Non-disruption_J-TEXT.json`
  - `shot_list/J-TEXT/AdvancedTime_J-TEXT.json`

## 标签/元数据审计与清洗
- 炮号列表原始计数：
  - 破裂炮：`378`
  - 非破裂炮：`1356`
  - 跨列表冲突：`0`
  - 重复项（每个列表）：`0`
- 去重/冲突处理后的候选炮：`1734`
- 保留的清洁炮：`1733`
- 排除的炮：`1`
  - 原因：`missing_hdf5=1`
- 炮号列表 vs HDF5 标签一致性：
  - 与 `meta/IsDisrupt` 进行比对
  - 按规则移除的不匹配项（本次运行中观察到的数量：`0`）

## 特征覆盖策略 (20+ 信号)
- 所需特征覆盖 (从 HDF5 `data/*` 推断)：`23` 个信号。
- 策略：**默认全部使用，供下游训练**。
  - 本次运行强制每个保留炮具有所有推断的必需信号。
  - 缺少关键特征的炮按规则排除（本次运行中观察到的数量：`0`）。
- 特征列表产物：`artifacts/datasets/jtext_v1/required_features.json`

## 时间轴与 FLS/灰区标注
- dt 映射：从 HDF5 时间元数据 (`meta/StartTime`、`meta/DownTime`) 推断，可用 TTD 回退。
- 本次运行的时间源使用情况：
  - `meta_start_down=1733`
- 应用的标注逻辑：
  - 有 advanced_time 的破裂炮：
    - `[t_end - advanced_time, t_end] -> 1`
    - 之前 `30 ms` 灰区丢弃
    - 更早 -> `0`
  - 无 advanced_time 的破裂炮（回退 25ms）：
    - `[t_end - 25 ms, t_end] -> 1`
    - 之前 `30 ms` 灰区丢弃
    - 更早 -> `0`
  - 非破裂炮：
    - 全部 `0`
- FLS 来源计数：
  - `advanced_time=375`
  - `fallback_25ms=3`
  - `non_disruptive=1355`

## 拆分构建 (炮级分层，8/1/1)
- 训练集：`1386` 炮 (`302` 破裂，`1084` 非破裂)
- 验证集：`174` 炮 (`38` 破裂，`136` 非破裂)
- 测试集：`173` 炮 (`38` 破裂，`135` 非破裂)
- 拆分文件：
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
- 炮次计数参考（权威来源）：
  - `artifacts/datasets/jtext_v1/summary.json`：
    - `splits.train.shots=1386`
    - `splits.val.shots=174`
    - `splits.test.shots=173`
  - 拆分 ID 文件（每行一个炮号 ID；行数应匹配）：
    - `splits/train.txt`
    - `splits/val.txt`
    - `splits/test.txt`

## 类别不平衡处理
- 已考虑：
  - `weighted_random_sampler`（加权随机采样器）
  - `focal_loss_or_hard_negative_mining`（焦点损失或难例挖掘）
- 已实现：
  - `class_weighting`（炮级 + 时间点级 `bce_pos_weight`）
- 产物：
  - `artifacts/datasets/jtext_v1/class_weights.json`

## 数据集统计
- 原始数据点总数：`592175`
- 使用的数据点总数（灰区丢弃后）：`580954`
- 正类点：`7654`
- 负类点：`573300`
- 灰区丢弃点：`11221`

## 3 炮 y(t) 示例
- 说明：此处仅有意生成 3 个 `y(t)` 图作为代表性审计样本
  （每种标注场景一个），以保持此 MVP 审计的轻量和可读性。
- 完整测试炮时间线由下游评估单独保存：
  - `reports/plots/probability_timelines_test.csv`（所有测试炮时间线行）
  - `reports/plots/probability/`（渲染的时间线 PNG 子集）
- 有 advanced_time 的破裂炮：
  - 炮号 `1051501`
  - 图：`reports/plots/data_audit/shot_1051501_disrupt_with_advanced.png`
- 无 advanced_time 的破裂炮（回退 25ms）：
  - 炮号 `1053639`
  - 图：`reports/plots/data_audit/shot_1053639_disrupt_without_advanced.png`
- 非破裂炮：
  - 炮号 `1051500`
  - 图：`reports/plots/data_audit/shot_1051500_non_disruptive.png`

## 输出产物（小型）
- 核心统计与摘要：
  - `artifacts/datasets/jtext_v1/summary.json`
  - `artifacts/datasets/jtext_v1/stats.json`
- 小型表格输出：
  - `artifacts/datasets/jtext_v1/clean_shots.csv`
  - `artifacts/datasets/jtext_v1/excluded_shots.csv`
  - `artifacts/datasets/jtext_v1/example_shots.csv`
  - `artifacts/datasets/jtext_v1/label_examples.csv`
- 小型样本二进制文件：
  - `artifacts/datasets/jtext_v1/small_sample.npz`
- 拆分 ID 文件：
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`
