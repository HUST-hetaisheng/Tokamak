# 多 Agent 进度看板

最后更新：2026-02-19

## Agent-1 (研究员 / 学习者)
状态：已重启 (进行中)
已完成：
- 阅读 `ref/paper_131.txt` 及与机制提取和方法链总结相关的项目文档。
- 完成了机制分类和可观测特征关系（密度极限、锁模、VDE/低 q、杂质/辐射路径）。
- 完成了 PGFE、FLS、(S-)CORAL、DART、SHAP 和 EFD 的方法链总结及可复现性说明。
- 保留了现有研究文档，并在 `docs/requirements.md` 中添加了将输出产物映射到物理可解释性概念的附录。
待办：
- 随着 Agent-2/Agent-3 输出的演进，维护产物-可解释性映射。
- 如评审反馈要求更严格的可追溯性，修订术语/引用。
阻塞项：
- 无。
产物：
- `docs/literature_review.md`
- `docs/requirements.md`
- `docs/feature_physics_map.md`

## Agent-2 (数据工程师)
状态：已完成
已完成：
- 按要求顺序检测数据根目录并选择了 `D:/Fuison/data/J-TEXT/unified_hdf5`。
- 定位了 `shot_list/J-TEXT/` 下的 J-TEXT 炮号列表和高级时间元数据文件。
- 实现了可复现的审计/清洗/拆分管线：`src/data/build_dataset.py`。
- 运行了有限的 MVP 构建并在 `artifacts/datasets/jtext_v1/` 下生成了小型产物。
- 构建了分层炮次拆分 (8/1/1) 到 `splits/train.txt`、`splits/val.txt`、`splits/test.txt`。
- 在 `reports/data_audit.md` 和 `reports/plots/data_audit/` 中生成了数据审计报告和 3 炮 `y(t)` 图。
- 在 `reports/data_audit.md` 中说明了 3 炮 `y(t)` 图仅为样本审计可视化；完整测试时间线位于 `reports/plots/probability_timelines_test.csv`（渲染子集位于 `reports/plots/probability/`）。
- 在 `reports/data_audit.md` 和 `artifacts/datasets/jtext_v1/summary.json` 中添加了明确的训练/验证/测试炮次计数参考，拆分 ID 来源为 `splits/train.txt`、`splits/val.txt` 和 `splits/test.txt`。
待办：
- 将拆分/产物输出移交给 Agent-3 用于下游训练。
- 如评审/建模反馈要求替代比例或更严格的过滤器，扩展管线选项。
阻塞项：
- 无。
产物：
- `src/data/build_dataset.py`
- `artifacts/datasets/jtext_v1/summary.json`
- `artifacts/datasets/jtext_v1/stats.json`
- `artifacts/datasets/jtext_v1/small_sample.npz`
- `artifacts/datasets/jtext_v1/clean_shots.csv`
- `artifacts/datasets/jtext_v1/excluded_shots.csv`
- `artifacts/datasets/jtext_v1/class_weights.json`
- `artifacts/datasets/jtext_v1/required_features.json`
- `artifacts/datasets/jtext_v1/example_shots.csv`
- `artifacts/datasets/jtext_v1/label_examples.csv`
- `splits/train.txt`
- `splits/val.txt`
- `splits/test.txt`
- `reports/data_audit.md`

## Agent-3 (建模者 / 实验者)
状态：已完成
已完成：
- 添加了训练 CLI 绘图控制 `--plot-shot-limit` 和 `--plot-all-test-shots`，以解决有限的时间线导出问题。
- 添加了验证阈值目标选择 `--threshold-objective {youden,accuracy,shot_fpr_constrained}`。
- 在完整拆分规模上运行了续训（训练=1386，验证=174，测试=173）；验证集拆分为校准=87 和阈值=87 炮。
- 保持了 23 特征全用策略 (23/23)，并在 `training_config.json` 中持久化。
- 生成了 `173` 张测试炮概率时间线 PNG。
- 生成了 `38` 个单破裂炮原因行（TEST 中预期破裂炮：38）。
- 当前测试指标：accuracy=0.990885，roc_auc=0.978437，shot_accuracy=0.953757，threshold=0.668796 (shot_fpr_constrained)。
待办：
- 与评审人协调阈值目标权衡和校准保留策略。
阻塞项：
- 无。
产物：
- `src/models/train.py`
- `src/models/eval.py`
- `src/models/calibrate.py`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/model_xgb_dart.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/calibrator.joblib`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/training_config.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/metrics_summary.json`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/shap_topk.csv`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/warning_summary_test.csv`
- `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/disruption_reason_per_shot.csv`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/metrics.md`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/calibration_curve_test.png`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/probability_timelines_test.csv`
- `reports/iters/sfpr002_d4_e260_lr004_s3_reason/plots/probability`
## Agent-4 (评审人 / 维护者)
状态：进行中
已完成：
- 完成了初始仓库扫描和风险检查清单设置。
- 在 `docs/code_review_log.md` 中添加了第一个带日期的条目。
- 评审了 Agent-2 管线和元数据产物：`src/data/build_dataset.py`、`reports/data_audit.md`、`artifacts/datasets/jtext_v1/*` 和 `splits/*.txt`。
- 在 `docs/code_review_log.md` 中追加了带严重等级标签的发现，含文件/函数引用。
- 评审了 Agent-3 输出：`src/models/train.py`、`src/models/eval.py`、`src/models/calibrate.py`、`artifacts/models/best/metrics_summary.json`、`reports/metrics.md`。
- 在 `docs/code_review_log.md` 中追加了按严重等级排序的 Agent-3 发现（阈值策略不匹配、校准过拟合风险、标签完整性检查和可复现性缺口）。
- 评审了 Agent-3 重启增量并在 `docs/code_review_log.md` 中追加了关于阈值目标正确性、校准有效性偏差、标签完整性泄漏防护缺口和绘图行为诊断的按严重等级排序的发现。
待办：
- 使用重跑产物验证 Agent-3 的修复，优先关注与炮级误报率目标对齐的阈值搜索和校准/阈值拆分隔离。
- 重新检查绘图输出中代表性的 TP/FP/TN/FN 时间线和修正后的校准区间计数报告。
阻塞项：
- 等待 Agent-3 对 `docs/code_review_log.md` 中记录的重启发现进行修复更新。
产物：
- `docs/code_review_log.md`

## Agent-5 (技术写作员)
状态：已完成
已完成：
- 在 `README.md` 中添加了产物指南（路径 + 角色 + 查阅时机），涵盖重启关键文件。
- 明确说明了默认仅生成少量炮号 PNG，并指向完整的 173 炮 CSV 产物。
- 使用当前训练参数更新了 `README.md` 中的全炮次重跑命令（`--max-train-shots 0`、`--max-val-shots 0`、`--max-test-shots 0`、`--plot-all-test-shots`、`--threshold-objective accuracy`）。
- 在 `docs/changelog.md` 中添加了重启阶段备注。
待办：
- 随着未来 EAST/J-TEXT 重跑和输出路径变更，保持产物指针同步。
阻塞项：
- 无。
产物：
- `README.md`
- `docs/changelog.md`
- `docs/progress.md`

## Agent-6 (高级建模者)
状态：已完成
已完成：
- 添加了 `src/models/sequence_arch.py`，包含 `TemporalTransformerClassifier`（时序Transformer分类器）、`MambaLiteClassifier`（轻量Mamba分类器）和 `GRUClassifier`（GRU门控循环单元分类器）。
- 添加了 `src/models/train_advanced.py`，包含受限高级训练扫描、校准、炮级阈值选择、概率时间线导出和梯度×输入原因分析。
- 执行了 3 次公平窗口运行 (`window_size=128`, `stride=16`)：transformer_small / mamba_lite / gru。
- 最佳运行：`adv_mamba_lite_ws128_st16_e5_s42`，test_acc=0.988993，roc_auc=0.990111，shot_acc=0.976879，shot_fpr=0.000000。
待办：
- 扩展到更大窗口/视野消融实验，并在相同架构骨干上添加跨装置迁移挂接点。
阻塞项：
- 无。
产物：
- `src/models/sequence_arch.py`
- `src/models/train_advanced.py`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_transformer_small_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_transformer_small_ws128_st16_e5_s42/plots/probability`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_mamba_lite_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_mamba_lite_ws128_st16_e5_s42/plots/probability`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/training_config.json`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/metrics_summary.json`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/warning_summary_test.csv`
- `artifacts/models/iters/adv_gru_ws128_st16_e5_s42/disruption_reason_per_shot.csv`
- `reports/iters/adv_gru_ws128_st16_e5_s42/metrics.md`
- `reports/iters/adv_gru_ws128_st16_e5_s42/plots/probability_timelines_test.csv`
- `reports/iters/adv_gru_ws128_st16_e5_s42/plots/probability`
- `reports/iters/advanced_summary.csv`
- `reports/iters/advanced_summary.md`