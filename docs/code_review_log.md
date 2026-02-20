# 代码评审记录

最后更新：2026-02-19

## Agent-4 评审检查清单
- 数据泄漏：训练/验证/测试在炮级和装置级的隔离。
- 标签泄漏：破裂标签或高级时间标签未直接或通过代理泄漏到特征中。
- 时间轴完整性：仅因果窗口，无未来样本，信号间时钟对齐。
- 指标有效性：事件级和时间点级指标均以明确的阈值协议报告。
- 概率质量：在做出运行声明前检查校准（如可靠性曲线、ECE/Brier）。
- 可复现性：固定种子，记录确定性设置，拆分产物版本化。

## 2026-02-19 初始基线扫描

范围：
- 审阅 `docs/progress.md` 了解 Agent-2/3/5 的产物状态。
- 扫描仓库文件树和 `analysis/` 中的可执行训练/评估代码。
- 使用 `rg --files -g "*.py"` 检查现有 Python 源文件。

发现：
- [信息] Agent-2、Agent-3 或 Agent-5 尚无可审阅的输出。当前 `docs/progress.md` 将其标记为排队或依赖等待。
- [中等] 当前工作区状态存在可复现性风险：未检测到 Python 源文件，因此无法执行冒烟验证和泄漏检查。

可操作建议：
- Agent-2：添加数据集构建脚本时，将拆分生成和特征组装公开为函数，带确定性种子参数和明确的炮级拆分断言。
- Agent-3：添加训练/评估脚本时，按拆分分离拟合/变换阶段并包含校准评估产物（`calibration_curve`、`brier_score` 或 ECE）。
- Agent-5：在变更日志/架构文档中记录确切命令、种子、数据根目录和产物路径以保持运行可复现。

## 2026-02-19 评审周期：Agent-2 数据集管线

范围：
- 审阅 `src/data/build_dataset.py`。
- 审阅 `reports/data_audit.md`。
- 审阅 `artifacts/datasets/jtext_v1/` 下的数据集元数据产物。
- 验证 `splits/train.txt`、`splits/val.txt`、`splits/test.txt` 中拆分文件的不相交性。

发现：
- [中等][可复现性/数据源策略] 硬编码的 J-TEXT 根目录选择未遵循仓库默认数据根策略。
  - 证据：`src/data/build_dataset.py:58`、`src/data/build_dataset.py:60`、`artifacts/datasets/jtext_v1/summary.json:7`。
  - 风险：运行可能静默使用机器本地的数据快照，降低跨环境可复现性。
  - 建议：添加明确的 `--data-root`（必填，除非用户覆盖），默认值来自仓库策略根目录，并在摘要元数据中持久化解析后的来源。

- [中等][数据泄漏/输入漂移风险] 元数据 JSON 发现是对所有仓库 JSON 文件的启发式搜索。
  - 证据：`src/data/build_dataset.py:76`、`src/data/build_dataset.py:85`、`src/data/build_dataset.py:104`。
  - 风险：如果存在多个 J-TEXT JSON 变体，管线可能在不失败的情况下绑定到非预期文件，导致拆分/标签漂移或从衍生文件的意外泄漏。
  - 建议：首先解析 `shot_list/J-TEXT/` 下的固定规范路径，遇到歧义时封闭失败，并记录文件哈希。

- [中等][时间/标签有效性] 混合破裂标注策略处于活跃状态（`advanced_time` + 缺失高级时间炮的固定窗口回退）。
  - 证据：`src/data/build_dataset.py:243`、`src/data/build_dataset.py:247`、`artifacts/datasets/jtext_v1/summary.json:30`、`artifacts/datasets/jtext_v1/summary.json:34`。
  - 风险：破裂类的训练目标不均一；回退标注的炮可能偏移边界行为并扭曲预警时间指标。
  - 建议：分别报告 `fls_source=advanced_time` 和 `fallback_25ms` 的指标，并倾向于在高级标签完整之前将回退标注的破裂炮从验证/测试集中排除。

- [中等][评估偏差风险] 拆分策略仅为炮级分层，未按时间/实验会话分组。
  - 证据：`src/data/build_dataset.py:279`、`src/data/build_dataset.py:500`。
  - 风险：来自同一实验运行的相邻炮可能被拆分到训练/验证/测试中，夸大表面泛化能力。
  - 建议：添加分组时间拆分选项（如按连续炮号范围或实验会话 ID），并与随机分层拆分进行比较。

- [中等][可复现性缺口] 运行定义的超参数未完全捕获在持久化摘要中。
  - 证据：`src/data/build_dataset.py:673`（摘要内容），缺少 `seed`、`gray_ms`、`fallback_fls_ms`、`fallback_dt_ms`、`reconcile_len_tol`。
  - 风险：仅凭产物无法保证精确的数据集重建。
  - 建议：在 `summary.json` 中持久化完整的 CLI/配置块和脚本版本/提交信息。

检查通过：
- [信息] 当前产物集中各拆分无炮号重叠，各拆分内无重复 ID。
- [信息] `clean_shots.csv` 中无非破裂炮带正标签，也无破裂炮零正标签的情况。
- [信息] 所有审阅炮的 `positive_start_ms` 保持在估计时间范围内。

待下次评审触发条件：
- `src/models/*.py`（尚未存在）。
- `reports/metrics.md`（尚未存在）。

## 2026-02-19 评审周期：Agent-3 模型/评估/校准

范围：
- 审阅 `src/models/train.py`。
- 审阅 `src/models/eval.py`。
- 审阅 `src/models/calibrate.py`。
- 审阅 `artifacts/models/best/metrics_summary.json`。
- 审阅 `reports/metrics.md`。

发现（按严重等级排序）：
- [严重][指标/预警策略不对齐] 阈值通过时间点级约登 J 指数选择，产生过于激进的低操作点，在部署级指标中具有高误报率。
  - 证据：`src/models/train.py:606`、`src/models/eval.py:59`、`artifacts/models/best/metrics_summary.json:124`、`artifacts/models/best/metrics_summary.json:129`、`artifacts/models/best/metrics_summary.json:142`、`reports/metrics.md:25`、`reports/metrics.md:30`。
  - 观察到的影响：`theta=0.019833`，时间点 `fpr=0.194614`，炮级 `shot_fpr=0.288889`。
  - 风险：预警流可靠性低（误报在运行中代价高昂），尽管召回率高。
  - 建议：在明确 FAR（误报率）上限下使用炮级目标选择阈值（如在 `shot_fpr <= target` 约束下优化提前量/TPR），而非仅使用时间点级约登 J 指数。

- [严重][校准过拟合风险] 等保校准在同一验证集上拟合和评估，该同一校准后的验证分布也用于选择阈值。
  - 证据：`src/models/train.py:601`、`src/models/train.py:604`、`src/models/train.py:606`、`artifacts/models/best/metrics_summary.json:152`、`artifacts/models/best/metrics_summary.json:159`。
  - 观察到的信号：验证 ECE 在校准拟合拆分上坍缩至近零 (`2.499e-09`)，这是乐观的。
  - 风险：校准质量可能被高估；部署阈值可能存在偏差。
  - 建议：使用专用的校准保留集（或交叉拟合校准器），与阈值选择数据分离，并将保留/测试校准作为首要报告。

- [中等][标签完整性漂移风险] 训练忽略加载时从 HDF5 读取的单炮元数据标签，仅信任 `clean_shots.csv` 标签。
  - 证据：`src/models/train.py:216`、`src/models/train.py:253`、`src/models/train.py:530`。
  - 风险：如果 HDF5 内容在数据集构建后漂移，静默标签不匹配可污染训练/评估。
  - 建议：在训练数据加载期间断言 `meta_label == expected_label`，不匹配时带炮号硬失败。

- [中等][时间/标注协议不一致] `train.py` 在 `advanced_ms` 存在时始终使用（包括非正值），而数据集构建逻辑将非正高级时间视为回退。
  - 证据：`src/models/train.py:172`、`src/data/build_dataset.py:243`。
  - 风险：未来的元数据异常（`advanced_time <= 0`）可能创建无效正窗口和不稳定指标。
  - 建议：在训练/评估标签生成中镜像数据集构建条件（`advanced_ms is not None and advanced_ms > 0`）。

- [中等][可复现性缺口] 运行确定性不完整；多线程 XGBoost 和不完整的运行指纹可能在不同环境间产生不可复现的差异。
  - 证据：`src/models/train.py:84`、`src/models/train.py:327`、`src/models/train.py:683`。
  - 风险：重复运行可能存在细微差异且无法追溯原因。
  - 建议：添加确定性模式选项（可复现运行时 `n_jobs=1` 开关），并在 `training_config.json` 中持久化库版本 + 拆分文件哈希。

检查通过：
- [信息] 校准在模型训练之后、仅在验证集上拟合（未检测到直接的测试集拟合泄漏）。
- [信息] 测试集在本次运行中仅用于最终报告指标的评估。

## 2026-02-19 跟进：阈值/校准修复已应用

范围：
- 审阅以下实现更新：
  - `src/models/eval.py`（`choose_threshold_by_shot_fpr`）
  - `src/models/train.py`（验证集拆分为校准/阈值子集）
  - `artifacts/models/iters/sfpr002_d4_e260_lr004_s3_reason/` 下的新运行产物

发现：
- [信息] 先前的校准过拟合风险已降低：校准器拟合和阈值选择现使用不相交的验证炮子集。
- [信息] 阈值搜索现支持炮级 FAR 约束策略（`threshold_objective=shot_fpr_constrained`，`max_shot_fpr` 可配置）。
- [中等] 仍然建议：为阈值子集的鲁棒性添加专用的时间/会话拆分选项以应对实验运行漂移。

## 2026-02-19 评审周期：Agent-3 重启增量 (阈值/泄漏/校准/绘图)

范围：
- 重新审阅 `src/models/train.py`、`src/models/eval.py` 和 `src/models/calibrate.py`。
- 重新审阅重启产物：`artifacts/models/best/metrics_summary.json`、`artifacts/models/best/calibration_curve_points_test.csv`、`artifacts/models/best/warning_summary_test.csv`、`reports/metrics.md` 和 `reports/plots/probability_timelines_test.csv`。
- 使用当前持续预警逻辑从导出的测试时间线重放炮级策略结果。

发现（按严重等级排序）：
- [严重][阈值目标不正确] 阈值搜索仍为时间点级约登 J 指数，与部署的炮级预警目标和 FAR 控制脱钩。
  - 证据：`src/models/eval.py:59`、`src/models/train.py:606`、`src/models/train.py:618`、`artifacts/models/best/metrics_summary.json:163`、`artifacts/models/best/metrics_summary.json:142`。
  - 观察到的影响（从 `reports/plots/probability_timelines_test.csv` 使用 `src/models/eval.py:84` 逻辑的策略重放）：当前 `theta=0.019833` 给出 `shot_fpr=0.288889`，而 `theta=0.100000` 给出 `shot_fpr=0.051852`，`shot_tpr=0.921053` 且 `shot_accuracy` 显著更高。
  - 风险：预警流操作点与部署可靠性目标不对齐。
  - 建议：将仅约登搜索替换为在明确 FAR 上限下的炮级策略优化（如在 `shot_fpr <= target` 约束下优化提前量/TPR）。

- [严重][校准有效性偏差] 等保校准器拟合、校准质量报告和阈值选择仍使用同一验证分布。
  - 证据：`src/models/train.py:601`、`src/models/train.py:604`、`src/models/train.py:606`、`artifacts/models/best/metrics_summary.json:118`、`artifacts/models/best/metrics_summary.json:159`。
  - 观察到的信号：验证 ECE 在用于拟合等保和选择阈值的同一拆分上坍缩至近零 (`2.499e-09`)。
  - 风险：校准和阈值质量过于乐观；操作点可能无法稳健迁移。
  - 建议：使用独立的校准保留集或交叉拟合，并在非校准器参数拟合数据上进行阈值搜索。

- [中等][标签完整性/泄漏防护缺失] HDF5 破裂标签被读取但丢弃，未对 `clean_shots.csv` 强制一致性断言。
  - 证据：`src/models/train.py:209`、`src/models/train.py:216`、`src/models/train.py:253`、`src/models/train.py:530`。
  - 风险：静默元数据漂移可在未被检测的情况下污染训练/评估标签。
  - 建议：对每炮断言 `meta_label == expected_label`，不匹配时带炮号硬失败。

- [中等][绘图代表性缺口] 时间线图选择逻辑确定性地选取前三个有预警的破裂炮，排除了负样本/误报示例。
  - 证据：`src/models/train.py:632`、`src/models/train.py:635`、`reports/plots/probability/shot_1051510_timeline.png`、`reports/plots/probability/shot_1051684_timeline.png`、`reports/plots/probability/shot_1051701_timeline.png`、`artifacts/models/best/warning_summary_test.csv:2`。
  - 风险：可视化高估了定性行为，隐藏了当前 FAR 关注点中主导的假阳性动态。
  - 建议：导出固定的代表性集合（如 TP/FP/TN/FN 或风险最高的非破裂炮），而非按排序取前 3 个。

- [中等][校准图 CSV 不一致] `calibration_curve_points_test.csv` 仅记录原始概率区间的 `count`；校准后区间的样本量未记录且可能有显著差异。
  - 证据：`src/models/eval.py:189`、`src/models/eval.py:195`、`artifacts/models/best/calibration_curve_points_test.csv:1`、`reports/plots/probability_timelines_test.csv:1`。
  - 观察到的信号：本次运行中所有 15 个区间的原始 vs 校准后区间人口均有差异。
  - 风险：下游读者可能以不正确的区间样本量误解校准后的可靠性点。
  - 建议：分别写入 `count_raw` 和 `count_cal` 列（或单独的表），并使用各自系列的计数。

检查通过：
- [信息] 重启中未引入直接的训练/测试泄漏路径：模型拟合使用训练拆分；校准器拟合排除测试拆分；测试仅用于评估。
