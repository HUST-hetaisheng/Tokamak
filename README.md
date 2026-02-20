# Fuison: 托卡马克破裂预测

本仓库面向实时托卡马克等离子体破裂预测，核心设计对齐相关前沿研究（paper_131）：
- 物理引导特征管线 (PGFE)
- 灰区感知标注 (FLS 兼容)
- 校准后的破裂概率输出
- 包含传统机器学习基线(XGBoost)和深度时序模型(Transformer, Mamba, GRU)
- 面向不同装置（EAST, J-TEXT）的迁移学习与适配。

## 目录结构

```text
/
├── src/
│   ├── data/
│   │   ├── build_dataset.py       # 数据集构建脚本
│   │   └── read_hdf5_structure.py # HDF5数据读取工具
│   ├── models/
│   │   ├── baseline/
│   │   │   └── train_xgb.py       # XGBoost 等传统简单模型训练脚本
│   │   └── advanced/
│   │       ├── train_sequence.py  # 包含 Transformer, MambaLite, GRU 等前沿时序模型训练脚本
│   │       └── sequence_arch.py   # 深度学习网络架构定义
│   └── evaluation/
│       ├── eval.py                # 评估指标脚本
│       ├── calibrate.py           # 概率校准脚本
│       └── generate_reason_report.py # 生成破裂原因报告
├── docs/
│   ├── architecture.md            # 系统架构文档
│   ├── literature/                # 参考文献与前沿调研
│   │   └── paper_131.txt
│   ├── notes/                     # 项目会议、个人随记、图片整理
│   └── research/                  # 方法调研与框架分析
├── data/                          # 数据集存放处（软链接或本地挂载）
├── shot_list/                     # 各装置的炮号列表 (EAST, J-TEXT 等)
├── splits/                        # 数据切分配置文件
├── artifacts/                     # 训练产生的模型文件与数据集特征字典
└── reports/                       # 生成的指标报告与图表
```

## 数据路径配置

本仓库推荐的数据根目录：

```powershell
$env:FUISON_DATA_ROOT = "G:\我的云端硬盘\Fuison\data"
if (-not (Test-Path $env:FUISON_DATA_ROOT)) { throw "FUISON_DATA_ROOT not found: $env:FUISON_DATA_ROOT" }
```

如果 HDF5 仅存放在云端根目录，请创建一次链接：

```powershell
if (-not (Test-Path "data/EAST/unified_hdf5")) {
  cmd /c mklink /J data\EAST\unified_hdf5 "$env:FUISON_DATA_ROOT\EAST\unified_hdf5"
}
```

## 环境搭建

推荐使用 Python 3.10+ 创建虚拟环境，并安装相关依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas h5py scikit-learn torch matplotlib joblib xgboost shap
```

## 运行说明

### 1) 构建数据集
```powershell
python src/data/build_dataset.py --target-dir artifacts/datasets/jtext_v1
```

### 2) 训练 Baseline (XGBoost)
```powershell
python src/models/baseline/train_xgb.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/best
```

### 3) 训练 Advanced Models (Mamba / Transformer / GRU)
```powershell
python src/models/advanced/train_sequence.py --dataset-dir artifacts/datasets/jtext_v1 --output-dir artifacts/models/iters/adv_mamba_lite --models mamba_lite
```

### 4) 生成评估和归因报告
```powershell
python src/evaluation/generate_reason_report.py --run-dir artifacts/models/iters/adv_mamba_lite --dataset-dir artifacts/datasets/jtext_v1
```
