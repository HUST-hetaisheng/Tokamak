# Ultra Transfer Workspace

This directory is intentionally separated from `src/models/advanced` and
`src/models/baseline` so new transfer experiments do not mix with previous
pipelines.

## Entry Point

- `src/models/ultra_transfer/train_transfer_sequence.py`

Default mode is direct ultra-transfer pipeline execution (`--mode run`).
Compatibility mode is still available for legacy forwarding (`--mode forward`).

## Core Modules (new)

- `src/models/ultra_transfer/config.py`：独立工作区配置
- `src/models/ultra_transfer/data.py`：数据加载、窗口化、归一化
- `src/models/ultra_transfer/trainer.py`：模型训练与推理封装
- `src/models/ultra_transfer/evaluation_adapter.py`：校准、阈值、告警与产物落盘适配

The launcher forwards all unknown CLI args to:

- `src/models/advanced/train_sequence.py`

while enforcing isolated defaults:

- `--output-root artifacts/models/ultra_transfer`
- `--report-root reports/ultra_transfer`

## Example

```bash
python src/models/ultra_transfer/train_transfer_sequence.py --mode run --models mamba_lite --epochs 8
```

```bash
python src/models/ultra_transfer/train_transfer_sequence.py --mode module-check
```
