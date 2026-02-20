"""Dedicated transfer workspace for plasma disruption modeling.

This package isolates transfer experiments from existing baseline/advanced
pipelines to avoid artifact and path confusion.
"""

from src.models.ultra_transfer.config import TransferWorkspaceConfig
from src.models.ultra_transfer.data import PreparedTransferData, prepare_transfer_data
from src.models.ultra_transfer.evaluation_adapter import (
    TransferEvaluationOutput,
    append_stability_if_requested,
    evaluate_transfer_predictions,
    persist_transfer_outputs,
)
from src.models.ultra_transfer.trainer import (
    TransferTrainOutput,
    compute_reason_attribution,
    train_and_infer,
)

__all__ = [
    "TransferWorkspaceConfig",
    "PreparedTransferData",
    "prepare_transfer_data",
    "TransferTrainOutput",
    "train_and_infer",
    "compute_reason_attribution",
    "TransferEvaluationOutput",
    "evaluate_transfer_predictions",
    "persist_transfer_outputs",
    "append_stability_if_requested",
]
