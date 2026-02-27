# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Model abstraction layer for trajectory prediction models."""

from .ar1_model import AR1Model
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction
# ManualModel requires pygame - lazy load only when needed
from .transfuser_model import TransfuserModel
from .vam_model import VAMModel


def _lazy_load_manual_model():
    """Lazy import ManualModel to avoid requiring pygame dependency."""
    from .manual_model import ManualModel
    return ManualModel


__all__ = [
    "AR1Model",
    "BaseTrajectoryModel",
    "DriveCommand",
    "ModelPrediction",
    "TransfuserModel",
    "VAMModel",
    "_lazy_load_manual_model",
]
