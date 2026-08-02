"""Trajectory model wrappers built on artifex generative models."""

from simulacrax.models.checkpointing import (
    CheckpointConfig,
    CheckpointCorruptError,
    SimulacraxCheckpointManager,
    TrainingState,
)
from simulacrax.models.factorized_backbone import (
    FactorizedSceneBackbone,
    FactorizedSceneBackboneConfig,
)
from simulacrax.models.trainer import (
    TrainerConfig,
    TrainingMetrics,
    TrajectoryTrainer,
)
from simulacrax.models.trajectory_diffusion import (
    create_trajectory_model,
    DiffusionLossOutputs,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)


__all__ = [
    "CheckpointConfig",
    "CheckpointCorruptError",
    "DiffusionLossOutputs",
    "FactorizedSceneBackbone",
    "FactorizedSceneBackboneConfig",
    "SimulacraxCheckpointManager",
    "TrainerConfig",
    "TrainingMetrics",
    "TrainingState",
    "TrajectoryDiffusionConfig",
    "TrajectoryDiffusionModel",
    "TrajectoryTrainer",
    "create_trajectory_model",
]
