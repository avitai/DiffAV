"""Smoke checks for sister-repo APIs consumed by Simulacrax."""

from __future__ import annotations


def test_datarax_symbols_importable() -> None:
    """Critical datarax interfaces used by Simulacrax should be importable."""
    from datarax.core.config import OperatorConfig, StructuralConfig
    from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
    from datarax.core.data_source import DataSourceModule
    from datarax.core.operator import OperatorModule
    from datarax.operators.composite_operator import CompositeOperatorModule

    assert OperatorConfig is not None
    assert StructuralConfig is not None
    assert CrossModalOperator is not None
    assert CrossModalOperatorConfig is not None
    assert DataSourceModule is not None
    assert OperatorModule is not None
    assert CompositeOperatorModule is not None


def test_artifex_symbols_importable() -> None:
    """Critical artifex interfaces used by Simulacrax should be importable."""
    from artifex.generative_models.core.configuration import NoiseScheduleConfig
    from artifex.generative_models.core.layers.egnn import EGNNLayer
    from artifex.generative_models.core.layers.transformers import (
        TransformerDecoderBlock,
        TransformerEncoderBlock,
    )
    from artifex.generative_models.core.noise_schedule import create_noise_schedule
    from artifex.generative_models.modalities.multi_modal.representations import (
        CrossModalAttention,
    )
    from artifex.generative_models.training.rl.rewards import CompositeReward

    assert NoiseScheduleConfig is not None
    assert EGNNLayer is not None
    assert TransformerDecoderBlock is not None
    assert TransformerEncoderBlock is not None
    assert create_noise_schedule is not None
    assert CrossModalAttention is not None
    assert CompositeReward is not None


def test_opifex_symbols_importable() -> None:
    """Critical opifex interfaces used by Simulacrax should be importable."""
    from opifex.core.physics.losses import AdaptiveWeightScheduler
    from opifex.core.training.components.checkpoint_store import OrbaxCheckpointStore
    from opifex.core.training.components.recovery import ErrorRecoveryManager
    from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

    assert AdaptiveWeightScheduler is not None
    assert OrbaxCheckpointStore is not None
    assert ErrorRecoveryManager is not None
    assert OptimizerConfig is not None
    assert create_optimizer is not None


def test_calibrax_symbols_importable() -> None:
    """Representative calibrax interfaces should remain importable."""
    from calibrax.exporters import PublicationGenerator
    from calibrax.profiling import ResourceMonitor, TimingCollector

    assert PublicationGenerator is not None
    assert TimingCollector is not None
    assert ResourceMonitor is not None
