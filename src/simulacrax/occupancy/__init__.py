"""Occupancy flow prediction via Fourier Neural Operators.

This subpackage provides:
- OccupancyFlowModel: MultiScaleFNO-based future occupancy + flow prediction
- SceneRasterizer: differentiable SceneContext → OccupancyGrid conversion
- FlowConsistencyLoss: continuity equation ∂ρ/∂t + ∇·(ρv) = 0 as training loss
"""

from __future__ import annotations

from simulacrax.occupancy.flow_model import (
    create_occupancy_flow_model,
    OccupancyFlowConfig,
    OccupancyFlowModel,
    OccupancyGrid,
    OccupancyGridPrediction,
)
from simulacrax.occupancy.losses import FlowConsistencyLoss, FlowConsistencyLossConfig
from simulacrax.occupancy.rasterizer import RasterizerConfig, SceneRasterizer


__all__ = [
    "OccupancyFlowConfig",
    "OccupancyFlowModel",
    "OccupancyGrid",
    "OccupancyGridPrediction",
    "create_occupancy_flow_model",
    "FlowConsistencyLoss",
    "FlowConsistencyLossConfig",
    "RasterizerConfig",
    "SceneRasterizer",
]
