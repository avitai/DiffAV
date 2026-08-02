"""Shared fixtures for occupancy tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.core.types import AgentState, AgentType, SceneContext
from simulacrax.occupancy.flow_model import OccupancyFlowConfig, OccupancyFlowModel, OccupancyGrid
from simulacrax.occupancy.rasterizer import RasterizerConfig, SceneRasterizer


@pytest.fixture()
def small_config() -> OccupancyFlowConfig:
    """Minimal config for fast CPU tests (32x32 grid, tiny FNO)."""
    return OccupancyFlowConfig(
        grid_resolution=32,
        modes_per_scale=(2, 1, 1),
        num_layers_per_scale=(1, 1, 1),
        hidden_channels=4,
        attention_heads=4,
        use_gradient_checkpointing=False,
    )


@pytest.fixture()
def small_model(small_config: OccupancyFlowConfig) -> OccupancyFlowModel:
    """Session-scoped OccupancyFlowModel for shape/gradient tests."""
    from simulacrax.occupancy.flow_model import create_occupancy_flow_model

    return create_occupancy_flow_model(small_config, nnx.Rngs(params=jax.random.key(0)))


@pytest.fixture
def small_grid() -> OccupancyGrid:
    """Empty 32x32 OccupancyGrid for forward-pass tests."""
    H = 32
    return OccupancyGrid(
        occupancy=jnp.zeros((H, H, 3)),
        flow=jnp.zeros((H, H, 2)),
        map_features=jnp.zeros((H, H, 3)),
    )


@pytest.fixture
def simple_scene() -> SceneContext:
    """Minimal SceneContext with ego vehicle at origin."""
    ego = AgentState(
        position=jnp.zeros(2),
        heading=0.0,
        velocity=0.0,
        acceleration=0.0,
        agent_type=AgentType.VEHICLE,
    )
    return SceneContext(
        ego_state=ego,
        agent_states=(),
        map_features=(),
        timestamps=jnp.array([0.0]),
    )


@pytest.fixture
def small_rasterizer() -> SceneRasterizer:
    """32x32 SceneRasterizer for rasterizer tests."""
    return SceneRasterizer(RasterizerConfig(grid_resolution=32))
