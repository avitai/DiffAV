"""Shared helpers for model tests."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)
from tests import support


# Shared test constants
TEST_HIDDEN = 32
TEST_BLOCKS = 1
TEST_TEMPORAL_LAYERS = 1
TEST_SOCIAL_LAYERS = 1
TEST_HEADS = 2
TEST_FUTURE = 4
TEST_AGENTS = 4
TEST_STEPS = 10
TEST_CTX_DIM = 16
TEST_STATE_DIM = 4


def small_diffusion_config(**overrides: Any) -> TrajectoryDiffusionConfig:
    """Create a small TrajectoryDiffusionConfig for testing.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        Configuration suitable for fast tests.
    """
    defaults: dict[str, Any] = {
        "hidden_dim": TEST_HIDDEN,
        "num_blocks": TEST_BLOCKS,
        "num_temporal_layers": TEST_TEMPORAL_LAYERS,
        "num_social_layers": TEST_SOCIAL_LAYERS,
        "num_heads": TEST_HEADS,
        "future_steps": TEST_FUTURE,
        "num_agents_max": TEST_AGENTS,
        "num_timesteps": TEST_STEPS,
        "context_dim": TEST_CTX_DIM,
        "state_dim": TEST_STATE_DIM,
    }
    defaults.update(overrides)
    return support.make_diffusion_config(**defaults)


def make_model(
    config: TrajectoryDiffusionConfig | None = None,
    seed: int = 0,
) -> TrajectoryDiffusionModel:
    """Create a small TrajectoryDiffusionModel for testing.

    Args:
        config: Optional config override.
        seed: Parameter initialization seed.

    Returns:
        Initialized model.
    """
    if config is None:
        config = small_diffusion_config()
    return support.make_diffusion_model(config, seed=seed)


def sample_data(
    num_agents: int = 3,
) -> tuple[jax.Array, jax.Array]:
    """Create sample (trajectories, scene_context) for testing.

    Args:
        num_agents: Number of agents.

    Returns:
        Tuple of (trajectories, scene_context).
    """
    trajectories = jnp.ones((num_agents, TEST_FUTURE, TEST_STATE_DIM))
    scene_context = jnp.ones((num_agents, TEST_CTX_DIM))
    return trajectories, scene_context
