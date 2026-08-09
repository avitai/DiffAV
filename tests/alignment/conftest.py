"""Shared fixtures for alignment module tests.

Provides synthetic trajectory data for reward and preference testing.
Trajectories use shape ``(batch, num_agents, future_steps, 4)`` with
state ``[x, y, heading, velocity]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)
from tests import support
from tests.alignment.helpers import (
    BATCH_SIZE,
    CONTEXT_DIM,
    FUTURE_STEPS,
    NUM_AGENTS,
    STATE_DIM,
)


@pytest.fixture()
def straight_trajectories() -> jax.Array:
    """Constant-velocity straight-line trajectories.

    Agents are well-separated in the y direction (10 m apart)
    travelling at 5 m/s along x with heading=0.
    Shape: ``(BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, 4)``.
    """
    dt = 0.1
    t = jnp.arange(FUTURE_STEPS) * dt
    # Per-agent x = velocity * time
    x = jnp.broadcast_to(t * 5.0, (NUM_AGENTS, FUTURE_STEPS))
    # Separate agents by 10m in y
    y_offsets = jnp.arange(NUM_AGENTS)[:, None] * 10.0
    y = jnp.zeros((NUM_AGENTS, FUTURE_STEPS)) + y_offsets
    heading = jnp.zeros((NUM_AGENTS, FUTURE_STEPS))
    velocity = jnp.full((NUM_AGENTS, FUTURE_STEPS), 5.0)
    single = jnp.stack([x, y, heading, velocity], axis=-1)
    # Replicate across batch
    return jnp.broadcast_to(single[None], (BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, STATE_DIM))


@pytest.fixture()
def collision_trajectories() -> jax.Array:
    """All agents at the same position (maximum collision).

    Shape: ``(BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, 4)``.
    """
    dt = 0.1
    t = jnp.arange(FUTURE_STEPS) * dt
    single_agent = jnp.stack(
        [
            t * 5.0,
            jnp.zeros(FUTURE_STEPS),
            jnp.zeros(FUTURE_STEPS),
            jnp.full(FUTURE_STEPS, 5.0),
        ],
        axis=-1,
    )
    # All agents identical position
    single_scene = jnp.broadcast_to(single_agent[None], (NUM_AGENTS, FUTURE_STEPS, STATE_DIM))
    return jnp.broadcast_to(single_scene[None], (BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, STATE_DIM))


@pytest.fixture()
def safe_trajectories() -> jax.Array:
    """Well-separated agents with large y offsets (50 m apart).

    Shape: ``(BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, 4)``.
    """
    dt = 0.1
    t = jnp.arange(FUTURE_STEPS) * dt
    x = jnp.broadcast_to(t * 5.0, (NUM_AGENTS, FUTURE_STEPS))
    y_offsets = jnp.arange(NUM_AGENTS)[:, None] * 50.0
    y = jnp.zeros((NUM_AGENTS, FUTURE_STEPS)) + y_offsets
    heading = jnp.zeros((NUM_AGENTS, FUTURE_STEPS))
    velocity = jnp.full((NUM_AGENTS, FUTURE_STEPS), 5.0)
    single = jnp.stack([x, y, heading, velocity], axis=-1)
    return jnp.broadcast_to(single[None], (BATCH_SIZE, NUM_AGENTS, FUTURE_STEPS, STATE_DIM))


@pytest.fixture()
def small_diffusion_model() -> TrajectoryDiffusionModel:
    """Small model for fast DPO tests."""
    config = support.make_diffusion_config(
        future_steps=FUTURE_STEPS,
        num_agents_max=NUM_AGENTS + 1,
        context_dim=CONTEXT_DIM,
        state_dim=STATE_DIM,
    )
    return support.make_diffusion_model(config, seed=0)


@pytest.fixture()
def scene_context() -> jax.Array:
    """Per-agent scene context for DPO tests. Shape: ``(NUM_AGENTS, context_dim)``.

    One row per agent: row ``i`` conditions agent ``i``.
    """
    return jnp.ones((NUM_AGENTS, CONTEXT_DIM))


@pytest.fixture()
def dpo_batch(
    straight_trajectories: jax.Array,
    collision_trajectories: jax.Array,
    scene_context: jax.Array,
) -> dict[str, jax.Array]:
    """DPO batch with chosen (safe) and rejected (collision) trajectories."""
    return {
        "chosen": straight_trajectories,
        "rejected": collision_trajectories,
        "scene_contexts": jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM)),
    }
