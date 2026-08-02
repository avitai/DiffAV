"""Shared fixtures for physics tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tests import support


# Shared test constants
NUM_AGENTS = 3
FUTURE_STEPS = 8
STATE_DIM = 4


def straight_line_trajectory(
    num_agents: int = NUM_AGENTS,
    future_steps: int = FUTURE_STEPS,
    velocity: float = 10.0,
    dt: float = 0.1,
) -> jax.Array:
    """Build a straight-line constant-velocity trajectory along the x-axis.

    State = [x, y, heading, velocity].

    Args:
        num_agents: Number of agents.
        future_steps: Prediction horizon steps.
        velocity: Constant velocity.
        dt: Simulation timestep.

    Returns:
        Trajectories array, shape ``(num_agents, future_steps, 4)``.
    """
    return support.straight_line_trajectory(num_agents, future_steps, velocity, dt)


def stationary_trajectory(
    num_agents: int = NUM_AGENTS,
    future_steps: int = FUTURE_STEPS,
) -> jax.Array:
    """Build a stationary trajectory with zero velocity and heading.

    Args:
        num_agents: Number of agents.
        future_steps: Prediction horizon steps.

    Returns:
        Zero trajectories array, shape ``(num_agents, future_steps, 4)``.
    """
    return jnp.zeros((num_agents, future_steps, STATE_DIM))
