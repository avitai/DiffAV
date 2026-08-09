"""Shared test builders used across test packages.

Consolidates the small-model, optimizer, and trajectory builders that were
previously duplicated per package. Package-specific dimensions stay in each
package's ``helpers.py``; this module holds the parameterized construction
logic only.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

from diffav.models.factorized_backbone import FactorizedSceneBackbone
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)


def make_diffusion_config(
    *,
    future_steps: int,
    num_agents_max: int,
    context_dim: int,
    hidden_dim: int = 32,
    num_blocks: int = 1,
    num_temporal_layers: int = 1,
    num_social_layers: int = 1,
    num_heads: int = 2,
    num_timesteps: int = 10,
    state_dim: int = 4,
    **overrides: Any,
) -> TrajectoryDiffusionConfig:
    """Build a small diffusion config for fast tests.

    Args:
        future_steps: Prediction horizon steps.
        num_agents_max: Maximum agents per scene.
        context_dim: Scene context embedding dimension.
        hidden_dim: Transformer hidden dimension.
        num_blocks: Factorized backbone blocks.
        num_temporal_layers: Temporal attention sublayers per block.
        num_social_layers: Social attention sublayers per block.
        num_heads: Attention heads.
        num_timesteps: Diffusion timesteps.
        state_dim: Per-step state dimension.
        **overrides: Any further ``TrajectoryDiffusionConfig`` fields.

    Returns:
        The test configuration.
    """
    return TrajectoryDiffusionConfig(
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_temporal_layers=num_temporal_layers,
        num_social_layers=num_social_layers,
        num_heads=num_heads,
        future_steps=future_steps,
        num_agents_max=num_agents_max,
        num_timesteps=num_timesteps,
        context_dim=context_dim,
        state_dim=state_dim,
        **overrides,
    )


def make_diffusion_model(
    config: TrajectoryDiffusionConfig,
    seed: int = 0,
) -> TrajectoryDiffusionModel:
    """Build a diffusion model with a seeded parameter init.

    Args:
        config: Model configuration.
        seed: Parameter initialization seed.

    Returns:
        The initialized model.
    """
    return TrajectoryDiffusionModel(config, rngs=nnx.Rngs(params=jax.random.key(seed)))


def make_adam_optimizer(
    model: nnx.Module,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
) -> nnx.Optimizer:
    """Build the standard clipped-Adam test optimizer.

    Args:
        model: Model to optimize.
        learning_rate: Adam learning rate.
        gradient_clip: Global-norm gradient clip.

    Returns:
        The configured optimizer.
    """
    tx = create_optimizer(
        OptimizerConfig(
            optimizer_type="adam", learning_rate=learning_rate, gradient_clip=gradient_clip
        )
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def straight_line_trajectory(
    num_agents: int,
    future_steps: int,
    velocity: float = 10.0,
    dt: float = 0.1,
) -> jax.Array:
    """Straight-line constant-velocity trajectories along the x-axis.

    State = [x, y, heading, velocity].

    Args:
        num_agents: Number of agents.
        future_steps: Prediction horizon steps.
        velocity: Constant velocity in m/s.
        dt: Simulation timestep in seconds.

    Returns:
        Trajectories, shape ``(num_agents, future_steps, 4)``.
    """
    t = jnp.arange(future_steps, dtype=jnp.float32)
    x = velocity * dt * t
    y = jnp.zeros(future_steps)
    heading = jnp.zeros(future_steps)
    vel = jnp.full(future_steps, velocity)
    single = jnp.stack([x, y, heading, vel], axis=-1)
    return jnp.broadcast_to(single[None, :, :], (num_agents, future_steps, 4))


def seeded_normal(*shape: int, seed: int = 0) -> np.ndarray:
    """Reproducible standard-normal samples (replaces bare ``np.random.randn``).

    Args:
        *shape: Output array shape.
        seed: Generator seed.

    Returns:
        Float64 samples of the requested shape.
    """
    return np.random.default_rng(seed).standard_normal(shape)


def randomize_adaln(backbone: FactorizedSceneBackbone, *, seed: int) -> None:
    """Fill a factorized backbone's adaLN projections with signal, in place.

    adaLN is zero-initialised (DiT convention), so an untrained backbone is
    the identity in its conditioning: context, timestep, and map cross-
    attention have no effect until training learns non-zero adaLN weights.
    Filling every projection exercises the conditioning pathway, as a trained
    model would.

    Args:
        backbone: The factorized scene backbone to perturb in place.
        seed: Seed for the random adaLN weights.
    """
    linears = [block.adaln for block in backbone.blocks]
    linears.append(backbone.output_adaln)
    key = jax.random.key(seed + 100)
    for index, linear in enumerate(linears):
        shape = linear.kernel[...].shape
        linear.kernel[...] = 0.5 * jax.random.normal(jax.random.fold_in(key, index), shape)
