"""Safety reward functions for preference-based trajectory alignment.

Provides domain-specific reward functions that satisfy the artifex
:class:`~artifex.generative_models.training.rl.rewards.RewardFunction`
protocol. Each reward quantifies a safety aspect of generated multi-agent
trajectories, returning negative penalties (higher = safer) following
standard RL convention.

The composite :class:`SafetyReward` wraps individual rewards via artifex's
:class:`~artifex.generative_models.training.rl.rewards.CompositeReward`
for weighted combination used by the preference pair builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from artifex.generative_models.training.rl.rewards import CompositeReward, RewardFunction

from diffav.core.constants import WOD_STEP_DURATION_SECONDS
from diffav.core.geometry import mean_offroad_penalty
from diffav.physics.kinematics import BicycleModelConfig, BicycleModelConstraint


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetyRewardConfig:
    """Configuration for the composite safety reward.

    Attributes:
        collision_threshold: Minimum safe distance between agents (metres).
        collision_weight: Weight for collision penalty reward.
        kinematic_weight: Weight for kinematic residual reward.
        boundary_weight: Weight for road boundary violation reward.
        comfort_weight: Weight for jerk minimisation reward.
        bicycle_config: Optional bicycle model configuration.
        dt: Simulation timestep in seconds.
    """

    collision_threshold: float = 2.0
    collision_weight: float = 1.0
    kinematic_weight: float = 1.0
    boundary_weight: float = 1.0
    comfort_weight: float = 0.5
    bicycle_config: BicycleModelConfig | None = None
    dt: float = WOD_STEP_DURATION_SECONDS

    def __post_init__(self) -> None:
        """Validate that weights are non-negative."""
        for name in ("collision_weight", "kinematic_weight", "boundary_weight", "comfort_weight"):
            value = getattr(self, name)
            if value < 0:
                msg = f"{name} must be non-negative, got {value}"
                raise ValueError(msg)


def _compute_collision_penalty(
    trajectories: jax.Array,
    threshold: float,
) -> jax.Array:
    """Compute pairwise collision penalty for a single scene.

    Args:
        trajectories: Shape ``(num_agents, future_steps, 4)``.
        threshold: Minimum safe distance (metres).

    Returns:
        Scalar collision penalty (non-negative).
    """
    positions = trajectories[:, :, :2]
    num_agents = positions.shape[0]

    # Pairwise distances: (A, A, T)
    diff = positions[:, None, :, :] - positions[None, :, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

    # Soft penalty: max(0, threshold - dist)^2
    penalty = jnp.maximum(0.0, threshold - dist) ** 2

    # Zero out self-pairs
    mask = 1.0 - jnp.eye(num_agents)[:, :, None]
    penalty = penalty * mask

    # Mean over all pairs and timesteps
    return jnp.sum(penalty) / jnp.maximum(num_agents * (num_agents - 1) * positions.shape[1], 1.0)


class CollisionReward:
    """Pairwise agent distance penalty, negated (higher = safer).

    Input shape: ``(batch, num_agents, future_steps, 4)``.
    Output shape: ``(batch,)``.

    Math: ``-mean(max(0, threshold - dist)^2)`` per sample.
    """

    def __init__(self, threshold: float = 2.0) -> None:
        """Initialize with collision distance threshold.

        Args:
            threshold: Minimum safe distance between agents (metres).
        """
        self.threshold = threshold

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute collision reward for a batch of trajectories.

        Args:
            samples: Trajectories ``(batch, num_agents, future_steps, 4)``.
            conditions: Unused, present for protocol compatibility.
            **kwargs: Unused.

        Returns:
            Collision rewards ``(batch,)``, non-positive.
        """
        del conditions, kwargs
        return -jax.vmap(lambda traj: _compute_collision_penalty(traj, self.threshold))(samples)


class KinematicReward:
    """Bicycle model kinematic residual, negated (higher = more plausible).

    Wraps :meth:`BicycleModelConstraint.compute_residuals` via ``jax.vmap``
    over the batch dimension.

    Input shape: ``(batch, num_agents, future_steps, 4)``.
    Output shape: ``(batch,)``.
    """

    def __init__(self, config: BicycleModelConfig | None = None) -> None:
        """Initialize with optional bicycle model configuration.

        Args:
            config: Bicycle model parameters. Uses defaults if ``None``.
        """
        self._constraint = BicycleModelConstraint(config)

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute kinematic reward for a batch of trajectories.

        Args:
            samples: Trajectories ``(batch, num_agents, future_steps, 4)``.
            conditions: Unused, present for protocol compatibility.
            **kwargs: Unused.

        Returns:
            Kinematic rewards ``(batch,)``, non-positive.
        """
        del conditions, kwargs
        return -jax.vmap(self._constraint.compute_residuals)(samples)


class BoundaryReward:
    """Off-road violation penalty, negated (higher = safer).

    Hinges on the signed distance to oriented road edges: on-road agents
    receive zero reward, off-road agents are penalized by their squared
    distance to the boundary. Expects ``road_edges`` as a keyword argument
    holding a :class:`~diffav.core.geometry.RoadEdges`; returns zero
    when no road edges are provided.

    Input shape: ``(batch, num_agents, future_steps, 4)``.
    Output shape: ``(batch,)``.
    """

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute boundary reward for a batch of trajectories.

        Args:
            samples: Trajectories ``(batch, num_agents, future_steps, 4)``.
            conditions: Unused, present for protocol compatibility.
            **kwargs: Must include ``road_edges`` (a :class:`RoadEdges`)
                or omit for zero.

        Returns:
            Boundary rewards ``(batch,)``, non-positive.
        """
        del conditions
        road_edges = kwargs.get("road_edges")
        if road_edges is None:
            return jnp.zeros(samples.shape[0])
        return -jax.vmap(lambda traj: mean_offroad_penalty(traj[:, :, :2], road_edges))(samples)


class ComfortReward:
    """Jerk minimisation reward (higher = smoother ride).

    Computes jerk as the second finite difference of velocity divided by
    ``dt^2``, returning ``-mean(|jerk|)`` per sample.

    Input shape: ``(batch, num_agents, future_steps, 4)``.
    Output shape: ``(batch,)``.
    """

    def __init__(self, dt: float = WOD_STEP_DURATION_SECONDS) -> None:
        """Initialize with simulation timestep.

        Args:
            dt: Simulation timestep in seconds.
        """
        self.dt = dt

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute comfort reward for a batch of trajectories.

        Args:
            samples: Trajectories ``(batch, num_agents, future_steps, 4)``.
            conditions: Unused, present for protocol compatibility.
            **kwargs: Unused.

        Returns:
            Comfort rewards ``(batch,)``, non-positive.
        """
        del conditions, kwargs
        return jax.vmap(self._compute_single)(samples)

    def _compute_single(self, trajectories: jax.Array) -> jax.Array:
        """Compute comfort reward for a single scene.

        Args:
            trajectories: Shape ``(num_agents, future_steps, 4)``.

        Returns:
            Scalar comfort reward.
        """
        velocity = trajectories[:, :, 3]
        # Second finite difference of velocity = jerk * dt^2
        jerk = jnp.diff(velocity, n=2, axis=-1) / (self.dt**2)
        return -jnp.mean(jnp.abs(jerk))


_RewardComponent: TypeAlias = CollisionReward | KinematicReward | BoundaryReward | ComfortReward


class SafetyReward:
    """Composite safety reward combining collision, kinematic, boundary, and comfort.

    Wraps individual reward functions via artifex
    :class:`~artifex.generative_models.training.rl.rewards.CompositeReward`
    with configurable weights.
    """

    def __init__(self, config: SafetyRewardConfig | None = None) -> None:
        """Initialize with optional configuration.

        Args:
            config: Safety reward configuration. Uses defaults if ``None``.
        """
        self.config = config or SafetyRewardConfig()

        bicycle_cfg = self.config.bicycle_config
        self._components: dict[str, _RewardComponent] = {
            "collision": CollisionReward(threshold=self.config.collision_threshold),
            "kinematic": KinematicReward(config=bicycle_cfg),
            "boundary": BoundaryReward(),
            "comfort": ComfortReward(dt=self.config.dt),
        }

        reward_fns: list[RewardFunction] = list(self._components.values())
        weights = [
            self.config.collision_weight,
            self.config.kinematic_weight,
            self.config.boundary_weight,
            self.config.comfort_weight,
        ]
        self._composite = CompositeReward(reward_fns=reward_fns, weights=weights)

    @property
    def component_rewards(self) -> dict[str, _RewardComponent]:
        """Access individual reward components for diagnostics.

        Returns:
            Mapping from component name to reward function.
        """
        return dict(self._components)

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute composite safety reward for a batch of trajectories.

        Args:
            samples: Trajectories ``(batch, num_agents, future_steps, 4)``.
            conditions: Unused, present for protocol compatibility.
            **kwargs: Passed to all component rewards (e.g. ``road_edges``).

        Returns:
            Combined safety rewards ``(batch,)``.
        """
        return self._composite(samples, conditions, **kwargs)
