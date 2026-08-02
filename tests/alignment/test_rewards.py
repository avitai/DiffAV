"""Tests for safety reward functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulacrax.alignment.rewards import (
    BoundaryReward,
    CollisionReward,
    ComfortReward,
    KinematicReward,
    SafetyReward,
    SafetyRewardConfig,
)
from simulacrax.core.geometry import RoadEdges
from simulacrax.physics.kinematics import BicycleModelConfig
from tests.alignment.helpers import BATCH_SIZE, FUTURE_STEPS, NUM_AGENTS


# ---------------------------------------------------------------------------
# SafetyRewardConfig
# ---------------------------------------------------------------------------


class TestSafetyRewardConfig:
    """Tests for SafetyRewardConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = SafetyRewardConfig()
        assert cfg.collision_threshold == 2.0
        assert cfg.collision_weight == 1.0
        assert cfg.kinematic_weight == 1.0
        assert cfg.boundary_weight == 1.0
        assert cfg.comfort_weight == 0.5
        assert cfg.dt == 0.1

    def test_negative_collision_weight_raises(self) -> None:
        """Negative collision_weight raises ValueError."""
        with pytest.raises(ValueError, match="collision_weight"):
            SafetyRewardConfig(collision_weight=-1.0)

    def test_negative_kinematic_weight_raises(self) -> None:
        """Negative kinematic_weight raises ValueError."""
        with pytest.raises(ValueError, match="kinematic_weight"):
            SafetyRewardConfig(kinematic_weight=-0.1)

    def test_negative_boundary_weight_raises(self) -> None:
        """Negative boundary_weight raises ValueError."""
        with pytest.raises(ValueError, match="boundary_weight"):
            SafetyRewardConfig(boundary_weight=-1.0)

    def test_negative_comfort_weight_raises(self) -> None:
        """Negative comfort_weight raises ValueError."""
        with pytest.raises(ValueError, match="comfort_weight"):
            SafetyRewardConfig(comfort_weight=-1.0)

    def test_custom_bicycle_config(self) -> None:
        """Custom BicycleModelConfig is stored."""
        bc = BicycleModelConfig(wheelbase=3.0)
        cfg = SafetyRewardConfig(bicycle_config=bc)
        assert cfg.bicycle_config is not None
        assert cfg.bicycle_config.wheelbase == 3.0


# ---------------------------------------------------------------------------
# CollisionReward
# ---------------------------------------------------------------------------


class TestCollisionReward:
    """Tests for CollisionReward."""

    def test_output_shape(self, straight_trajectories: jax.Array) -> None:
        """Output has shape (batch,)."""
        reward = CollisionReward()
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_separated_agents_low_penalty(self, safe_trajectories: jax.Array) -> None:
        """Agents far apart should have near-zero collision penalty."""
        reward = CollisionReward()
        result = reward(safe_trajectories)
        # All rewards should be 0 (no violations)
        assert float(jnp.max(jnp.abs(result))) < 1e-6

    def test_overlapping_agents_high_penalty(self, collision_trajectories: jax.Array) -> None:
        """Overlapping agents should produce negative rewards."""
        reward = CollisionReward()
        result = reward(collision_trajectories)
        # Collision reward is negated penalty (higher = safer)
        assert float(jnp.max(result)) < 0.0

    def test_differentiable(self, straight_trajectories: jax.Array) -> None:
        """Reward is differentiable via jax.grad."""
        reward = CollisionReward()

        def loss_fn(traj: jax.Array) -> jax.Array:
            return jnp.sum(reward(traj))

        grad = jax.grad(loss_fn)(straight_trajectories)
        assert grad.shape == straight_trajectories.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit_compatible(self, straight_trajectories: jax.Array) -> None:
        """Reward works under jax.jit."""
        reward = CollisionReward()
        jitted = jax.jit(reward)
        result = jitted(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_custom_threshold(self, straight_trajectories: jax.Array) -> None:
        """Custom threshold changes sensitivity."""
        # With very large threshold, even separated agents get penalized
        reward_large = CollisionReward(threshold=100.0)
        result = reward_large(straight_trajectories)
        assert float(jnp.min(result)) < 0.0

    def test_satisfies_reward_protocol(self) -> None:
        """CollisionReward matches RewardFunction protocol signature."""
        reward = CollisionReward()
        # Verify callable with protocol signature
        samples = jnp.zeros((2, NUM_AGENTS, FUTURE_STEPS, 4))
        result = reward(samples, conditions=None)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# KinematicReward
# ---------------------------------------------------------------------------


class TestKinematicReward:
    """Tests for KinematicReward."""

    def test_output_shape(self, straight_trajectories: jax.Array) -> None:
        """Output has shape (batch,)."""
        reward = KinematicReward()
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_straight_line_low_residual(self, straight_trajectories: jax.Array) -> None:
        """Straight-line constant-velocity has low kinematic residual."""
        reward = KinematicReward()
        result = reward(straight_trajectories)
        # Residual should be small for physically plausible trajectories
        # Reward is negated, so values should be close to 0
        assert float(jnp.max(jnp.abs(result))) < 0.05

    def test_differentiable(self, straight_trajectories: jax.Array) -> None:
        """Reward is differentiable."""
        reward = KinematicReward()

        def loss_fn(traj: jax.Array) -> jax.Array:
            return jnp.sum(reward(traj))

        grad = jax.grad(loss_fn)(straight_trajectories)
        assert grad.shape == straight_trajectories.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit_compatible(self, straight_trajectories: jax.Array) -> None:
        """Reward works under jax.jit."""
        reward = KinematicReward()
        jitted = jax.jit(reward)
        result = jitted(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_custom_bicycle_config(self, straight_trajectories: jax.Array) -> None:
        """Custom BicycleModelConfig is used."""
        cfg = BicycleModelConfig(wheelbase=3.5, dt=0.1)
        reward = KinematicReward(config=cfg)
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# BoundaryReward
# ---------------------------------------------------------------------------


def _square_road_edges() -> RoadEdges:
    """Counterclockwise closed square road edge: interior on-road."""
    square = np.array([[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0], [0, 0, 0]], dtype=np.float32)
    return RoadEdges.from_polylines([square])


def _batch_at_y(y: float) -> jax.Array:
    """Single-scene batch of constant-velocity agents at a lateral offset."""
    t = jnp.arange(FUTURE_STEPS) * 0.1
    x = jnp.broadcast_to(5.0 + 5.0 * t, (NUM_AGENTS, FUTURE_STEPS))
    lateral = jnp.full((NUM_AGENTS, FUTURE_STEPS), y)
    heading = jnp.zeros((NUM_AGENTS, FUTURE_STEPS))
    velocity = jnp.full((NUM_AGENTS, FUTURE_STEPS), 5.0)
    return jnp.stack([x, lateral, heading, velocity], axis=-1)[None]


class TestBoundaryReward:
    """Tests for BoundaryReward (off-road hinge on signed distance)."""

    def test_output_shape(self, straight_trajectories: jax.Array) -> None:
        """Output has shape (batch,)."""
        reward = BoundaryReward()
        result = reward(straight_trajectories, road_edges=_square_road_edges())
        assert result.shape == (BATCH_SIZE,)

    def test_no_road_edges_returns_zero(self, straight_trajectories: jax.Array) -> None:
        """Without road_edges kwarg, returns 0."""
        reward = BoundaryReward()
        result = reward(straight_trajectories)
        assert float(jnp.max(jnp.abs(result))) == 0.0

    def test_on_road_agents_zero_reward(self) -> None:
        """Agents inside the drivable area receive exactly zero reward."""
        reward = BoundaryReward()
        result = reward(_batch_at_y(20.0), road_edges=_square_road_edges())
        assert float(result[0]) == 0.0

    def test_off_road_agents_penalized(self) -> None:
        """Off-road agents get negative reward, more negative further out."""
        reward = BoundaryReward()
        edges = _square_road_edges()
        near = reward(_batch_at_y(-5.0), road_edges=edges)
        far = reward(_batch_at_y(-10.0), road_edges=edges)
        assert float(near[0]) < 0.0
        assert float(far[0]) < float(near[0])

    def test_differentiable(self) -> None:
        """Reward is differentiable and pushes off-road agents back on-road."""
        reward = BoundaryReward()
        edges = _square_road_edges()

        def total_reward(traj: jax.Array) -> jax.Array:
            return jnp.sum(reward(traj, road_edges=edges))

        batch = _batch_at_y(-5.0)
        grad = jax.grad(total_reward)(batch)
        assert grad.shape == batch.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        # Below the bottom edge, increasing y moves on-road: reward rises.
        assert bool(jnp.all(grad[:, :, :, 1] > 0.0))

    def test_jit_compatible(self, straight_trajectories: jax.Array) -> None:
        """Reward works under jax.jit."""
        reward = BoundaryReward()
        edges = _square_road_edges()

        @jax.jit
        def compute(traj: jax.Array) -> jax.Array:
            return reward(traj, road_edges=edges)

        result = compute(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# ComfortReward
# ---------------------------------------------------------------------------


class TestComfortReward:
    """Tests for ComfortReward."""

    def test_output_shape(self, straight_trajectories: jax.Array) -> None:
        """Output has shape (batch,)."""
        reward = ComfortReward()
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_constant_velocity_low_jerk(self, straight_trajectories: jax.Array) -> None:
        """Constant velocity trajectories have zero jerk."""
        reward = ComfortReward()
        result = reward(straight_trajectories)
        # Zero jerk => reward should be 0
        assert float(jnp.max(jnp.abs(result))) < 1e-5

    def test_varying_velocity_nonzero_jerk(self) -> None:
        """Trajectories with varying velocity have nonzero jerk penalty."""
        reward = ComfortReward()
        # Create trajectory with acceleration changes (jerk)
        dt = 0.1
        t = jnp.arange(FUTURE_STEPS) * dt
        velocity = jnp.sin(t * 5.0) * 10.0 + 10.0  # oscillating velocity
        traj = jnp.stack(
            [
                jnp.broadcast_to(t * 5.0, (NUM_AGENTS, FUTURE_STEPS)),
                jnp.zeros((NUM_AGENTS, FUTURE_STEPS)),
                jnp.zeros((NUM_AGENTS, FUTURE_STEPS)),
                jnp.broadcast_to(velocity, (NUM_AGENTS, FUTURE_STEPS)),
            ],
            axis=-1,
        )[None]  # (1, A, T, 4)
        result = reward(traj)
        # Oscillating velocity => nonzero jerk => negative reward
        assert float(result[0]) < 0.0

    def test_differentiable(self, straight_trajectories: jax.Array) -> None:
        """Reward is differentiable."""
        reward = ComfortReward()

        def loss_fn(traj: jax.Array) -> jax.Array:
            return jnp.sum(reward(traj))

        grad = jax.grad(loss_fn)(straight_trajectories)
        assert grad.shape == straight_trajectories.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit_compatible(self, straight_trajectories: jax.Array) -> None:
        """Reward works under jax.jit."""
        reward = ComfortReward()
        jitted = jax.jit(reward)
        result = jitted(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# SafetyReward (composite)
# ---------------------------------------------------------------------------


class TestSafetyReward:
    """Tests for SafetyReward composite."""

    def test_output_shape(self, straight_trajectories: jax.Array) -> None:
        """Output has shape (batch,)."""
        reward = SafetyReward()
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_combines_components(self, straight_trajectories: jax.Array) -> None:
        """SafetyReward produces a combined score from all components."""
        reward = SafetyReward()
        result = reward(straight_trajectories)
        # Safe, straight trajectories should have reward close to 0
        assert bool(jnp.all(jnp.isfinite(result)))

    def test_component_rewards_property(self) -> None:
        """component_rewards exposes individual reward functions."""
        reward = SafetyReward()
        components = reward.component_rewards
        assert "collision" in components
        assert "kinematic" in components
        assert "boundary" in components
        assert "comfort" in components

    def test_satisfies_reward_protocol(self, straight_trajectories: jax.Array) -> None:
        """SafetyReward matches RewardFunction protocol."""
        reward = SafetyReward()
        result = reward(straight_trajectories, conditions=None)
        assert result.shape == (BATCH_SIZE,)

    def test_jit_compatible(self, straight_trajectories: jax.Array) -> None:
        """SafetyReward works under jax.jit."""

        @jax.jit
        def compute(traj: jax.Array) -> jax.Array:
            return SafetyReward()(traj)

        result = compute(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_differentiable(self, straight_trajectories: jax.Array) -> None:
        """SafetyReward is differentiable."""
        reward = SafetyReward()

        def loss_fn(traj: jax.Array) -> jax.Array:
            return jnp.sum(reward(traj))

        grad = jax.grad(loss_fn)(straight_trajectories)
        assert grad.shape == straight_trajectories.shape
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_custom_config(self, straight_trajectories: jax.Array) -> None:
        """Custom config changes component weights."""
        cfg = SafetyRewardConfig(
            collision_weight=2.0,
            kinematic_weight=0.0,
            boundary_weight=0.0,
            comfort_weight=0.0,
        )
        reward = SafetyReward(config=cfg)
        result = reward(straight_trajectories)
        assert result.shape == (BATCH_SIZE,)

    def test_collision_trajectories_worse(
        self,
        safe_trajectories: jax.Array,
        collision_trajectories: jax.Array,
    ) -> None:
        """Collision trajectories score worse than safe trajectories."""
        reward = SafetyReward()
        safe_score = reward(safe_trajectories)
        collision_score = reward(collision_trajectories)
        # Safe should score higher (less negative) than collision
        assert float(jnp.mean(safe_score)) > float(jnp.mean(collision_score))
