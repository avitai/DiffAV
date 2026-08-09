"""Tests for driving-domain physics losses."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffav.core.geometry import RoadEdges
from diffav.physics.losses import DiffAVPhysicsConfig, DiffAVPhysicsLoss
from tests.physics.helpers import FUTURE_STEPS as _FUTURE_STEPS, STATE_DIM as _STATE_DIM


_NUM_AGENTS = 4

# Counterclockwise closed square road edge: interior on-road.
_ROAD_EDGE_SQUARE = np.array(
    [[0, 0, 0], [40, 0, 0], [40, 40, 0], [0, 40, 0], [0, 0, 0]], dtype=np.float32
)


def _square_road_edges() -> RoadEdges:
    return RoadEdges.from_polylines([_ROAD_EDGE_SQUARE])


def _trajectories_at_y(y: float, num_agents: int = 2, future_steps: int = 5) -> jax.Array:
    """Constant-velocity trajectories at a fixed lateral offset."""
    t = jnp.arange(future_steps) * 0.1
    x = jnp.broadcast_to(5.0 + 5.0 * t, (num_agents, future_steps))
    lateral = jnp.full((num_agents, future_steps), y)
    heading = jnp.zeros((num_agents, future_steps))
    velocity = jnp.full((num_agents, future_steps), 5.0)
    return jnp.stack([x, lateral, heading, velocity], axis=-1)


def _straight_trajectories(
    num_agents: int = _NUM_AGENTS,
    future_steps: int = _FUTURE_STEPS,
    velocity: float = 5.0,
    dt: float = 0.1,
) -> jax.Array:
    """Create straight-line constant-velocity trajectories with y-offsets."""
    t = jnp.arange(future_steps) * dt
    x = jnp.broadcast_to(t[None, :], (num_agents, future_steps))
    y_offsets = jnp.arange(num_agents)[:, None] * 10.0
    y = jnp.broadcast_to(jnp.zeros(future_steps)[None, :], (num_agents, future_steps))
    y = y + y_offsets
    heading = jnp.zeros((num_agents, future_steps))
    vel = jnp.full((num_agents, future_steps), velocity)
    return jnp.stack([x, y, heading, vel], axis=-1)


def _separated_trajectories() -> jax.Array:
    """Create agents far apart from each other."""
    return _straight_trajectories(num_agents=3, velocity=5.0)


def _overlapping_trajectories() -> jax.Array:
    """Create agents at the same position (collision)."""
    t = jnp.arange(_FUTURE_STEPS) * 0.1
    traj_single = jnp.stack(
        [
            t,
            jnp.zeros(_FUTURE_STEPS),
            jnp.zeros(_FUTURE_STEPS),
            jnp.full(_FUTURE_STEPS, 5.0),
        ],
        axis=-1,
    )
    return jnp.broadcast_to(traj_single[None, :, :], (3, _FUTURE_STEPS, _STATE_DIM))


class TestDiffAVPhysicsConfig:
    """Tests for DiffAVPhysicsConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = DiffAVPhysicsConfig()
        assert cfg.kinematic_weight == 1.0
        assert cfg.collision_threshold == 2.0
        assert cfg.adaptive_weighting is True
        assert cfg.schedule_type == "exponential"
        assert cfg.transition_epochs == 100

    def test_negative_kinematic_weight_raises(self) -> None:
        """Negative kinematic weight raises ValueError."""
        with pytest.raises(ValueError, match="kinematic_weight"):
            DiffAVPhysicsConfig(kinematic_weight=-1.0)

    def test_negative_collision_weight_raises(self) -> None:
        """Negative collision weight raises ValueError."""
        with pytest.raises(ValueError, match="collision_weight"):
            DiffAVPhysicsConfig(collision_weight=-0.5)

    def test_negative_road_boundary_weight_raises(self) -> None:
        """Negative road boundary weight raises ValueError."""
        with pytest.raises(ValueError, match="road_boundary_weight"):
            DiffAVPhysicsConfig(road_boundary_weight=-1.0)

    def test_invalid_schedule_type_raises(self) -> None:
        """Invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid PhysicsScheduleType"):
            DiffAVPhysicsConfig(schedule_type=cast(Any, "cosine"))

    def test_non_positive_offroad_chunk_size_raises(self) -> None:
        """A zero or negative offroad_chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="offroad_chunk_size"):
            DiffAVPhysicsConfig(offroad_chunk_size=0)

    def test_zero_collision_threshold_raises(self) -> None:
        """Zero collision threshold raises ValueError."""
        with pytest.raises(ValueError, match="collision_threshold"):
            DiffAVPhysicsConfig(collision_threshold=0.0)

    def test_negative_momentum_weight_raises(self) -> None:
        """Negative momentum-variation weight raises ValueError."""
        with pytest.raises(ValueError, match="momentum_variation_weight"):
            DiffAVPhysicsConfig(momentum_variation_weight=-0.1)


class TestDiffAVPhysicsLoss:
    """Tests for DiffAVPhysicsLoss compute_loss."""

    def test_returns_scalar_and_dict(self) -> None:
        """compute_loss returns (scalar, dict)."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()
        total, components = loss_fn.compute_loss(traj, epoch=0)
        assert total.shape == ()
        assert isinstance(components, dict)
        assert "kinematic_loss" in components
        assert "collision_loss" in components
        assert "boundary_loss" in components
        assert "momentum_loss" in components
        assert "total_loss" in components

    def test_loss_nonnegative(self) -> None:
        """Total loss is non-negative."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()
        total, _ = loss_fn.compute_loss(traj, epoch=0)
        assert float(total) >= 0.0

    def test_differentiable(self) -> None:
        """Loss is differentiable with respect to trajectories."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()

        def scalar_loss(t: jax.Array) -> jax.Array:
            total, _ = loss_fn.compute_loss(t, epoch=0)
            return total

        grad = jax.grad(scalar_loss)(traj)
        assert grad.shape == traj.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_jit_compatible(self) -> None:
        """Loss computes correctly under JIT."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()

        @jax.jit
        def jitted_loss(t: jax.Array) -> jax.Array:
            total, _ = loss_fn.compute_loss(t, epoch=0)
            return total

        eager = loss_fn.compute_loss(traj, epoch=0)[0]
        jitted = jitted_loss(traj)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_collision_separated_agents_zero(self) -> None:
        """Well-separated agents have zero collision loss."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _separated_trajectories()
        _, components = loss_fn.compute_loss(traj, epoch=0)
        assert float(components["collision_loss"]) == pytest.approx(0.0, abs=1e-6)

    def test_collision_overlapping_agents_positive(self) -> None:
        """Overlapping agents produce positive collision loss."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _overlapping_trajectories()
        _, components = loss_fn.compute_loss(traj, epoch=0)
        assert float(components["collision_loss"]) > 0.0

    def test_adaptive_weight_changes_with_epoch(self) -> None:
        """Adaptive weight increases from early to late epochs."""
        loss_fn = DiffAVPhysicsLoss()
        weight_early = float(loss_fn.get_current_weight(0))
        weight_late = float(loss_fn.get_current_weight(1000))
        assert weight_late > weight_early

    def test_no_road_edges_zero_boundary_loss(self) -> None:
        """No road edges → zero boundary loss."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()
        _, components = loss_fn.compute_loss(traj, epoch=0, road_edges=None)
        assert float(components["boundary_loss"]) == pytest.approx(0.0)

    def test_on_road_zero_boundary_loss(self) -> None:
        """Agents inside the drivable area incur no boundary loss."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _trajectories_at_y(20.0)
        _, components = loss_fn.compute_loss(traj, epoch=0, road_edges=_square_road_edges())
        assert float(components["boundary_loss"]) == pytest.approx(0.0)

    def test_off_road_boundary_loss_is_squared_distance(self) -> None:
        """Off-road agents are penalized by squared distance to the boundary."""
        loss_fn = DiffAVPhysicsLoss()
        # 5 m below the bottom road edge → signed distance +5 → penalty 25.
        traj = _trajectories_at_y(-5.0)
        _, components = loss_fn.compute_loss(traj, epoch=0, road_edges=_square_road_edges())
        assert float(components["boundary_loss"]) == pytest.approx(25.0, rel=1e-4)

    def test_boundary_loss_grows_with_off_road_distance(self) -> None:
        """Driving further off-road increases the boundary loss."""
        loss_fn = DiffAVPhysicsLoss()
        edges = _square_road_edges()
        _, near = loss_fn.compute_loss(_trajectories_at_y(-5.0), epoch=0, road_edges=edges)
        _, far = loss_fn.compute_loss(_trajectories_at_y(-10.0), epoch=0, road_edges=edges)
        assert float(far["boundary_loss"]) > float(near["boundary_loss"])

    def test_boundary_gradient_points_back_toward_road(self) -> None:
        """For off-road agents, moving back on-road reduces the boundary loss."""
        loss_fn = DiffAVPhysicsLoss()
        edges = _square_road_edges()

        def boundary_loss(traj: jax.Array) -> jax.Array:
            _, components = loss_fn.compute_loss(traj, epoch=0, road_edges=edges)
            return components["boundary_loss"]

        grads = jax.grad(boundary_loss)(_trajectories_at_y(-5.0))
        assert bool(jnp.all(jnp.isfinite(grads)))
        # Below the bottom edge, increasing y moves on-road: negative y-gradient.
        assert bool(jnp.all(grads[:, :, 1] < 0.0))

    def test_momentum_variation_in_components(self) -> None:
        """Momentum loss is present in component dict."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories()
        _, components = loss_fn.compute_loss(traj, epoch=0)
        assert "momentum_loss" in components

    def test_default_config(self) -> None:
        """DiffAVPhysicsLoss() works without arguments."""
        loss_fn = DiffAVPhysicsLoss()
        assert loss_fn.config.kinematic_weight == 1.0

    def test_disabled_adaptive_weighting(self) -> None:
        """Disabled adaptive weighting returns constant weight."""
        cfg = DiffAVPhysicsConfig(adaptive_weighting=False)
        loss_fn = DiffAVPhysicsLoss(cfg)
        w0 = float(loss_fn.get_current_weight(0))
        w100 = float(loss_fn.get_current_weight(100))
        assert w0 == pytest.approx(w100)

    def test_single_agent_no_collision(self) -> None:
        """Single agent produces zero collision loss."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories(num_agents=1)
        _, components = loss_fn.compute_loss(traj, epoch=0)
        assert float(components["collision_loss"]) == pytest.approx(0.0)


class TestValidityAwarePhysics:
    """The physics loss excludes padded (invalid) agents from every term."""

    def test_valid_mask_excludes_padded_agent_from_collision(self) -> None:
        """A padded slot sitting on a real agent collides only when unmasked."""
        loss_fn = DiffAVPhysicsLoss()
        two = _straight_trajectories(num_agents=2, future_steps=6)
        overlap = two[:1]  # a duplicate sitting exactly on agent 0
        three = jnp.concatenate([two, overlap], axis=0)

        _, unmasked = loss_fn.compute_loss(three, epoch=0, valid_mask=jnp.ones((3, 6), dtype=bool))
        valid = jnp.array([[True] * 6, [True] * 6, [False] * 6])
        _, masked = loss_fn.compute_loss(three, epoch=0, valid_mask=valid)
        assert float(unmasked["collision_loss"]) > 0.0
        assert float(masked["collision_loss"]) == pytest.approx(0.0, abs=1e-6)

    def test_valid_mask_excludes_padded_agent_from_kinematic(self) -> None:
        """A teleporting padded agent inflates the kinematic residual unless masked."""
        loss_fn = DiffAVPhysicsLoss()
        good = _straight_trajectories(num_agents=1, future_steps=6)
        bad = (
            jnp.zeros((1, 6, _STATE_DIM))
            .at[0, :, 0]
            .set(jnp.array([0.0, 50.0, 0.0, 50.0, 0.0, 50.0]))
        )
        pair = jnp.concatenate([good, bad], axis=0)

        _, unmasked = loss_fn.compute_loss(pair, epoch=0, valid_mask=jnp.ones((2, 6), dtype=bool))
        valid = jnp.array([[True] * 6, [False] * 6])
        _, masked = loss_fn.compute_loss(pair, epoch=0, valid_mask=valid)
        _, only_good = loss_fn.compute_loss(good, epoch=0)
        # Masking the padded agent recovers the single-agent residual, and the
        # padded agent inflates the residual when it is counted.
        assert float(masked["kinematic_loss"]) == pytest.approx(
            float(only_good["kinematic_loss"]), rel=1e-4
        )
        assert float(unmasked["kinematic_loss"]) > float(masked["kinematic_loss"])

    def test_valid_mask_excludes_padded_agent_from_boundary(self) -> None:
        """An off-road padded agent adds boundary loss only when unmasked."""
        loss_fn = DiffAVPhysicsLoss()
        on_road = _trajectories_at_y(10.0, num_agents=2, future_steps=6)
        off_road = jnp.full((1, 6, _STATE_DIM), 100.0)  # far outside the square
        three = jnp.concatenate([on_road, off_road], axis=0)
        edges = _square_road_edges()

        _, unmasked = loss_fn.compute_loss(
            three, epoch=0, road_edges=edges, valid_mask=jnp.ones((3, 6), dtype=bool)
        )
        valid = jnp.array([[True] * 6, [True] * 6, [False] * 6])
        _, masked = loss_fn.compute_loss(three, epoch=0, road_edges=edges, valid_mask=valid)
        assert float(unmasked["boundary_loss"]) > 0.0
        assert float(masked["boundary_loss"]) == pytest.approx(0.0, abs=1e-6)

    def test_none_mask_matches_unmasked(self) -> None:
        """valid_mask=None reproduces the unmasked loss exactly."""
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories(num_agents=3, future_steps=6)
        edges = _square_road_edges()
        total_default, _ = loss_fn.compute_loss(traj, epoch=0, road_edges=edges)
        total_none, _ = loss_fn.compute_loss(traj, epoch=0, road_edges=edges, valid_mask=None)
        assert float(total_default) == float(total_none)

    def test_float_valid_mask_matches_bool(self) -> None:
        """A 0/1 float mask (as the data pipeline emits) matches a bool mask.

        The kinematic term combines per-step validity with a boolean ``&``, so
        a float32 mask must be accepted rather than raising in ``bitwise_and``.
        """
        loss_fn = DiffAVPhysicsLoss()
        traj = _straight_trajectories(num_agents=3, future_steps=6)
        edges = _square_road_edges()
        bool_mask = jnp.array([[True] * 6, [False] * 6, [True] * 6])
        total_bool, _ = loss_fn.compute_loss(traj, epoch=0, road_edges=edges, valid_mask=bool_mask)
        total_float, _ = loss_fn.compute_loss(
            traj, epoch=0, road_edges=edges, valid_mask=bool_mask.astype(jnp.float32)
        )
        assert float(total_float) == pytest.approx(float(total_bool), rel=1e-6)

    def test_offroad_chunk_size_leaves_boundary_unchanged(self) -> None:
        """The offroad_chunk_size memory knob does not change the boundary loss.

        Several road-edge polylines make a small chunk scan multiple chunks, so
        this exercises the memory-bounded path end to end through the loss.
        """
        edges = RoadEdges.from_polylines(
            [_ROAD_EDGE_SQUARE, _ROAD_EDGE_SQUARE + 30.0, _ROAD_EDGE_SQUARE + 60.0]
        )
        traj = _straight_trajectories(num_agents=3, future_steps=6)
        chunked = DiffAVPhysicsLoss(DiffAVPhysicsConfig(offroad_chunk_size=2))
        unchunked = DiffAVPhysicsLoss(DiffAVPhysicsConfig())
        _, chunked_components = chunked.compute_loss(traj, epoch=0, road_edges=edges)
        _, unchunked_components = unchunked.compute_loss(traj, epoch=0, road_edges=edges)
        assert float(chunked_components["boundary_loss"]) == pytest.approx(
            float(unchunked_components["boundary_loss"]), rel=1e-6
        )
