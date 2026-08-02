"""Tests for bicycle model kinematic constraints."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from simulacrax.physics.kinematics import (
    AckermanSteeringConstraint,
    BicycleModelConfig,
    BicycleModelConstraint,
    compute_kinematic_features,
    compute_kinematic_validity,
    KinematicValidationResult,
)
from tests.physics.helpers import (
    FUTURE_STEPS as _FUTURE_STEPS,
    NUM_AGENTS as _NUM_AGENTS,
    STATE_DIM as _STATE_DIM,
    stationary_trajectory as _stationary_trajectory,
    straight_line_trajectory as _straight_line_trajectory,
)


# ---------------------------------------------------------------------------
# TestBicycleModelConfig
# ---------------------------------------------------------------------------


class TestBicycleModelConfig:
    """Tests for BicycleModelConfig frozen dataclass."""

    def test_defaults(self) -> None:
        """Default config has expected field values."""
        cfg = BicycleModelConfig()
        assert cfg.wheelbase == 2.7
        assert cfg.dt == 0.1
        assert cfg.max_acceleration == 8.0
        assert cfg.max_steering_angle == 0.7
        assert cfg.max_velocity == 40.0

    @pytest.mark.parametrize(
        "field",
        ["wheelbase", "dt", "max_acceleration", "max_steering_angle", "max_velocity"],
    )
    def test_validation_positive_fields(self, field: str) -> None:
        """Setting any field to zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            BicycleModelConfig(**{field: 0.0})

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        cfg = BicycleModelConfig(
            wheelbase=3.0,
            dt=0.05,
            max_acceleration=6.0,
            max_steering_angle=0.5,
            max_velocity=30.0,
        )
        assert cfg.wheelbase == 3.0
        assert cfg.dt == 0.05
        assert cfg.max_acceleration == 6.0
        assert cfg.max_steering_angle == 0.5
        assert cfg.max_velocity == 30.0


# ---------------------------------------------------------------------------
# TestBicycleModelConstraint
# ---------------------------------------------------------------------------


class TestBicycleModelConstraint:
    """Tests for BicycleModelConstraint residual computation."""

    def test_straight_line_near_zero_residual(self) -> None:
        """Straight constant-velocity trajectory produces near-zero residual."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        residual = constraint.compute_residuals(traj)
        assert float(residual) == pytest.approx(0.0, abs=1e-4)

    def test_stationary_agent_zero_residual(self) -> None:
        """Stationary trajectory (all zeros) produces near-zero residual."""
        constraint = BicycleModelConstraint()
        traj = _stationary_trajectory()
        residual = constraint.compute_residuals(traj)
        assert float(residual) == pytest.approx(0.0, abs=1e-4)

    def test_random_trajectory_nonzero_residual(self) -> None:
        """Random trajectories produce a nonzero residual."""
        constraint = BicycleModelConstraint()
        key = jax.random.key(42)
        traj = jax.random.normal(key, (_NUM_AGENTS, _FUTURE_STEPS, _STATE_DIM))
        residual = constraint.compute_residuals(traj)
        assert float(residual) > 0.0

    def test_residual_is_scalar(self) -> None:
        """Residual output has scalar shape."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        residual = constraint.compute_residuals(traj)
        assert residual.shape == ()

    def test_residual_differentiable(self) -> None:
        """Gradient of residual w.r.t. trajectories can be computed."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()

        grad_fn = jax.grad(lambda t: constraint.compute_residuals(t))
        grad = grad_fn(traj)
        assert grad.shape == traj.shape

    def test_residual_jit_compatible(self) -> None:
        """JIT-compiled residual matches eager result."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()

        eager = constraint.compute_residuals(traj)
        jitted = jax.jit(constraint.compute_residuals)(traj)
        assert float(jnp.abs(eager - jitted)) < 1e-6

    def test_residual_accepts_float_valid_mask(self) -> None:
        """A 0/1 float mask (as the data pipeline emits) is accepted.

        The transition validity uses a boolean AND, which rejects a float
        dtype; the residual must normalize it rather than raise.
        """
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        steps = traj.shape[1]
        bool_mask = jnp.ones((traj.shape[0], steps), dtype=bool)
        residual_bool = constraint.compute_residuals(traj, bool_mask)
        residual_float = constraint.compute_residuals(traj, bool_mask.astype(jnp.float32))
        assert float(residual_float) == pytest.approx(float(residual_bool), rel=1e-6)

    def test_default_config(self) -> None:
        """BicycleModelConstraint can be constructed without arguments."""
        constraint = BicycleModelConstraint()
        assert constraint.config.wheelbase == 2.7


# ---------------------------------------------------------------------------
# TestKinematicValidation
# ---------------------------------------------------------------------------


def _bicycle_rollout(
    config: BicycleModelConfig,
    steering_angle: float,
    velocity: float,
    steps: int,
    heading_rate_override: float | None = None,
) -> jax.Array:
    """Roll out one agent under exact bicycle kinematics.

    Positions integrate the actual headings, so the position residual is
    zero by construction; the heading rate follows the bicycle model for
    the given steering angle unless overridden.
    """
    x, y, theta = 0.0, 0.0, 0.0
    if heading_rate_override is None:
        heading_rate = velocity / config.wheelbase * math.tan(steering_angle)
    else:
        heading_rate = heading_rate_override
    states = []
    for _ in range(steps):
        states.append([x, y, theta, velocity])
        x += velocity * math.cos(theta) * config.dt
        y += velocity * math.sin(theta) * config.dt
        theta += heading_rate * config.dt
    return jnp.array([states])


class TestSteeringDerivedHeadingResidual:
    """The heading residual penalizes only steering-infeasible turning."""

    def test_feasible_turn_zero_residual(self) -> None:
        """A turn within the steering limit incurs no residual."""
        config = BicycleModelConfig()
        constraint = BicycleModelConstraint(config=config)
        traj = _bicycle_rollout(config, steering_angle=0.5, velocity=5.0, steps=12)
        assert float(constraint.compute_residuals(traj)) == pytest.approx(0.0, abs=1e-6)

    def test_infeasible_heading_rate_positive_residual(self) -> None:
        """Turning faster than max steering allows is penalized."""
        config = BicycleModelConfig()
        constraint = BicycleModelConstraint(config=config)
        # Max feasible heading rate at 1 m/s: tan(0.7)/2.7 ≈ 0.31 rad/s;
        # spin at 3 rad/s while positions still track the headings.
        traj = _bicycle_rollout(
            config, steering_angle=0.0, velocity=1.0, steps=12, heading_rate_override=3.0
        )
        assert float(constraint.compute_residuals(traj)) > 0.0


class TestKinematicValidation:
    """Tests for BicycleModelConstraint.validate detailed output."""

    def test_validate_returns_result(self) -> None:
        """Validate returns a KinematicValidationResult instance."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        result = constraint.validate(traj)
        assert isinstance(result, KinematicValidationResult)

    def test_valid_straight_line(self) -> None:
        """Straight constant-velocity trajectory is valid."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        result = constraint.validate(traj)
        assert bool(result.is_valid) is True

    def test_detect_acceleration_violation(self) -> None:
        """Sudden velocity jump triggers acceleration violation."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory(velocity=5.0)
        # Inject a sudden velocity jump at step 4
        traj_mut = jnp.array(traj)
        traj_mut = traj_mut.at[:, 4, 3].set(50.0)  # velocity spike
        result = constraint.validate(traj_mut)
        assert float(jnp.sum(result.acceleration_violation)) > 0.0
        assert bool(result.is_valid) is False

    def test_detect_velocity_violation(self) -> None:
        """Velocity exceeding max_velocity triggers velocity violation."""
        cfg = BicycleModelConfig(max_velocity=5.0)
        constraint = BicycleModelConstraint(config=cfg)
        traj = _straight_line_trajectory(velocity=10.0, dt=cfg.dt)
        result = constraint.validate(traj)
        assert float(jnp.sum(result.velocity_violation)) > 0.0
        assert bool(result.is_valid) is False

    def test_validate_jit_compatible(self) -> None:
        """validate traces under jax.jit and matches the eager result."""
        constraint = BicycleModelConstraint()
        traj = _straight_line_trajectory()
        eager = constraint.validate(traj)
        jitted = jax.jit(constraint.validate)(traj)
        assert bool(jitted.is_valid) == bool(eager.is_valid)
        assert jnp.allclose(jitted.position_residual, eager.position_residual)
        assert jnp.allclose(jitted.acceleration_violation, eager.acceleration_violation)

    def test_per_constraint_components_have_native_units(self) -> None:
        """Validity decomposes per constraint; no mixed-unit total is exposed."""
        constraint = BicycleModelConstraint()
        result = constraint.validate(_straight_line_trajectory())
        assert not hasattr(result, "total_violation")
        for component in (
            result.position_residual,
            result.heading_residual,
            result.acceleration_violation,
            result.steering_violation,
            result.velocity_violation,
        ):
            assert component.shape == (_NUM_AGENTS,)


# ---------------------------------------------------------------------------
# TestAckermanSteering
# ---------------------------------------------------------------------------


class TestAckermanSteering:
    """Tests for AckermanSteeringConstraint."""

    def test_min_turning_radius(self) -> None:
        """Minimum turning radius is positive and matches analytic formula."""
        ackerman = AckermanSteeringConstraint()
        radius = ackerman.compute_min_turning_radius()
        assert radius > 0.0
        expected = ackerman.config.wheelbase / math.tan(ackerman.config.max_steering_angle)
        assert radius == pytest.approx(expected, rel=1e-6)

    def test_straight_line_zero_violation(self) -> None:
        """Straight-line trajectory produces near-zero violation."""
        ackerman = AckermanSteeringConstraint()
        traj = _straight_line_trajectory()
        violation = ackerman.compute_violation(traj)
        assert float(violation) == pytest.approx(0.0, abs=1e-2)

    def test_default_config(self) -> None:
        """AckermanSteeringConstraint can be constructed without arguments."""
        ackerman = AckermanSteeringConstraint()
        assert ackerman.config.wheelbase == 2.7


# ---------------------------------------------------------------------------
# Kinematic features — golden values generated by executing the official
# waymo-open-dataset reference (compute_kinematic_features and
# compute_kinematic_validity in wdl_limited/sim_agents_metrics/
# trajectory_features.py) on these inputs.
# ---------------------------------------------------------------------------


def _curved_trajectory() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Curved 3D trajectory whose heading wraps across the +-pi boundary."""
    x = jnp.array([0.0, 1.0, 2.5, 4.5, 7.0, 9.0, 10.5, 11.5])
    y = jnp.array([0.0, 0.2, 0.8, 1.8, 3.0, 4.6, 6.4, 8.4])
    z = jnp.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5])
    heading = jnp.array([3.0, 3.1, -3.1, -3.0, -2.8, -2.9, 3.1, 2.9])
    return x, y, z, heading


class TestComputeKinematicFeatures:
    """Reference-validated tests for central-difference kinematic features."""

    def test_curved_trajectory_matches_reference(self) -> None:
        """Speeds/accelerations incl. angle wrap match the official values."""
        x, y, z, heading = _curved_trajectory()
        speed, accel, ang_speed, ang_accel = compute_kinematic_features(
            x, y, z, heading, seconds_per_step=0.1
        )
        nan = float("nan")
        expected_speed = jnp.array(
            [nan, 13.133925, 19.25649, 25.064917, 26.51061, 24.402868, 22.755491, nan]
        )
        expected_accel = jnp.array([nan, nan, 59.654957, 36.2706, -3.310242, -18.775597, nan, nan])
        expected_ang_speed = jnp.array(
            [nan, 0.915928, 0.915928, 1.5, 0.5, -1.915928, -2.415925, nan]
        )
        expected_ang_accel = jnp.array(
            [nan, nan, 2.920353, -2.079642, -17.07964, -14.579618, nan, nan]
        )
        for result, expected in (
            (speed, expected_speed),
            (accel, expected_accel),
            (ang_speed, expected_ang_speed),
            (ang_accel, expected_ang_accel),
        ):
            assert jnp.allclose(result, expected, atol=1e-4, equal_nan=True)

    def test_constant_velocity_batched(self) -> None:
        """Batched constant-velocity agents give constant speed, zero accel."""
        steps = jnp.arange(8, dtype=jnp.float32)
        x = jnp.stack([steps * 2.0, steps * -1.0])
        zeros = jnp.zeros_like(x)
        speed, accel, _, _ = compute_kinematic_features(
            x, zeros, zeros, zeros, seconds_per_step=0.1
        )
        nan = float("nan")
        expected_speed = jnp.array(
            [
                [nan, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, nan],
                [nan, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, nan],
            ]
        )
        expected_accel = jnp.array(
            [
                [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan],
                [nan, nan, 0.0, 0.0, 0.0, 0.0, nan, nan],
            ]
        )
        assert jnp.allclose(speed, expected_speed, atol=1e-5, equal_nan=True)
        assert jnp.allclose(accel, expected_accel, atol=1e-5, equal_nan=True)

    def test_validity_matches_reference(self) -> None:
        """Validity requires both neighbours valid (element itself exempt)."""
        valid = jnp.array(
            [
                [True, True, True, False, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )
        speed_valid, accel_valid = compute_kinematic_validity(valid)
        expected_speed_valid = jnp.array(
            [
                [False, True, False, True, False, True, True, False],
                [False, True, True, True, True, True, True, False],
            ]
        )
        expected_accel_valid = jnp.array(
            [
                [False, False, True, False, True, False, False, False],
                [False, False, True, True, True, True, False, False],
            ]
        )
        assert jnp.array_equal(speed_valid, expected_speed_valid)
        assert jnp.array_equal(accel_valid, expected_accel_valid)

    def test_jit_compatible(self) -> None:
        """Features trace under jit and match the eager result."""
        x, y, z, heading = _curved_trajectory()
        jitted = jax.jit(compute_kinematic_features, static_argnames=("seconds_per_step",))
        eager = compute_kinematic_features(x, y, z, heading, seconds_per_step=0.1)
        traced = jitted(x, y, z, heading, seconds_per_step=0.1)
        for a, b in zip(traced, eager, strict=True):
            assert jnp.allclose(a, b, equal_nan=True)

    def test_grad_flows_through_interior(self) -> None:
        """Gradients w.r.t. positions are finite on interior (non-pad) steps."""
        x, y, z, heading = _curved_trajectory()

        def interior_speed_sum(px: jax.Array) -> jax.Array:
            speed, _, _, _ = compute_kinematic_features(px, y, z, heading, seconds_per_step=0.1)
            return jnp.sum(speed[1:-1])

        grads = jax.grad(interior_speed_sum)(x)
        assert grads.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(grads)))

    def test_vmap_over_agents(self) -> None:
        """Features vmap over a leading agent axis."""
        x, y, z, heading = _curved_trajectory()
        stack = lambda a: jnp.stack([a, a])  # noqa: E731
        batched = jax.vmap(
            lambda px, py, pz, ph: compute_kinematic_features(px, py, pz, ph, seconds_per_step=0.1)
        )(stack(x), stack(y), stack(z), stack(heading))
        eager = compute_kinematic_features(x, y, z, heading, seconds_per_step=0.1)
        for b, e in zip(batched, eager, strict=True):
            assert b.shape == (2, x.shape[0])
            assert jnp.allclose(b[0], e, equal_nan=True)
