"""Tests for JAX-native WOD evaluation metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simulacrax.core.geometry import RoadEdges
from simulacrax.evaluation.metrics import (
    ade,
    benchmark_min_ade,
    fde,
    min_ade,
    min_fde,
    miss_rate,
    MotionMetrics,
    MotionMetricsConfig,
    offroad_rate,
    SimAgentMetrics,
    SimAgentMetricsConfig,
    SimAgentMetricsResult,
)


def _square_edges() -> RoadEdges:
    """Counterclockwise 20 m square centred at the origin (interior on-road)."""
    square = np.array(
        [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0], [-10, -10, 0]],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


class TestOffroadRate:
    """Fraction of valid agent-steps off the road, via the exact WOSAC distance."""

    def _rollouts(self, x: float, y: float) -> jax.Array:
        """One rollout, two agents held at a constant position (K=1, A=2, T=3, 4)."""
        traj = jnp.zeros((1, 2, 3, 4))
        return traj.at[..., :2].set(jnp.array([x, y]))

    def test_on_road_positions_are_zero(self) -> None:
        """Agents inside the square incur no off-road."""
        valid = jnp.ones((2, 3), dtype=bool)
        assert float(offroad_rate(self._rollouts(0.0, 0.0), _square_edges(), valid)) == 0.0

    def test_off_road_positions_are_one(self) -> None:
        """Agents far outside the square are off-road at every valid step."""
        valid = jnp.ones((2, 3), dtype=bool)
        assert float(offroad_rate(self._rollouts(50.0, 50.0), _square_edges(), valid)) == 1.0

    def test_excludes_invalid_steps(self) -> None:
        """A fully-invalid batch yields a finite zero rate (guarded denominator)."""
        valid = jnp.zeros((2, 3), dtype=bool)
        rate = offroad_rate(self._rollouts(50.0, 50.0), _square_edges(), valid)
        assert float(rate) == 0.0

    def test_chunk_size_invariant(self) -> None:
        """The chunk_size memory knob does not change the rate."""
        edges = RoadEdges.from_polylines(
            [
                np.array(
                    [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0], [-10, -10, 0]],
                    np.float32,
                ),
                np.array([[90, 90, 0], [110, 90, 0], [110, 110, 0], [90, 90, 0]], np.float32),
            ]
        )
        valid = jnp.ones((2, 3), dtype=bool)
        rollouts = self._rollouts(50.0, 50.0)
        assert float(offroad_rate(rollouts, edges, valid, chunk_size=1)) == float(
            offroad_rate(rollouts, edges, valid)
        )


class TestAde:
    def test_zero_displacement(self) -> None:
        pred = jnp.zeros((10, 2))
        gt = jnp.zeros((10, 2))
        valid = jnp.ones(10)
        assert float(ade(pred, gt, valid)) == pytest.approx(0.0)

    def test_unit_step(self) -> None:
        pred = jnp.ones((4, 2))
        gt = jnp.zeros((4, 2))
        valid = jnp.ones(4)
        expected = float(jnp.linalg.norm(jnp.ones(2)))  # sqrt(2)
        assert float(ade(pred, gt, valid)) == pytest.approx(expected, rel=1e-5)

    def test_partial_validity(self) -> None:
        pred = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        gt = jnp.zeros((3, 2))
        valid = jnp.array([1.0, 0.0, 0.0])
        assert float(ade(pred, gt, valid)) == pytest.approx(1.0, rel=1e-5)

    def test_all_invalid_returns_zero(self) -> None:
        pred = jnp.ones((4, 2))
        gt = jnp.zeros((4, 2))
        valid = jnp.zeros(4)
        assert float(ade(pred, gt, valid)) == pytest.approx(0.0)


class TestFde:
    def test_zero_displacement(self) -> None:
        pred = jnp.zeros((5, 2))
        gt = jnp.zeros((5, 2))
        valid = jnp.ones(5)
        assert float(fde(pred, gt, valid)) == pytest.approx(0.0)

    def test_unit_error_at_last_step(self) -> None:
        pred = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        gt = jnp.zeros((3, 2))
        valid = jnp.ones(3)
        assert float(fde(pred, gt, valid)) == pytest.approx(1.0, rel=1e-5)

    def test_uses_last_valid_step(self) -> None:
        pred = jnp.array([[0.0, 0.0], [2.0, 0.0], [99.0, 0.0]])
        gt = jnp.zeros((3, 2))
        valid = jnp.array([1.0, 1.0, 0.0])
        assert float(fde(pred, gt, valid)) == pytest.approx(2.0, rel=1e-5)

    def test_last_valid_step_with_occlusion_gap(self) -> None:
        """Non-contiguous validity (occlusion) uses the last valid index."""
        pred = jnp.array([[0.0, 0.0], [99.0, 0.0], [3.0, 0.0], [50.0, 0.0]])
        gt = jnp.zeros((4, 2))
        valid = jnp.array([1.0, 0.0, 1.0, 0.0])
        assert float(fde(pred, gt, valid)) == pytest.approx(3.0, rel=1e-5)

    def test_all_invalid_returns_zero(self) -> None:
        pred = jnp.full((3, 2), 7.0)
        gt = jnp.zeros((3, 2))
        valid = jnp.zeros(3)
        assert float(fde(pred, gt, valid)) == pytest.approx(0.0)


class TestMinAde:
    def test_selects_best_hypothesis(self) -> None:
        pred_k = jnp.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0]],
            ]
        )
        gt = jnp.zeros((2, 2))
        valid = jnp.ones(2)
        assert float(min_ade(pred_k, gt, valid)) == pytest.approx(0.0)

    def test_single_hypothesis(self) -> None:
        pred_k = jnp.ones((1, 4, 2))
        gt = jnp.zeros((4, 2))
        valid = jnp.ones(4)
        expected = float(jnp.linalg.norm(jnp.ones(2)))
        assert float(min_ade(pred_k, gt, valid)) == pytest.approx(expected, rel=1e-5)


class TestMinFde:
    def test_selects_best_final(self) -> None:
        pred_k = jnp.array(
            [
                [[5.0, 0.0], [3.0, 0.0]],
                [[5.0, 0.0], [0.0, 0.0]],
            ]
        )
        gt = jnp.zeros((2, 2))
        valid = jnp.ones(2)
        assert float(min_fde(pred_k, gt, valid)) == pytest.approx(0.0)


class TestMissRate:
    def test_all_miss(self) -> None:
        pred_k = jnp.full((3, 4, 2), 10.0)
        gt = jnp.zeros((4, 2))
        valid = jnp.ones(4)
        assert float(miss_rate(pred_k, gt, valid, threshold=2.0)) == pytest.approx(1.0)

    def test_one_hit(self) -> None:
        pred_k = jnp.array(
            [
                [[10.0, 0.0], [10.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        gt = jnp.zeros((2, 2))
        valid = jnp.ones(2)
        assert float(miss_rate(pred_k, gt, valid, threshold=2.0)) == pytest.approx(0.0)


class TestMotionMetrics:
    """Tests for MotionMetrics class."""

    def _make_config(self) -> MotionMetricsConfig:
        return MotionMetricsConfig(miss_rate_threshold=2.0, top_k=2, num_future_steps=3)

    def _make_inputs(
        self, b: int = 1, m: int = 1, k: int = 2, n: int = 2, t: int = 3
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        predictions = jnp.zeros((b, m, k, n, t, 2))
        scores = jnp.ones((b, m, k)) / k
        ground_truth = jnp.zeros((b, n, t, 7))
        valid = jnp.ones((b, n, t))
        object_type = jnp.zeros((b, n), dtype=jnp.int32)
        return predictions, scores, ground_truth, valid, object_type

    def test_returns_dict(self) -> None:
        metrics = MotionMetrics(self._make_config())
        result = metrics.compute(*self._make_inputs())
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_zero_error_perfect_predictions(self) -> None:
        metrics = MotionMetrics(self._make_config())
        result = metrics.compute(*self._make_inputs())
        for key, val in result.items():
            if "minADE" in key or "minFDE" in key or "MissRate" in key:
                assert val == pytest.approx(0.0, abs=1e-5), f"{key}={val}"

    def test_per_type_vehicle_keys_present(self) -> None:
        metrics = MotionMetrics(self._make_config())
        result = metrics.compute(*self._make_inputs())
        assert any("VEHICLE" in k for k in result)

    def test_config_immutable(self) -> None:
        config = MotionMetricsConfig()
        with pytest.raises(AttributeError):
            config.top_k = 99  # type: ignore[misc]

    def test_multiple_prediction_groups_align_with_ground_truth(self) -> None:
        """With M>1, group m / agent n maps to ground-truth agent m*N+n."""
        b, m, k, n, t = 1, 2, 1, 1, 3
        gt = jnp.zeros((b, m * n, t, 7))
        # Agent 0 (vehicle) drives straight in x; agent 1 (pedestrian) in y.
        gt = gt.at[0, 0, :, 0].set(jnp.arange(t, dtype=jnp.float32))
        gt = gt.at[0, 1, :, 1].set(jnp.arange(t, dtype=jnp.float32))
        predictions = jnp.stack([gt[0, 0, :, :2], gt[0, 1, :, :2]], axis=0).reshape(
            b, m, k, n, t, 2
        )
        scores = jnp.ones((b, m, k))
        valid = jnp.ones((b, m * n, t))
        object_type = jnp.array([[0, 1]], dtype=jnp.int32)

        metrics = MotionMetrics(self._make_config())
        result = metrics.compute(predictions, scores, gt, valid, object_type)
        assert result["VEHICLE/minADE"] == pytest.approx(0.0, abs=1e-5)
        assert result["PEDESTRIAN/minADE"] == pytest.approx(0.0, abs=1e-5)

    def test_agent_count_mismatch_raises(self) -> None:
        """A != M*N is rejected instead of silently mis-reshaping."""
        predictions = jnp.zeros((1, 2, 1, 2, 3, 2))  # M*N = 4
        scores = jnp.ones((1, 2, 1))
        gt = jnp.zeros((1, 3, 3, 7))  # A = 3
        valid = jnp.ones((1, 3, 3))
        object_type = jnp.zeros((1, 3), dtype=jnp.int32)
        metrics = MotionMetrics(self._make_config())
        with pytest.raises(ValueError, match="agents"):
            metrics.compute(predictions, scores, gt, valid, object_type)

    def test_map_uses_configured_threshold(self) -> None:
        """mAP hit/miss uses the configured miss-rate threshold."""
        b, m, k, n, t = 1, 1, 1, 1, 2
        gt = jnp.zeros((b, n, t, 7))
        # Constant 1 m final displacement.
        predictions = jnp.full((b, m, k, n, t, 2), 1.0 / jnp.sqrt(2.0))
        scores = jnp.ones((b, m, k))
        valid = jnp.ones((b, n, t))
        object_type = jnp.zeros((b, n), dtype=jnp.int32)

        loose = MotionMetrics(MotionMetricsConfig(miss_rate_threshold=2.0))
        tight = MotionMetrics(MotionMetricsConfig(miss_rate_threshold=0.5))
        loose_map = loose.compute(predictions, scores, gt, valid, object_type)
        tight_map = tight.compute(predictions, scores, gt, valid, object_type)
        assert loose_map["VEHICLE/mAP"] == pytest.approx(1.0)
        assert tight_map["VEHICLE/mAP"] == pytest.approx(0.0)


class TestSimAgentMetrics:
    """Tests for SimAgentMetrics class."""

    def _make_config(self) -> SimAgentMetricsConfig:
        return SimAgentMetricsConfig(
            kinematic_weight=0.4,
            interactive_weight=0.4,
            map_weight=0.2,
            collision_threshold=1.0,
        )

    def test_returns_result(self) -> None:
        metrics = SimAgentMetrics(self._make_config())
        trajectories = jnp.zeros((2, 3, 4, 4))
        road_edge_distances = jnp.full((3, 4), -5.0)
        result = metrics.compute(trajectories, road_edge_distances)
        assert isinstance(result, SimAgentMetricsResult)

    def test_metametric_in_range(self) -> None:
        metrics = SimAgentMetrics(self._make_config())
        result = metrics.compute(jnp.zeros((2, 3, 4, 4)), jnp.full((3, 4), -5.0))
        assert 0.0 <= result.metametric <= 1.0

    def test_offroad_lowers_map_score(self) -> None:
        """Positive signed road-edge distance (off-road) lowers the score."""
        metrics = SimAgentMetrics(self._make_config())
        traj = jnp.zeros((2, 3, 4, 4))
        result_ok = metrics.compute(traj, jnp.full((3, 4), -5.0))
        result_bad = metrics.compute(traj, jnp.full((3, 4), +5.0))
        assert result_bad.map_score < result_ok.map_score

    def test_result_has_per_feature(self) -> None:
        metrics = SimAgentMetrics(self._make_config())
        result = metrics.compute(jnp.zeros((2, 3, 4, 4)), jnp.full((3, 4), -5.0))
        assert isinstance(result.per_feature, dict)
        assert len(result.per_feature) > 0

    def test_config_immutable(self) -> None:
        config = SimAgentMetricsConfig()
        with pytest.raises(AttributeError):
            config.kinematic_weight = 0.9  # type: ignore[misc]

    @staticmethod
    def _straight_rollouts(displacement_per_step: float, steps: int = 12) -> jax.Array:
        """One rollout, two well-separated agents moving straight in x."""
        t = jnp.arange(steps, dtype=jnp.float32)
        x = displacement_per_step * t
        traj = jnp.zeros((1, 2, steps, 4))
        traj = traj.at[0, :, :, 0].set(x[jnp.newaxis, :])
        traj = traj.at[0, 1, :, 1].set(100.0)  # agent 2 offset far away
        traj = traj.at[0, :, :, 3].set(displacement_per_step / 0.1)
        return traj

    def test_kinematics_derived_from_positions_with_dt(self) -> None:
        """Speeds come from positions at 10 Hz, not the speed channel.

        Positions advance 4 m per 0.1 s step (40 m/s > the 25 m/s WOSAC
        envelope) while the speed channel claims a modest value — the
        position-derived check must flag it.
        """
        metrics = SimAgentMetrics(self._make_config())
        traj = self._straight_rollouts(displacement_per_step=4.0)
        traj = traj.at[..., 3].set(10.0)  # speed channel lies
        result = metrics.compute(traj, jnp.full((2, traj.shape[2]), -5.0))
        assert result.per_feature["linear_speed_ok"] < 1.0

    def test_acceleration_bound_uses_dt(self) -> None:
        """Per-step speed changes are converted to m/s^2 before checking.

        Speed ramps 2 m/s per 0.1 s step (20 m/s^2), far beyond the
        12 m/s^2 envelope but under the raw per-step magnitude of 6 used
        before the fix.
        """
        metrics = SimAgentMetrics(self._make_config())
        steps = 12
        t = jnp.arange(steps, dtype=jnp.float32)
        # x(t) = 0.5 * a * (t*dt)^2 with a = 20 m/s^2.
        x = 0.5 * 20.0 * (t * 0.1) ** 2
        traj = jnp.zeros((1, 2, steps, 4))
        traj = traj.at[0, :, :, 0].set(x[jnp.newaxis, :])
        traj = traj.at[0, 1, :, 1].set(100.0)
        result = metrics.compute(traj, jnp.full((2, steps), -5.0))
        assert result.per_feature["linear_acceleration_ok"] < 1.0

    def test_feasible_motion_scores_full_kinematics(self) -> None:
        """In-envelope straight motion scores a full kinematic bucket."""
        metrics = SimAgentMetrics(self._make_config())
        traj = self._straight_rollouts(displacement_per_step=1.0)  # 10 m/s
        result = metrics.compute(traj, jnp.full((2, traj.shape[2]), -5.0))
        assert result.kinematic_score == pytest.approx(1.0)

    def test_rollouts_scored_individually_not_averaged(self) -> None:
        """Collisions are scored per rollout, not on the rollout-mean.

        Rollout 1 has both agents at the same position (full collision);
        rollout 2 keeps them far apart. Averaging trajectories across
        rollouts first would place the agents 5 m apart and hide the
        collision entirely.
        """
        metrics = SimAgentMetrics(self._make_config())
        steps = 6
        traj = jnp.zeros((2, 2, steps, 4))
        # Rollout 1: both agents at the origin — colliding at every step.
        # Rollout 2: agent 2 at y=10 — no collision.
        traj = traj.at[1, 1, :, 1].set(10.0)
        result = metrics.compute(traj, jnp.full((2, steps), -5.0))
        assert result.interactive_score == pytest.approx(0.5)


class TestBenchmarkMinAde:
    """WOMD-style minADE: 2 Hz subsample, 3/5/8 s horizon mean, min over K."""

    def test_perfect_prediction_is_zero(self) -> None:
        """Predicting the ground truth exactly gives zero."""
        out = benchmark_min_ade(jnp.zeros((1, 80, 2)), jnp.zeros((80, 2)), jnp.ones((80,), bool))
        assert float(out) == pytest.approx(0.0, abs=1e-6)

    def test_ignores_error_between_subsampled_points(self) -> None:
        """Only the 2 Hz-subsampled steps are scored; error between them is unseen."""
        pred = jnp.full((1, 80, 2), 100.0).at[:, 4::5, :].set(0.0)  # perfect at 2 Hz points
        out = benchmark_min_ade(pred, jnp.zeros((80, 2)), jnp.ones((80,), bool))
        assert float(out) == pytest.approx(0.0, abs=1e-6)

    def test_averages_over_the_three_horizons(self) -> None:
        """Perfect for the first 3 s then 10 m off: the 3/5/8 s horizon ADEs are
        0 / 4 / 6.25, so the reported minADE is their mean."""
        pred = jnp.zeros((1, 80, 2)).at[:, 30:, 0].set(10.0)
        out = benchmark_min_ade(pred, jnp.zeros((80, 2)), jnp.ones((80,), bool))
        assert float(out) == pytest.approx((0.0 + 4.0 + 6.25) / 3.0, abs=1e-4)

    def test_min_over_k_picks_the_closest_hypothesis(self) -> None:
        """The minimum over K is taken per horizon."""
        pred = jnp.stack([jnp.full((80, 2), 10.0), jnp.zeros((80, 2))])  # (2, 80, 2)
        out = benchmark_min_ade(pred, jnp.zeros((80, 2)), jnp.ones((80,), bool))
        assert float(out) == pytest.approx(0.0, abs=1e-6)

    def test_falls_back_to_dense_when_shorter_than_interval(self) -> None:
        """A trajectory shorter than the sampling interval is scored densely."""
        pred = jnp.zeros((1, 4, 2)).at[:, :, 0].set(3.0)  # 3 m offset along x
        out = benchmark_min_ade(pred, jnp.zeros((4, 2)), jnp.ones((4,), bool))
        assert float(out) == pytest.approx(3.0, rel=1e-5)

    def test_jit_compatible(self) -> None:
        """Composes with jit (static horizon masks, no dynamic slicing)."""
        out = jax.jit(benchmark_min_ade)(
            jnp.zeros((3, 80, 2)), jnp.zeros((80, 2)), jnp.ones((80,), bool)
        )
        assert float(out) == pytest.approx(0.0, abs=1e-6)
