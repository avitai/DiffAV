"""Tests for the WOSAC-style histogram log-likelihood metametric."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffav.core.geometry import RoadEdges
from diffav.evaluation.wosac_metametric import (
    compute_metametric_features,
    MetametricFeatures,
    WosacMetametric,
    WosacMetametricResult,
)


_STEPS = 12
_AGENTS = 2


def _road_edges() -> RoadEdges:
    """Counterclockwise closed square: interior on-road."""
    square = np.array(
        [[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0], [-50, -50, 0]],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


def _rollouts(num_rollouts: int, speed: float = 5.0, lateral_offset: float = 0.0) -> jax.Array:
    """Straight constant-velocity rollouts, two well-separated agents."""
    t = jnp.arange(_STEPS, dtype=jnp.float32)
    x = speed * 0.1 * t
    traj = jnp.zeros((num_rollouts, _AGENTS, _STEPS, 4))
    traj = traj.at[:, :, :, 0].set(x[jnp.newaxis, jnp.newaxis, :])
    traj = traj.at[:, 1, :, 1].set(20.0 + lateral_offset)
    traj = traj.at[:, :, :, 3].set(speed)
    return traj


class TestComputeMetametricFeatures:
    """Feature assembly from rollout trajectories."""

    def test_shapes(self) -> None:
        """All per-rollout features carry (K, N, T); valid carries (N, T)."""
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        features = compute_metametric_features(
            _rollouts(3), valid, _road_edges(), collision_threshold=1.0
        )
        assert isinstance(features, MetametricFeatures)
        for array in (
            features.linear_speed,
            features.linear_acceleration,
            features.angular_speed,
            features.angular_acceleration,
            features.distance_to_nearest_object,
            features.collision_per_step,
            features.distance_to_road_edge,
            features.offroad_per_step,
        ):
            assert array.shape == (3, _AGENTS, _STEPS)
        assert features.valid.shape == (_AGENTS, _STEPS)

    def test_kinematics_and_map_features_are_consistent(self) -> None:
        """Constant motion inside the square yields the expected features."""
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        features = compute_metametric_features(
            _rollouts(1, speed=5.0), valid, _road_edges(), collision_threshold=1.0
        )
        interior_speed = features.linear_speed[0, :, 1:-1]
        assert jnp.allclose(interior_speed, 5.0, atol=1e-4)
        # Agents at y=0 and y=20 inside the 100 m square: on-road (negative).
        assert bool(jnp.all(features.distance_to_road_edge < 0.0))
        assert not bool(jnp.any(features.offroad_per_step))
        # 20 m apart: no collision at threshold 1 m.
        assert not bool(jnp.any(features.collision_per_step))
        assert jnp.allclose(features.distance_to_nearest_object[0, :, 0], 20.0, atol=1e-4)

    def test_registered_pytree(self) -> None:
        """MetametricFeatures flattens into arrays for jit/vmap use."""
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        features = compute_metametric_features(
            _rollouts(2), valid, _road_edges(), collision_threshold=1.0
        )
        leaves = jax.tree_util.tree_leaves(features)
        assert len(leaves) == 9

    def test_offroad_chunk_size_leaves_distance_unchanged(self) -> None:
        """The offroad_chunk_size memory knob does not change the distance.

        Several road-edge polylines make a small chunk scan multiple chunks, so
        this exercises the memory-bounded eval path; the full-resolution result
        must be reproduced exactly.
        """
        square = np.array(
            [[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0], [-50, -50, 0]],
            dtype=np.float32,
        )
        edges = RoadEdges.from_polylines([square, square + 200.0, square + 400.0])
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        chunked = compute_metametric_features(_rollouts(2), valid, edges, offroad_chunk_size=1)
        monolithic = compute_metametric_features(_rollouts(2), valid, edges)
        assert jnp.allclose(
            chunked.distance_to_road_edge, monolithic.distance_to_road_edge, atol=1e-5
        )


class TestWosacMetametric:
    """Scoring flow: likelihoods, buckets, and the weighted metametric."""

    def _score(self, sim: jax.Array, log: jax.Array) -> WosacMetametricResult:
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        edges = _road_edges()
        log_features = compute_metametric_features(log, valid, edges, collision_threshold=1.0)
        sim_features = compute_metametric_features(sim, valid, edges, collision_threshold=1.0)
        return WosacMetametric().compute(log_features, sim_features)

    def test_matching_distributions_score_high(self) -> None:
        """Sim rollouts identical to the log scenario score near the maximum."""
        log = _rollouts(1, speed=5.0)
        sim = _rollouts(8, speed=5.0)
        result = self._score(sim, log)
        assert result.active_weight_sum == pytest.approx(0.85)
        assert 0.0 < result.metametric <= result.active_weight_sum + 1e-6
        assert result.normalized_metametric == pytest.approx(
            result.metametric / result.active_weight_sum
        )
        assert result.normalized_metametric > 0.85

    def test_shifted_speeds_lower_the_kinematic_bucket(self) -> None:
        """Sim speeds far from the log speed lower kinematic likelihoods."""
        log = _rollouts(1, speed=5.0)
        good = self._score(_rollouts(8, speed=5.0), log)
        shifted = self._score(_rollouts(8, speed=20.0), log)
        assert shifted.kinematic_score < good.kinematic_score
        assert shifted.metametric < good.metametric

    def test_injected_collisions_lower_the_interactive_bucket(self) -> None:
        """Sim rollouts that collide (log does not) lower the collision score."""
        log = _rollouts(1)
        colliding = _rollouts(8, lateral_offset=-20.0)  # both agents at y=0
        good = self._score(_rollouts(8), log)
        bad = self._score(colliding, log)
        assert bad.interactive_score < good.interactive_score
        assert bad.metametric < good.metametric

    def test_per_feature_likelihoods_in_unit_interval(self) -> None:
        """Every per-feature likelihood is a probability."""
        result = self._score(_rollouts(4), _rollouts(1))
        assert set(result.per_feature) == {
            "linear_speed",
            "linear_acceleration",
            "angular_speed",
            "angular_acceleration",
            "distance_to_nearest_object",
            "collision_indication",
            "distance_to_road_edge",
            "offroad_indication",
        }
        for value in result.per_feature.values():
            assert 0.0 <= value <= 1.0

    def test_validity_masking_ignores_invalid_steps(self) -> None:
        """Features at invalid log steps do not affect the scores."""
        edges = _road_edges()
        log = _rollouts(1)
        sim = _rollouts(4)
        valid = jnp.ones((_AGENTS, _STEPS), dtype=bool)
        # Corrupt the log scenario at a step marked invalid.
        log_corrupt = log.at[0, :, 5, 0].add(1000.0)
        valid_masked = valid.at[:, 4:7].set(False)

        clean = WosacMetametric().compute(
            compute_metametric_features(log, valid_masked, edges, collision_threshold=1.0),
            compute_metametric_features(sim, valid_masked, edges, collision_threshold=1.0),
        )
        corrupt = WosacMetametric().compute(
            compute_metametric_features(log_corrupt, valid_masked, edges, collision_threshold=1.0),
            compute_metametric_features(sim, valid_masked, edges, collision_threshold=1.0),
        )
        # The corruption is confined to invalid steps (central differences
        # reach one step each side), so kinematic scores must be identical.
        assert corrupt.per_feature["linear_speed"] == pytest.approx(
            clean.per_feature["linear_speed"], abs=1e-6
        )
