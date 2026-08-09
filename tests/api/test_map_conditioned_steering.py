"""Tests for the map-conditioned adversarial candidate spine.

Cover the offline steering pipeline over a synthetic validation scene: drawing a
world-frame candidate pool, ranking it by the adversarial victim-approach reward,
gating feasibility on real road edges, and emitting same-scene preference pairs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffav.alignment.steering_spine import AdversarialRewardConfig
from diffav.api.map_conditioned_steering import (
    build_scene_pairs,
    sample_scene_candidates,
    score_scene_candidates,
)
from diffav.core.geometry import RoadEdges
from diffav.evaluation.map_conditioned_evaluator import ValidationScene
from tests.api.test_map_conditioned import _agent_rows, _model, _scene


_FUTURE = 4
_SELECTED = 2
_ADVERSARY = 0
_VICTIM = 1


def _road_edges(half: float) -> RoadEdges:
    """A counterclockwise square of side ``2 * half`` centred at the origin."""
    square = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


def _validation_scene(*, road_half: float) -> ValidationScene:
    """A synthetic two-agent validation scene with a square drivable region."""
    return ValidationScene(
        scene=_scene(),
        trajectories=jnp.zeros((_SELECTED, _FUTURE, 4)),
        agent_rows=_agent_rows(),
        valid=jnp.ones((_SELECTED, _FUTURE)),
        tracks_to_predict=jnp.array([False, True]),
        reference_pose=jnp.zeros((_SELECTED, 3)),
        road_edges=_road_edges(road_half),
    )


def _config() -> AdversarialRewardConfig:
    return AdversarialRewardConfig()


class TestSampleSceneCandidates:
    def test_candidate_shape(self) -> None:
        candidates = sample_scene_candidates(
            _model(),
            _validation_scene(road_half=5.0),
            num_candidates=4,
            key=jax.random.key(0),
        )
        assert candidates.shape == (4, _SELECTED, _FUTURE, 4)

    def test_too_few_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            sample_scene_candidates(
                _model(), _validation_scene(road_half=5.0), num_candidates=1, key=jax.random.key(0)
            )

    def test_deterministic_for_fixed_key(self) -> None:
        scene = _validation_scene(road_half=5.0)
        first = sample_scene_candidates(_model(), scene, num_candidates=4, key=jax.random.key(3))
        second = sample_scene_candidates(_model(), scene, num_candidates=4, key=jax.random.key(3))
        assert jnp.allclose(first, second)


class TestScoreSceneCandidates:
    def test_gate_reads_real_road_edges(self) -> None:
        """With a tight road, the adversary off-road gate is active, not vacuous."""
        scene = _validation_scene(road_half=0.1)
        candidates = sample_scene_candidates(
            _model(), scene, num_candidates=8, key=jax.random.key(1)
        )
        pool = score_scene_candidates(
            candidates,
            adversary_index=_ADVERSARY,
            victim_index=_VICTIM,
            scene=scene,
            config=_config(),
        )
        # A tight square leaves at least some adversary positions off-road, so the
        # gate computed genuine fractions (not the no-map zero fallback).
        assert bool(jnp.any(pool.offroad_fraction > 0.0))
        # feasible is exactly the threshold applied to those fractions.
        assert bool(jnp.all(pool.feasible == (pool.offroad_fraction <= 0.0)))

    def test_reward_rewards_proximity(self) -> None:
        """A candidate whose adversary hugs the victim outscores a far-off one.

        Uses hand-built candidates (not sampler draws) so the assertion tests the
        reward's proximity semantics directly, independent of the soft-min-vs-min
        reduction subtlety that makes sampled orderings ambiguous.
        """
        scene = _validation_scene(road_half=100.0)  # feasibility inert
        victim = jnp.zeros((_FUTURE, 4))  # victim parked at the origin
        near = victim.at[:, 0].set(0.5)  # adversary 0.5 m away
        far = victim.at[:, 0].set(50.0)  # adversary 50 m away
        candidates = jnp.stack(
            [jnp.stack([near, victim]), jnp.stack([far, victim])]
        )  # (2, agents=2, T, 4)
        pool = score_scene_candidates(
            candidates,
            adversary_index=_ADVERSARY,
            victim_index=_VICTIM,
            scene=scene,
            config=_config(),
        )
        assert float(pool.rewards[0]) > float(pool.rewards[1])


class TestBuildScenePairs:
    def test_pairs_shape_and_same_scene(self) -> None:
        pairs = build_scene_pairs(
            _model(),
            _validation_scene(road_half=100.0),
            adversary_index=_ADVERSARY,
            victim_index=_VICTIM,
            num_candidates=8,
            selection_pressure=0.5,
            num_pairs=3,
            config=_config(),
            feasibility_threshold=1.0,
            key=jax.random.key(4),
        )
        assert pairs is not None
        chosen, rejected, margins = pairs
        assert chosen.shape == (3, _SELECTED, _FUTURE, 4)
        assert rejected.shape == (3, _SELECTED, _FUTURE, 4)
        assert margins.shape == (3,)

    def test_chosen_outranks_rejected(self) -> None:
        pairs = build_scene_pairs(
            _model(),
            _validation_scene(road_half=100.0),
            adversary_index=_ADVERSARY,
            victim_index=_VICTIM,
            num_candidates=8,
            selection_pressure=0.75,
            num_pairs=3,
            config=_config(),
            feasibility_threshold=1.0,
            key=jax.random.key(5),
        )
        assert pairs is not None
        _, _, margins = pairs
        # Reward-ranked pairing: every chosen candidate outranks its rejected one.
        assert bool(jnp.all(margins >= 0.0))

    def test_no_feasible_returns_none(self) -> None:
        """An unreachable feasibility threshold gates every candidate out."""
        pairs = build_scene_pairs(
            _model(),
            _validation_scene(road_half=0.05),
            adversary_index=_ADVERSARY,
            victim_index=_VICTIM,
            num_candidates=6,
            selection_pressure=0.5,
            num_pairs=2,
            config=_config(),
            feasibility_threshold=-1.0,
            key=jax.random.key(6),
        )
        assert pairs is None
