"""Tests for the shared steering spine: sampling, scoring, gating, pairing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffav.alignment.steering_spine import (
    build_steering_pairs,
    candidate_offroad_fractions,
    CandidatePool,
    sample_and_score,
)
from diffav.core.geometry import RoadEdges
from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel
from tests.alignment.helpers import CONTEXT_DIM, FUTURE_STEPS, NUM_AGENTS


def _square_road() -> RoadEdges:
    """Counterclockwise 20x20 m square: inside is on-road."""
    square = np.array(
        [
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 10.0, 0.0],
            [-10.0, -10.0, 0.0],
        ]
    )
    return RoadEdges.from_polylines([square])


def _constant_position_candidates(positions: list[tuple[float, float]]) -> jax.Array:
    """One candidate per (x, y), every agent/step parked at that point."""
    candidates = []
    for x, y in positions:
        state = jnp.array([x, y, 0.0, 0.0])
        candidates.append(jnp.broadcast_to(state, (NUM_AGENTS, FUTURE_STEPS, 4)))
    return jnp.stack(candidates)


def _pool(
    rewards: list[float],
    feasible: list[bool],
) -> CandidatePool:
    """Pool with distinguishable trajectories (candidate index in x)."""
    num = len(rewards)
    trajectories = jnp.arange(num, dtype=jnp.float32)[:, None, None, None] * jnp.ones(
        (num, NUM_AGENTS, FUTURE_STEPS, 4)
    )
    return CandidatePool(
        trajectories=trajectories,
        rewards=jnp.array(rewards),
        feasible=jnp.array(feasible),
        offroad_fraction=jnp.where(jnp.array(feasible), 0.0, 1.0),
    )


def _candidate_index(trajectory: jax.Array) -> int:
    return int(trajectory[0, 0, 0])


class TestCandidateOffroadFractions:
    def test_inside_square_is_zero(self) -> None:
        fractions = candidate_offroad_fractions(
            _constant_position_candidates([(0.0, 0.0)]), _square_road()
        )
        assert float(fractions[0]) == 0.0

    def test_outside_square_is_one(self) -> None:
        fractions = candidate_offroad_fractions(
            _constant_position_candidates([(50.0, 50.0)]), _square_road()
        )
        assert float(fractions[0]) == 1.0

    def test_mixed_candidates_ordered(self) -> None:
        fractions = candidate_offroad_fractions(
            _constant_position_candidates([(0.0, 0.0), (50.0, 50.0)]), _square_road()
        )
        assert float(fractions[0]) < float(fractions[1])


class TestSampleAndScore:
    def test_pool_shapes(self, small_diffusion_model: TrajectoryDiffusionModel) -> None:
        pool = sample_and_score(
            small_diffusion_model,
            jnp.ones((NUM_AGENTS, CONTEXT_DIM)),
            num_agents=NUM_AGENTS,
            num_candidates=4,
            target_scenario="forward",
            reference_speed=10.0,
            road_edges=None,
            key=jax.random.key(0),
        )
        assert pool.trajectories.shape == (4, NUM_AGENTS, FUTURE_STEPS, 4)
        assert pool.rewards.shape == (4,)
        assert pool.feasible.shape == (4,)
        assert pool.offroad_fraction.shape == (4,)

    def test_no_road_edges_all_feasible(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        pool = sample_and_score(
            small_diffusion_model,
            jnp.ones((NUM_AGENTS, CONTEXT_DIM)),
            num_agents=NUM_AGENTS,
            num_candidates=3,
            target_scenario="forward",
            reference_speed=10.0,
            road_edges=None,
            key=jax.random.key(0),
        )
        assert bool(jnp.all(pool.feasible))
        assert float(jnp.max(pool.offroad_fraction)) == 0.0

    def test_candidates_differ_across_indices(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        pool = sample_and_score(
            small_diffusion_model,
            jnp.ones((NUM_AGENTS, CONTEXT_DIM)),
            num_agents=NUM_AGENTS,
            num_candidates=2,
            target_scenario="forward",
            reference_speed=10.0,
            road_edges=None,
            key=jax.random.key(0),
        )
        assert not jnp.array_equal(pool.trajectories[0], pool.trajectories[1])

    def test_deterministic_for_fixed_key(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        def run() -> CandidatePool:
            return sample_and_score(
                small_diffusion_model,
                jnp.ones((NUM_AGENTS, CONTEXT_DIM)),
                num_agents=NUM_AGENTS,
                num_candidates=2,
                target_scenario="forward",
                reference_speed=10.0,
                road_edges=None,
                key=jax.random.key(3),
            )

        first = run()
        second = run()
        assert jnp.array_equal(first.trajectories, second.trajectories)
        assert jnp.array_equal(first.rewards, second.rewards)

    def test_too_few_candidates_raises(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        with pytest.raises(ValueError, match="num_candidates"):
            sample_and_score(
                small_diffusion_model,
                jnp.ones((NUM_AGENTS, CONTEXT_DIM)),
                num_agents=NUM_AGENTS,
                num_candidates=1,
                target_scenario="forward",
                reference_speed=10.0,
                road_edges=None,
                key=jax.random.key(0),
            )


class TestBuildSteeringPairs:
    def test_chosen_outranks_rejected(self) -> None:
        pool = _pool([0.9, 0.1, 0.5, -0.3], [True, True, True, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=1.0, num_pairs=1, key=jax.random.key(0)
        )
        assert pairs is not None
        chosen, rejected, margins = pairs
        assert _candidate_index(chosen[0]) == 0  # reward 0.9
        assert _candidate_index(rejected[0]) == 3  # reward -0.3
        assert float(margins[0]) == pytest.approx(1.2)

    def test_infeasible_never_chosen(self) -> None:
        # Highest reward candidate is infeasible
        pool = _pool([0.9, 0.1, 0.5], [False, True, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=1.0, num_pairs=4, key=jax.random.key(0)
        )
        assert pairs is not None
        chosen, _, _ = pairs
        chosen_indices = {_candidate_index(chosen[i]) for i in range(4)}
        assert 0 not in chosen_indices

    def test_infeasible_preferred_as_rejected(self) -> None:
        pool = _pool([0.9, 0.1, 0.5], [True, False, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=1.0, num_pairs=1, key=jax.random.key(0)
        )
        assert pairs is not None
        _, rejected, _ = pairs
        assert _candidate_index(rejected[0]) == 1

    def test_no_feasible_returns_none(self) -> None:
        pool = _pool([0.9, 0.1], [False, False])
        assert (
            build_steering_pairs(pool, selection_pressure=0.5, num_pairs=1, key=jax.random.key(0))
            is None
        )

    def test_zero_pressure_pairs_randomly(self) -> None:
        pool = _pool([0.9, 0.1, 0.5, -0.3], [True, True, True, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=0.0, num_pairs=2, key=jax.random.key(0)
        )
        assert pairs is not None
        chosen, rejected, _ = pairs
        assert chosen.shape == (2, NUM_AGENTS, FUTURE_STEPS, 4)
        for i in range(2):
            assert _candidate_index(chosen[i]) != _candidate_index(rejected[i])

    def test_never_pairs_candidate_with_itself(self) -> None:
        # Wide pools (low pressure) overlap in the middle of the ranking
        pool = _pool([0.9, 0.1, 0.5], [True, True, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=0.01, num_pairs=6, key=jax.random.key(0)
        )
        assert pairs is not None
        chosen, rejected, _ = pairs
        for i in range(6):
            assert _candidate_index(chosen[i]) != _candidate_index(rejected[i])

    def test_num_pairs_shape(self) -> None:
        pool = _pool([0.9, 0.1], [True, True])
        pairs = build_steering_pairs(
            pool, selection_pressure=1.0, num_pairs=3, key=jax.random.key(0)
        )
        assert pairs is not None
        chosen, rejected, margins = pairs
        assert chosen.shape[0] == 3
        assert rejected.shape[0] == 3
        assert margins.shape == (3,)


class TestMakeSteeringGuidance:
    def test_returns_scalar_reward(self) -> None:
        from diffav.alignment.steering_spine import make_steering_guidance

        guidance = make_steering_guidance("forward", reference_speed=10.0)
        x_0 = jnp.ones((NUM_AGENTS, FUTURE_STEPS, 4))
        reward = guidance(x_0)
        assert reward.shape == ()
        assert bool(jnp.isfinite(reward))

    def test_gradient_is_finite_and_nonzero(self) -> None:
        from diffav.alignment.steering_spine import make_steering_guidance

        guidance = make_steering_guidance("forward", reference_speed=100.0)
        x_0 = jnp.linspace(0.0, 1.0, NUM_AGENTS * FUTURE_STEPS * 4).reshape(
            NUM_AGENTS, FUTURE_STEPS, 4
        )
        grad = jax.grad(guidance)(x_0)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0


_ADV_AGENTS = 4
_ADV_STEPS = 3
_ADVERSARY = 0
_VICTIM = 1


def _adv_x0(coords: list[tuple[float, float]]) -> jax.Array:
    """Local-frame x̂₀ ``(A, T, 4)`` with each agent parked at ``(x, y)``.

    Channels are ``[x, y, heading, speed]``; heading/speed are zero, so a zero
    reference pose makes the world frame equal the local frame.
    """
    states = jnp.stack([jnp.array([x, y, 0.0, 0.0]) for x, y in coords])
    return jnp.broadcast_to(states[:, None, :], (len(coords), _ADV_STEPS, 4))


def _zero_pose() -> jax.Array:
    return jnp.zeros((_ADV_AGENTS, 3))


class TestMakeAdversarialGuidance:
    """The adversarial guidance rewards approaching the victim, feasibly.

    Grounded in STRIVE/Safe-Sim: a soft-min-over-time distance-to-victim reward,
    an off-road penalty and a non-victim collision penalty on the adversary, and
    an adversary-only gradient so only the attacker is perturbed.
    """

    def _guidance(self, **overrides: object):  # noqa: ANN202 - test-local closure type
        from diffav.alignment.steering_spine import make_adversarial_guidance

        kwargs: dict[str, object] = {
            "adversary_index": _ADVERSARY,
            "victim_index": _VICTIM,
            "reference_pose": _zero_pose(),
            "offroad_weight": 0.0,
            "other_collision_weight": 0.0,
        }
        kwargs.update(overrides)
        return make_adversarial_guidance(**kwargs)  # type: ignore[arg-type]

    def test_reward_higher_when_adversary_closer_to_victim(self) -> None:
        """Approaching the victim raises the reward."""
        guidance = self._guidance()
        # Agents: 0=adversary, 1=victim at origin, 2/3 parked far away.
        near = _adv_x0([(1.0, 0.0), (0.0, 0.0), (50.0, 50.0), (60.0, 60.0)])
        far = _adv_x0([(20.0, 0.0), (0.0, 0.0), (50.0, 50.0), (60.0, 60.0)])
        assert float(guidance(near)) > float(guidance(far))

    def test_gradient_isolated_to_adversary(self) -> None:
        """Only the adversary's rows receive gradient (adversary-only steering)."""
        guidance = self._guidance(
            road_edges=_square_road(),
            offroad_weight=1.0,
            other_collision_weight=1.0,
            other_collision_margin=5.0,
        )
        x_0 = _adv_x0([(1.0, 0.0), (0.0, 0.0), (3.0, 0.0), (2.0, 2.0)])
        grad = jax.grad(guidance)(x_0)
        assert float(jnp.max(jnp.abs(grad[_ADVERSARY]))) > 0.0
        assert float(jnp.max(jnp.abs(grad[_VICTIM]))) == 0.0
        assert float(jnp.max(jnp.abs(grad[2]))) == 0.0
        assert float(jnp.max(jnp.abs(grad[3]))) == 0.0

    def test_offroad_penalty_lowers_reward(self) -> None:
        """An off-road adversary scores below the same geometry without edges."""
        # Victim at origin; adversary far outside the 20x20 square (off-road).
        coords = [(100.0, 100.0), (0.0, 0.0), (50.0, 50.0), (60.0, 60.0)]
        x_0 = _adv_x0(coords)
        without = self._guidance()(x_0)
        with_edges = self._guidance(road_edges=_square_road(), offroad_weight=1.0)(x_0)
        assert float(with_edges) < float(without)

    def test_non_victim_collision_penalizes(self) -> None:
        """Crowding a non-victim agent lowers the reward."""
        guidance = self._guidance(other_collision_weight=1.0, other_collision_margin=5.0)
        # Adversary equally near the victim in both; agent 2 near vs far.
        crowding = _adv_x0([(1.0, 0.0), (0.0, 0.0), (1.5, 0.0), (60.0, 60.0)])
        clear = _adv_x0([(1.0, 0.0), (0.0, 0.0), (50.0, 0.0), (60.0, 60.0)])
        assert float(guidance(crowding)) < float(guidance(clear))

    def test_valid_agents_mask_excludes_padding(self) -> None:
        """A padded (invalid) neighbour does not trigger the collision penalty."""
        crowding = _adv_x0([(1.0, 0.0), (0.0, 0.0), (1.5, 0.0), (60.0, 60.0)])
        counted = self._guidance(other_collision_weight=1.0, other_collision_margin=5.0)
        masked = self._guidance(
            other_collision_weight=1.0,
            other_collision_margin=5.0,
            valid_agents=jnp.array([True, True, False, True]),
        )
        # Masking the crowding neighbour removes its penalty, raising the reward.
        assert float(masked(crowding)) > float(counted(crowding))

    def test_reference_pose_maps_to_shared_frame(self) -> None:
        """World-frame distance (not local) drives the reward via the pose.

        Identical local positions, but a reference pose that translates the
        adversary 20 m away from the victim must reduce the reward.
        """
        from diffav.alignment.steering_spine import make_adversarial_guidance

        local = _adv_x0([(0.0, 0.0), (0.0, 0.0), (50.0, 50.0), (60.0, 60.0)])
        aligned_pose = jnp.zeros((_ADV_AGENTS, 3))
        separated_pose = aligned_pose.at[_ADVERSARY, 0].set(20.0)
        aligned = make_adversarial_guidance(
            adversary_index=_ADVERSARY, victim_index=_VICTIM, reference_pose=aligned_pose
        )
        separated = make_adversarial_guidance(
            adversary_index=_ADVERSARY, victim_index=_VICTIM, reference_pose=separated_pose
        )
        assert float(aligned(local)) > float(separated(local))

    def test_reward_is_finite_scalar(self) -> None:
        """The reward is a finite scalar with all terms active."""
        guidance = self._guidance(
            road_edges=_square_road(), offroad_weight=1.0, other_collision_weight=1.0
        )
        reward = guidance(_adv_x0([(1.0, 0.0), (0.0, 0.0), (3.0, 0.0), (2.0, 2.0)]))
        assert reward.shape == ()
        assert bool(jnp.isfinite(reward))
