"""Tests for preference pair construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from simulacrax.alignment.preferences import (
    PreferenceBatch,
    PreferencePair,
    PreferencePairBuilder,
    PreferencePairConfig,
    RankingStrategy,
)
from simulacrax.alignment.rewards import CollisionReward, SafetyReward
from tests.alignment.helpers import FUTURE_STEPS, NUM_AGENTS, STATE_DIM


# ---------------------------------------------------------------------------
# RankingStrategy
# ---------------------------------------------------------------------------


class TestRankingStrategy:
    """Tests for RankingStrategy enum."""

    def test_values(self) -> None:
        """Enum has expected members."""
        assert RankingStrategy.BEST_VS_WORST == "best_vs_worst"
        assert RankingStrategy.ADJACENT == "adjacent"


# ---------------------------------------------------------------------------
# PreferencePairConfig
# ---------------------------------------------------------------------------


class TestPreferencePairConfig:
    """Tests for PreferencePairConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = PreferencePairConfig()
        assert cfg.ranking_strategy == RankingStrategy.BEST_VS_WORST
        assert cfg.num_candidates == 8
        assert cfg.top_k == 1
        assert cfg.min_reward_margin == 0.0

    def test_negative_num_candidates_raises(self) -> None:
        """Non-positive num_candidates raises ValueError."""
        with pytest.raises(ValueError, match="num_candidates"):
            PreferencePairConfig(num_candidates=0)

    def test_negative_top_k_raises(self) -> None:
        """Non-positive top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k"):
            PreferencePairConfig(top_k=0)

    def test_negative_min_margin_raises(self) -> None:
        """Negative min_reward_margin raises ValueError."""
        with pytest.raises(ValueError, match="min_reward_margin"):
            PreferencePairConfig(min_reward_margin=-0.1)


# ---------------------------------------------------------------------------
# PreferencePair
# ---------------------------------------------------------------------------


class TestPreferencePair:
    """Tests for PreferencePair data container."""

    def test_frozen(self) -> None:
        """PreferencePair is immutable."""
        pair = PreferencePair(
            chosen=jnp.ones((NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            rejected=jnp.zeros((NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            chosen_reward=1.0,
            rejected_reward=-1.0,
            reward_margin=2.0,
        )
        with pytest.raises(AttributeError):
            pair.chosen_reward = 0.5  # type: ignore[misc]

    def test_fields_correct(self) -> None:
        """Fields store expected values."""
        chosen = jnp.ones((NUM_AGENTS, FUTURE_STEPS, STATE_DIM))
        rejected = jnp.zeros((NUM_AGENTS, FUTURE_STEPS, STATE_DIM))
        pair = PreferencePair(
            chosen=chosen,
            rejected=rejected,
            chosen_reward=1.5,
            rejected_reward=-0.5,
            reward_margin=2.0,
            scene_context=jnp.ones((10, 64)),
        )
        assert pair.chosen.shape == (NUM_AGENTS, FUTURE_STEPS, STATE_DIM)
        assert pair.rejected.shape == (NUM_AGENTS, FUTURE_STEPS, STATE_DIM)
        assert pair.chosen_reward == 1.5
        assert pair.rejected_reward == -0.5
        assert pair.reward_margin == 2.0
        assert pair.scene_context is not None
        assert pair.scene_context.shape == (10, 64)


# ---------------------------------------------------------------------------
# PreferenceBatch
# ---------------------------------------------------------------------------


class TestPreferenceBatch:
    """Tests for PreferenceBatch data container."""

    def _make_batch(self, size: int = 4) -> PreferenceBatch:
        """Create a test batch."""
        return PreferenceBatch(
            chosen=jnp.ones((size, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            rejected=jnp.zeros((size, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            chosen_rewards=jnp.array([1.0, 0.5, 0.3, 0.1][:size]),
            rejected_rewards=jnp.array([-1.0, -0.5, -0.3, -0.1][:size]),
            margins=jnp.array([2.0, 1.0, 0.6, 0.2][:size]),
        )

    def test_to_dpo_batch_format(self) -> None:
        """to_dpo_batch returns expected dict format."""
        batch = self._make_batch()
        dpo = batch.to_dpo_batch()
        assert "chosen" in dpo
        assert "rejected" in dpo
        assert dpo["chosen"].shape == (4, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)
        assert dpo["rejected"].shape == (4, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)

    def test_to_dpo_batch_values(self) -> None:
        """to_dpo_batch preserves data correctly."""
        batch = self._make_batch()
        dpo = batch.to_dpo_batch()
        assert float(jnp.sum(dpo["chosen"])) > 0
        assert float(jnp.sum(dpo["rejected"])) == 0.0

    def test_frozen(self) -> None:
        """PreferenceBatch is immutable."""
        batch = self._make_batch()
        with pytest.raises(AttributeError):
            batch.margins = jnp.zeros(4)  # type: ignore[misc]

    def test_with_scene_contexts(self) -> None:
        """PreferenceBatch can hold scene contexts."""
        batch = PreferenceBatch(
            chosen=jnp.ones((2, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            rejected=jnp.zeros((2, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            chosen_rewards=jnp.array([1.0, 0.5]),
            rejected_rewards=jnp.array([-1.0, -0.5]),
            margins=jnp.array([2.0, 1.0]),
            scene_contexts=jnp.ones((2, 10, 64)),
        )
        assert batch.scene_contexts is not None
        assert batch.scene_contexts.shape == (2, 10, 64)

    def test_to_dpo_batch_includes_scene_contexts(self) -> None:
        """to_dpo_batch includes scene_contexts when present."""
        batch = PreferenceBatch(
            chosen=jnp.ones((2, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            rejected=jnp.zeros((2, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)),
            chosen_rewards=jnp.array([1.0, 0.5]),
            rejected_rewards=jnp.array([-1.0, -0.5]),
            margins=jnp.array([2.0, 1.0]),
            scene_contexts=jnp.ones((2, 10, 64)),
        )
        dpo = batch.to_dpo_batch()
        assert "scene_contexts" in dpo
        assert dpo["scene_contexts"].shape == (2, 10, 64)

    def test_to_dpo_batch_excludes_scene_contexts_when_none(self) -> None:
        """to_dpo_batch omits scene_contexts when not set."""
        batch = self._make_batch()
        dpo = batch.to_dpo_batch()
        assert "scene_contexts" not in dpo


# ---------------------------------------------------------------------------
# PreferencePairBuilder
# ---------------------------------------------------------------------------


class TestPreferencePairBuilder:
    """Tests for PreferencePairBuilder."""

    def _make_candidates(self, num_candidates: int = 8) -> jax.Array:
        """Create candidate trajectories with varying collision levels.

        Each candidate has agents at different y-separations so scores differ.
        Shape: ``(num_candidates, NUM_AGENTS, FUTURE_STEPS, STATE_DIM)``.
        """
        dt = 0.1
        t = jnp.arange(FUTURE_STEPS) * dt
        candidates = []
        for i in range(num_candidates):
            # Vary y-separation: more separation = safer = higher reward
            separation = (i + 1) * 3.0
            x = jnp.broadcast_to(t * 5.0, (NUM_AGENTS, FUTURE_STEPS))
            y_offsets = jnp.arange(NUM_AGENTS)[:, None] * separation
            y = jnp.zeros((NUM_AGENTS, FUTURE_STEPS)) + y_offsets
            heading = jnp.zeros((NUM_AGENTS, FUTURE_STEPS))
            velocity = jnp.full((NUM_AGENTS, FUTURE_STEPS), 5.0)
            traj = jnp.stack([x, y, heading, velocity], axis=-1)
            candidates.append(traj)
        return jnp.stack(candidates)

    def test_score_candidates_shape(self) -> None:
        """score_candidates returns shape (num_candidates,)."""
        reward = CollisionReward()
        builder = PreferencePairBuilder(reward_fn=reward)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        assert scores.shape == (8,)

    def test_score_candidates_jit(self) -> None:
        """score_candidates is JIT-compatible."""
        reward = CollisionReward()
        builder = PreferencePairBuilder(reward_fn=reward)
        candidates = self._make_candidates(8)

        @jax.jit
        def score(c: jax.Array) -> jax.Array:
            return builder.score_candidates(c, conditions=None)

        result = score(candidates)
        assert result.shape == (8,)

    def test_rank_candidates_descending(self) -> None:
        """rank_candidates sorts by reward descending."""
        reward = CollisionReward()
        builder = PreferencePairBuilder(reward_fn=reward)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        sorted_candidates, sorted_rewards = builder.rank_candidates(candidates, scores)
        # Rewards should be monotonically non-increasing
        for i in range(len(sorted_rewards) - 1):
            assert float(sorted_rewards[i]) >= float(sorted_rewards[i + 1])
        assert sorted_candidates.shape == candidates.shape

    def test_rank_candidates_jit(self) -> None:
        """rank_candidates is JIT-compatible."""
        reward = CollisionReward()
        builder = PreferencePairBuilder(reward_fn=reward)
        candidates = self._make_candidates(8)
        scores = jnp.array([0.0, -1.0, -0.5, -2.0, -0.1, -3.0, -0.8, -1.5])

        @jax.jit
        def rank(c: jax.Array, r: jax.Array) -> tuple[jax.Array, jax.Array]:
            return builder.rank_candidates(c, r)

        sorted_c, sorted_r = rank(candidates, scores)
        assert sorted_c.shape == candidates.shape

    def test_build_pairs_best_vs_worst(self) -> None:
        """BEST_VS_WORST pairs top-k with bottom-k."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.BEST_VS_WORST,
            num_candidates=8,
            top_k=2,
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        sorted_c, sorted_r = builder.rank_candidates(candidates, scores)
        pairs = builder.build_pairs(sorted_c, sorted_r)

        # top_k=2 should produce 2 pairs
        assert len(pairs) == 2
        for pair in pairs:
            assert pair.chosen_reward >= pair.rejected_reward
            assert pair.reward_margin >= 0.0

    def test_build_pairs_adjacent(self) -> None:
        """ADJACENT pairs consecutive positions."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.ADJACENT,
            num_candidates=8,
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        sorted_c, sorted_r = builder.rank_candidates(candidates, scores)
        pairs = builder.build_pairs(sorted_c, sorted_r)

        # Adjacent: 8 candidates => 4 pairs
        assert len(pairs) == 4
        for pair in pairs:
            assert pair.chosen_reward >= pair.rejected_reward

    def test_build_pairs_min_margin_filter(self) -> None:
        """min_reward_margin filters out pairs with small margins."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.ADJACENT,
            num_candidates=8,
            min_reward_margin=100.0,  # Very high margin => filter most
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        sorted_c, sorted_r = builder.rank_candidates(candidates, scores)
        pairs = builder.build_pairs(sorted_c, sorted_r)

        # Very high margin should filter out most/all pairs
        assert len(pairs) == 0

    def test_build_batch_shapes(self) -> None:
        """build_batch returns PreferenceBatch with correct shapes."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.BEST_VS_WORST,
            num_candidates=8,
            top_k=2,
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        batch = builder.build_batch(candidates, conditions=None)

        assert batch.chosen.shape[0] == 2  # top_k=2
        assert batch.rejected.shape[0] == 2
        assert batch.chosen_rewards.shape == (2,)
        assert batch.rejected_rewards.shape == (2,)
        assert batch.margins.shape == (2,)

    def test_build_batch_from_scenes(self) -> None:
        """build_batch_from_scenes stacks multiple scenes."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.BEST_VS_WORST,
            num_candidates=8,
            top_k=1,
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        scene1 = self._make_candidates(8)
        scene2 = self._make_candidates(8)
        batch = builder.build_batch_from_scenes(
            candidates_per_scene=[scene1, scene2],
            conditions_per_scene=[None, None],
        )

        # 2 scenes * top_k=1 = 2 pairs
        assert batch.chosen.shape[0] == 2
        assert batch.rejected.shape[0] == 2

    def test_build_batch_to_dpo(self) -> None:
        """Full pipeline: build_batch -> to_dpo_batch."""
        reward = SafetyReward()
        cfg = PreferencePairConfig(
            ranking_strategy=RankingStrategy.BEST_VS_WORST,
            num_candidates=8,
            top_k=1,
        )
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        batch = builder.build_batch(candidates, conditions=None)
        dpo = batch.to_dpo_batch()

        assert "chosen" in dpo
        assert "rejected" in dpo
        assert dpo["chosen"].shape[0] == 1

    def test_works_with_any_reward_function(self) -> None:
        """Builder works with any callable matching RewardFunction protocol."""

        class CustomReward:
            def __call__(
                self,
                samples: jax.Array,
                conditions: jax.Array | None = None,
                **kwargs: object,
            ) -> jax.Array:
                del conditions, kwargs
                return jnp.sum(samples, axis=(1, 2, 3))

        candidates = self._make_candidates(4)
        cfg = PreferencePairConfig(num_candidates=4, top_k=1)
        builder = PreferencePairBuilder(reward_fn=CustomReward(), config=cfg)
        batch = builder.build_batch(candidates, conditions=None)
        assert batch.chosen.shape[0] == 1

    def test_too_few_candidates(self) -> None:
        """Fewer than 2 candidates raises ValueError."""
        reward = CollisionReward()
        builder = PreferencePairBuilder(reward_fn=reward)
        candidates = self._make_candidates(1)
        with pytest.raises(ValueError, match="[Aa]t least 2"):
            builder.build_batch(candidates, conditions=None)

    def test_build_pairs_with_scene_context(self) -> None:
        """build_pairs passes scene_context to PreferencePair."""
        reward = CollisionReward()
        cfg = PreferencePairConfig(top_k=1)
        builder = PreferencePairBuilder(reward_fn=reward, config=cfg)
        candidates = self._make_candidates(8)
        scores = builder.score_candidates(candidates, conditions=None)
        sorted_c, sorted_r = builder.rank_candidates(candidates, scores)
        context = jnp.ones((10, 64))
        pairs = builder.build_pairs(sorted_c, sorted_r, scene_context=context)

        assert len(pairs) >= 1
        assert pairs[0].scene_context is not None
        assert pairs[0].scene_context.shape == (10, 64)
