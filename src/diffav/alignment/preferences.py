"""Reference preference-pair construction for DPO-based trajectory alignment.

Builds ranked preference pairs from candidate trajectories scored by a
reward function. The resulting :class:`PreferenceBatch` feeds a DPO training
loop via :meth:`PreferenceBatch.to_dpo_batch`.

This module is the **pedagogical / reference** preference builder used by the
alignment examples: general reward-ranking with no feasibility gate. The
**production** steering path is
:func:`diffav.alignment.steering_spine.build_steering_pairs` over a
:class:`~diffav.alignment.steering_spine.CandidatePool`, which adds
feasibility gating (an infeasible candidate can never be chosen) and
selection-pressure pairing. Prefer that path for map-conditioned steering;
use this module to learn or prototype preference construction.

.. note::
    ``PreferencePair``, ``PreferenceBatch``, and the generic scoring
    pipeline should migrate to
    ``artifex.generative_models.training.rl.data`` once stabilised.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from diffav.core.config import validate_positive


class RankingStrategy(enum.StrEnum):
    """Strategy for constructing preference pairs from ranked candidates."""

    BEST_VS_WORST = "best_vs_worst"
    ADJACENT = "adjacent"


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferencePairConfig:
    """Configuration for preference pair construction.

    Attributes:
        ranking_strategy: How to pair ranked candidates.
        num_candidates: Number of candidate trajectories per scene.
        top_k: Number of top/bottom pairs for ``BEST_VS_WORST``.
        min_reward_margin: Minimum reward gap to keep a pair.
    """

    ranking_strategy: RankingStrategy = RankingStrategy.BEST_VS_WORST
    num_candidates: int = 8
    top_k: int = 1
    min_reward_margin: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        validate_positive("num_candidates", self.num_candidates)
        validate_positive("top_k", self.top_k)
        if self.min_reward_margin < 0:
            msg = f"min_reward_margin must be non-negative, got {self.min_reward_margin}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferencePair:
    """A single chosen/rejected trajectory pair with reward metadata.

    Attributes:
        chosen: Preferred trajectory ``(num_agents, future_steps, state_dim)``.
        rejected: Non-preferred trajectory, same shape.
        chosen_reward: Scalar reward for the chosen trajectory.
        rejected_reward: Scalar reward for the rejected trajectory.
        reward_margin: ``chosen_reward - rejected_reward``.
        scene_context: Optional scene conditioning array.
    """

    chosen: jax.Array
    rejected: jax.Array
    chosen_reward: float
    rejected_reward: float
    reward_margin: float
    scene_context: jax.Array | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceBatch:
    """Batched preference pairs ready for DPO training.

    Attributes:
        chosen: Preferred trajectories ``(batch, num_agents, future_steps, state_dim)``.
        rejected: Non-preferred trajectories, same shape.
        chosen_rewards: Rewards for chosen ``(batch,)``.
        rejected_rewards: Rewards for rejected ``(batch,)``.
        margins: Reward margins ``(batch,)``.
        scene_contexts: Optional scene contexts ``(batch, ...)``.
    """

    chosen: jax.Array
    rejected: jax.Array
    chosen_rewards: jax.Array
    rejected_rewards: jax.Array
    margins: jax.Array
    scene_contexts: jax.Array | None = None

    def to_dpo_batch(self) -> dict[str, jax.Array]:
        """Convert to DPO trainer input format.

        Returns:
            Dict with ``"chosen"`` and ``"rejected"`` arrays, plus
            ``"scene_contexts"`` when available. Matches the expected
            format for ``DPOAlignmentTrainer.train_step(batch)``.
        """
        batch: dict[str, jax.Array] = {"chosen": self.chosen, "rejected": self.rejected}
        if self.scene_contexts is not None:
            batch["scene_contexts"] = self.scene_contexts
        return batch


class PreferencePairBuilder:
    """Builds preference pairs from candidate trajectories and a reward function.

    The pipeline is: ``score_candidates`` -> ``rank_candidates`` ->
    ``build_pairs`` -> stack into ``PreferenceBatch``. Scoring and ranking
    use pure JAX ops (JIT-compatible). Pair construction is Python-level
    due to variable-length filtering.

    .. note::
        The generic score-rank-pair pipeline should migrate to artifex
        once the API stabilises.
    """

    def __init__(
        self,
        reward_fn: Callable[..., jax.Array],
        config: PreferencePairConfig | None = None,
    ) -> None:
        """Initialize with a reward function and optional configuration.

        Args:
            reward_fn: Callable satisfying the ``RewardFunction`` protocol.
            config: Pair construction config. Uses defaults if ``None``.
        """
        self.reward_fn = reward_fn
        self.config = config or PreferencePairConfig()

    def score_candidates(
        self,
        candidates: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Score candidate trajectories using the reward function.

        Args:
            candidates: ``(num_candidates, num_agents, future_steps, state_dim)``.
            conditions: Optional conditioning passed to reward function.
            **kwargs: Extra kwargs passed to reward function.

        Returns:
            Rewards ``(num_candidates,)``.
        """
        return self.reward_fn(candidates, conditions, **kwargs)

    def rank_candidates(
        self,
        candidates: jax.Array,
        rewards: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sort candidates by reward in descending order.

        Args:
            candidates: ``(num_candidates, ...)``.
            rewards: ``(num_candidates,)``.

        Returns:
            Tuple of (sorted_candidates, sorted_rewards), best first.
        """
        # argsort ascending, then reverse for descending
        order = jnp.argsort(-rewards)
        return candidates[order], rewards[order]

    def build_pairs(
        self,
        candidates: jax.Array,
        rewards: jax.Array,
        scene_context: jax.Array | None = None,
    ) -> list[PreferencePair]:
        """Build preference pairs from sorted candidates.

        Args:
            candidates: Sorted candidates ``(num_candidates, ...)``, best first.
            rewards: Sorted rewards ``(num_candidates,)``, descending.
            scene_context: Optional scene context array.

        Returns:
            List of preference pairs after margin filtering.
        """
        num = candidates.shape[0]
        pairs: list[PreferencePair] = []

        def _maybe_add_pair(chosen_idx: int, rejected_idx: int) -> None:
            chosen_r = float(rewards[chosen_idx])
            rejected_r = float(rewards[rejected_idx])
            margin = chosen_r - rejected_r
            if margin >= self.config.min_reward_margin:
                pairs.append(
                    PreferencePair(
                        chosen=candidates[chosen_idx],
                        rejected=candidates[rejected_idx],
                        chosen_reward=chosen_r,
                        rejected_reward=rejected_r,
                        reward_margin=margin,
                        scene_context=scene_context,
                    )
                )

        if self.config.ranking_strategy == RankingStrategy.BEST_VS_WORST:
            k = min(self.config.top_k, num // 2)
            for i in range(k):
                _maybe_add_pair(i, num - 1 - i)
        elif self.config.ranking_strategy == RankingStrategy.ADJACENT:
            for i in range(0, num - 1, 2):
                _maybe_add_pair(i, i + 1)

        return pairs

    def build_batch(
        self,
        candidates: jax.Array,
        conditions: jax.Array | None = None,
        scene_context: jax.Array | None = None,
        **kwargs: Any,
    ) -> PreferenceBatch:
        """Full pipeline: score, rank, pair, and stack into a batch.

        Args:
            candidates: ``(num_candidates, num_agents, future_steps, state_dim)``.
            conditions: Optional conditioning for reward function.
            scene_context: Optional scene context array.
            **kwargs: Extra kwargs passed to reward function.

        Returns:
            Stacked preference batch.

        Raises:
            ValueError: If fewer than 2 candidates are provided.
        """
        if candidates.shape[0] < 2:
            msg = f"At least 2 candidates required, got {candidates.shape[0]}"
            raise ValueError(msg)

        scores = self.score_candidates(candidates, conditions, **kwargs)
        sorted_candidates, sorted_rewards = self.rank_candidates(candidates, scores)
        pairs = self.build_pairs(sorted_candidates, sorted_rewards, scene_context)
        return _stack_pairs(pairs)

    def build_batch_from_scenes(
        self,
        candidates_per_scene: Sequence[jax.Array],
        conditions_per_scene: Sequence[jax.Array | None],
        scene_contexts: Sequence[jax.Array | None] | None = None,
        **kwargs: Any,
    ) -> PreferenceBatch:
        """Build a preference batch from multiple scenes.

        Args:
            candidates_per_scene: List of candidate arrays, one per scene.
            conditions_per_scene: List of condition arrays, one per scene.
            scene_contexts: Optional list of scene context arrays.
            **kwargs: Extra kwargs passed to reward function.

        Returns:
            Combined preference batch across all scenes.
        """
        all_pairs: list[PreferencePair] = []
        for i, (candidates, conditions) in enumerate(
            zip(candidates_per_scene, conditions_per_scene)
        ):
            ctx = scene_contexts[i] if scene_contexts is not None else None
            scores = self.score_candidates(candidates, conditions, **kwargs)
            sorted_c, sorted_r = self.rank_candidates(candidates, scores)
            pairs = self.build_pairs(sorted_c, sorted_r, scene_context=ctx)
            all_pairs.extend(pairs)
        return _stack_pairs(all_pairs)


def _stack_pairs(pairs: list[PreferencePair]) -> PreferenceBatch:
    """Stack a list of PreferencePair into a PreferenceBatch.

    Args:
        pairs: Non-empty list of preference pairs.

    Returns:
        Batched preference data.

    Raises:
        ValueError: If the pairs list is empty.
    """
    if not pairs:
        msg = "Cannot stack an empty list of preference pairs"
        raise ValueError(msg)
    has_context = pairs[0].scene_context is not None
    return PreferenceBatch(
        chosen=jnp.stack([p.chosen for p in pairs]),
        rejected=jnp.stack([p.rejected for p in pairs]),
        chosen_rewards=jnp.array([p.chosen_reward for p in pairs]),
        rejected_rewards=jnp.array([p.rejected_reward for p in pairs]),
        margins=jnp.array([p.reward_margin for p in pairs]),
        scene_contexts=(
            jnp.stack([p.scene_context for p in pairs])  # type: ignore[arg-type]
            if has_context
            else None
        ),
    )
