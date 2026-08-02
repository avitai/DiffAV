"""Map-conditioned adversarial candidate spine over real validation scenes.

The offline counterpart to the test-time guidance loop: instead of steering a
single sample with x̂₀ guidance, draw a pool of full-scene rollouts from the
map-conditioned diffuser, rank them by the *same* grounded adversarial
victim-approach reward, gate feasibility on the scene's real road edges, and
emit reward-ranked chosen/rejected preference pairs for Diffusion-DPO.

Candidates are sampled and scored in the shared (world) frame — directly
comparable across agents and to the logged futures. The downstream DPO
log-probability re-projects the winning/losing sets into each agent's local
frame itself (:meth:`MapConditionedTrajectoryModel.denoising_log_prob`), so the
pairs stay world-frame here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simulacrax.alignment.steering_spine import (
    adversarial_victim_reward,
    AdversarialRewardConfig,
    build_steering_pairs,
    candidate_offroad_fractions,
    CandidatePool,
    draw_candidates,
)
from simulacrax.api.map_conditioned import MapConditionedTrajectoryModel
from simulacrax.evaluation.map_conditioned_evaluator import ValidationScene
from simulacrax.models.trajectory_diffusion import GuidanceSpec


def sample_scene_candidates(
    model: MapConditionedTrajectoryModel,
    scene: ValidationScene,
    *,
    num_candidates: int,
    guidance: GuidanceSpec | None = None,
    key: jax.Array,
) -> jax.Array:
    """Draw a pool of full-scene rollouts from the map-conditioned diffuser.

    Every candidate rolls out all of the scene's agents in the shared frame with
    a distinct ``split`` key. Candidates are selection material only
    (``stop_gradient``): the steering gradient flows through the downstream DPO
    loss on the selected pairs, never through the (non-differentiable) sampler.

    Args:
        model: The map-conditioned model to sample from.
        scene: The validation scene supplying the raw scene dict, anchor rows and
            per-agent reference pose.
        num_candidates: Candidates to draw (at least 2).
        guidance: Optional test-time x̂₀ guidance forwarded to the sampler.
        key: JAX random key.

    Returns:
        World-frame candidates, shape ``(num_candidates, num_agents,
        future_steps, 4)``.

    Raises:
        ValueError: If ``num_candidates`` < 2.
    """

    def sample_one(merged: MapConditionedTrajectoryModel, sample_key: jax.Array) -> jax.Array:
        return merged.sample(
            scene.scene,
            scene.agent_rows,
            reference_pose=scene.reference_pose,
            key=sample_key,
            guidance=guidance,
        ).trajectories

    keys = jax.random.split(key, num_candidates)
    return draw_candidates(model, sample_one, num_candidates=num_candidates, keys=keys)


def score_scene_candidates(
    candidates: jax.Array,
    *,
    adversary_index: int,
    victim_index: int,
    scene: ValidationScene,
    config: AdversarialRewardConfig,
    feasibility_threshold: float = 0.0,
) -> CandidatePool:
    """Score a candidate pool by the adversarial reward and gate feasibility.

    Ranking is by the grounded adversarial victim-approach reward alone (which
    already regularizes the adversary's off-road excursions and non-victim
    crowding internally) — a generic whole-scene safety reward is deliberately
    omitted because it would penalize the very adversary-to-victim approach being
    steered for. Feasibility gates on the *adversary's* off-road fraction against
    the scene's real road edges: the other agents follow the model's prior and
    must not disqualify a candidate.

    Args:
        candidates: World-frame candidates, shape ``(num_candidates, num_agents,
            future_steps, 4)``.
        adversary_index: Row of the steered adversary agent.
        victim_index: Row of the victim (target) agent.
        scene: The validation scene supplying the real road edges and validity.
        config: Adversarial reward weights.
        feasibility_threshold: Maximum allowed adversary off-road fraction.

    Returns:
        A scored, feasibility-gated :class:`CandidatePool`.
    """
    valid_agents = jnp.any(scene.valid.astype(bool), axis=1)

    def score_one(candidate: jax.Array) -> jax.Array:
        return adversarial_victim_reward(
            candidate[..., :2],
            adversary_index=adversary_index,
            victim_index=victim_index,
            road_edges=scene.road_edges,
            valid_agents=valid_agents,
            config=config,
        )

    rewards = jax.vmap(score_one)(candidates)

    adversary = candidates[:, adversary_index : adversary_index + 1]
    offroad_fraction = candidate_offroad_fractions(adversary, scene.road_edges)
    feasible = offroad_fraction <= feasibility_threshold

    return CandidatePool(
        trajectories=candidates,
        rewards=rewards,
        feasible=feasible,
        offroad_fraction=offroad_fraction,
    )


def build_scene_pairs(
    model: MapConditionedTrajectoryModel,
    scene: ValidationScene,
    *,
    adversary_index: int,
    victim_index: int,
    num_candidates: int,
    selection_pressure: float,
    num_pairs: int,
    config: AdversarialRewardConfig,
    feasibility_threshold: float = 0.0,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Sample, score and pair adversarial candidates for one scene.

    Composes :func:`sample_scene_candidates`, :func:`score_scene_candidates` and
    :func:`~simulacrax.alignment.steering_spine.build_steering_pairs` into the
    per-scene half of the ranked-DPO data pipeline. All pairs come from the same
    scene (they share ``model``/``scene``), so the downstream DPO loss conditions
    both branches on one encoded map.

    Args:
        model: The map-conditioned model to sample from.
        scene: The validation scene.
        adversary_index: Row of the steered adversary agent.
        victim_index: Row of the victim (target) agent.
        num_candidates: Candidates to draw (at least 2).
        selection_pressure: α ∈ [0, 1]; 0 pairs randomly (the no-steering
            baseline), larger tightens the chosen set toward the reward extreme.
        num_pairs: Preference pairs to emit for this scene.
        config: Adversarial reward weights.
        feasibility_threshold: Maximum allowed adversary off-road fraction.
        key: JAX random key (split for sampling and pairing).

    Returns:
        Tuple of (chosen ``(num_pairs, A, T, 4)``, rejected same shape, margins
        ``(num_pairs,)``) in the shared frame, or ``None`` when no valid pairing
        exists (e.g. no feasible candidate).
    """
    sample_key, pair_key = jax.random.split(key)
    candidates = sample_scene_candidates(
        model, scene, num_candidates=num_candidates, key=sample_key
    )
    pool = score_scene_candidates(
        candidates,
        adversary_index=adversary_index,
        victim_index=victim_index,
        scene=scene,
        config=config,
        feasibility_threshold=feasibility_threshold,
    )
    return build_steering_pairs(
        pool,
        selection_pressure=selection_pressure,
        num_pairs=num_pairs,
        key=pair_key,
    )
