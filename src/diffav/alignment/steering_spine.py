"""Shared steering spine: sample candidates, score rewards, gate feasibility.

Every steering strategy couples the reward to the model through samples the
model itself generated. This module provides that shared pipeline: draw
``num_candidates`` trajectories per scene context, score each with the
scenario reward (plus the composite safety reward), gate feasibility on the
signed distance to road edges, and construct chosen/rejected preference
pairs from the ranked pool.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from diffav.alignment.rewards import SafetyReward
from diffav.core.geometry import (
    from_agent_frame,
    mean_offroad_penalty,
    RoadEdges,
    signed_distances_to_road_edges,
)
from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidatePool:
    """Scored and feasibility-gated candidate trajectories for one scene.

    Attributes:
        trajectories: Sampled candidates, shape ``(num_candidates, num_agents,
            future_steps, 4)``.
        rewards: Combined score per candidate — the weighted sum of the
            rank-normalized scenario and safety rewards — shape
            ``(num_candidates,)``. Higher is better; only the ordering is
            meaningful.
        feasible: Whether each candidate's off-road fraction is within the
            feasibility threshold, shape ``(num_candidates,)``.
        offroad_fraction: Fraction of (agent, step) positions with positive
            signed distance to the road edges, shape ``(num_candidates,)``.
            Zero everywhere when no road edges were provided.
    """

    trajectories: jax.Array
    rewards: jax.Array
    feasible: jax.Array
    offroad_fraction: jax.Array


def _rank01(values: jax.Array) -> jax.Array:
    """Map values to their normalized ranks in [0, 1] (1 = highest value).

    Args:
        values: Scores, shape ``(num_candidates,)`` with at least 2 entries.

    Returns:
        Normalized ranks, shape ``(num_candidates,)``.
    """
    order = jnp.argsort(values)
    ranks = jnp.zeros_like(values).at[order].set(jnp.arange(values.shape[0], dtype=values.dtype))
    return ranks / (values.shape[0] - 1)


def candidate_offroad_fractions(
    trajectories: jax.Array,
    road_edges: RoadEdges,
) -> jax.Array:
    """Fraction of off-road position samples per candidate.

    Args:
        trajectories: Candidates, shape ``(num_candidates, num_agents,
            future_steps, state_dim)`` with ``[x, y, ...]`` leading state.
        road_edges: Oriented road-edge polylines (counterclockwise winding,
            positive signed distance = off-road).

    Returns:
        Off-road fractions in [0, 1], shape ``(num_candidates,)``.
    """
    positions = trajectories[..., :2]
    flat = positions.reshape(positions.shape[0], -1, 2)

    def per_candidate(points: jax.Array) -> jax.Array:
        signed = signed_distances_to_road_edges(points, road_edges)
        return jnp.mean((signed > 0.0).astype(jnp.float32))

    return jax.vmap(per_candidate)(flat)


def draw_candidates(
    model: nnx.Module,
    sample_fn: Callable[[Any, jax.Array], jax.Array],
    *,
    num_candidates: int,
    keys: jax.Array,
) -> jax.Array:
    """Vmap a per-candidate sampler over keys, detached as selection material.

    Splits the model once, then re-merges it inside a vmapped closure so each
    candidate samples independently. The pool is ``stop_gradient``-ed: gradients
    flow through the downstream DPO loss on the selected pairs, never through the
    (non-differentiable) sampler. Callers supply the model-specific sampling call
    and the per-candidate keys (the two steering models differ in both).

    Args:
        model: The diffusion model to sample from.
        sample_fn: Maps a re-merged model and one key to a candidate's
            trajectories, shape ``(num_agents, future_steps, 4)``.
        num_candidates: Candidates to draw (at least 2).
        keys: Per-candidate keys, shape ``(num_candidates, ...)``.

    Returns:
        Detached candidates, shape ``(num_candidates, num_agents, future_steps,
        4)``.

    Raises:
        ValueError: If ``num_candidates`` < 2.
    """
    if num_candidates < 2:
        msg = f"num_candidates must be at least 2, got {num_candidates}"
        raise ValueError(msg)

    graphdef, state = nnx.split(model)

    def sample_one(sample_key: jax.Array) -> jax.Array:
        merged = nnx.merge(graphdef, state)
        return sample_fn(merged, sample_key)

    return jax.lax.stop_gradient(jax.vmap(sample_one)(keys))


def sample_and_score(
    model: TrajectoryDiffusionModel,
    context_arr: jax.Array,
    *,
    num_agents: int,
    num_candidates: int,
    target_scenario: str,
    reference_speed: float,
    road_edges: RoadEdges | None,
    feasibility_threshold: float = 0.0,
    scenario_weight: float = 1.0,
    safety_weight: float = 1.0,
    safety_reward: SafetyReward | None = None,
    key: jax.Array,
) -> CandidatePool:
    """Sample candidates from the model and score them for steering.

    Candidates are sampled with per-candidate ``fold_in`` keys and treated as
    selection material only (``stop_gradient``): gradients flow through the
    downstream DPO loss evaluated on the selected pairs, not through the
    sampling process.

    Args:
        model: Diffusion model to sample from.
        context_arr: Per-agent scene context, shape ``(num_agents,
            context_dim)`` — one row per agent; the sampled agent count is
            inferred from ``context_arr.shape[0]``.
        num_agents: Agents per candidate scene (matches ``context_arr``'s
            row count).
        num_candidates: Candidates to draw (at least 2).
        target_scenario: Scenario label for the rule-based reward.
        reference_speed: Reference speed in m/s for reward normalisation.
        road_edges: Road edges for the feasibility gate, or ``None`` when the
            scene carries no map (all candidates then count as feasible).
        feasibility_threshold: Maximum allowed off-road fraction.
        scenario_weight: Weight of the scenario reward in the combined score.
        safety_weight: Weight of the composite safety reward.
        safety_reward: Optional pre-built safety reward (a default is created
            otherwise).
        key: JAX random key.

    Returns:
        Scored and gated ``CandidatePool``.

    Raises:
        ValueError: If ``num_candidates`` < 2.
    """
    # Imported here to avoid a module-level cycle: scenario_steering imports
    # the spine for its trainer, while the reward stays defined there.
    from diffav.alignment.scenario_steering import compute_scenario_reward  # noqa: PLC0415

    keys = jnp.stack([jax.random.fold_in(key, i) for i in range(num_candidates)])
    trajectories = draw_candidates(
        model,
        lambda merged, sample_key: merged.sample(context_arr, key=sample_key).trajectories,
        num_candidates=num_candidates,
        keys=keys,
    )

    scenario_rewards = jax.vmap(
        lambda t: compute_scenario_reward(t, target_scenario, reference_speed=reference_speed)
    )(trajectories)
    safety = safety_reward if safety_reward is not None else SafetyReward()
    safety_rewards = safety(trajectories)
    # Combine rank-normalized components: the scenario reward is bounded in
    # [-1, 1] while the safety composite is unbounded (large negative on
    # heavily violating samples), so raw addition lets safety drown the
    # steering target. Ranks are scale-free.
    rewards = scenario_weight * _rank01(scenario_rewards) + safety_weight * _rank01(safety_rewards)

    if road_edges is None:
        logger.warning("sample_and_score called without road_edges; feasibility gate is inactive.")
        offroad_fraction = jnp.zeros(num_candidates)
        feasible = jnp.ones(num_candidates, dtype=bool)
    else:
        offroad_fraction = candidate_offroad_fractions(trajectories, road_edges)
        feasible = offroad_fraction <= feasibility_threshold

    return CandidatePool(
        trajectories=trajectories,
        rewards=rewards,
        feasible=feasible,
        offroad_fraction=offroad_fraction,
    )


def make_steering_guidance(
    target_scenario: str,
    *,
    reference_speed: float,
    scenario_weight: float = 1.0,
    safety_weight: float = 1.0,
    safety_reward: SafetyReward | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Build a differentiable reward of x̂₀ for test-time guidance.

    The returned callable maps clean trajectory estimates
    ``(num_agents, future_steps, 4)`` to a scalar reward — the weighted sum
    of the scenario reward and the composite safety reward — suitable for
    ``TrajectoryDiffusionModel.sample(guidance_fn=...)``. Raw (unranked)
    rewards are used here because guidance needs a continuous gradient of a
    single sample, not a pool ordering; ``guidance_scale`` calibrates
    against the resulting gradient magnitude.

    Args:
        target_scenario: Scenario label for the rule-based reward.
        reference_speed: Reference speed in m/s for reward normalisation.
        scenario_weight: Weight of the scenario reward.
        safety_weight: Weight of the composite safety reward.
        safety_reward: Optional pre-built safety reward.

    Returns:
        Differentiable scalar reward function of x̂₀.
    """
    # Imported here to avoid a module-level cycle (see sample_and_score).
    from diffav.alignment.scenario_steering import compute_scenario_reward  # noqa: PLC0415

    safety = safety_reward if safety_reward is not None else SafetyReward()

    def guidance(x_0: jax.Array) -> jax.Array:
        scenario = compute_scenario_reward(x_0, target_scenario, reference_speed=reference_speed)
        return scenario_weight * scenario + safety_weight * safety(x_0[None])[0]

    return guidance


@dataclass(frozen=True, slots=True, kw_only=True)
class AdversarialRewardConfig:
    """Weights of the grounded adversarial victim-approach reward.

    Shared by the test-time guidance closure and the offline candidate scorer so
    both rank trajectories by the identical objective.

    Attributes:
        offroad_weight: Weight of the adversary off-road squared-hinge penalty.
        other_collision_weight: Weight of the non-victim crowding penalty.
        other_collision_margin: Distance (m) below which the adversary is
            penalized for crowding another (non-victim, valid) agent.
        softmin_temperature: Temperature of the soft-min over time (smaller is
            closer to a hard minimum).
    """

    offroad_weight: float = 1.0
    other_collision_weight: float = 1.0
    other_collision_margin: float = 2.0
    softmin_temperature: float = 1.0


def _safe_distance(offsets: jax.Array) -> jax.Array:
    """Euclidean norm with an epsilon under the root.

    The raw norm's gradient is undefined at a zero offset — which happens both
    for the adversary versus itself and, crucially, when the adversary reaches
    the victim (the steering target) — and would poison the gradient with NaNs.

    Args:
        offsets: Offset vectors with the coordinate axis last.

    Returns:
        Distances with the coordinate axis reduced.
    """
    return jnp.sqrt(jnp.sum(offsets**2, axis=-1) + 1e-12)


def adversarial_victim_reward(
    world_positions: jax.Array,
    *,
    adversary_index: int,
    victim_index: int,
    road_edges: RoadEdges | None = None,
    valid_agents: jax.Array | None = None,
    config: AdversarialRewardConfig,
) -> jax.Array:
    """Scalar adversarial victim-approach reward on shared-frame positions.

    The grounded reward (STRIVE / Safe-Sim / CTG++) of one *adversary* agent
    steered toward one *victim* while staying drivable. It reads the adversary's
    rows with gradient and treats every other agent as a fixed obstacle
    (``stop_gradient``), so a differentiable caller perturbs the adversary alone;
    an offline caller passing already-detached candidates is unaffected (the
    extra ``stop_gradient`` is a no-op there). See
    :func:`make_adversarial_guidance` for the term-by-term grounding.

    Args:
        world_positions: Shared-frame ``(x, y)`` positions of every agent, shape
            ``(num_agents, future_steps, 2)``.
        adversary_index: Row of the steered adversary agent.
        victim_index: Row of the victim (target) agent.
        road_edges: Oriented road edges for the off-road term, or ``None`` to
            drop it.
        valid_agents: Optional boolean ``(num_agents,)`` mask restricting the
            non-victim collision term to valid (non-padded) agents.
        config: Reward weights and soft-min temperature.

    Returns:
        Scalar reward; higher means a more critical, still-drivable approach.
    """
    positions = world_positions
    adversary = positions[adversary_index]
    # Every other agent is a fixed obstacle: no gradient flows into it, so a
    # differentiable caller perturbs only the adversary's trajectory.
    frozen = jax.lax.stop_gradient(positions)
    victim = frozen[victim_index]

    # Adversarial term: soft-min-over-time distance to the victim, negated so
    # that approaching the victim increases the reward.
    victim_distance = _safe_distance(adversary - victim)
    softmin_weights = jax.nn.softmax(-victim_distance / config.softmin_temperature)
    reward = -jnp.sum(softmin_weights * victim_distance)

    if road_edges is not None:
        reward -= config.offroad_weight * mean_offroad_penalty(adversary, road_edges)

    num_agents = positions.shape[0]
    agent_ids = jnp.arange(num_agents)
    others_mask = (agent_ids != adversary_index) & (agent_ids != victim_index)
    if valid_agents is not None:
        others_mask = others_mask & valid_agents
    other_distance = _safe_distance(adversary[None] - frozen)
    proximity = jax.nn.relu(config.other_collision_margin - other_distance)
    future_steps = other_distance.shape[-1]
    reward -= (
        config.other_collision_weight * jnp.sum(proximity * others_mask[:, None]) / future_steps
    )

    return reward


def make_adversarial_guidance(
    *,
    adversary_index: int,
    victim_index: int,
    reference_pose: jax.Array,
    road_edges: RoadEdges | None = None,
    valid_agents: jax.Array | None = None,
    offroad_weight: float = 1.0,
    other_collision_weight: float = 1.0,
    other_collision_margin: float = 2.0,
    softmin_temperature: float = 1.0,
) -> Callable[[jax.Array], jax.Array]:
    """Build a differentiable adversarial reward of x̂₀ for test-time guidance.

    Grounded in the safety-critical traffic-generation literature (STRIVE,
    Safe-Sim, CTG++): steer one *adversary* agent toward one *victim* while
    keeping the scenario drivable. The returned callable maps the diffuser's
    clean estimate ``(num_agents, future_steps, 4)`` in each agent's **local**
    frame to a scalar reward suitable for
    :meth:`~diffav.api.map_conditioned.MapConditionedTrajectoryModel.sample`
    ``(guidance_fn=...)``. It maps to the shared frame itself via
    ``reference_pose`` so the inter-agent distance is meaningful, and reads only
    the adversary's rows with gradient — every other agent is a fixed obstacle
    (``stop_gradient``) — so guidance perturbs the adversary alone (the Safe-Sim
    adversary-only rule that prevents degenerate whole-scene collapse).

    The reward sums three grounded terms:

    - **Adversarial** — the negated soft-min-over-time distance from the
      adversary to the victim (STRIVE ``adv_crash`` / Safe-Sim ``J_coll``): the
      soft-min auto-selects the closest-approach step without hand-specifying
      when the collision happens.
    - **Off-road** — a squared-hinge penalty on the adversary's signed distance
      to the road edges (KING ``φ_adv_dev`` / Safe-Sim ``J_route``), keeping the
      attack on the drivable surface. Skipped when ``road_edges`` is ``None``.
    - **Non-victim collision** — a hinge penalty on the adversary crowding any
      other (non-victim, valid) agent within ``other_collision_margin`` (STRIVE
      ``L_coll`` / Safe-Sim ``J_Gauss``), so the adversary does not plough
      through the rest of the scene to reach the victim.

    Args:
        adversary_index: Row of the steered adversary agent.
        victim_index: Row of the victim (target) agent.
        reference_pose: Per-agent current pose ``(num_agents, 3)`` mapping each
            agent's local-frame estimate back to the shared frame.
        road_edges: Oriented road edges for the off-road term, or ``None`` to
            drop it (e.g. a scene without a usable map).
        valid_agents: Optional boolean ``(num_agents,)`` mask; when set, only
            valid agents contribute to the non-victim collision term (excludes
            padded slots).
        offroad_weight: Weight of the off-road penalty.
        other_collision_weight: Weight of the non-victim collision penalty.
        other_collision_margin: Distance (m) below which the adversary is
            penalized for crowding another agent.
        softmin_temperature: Temperature of the soft-min over time (smaller is
            closer to a hard minimum).

    Returns:
        Differentiable scalar reward of the local-frame x̂₀.
    """
    config = AdversarialRewardConfig(
        offroad_weight=offroad_weight,
        other_collision_weight=other_collision_weight,
        other_collision_margin=other_collision_margin,
        softmin_temperature=softmin_temperature,
    )

    def guidance(x_0: jax.Array) -> jax.Array:
        # The inner estimate is per-agent local; map to the shared frame so the
        # inter-agent distances are meaningful before scoring.
        world = from_agent_frame(x_0, reference_pose)
        return adversarial_victim_reward(
            world[..., :2],
            adversary_index=adversary_index,
            victim_index=victim_index,
            road_edges=road_edges,
            valid_agents=valid_agents,
            config=config,
        )

    return guidance


def build_steering_pairs(
    pool: CandidatePool,
    *,
    selection_pressure: float,
    num_pairs: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Construct chosen/rejected pairs from a scored candidate pool.

    With ``selection_pressure`` α > 0, chosen candidates come from the top
    ``ceil((1 − α) · num_feasible)`` of the feasible ranking (higher α →
    tighter extreme → larger reward gaps → stronger steering) and rejected
    candidates from the infeasible set plus the bottom of the feasible
    ranking — an infeasible candidate can never be chosen. With α = 0 the
    pairing is random regardless of reward: the no-steering baseline.

    Args:
        pool: Scored candidate pool.
        selection_pressure: α ∈ [0, 1].
        num_pairs: Number of pairs to emit (pools are cycled).
        key: JAX random key (used for the α = 0 shuffle).

    Returns:
        Tuple of (chosen ``(num_pairs, A, T, 4)``, rejected same shape,
        margins ``(num_pairs,)``), or ``None`` when no valid pairing exists
        (no feasible candidate, or a single candidate total).
    """
    pools = _select_candidate_pools(pool, selection_pressure, key)
    if pools is None:
        return None
    chosen_pool, rejected_pool = pools

    indices = _cycle_pair_indices(chosen_pool, rejected_pool, num_pairs)
    if indices is None:
        return None
    chosen_idx, rejected_idx = indices

    chosen = pool.trajectories[jnp.array(chosen_idx)]
    rejected = pool.trajectories[jnp.array(rejected_idx)]
    margins = pool.rewards[jnp.array(chosen_idx)] - pool.rewards[jnp.array(rejected_idx)]
    return chosen, rejected, margins


def _select_candidate_pools(
    pool: CandidatePool,
    selection_pressure: float,
    key: jax.Array,
) -> tuple[list[int], list[int]] | None:
    """Select chosen/rejected index pools per the selection-pressure rules."""
    if selection_pressure == 0.0:
        pools = _random_pools(pool, key)
    else:
        pools = _ranked_pools(pool, selection_pressure)
    if pools is None:
        return None
    chosen_pool, rejected_pool = pools
    if not chosen_pool or not rejected_pool:
        logger.warning("build_steering_pairs: not enough candidates to pair; skipping.")
        return None
    return pools


def _random_pools(pool: CandidatePool, key: jax.Array) -> tuple[list[int], list[int]]:
    """Random half/half split — the α = 0 no-steering baseline."""
    num_candidates = int(pool.trajectories.shape[0])
    order = [int(i) for i in jax.random.permutation(key, num_candidates)]
    half = num_candidates // 2
    return order[:half], order[half : 2 * half]


def _ranked_pools(
    pool: CandidatePool,
    selection_pressure: float,
) -> tuple[list[int], list[int]] | None:
    """Top/bottom reward-ranked pools with the feasibility gate applied."""
    num_candidates = int(pool.trajectories.shape[0])
    feasible_mask = [bool(f) for f in pool.feasible]
    rewards = [float(r) for r in pool.rewards]
    feasible_ranked = sorted(
        (i for i in range(num_candidates) if feasible_mask[i]),
        key=lambda i: -rewards[i],
    )
    if not feasible_ranked:
        logger.warning("build_steering_pairs: no feasible candidates; skipping.")
        return None
    infeasible = sorted(
        (i for i in range(num_candidates) if not feasible_mask[i]),
        key=lambda i: rewards[i],
    )
    breadth = min(
        max(1, math.ceil((1.0 - selection_pressure) * len(feasible_ranked))),
        len(feasible_ranked),
    )
    # Rejected material: infeasible first (worst reward first), then the
    # bottom `breadth` feasible candidates from worst to best.
    return feasible_ranked[:breadth], infeasible + feasible_ranked[-breadth:][::-1]


def _cycle_pair_indices(
    chosen_pool: list[int],
    rejected_pool: list[int],
    num_pairs: int,
) -> tuple[list[int], list[int]] | None:
    """Cycle both pools into pair index lists, never pairing a candidate with itself."""
    chosen_idx = [chosen_pool[i % len(chosen_pool)] for i in range(num_pairs)]
    rejected_idx = [rejected_pool[i % len(rejected_pool)] for i in range(num_pairs)]
    for pair_index in range(num_pairs):
        if chosen_idx[pair_index] != rejected_idx[pair_index]:
            continue
        # Wide pools can overlap in the middle; advance to the next distinct entry.
        offset = 1
        while rejected_pool[(pair_index + offset) % len(rejected_pool)] == chosen_idx[
            pair_index
        ] and offset < len(rejected_pool):
            offset += 1
        rejected_idx[pair_index] = rejected_pool[(pair_index + offset) % len(rejected_pool)]
        if rejected_idx[pair_index] == chosen_idx[pair_index]:
            logger.warning("build_steering_pairs: degenerate single-candidate pool.")
            return None
    return chosen_idx, rejected_idx
