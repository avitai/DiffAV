"""Scenario-type steering for DPO-aligned trajectory generation.

Steers diffusion trajectories toward a target scenario class by training on
preference pairs the model itself generated: candidates are sampled from the
current policy, scored by the rule-based scenario reward plus the composite
safety reward, gated on road-edge feasibility, and ranked into chosen/rejected
pairs (see :mod:`simulacrax.alignment.steering_spine`). The DPO loss on those
pairs carries the steering gradient — the reward shapes the training
distribution rather than entering the loss as a constant.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simulacrax.alignment.dpo_trainer import (
    _build_alignment_metrics,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from simulacrax.alignment.steering_spine import build_steering_pairs, sample_and_score
from simulacrax.core.geometry import RoadEdges


logger = logging.getLogger(__name__)


class SteeringStrategy(enum.StrEnum):
    """Steering strategy selector for :meth:`ScenarioMiner.steer`.

    ``RANKED_DPO`` fine-tunes on reward-ranked pairs with
    ``steering_strength`` as selection pressure. ``WEIGHT_SOUP`` trains a
    full-pressure expert the same way, then interpolates parameters between
    the base model and the expert with ``steering_strength`` as the
    interpolation weight λ — one expert run serves every strength.
    ``GUIDANCE`` trains nothing: sampling takes reward-gradient ascent steps
    in x̂₀-space with ``steering_strength`` as the guidance scale η.
    """

    RANKED_DPO = "ranked_dpo"
    WEIGHT_SOUP = "weight_soup"
    GUIDANCE = "guidance"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioSteeringConfig:
    """Configuration for scenario-type steering.

    Attributes:
        target_scenario: Canonical scenario type label to steer toward.
            Must match labels used in ``ScenarioMiner.generate()``
            (e.g. ``"forward"``, ``"lane_change"``, ``"unprotected_left_turn"``).
        strategy: Steering strategy to apply.
        steering_strength: Selection pressure α ∈ [0.0, 1.0]. Chosen/rejected
            pools span the top/bottom ``ceil((1 − α) · num_feasible)`` of the
            reward ranking, so higher α widens the reward gap between pairs;
            ``0.0`` pairs candidates randomly (no steering).
        num_steering_steps: Number of gradient steps applied per call to
            :meth:`ScenarioMiner.steer`.
        num_candidates: Candidates sampled per scene context when building
            preference pairs (at least 2).
        feasibility_threshold: Maximum off-road fraction for a candidate to
            remain eligible as "chosen".
    """

    target_scenario: str
    strategy: SteeringStrategy = SteeringStrategy.RANKED_DPO
    steering_strength: float = 0.3
    num_steering_steps: int = 1
    num_candidates: int = 8
    feasibility_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        if not (0.0 <= self.steering_strength <= 1.0):
            raise ValueError(
                f"steering_strength must be in [0.0, 1.0]; got {self.steering_strength}"
            )
        if self.num_steering_steps < 1:
            raise ValueError(f"num_steering_steps must be >= 1; got {self.num_steering_steps}")
        if self.num_candidates < 2:
            raise ValueError(f"num_candidates must be >= 2; got {self.num_candidates}")
        if self.feasibility_threshold < 0.0:
            raise ValueError(
                f"feasibility_threshold must be non-negative; got {self.feasibility_threshold}"
            )


def compute_scenario_reward(
    trajectories: jax.Array,
    target_scenario: str,
    *,
    reference_speed: float,
) -> jax.Array:
    """Compute rule-based steering reward for target scenario alignment.

    Returns a scalar in [-1.0, 1.0] indicating how well ``trajectories``
    match the kinematic signature of ``target_scenario``.

    The reward is JIT-compatible: scenario dispatch happens at trace time
    because ``target_scenario`` is a Python string.

    Scenario rules:

    - ``"forward"``: Rewards forward progress (mean x-displacement relative
      to the progress expected at ``reference_speed``).
    - ``"lane_change"``: Rewards significant lateral displacement (mean
      absolute y-displacement against a 3.5 m lane width).
    - All other labels: Reward based on trajectory finiteness and boundedness
      (1.0 if all finite and abs-max <= 200 m, −1.0 otherwise).

    Args:
        trajectories: Predicted trajectories, shape ``(num_agents, T, state_dim)``
            where state is ``[x, y, heading, velocity]``.
        target_scenario: Target scenario type label (a Python string literal,
            dispatched at trace time).
        reference_speed: Reference speed in m/s used to normalise forward
            progress (clamped to at least 1 m/s).

    Returns:
        Scalar reward in [-1.0, 1.0]; higher means better alignment.
    """
    speed = jnp.maximum(jnp.abs(jnp.array(reference_speed)), 1.0)

    if target_scenario == "forward":
        x_positions = trajectories[..., 0]  # (num_agents, T)
        num_steps = trajectories.shape[-2]
        total_x_disp = x_positions[..., -1] - x_positions[..., 0]  # (num_agents,)
        mean_x_disp = jnp.mean(total_x_disp)
        # Normalise: expected forward progress = reference_speed * T * dt (dt=0.1)
        expected_progress = speed * num_steps * 0.1
        reward_raw = mean_x_disp / jnp.maximum(expected_progress, 1.0)
        return jnp.clip(reward_raw, -1.0, 1.0)

    if target_scenario == "lane_change":
        y_positions = trajectories[..., 1]  # (num_agents, T)
        lateral_disp = jnp.abs(y_positions[..., -1] - y_positions[..., 0])  # (num_agents,)
        mean_lateral = jnp.mean(lateral_disp)
        # 3.5 m lane width as normalisation reference
        reward_raw = (mean_lateral / 3.5) - 1.0
        return jnp.clip(reward_raw, -1.0, 1.0)

    # Default: reward finite, bounded trajectories
    all_finite = jnp.all(jnp.isfinite(trajectories))
    bounded = jnp.max(jnp.abs(trajectories[..., :2])) <= 200.0
    return jnp.where(all_finite & bounded, jnp.array(1.0), jnp.array(-1.0))


class ScenarioSteeringTrainer:
    """Ranked-DPO steering: DPO on reward-ranked, feasibility-gated pairs.

    Each step samples ``config.num_candidates`` trajectories per scene
    context from the *current* policy, ranks them by scenario + safety
    reward, gates feasibility on road edges, and trains with the standard
    DPO loss on the resulting chosen/rejected pairs. The steering gradient
    flows through the DPO term on model-generated samples — there is no
    reward term in the loss itself.

    Args:
        dpo_trainer: Configured DPO trainer to wrap.
        config: Steering configuration.
    """

    def __init__(
        self,
        dpo_trainer: DPOAlignmentTrainer,
        config: ScenarioSteeringConfig,
    ) -> None:
        """Initialize the scenario steering trainer.

        Args:
            dpo_trainer: Configured DPO trainer to wrap.
            config: Steering configuration.
        """
        self.dpo_trainer = dpo_trainer
        self.config = config

    def build_step_batch(
        self,
        scene_contexts: jax.Array,
        key: jax.Array,
        *,
        num_agents: int,
        reference_speed: float = 10.0,
        road_edges: RoadEdges | None = None,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Build a DPO batch of ranked pairs from model-generated candidates.

        Args:
            scene_contexts: Per-agent scene contexts, shape
                ``(num_scenes, num_agents, context_dim)`` — one row per agent.
            key: JAX random key (per-scene keys are derived via ``fold_in``).
            num_agents: Agents per candidate scene (matches each scene
                context's row count).
            reference_speed: Reference speed for reward normalisation.
            road_edges: Optional road edges for the feasibility gate.

        Returns:
            Tuple of (DPO batch dict with ``"chosen"``/``"rejected"``/
            ``"scene_contexts"``, reward margins ``(num_pairs,)``).

        Raises:
            RuntimeError: If no scene yields a valid preference pair (e.g.
                every candidate is infeasible).
        """
        chosen_list: list[jax.Array] = []
        rejected_list: list[jax.Array] = []
        margin_list: list[jax.Array] = []
        context_list: list[jax.Array] = []

        for scene_index in range(scene_contexts.shape[0]):
            scene_key = jax.random.fold_in(key, scene_index)
            pool_key, pair_key = jax.random.split(scene_key)
            pool = sample_and_score(
                self.dpo_trainer.model,
                scene_contexts[scene_index],
                num_agents=num_agents,
                num_candidates=self.config.num_candidates,
                target_scenario=self.config.target_scenario,
                reference_speed=reference_speed,
                road_edges=road_edges,
                feasibility_threshold=self.config.feasibility_threshold,
                key=pool_key,
            )
            pairs = build_steering_pairs(
                pool,
                selection_pressure=self.config.steering_strength,
                num_pairs=1,
                key=pair_key,
            )
            if pairs is None:
                continue
            chosen, rejected, margins = pairs
            chosen_list.append(chosen[0])
            rejected_list.append(rejected[0])
            margin_list.append(margins[0])
            context_list.append(scene_contexts[scene_index])

        if not chosen_list:
            msg = "No scene yielded a valid preference pair (all candidates infeasible?)"
            raise RuntimeError(msg)

        batch = {
            "chosen": jnp.stack(chosen_list),
            "rejected": jnp.stack(rejected_list),
            "scene_contexts": jnp.stack(context_list),
        }
        return batch, jnp.stack(margin_list)

    def steer_step(
        self,
        scene_contexts: jax.Array,
        key: jax.Array,
        *,
        num_agents: int,
        reference_speed: float = 10.0,
        road_edges: RoadEdges | None = None,
    ) -> DPOAlignmentMetrics:
        """Execute one ranked-DPO steering step.

        Splits ``key`` into a sampling key (candidate generation + pairing)
        and a loss key, builds the ranked-pair batch via
        :meth:`build_step_batch`, and applies one standard DPO update on it.
        With ``steering_strength=0.0`` the pairs are random, so the step is
        numerically identical to
        ``dpo_trainer.train_step(build_step_batch(...)[0], loss_key)``.

        Args:
            scene_contexts: Per-agent scene contexts, shape
                ``(num_scenes, num_agents, context_dim)`` — one row per agent.
            key: JAX random key.
            num_agents: Agents per candidate scene (matches each scene
                context's row count).
            reference_speed: Reference speed for reward normalisation.
            road_edges: Optional road edges for the feasibility gate.

        Returns:
            ``DPOAlignmentMetrics`` for this step.
        """
        sample_key, loss_key = jax.random.split(key)
        batch, margins = self.build_step_batch(
            scene_contexts,
            sample_key,
            num_agents=num_agents,
            reference_speed=reference_speed,
            road_edges=road_edges,
        )
        loss, aux = self.dpo_trainer.compute_dpo_step(
            self.dpo_trainer.model, self.dpo_trainer.optimizer, batch, loss_key
        )

        if bool(aux["has_nan"]):
            logger.warning("NaN detected in steer_step, gradients zeroed")
        logger.info("steer_step mean reward margin: %.4f", float(jnp.mean(margins)))

        return _build_alignment_metrics(loss, aux, float(aux["grad_norm"]))
