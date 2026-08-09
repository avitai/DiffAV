"""Public ScenarioMiner SDK: generate, evaluate, and adversarially search scenarios."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

from diffav.alignment.scenario_steering import (
    ScenarioSteeringConfig,
    ScenarioSteeringTrainer,
    SteeringStrategy,
)
from diffav.alignment.steering_spine import make_steering_guidance
from diffav.alignment.weight_soup import make_weight_soup
from diffav.api.config import FailureCase, MinerConfig, Scenario
from diffav.core.constants import (
    MINER_STATE_OFFSETS,
    MINER_STATE_SCALES,
    SCENE_BACKBONE_ARCHITECTURE_VERSION,
    WOD_HISTORY_STEPS,
    WOD_STEP_DURATION_SECONDS,
)
from diffav.core.training_utils import nan_safe_gradients
from diffav.core.types import (
    AgentState,
    AgentType,
    Density,
    MetricsReport,
    ScenarioMetadata,
    ScenarioType,
    SceneContext,
    TrajectoryPrediction,
)
from diffav.evaluation.metrics import ade as _ade, fde as _fde
from diffav.models.trajectory_diffusion import (
    GuidanceSpec,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)
from diffav.physics.losses import DiffAVPhysicsLoss


logger = logging.getLogger(__name__)

# ── density → agent count ──────────────────────────────────────────────────
_DENSITY_AGENTS: dict[Density, int] = {Density.LOW: 2, Density.MEDIUM: 8, Density.HIGH: 16}


def _agent_count_for_density(density: Density, max_agents: int) -> int:
    """Return number of agents for a density level, capped at max_agents."""
    return min(_DENSITY_AGENTS[density], max_agents)


def _validate_generation_inputs(
    scenario_type: str | ScenarioType,
    density: str | Density,
    count: int,
) -> tuple[ScenarioType, Density]:
    """Fail fast on invalid SDK boundary inputs.

    Args:
        scenario_type: Requested scene template label.
        density: Requested agent density label.
        count: Requested number of scenarios.

    Returns:
        The validated ``(ScenarioType, Density)`` pair.

    Raises:
        ValueError: If any input is outside the SDK vocabulary or negative.
    """
    try:
        validated_type = ScenarioType(scenario_type)
    except ValueError as exc:
        valid_types = [t.value for t in ScenarioType]
        msg = f"scenario_type must be one of {valid_types}, got {scenario_type!r}"
        raise ValueError(msg) from exc
    try:
        validated_density = Density(density)
    except ValueError as exc:
        valid_densities = [d.value for d in Density]
        msg = f"density must be one of {valid_densities}, got {density!r}"
        raise ValueError(msg) from exc
    if count < 0:
        msg = f"count must be non-negative, got {count}"
        raise ValueError(msg)
    return validated_type, validated_density


def _build_scene_context(scenario_type: str, n_agents: int, scene_index: int) -> SceneContext:
    """Build a synthetic SceneContext for a given scenario type.

    Args:
        scenario_type: Template label (e.g. ``"forward"``, ``"lane_change"``).
        n_agents: Number of surrounding agents.
        scene_index: Index used to vary agent positions.

    Returns:
        Synthetic ``SceneContext``.
    """
    spacing = 8.0 + float(scene_index) * 0.5
    heading_delta = 0.1 if scenario_type == "lane_change" else 0.0

    ego = AgentState(
        position=jnp.zeros(2),
        heading=0.0,
        velocity=10.0,
        acceleration=0.0,
        agent_type=AgentType.VEHICLE,
    )
    agents = tuple(
        AgentState(
            position=jnp.array([spacing * (i + 1), float(i % 3) * 3.5]),
            heading=heading_delta * i,
            velocity=8.0 + float(i),
            acceleration=0.0,
            agent_type=AgentType.VEHICLE,
        )
        for i in range(n_agents)
    )
    return SceneContext(
        ego_state=ego,
        agent_states=agents,
        map_features=(),
        timestamps=jnp.linspace(
            0.0, WOD_STEP_DURATION_SECONDS * (WOD_HISTORY_STEPS - 1), WOD_HISTORY_STEPS
        ),
    )


def _scene_to_context_array(scene: SceneContext, context_dim: int) -> jax.Array:
    """Encode a SceneContext as a flat embedding array for model input.

    Stacks per-agent features [x, y, vx, vy], zero-padded to context_dim.

    Args:
        scene: The scene to encode.
        context_dim: Target feature dimension (must be >= 4).

    Returns:
        Float32 array of shape ``(n_agents, context_dim)``.
    """
    if context_dim < 4:
        raise ValueError(f"context_dim must be >= 4, got {context_dim}")
    rows = []
    for state in scene.agent_states:
        vx = state.velocity * jnp.cos(jnp.array(state.heading))
        vy = state.velocity * jnp.sin(jnp.array(state.heading))
        raw = jnp.array([state.position[0], state.position[1], vx, vy])
        pad_len = max(context_dim - raw.shape[0], 0)
        rows.append(jnp.pad(raw, (0, pad_len))[:context_dim])
    if not rows:
        return jnp.zeros((0, context_dim), dtype=jnp.float32)
    return jnp.stack(rows)  # (n_agents, context_dim)


def _classify_failure_mode(loss_components: dict[str, jax.Array]) -> str:
    """Return a human-readable failure mode label from physics loss components.

    Args:
        loss_components: Dict from ``DiffAVPhysicsLoss.compute_loss``.

    Returns:
        ``"kinematics_violation"`` or ``"collision_risk"``.
    """
    kin = float(loss_components.get("kinematic_loss", jnp.array(0.0)))
    col = float(loss_components.get("collision_loss", jnp.array(0.0)))
    return "kinematics_violation" if kin >= col else "collision_risk"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioMiner:
    """High-level SDK for generating and evaluating autonomous driving scenarios.

    Wraps a ``TrajectoryDiffusionModel`` to provide three public entry points:

    - :meth:`generate` — synthesise scenario batches from a scene template.
    - :meth:`evaluate_planner` — measure an external planner against reference
      trajectories using ADE/FDE metrics.
    - :meth:`adversarial_search` — use gradient ascent on scene context embeddings
      to find scenes that maximise physics violations in a planner's output.

    Create via :func:`create_scenario_miner` rather than directly.

    Attributes:
        model: Loaded trajectory diffusion model.
        config: SDK configuration.
    """

    model: TrajectoryDiffusionModel
    config: MinerConfig

    def generate(
        self,
        scenario_type: str | ScenarioType,
        density: str | Density,
        *,
        count: int,
        key: jax.Array | None = None,
    ) -> list[Scenario]:
        """Generate a batch of synthetic scenarios.

        Args:
            scenario_type: Scene template — one of
                :class:`~diffav.core.types.ScenarioType` (``"forward"``,
                ``"lane_change"``, ``"unprotected_left_turn"``,
                ``"adversarial"``).
            density: Agent density — one of
                :class:`~diffav.core.types.Density`: ``"low"`` (2),
                ``"medium"`` (8), or ``"high"`` (16), capped at
                ``config.max_agents``.
            count: Number of scenarios to generate (non-negative).
            key: Optional JAX random key. Defaults to ``jax.random.key(0)``.

        Returns:
            List of ``Scenario`` objects with reference predictions. Sampling
            runs in ``jax.vmap`` chunks of ``config.batch_size``; each
            scenario derives its own subkey, so results do not depend on the
            chunk layout.

        Raises:
            ValueError: If ``scenario_type`` or ``density`` is outside the
                SDK vocabulary, or ``count`` is negative.
        """
        scenario_type, density = _validate_generation_inputs(scenario_type, density, count)
        if count == 0:
            return []
        rng: jax.Array = jax.random.key(0) if key is None else key
        return self._generate_with_model(self.model, scenario_type, density, count, rng)

    def _generate_with_model(
        self,
        model: TrajectoryDiffusionModel,
        scenario_type: str,
        density: str,
        count: int,
        rng: jax.Array,
        *,
        guidance_fn: Callable[[jax.Array], jax.Array] | None = None,
        guidance_scale: float = 0.0,
    ) -> list[Scenario]:
        """Chunked vmapped generation shared by ``generate`` and ``steer``.

        Args:
            model: Model to sample from (the miner's own or a steered one).
            scenario_type: Validated scene template label.
            density: Validated density label.
            count: Number of scenarios (positive).
            rng: JAX random key.
            guidance_fn: Optional test-time x̂₀ guidance reward.
            guidance_scale: Guidance step size η (0 disables guidance).

        Returns:
            List of ``Scenario`` objects with generation metadata.
        """
        n_agents = _agent_count_for_density(Density(density), self.config.max_agents)
        contexts = [
            _build_scene_context(scenario_type, n_agents, scene_index=i) for i in range(count)
        ]
        subkeys: list[jax.Array] = []
        for _ in range(count):
            rng, subkey = jax.random.split(rng)
            subkeys.append(subkey)

        # Sampling is vmapped over batch_size-sized chunks; each scenario keeps
        # its own subkey, so results are invariant to the chunk layout.
        graphdef, state = nnx.split(model)
        guidance = (
            GuidanceSpec(reward_fn=guidance_fn, scale=guidance_scale)
            if guidance_fn is not None
            else None
        )

        def _sample_trajectories(context_arr: jax.Array, sample_key: jax.Array) -> jax.Array:
            merged = nnx.merge(graphdef, state)
            prediction = merged.sample(context_arr, key=sample_key, guidance=guidance)
            return prediction.trajectories

        batched_sample = jax.vmap(_sample_trajectories)
        agent_ids = tuple(f"agent_{i}" for i in range(n_agents))

        scenarios: list[Scenario] = []
        for start in range(0, count, self.config.batch_size):
            chunk = range(start, min(start + self.config.batch_size, count))
            context_stack = jnp.stack(
                [_scene_to_context_array(contexts[i], self.config.context_dim) for i in chunk]
            )
            key_stack = jnp.stack([subkeys[i] for i in chunk])
            chunk_trajectories = batched_sample(context_stack, key_stack)
            for offset, i in enumerate(chunk):
                scenarios.append(
                    Scenario(
                        context=contexts[i],
                        predictions=TrajectoryPrediction(
                            trajectories=chunk_trajectories[offset],
                            agent_ids=agent_ids,
                        ),
                        metadata=ScenarioMetadata(
                            scenario_id=f"{scenario_type}_{i:06d}",
                            source_dataset="diffav_synthetic",
                            tags=(scenario_type, density),
                        ),
                    )
                )

        return scenarios

    def evaluate_planner(
        self,
        planner_fn: Callable[[SceneContext], TrajectoryPrediction],
        scenarios: list[Scenario],
    ) -> MetricsReport:
        """Evaluate an external planner against reference scenario predictions.

        Uses standalone :func:`~diffav.evaluation.metrics.ade` and
        :func:`~diffav.evaluation.metrics.fde` from
        ``diffav.evaluation.metrics``, vmapped over agents.

        Args:
            planner_fn: Callable mapping ``SceneContext`` → ``TrajectoryPrediction``.
            scenarios: List of scenarios to evaluate.

        Returns:
            ``MetricsReport`` with ``"ade"`` and ``"fde"`` averaged over all
            scenarios and agents.
        """
        if not scenarios:
            return MetricsReport(metric_values={})

        ade_vals: list[float] = []
        fde_vals: list[float] = []

        for scenario in scenarios:
            planner_pred = planner_fn(scenario.context)
            pred_xy = planner_pred.trajectories[..., :2]  # (N, T, 2)
            ref_xy = scenario.predictions.trajectories[..., :2]  # (N, T, 2)
            n_agents, n_steps = pred_xy.shape[:2]
            valid = jnp.ones((n_agents, n_steps))

            per_agent_ade = jax.vmap(_ade)(pred_xy, ref_xy, valid)  # (N,)
            per_agent_fde = jax.vmap(_fde)(pred_xy, ref_xy, valid)  # (N,)
            ade_vals.append(float(jnp.mean(per_agent_ade)))
            fde_vals.append(float(jnp.mean(per_agent_fde)))

        return MetricsReport(
            metric_values={
                "ade": sum(ade_vals) / len(ade_vals),
                "fde": sum(fde_vals) / len(fde_vals),
            }
        )

    def steer(
        self,
        scenario_type: str | ScenarioType,
        density: str | Density,
        *,
        steering_config: ScenarioSteeringConfig,
        count: int,
        key: jax.Array | None = None,
    ) -> list[Scenario]:
        """Generate scenarios steered toward a target scenario type.

        Dispatches on ``steering_config.strategy``. For ``RANKED_DPO``, a
        cloned model is fine-tuned for ``num_steering_steps`` DPO steps on
        reward-ranked, feasibility-gated preference pairs sampled from the
        model itself (see ``ScenarioSteeringTrainer``), then ``count``
        scenarios are generated with the tuned weights.

        Args:
            scenario_type: Base scene template for probe contexts
                (e.g. ``"forward"``, ``"lane_change"``).
            density: Agent density (``"low"``, ``"medium"``, or ``"high"``).
            steering_config: Steering configuration including target scenario
                type, strategy, and selection pressure.
            count: Number of steered scenarios to generate.
            key: Optional JAX random key. Defaults to ``jax.random.key(0)``.

        Returns:
            List of steered ``Scenario`` objects.

        Raises:
            ValueError: If ``scenario_type`` or ``density`` is outside the
                SDK vocabulary, ``count`` is negative, or the target scenario
                is outside the SDK vocabulary.
        """
        scenario_type, density = _validate_generation_inputs(scenario_type, density, count)
        try:
            ScenarioType(steering_config.target_scenario)
        except ValueError as error:
            valid = ", ".join(member.value for member in ScenarioType)
            msg = (
                f"target_scenario must be one of [{valid}], got {steering_config.target_scenario!r}"
            )
            raise ValueError(msg) from error
        if count == 0:
            return []

        rng: jax.Array = jax.random.key(0) if key is None else key
        rng, steer_key, gen_key = jax.random.split(rng, 3)

        n_agents = _agent_count_for_density(density, self.config.max_agents)

        # Probe contexts the steering steps sample candidates against
        num_probes = min(4, max(count, 2))
        probe_contexts = jnp.stack(
            [
                _scene_to_context_array(
                    _build_scene_context(scenario_type, n_agents, scene_index=i),
                    self.config.context_dim,
                )
                for i in range(num_probes)
            ]
        )

        if steering_config.strategy is SteeringStrategy.GUIDANCE:
            # No training: reward-gradient ascent in x̂₀-space at sample time,
            # with steering_strength as the guidance scale η.
            guidance_fn = make_steering_guidance(
                steering_config.target_scenario, reference_speed=10.0
            )
            raw = self._generate_with_model(
                self.model,
                scenario_type,
                density,
                count,
                gen_key,
                guidance_fn=guidance_fn,
                guidance_scale=steering_config.steering_strength,
            )
        else:
            if steering_config.strategy is SteeringStrategy.WEIGHT_SOUP:
                # Train a full-pressure expert once; steering_strength becomes
                # the parameter-interpolation weight.
                expert_config = replace(
                    steering_config,
                    strategy=SteeringStrategy.RANKED_DPO,
                    steering_strength=1.0,
                )
                expert = self._train_steering_expert(
                    expert_config, probe_contexts, steer_key, n_agents
                )
                steered_model = make_weight_soup(
                    self.model, expert, steering_config.steering_strength
                )
            else:
                steered_model = self._train_steering_expert(
                    steering_config, probe_contexts, steer_key, n_agents
                )
            raw = self._generate_with_model(steered_model, scenario_type, density, count, gen_key)
        target = steering_config.target_scenario
        return [
            replace(
                scenario,
                metadata=ScenarioMetadata(
                    scenario_id=f"steered_{target}_{i:06d}",
                    source_dataset="diffav_steered",
                    tags=(scenario_type, density, f"steered_{target}"),
                ),
            )
            for i, scenario in enumerate(raw)
        ]

    def _train_steering_expert(
        self,
        steering_config: ScenarioSteeringConfig,
        probe_contexts: jax.Array,
        key: jax.Array,
        num_agents: int,
    ) -> TrajectoryDiffusionModel:
        """Fine-tune a clone of the miner's model with ranked-DPO steering.

        Args:
            steering_config: Steering configuration for the expert.
            probe_contexts: Scene contexts the steering steps sample against.
            key: JAX random key (per-step keys derive via ``fold_in``).
            num_agents: Agents per candidate scene.

        Returns:
            The fine-tuned clone; the miner's own model stays untouched.
        """
        from diffav.alignment.dpo_trainer import (  # noqa: PLC0415
            DPOAlignmentConfig,
            DPOAlignmentTrainer,
        )

        expert = nnx.clone(self.model)
        tx = create_optimizer(
            OptimizerConfig(optimizer_type="adam", learning_rate=1e-4, gradient_clip=1.0)
        )
        optimizer = nnx.Optimizer(expert, tx, wrt=nnx.Param)
        dpo_trainer = DPOAlignmentTrainer(
            model=expert,
            optimizer=optimizer,
            config=DPOAlignmentConfig(reference_free=True),
        )
        steer_trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=steering_config)

        for step in range(steering_config.num_steering_steps):
            step_key = jax.random.fold_in(key, step)
            steer_trainer.steer_step(probe_contexts, step_key, num_agents=num_agents)
        return expert

    def adversarial_search(
        self,
        planner_fn: Callable[[SceneContext], TrajectoryPrediction],
        budget: int,
        *,
        severity_threshold: float = 0.0,
        perturbation_steps: int = 10,
        step_size: float = 0.05,
        key: jax.Array | None = None,
    ) -> list[FailureCase]:
        """Find adversarial scenarios using gradient-based context perturbation.

        Explores exactly ``budget`` candidate scenes. For each, Adam gradient
        ascent (via ``opifex.core.training.optimizers``) perturbs the scene
        context embedding to maximise the physics violation score of the
        internal model's predictions; the perturbed context is then passed to
        ``planner_fn`` and its output scored by ``DiffAVPhysicsLoss``. A
        candidate becomes a ``FailureCase`` only when the planner's severity
        exceeds ``severity_threshold`` — a healthy planner yields fewer than
        ``budget`` cases, possibly none.

        The candidate key is the first split of ``key``, so the unperturbed
        seed scenarios are reproducible via
        ``generate(ScenarioType.ADVERSARIAL, Density.MEDIUM, count=budget,
        key=jax.random.split(key)[0])``. Failure cases carry reference
        predictions regenerated on the *perturbed* context.

        NaN gradients are suppressed by
        ``diffav.core.training_utils.nan_safe_gradients``.

        Note:
            ``planner_fn`` does NOT need to be JAX-differentiable.

        Args:
            planner_fn: Callable mapping ``SceneContext`` → ``TrajectoryPrediction``.
            budget: Number of candidate scenes to explore (non-negative).
            severity_threshold: Minimum planner severity (exclusive) for a
                candidate to count as a failure.
            perturbation_steps: Adam gradient-ascent steps per candidate.
            step_size: Adam learning rate for perturbation.
            key: Optional JAX random key. Defaults to ``jax.random.key(0)``.

        Returns:
            List of at most ``budget`` ``FailureCase`` objects, sorted by
            severity descending.

        Raises:
            ValueError: If ``budget`` is negative.
        """
        if budget < 0:
            msg = f"budget must be non-negative, got {budget}"
            raise ValueError(msg)
        if budget == 0:
            return []

        rng: jax.Array = jax.random.key(0) if key is None else key
        candidate_key, perturb_key = jax.random.split(rng)

        physics = DiffAVPhysicsLoss()
        opt_cfg = OptimizerConfig(optimizer_type="adam", learning_rate=step_size)
        tx = create_optimizer(opt_cfg)

        candidates = self.generate(
            scenario_type=ScenarioType.ADVERSARIAL,
            density=Density.MEDIUM,
            count=budget,
            key=candidate_key,
        )
        failure_cases: list[FailureCase] = []

        # Split model state once so the loss function uses a stable pytree
        # of concrete arrays rather than NNX variables — follows the nnx.split/
        # merge pattern used by opifex (NNXHybridOptimizer) and avoids per-
        # candidate JIT recompilation from new lambda closures.
        graphdef, model_state = nnx.split(self.model)
        n_agents = _agent_count_for_density(Density.MEDIUM, self.config.max_agents)

        def _physics_loss(delta: jax.Array, base: jax.Array, sample_key: jax.Array) -> jax.Array:
            """Physics loss of model predictions at perturbed context."""
            frozen_model = nnx.merge(graphdef, model_state)
            pred = frozen_model.sample(base + delta, key=sample_key)
            loss, _ = physics.compute_loss(pred.trajectories, epoch=0)
            return loss

        # Compile once for all candidates (shapes are identical across the
        # budget); differentiate w.r.t. delta only. Without jit this loop
        # re-executes the full reverse diffusion eagerly per step.
        grad_fn = jax.jit(jax.value_and_grad(_physics_loss, argnums=0))

        for index, candidate in enumerate(candidates):
            sample_key = jax.random.fold_in(perturb_key, index)
            base_ctx = _scene_to_context_array(candidate.context, self.config.context_dim)
            delta = jnp.zeros_like(base_ctx)
            opt_state = tx.init(delta)

            for _ in range(perturbation_steps):
                loss_val, grad = grad_fn(delta, base_ctx, sample_key)
                safe_grad, has_nan = nan_safe_gradients(loss_val, grad)
                if bool(has_nan):
                    logger.debug("NaN gradient in adversarial search — skipping step")
                    break
                # Negate gradient to perform ascent (maximise physics violation).
                updates, opt_state = tx.update(-safe_grad, opt_state)
                delta = cast(jax.Array, optax.apply_updates(delta, updates))

            delta_pos = delta[:n_agents, :2]
            adv_states = tuple(
                AgentState(
                    position=s.position + delta_pos[i],
                    heading=s.heading,
                    velocity=s.velocity,
                    acceleration=s.acceleration,
                    agent_type=s.agent_type,
                )
                for i, s in enumerate(candidate.context.agent_states)
            )
            adv_context = SceneContext(
                ego_state=candidate.context.ego_state,
                agent_states=adv_states,
                map_features=candidate.context.map_features,
                timestamps=candidate.context.timestamps,
            )

            planner_output = planner_fn(adv_context)
            planner_loss, loss_components = physics.compute_loss(
                planner_output.trajectories, epoch=0
            )
            severity = max(float(planner_loss), 0.0)
            if severity <= severity_threshold:
                continue

            # Regenerate reference predictions on the perturbed context so
            # the failure case describes the scene that caused the failure.
            frozen_model = nnx.merge(graphdef, model_state)
            adv_predictions = frozen_model.sample(base_ctx + delta, key=sample_key)
            adv_scenario = Scenario(
                context=adv_context,
                predictions=adv_predictions,
                metadata=ScenarioMetadata(
                    scenario_id=f"adv_{candidate.metadata.scenario_id}",
                    source_dataset="diffav_adversarial",
                    tags=("adversarial",),
                ),
            )

            failure_cases.append(
                FailureCase(
                    scenario=adv_scenario,
                    failure_mode=_classify_failure_mode(loss_components),
                    severity=severity,
                )
            )

        failure_cases.sort(key=lambda fc: fc.severity, reverse=True)
        return failure_cases


def create_scenario_miner(
    config: MinerConfig,
    *,
    rngs: nnx.Rngs | None = None,
) -> ScenarioMiner:
    """Create a ScenarioMiner, loading a checkpoint if ``config.model_path`` is set.

    If ``config.model_path`` is empty, a freshly initialised model is created.

    Args:
        config: SDK configuration.
        rngs: Optional NNX random keys. Defaults to ``nnx.Rngs(params=jax.random.key(0))``.

    Returns:
        A ``ScenarioMiner`` ready for use.

    Raises:
        FileNotFoundError: If ``config.model_path`` is set but does not exist.
    """
    if rngs is None:
        rngs = nnx.Rngs(params=jax.random.key(0))

    model_cfg = TrajectoryDiffusionConfig(
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_temporal_layers=config.num_temporal_layers,
        num_social_layers=config.num_social_layers,
        use_social_interaction=config.use_social_interaction,
        num_heads=config.num_heads,
        num_agents_max=config.max_agents,
        future_steps=config.prediction_horizon,
        context_dim=config.context_dim,
        num_timesteps=config.num_diffusion_steps,
        # Diffusion runs in normalized space with the standard unit clamp;
        # samples and losses stay in metres (see TrajectoryDiffusionConfig).
        state_offsets=MINER_STATE_OFFSETS,
        state_scales=MINER_STATE_SCALES,
        x0_clip_bound=1.0,
    )
    model = TrajectoryDiffusionModel(model_cfg, rngs=rngs)

    if config.model_path:
        _load_checkpoint_into_model(model, config.model_path)

    return ScenarioMiner(model=model, config=config)


def _load_checkpoint_into_model(model: TrajectoryDiffusionModel, checkpoint_path: str) -> None:
    """Restore model parameters from a DiffAVCheckpointManager checkpoint.

    The model is updated in place from the latest checkpoint in the
    directory.

    Args:
        model: Model whose parameters will be restored.
        checkpoint_path: Path to the checkpoint directory.

    Raises:
        FileNotFoundError: If the checkpoint directory does not exist or
            contains no checkpoint.
        CheckpointCorruptError: If the latest checkpoint cannot be restored.
    """
    from diffav.models.checkpointing import (  # noqa: PLC0415
        CheckpointConfig,
        DiffAVCheckpointManager,
    )

    path = Path(checkpoint_path)
    if not path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    with DiffAVCheckpointManager(CheckpointConfig(checkpoint_dir=checkpoint_path)) as manager:
        state = manager.restore_latest(
            model, expected_architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION
        )
    if state is None:
        msg = f"No checkpoint found in: {checkpoint_path}"
        raise FileNotFoundError(msg)
    logger.info("Loaded checkpoint from %s (step=%d)", checkpoint_path, state.step)
