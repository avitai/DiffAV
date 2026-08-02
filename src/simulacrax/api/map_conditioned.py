"""Map-conditioned trajectory diffusion: SceneTokenizer + factorized backbone.

Composes the differentiable :class:`~simulacrax.data.tokenizer.SceneTokenizer`
scene encoder with the :class:`~simulacrax.models.trajectory_diffusion.
TrajectoryDiffusionModel`. The tokenizer encodes a raw WOD scene into a fused
scene-token set (map + agent + ego) plus per-agent embeddings; each trajectory
agent's embedding is the adaLN anchor, and the fused tokens are the keys/values
the backbone cross-attends to (the CTG++/MotionDiffuser conditioning). Tokenizer
and backbone parameters train jointly under the diffusion loss, so the map
encoder is learned end to end rather than frozen.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
from flax import nnx

from simulacrax.core.constants import (
    AGENT_EMBEDDING,
    AGENT_LOCAL_STATE_OFFSETS,
    AGENT_LOCAL_STATE_SCALES,
    SCENE_BACKBONE_ARCHITECTURE_VERSION,
    SCENE_EMBEDDING,
    SCENE_TOKEN_VALID,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.geometry import from_agent_frame, RoadEdges, to_agent_frame
from simulacrax.core.types import ModalityMode, PredictionType, TrajectoryPrediction
from simulacrax.data.tokenizer import SceneTokenizer, TokenizerConfig
from simulacrax.models.checkpointing import CheckpointConfig, SimulacraxCheckpointManager
from simulacrax.models.trajectory_diffusion import (
    GuidanceSpec,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)
from simulacrax.physics.losses import SimulacraxPhysicsConfig, SimulacraxPhysicsLoss


@dataclass(frozen=True, slots=True, kw_only=True)
class MapConditionedTrajectoryConfig:
    """Configuration coupling a scene tokenizer to a map-conditioned diffuser.

    Attributes:
        tokenizer: Scene tokenizer configuration; its ``embed_dim`` sets both
            the fused-scene-token width and the per-agent anchor width.
        diffusion: Diffusion model configuration. It must enable
            ``use_map_cross_attention`` and set both ``context_dim`` (the
            anchor width) and ``scene_token_dim`` (the key/value width) equal
            to the tokenizer ``embed_dim``.
        physics: Optional physics-loss configuration. When set, a
            validity-aware physics penalty on the diffuser's x̂₀ reconstruction
            is added to the diffusion loss; ``None`` (default) trains on the
            pure diffusion objective.
        physics_x0_annealing: Scale the physics penalty by the drawn timestep's
            ᾱ_t so the reconstruction is trusted in proportion to the model's
            confidence, mirroring the sampling path's x̂₀-guidance anneal. Only
            applies when ``physics`` is set.
    """

    tokenizer: TokenizerConfig
    diffusion: TrajectoryDiffusionConfig
    physics: SimulacraxPhysicsConfig | None = None
    physics_x0_annealing: bool = True

    def __post_init__(self) -> None:
        """Validate the tokenizer/diffusion dimension coupling.

        Raises:
            ValueError: If the diffusion model is map-blind, or its context or
                scene-token width does not match the tokenizer ``embed_dim``.
        """
        if not self.diffusion.use_map_cross_attention:
            msg = "diffusion.use_map_cross_attention must be True for map conditioning"
            raise ValueError(msg)
        embed_dim = self.tokenizer.embed_dim
        if self.diffusion.context_dim != embed_dim:
            msg = (
                f"diffusion.context_dim ({self.diffusion.context_dim}) must equal the "
                f"tokenizer embed_dim ({embed_dim}) — the per-agent anchor width"
            )
            raise ValueError(msg)
        if self.diffusion.scene_token_dim != embed_dim:
            msg = (
                f"diffusion.scene_token_dim ({self.diffusion.scene_token_dim}) must equal "
                f"the tokenizer embed_dim ({embed_dim})"
            )
            raise ValueError(msg)


class MapConditionedTrajectoryModel(nnx.Module):
    """Jointly-trained scene tokenizer and map-conditioned diffusion backbone.

    The tokenizer encodes a raw WOD scene into a fused scene-token set and
    per-agent embeddings; the trajectory agents' rows form the adaLN anchor and
    the fused tokens are the backbone's cross-attention keys/values. All
    parameters train together under the diffusion loss.
    """

    def __init__(
        self,
        config: MapConditionedTrajectoryConfig,
        *,
        rngs: nnx.Rngs,
        element_spec: Mapping[str, object] | None = None,
    ) -> None:
        """Build the tokenizer and the diffusion model.

        Args:
            config: Coupled tokenizer/diffusion configuration.
            rngs: NNX random number generators.
            element_spec: Optional data-source element spec for tokenizer
                modality negotiation (see :class:`SceneTokenizer`).
        """
        self.config = config
        self.tokenizer = SceneTokenizer(config.tokenizer, element_spec=element_spec, rngs=rngs)
        self.diffusion = TrajectoryDiffusionModel(config.diffusion, rngs=rngs)
        # Parameter-free helper; wrap in nnx.static so nnx.split/merge and jit
        # keep it out of the Param/Variable partition (a stable graphdef leaf)
        # while gradients flow only through the x̂₀ reconstruction.
        self._physics: SimulacraxPhysicsLoss | None = nnx.static(
            SimulacraxPhysicsLoss(config.physics) if config.physics is not None else None
        )

    def _encode_scene(self, scene: Mapping[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Tokenize a raw scene into fused tokens, agent embeddings, validity.

        Args:
            scene: Raw WOD scenario dict (state and roadgraph arrays).

        Returns:
            ``(scene_tokens, agent_emb, scene_token_valid)`` where
            ``scene_tokens`` is ``(num_tokens, embed_dim)``, ``agent_emb`` is
            ``(num_raw_agents, embed_dim)``, and ``scene_token_valid`` is a
            boolean ``(num_tokens,)`` marking padded tokens (invalid agents,
            empty polyline slots) so the backbone excludes them.
        """
        tokenized, _, _ = self.tokenizer.apply(scene, {}, {})
        return (
            tokenized[SCENE_EMBEDDING],
            tokenized[AGENT_EMBEDDING],
            tokenized[SCENE_TOKEN_VALID],
        )

    def compute_loss(
        self,
        scene: Mapping[str, Any],
        trajectories: jax.Array,
        agent_rows: jax.Array,
        *,
        reference_pose: jax.Array,
        key: jax.Array,
        epoch: jax.Array | int = 0,
        road_edges: RoadEdges | None = None,
        valid_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Map-conditioned diffusion (plus optional physics) loss for one scene.

        Args:
            scene: Raw WOD scenario dict (agents + roadgraph).
            trajectories: Ground-truth futures ``(num_agents, future_steps,
                state_dim)`` for the selected agents.
            agent_rows: Indices selecting, from the tokenizer's per-agent
                embeddings, the anchor for each trajectory agent — row ``k`` of
                ``trajectories`` is conditioned by ``agent_emb[agent_rows[k]]``.
            reference_pose: Per-agent current pose ``(num_agents, 3)`` defining
                each agent's local frame. The diffusion target is expressed in
                that frame (small local displacements the denoiser can regress),
                and the x̂₀ reconstruction is mapped back to the shared frame
                before the physics terms, which need a common frame.
            key: JAX random key.
            epoch: Current training epoch (traced or static) for the physics
                loss's adaptive weight schedule. Ignored when physics is off.
            road_edges: Optional oriented road edges for the physics boundary
                term. Ignored when physics is off.
            valid_mask: Optional per-step validity ``(num_agents,
                future_steps)``. Masks the diffusion loss and, when physics is
                enabled, excludes padded agents and out-of-horizon steps from
                the physics terms.

        Returns:
            Scalar loss carrying gradients into both the tokenizer and the
            backbone. When physics is configured, the penalty scores the
            diffuser's x̂₀ reconstruction so its gradients shape the denoiser.
        """
        scene_tokens, agent_emb, scene_token_valid = self._encode_scene(scene)
        anchor = agent_emb[agent_rows]
        local_trajectories = to_agent_frame(trajectories, reference_pose)
        outputs = self.diffusion.compute_loss_outputs(
            local_trajectories,
            anchor,
            key=key,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            valid_mask=valid_mask,
        )
        if self._physics is None:
            return outputs.loss
        # Physics scores the x̂₀ reconstruction; the collision term needs all
        # agents in one frame, so map the per-agent-local prediction back.
        prediction = from_agent_frame(outputs.prediction, reference_pose)
        physics_total, _ = self._physics.compute_loss(prediction, epoch, road_edges, valid_mask)
        if self.config.physics_x0_annealing:
            # Trust the physics score of x̂₀ in proportion to ᾱ_t, mirroring the
            # reconstruction-guidance anneal the sampling path applies.
            physics_total = outputs.alpha_bar_t * physics_total
        return outputs.loss + physics_total

    def sample(
        self,
        scene: Mapping[str, Any],
        agent_rows: jax.Array,
        *,
        reference_pose: jax.Array,
        key: jax.Array,
        guidance: GuidanceSpec | None = None,
    ) -> TrajectoryPrediction:
        """Generate trajectories for the selected agents conditioned on the map.

        The diffuser generates in each agent's local frame; the samples are
        mapped back to the shared (ego/world) frame via ``reference_pose`` so the
        returned trajectories are directly comparable to the logged futures.

        Args:
            scene: Raw WOD scenario dict.
            agent_rows: Anchor-row indices for the agents to generate.
            reference_pose: Per-agent current pose ``(num_agents, 3)`` defining
                each agent's local frame (must match training).
            key: JAX random key.
            guidance: Optional test-time x̂₀ guidance
                (:class:`~simulacrax.models.trajectory_diffusion.GuidanceSpec`).
                Its reward receives the clean estimate for all selected agents,
                shape ``(num_agents, future_steps, state_dim)``, in each agent's
                **local** frame (the diffuser's native frame); an inter-agent
                reward must map to the shared frame itself via ``reference_pose``.

        Returns:
            A :class:`TrajectoryPrediction` with one trajectory per selected
            agent row, in the shared frame.
        """
        scene_tokens, agent_emb, scene_token_valid = self._encode_scene(scene)
        anchor = agent_emb[agent_rows]
        local = self.diffusion.sample(
            anchor,
            key=key,
            scene_tokens=scene_tokens,
            scene_token_valid=scene_token_valid,
            guidance=guidance,
        )
        world = from_agent_frame(local.trajectories, reference_pose)
        return TrajectoryPrediction(trajectories=world, agent_ids=local.agent_ids)

    def denoising_log_prob(
        self,
        trajectories: jax.Array,
        agent_rows: jax.Array,
        scene: Mapping[str, Any],
        *,
        reference_pose: jax.Array,
        key: jax.Array,
        num_samples: int,
        freeze_tokenizer: bool = False,
    ) -> jax.Array:
        """Monte-Carlo diffusion log-probability proxy for Diffusion-DPO.

        Returns the negative mean denoising loss over ``num_samples`` stratified
        timesteps — the Diffusion-DPO log-prob estimator, evaluated in the
        model's own parameterization (so it is correct for x̂₀-prediction, not
        just ε). The trajectories are mapped into each agent's local frame (the
        diffuser's native target frame), and the scene conditioning is encoded
        once and shared across the samples.

        Args:
            trajectories: Trajectories in the shared frame ``(num_agents,
                future_steps, 4)`` to score (e.g. a chosen or rejected sample).
            agent_rows: Anchor-row indices for the scored agents.
            scene: Raw WOD scenario dict.
            reference_pose: Per-agent current pose ``(num_agents, 3)`` defining
                each agent's local frame (must match training).
            key: JAX random key.
            num_samples: Monte-Carlo (timestep, noise) draws; timesteps are
                stratified across the schedule for variance reduction.
            freeze_tokenizer: When ``True``, the scene conditioning is
                stop-gradiented so no gradient reaches the tokenizer — the
                Diffusion-DPO "shared, fixed conditioning" regime that adapts
                only the diffusion backbone. When ``False``, gradients flow into
                the tokenizer too (end-to-end fine-tuning of the map encoder).

        Returns:
            Scalar log-probability proxy (non-positive).
        """
        anchor, scene_tokens, scene_token_valid = self.encode_conditioning(
            scene, agent_rows, freeze_tokenizer=freeze_tokenizer
        )
        return self.log_prob_given_conditioning(
            trajectories,
            anchor,
            scene_tokens,
            scene_token_valid,
            reference_pose=reference_pose,
            key=key,
            num_samples=num_samples,
        )

    def encode_conditioning(
        self,
        scene: Mapping[str, Any],
        agent_rows: jax.Array,
        *,
        freeze_tokenizer: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Encode the shared scene conditioning for a set of agents.

        Splitting the encode out of :meth:`denoising_log_prob` lets a caller
        scoring many trajectory sets against one scene (e.g. the Diffusion-DPO
        loss over a scene's preference pairs) tokenize it **once** and reuse the
        result — the map encoder is the model's heaviest activation, so
        re-encoding per pair is what makes a batched loss exhaust GPU memory.

        Args:
            scene: Raw WOD scenario dict.
            agent_rows: Anchor-row indices for the scored agents.
            freeze_tokenizer: When ``True``, stop-gradient the scene tokens *and*
                the anchor so no gradient reaches the tokenizer (the Diffusion-DPO
                fixed-conditioning regime that adapts only the backbone).

        Returns:
            ``(anchor, scene_tokens, scene_token_valid)`` — the per-agent anchor
            ``(num_agents, embed_dim)``, the fused scene tokens and their validity
            mask, ready to pass to :meth:`log_prob_given_conditioning`.
        """
        scene_tokens, agent_emb, scene_token_valid = self._encode_scene(scene)
        anchor = agent_emb[agent_rows]
        if freeze_tokenizer:
            scene_tokens = jax.lax.stop_gradient(scene_tokens)
            anchor = jax.lax.stop_gradient(anchor)
        return anchor, scene_tokens, scene_token_valid

    def log_prob_given_conditioning(
        self,
        trajectories: jax.Array,
        anchor: jax.Array,
        scene_tokens: jax.Array,
        scene_token_valid: jax.Array,
        *,
        reference_pose: jax.Array,
        key: jax.Array,
        num_samples: int,
    ) -> jax.Array:
        """Monte-Carlo diffusion log-prob proxy given pre-encoded conditioning.

        The frame-mapping and stratified-timestep estimator of
        :meth:`denoising_log_prob`, but taking already-encoded conditioning so it
        can be shared across trajectory sets that condition on the same scene.

        Args:
            trajectories: Trajectories in the shared frame ``(num_agents,
                future_steps, 4)`` to score.
            anchor: Per-agent anchor from :meth:`encode_conditioning`.
            scene_tokens: Fused scene tokens from :meth:`encode_conditioning`.
            scene_token_valid: Scene-token validity mask.
            reference_pose: Per-agent current pose ``(num_agents, 3)``.
            key: JAX random key.
            num_samples: Monte-Carlo (timestep, noise) draws.

        Returns:
            Scalar log-probability proxy (non-positive).
        """
        local = to_agent_frame(trajectories, reference_pose)

        @jax.checkpoint
        def sample_loss(index: jax.Array, step_key: jax.Array) -> jax.Array:
            """Denoising loss at the ``index``-th stratified timestep."""
            return self.diffusion.compute_loss_outputs(
                local,
                anchor,
                key=step_key,
                scene_tokens=scene_tokens,
                scene_token_valid=scene_token_valid,
                timestep_stratum=(index, num_samples),
                deterministic=True,
            ).loss

        def body(index: jax.Array, carry: jax.Array) -> jax.Array:
            """Accumulate the denoising loss for one stratified sample."""
            return carry + sample_loss(index, jax.random.fold_in(key, index))

        total = jax.lax.fori_loop(0, num_samples, body, jnp.array(0.0))
        return -(total / num_samples)


@dataclass(frozen=True, slots=True, kw_only=True)
class MapConditionedBuildSpec:
    """Architecture dimensions and physics weights a training run varies.

    The non-varied architecture *policy* — x̂₀ prediction, map cross-attention,
    unit-variance state normalization, and a kinematic+collision-only physics
    penalty — is fixed by :func:`build_map_conditioned_model`; this spec carries
    only the dimensions and weights a run configures. :meth:`from_run_config`
    re-types the stringified ``vars(args)`` a run records in ``run_config.json``.

    Attributes:
        hidden_dim: Backbone hidden width; also the tokenizer ``embed_dim`` and
            the diffusion ``context_dim``/``scene_token_dim`` (the coupling the
            composed config enforces).
        num_heads: Attention heads (tokenizer and diffusion).
        num_blocks: Diffusion backbone blocks.
        num_temporal_layers: Per-block temporal attention layers.
        num_social_layers: Per-block social (cross-agent) attention layers.
        max_agents: Maximum agents per scene.
        future_steps: Predicted future horizon.
        diffusion_steps: Number of diffusion timesteps.
        kinematic_weight: Bicycle-model kinematic penalty weight on x̂₀.
        collision_weight: Inter-agent collision penalty weight on x̂₀.
        gradient_checkpointing: Rematerialize backbone blocks to save memory.
        history_steps: History length the tokenizer consumes; defaults to the
            canonical WOD value (not a per-run argument).
    """

    hidden_dim: int
    num_heads: int
    num_blocks: int
    num_temporal_layers: int
    num_social_layers: int
    max_agents: int
    future_steps: int
    diffusion_steps: int
    kinematic_weight: float
    collision_weight: float
    gradient_checkpointing: bool = False
    history_steps: int = WOD_HISTORY_STEPS

    @classmethod
    def from_run_config(cls, run_config: Mapping[str, str]) -> Self:
        """Rebuild a spec from a run's stringified ``run_config.json``.

        A training run records ``vars(args)`` with every value stringified, so
        each field is parsed back to its type. Extra keys (learning rate, data
        split, …) that do not shape the architecture are ignored.

        Args:
            run_config: The stringified argument mapping a run recorded.

        Returns:
            The typed build spec.
        """
        return cls(
            hidden_dim=int(run_config["hidden_dim"]),
            num_heads=int(run_config["num_heads"]),
            num_blocks=int(run_config["num_blocks"]),
            num_temporal_layers=int(run_config["num_temporal_layers"]),
            num_social_layers=int(run_config["num_social_layers"]),
            max_agents=int(run_config["max_agents"]),
            future_steps=int(run_config["future_steps"]),
            diffusion_steps=int(run_config["diffusion_steps"]),
            kinematic_weight=float(run_config["kinematic_weight"]),
            collision_weight=float(run_config["collision_weight"]),
            # bool("False") is truthy, so compare the recorded string explicitly.
            gradient_checkpointing=run_config.get("gradient_checkpointing", "False") == "True",
        )


def build_map_conditioned_model(
    spec: MapConditionedBuildSpec,
    *,
    rngs: nnx.Rngs,
    element_spec: Mapping[str, object] | None = None,
) -> MapConditionedTrajectoryModel:
    """Build a map-conditioned model under the fixed training-recipe policy.

    Centralizes the tokenizer/diffusion/physics coupling so the training script
    and the checkpoint loader construct byte-identical architectures. The policy
    held constant here (predict x̂₀, map cross-attention on, no x̂₀ clamp under
    unit-variance normalization, kinematic+collision physics on x̂₀ without the
    boundary/momentum terms) matches the tier-0 recipe the checkpoints were
    trained under; the architecture-version guard rejects any structural drift.

    Args:
        spec: Dimensions and physics weights to build.
        rngs: NNX random number generators.
        element_spec: Optional data-source element spec for tokenizer modality
            negotiation. WOD carries no sensors, so ``None`` (the default) works
            for the WOD showcase.

    Returns:
        The constructed model (freshly initialized; restore weights separately).
    """
    tokenizer_config = TokenizerConfig(
        embed_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        history_steps=spec.history_steps,
        lidar_modality=ModalityMode.OFF,
        camera_modality=ModalityMode.OFF,
    )
    diffusion_config = TrajectoryDiffusionConfig(
        hidden_dim=spec.hidden_dim,
        num_heads=spec.num_heads,
        num_blocks=spec.num_blocks,
        num_temporal_layers=spec.num_temporal_layers,
        num_social_layers=spec.num_social_layers,
        num_agents_max=spec.max_agents,
        future_steps=spec.future_steps,
        num_timesteps=spec.diffusion_steps,
        context_dim=spec.hidden_dim,
        scene_token_dim=spec.hidden_dim,
        use_map_cross_attention=True,
        gradient_checkpointing=spec.gradient_checkpointing,
        state_offsets=AGENT_LOCAL_STATE_OFFSETS,
        state_scales=AGENT_LOCAL_STATE_SCALES,
        # No x̂₀ clamp: targets are unit-variance-normalized (not bounded like
        # image data), so any finite clamp would chop legitimate-manoeuvre tails.
        x0_clip_bound=None,
        snr_gamma=5.0,
        # Predict the clean trajectory (x̂₀) — the SOTA parameterization for
        # trajectory diffusion (MotionDiffuser, VBD).
        prediction_type=PredictionType.X0,
    )
    physics_config = SimulacraxPhysicsConfig(
        kinematic_weight=spec.kinematic_weight,
        collision_weight=spec.collision_weight,
        # Off-road is an eval/guidance concern with its own representation;
        # momentum is weighted out for iteration 1. The ᾱ_t anneal trusts the
        # x̂₀ physics score by confidence, so a constant weight suffices.
        road_boundary_weight=0.0,
        momentum_variation_weight=0.0,
        adaptive_weighting=False,
    )
    config = MapConditionedTrajectoryConfig(
        tokenizer=tokenizer_config,
        diffusion=diffusion_config,
        physics=physics_config,
        physics_x0_annealing=True,
    )
    return MapConditionedTrajectoryModel(config, rngs=rngs, element_spec=element_spec)


def load_map_conditioned_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    rngs: nnx.Rngs,
) -> MapConditionedTrajectoryModel:
    """Rebuild and restore a map-conditioned model from a checkpoint directory.

    Reads the ``run_config.json`` a training run wrote beside its checkpoints,
    rebuilds the exact architecture via :func:`build_map_conditioned_model`, and
    restores the latest saved weights (the EMA-smoothed weights a run persists),
    rejecting a checkpoint whose backbone version differs from the running code.

    Args:
        checkpoint_dir: Directory holding ``run_config.json`` and the Orbax
            checkpoints.
        rngs: NNX random number generators for the freshly built model (its
            initialization is overwritten by the restored weights).

    Returns:
        The model with restored weights.

    Raises:
        FileNotFoundError: If no checkpoint exists in ``checkpoint_dir``.
        CheckpointCorruptError: If the checkpoint is unreadable, its payload
            structure does not match, or its architecture version differs.
    """
    checkpoint_dir = Path(checkpoint_dir)
    run_config = json.loads((checkpoint_dir / "run_config.json").read_text())
    spec = MapConditionedBuildSpec.from_run_config(run_config)
    model = build_map_conditioned_model(spec, rngs=rngs)
    manager_config = CheckpointConfig(checkpoint_dir=str(checkpoint_dir))
    with SimulacraxCheckpointManager(manager_config) as manager:
        state = manager.restore_latest(
            model, expected_architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION
        )
    if state is None:
        msg = f"No checkpoint found in {str(checkpoint_dir)!r}"
        raise FileNotFoundError(msg)
    return model
