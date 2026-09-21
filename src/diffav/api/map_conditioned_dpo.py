"""Map-conditioned Diffusion-DPO: preference fine-tuning of the scene diffuser.

Extends Diffusion-DPO to the map-conditioned trajectory model. The preference
log-probabilities condition on the *structured* scene (fused tokens + per-agent
anchor) rather than a flat context, and are computed in each agent's local
frame — the diffuser's native target frame. A ``freeze_tokenizer`` toggle
controls the central ablation: whether the DPO gradient adapts only the
diffusion backbone (frozen map encoder, the Diffusion-DPO / Gen-Drive norm) or
flows end-to-end into the tokenizer as well (optimizing the raw-data→result
pipeline through the map encoder).

The DPO objective algebra is reused from
:func:`~diffav.alignment.dpo_trainer.dpo_loss_from_log_probs`; only the
log-probability estimation is map-conditioned here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.optim import create_optimizer, OptimizerConfig

from diffav.alignment.dpo_trainer import dpo_loss_from_log_probs, DPOAlignmentConfig
from diffav.api.map_conditioned import MapConditionedTrajectoryModel
from diffav.core.training_utils import nan_safe_gradients
from diffav.evaluation.map_conditioned_evaluator import ValidationScene


def _scene_pair_log_probs(
    model: MapConditionedTrajectoryModel,
    scenes: Any,
    agent_rows: jax.Array,
    reference_pose: jax.Array,
    chosen: jax.Array,
    rejected: jax.Array,
    *,
    key: jax.Array,
    num_samples: int,
    freeze_tokenizer: bool,
) -> tuple[jax.Array, jax.Array]:
    """Chosen/rejected log-probs, encoding each scene's conditioning once.

    For every scene the map is tokenized a single time and that conditioning is
    shared across all of the scene's preference pairs (inner ``vmap``) and both
    the chosen and rejected sets — the Diffusion-DPO fixed-``c`` rule. Re-encoding
    per pair (the previous flat-batch design) materialized the heaviest activation
    once per pair and overflowed GPU memory as scenes accumulated.

    Args:
        model: The map-conditioned model.
        scenes: Stacked raw scene dicts, leading axis ``S``.
        agent_rows: Anchor rows ``(S, A)``.
        reference_pose: Per-agent poses ``(S, A, 3)``.
        chosen: Chosen trajectory sets ``(S, P, A, T, 4)``.
        rejected: Rejected trajectory sets ``(S, P, A, T, 4)``.
        key: JAX random key.
        num_samples: Monte-Carlo draws per log-prob.
        freeze_tokenizer: Freeze the map encoder for this evaluation.

    Returns:
        ``(chosen_log_probs, rejected_log_probs)``, each ``(S, P)``.
    """

    def per_scene(
        scene: Any, rows: jax.Array, pose: jax.Array, chosen_set: jax.Array, rejected_set: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Encode one scene once, then score all its pairs against that ``c``."""
        anchor, scene_tokens, scene_token_valid = model.encode_conditioning(
            scene, rows, freeze_tokenizer=freeze_tokenizer
        )

        def log_prob(trajectories: jax.Array) -> jax.Array:
            """Shared-conditioning log-prob for one trajectory set."""
            return model.log_prob_given_conditioning(
                trajectories,
                anchor,
                scene_tokens,
                scene_token_valid,
                reference_pose=pose,
                key=key,
                num_samples=num_samples,
            )

        return jax.vmap(log_prob)(chosen_set), jax.vmap(log_prob)(rejected_set)

    return jax.vmap(per_scene)(scenes, agent_rows, reference_pose, chosen, rejected)


def map_conditioned_dpo_loss(
    policy_model: MapConditionedTrajectoryModel,
    reference_model: MapConditionedTrajectoryModel | None,
    batch: Mapping[str, Any],
    key: jax.Array,
    *,
    config: DPOAlignmentConfig,
    num_samples: int,
    freeze_tokenizer: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Map-conditioned Diffusion-DPO loss over a scene-grouped batch of pairs.

    Each scene's map is encoded once and shared across all of its preference
    pairs and both the chosen and rejected sets, under the policy (and, unless
    ``config.reference_free``, the frozen reference). Policy and reference share
    the estimator ``key`` so per-sample Monte Carlo noise cancels in the
    log-ratio.

    Args:
        policy_model: The map-conditioned policy being fine-tuned.
        reference_model: Frozen reference for the KL anchor; ``None`` only in
            ``config.reference_free`` mode.
        batch: Scene-grouped mapping with a leading scene axis ``S`` and a
            per-scene pair axis ``P``: ``"scene"`` (stacked raw scene dicts),
            ``"agent_rows"`` ``(S, A)``, ``"chosen"`` / ``"rejected"``
            ``(S, P, A, T, 4)`` in the shared frame, and ``"reference_pose"``
            ``(S, A, 3)``.
        key: JAX random key.
        config: DPO configuration (beta, reference_free, label smoothing, …).
        num_samples: Monte-Carlo timestep/noise draws per log-prob.
        freeze_tokenizer: Passed to the policy log-prob — freeze the map encoder
            (backbone-only DPO) or let gradients flow into it.

    Returns:
        Tuple of ``(scalar_loss, aux)`` with the implicit-reward metrics.
    """
    scenes = batch["scene"]
    rows = batch["agent_rows"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    pose = batch["reference_pose"]

    policy_chosen, policy_rejected = _scene_pair_log_probs(
        policy_model,
        scenes,
        rows,
        pose,
        chosen,
        rejected,
        key=key,
        num_samples=num_samples,
        freeze_tokenizer=freeze_tokenizer,
    )
    policy_chosen_lp = policy_chosen.reshape(-1)
    policy_rejected_lp = policy_rejected.reshape(-1)

    reference_chosen_lp: jax.Array | None = None
    reference_rejected_lp: jax.Array | None = None
    if not config.reference_free:
        if reference_model is None:
            msg = "reference_model is required when reference_free=False"
            raise ValueError(msg)
        reference_chosen, reference_rejected = _scene_pair_log_probs(
            reference_model,
            scenes,
            rows,
            pose,
            chosen,
            rejected,
            key=key,
            num_samples=num_samples,
            freeze_tokenizer=True,
        )
        reference_chosen_lp = reference_chosen.reshape(-1)
        reference_rejected_lp = reference_rejected.reshape(-1)

    return dpo_loss_from_log_probs(
        policy_chosen_lp,
        policy_rejected_lp,
        reference_chosen_lp,
        reference_rejected_lp,
        config,
    )


def assemble_dpo_batch(
    entries: Sequence[tuple[ValidationScene, jax.Array, jax.Array]],
) -> dict[str, Any]:
    """Stack per-scene preference pairs into a scene-grouped DPO batch.

    Scenes stack along a leading axis ``S`` and their pairs along a per-scene axis
    ``P`` — the structure :func:`map_conditioned_dpo_loss` needs to encode each
    scene once and share that conditioning across its pairs. Every scene must
    contribute the same number of pairs so the pair axis is rectangular (the
    steering spine emits a fixed ``num_pairs`` per scene).

    Args:
        entries: One ``(scene, chosen, rejected)`` per source scene, with
            ``chosen`` / ``rejected`` shaped ``(num_pairs, num_agents,
            future_steps, 4)`` in the shared frame.

    Returns:
        A batch mapping: ``"scene"`` (raw scene dicts stacked to leading ``S``),
        ``"agent_rows"`` ``(S, A)``, ``"reference_pose"`` ``(S, A, 3)``, and
        ``"chosen"`` / ``"rejected"`` ``(S, P, A, T, 4)``.

    Raises:
        ValueError: If ``entries`` is empty or scenes contribute differing pair
            counts.
    """
    if not entries:
        msg = "assemble_dpo_batch requires at least one scene entry"
        raise ValueError(msg)

    num_pairs = entries[0][1].shape[0]
    if any(
        chosen.shape[0] != num_pairs or rejected.shape[0] != num_pairs
        for _, chosen, rejected in entries
    ):
        msg = "assemble_dpo_batch requires every scene to contribute the same number of pairs"
        raise ValueError(msg)

    scene_dicts = [dict(scene.scene) for scene, _, _ in entries]
    return {
        "scene": jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *scene_dicts),
        "agent_rows": jnp.stack([jnp.asarray(scene.agent_rows) for scene, _, _ in entries]),
        "reference_pose": jnp.stack([jnp.asarray(scene.reference_pose) for scene, _, _ in entries]),
        "chosen": jnp.stack([chosen for _, chosen, _ in entries]),
        "rejected": jnp.stack([rejected for _, _, rejected in entries]),
    }


def build_dpo_arm(
    model: MapConditionedTrajectoryModel,
    *,
    learning_rate: float = 1e-6,
    gradient_clip: float = 1.0,
) -> tuple[MapConditionedTrajectoryModel, MapConditionedTrajectoryModel, nnx.Optimizer]:
    """Clone the fine-tuning policy and frozen reference for one ablation arm.

    Both arms of the freeze-vs-learnable ablation share this construction; they
    differ only in the ``freeze_tokenizer`` flag passed to
    :func:`map_conditioned_dpo_step`. That flag stop-gradients the encoded scene
    (tokens *and* anchor) in the frozen arm, so the tokenizer receives exactly
    zero gradient and Adam leaves it untouched — the optimizer therefore
    optimizes ``nnx.Param`` in both arms and the freeze is carried entirely by
    the toggle (a single-variable ablation).

    Args:
        model: The base map-conditioned model to fine-tune (left untouched).
        learning_rate: Adam learning rate. Defaults to ``1e-6`` — the
            over-optimization-safe rate root-caused in the ablation; larger
            rates (the earlier ``1e-4``) drive the policy into reward hacking.
        gradient_clip: Global-norm gradient clip.

    Returns:
        Tuple of ``(policy, reference, optimizer)`` — ``policy`` and
        ``reference`` are independent clones; ``reference`` is the frozen KL
        anchor.
    """
    policy = nnx.clone(model)
    reference = nnx.clone(model)
    optimizer = create_optimizer(
        policy,
        OptimizerConfig(
            optimizer_type="adam", learning_rate=learning_rate, gradient_clip_norm=gradient_clip
        ),
    )
    return policy, reference, optimizer


@partial(nnx.jit, static_argnames=("config", "num_samples", "freeze_tokenizer"))
def _chunk_value_and_grad(
    policy_model: MapConditionedTrajectoryModel,
    reference_model: MapConditionedTrajectoryModel | None,
    sub_batch: Mapping[str, Any],
    key: jax.Array,
    *,
    config: DPOAlignmentConfig,
    num_samples: int,
    freeze_tokenizer: bool,
) -> tuple[tuple[jax.Array, dict[str, jax.Array]], Any]:
    """Jitted (value, aux), gradient of the DPO loss over one scene chunk.

    Compiled once per chunk shape (and ``config`` / ``num_samples`` /
    ``freeze_tokenizer``) and reused across every chunk and optimizer step, which
    is the difference between a seconds-long and an hours-long accumulation run
    (the eager loop dispatches each op unfused for every scene of every step).
    The unweighted gradient is returned; the caller scales it by the chunk's pair
    share, so a ragged final chunk does not trigger a recompile.
    """

    def loss_fn(model: MapConditionedTrajectoryModel) -> tuple[jax.Array, dict[str, jax.Array]]:
        """DPO loss of the traced policy over this chunk."""
        return map_conditioned_dpo_loss(
            model,
            reference_model,
            sub_batch,
            key,
            config=config,
            num_samples=num_samples,
            freeze_tokenizer=freeze_tokenizer,
        )

    return nnx.value_and_grad(loss_fn, has_aux=True)(policy_model)


def _slice_scene_batch(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Take scenes ``[start:stop]`` from a scene-grouped batch."""
    return {
        "scene": jax.tree.map(lambda leaf: leaf[start:stop], batch["scene"]),
        "agent_rows": batch["agent_rows"][start:stop],
        "reference_pose": batch["reference_pose"][start:stop],
        "chosen": batch["chosen"][start:stop],
        "rejected": batch["rejected"][start:stop],
    }


def map_conditioned_dpo_step(
    policy_model: MapConditionedTrajectoryModel,
    reference_model: MapConditionedTrajectoryModel | None,
    optimizer: nnx.Optimizer,
    batch: Mapping[str, Any],
    key: jax.Array,
    *,
    config: DPOAlignmentConfig,
    num_samples: int,
    freeze_tokenizer: bool,
    scene_microbatch: int = 0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Apply one map-conditioned Diffusion-DPO update over a scene-grouped batch.

    Differentiates :func:`map_conditioned_dpo_loss` with respect to the policy,
    zeros the gradients on any NaN, and applies the optimizer update in place.
    The ``freeze_tokenizer`` flag is the ablation's single independent variable:
    ``True`` freezes the map encoder (backbone-only DPO), ``False`` lets the
    gradient flow end-to-end into the tokenizer.

    Peak training memory grows linearly with the number of scenes evaluated in a
    single backward pass. ``scene_microbatch`` splits the batch's scenes into
    chunks, differentiates each chunk separately, and accumulates the gradients —
    the standard Diffusion-DPO recipe (local batch of one, gradient accumulation
    to the effective batch). Each chunk's gradient is weighted by its share of the
    total pairs, so the accumulated update equals the full-batch update exactly,
    even when the final chunk is short (the mean-over-pairs DPO loss would
    otherwise over-weight a ragged chunk).

    Args:
        policy_model: The policy being fine-tuned (updated in place).
        reference_model: Frozen reference for the KL anchor; ``None`` only in
            ``config.reference_free`` mode.
        optimizer: NNX optimizer wrapping ``policy_model``.
        batch: Scene-grouped batch from :func:`assemble_dpo_batch`.
        key: JAX random key.
        config: DPO configuration.
        num_samples: Monte-Carlo timestep/noise draws per log-prob.
        freeze_tokenizer: Freeze the map encoder (frozen arm) or adapt it
            end-to-end (learnable arm).
        scene_microbatch: Scenes per backward pass; ``0`` (default) evaluates all
            scenes at once, matching the un-accumulated behavior.

    Returns:
        Tuple of ``(loss, aux)`` with the implicit-reward metrics plus
        ``grad_norm`` and ``has_nan``. ``loss`` and the reward metrics equal the
        full-batch values regardless of ``scene_microbatch``.
    """
    num_scenes = int(batch["chosen"].shape[0])
    pairs_per_scene = int(batch["chosen"].shape[1])
    chunk = num_scenes if scene_microbatch <= 0 else min(scene_microbatch, num_scenes)
    total_pairs = num_scenes * pairs_per_scene

    total_loss = jnp.array(0.0)
    accumulated: Any = None
    aggregated: dict[str, jax.Array] = {}
    for start in range(0, num_scenes, chunk):
        stop = min(start + chunk, num_scenes)
        sub_batch = _slice_scene_batch(batch, start, stop)
        weight = ((stop - start) * pairs_per_scene) / total_pairs

        # The per-chunk value_and_grad is jitted (compiled once per chunk shape and
        # reused across chunks and steps — a >100x speedup over the eager loop),
        # while the pair-count weight is applied outside so a ragged final chunk
        # does not force a recompile.
        (chunk_loss, aux), grads = _chunk_value_and_grad(
            policy_model,
            reference_model,
            sub_batch,
            key,
            config=config,
            num_samples=num_samples,
            freeze_tokenizer=freeze_tokenizer,
        )
        total_loss = total_loss + chunk_loss * weight
        weighted_grads = jax.tree_util.tree_map(lambda g, w=weight: g * w, grads)
        accumulated = (
            weighted_grads
            if accumulated is None
            else jax.tree_util.tree_map(jnp.add, accumulated, weighted_grads)
        )
        for name, value in aux.items():
            aggregated[name] = aggregated.get(name, jnp.array(0.0)) + weight * value

    grad_leaves = jax.tree_util.tree_leaves(accumulated)
    squared_norms = [jnp.sum(g**2) for g in grad_leaves if hasattr(g, "shape")]
    grad_norm = jnp.sqrt(sum(squared_norms)) if squared_norms else jnp.array(0.0)

    safe_grads, has_nan = nan_safe_gradients(total_loss, accumulated)
    optimizer.update(policy_model, safe_grads)

    aggregated["grad_norm"] = grad_norm
    aggregated["has_nan"] = has_nan
    return total_loss, aggregated
