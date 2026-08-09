"""Serious training entrypoint for the map-conditioned trajectory model.

Composes the map-conditioned model with a warmup-cosine AdamW optimizer, an
exponential moving average of the parameters, periodic held-out validation, and
the caller's checkpointing. The loop is house-style: a jitted step over
``(model, optimizer, batch)`` with ``nnx.value_and_grad`` then
``optimizer.update``; the EMA is updated outside ``jit`` (an in-place attribute
write); and validation runs against the smoothed weights via
``ParamEMA.swap_in``. Validation is an injected callback, so this module does
not depend on the evaluation layer.

The optimizer is the standard optax composition
(``adamw(learning_rate=warmup_cosine_schedule)`` behind global-norm clipping)
rather than a config-builder that would double-apply the learning rate; the EMA
reuses ``opifex``'s parameter-filtered :class:`ParamEMA`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from opifex.neural.atomistic.training import ParamEMA

from diffav.api.map_conditioned import MapConditionedTrajectoryModel
from diffav.core.config import validate_positive


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainBatch:
    """One batched training step's inputs, stacked along a leading scene axis.

    Attributes:
        scene: Batched raw tokenizer keys; each value ``(batch, ...)``.
        trajectories: Ground-truth futures, ``(batch, num_agents, future_steps,
            state_dim)``.
        agent_rows: Anchor rows per scene, ``(batch, num_agents)``.
        valid: Per-step validity, ``(batch, num_agents, future_steps)``.
        reference_pose: Per-agent current pose, ``(batch, num_agents, 3)``,
            defining each agent's local frame for the diffusion target.
    """

    scene: Mapping[str, jax.Array]
    trajectories: jax.Array
    agent_rows: jax.Array
    valid: jax.Array
    reference_pose: jax.Array


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConfig:
    """Optimizer, EMA, and schedule configuration for the training loop.

    Attributes:
        total_steps: Total optimizer steps; the cosine decay spans them.
        peak_learning_rate: Peak learning rate at the end of warmup.
        warmup_steps: Linear warmup steps from zero to the peak.
        final_lr_fraction: Cosine end value as a fraction of the peak.
        weight_decay: AdamW decoupled weight decay.
        gradient_clip: Global-norm gradient clip applied before AdamW.
        ema_decay: Parameter EMA decay in ``[0, 1)``.
        eval_every: Run the validation callback every this many steps
            (``0`` disables periodic validation).
    """

    total_steps: int
    peak_learning_rate: float = 2e-4
    warmup_steps: int = 1000
    final_lr_fraction: float = 0.1
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    eval_every: int = 2000

    def __post_init__(self) -> None:
        """Validate the schedule and EMA bounds."""
        validate_positive("total_steps", self.total_steps)
        validate_positive("peak_learning_rate", self.peak_learning_rate)
        if self.warmup_steps < 0:
            msg = f"warmup_steps must be non-negative, got {self.warmup_steps}"
            raise ValueError(msg)
        if self.warmup_steps >= self.total_steps:
            msg = (
                f"warmup_steps ({self.warmup_steps}) must be less than total_steps "
                f"({self.total_steps}) so the cosine decay spans at least one step"
            )
            raise ValueError(msg)
        if not 0.0 <= self.final_lr_fraction <= 1.0:
            msg = f"final_lr_fraction must be in [0, 1], got {self.final_lr_fraction}"
            raise ValueError(msg)
        if not 0.0 <= self.ema_decay < 1.0:
            msg = f"ema_decay must be in [0, 1), got {self.ema_decay}"
            raise ValueError(msg)


def build_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Global-norm-clipped AdamW under a warmup-cosine learning-rate schedule.

    Args:
        config: Training configuration.

    Returns:
        The composed optax gradient transformation.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=config.peak_learning_rate * config.final_lr_fraction,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )


def _train_step(
    model: MapConditionedTrajectoryModel,
    optimizer: nnx.Optimizer,
    scene: Mapping[str, jax.Array],
    trajectories: jax.Array,
    agent_rows: jax.Array,
    valid: jax.Array,
    reference_pose: jax.Array,
    epoch: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """One batched joint step: mean per-scene loss, gradients, AdamW update."""

    def loss_fn(traced_model: MapConditionedTrajectoryModel) -> jax.Array:
        keys = jax.random.split(key, trajectories.shape[0])
        losses = jax.vmap(
            lambda sc, tr, rw, vm, rp, k: traced_model.compute_loss(
                sc, tr, rw, reference_pose=rp, key=k, epoch=epoch, valid_mask=vm
            )
        )(scene, trajectories, agent_rows, valid, reference_pose, keys)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


ValidationCallback = Callable[[MapConditionedTrajectoryModel], Mapping[str, float]]


class MapConditionedTrainer:
    """Trains a map-conditioned model with warmup-cosine AdamW, EMA, and val-eval."""

    def __init__(self, model: MapConditionedTrajectoryModel, config: TrainingConfig) -> None:
        """Build the optimizer and the parameter EMA for ``model``.

        Args:
            model: The map-conditioned model to train (updated in place).
            config: Training configuration.
        """
        self.model = model
        self._config = config
        self.optimizer = nnx.Optimizer(model, build_optimizer(config), wrt=nnx.Param)
        # ParamEMA is parameter-filtered and model-agnostic (it only reads/writes
        # nnx.Param via nnx.state / nnx.update); cast past its opifex-specific
        # annotation.
        self._ema = ParamEMA(cast(Any, model), decay=config.ema_decay)
        self._jit_step = nnx.jit(_train_step)
        self._step = 0

    @property
    def ema(self) -> ParamEMA:
        """The parameter exponential moving average."""
        return self._ema

    def fit(
        self,
        batches: Iterable[TrainBatch],
        *,
        key: jax.Array,
        validation_callback: ValidationCallback | None = None,
        on_step: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Train over the batches, updating the EMA and running periodic eval.

        The training loss is kept as an on-device array rather than blocked on
        each step, so JAX async-dispatches the steps and the GPU runs ahead
        (the flax nnx loop convention). Callers realize it (``float(...)``) only
        when they log or checkpoint, which is the loop's only periodic sync.

        Args:
            batches: Iterable of batched training inputs.
            key: JAX random key; each step folds in its index.
            validation_callback: Optional callable scoring the model; run every
                ``config.eval_every`` steps against the EMA weights (via
                ``ParamEMA.swap_in``) and merged into that step's record.
            on_step: Optional callback invoked after each step with the step
                index and its record (``loss`` is an on-device array), e.g. for
                logging or checkpointing.

        Returns:
            One record per step with the step index, the training loss (an
            on-device array), and any validation metrics on evaluation steps.
        """
        history: list[dict[str, Any]] = []
        # Train mode enables encoder/tokenizer dropout when ``dropout_rate > 0``;
        # at the default rate of 0.0 it is a numerical no-op. Restored to eval
        # mode on exit so a subsequent ``sample`` runs deterministically.
        self.model.train()
        try:
            history = self._run_epoch(batches, key, validation_callback, on_step, history)
        finally:
            self.model.eval()
        return history

    def _run_epoch(
        self,
        batches: Iterable[TrainBatch],
        key: jax.Array,
        validation_callback: ValidationCallback | None,
        on_step: Callable[[int, dict[str, Any]], None] | None,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run the training loop body; see :meth:`fit` for semantics."""
        for batch in batches:
            step_key = jax.random.fold_in(key, self._step)
            loss = self._jit_step(
                self.model,
                self.optimizer,
                batch.scene,
                batch.trajectories,
                batch.agent_rows,
                batch.valid,
                batch.reference_pose,
                jnp.asarray(self._step),
                step_key,
            )
            self._ema.update(cast(Any, self.model))
            record: dict[str, Any] = {"step": self._step, "loss": loss}

            if (
                validation_callback is not None
                and self._config.eval_every > 0
                and (self._step + 1) % self._config.eval_every == 0
            ):
                # Evaluate deterministically (dropout off) even mid-training.
                self.model.eval()
                try:
                    with self._ema.swap_in(cast(Any, self.model)):
                        record.update(validation_callback(self.model))
                finally:
                    self.model.train()

            if on_step is not None:
                on_step(self._step, record)
            history.append(record)
            self._step += 1
        return history
