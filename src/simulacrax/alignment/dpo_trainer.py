"""DPO fine-tuning trainer for trajectory diffusion models.

Implements the Diffusion-DPO approach (Wallace et al. 2023) for aligning
trajectory generation toward safer, physically plausible scenarios.
Unlike discrete-token DPO, log-probabilities are estimated via Monte Carlo
averaging of negative diffusion noise prediction error:

    log p_theta(x) ~ -E_{t,eps}[ ||eps - eps_theta(x_t, t)||^2 ]

The trainer supports:
- Standard DPO with a frozen reference model (via ``nnx.clone``)
- Reference-free SimPO mode (``reference_free=True``)
- Optional physics regularisation using ``SimulacraxPhysicsLoss``
- NaN-safe gradient handling following ``TrajectoryTrainer`` patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from simulacrax.core.config import validate_positive
from simulacrax.core.training_utils import nan_safe_gradients
from simulacrax.models.sampling_utils import stratified_timestep
from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionModel
from simulacrax.physics.losses import SimulacraxPhysicsLoss


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class DPOAlignmentConfig:
    """Configuration for DPO alignment training.

    Attributes:
        beta: DPO temperature controlling preference strength. Diffusion
            log-prob proxies are MSE-scale, so useful values are far larger
            than token-DPO's 0.1-0.5 (Diffusion-DPO uses 2000-5000).
        label_smoothing: Robust DPO label smoothing in ``[0, 0.5)``.
        reference_free: Drop the reference-model anchor (CPO/SimPO family).
        simpo_gamma: SimPO target reward margin, already beta-scaled;
            subtracted from the logits in reference-free mode. ``0``
            recovers plain reference-free DPO.
        num_log_prob_samples: Monte Carlo samples for diffusion log-prob.
            Timesteps are stratified: each of the K draws lands in its own
            bin of the schedule, reducing estimator variance without bias.
        physics_weight: Weight for physics regularisation loss.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
    simpo_gamma: float = 0.0
    num_log_prob_samples: int = 4
    physics_weight: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        validate_positive("beta", self.beta)
        if self.label_smoothing < 0 or self.label_smoothing >= 0.5:
            msg = f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}"
            raise ValueError(msg)
        if self.simpo_gamma < 0:
            msg = f"simpo_gamma must be non-negative, got {self.simpo_gamma}"
            raise ValueError(msg)
        validate_positive("num_log_prob_samples", self.num_log_prob_samples)
        if self.physics_weight < 0:
            msg = f"physics_weight must be non-negative, got {self.physics_weight}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class DPOAlignmentMetrics:
    """Immutable snapshot of metrics from a DPO training step.

    Attributes:
        dpo_loss: Total DPO loss (including physics regularisation).
        policy_chosen_log_prob: Mean log-prob of chosen trajectories.
        policy_rejected_log_prob: Mean log-prob of rejected trajectories.
        reward_accuracy: Fraction where chosen_logp > rejected_logp.
        reward_margin: Mean(chosen_logp - rejected_logp).
        grad_norm: Global gradient norm before optimizer update.
    """

    dpo_loss: float
    policy_chosen_log_prob: float
    policy_rejected_log_prob: float
    reward_accuracy: float
    reward_margin: float
    grad_norm: float


def _build_alignment_metrics(
    loss: jax.Array,
    aux: dict[str, jax.Array],
    grad_norm: float,
) -> DPOAlignmentMetrics:
    """Construct a ``DPOAlignmentMetrics`` from a loss scalar and aux dict.

    Args:
        loss: Scalar DPO loss array.
        aux: Auxiliary outputs containing log-prob and reward keys.
        grad_norm: Gradient norm before optimizer update.

    Returns:
        Immutable ``DPOAlignmentMetrics`` snapshot.
    """
    return DPOAlignmentMetrics(
        dpo_loss=float(loss),
        policy_chosen_log_prob=float(aux["policy_chosen_log_prob"]),
        policy_rejected_log_prob=float(aux["policy_rejected_log_prob"]),
        reward_accuracy=float(aux["reward_accuracy"]),
        reward_margin=float(aux["reward_margin"]),
        grad_norm=grad_norm,
    )


def dpo_loss_from_log_probs(
    policy_chosen_log_prob: jax.Array,
    policy_rejected_log_prob: jax.Array,
    reference_chosen_log_prob: jax.Array | None,
    reference_rejected_log_prob: jax.Array | None,
    config: DPOAlignmentConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """DPO preference loss and reward metrics from four log-prob arrays.

    The parameterization-agnostic core of the DPO objective, shared by the
    bare-context and map-conditioned trainers: it forms the (reference-anchored
    or reference-free) log-ratios, the ``beta``-scaled logits with optional
    SimPO margin and label smoothing, and the implicit-reward metrics. The
    log-prob *estimation* (which differs by conditioning and parameterization)
    is the caller's responsibility.

    Args:
        policy_chosen_log_prob: Policy log-prob proxies for chosen ``(batch,)``.
        policy_rejected_log_prob: Policy log-prob proxies for rejected.
        reference_chosen_log_prob: Reference log-probs for chosen, or ``None``
            in reference-free mode.
        reference_rejected_log_prob: Reference log-probs for rejected, or
            ``None`` in reference-free mode.
        config: DPO configuration.

    Returns:
        Tuple of ``(scalar_loss, aux)`` where aux carries
        ``policy_chosen_log_prob``, ``policy_rejected_log_prob``,
        ``reward_accuracy``, and ``reward_margin``.

    Raises:
        ValueError: If reference log-probs are missing when
            ``reference_free`` is ``False``.
    """
    if config.reference_free:
        log_ratio_chosen = policy_chosen_log_prob
        log_ratio_rejected = policy_rejected_log_prob
    else:
        if reference_chosen_log_prob is None or reference_rejected_log_prob is None:
            msg = "reference log-probs are required when reference_free=False"
            raise ValueError(msg)
        log_ratio_chosen = policy_chosen_log_prob - reference_chosen_log_prob
        log_ratio_rejected = policy_rejected_log_prob - reference_rejected_log_prob

    logits = config.beta * (log_ratio_chosen - log_ratio_rejected)
    if config.reference_free and config.simpo_gamma > 0:
        logits = logits - config.simpo_gamma

    if config.label_smoothing > 0:
        ls = config.label_smoothing
        dpo_loss = (1 - ls) * (-jax.nn.log_sigmoid(logits)) + ls * (-jax.nn.log_sigmoid(-logits))
    else:
        dpo_loss = -jax.nn.log_sigmoid(logits)

    chosen_reward = config.beta * log_ratio_chosen
    rejected_reward = config.beta * log_ratio_rejected
    aux = {
        "policy_chosen_log_prob": jnp.mean(policy_chosen_log_prob),
        "policy_rejected_log_prob": jnp.mean(policy_rejected_log_prob),
        "reward_accuracy": jnp.mean((chosen_reward > rejected_reward).astype(jnp.float32)),
        "reward_margin": jnp.mean(chosen_reward - rejected_reward),
    }
    return jnp.mean(dpo_loss), aux


def create_reference_model(
    model: TrajectoryDiffusionModel,
) -> TrajectoryDiffusionModel:
    """Create a frozen reference copy of the policy model.

    Uses ``nnx.clone`` for a deep copy with independent parameters.
    The returned model should not be passed to an optimizer.

    Args:
        model: Policy model to clone.

    Returns:
        Independent copy of the model.
    """
    return nnx.clone(model)


def _compute_single_log_prob(
    model: TrajectoryDiffusionModel,
    trajectory: jax.Array,
    scene_ctx: jax.Array,
    key: jax.Array,
    num_samples: int,
) -> jax.Array:
    """Estimate log p(trajectory) for a single scene via Monte Carlo.

    Averages negative MSE over ``num_samples`` (timestep, noise) draws with
    timesteps stratified across the diffusion schedule: draw ``i`` falls in
    schedule bin ``i`` (see :func:`stratified_timestep`), so one estimator
    call always covers early, middle, and late noise levels instead of
    clustering wherever K uniform draws happen to land.

    Args:
        model: Diffusion model for noise prediction.
        trajectory: Single scene trajectories ``(num_agents, future_steps, 4)``.
        scene_ctx: Per-agent scene context ``(num_agents, ctx_dim)`` —
            one row per agent.
        key: JAX random key.
        num_samples: Number of Monte Carlo samples (K).

    Returns:
        Scalar log-probability proxy (non-positive).
    """
    num_timesteps = model.config.num_timesteps

    # Checkpoint the per-sample MSE so differentiating the Monte Carlo
    # loop recomputes each sample's activations in the backward pass
    # instead of storing all of them: memory stays O(1) in num_samples
    # (a DPO step evaluates policy and reference on both pair sides, so
    # uncheckpointed activations scale 4x num_samples x batch).
    @jax.checkpoint
    def sample_mse(sample_index: jax.Array, step_key: jax.Array) -> jax.Array:
        t_key, noise_key = jax.random.split(step_key)
        t = stratified_timestep(t_key, sample_index, num_samples, num_timesteps)
        noise = jax.random.normal(noise_key, trajectory.shape)
        noisy = model.q_sample(trajectory, t, noise)
        pred = model.predict_noise(noisy, t, scene_ctx, deterministic=True)
        return jnp.mean((noise - pred) ** 2)

    def body_fn(i: jax.Array, carry: jax.Array) -> jax.Array:
        """Accumulate MSE for one (timestep, noise) sample."""
        return carry + sample_mse(i, jax.random.fold_in(key, i))

    total_mse = jax.lax.fori_loop(0, num_samples, body_fn, jnp.array(0.0))
    return -(total_mse / num_samples)


class DPOAlignmentTrainer:
    """DPO trainer for trajectory diffusion model alignment.

    Consumes preference pairs (chosen/rejected trajectories) to fine-tune
    the diffusion model using the DPO objective. Supports optional physics
    regularisation and reference-free (SimPO) mode.

    The trainer follows the ``TrajectoryTrainer`` pattern for NaN-safe
    gradient handling and optimizer updates.
    """

    def __init__(
        self,
        model: TrajectoryDiffusionModel,
        optimizer: nnx.Optimizer,
        config: DPOAlignmentConfig | None = None,
        reference_model: TrajectoryDiffusionModel | None = None,
        physics: SimulacraxPhysicsLoss | None = None,
    ) -> None:
        """Initialize the DPO alignment trainer.

        Args:
            model: Policy model to fine-tune.
            optimizer: Pre-configured NNX optimizer (with gradient clipping).
            config: DPO training configuration. Uses defaults if ``None``.
            reference_model: Frozen reference model for standard DPO.
                Required when ``config.reference_free`` is ``False``.
            physics: Optional physics loss for regularisation.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or DPOAlignmentConfig()
        self.reference_model = reference_model
        self.physics = physics

        # Compiled distributed step, built once (a fresh nnx.jit wrapper
        # per call would retrace and recompile on every step)
        self._jitted_dpo_step = nnx.jit(self.compute_dpo_step)

    def compute_trajectory_log_prob(
        self,
        model: TrajectoryDiffusionModel,
        trajectories: jax.Array,
        scene_contexts: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Estimate diffusion log-probabilities for batched trajectories.

        Uses the Diffusion-DPO Monte Carlo estimator: averages negative
        noise prediction MSE over K (timestep, noise) samples, with
        timesteps stratified across the diffusion schedule (one draw per
        schedule bin) to reduce estimator variance without bias.

        Args:
            model: Diffusion model for noise prediction.
            trajectories: Batched trajectories
                ``(batch, num_agents, future_steps, 4)``.
            scene_contexts: Batched per-agent scene contexts
                ``(batch, num_agents, ctx_dim)`` — one row per agent.
            key: JAX random key.

        Returns:
            Log-probability proxies ``(batch,)``, non-positive.
        """
        num_samples = self.config.num_log_prob_samples
        batch_keys = jax.random.split(key, trajectories.shape[0])

        compute_fn = partial(
            _compute_single_log_prob,
            model,
            num_samples=num_samples,
        )
        return jax.vmap(compute_fn)(trajectories, scene_contexts, batch_keys)

    def compute_dpo_loss(
        self,
        batch: dict[str, jax.Array],
        key: jax.Array,
        policy_model: TrajectoryDiffusionModel | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute the DPO loss with optional physics regularisation.

        Args:
            batch: Dict with ``"chosen"``, ``"rejected"``, and
                ``"scene_contexts"`` arrays.
            key: JAX random key.
            policy_model: Policy model to compute log-probs with.
                Defaults to ``self.model`` when ``None``. Passed
                explicitly by ``train_step`` for gradient tracing.

        Returns:
            Tuple of ``(scalar_loss, aux_dict)`` where aux contains
            per-metric values for logging.
        """
        model = policy_model if policy_model is not None else self.model
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        scene_contexts = batch.get("scene_contexts")
        if scene_contexts is None:
            msg = "batch must contain 'scene_contexts' for DPO log-prob estimation"
            raise ValueError(msg)

        # ONE key everywhere: Diffusion-DPO evaluates policy/reference AND
        # chosen/rejected on the SAME (timestep, noise) draws so per-sample
        # Monte Carlo noise cancels in both the log-ratio and the pair
        # difference instead of dominating them.
        policy_chosen_lp = self.compute_trajectory_log_prob(
            model,
            chosen,
            scene_contexts,
            key,
        )
        policy_rejected_lp = self.compute_trajectory_log_prob(
            model,
            rejected,
            scene_contexts,
            key,
        )

        # Reference log-ratios (skipped in reference-free mode).
        ref_chosen_lp: jax.Array | None = None
        ref_rejected_lp: jax.Array | None = None
        if not self.config.reference_free:
            if self.reference_model is None:
                msg = "reference_model is required when reference_free=False"
                raise ValueError(msg)
            ref_chosen_lp = self.compute_trajectory_log_prob(
                self.reference_model,
                chosen,
                scene_contexts,
                key,
            )
            ref_rejected_lp = self.compute_trajectory_log_prob(
                self.reference_model,
                rejected,
                scene_contexts,
                key,
            )

        total_loss, dpo_aux = dpo_loss_from_log_probs(
            policy_chosen_lp,
            policy_rejected_lp,
            ref_chosen_lp,
            ref_rejected_lp,
            self.config,
        )

        # Physics regularisation
        physics_loss = jnp.array(0.0)
        if self.physics is not None and self.config.physics_weight > 0:
            # vmap over batch: compute_loss expects (A, T, 4) not batched
            def single_physics(traj: jax.Array) -> jax.Array:
                loss, _ = self.physics.compute_loss(traj, epoch=0)  # type: ignore[union-attr]
                return loss

            per_scene_physics = jax.vmap(single_physics)(chosen)
            physics_loss = jnp.mean(per_scene_physics)
            total_loss = total_loss + self.config.physics_weight * physics_loss

        # dpo_aux carries the implicit-reward metrics (reference-anchored when a
        # reference model is used); append the physics regularisation term.
        aux = {**dpo_aux, "physics_loss": physics_loss}
        return total_loss, aux

    def compute_dpo_step(
        self,
        model: TrajectoryDiffusionModel,
        optimizer: nnx.Optimizer,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """JIT-compatible core of a DPO training step.

        Computes DPO loss, gradients, NaN-safe update, and auxiliary metrics.
        Follows the same two-layer pattern as
        :meth:`~simulacrax.models.trainer.TrajectoryTrainer.compute_train_step`:
        ``model`` and ``optimizer`` are explicit arguments (NNX-traced by
        ``nnx.jit``), while config and reference model are captured in the
        bound method's closure (static).

        Wrap with ``nnx.jit`` for compiled performance, or use inside a
        :func:`jax.set_mesh` context for SPMD distributed execution::

            jit_step = nnx.jit(trainer.compute_dpo_step)
            with jax.set_mesh(mesh):
                loss, aux = jit_step(
                    trainer.model, trainer.optimizer, batch, key
                )

        Args:
            model: Policy model (NNX-traced).
            optimizer: Flax NNX optimizer (NNX-traced).
            batch: DPO batch dict with ``"chosen"``, ``"rejected"``, and
                ``"scene_contexts"`` arrays.
            key: JAX random key.

        Returns:
            Tuple of ``(loss, aux_dict)`` where ``aux_dict`` contains
            ``policy_chosen_log_prob``, ``policy_rejected_log_prob``,
            ``reward_accuracy``, ``reward_margin``, ``grad_norm``, and
            ``has_nan``.
        """

        def loss_fn(
            m: TrajectoryDiffusionModel,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            return self.compute_dpo_loss(batch, key, policy_model=m)

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Gradient norm (JIT-compatible)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        squared_norms = [jnp.sum(g**2) for g in grad_leaves if hasattr(g, "shape")]
        grad_norm = jnp.sqrt(sum(squared_norms)) if squared_norms else jnp.array(0.0)

        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        optimizer.update(model, safe_grads)

        aux["grad_norm"] = grad_norm
        aux["has_nan"] = has_nan

        return loss, aux

    def train_step(
        self,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> DPOAlignmentMetrics:
        """Execute a single DPO training step.

        Runs the prebuilt ``nnx.jit``-compiled step and wraps it with
        Python-level operations (NaN logging) that cannot run inside JIT.

        For uncompiled debugging, call :meth:`compute_dpo_step` directly.

        Args:
            batch: DPO batch dict from ``PreferenceBatch.to_dpo_batch()``.
            key: JAX random key.

        Returns:
            Training metrics for this step.
        """
        loss, aux = self._jitted_dpo_step(self.model, self.optimizer, batch, key)

        if bool(aux["has_nan"]):
            logger.warning("NaN detected in DPO step, gradients zeroed")

        return _build_alignment_metrics(loss, aux, float(aux["grad_norm"]))

    def train_step_distributed(
        self,
        batch: dict[str, jax.Array],
        key: jax.Array,
        *,
        mesh: jax.sharding.Mesh,
    ) -> DPOAlignmentMetrics:
        """Execute a distributed DPO training step using SPMD via ``nnx.jit``.

        Compiles :meth:`compute_dpo_step` with ``nnx.jit`` and executes it
        inside a :func:`jax.set_mesh` context.  XLA's SPMD partitioner
        automatically replicates parameters and all-reduces gradients across
        the mesh — no manual ``shard_batch`` call is needed.

        On a single-device mesh this is functionally identical to
        :meth:`train_step`.

        Args:
            batch: DPO batch dict from ``PreferenceBatch.to_dpo_batch()``.
            key: JAX random key.
            mesh: JAX device mesh from
                :func:`~simulacrax.core.distributed.create_device_mesh`.

        Returns:
            DPO alignment metrics for this step.
        """
        with jax.set_mesh(mesh):
            loss, aux = self._jitted_dpo_step(self.model, self.optimizer, batch, key)

        if bool(aux["has_nan"]):
            logger.warning("NaN detected in distributed DPO step, gradients zeroed")

        return DPOAlignmentMetrics(
            dpo_loss=float(loss),
            policy_chosen_log_prob=float(aux["policy_chosen_log_prob"]),
            policy_rejected_log_prob=float(aux["policy_rejected_log_prob"]),
            reward_accuracy=float(aux["reward_accuracy"]),
            reward_margin=float(aux["reward_margin"]),
            grad_norm=float(aux["grad_norm"]),
        )
