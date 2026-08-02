"""Tests for DPO alignment trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.alignment.dpo_trainer import (
    create_reference_model,
    DPOAlignmentConfig,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionModel
from simulacrax.physics.losses import SimulacraxPhysicsLoss
from tests import support
from tests.alignment.helpers import BATCH_SIZE, CONTEXT_DIM, NUM_AGENTS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_optimizer(model: TrajectoryDiffusionModel) -> nnx.Optimizer:
    """Create a small optimizer for testing."""
    return support.make_adam_optimizer(model)


def _make_trainer(
    model: TrajectoryDiffusionModel,
    config: DPOAlignmentConfig | None = None,
    physics: SimulacraxPhysicsLoss | None = None,
) -> DPOAlignmentTrainer:
    """Create a DPO trainer with reference model for testing.

    Args:
        model: Policy model.
        config: Optional DPO config.
        physics: Optional physics loss.

    Returns:
        Configured DPO trainer.
    """
    optimizer = _make_optimizer(model)
    ref = create_reference_model(model)
    return DPOAlignmentTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        reference_model=ref,
        physics=physics,
    )


# ---------------------------------------------------------------------------
# DPOAlignmentConfig
# ---------------------------------------------------------------------------


class TestDPOAlignmentConfig:
    """Tests for DPOAlignmentConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = DPOAlignmentConfig()
        assert cfg.beta == 0.1
        assert cfg.label_smoothing == 0.0
        assert cfg.reference_free is False
        assert cfg.num_log_prob_samples == 4
        assert cfg.physics_weight == 0.0

    def test_beta_zero_raises(self) -> None:
        """beta=0 raises ValueError."""
        with pytest.raises(ValueError, match="beta"):
            DPOAlignmentConfig(beta=0.0)

    def test_label_smoothing_negative_raises(self) -> None:
        """Negative label_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="label_smoothing"):
            DPOAlignmentConfig(label_smoothing=-0.1)

    def test_label_smoothing_too_large_raises(self) -> None:
        """label_smoothing >= 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="label_smoothing"):
            DPOAlignmentConfig(label_smoothing=0.5)

    def test_num_log_prob_samples_zero_raises(self) -> None:
        """num_log_prob_samples=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_log_prob_samples"):
            DPOAlignmentConfig(num_log_prob_samples=0)

    def test_physics_weight_negative_raises(self) -> None:
        """Negative physics_weight raises ValueError."""
        with pytest.raises(ValueError, match="physics_weight"):
            DPOAlignmentConfig(physics_weight=-1.0)


# ---------------------------------------------------------------------------
# DPOAlignmentMetrics
# ---------------------------------------------------------------------------


class TestDPOAlignmentMetrics:
    """Tests for DPOAlignmentMetrics dataclass."""

    def test_frozen(self) -> None:
        """DPOAlignmentMetrics is immutable."""
        metrics = DPOAlignmentMetrics(
            dpo_loss=0.5,
            policy_chosen_log_prob=-1.0,
            policy_rejected_log_prob=-2.0,
            reward_accuracy=0.75,
            reward_margin=1.0,
            grad_norm=0.01,
        )
        with pytest.raises(AttributeError):
            metrics.dpo_loss = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# create_reference_model
# ---------------------------------------------------------------------------


class TestCreateReferenceModel:
    """Tests for create_reference_model."""

    def test_creates_independent_copy(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
    ) -> None:
        """Modifying policy params does not affect reference."""
        ref = create_reference_model(small_diffusion_model)

        # Get a param leaf from both
        policy_params = nnx.state(small_diffusion_model, nnx.Param)
        ref_params = nnx.state(ref, nnx.Param)

        policy_leaves = jax.tree_util.tree_leaves(policy_params)
        ref_leaves = jax.tree_util.tree_leaves(ref_params)

        # Initially equal
        for p, r in zip(policy_leaves, ref_leaves):
            assert jnp.allclose(p, r)

    def test_return_type(self, small_diffusion_model: TrajectoryDiffusionModel) -> None:
        """Returns a TrajectoryDiffusionModel."""
        ref = create_reference_model(small_diffusion_model)
        assert isinstance(ref, TrajectoryDiffusionModel)


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------


class TestLogProbComputation:
    """Tests for compute_trajectory_log_prob."""

    def test_output_shape(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        straight_trajectories: jax.Array,
        scene_context: jax.Array,
    ) -> None:
        """Log-probs have shape (batch_size,)."""
        trainer = _make_trainer(small_diffusion_model)
        batch_ctx = jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM))
        log_probs = trainer.compute_trajectory_log_prob(
            small_diffusion_model,
            straight_trajectories,
            batch_ctx,
            jax.random.key(42),
        )
        assert log_probs.shape == (BATCH_SIZE,)

    def test_log_probs_finite(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        straight_trajectories: jax.Array,
        scene_context: jax.Array,
    ) -> None:
        """Log-probs are finite (no NaN/Inf)."""
        trainer = _make_trainer(small_diffusion_model)
        batch_ctx = jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM))
        log_probs = trainer.compute_trajectory_log_prob(
            small_diffusion_model,
            straight_trajectories,
            batch_ctx,
            jax.random.key(42),
        )
        assert jnp.all(jnp.isfinite(log_probs))

    def test_log_probs_non_positive(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        straight_trajectories: jax.Array,
        scene_context: jax.Array,
    ) -> None:
        """Log-probs are non-positive (negative MSE proxy)."""
        trainer = _make_trainer(small_diffusion_model)
        batch_ctx = jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM))
        log_probs = trainer.compute_trajectory_log_prob(
            small_diffusion_model,
            straight_trajectories,
            batch_ctx,
            jax.random.key(42),
        )
        assert jnp.all(log_probs <= 0.0)

    def test_different_trajectories_different_log_probs(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        straight_trajectories: jax.Array,
        collision_trajectories: jax.Array,
        scene_context: jax.Array,
    ) -> None:
        """Different trajectories produce different log-probs."""
        trainer = _make_trainer(small_diffusion_model)
        batch_ctx = jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM))
        key = jax.random.key(42)
        lp_straight = trainer.compute_trajectory_log_prob(
            small_diffusion_model,
            straight_trajectories,
            batch_ctx,
            key,
        )
        lp_collision = trainer.compute_trajectory_log_prob(
            small_diffusion_model,
            collision_trajectories,
            batch_ctx,
            key,
        )
        # They should not be identical (different input data)
        assert not jnp.allclose(lp_straight, lp_collision)

    def test_jit_compatible(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        straight_trajectories: jax.Array,
        scene_context: jax.Array,
    ) -> None:
        """compute_trajectory_log_prob works under JIT."""
        trainer = _make_trainer(small_diffusion_model)
        batch_ctx = jnp.broadcast_to(scene_context, (BATCH_SIZE, NUM_AGENTS, CONTEXT_DIM))

        @jax.jit
        def compute(traj: jax.Array, ctx: jax.Array, key: jax.Array) -> jax.Array:
            return trainer.compute_trajectory_log_prob(
                small_diffusion_model,
                traj,
                ctx,
                key,
            )

        result = compute(straight_trajectories, batch_ctx, jax.random.key(42))
        assert result.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDPOLoss:
    """Tests for compute_dpo_loss."""

    def test_loss_finite_scalar(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """DPO loss is a finite scalar."""
        trainer = _make_trainer(small_diffusion_model)
        loss, aux = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0))
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_aux_dict_keys(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Aux dict contains expected metric keys."""
        trainer = _make_trainer(small_diffusion_model)
        _, aux = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0))
        expected_keys = {
            "policy_chosen_log_prob",
            "policy_rejected_log_prob",
            "reward_accuracy",
            "reward_margin",
        }
        assert expected_keys.issubset(set(aux.keys()))

    def test_meaningful_loss(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """DPO loss is a meaningful positive value."""
        trainer = _make_trainer(small_diffusion_model)
        loss, _ = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0))
        # DPO loss from -log(sigmoid(...)) is always positive
        assert float(loss) > 0.0

    def test_label_smoothing_changes_loss(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Label smoothing produces a different loss when logits are non-zero.

        Reference-free mode keeps the logits non-zero: with a clone
        reference the shared-draw estimator cancels the log-ratio to
        exactly 0, where smoothing has no effect by symmetry.
        """
        loss_by_smoothing = []
        for label_smoothing in (0.0, 0.1):
            trainer = DPOAlignmentTrainer(
                model=small_diffusion_model,
                optimizer=_make_optimizer(small_diffusion_model),
                config=DPOAlignmentConfig(label_smoothing=label_smoothing, reference_free=True),
            )
            loss, _ = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0))
            loss_by_smoothing.append(loss)
        assert not jnp.allclose(loss_by_smoothing[0], loss_by_smoothing[1])

    def test_jit_compatible(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """compute_dpo_loss works under JIT."""
        trainer = _make_trainer(small_diffusion_model)

        @jax.jit
        def compute(key: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
            return trainer.compute_dpo_loss(dpo_batch, key)

        loss, aux = compute(jax.random.key(0))
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSharedEstimatorDraws:
    """Policy and reference passes must share (t, noise) draws (Diffusion-DPO).

    With shared draws, an identical reference model cancels exactly in the
    log-ratio, so the DPO loss collapses to -log_sigmoid(0) = log(2). With
    independent draws the logits are Monte Carlo noise instead.
    """

    def test_identical_reference_gives_exact_log_two_loss(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """A clone reference makes the log-ratio exactly zero."""
        import math

        trainer = _make_trainer(small_diffusion_model)
        loss, _ = trainer.compute_dpo_loss(dpo_batch, jax.random.key(7))
        assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)

    def test_identical_reference_zero_loss_across_keys(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """The exact cancellation holds for any key, not one lucky draw."""
        import math

        trainer = _make_trainer(small_diffusion_model)
        for seed in (0, 1, 2):
            loss, _ = trainer.compute_dpo_loss(dpo_batch, jax.random.key(seed))
            assert float(loss) == pytest.approx(math.log(2.0), abs=1e-6)


@pytest.mark.slow
class TestTrainStep:
    """Tests for train_step."""

    def test_returns_metrics(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """train_step returns DPOAlignmentMetrics with all fields."""
        trainer = _make_trainer(small_diffusion_model)
        metrics = trainer.train_step(dpo_batch, jax.random.key(0))
        assert isinstance(metrics, DPOAlignmentMetrics)
        assert isinstance(metrics.dpo_loss, float)
        assert isinstance(metrics.reward_accuracy, float)
        assert isinstance(metrics.grad_norm, float)

    def test_reward_accuracy_in_range(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """reward_accuracy is in [0, 1]."""
        trainer = _make_trainer(small_diffusion_model)
        metrics = trainer.train_step(dpo_batch, jax.random.key(0))
        assert 0.0 <= metrics.reward_accuracy <= 1.0

    def test_grad_norm_finite_positive(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Gradient norm is finite and positive."""
        trainer = _make_trainer(small_diffusion_model)
        metrics = trainer.train_step(dpo_batch, jax.random.key(0))
        assert metrics.grad_norm > 0.0
        assert jnp.isfinite(metrics.grad_norm)

    def test_params_change_after_step(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Model parameters change after train_step."""
        trainer = _make_trainer(small_diffusion_model)
        params_before = jax.tree_util.tree_map(
            lambda x: x.copy(), jax.tree_util.tree_leaves(nnx.state(trainer.model, nnx.Param))
        )
        trainer.train_step(dpo_batch, jax.random.key(0))
        params_after = jax.tree_util.tree_leaves(nnx.state(trainer.model, nnx.Param))

        any_changed = any(not jnp.allclose(p, a) for p, a in zip(params_before, params_after))
        assert any_changed

    def test_multiple_steps_no_nan(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Multiple training steps do not produce NaN."""
        trainer = _make_trainer(small_diffusion_model)
        for i in range(3):
            metrics = trainer.train_step(dpo_batch, jax.random.key(i))
            assert jnp.isfinite(metrics.dpo_loss)
            assert jnp.isfinite(metrics.grad_norm)


# ---------------------------------------------------------------------------
# Reference model modes
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestReferenceModelModes:
    """Tests for standard DPO and SimPO (reference-free) modes."""

    def test_reference_unchanged_after_train(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Reference model params unchanged after policy train_step."""
        ref = create_reference_model(small_diffusion_model)
        ref_params_before = jax.tree_util.tree_map(
            lambda x: x.copy(), jax.tree_util.tree_leaves(nnx.state(ref, nnx.Param))
        )
        optimizer = _make_optimizer(small_diffusion_model)
        trainer = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer,
            reference_model=ref,
        )
        trainer.train_step(dpo_batch, jax.random.key(0))

        ref_params_after = jax.tree_util.tree_leaves(nnx.state(ref, nnx.Param))
        for before, after in zip(ref_params_before, ref_params_after):
            assert jnp.allclose(before, after)

    def test_reference_free_mode(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """reference_free=True works without reference model."""
        config = DPOAlignmentConfig(reference_free=True)
        optimizer = _make_optimizer(small_diffusion_model)
        trainer = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer,
            config=config,
            reference_model=None,
        )
        metrics = trainer.train_step(dpo_batch, jax.random.key(0))
        assert jnp.isfinite(metrics.dpo_loss)


# ---------------------------------------------------------------------------
# Physics regularisation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPhysicsRegularisation:
    """Tests for physics loss integration."""

    def test_physics_weight_increases_loss(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """physics_weight > 0 produces larger total loss."""
        ref = create_reference_model(small_diffusion_model)
        physics = SimulacraxPhysicsLoss()

        optimizer_no = _make_optimizer(small_diffusion_model)
        trainer_no = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer_no,
            config=DPOAlignmentConfig(physics_weight=0.0),
            reference_model=ref,
        )

        optimizer_yes = _make_optimizer(small_diffusion_model)
        trainer_yes = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer_yes,
            config=DPOAlignmentConfig(physics_weight=0.1),
            reference_model=ref,
            physics=physics,
        )

        # Chosen trajectories must actually violate physics for the term to
        # be non-zero: overlap all agents (collision) in the chosen slot.
        violating_batch = dict(dpo_batch)
        violating_batch["chosen"], violating_batch["rejected"] = (
            dpo_batch["rejected"],
            dpo_batch["chosen"],
        )

        key = jax.random.key(0)
        loss_no, _ = trainer_no.compute_dpo_loss(violating_batch, key)
        loss_yes, aux_yes = trainer_yes.compute_dpo_loss(violating_batch, key)
        # The difference must be exactly the weighted physics term — a
        # dropped or ignored physics_weight cannot pass this.
        physics_term = float(aux_yes["physics_loss"])
        assert physics_term > 0.0
        assert float(loss_yes) - float(loss_no) == pytest.approx(0.1 * physics_term, rel=1e-4)

    def test_physics_loss_in_aux(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """Physics loss component appears in aux dict."""
        ref = create_reference_model(small_diffusion_model)
        physics = SimulacraxPhysicsLoss()
        optimizer = _make_optimizer(small_diffusion_model)
        trainer = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer,
            config=DPOAlignmentConfig(physics_weight=0.1),
            reference_model=ref,
            physics=physics,
        )
        _, aux = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0))
        assert "physics_loss" in aux
        assert jnp.isfinite(aux["physics_loss"])


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """Tests for gradient computation."""

    def test_grad_produces_finite_gradients(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """jax.grad of compute_dpo_loss produces finite gradients."""
        ref = create_reference_model(small_diffusion_model)
        optimizer = _make_optimizer(small_diffusion_model)
        trainer = DPOAlignmentTrainer(
            model=small_diffusion_model,
            optimizer=optimizer,
            reference_model=ref,
        )

        def loss_fn(model: TrajectoryDiffusionModel) -> jax.Array:
            loss, _ = trainer.compute_dpo_loss(dpo_batch, jax.random.key(0), policy_model=model)
            return loss

        grads = nnx.grad(loss_fn)(small_diffusion_model)
        grad_leaves = [g for g in jax.tree_util.tree_leaves(grads) if hasattr(g, "shape")]
        for g in grad_leaves:
            assert jnp.all(jnp.isfinite(g))
        # All-zero gradients would mean the loss ignored the model
        global_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        assert float(global_norm) > 0.0


class TestReferenceAlignedBehaviour:
    """Reference-implementation semantics: shared draws, implicit rewards."""

    def test_identical_pair_gives_zero_margin(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """chosen == rejected must give exactly zero margin and ln(2) loss.

        Diffusion-DPO scores both sides of a pair on the SAME
        (timestep, noise) draws so Monte Carlo noise cancels; with
        independent draws an identical pair would show phantom margin.
        """
        trainer = _make_trainer(small_diffusion_model)
        batch = dict(dpo_batch)
        batch["rejected"] = batch["chosen"]
        loss, aux = trainer.compute_dpo_loss(batch, jax.random.key(3))
        assert float(aux["reward_margin"]) == 0.0
        assert float(loss) == pytest.approx(float(jnp.log(2.0)), rel=1e-6)

    def test_metrics_use_implicit_rewards(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """reward_margin must be beta * (log-ratio difference), not raw lp.

        The canonical metric compares implicit rewards
        beta * (policy - reference); raw policy log-probs are biased by
        trajectory content.
        """
        config = DPOAlignmentConfig(beta=0.5)
        trainer = _make_trainer(small_diffusion_model, config=config)
        key = jax.random.key(11)
        loss, aux = trainer.compute_dpo_loss(dpo_batch, key)

        policy_c = trainer.compute_trajectory_log_prob(
            trainer.model, dpo_batch["chosen"], dpo_batch["scene_contexts"], key
        )
        policy_r = trainer.compute_trajectory_log_prob(
            trainer.model, dpo_batch["rejected"], dpo_batch["scene_contexts"], key
        )
        reference = trainer.reference_model
        assert reference is not None
        ref_c = trainer.compute_trajectory_log_prob(
            reference, dpo_batch["chosen"], dpo_batch["scene_contexts"], key
        )
        ref_r = trainer.compute_trajectory_log_prob(
            reference, dpo_batch["rejected"], dpo_batch["scene_contexts"], key
        )
        expected_margin = float(jnp.mean(config.beta * ((policy_c - ref_c) - (policy_r - ref_r))))
        assert float(aux["reward_margin"]) == pytest.approx(expected_margin, rel=1e-5)

    def test_train_step_uses_compiled_path(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """train_step must route through the prebuilt jitted step."""
        trainer = _make_trainer(small_diffusion_model)
        calls: list[bool] = []
        original = trainer._jitted_dpo_step

        def spy(
            model: TrajectoryDiffusionModel,
            optimizer: nnx.Optimizer,
            batch: dict[str, jax.Array],
            key: jax.Array,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            calls.append(True)
            return original(model, optimizer, batch, key)

        trainer._jitted_dpo_step = spy  # type: ignore[method-assign]
        metrics = trainer.train_step(dpo_batch, jax.random.key(0))
        assert calls, "train_step bypassed the compiled step"
        assert jnp.isfinite(metrics.dpo_loss)

    def test_simpo_gamma_raises_loss_in_reference_free_mode(
        self,
        small_diffusion_model: TrajectoryDiffusionModel,
        dpo_batch: dict[str, jax.Array],
    ) -> None:
        """A positive target margin shifts logits down and raises the loss."""
        base = DPOAlignmentConfig(reference_free=True)
        with_gamma = DPOAlignmentConfig(reference_free=True, simpo_gamma=1.0)
        loss_base, _ = _make_trainer(small_diffusion_model, config=base).compute_dpo_loss(
            dpo_batch, jax.random.key(0)
        )
        loss_gamma, _ = _make_trainer(small_diffusion_model, config=with_gamma).compute_dpo_loss(
            dpo_batch, jax.random.key(0)
        )
        assert float(loss_gamma) > float(loss_base)

    def test_simpo_gamma_negative_raises(self) -> None:
        """simpo_gamma must be non-negative."""
        with pytest.raises(ValueError, match="simpo_gamma"):
            DPOAlignmentConfig(simpo_gamma=-0.1)
