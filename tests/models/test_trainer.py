"""Tests for TrajectoryTrainer physics-informed training loop."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from opifex.core.training.optimizers import OptimizerConfig

from diffav.core.constants import SCENE_BACKBONE_ARCHITECTURE_VERSION
from diffav.models.checkpointing import CheckpointConfig, CheckpointCorruptError
from diffav.models.sampling_utils import stratified_timestep
from diffav.models.trainer import (
    TrainerConfig,
    TrainingMetrics,
    TrajectoryTrainer,
)
from diffav.physics.losses import DiffAVPhysicsConfig
from tests.models.helpers import (
    make_model as _make_model,
    sample_data as _sample_data,
    TEST_CTX_DIM,
)


# ---- TrainerConfig tests ----


class TestTrainerConfig:
    """Tests for TrainerConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = TrainerConfig()
        assert cfg.num_epochs == 100
        assert cfg.log_interval == 50
        assert cfg.nan_detection is True
        assert cfg.physics_config is None
        assert cfg.checkpoint_config is None

    def test_uses_opifex_optimizer_config(self) -> None:
        """Default config uses OptimizerConfig from opifex."""
        cfg = TrainerConfig()
        assert isinstance(cfg.optimizer_config, OptimizerConfig)
        assert cfg.optimizer_config.optimizer_type == "adam"
        assert cfg.optimizer_config.learning_rate == 1e-4

    def test_custom_optimizer_config(self) -> None:
        """Custom OptimizerConfig is accepted."""
        opt_cfg = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=3e-4,
            weight_decay=0.01,
            gradient_clip=0.5,
        )
        cfg = TrainerConfig(optimizer_config=opt_cfg)
        assert cfg.optimizer_config.optimizer_type == "adamw"
        assert cfg.optimizer_config.learning_rate == 3e-4

    def test_zero_num_epochs_raises(self) -> None:
        """Zero num_epochs raises ValueError."""
        with pytest.raises(ValueError, match="num_epochs"):
            TrainerConfig(num_epochs=0)

    def test_negative_log_interval_raises(self) -> None:
        """Zero log_interval raises ValueError."""
        with pytest.raises(ValueError, match="log_interval"):
            TrainerConfig(log_interval=0)

    def test_negative_loss_explosion_threshold_raises(self) -> None:
        """Non-positive loss_explosion_threshold raises ValueError."""
        with pytest.raises(ValueError, match="loss_explosion_threshold"):
            TrainerConfig(loss_explosion_threshold=0.0)

    def test_with_physics_config(self) -> None:
        """Physics config can be set."""
        physics = DiffAVPhysicsConfig(kinematic_weight=2.0)
        cfg = TrainerConfig(physics_config=physics)
        assert cfg.physics_config is not None
        assert cfg.physics_config.kinematic_weight == 2.0

    def test_with_checkpoint_config(self) -> None:
        """Checkpoint config can be set."""
        ckpt = CheckpointConfig(save_interval_steps=500)
        cfg = TrainerConfig(checkpoint_config=ckpt)
        assert cfg.checkpoint_config is not None
        assert cfg.checkpoint_config.save_interval_steps == 500


# ---- TrainingMetrics tests ----


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_creation(self) -> None:
        """TrainingMetrics can be created with all fields."""
        m = TrainingMetrics(
            step=0,
            epoch=0,
            total_loss=1.0,
            diffusion_loss=0.8,
            physics_loss=0.2,
            physics_weight=0.01,
            grad_norm=0.5,
            learning_rate=1e-4,
            has_nan=False,
            wall_clock_sec=0.1,
        )
        assert m.step == 0
        assert m.total_loss == 1.0
        assert m.has_nan is False

    def test_frozen(self) -> None:
        """TrainingMetrics is immutable."""
        m = TrainingMetrics(
            step=0,
            epoch=0,
            total_loss=1.0,
            diffusion_loss=0.8,
            physics_loss=0.2,
            physics_weight=0.0,
            grad_norm=0.5,
            learning_rate=1e-4,
            has_nan=False,
            wall_clock_sec=0.1,
        )
        with pytest.raises(AttributeError):
            m.step = 1  # type: ignore[misc]


# ---- TrajectoryTrainer tests ----


@pytest.mark.slow
class TestTrajectoryTrainer:
    """Tests for TrajectoryTrainer."""

    def test_train_step_returns_metrics(self) -> None:
        """train_step returns TrainingMetrics."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert isinstance(metrics, TrainingMetrics)

    def test_train_step_loss_finite(self) -> None:
        """train_step produces finite loss values."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert jnp.isfinite(metrics.total_loss)
        assert jnp.isfinite(metrics.diffusion_loss)
        assert metrics.total_loss >= 0

    def test_train_step_uses_compiled_path(self) -> None:
        """train_step must route through the prebuilt jitted step."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        calls: list[bool] = []
        original = trainer._jitted_step

        def spy(*args: Any, **kwargs: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            calls.append(True)
            return original(*args, **kwargs)

        trainer._jitted_step = spy  # type: ignore[method-assign]
        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert calls, "train_step bypassed the compiled step"
        assert jnp.isfinite(metrics.total_loss)

    def test_train_step_increments_step(self) -> None:
        """Consecutive train_step calls increment the step counter."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()

        m0 = trainer.train_step(traj, ctx, key=jax.random.key(0))
        m1 = trainer.train_step(traj, ctx, key=jax.random.key(1))
        assert m0.step == 0
        assert m1.step == 1

    def test_multiple_steps_change_loss(self) -> None:
        """Multiple training steps change the loss value."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()

        m0 = trainer.train_step(traj, ctx, key=jax.random.key(0))
        for i in range(1, 5):
            trainer.train_step(traj, ctx, key=jax.random.key(i))
        m5 = trainer.train_step(traj, ctx, key=jax.random.key(5))

        # Loss should differ after several steps of optimization
        assert m0.total_loss != pytest.approx(m5.total_loss, abs=1e-6)

    def test_grad_norm_finite(self) -> None:
        """Gradient norm is finite and non-negative."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert jnp.isfinite(metrics.grad_norm)
        assert metrics.grad_norm >= 0

    def test_wall_clock_positive(self) -> None:
        """Wall clock time is positive."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert metrics.wall_clock_sec > 0

    def test_no_physics_loss_by_default(self) -> None:
        """Without physics config, physics loss is zero."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert metrics.physics_loss == pytest.approx(0.0)
        assert metrics.physics_weight == pytest.approx(0.0)

    def test_physics_loss_in_metrics(self) -> None:
        """With physics config, physics loss is tracked."""
        model = _make_model()
        cfg = TrainerConfig(physics_config=DiffAVPhysicsConfig())
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert metrics.physics_weight > 0
        # Physics loss may be zero for constant trajectories, but weight > 0

    def test_adaptive_physics_weight_changes(self) -> None:
        """Physics weight increases with epoch number."""
        model = _make_model()
        cfg = TrainerConfig(
            physics_config=DiffAVPhysicsConfig(
                adaptive_weighting=True,
                transition_epochs=10,
            ),
        )
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        # Epoch 0
        m_early = trainer.train_step(traj, ctx, key=jax.random.key(0))

        # Advance epoch
        trainer.set_epoch(100)
        m_late = trainer.train_step(traj, ctx, key=jax.random.key(1))

        assert m_late.physics_weight > m_early.physics_weight

    def test_custom_optimizer_accepted(self) -> None:
        """Custom optax optimizer overrides config."""
        model = _make_model()
        custom_tx = optax.sgd(learning_rate=0.1)
        trainer = TrajectoryTrainer(model, TrainerConfig(), optimizer=custom_tx)
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert jnp.isfinite(metrics.total_loss)

    def test_nan_detection_flags_nan(self) -> None:
        """NaN in scene context triggers has_nan=True."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig(nan_detection=True))
        traj, _ = _sample_data()
        nan_ctx = jnp.full((traj.shape[0], TEST_CTX_DIM), jnp.nan)
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, nan_ctx, key=key)
        assert metrics.has_nan is True

    def test_nan_detection_disabled(self) -> None:
        """When nan_detection=False, has_nan is always False."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig(nan_detection=False))
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        metrics = trainer.train_step(traj, ctx, key=key)
        assert metrics.has_nan is False

    def test_uses_error_recovery_manager(self) -> None:
        """Trainer composes opifex ErrorRecoveryManager for stability."""
        from opifex.core.training.components.recovery import (
            ErrorRecoveryManager,
        )

        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig(nan_detection=True))
        assert isinstance(trainer.recovery_manager, ErrorRecoveryManager)

    def test_no_recovery_manager_when_disabled(self) -> None:
        """No ErrorRecoveryManager when nan_detection is disabled."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig(nan_detection=False))
        assert trainer.recovery_manager is None

    def test_checkpoint_integration(self, tmp_path: object) -> None:
        """Trainer saves checkpoints at configured intervals."""
        model = _make_model()
        cfg = TrainerConfig(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=str(tmp_path),
                save_interval_steps=2,
                max_to_keep=3,
            ),
        )
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        # Steps 0, 1 - no checkpoint (not at interval)
        trainer.train_step(traj, ctx, key=jax.random.key(0))
        trainer.train_step(traj, ctx, key=jax.random.key(1))

        # Step 2 - should trigger checkpoint
        trainer.train_step(traj, ctx, key=jax.random.key(2))

        assert trainer.checkpoint_manager is not None
        latest = trainer.checkpoint_manager.latest_step()
        assert latest is not None

    def test_checkpoint_stamps_architecture_version(self, tmp_path: object) -> None:
        """Trainer checkpoints carry the backbone architecture version.

        A matching-version restore succeeds; a mismatched one is rejected —
        proving the version was stamped on save.
        """
        cfg = TrainerConfig(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=str(tmp_path),
                save_interval_steps=2,
                max_to_keep=3,
            ),
        )
        trainer = TrajectoryTrainer(_make_model(), cfg)
        traj, ctx = _sample_data()
        for i in range(3):
            trainer.train_step(traj, ctx, key=jax.random.key(i))

        fresh = TrajectoryTrainer(_make_model(), cfg)
        assert fresh.checkpoint_manager is not None
        with pytest.raises(CheckpointCorruptError):
            fresh.checkpoint_manager.restore_latest(
                fresh.model,
                optimizer=fresh.optimizer,
                expected_architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION + 1,
            )
        restored = fresh.checkpoint_manager.restore_latest(
            fresh.model,
            optimizer=fresh.optimizer,
            expected_architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION,
        )
        assert restored is not None

    def test_train_epoch_returns_list(self) -> None:
        """train_epoch returns a list of TrainingMetrics."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()

        # Simulate a data iterator with 3 batches
        data = [(traj, ctx)] * 3
        metrics_list = trainer.train_epoch(iter(data), key=jax.random.key(0))

        assert len(metrics_list) == 3
        assert all(isinstance(m, TrainingMetrics) for m in metrics_list)
        assert all(m.epoch == 0 for m in metrics_list)

    def test_train_epoch_increments_epoch(self) -> None:
        """Consecutive train_epoch calls increment epoch counter."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()

        data1 = [(traj, ctx)] * 2
        m1 = trainer.train_epoch(iter(data1), key=jax.random.key(0))
        assert all(m.epoch == 0 for m in m1)

        data2 = [(traj, ctx)] * 2
        m2 = trainer.train_epoch(iter(data2), key=jax.random.key(1))
        assert all(m.epoch == 1 for m in m2)

    def test_learning_rate_in_metrics(self) -> None:
        """Learning rate from OptimizerConfig is reported in metrics."""
        opt_cfg = OptimizerConfig(learning_rate=5e-4)
        cfg = TrainerConfig(optimizer_config=opt_cfg)
        model = _make_model()
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert metrics.learning_rate == pytest.approx(5e-4)

    def test_gradient_clipping_via_optimizer_config(self) -> None:
        """OptimizerConfig gradient_clip is respected."""
        opt_cfg = OptimizerConfig(
            learning_rate=1e-4,
            gradient_clip=0.01,
        )
        cfg = TrainerConfig(optimizer_config=opt_cfg)
        model = _make_model()
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert jnp.isfinite(metrics.grad_norm)

    def test_train_epoch_logs_at_interval(self) -> None:
        """train_epoch logs metrics at log_interval steps."""
        model = _make_model()
        # log_interval=1 so every step triggers the logging branch
        cfg = TrainerConfig(log_interval=1)
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        data = [(traj, ctx)] * 2
        metrics_list = trainer.train_epoch(iter(data), key=jax.random.key(0))
        assert len(metrics_list) == 2

    def test_train_full_loop(self) -> None:
        """train() runs for num_epochs and returns flat metrics list."""
        model = _make_model()
        cfg = TrainerConfig(num_epochs=2, log_interval=100)
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        data = [(traj, ctx)] * 2

        all_metrics = trainer.train(lambda: iter(data), key=jax.random.key(0))
        # 2 epochs * 2 batches = 4 metrics
        assert len(all_metrics) == 4
        assert all(isinstance(m, TrainingMetrics) for m in all_metrics)

    def test_close_without_checkpoint(self) -> None:
        """close() succeeds when no checkpoint manager is configured."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        trainer.close()  # should not raise

    def test_close_with_checkpoint(self, tmp_path: object) -> None:
        """close() releases checkpoint resources."""
        model = _make_model()
        cfg = TrainerConfig(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=str(tmp_path),
            ),
        )
        trainer = TrajectoryTrainer(model, cfg)
        trainer.close()  # should not raise

    def test_compute_train_step_jit_compatible(self) -> None:
        """compute_train_step can be JIT-compiled via nnx.jit."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        # Artifex pattern: wrap bound method, pass model/optimizer as args
        jit_step = nnx.jit(trainer.compute_train_step)
        total_loss, aux = jit_step(
            trainer.model,
            trainer.optimizer,
            traj,
            ctx,
            key,
            0,
        )

        assert jnp.isfinite(total_loss)
        assert "diffusion_loss" in aux
        assert "physics_loss" in aux
        assert "grad_norm" in aux
        assert "has_nan" in aux
        assert jnp.isfinite(aux["grad_norm"])

    def test_compute_train_step_jit_with_physics(self) -> None:
        """JIT-compiled compute_train_step works with physics loss."""
        model = _make_model()
        cfg = TrainerConfig(physics_config=DiffAVPhysicsConfig())
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        # Physics is captured in self (static closure), not a JIT arg
        jit_step = nnx.jit(trainer.compute_train_step)
        total_loss, aux = jit_step(
            trainer.model,
            trainer.optimizer,
            traj,
            ctx,
            key,
            0,
        )

        assert jnp.isfinite(total_loss)
        assert float(aux["physics_weight"]) > 0

    def test_compute_train_step_jit_multiple_calls(self) -> None:
        """JIT-compiled compute_train_step works across multiple calls."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()

        jit_step = nnx.jit(trainer.compute_train_step)

        loss0, _ = jit_step(
            trainer.model,
            trainer.optimizer,
            traj,
            ctx,
            jax.random.key(0),
            0,
        )
        loss1, _ = jit_step(
            trainer.model,
            trainer.optimizer,
            traj,
            ctx,
            jax.random.key(1),
            0,
        )

        # Both should be finite; model updates between calls
        assert jnp.isfinite(loss0)
        assert jnp.isfinite(loss1)

    def test_physics_loss_depends_on_model_predictions(self) -> None:
        """The physics term is computed on model predictions, not inputs.

        Two differently initialized models given identical inputs must
        produce different physics losses — the term carries model
        gradients rather than being a constant of the batch.
        """
        cfg = TrainerConfig(
            physics_config=DiffAVPhysicsConfig(adaptive_weighting=False),
        )
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        aux_values = []
        for seed in (0, 1):
            model = _make_model(seed=seed)
            trainer = TrajectoryTrainer(model, cfg)
            _, aux = trainer.compute_train_step(trainer.model, trainer.optimizer, traj, ctx, key, 0)
            aux_values.append(float(aux["physics_loss"]))

        assert aux_values[0] != pytest.approx(aux_values[1])

    def test_epoch_is_traced_under_jit(self) -> None:
        """The adaptive physics weight advances across epochs under one jit."""
        cfg = TrainerConfig(
            physics_config=DiffAVPhysicsConfig(
                adaptive_weighting=True,
                transition_epochs=10,
            ),
        )
        model = _make_model()
        trainer = TrajectoryTrainer(model, cfg)
        traj, ctx = _sample_data()

        jit_step = nnx.jit(trainer.compute_train_step)
        _, aux_early = jit_step(trainer.model, trainer.optimizer, traj, ctx, jax.random.key(0), 0)
        _, aux_late = jit_step(trainer.model, trainer.optimizer, traj, ctx, jax.random.key(0), 100)
        assert float(aux_late["physics_weight"]) > float(aux_early["physics_weight"])


# ---- Stability policy: real ErrorRecoveryManager wiring ----


class TestStabilityPolicy:
    """The opifex ErrorRecoveryManager must actually check, recover, raise."""

    def test_stable_step_records_stable_state(self) -> None:
        """A healthy step registers a stable snapshot with the manager."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig())
        traj, ctx = _sample_data()
        trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert trainer.recovery_manager is not None
        assert trainer.recovery_manager.last_stable_state is not None

    def test_persistent_loss_explosion_raises(self) -> None:
        """Unrecoverable instability surfaces as RuntimeError, not silence."""
        config = TrainerConfig(loss_explosion_threshold=1e-12)
        trainer = TrajectoryTrainer(_make_model(), config)
        traj, ctx = _sample_data()

        with pytest.raises(RuntimeError, match="recovery attempts"):
            for i in range(10):
                trainer.train_step(traj, ctx, key=jax.random.key(i))

    def test_recovery_restores_last_stable_state(self) -> None:
        """An exploding step rolls model parameters back to the stable snapshot."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig())
        traj, ctx = _sample_data()

        trainer.train_step(traj, ctx, key=jax.random.key(0))
        stable_params = jax.tree.map(jnp.copy, nnx.state(trainer.model, nnx.Param))

        # Force the next step to register as a loss explosion.
        assert trainer.recovery_manager is not None
        trainer.recovery_manager.loss_explosion_threshold = 1e-12
        trainer.train_step(traj, ctx, key=jax.random.key(1))

        restored = nnx.state(trainer.model, nnx.Param)
        leaves_equal = jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            jax.tree.leaves(stable_params),
            jax.tree.leaves(restored),
        )
        assert all(leaves_equal)

    def test_nan_detection_true_keeps_params_finite(self) -> None:
        """With nan_detection on, a NaN batch zeroes gradients (params stay finite)."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig(nan_detection=True))
        traj, ctx = _sample_data()
        nan_traj = traj.at[0, 0, 0].set(jnp.nan)

        metrics = trainer.train_step(nan_traj, ctx, key=jax.random.key(0))
        assert metrics.has_nan
        params = nnx.state(trainer.model, nnx.Param)
        assert all(bool(jnp.all(jnp.isfinite(p))) for p in jax.tree.leaves(params))

    def test_nan_detection_false_applies_raw_gradients(self) -> None:
        """With nan_detection off, gradients are applied untouched — honestly."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig(nan_detection=False))
        traj, ctx = _sample_data()
        nan_traj = traj.at[0, 0, 0].set(jnp.nan)

        metrics = trainer.train_step(nan_traj, ctx, key=jax.random.key(0))
        assert metrics.has_nan
        params = nnx.state(trainer.model, nnx.Param)
        finite = all(bool(jnp.all(jnp.isfinite(p))) for p in jax.tree.leaves(params))
        assert not finite


# ---- Learning-rate reporting ----


class TestLearningRateReporting:
    """metrics.learning_rate must reflect the optimizer actually in use."""

    def test_constant_lr_reported(self) -> None:
        """Without a schedule, the configured learning rate is reported."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig())
        traj, ctx = _sample_data()
        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert metrics.learning_rate == pytest.approx(1e-4)

    def test_scheduled_lr_advances_with_steps(self) -> None:
        """With an exponential schedule, the reported rate decays per step."""
        optimizer_config = OptimizerConfig(
            optimizer_type="adam",
            learning_rate=1e-3,
            schedule_type="exponential",
            transition_steps=1,
            decay_rate=0.5,
        )
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig(optimizer_config=optimizer_config))
        traj, ctx = _sample_data()

        m0 = trainer.train_step(traj, ctx, key=jax.random.key(0))
        m1 = trainer.train_step(traj, ctx, key=jax.random.key(1))

        assert m1.learning_rate == pytest.approx(m0.learning_rate * 0.5, rel=1e-5)

    def test_custom_optimizer_reports_nan(self) -> None:
        """A caller-supplied optimizer has an unknown rate — report NaN, not a guess."""
        import math

        trainer = TrajectoryTrainer(
            _make_model(),
            TrainerConfig(),
            optimizer=optax.sgd(3e-2),
        )
        traj, ctx = _sample_data()
        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert math.isnan(metrics.learning_rate)


# ---- FLOPs profiling ----


class TestFlopsProfiling:
    """flops_per_step is measured via calibrax when enabled, 0.0 otherwise."""

    def test_flops_measured_when_enabled(self) -> None:
        """profile_flops=True yields a positive, step-stable FLOP count."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig(profile_flops=True))
        traj, ctx = _sample_data()

        m0 = trainer.train_step(traj, ctx, key=jax.random.key(0))
        m1 = trainer.train_step(traj, ctx, key=jax.random.key(1))

        assert m0.flops_per_step > 0.0
        assert m1.flops_per_step == m0.flops_per_step

    def test_flops_zero_when_disabled(self) -> None:
        """The default reports an honest 0.0 (profiling off), never a fake count."""
        trainer = TrajectoryTrainer(_make_model(), TrainerConfig())
        traj, ctx = _sample_data()
        metrics = trainer.train_step(traj, ctx, key=jax.random.key(0))
        assert metrics.flops_per_step == 0.0


class TestBatchedTrainStep:
    """Reference-practice batched steps: mean loss over stacked scenes.

    CTG trains with ~100 scenes per step; per-step single-scene cycling
    produces loss variance that reference implementations never see.
    """

    def test_batched_input_returns_finite_metrics(self) -> None:
        """A (B, A, T, 4) batch trains with finite scalar losses."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        batch_traj = jnp.stack([traj, traj * 0.5, traj * 2.0])
        batch_ctx = jnp.stack([ctx, ctx, ctx])

        metrics = trainer.train_step(batch_traj, batch_ctx, key=jax.random.key(0))
        assert jnp.isfinite(metrics.total_loss)
        assert metrics.total_loss >= 0

    def test_batched_matches_mean_of_scene_losses(self) -> None:
        """Batched diffusion loss equals the mean of matched per-scene losses."""
        model = _make_model()
        trainer = TrajectoryTrainer(model, TrainerConfig())
        traj, ctx = _sample_data()
        batch_traj = jnp.stack([traj, traj * 0.3])
        batch_ctx = jnp.stack([ctx, ctx])
        key = jax.random.key(7)

        jit_step = nnx.jit(trainer.compute_train_step)
        # Evaluate the loss WITHOUT updating: use a throwaway model/optimizer
        eval_model = _make_model()
        eval_trainer = TrajectoryTrainer(eval_model, TrainerConfig())
        _, aux = nnx.jit(eval_trainer.compute_train_step)(
            eval_trainer.model,
            eval_trainer.optimizer,
            batch_traj,
            batch_ctx,
            key,
            0,
        )

        scene_keys = jax.random.split(key, 2)
        reference_model = _make_model()  # same seed as eval_model pre-update
        # The batched path assigns scene i its own schedule stratum (i, B),
        # so the matched per-scene reference must draw the same stratified
        # timestep — a plain uniform per-scene draw no longer matches.
        expected = jnp.mean(
            jnp.stack(
                [
                    reference_model.compute_loss_and_prediction(
                        batch_traj[i],
                        batch_ctx[i],
                        key=scene_keys[i],
                        timestep_stratum=(i, 2),
                    )[0]
                    for i in range(2)
                ]
            )
        )
        # Tolerance covers the float reduction-order difference between the
        # vmap batch-mean and this Python-loop reference (a timestep mismatch
        # would shift the mean by O(0.1), far above this bound).
        assert jnp.allclose(aux["diffusion_loss"], expected, rtol=1e-4)
        del jit_step

    def test_batched_scenes_draw_distinct_schedule_strata(self) -> None:
        """A B-scene batch spreads its diffusion timesteps across B strata."""
        num_timesteps = _make_model().config.num_timesteps
        batch_size = num_timesteps  # one stratum per timestep bin
        key = jax.random.key(3)
        scene_keys = jax.random.split(key, batch_size)
        # Reproduce the trainer's per-scene stratified draw (key_t is the
        # first split of the scene key, matching compute_loss_and_prediction).
        drawn = set()
        for i in range(batch_size):
            key_t, _ = jax.random.split(scene_keys[i])
            t = stratified_timestep(key_t, i, batch_size, num_timesteps)
            drawn.add(int(t))
        # With one stratum per bin the batch must cover every timestep once.
        assert drawn == set(range(num_timesteps))

    def test_batched_with_physics_is_per_scene(self) -> None:
        """Physics loss on a batch stays finite and non-negative per scene."""
        model = _make_model()
        config = TrainerConfig(
            physics_config=DiffAVPhysicsConfig(kinematic_weight=1.0, collision_weight=1.0)
        )
        trainer = TrajectoryTrainer(model, config)
        traj, ctx = _sample_data()
        batch_traj = jnp.stack([traj, traj])
        batch_ctx = jnp.stack([ctx, ctx])

        metrics = trainer.train_step(batch_traj, batch_ctx, key=jax.random.key(1))
        assert jnp.isfinite(metrics.physics_loss)
        assert metrics.physics_loss >= 0


class TestPhysicsAlphaBarAnnealing:
    """Tests for ``TrainerConfig.physics_x0_annealing`` (ᾱ_t-weighted physics).

    Mirrors the reconstruction-guidance anneal the sampling path already
    applies to x̂₀ guidance (``x̂₀ + η·ᾱ_t·∇R``): the physics penalty on the
    training-time x̂₀ reconstruction is trusted in proportion to ᾱ_t.
    """

    @staticmethod
    def _step_aux(annealing: bool) -> dict[str, jax.Array]:
        """Run one unbatched step and return its aux dict (pre-update params)."""
        model = _make_model()
        config = TrainerConfig(
            physics_config=DiffAVPhysicsConfig(
                kinematic_weight=1.0,
                collision_weight=1.0,
                adaptive_weighting=False,
            ),
            physics_x0_annealing=annealing,
        )
        trainer = TrajectoryTrainer(model, config)
        traj, ctx = _sample_data()
        _, aux = trainer.compute_train_step(
            trainer.model, trainer.optimizer, traj, ctx, jax.random.key(21), 0
        )
        return aux

    @staticmethod
    def _reference_outputs_and_physics() -> tuple[jax.Array, jax.Array]:
        """Recompute (ᾱ_t, unannealed physics loss) with matched seed/key."""
        from diffav.physics.losses import DiffAVPhysicsLoss

        reference_model = _make_model()
        traj, ctx = _sample_data()
        outputs = reference_model.compute_loss_outputs(traj, ctx, key=jax.random.key(21))
        physics = DiffAVPhysicsLoss(
            DiffAVPhysicsConfig(
                kinematic_weight=1.0,
                collision_weight=1.0,
                adaptive_weighting=False,
            )
        )
        physics_total, _ = physics.compute_loss(outputs.prediction, 0)
        return outputs.alpha_bar_t, physics_total

    def test_default_is_off(self) -> None:
        """The annealing flag defaults to False (bit-identical prior path)."""
        assert TrainerConfig().physics_x0_annealing is False

    def test_off_matches_unannealed_reference(self) -> None:
        """Flag off: physics loss equals the plain physics penalty on x̂₀."""
        aux = self._step_aux(annealing=False)
        _, expected = self._reference_outputs_and_physics()
        assert jnp.allclose(aux["physics_loss"], expected, rtol=1e-5)

    def test_on_matches_alpha_weighted_reference(self) -> None:
        """Flag on: physics loss equals ᾱ_t × the plain physics penalty."""
        aux = self._step_aux(annealing=True)
        alpha_bar, unannealed = self._reference_outputs_and_physics()
        expected = alpha_bar * unannealed
        assert jnp.allclose(aux["physics_loss"], expected, rtol=1e-5)
        # The anneal is a genuine down-weighting for any t with ᾱ_t < 1.
        assert float(aux["physics_loss"]) <= float(unannealed)

    def test_annealing_leaves_diffusion_loss_unchanged(self) -> None:
        """The anneal touches only the physics term."""
        aux_on = self._step_aux(annealing=True)
        aux_off = self._step_aux(annealing=False)
        assert jnp.allclose(aux_on["diffusion_loss"], aux_off["diffusion_loss"])
