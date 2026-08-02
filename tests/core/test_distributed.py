"""Tests for JAX mesh sharding utilities (distributed execution).

Covers DistributedConfig defaults/immutability, create_device_mesh(),
shard_batch() structure/value preservation, and trainer integration for
both TrajectoryTrainer and DPOAlignmentTrainer on single-device CPU/GPU.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from simulacrax.alignment.dpo_trainer import (
    DPOAlignmentConfig,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from simulacrax.core.distributed import (
    create_device_mesh,
    DistributedConfig,
    shard_batch,
)
from simulacrax.models.trainer import (
    TrainerConfig,
    TrainingMetrics,
    TrajectoryTrainer,
)
from simulacrax.models.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)
from tests import support


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HIDDEN = 32
_HEADS = 2
_FUTURE = 4
_AGENTS = 3
_STEPS = 10
_CTX_DIM = 16
_STATE_DIM = 4
_BATCH = 2
_CTX_LEN = 8


def _make_model() -> TrajectoryDiffusionModel:
    """Create a minimal diffusion model for testing."""
    config = support.make_diffusion_config(
        hidden_dim=_HIDDEN,
        num_blocks=1,
        num_temporal_layers=1,
        num_social_layers=1,
        num_heads=_HEADS,
        future_steps=_FUTURE,
        num_agents_max=_AGENTS + 1,
        num_timesteps=_STEPS,
        context_dim=_CTX_DIM,
        state_dim=_STATE_DIM,
    )
    return support.make_diffusion_model(config, seed=42)


def _make_trajectory_trainer() -> TrajectoryTrainer:
    """Create a minimal TrajectoryTrainer for testing."""
    model = _make_model()
    config = TrainerConfig(num_epochs=1, log_interval=50)
    return TrajectoryTrainer(model, config)


def _make_dpo_trainer() -> DPOAlignmentTrainer:
    """Create a minimal DPOAlignmentTrainer for testing."""
    model = _make_model()
    tx = optax.adam(1e-4)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    config = DPOAlignmentConfig(reference_free=True)
    return DPOAlignmentTrainer(model, optimizer, config=config)


def _make_dpo_batch() -> dict[str, jax.Array]:
    """Create a minimal DPO batch for testing (reference_free=True)."""
    chosen = jnp.ones((_BATCH, _AGENTS, _FUTURE, _STATE_DIM))
    rejected = jnp.zeros((_BATCH, _AGENTS, _FUTURE, _STATE_DIM))
    # One context row per agent: rows must match the agent count.
    scene_contexts = jnp.ones((_BATCH, _AGENTS, _CTX_DIM))
    return {"chosen": chosen, "rejected": rejected, "scene_contexts": scene_contexts}


# ---------------------------------------------------------------------------
# Core: DistributedConfig
# ---------------------------------------------------------------------------


class TestDistributedConfigDefaults:
    """DistributedConfig default values and field types."""

    def test_distributed_config_defaults(self) -> None:
        """Default config has mesh_shape=(1,1,1), sharding_strategy='ddp'."""
        config = DistributedConfig()
        assert config.mesh_shape == (1, 1, 1)
        assert config.sharding_strategy == "ddp"
        assert config.axis_names == ("data", "model", "pipeline")

    def test_distributed_config_frozen(self) -> None:
        """FrozenInstanceError is raised on mutation attempt."""
        config = DistributedConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.mesh_shape = (2, 1, 1)  # type: ignore[misc]

    def test_distributed_config_custom_mesh_shape(self) -> None:
        """Custom mesh_shape is stored correctly."""
        config = DistributedConfig(mesh_shape=(1, 1, 1))
        assert config.mesh_shape == (1, 1, 1)

    def test_distributed_config_unsupported_strategy_fsdp(self) -> None:
        """fsdp strategy raises NotImplementedError."""
        config = DistributedConfig(sharding_strategy="fsdp")
        with pytest.raises(NotImplementedError):
            create_device_mesh(config)

    def test_distributed_config_unsupported_strategy_mp(self) -> None:
        """mp strategy raises NotImplementedError."""
        config = DistributedConfig(sharding_strategy="mp")
        with pytest.raises(NotImplementedError):
            create_device_mesh(config)


# ---------------------------------------------------------------------------
# Core: create_device_mesh
# ---------------------------------------------------------------------------


class TestCreateDeviceMesh:
    """create_device_mesh() returns a correctly-shaped Mesh."""

    def test_create_mesh_single_device(self) -> None:
        """create_device_mesh(DistributedConfig()) returns a Mesh with total size 1."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        assert mesh.size == 1

    def test_mesh_shape_matches_config(self) -> None:
        """Mesh axis sizes match config.mesh_shape on single device."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        # On a single device the mesh collapses to (1, 1, 1) trivially
        assert mesh.size == 1
        # The mesh should have exactly the axis names from the config
        for name in config.axis_names:
            assert name in mesh.axis_names

    def test_mesh_is_context_manager(self) -> None:
        """Mesh can be used as a context manager."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        with mesh:
            assert mesh.size == 1


# ---------------------------------------------------------------------------
# Core: shard_batch
# ---------------------------------------------------------------------------


class TestShardBatch:
    """shard_batch() structure and value preservation."""

    def test_shard_batch_returns_pytree(self) -> None:
        """Output has the same PyTree structure as the dict input."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        batch = {
            "chosen": jnp.ones((_BATCH, _AGENTS, _FUTURE, _STATE_DIM)),
            "rejected": jnp.zeros((_BATCH, _AGENTS, _FUTURE, _STATE_DIM)),
        }
        sharded = shard_batch(batch, mesh)
        assert isinstance(sharded, dict)
        assert set(sharded.keys()) == set(batch.keys())

    def test_shard_batch_preserves_values(self) -> None:
        """Sharded values equal original values on single device."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        batch = {
            "x": jnp.array([1.0, 2.0, 3.0]),
            "y": jnp.array([[4.0, 5.0], [6.0, 7.0]]),
        }
        sharded = shard_batch(batch, mesh)
        assert jnp.allclose(sharded["x"], batch["x"])
        assert jnp.allclose(sharded["y"], batch["y"])

    def test_shard_batch_preserves_shapes(self) -> None:
        """Sharded arrays have the same shapes as original arrays."""
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        batch = {"arr": jnp.ones((_BATCH, _CTX_LEN, _CTX_DIM))}
        sharded = shard_batch(batch, mesh)
        assert sharded["arr"].shape == (_BATCH, _CTX_LEN, _CTX_DIM)


# ---------------------------------------------------------------------------
# Trainer integration: TrajectoryTrainer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTrajectoryTrainerDistributed:
    """TrajectoryTrainer.train_step_distributed() integration tests."""

    def test_trajectory_trainer_distributed_step(self) -> None:
        """train_step_distributed() returns TrainingMetrics."""
        trainer = _make_trajectory_trainer()
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        trajectories = jnp.ones((_AGENTS, _FUTURE, _STATE_DIM))
        scene_context = jnp.ones((_AGENTS, _CTX_DIM))
        key = jax.random.key(0)

        metrics = trainer.train_step_distributed(
            trajectories,
            scene_context,
            key=key,
            mesh=mesh,
        )
        assert isinstance(metrics, TrainingMetrics)

    def test_trajectory_trainer_distributed_returns_numeric_loss(self) -> None:
        """train_step_distributed() returns a finite loss."""
        trainer = _make_trajectory_trainer()
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        trajectories = jnp.ones((_AGENTS, _FUTURE, _STATE_DIM))
        scene_context = jnp.ones((_AGENTS, _CTX_DIM))
        key = jax.random.key(0)

        metrics = trainer.train_step_distributed(
            trajectories,
            scene_context,
            key=key,
            mesh=mesh,
        )
        assert jnp.isfinite(metrics.total_loss)

    def test_distributed_matches_single_device(self) -> None:
        """On 1 device, distributed loss is close to single-device loss."""
        model = _make_model()
        config = TrainerConfig(num_epochs=1, log_interval=50)
        trainer = TrajectoryTrainer(model, config)

        trajectories = jnp.ones((_AGENTS, _FUTURE, _STATE_DIM))
        scene_context = jnp.ones((_AGENTS, _CTX_DIM))
        key = jax.random.key(7)

        # Single-device step
        single_metrics = trainer.train_step(
            trajectories,
            scene_context,
            key=key,
        )

        # Build a fresh trainer with the same initial model weights
        model2 = _make_model()
        trainer2 = TrajectoryTrainer(model2, config)

        dist_config = DistributedConfig()
        mesh = create_device_mesh(dist_config)
        dist_metrics = trainer2.train_step_distributed(
            trajectories,
            scene_context,
            key=key,
            mesh=mesh,
        )

        # Losses should be close (same model init, same input, same key)
        assert abs(single_metrics.total_loss - dist_metrics.total_loss) < 1e-4

    def test_trajectory_trainer_distributed_has_flops_field(self) -> None:
        """TrainingMetrics returned by distributed step has flops_per_step field."""
        trainer = _make_trajectory_trainer()
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        trajectories = jnp.ones((_AGENTS, _FUTURE, _STATE_DIM))
        scene_context = jnp.ones((_AGENTS, _CTX_DIM))
        key = jax.random.key(0)

        metrics = trainer.train_step_distributed(
            trajectories,
            scene_context,
            key=key,
            mesh=mesh,
        )
        assert hasattr(metrics, "flops_per_step")
        assert isinstance(metrics.flops_per_step, float)


# ---------------------------------------------------------------------------
# Trainer integration: DPOAlignmentTrainer
# ---------------------------------------------------------------------------


class TestDPOTrainerDistributed:
    """DPOAlignmentTrainer.train_step_distributed() integration tests."""

    def test_dpo_trainer_distributed_step(self) -> None:
        """train_step_distributed() returns DPOAlignmentMetrics."""
        trainer = _make_dpo_trainer()
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        batch = _make_dpo_batch()
        key = jax.random.key(1)

        metrics = trainer.train_step_distributed(batch, key, mesh=mesh)
        assert isinstance(metrics, DPOAlignmentMetrics)

    def test_dpo_trainer_distributed_returns_finite_loss(self) -> None:
        """train_step_distributed() returns finite dpo_loss."""
        trainer = _make_dpo_trainer()
        config = DistributedConfig()
        mesh = create_device_mesh(config)
        batch = _make_dpo_batch()
        key = jax.random.key(2)

        metrics = trainer.train_step_distributed(batch, key, mesh=mesh)
        assert jnp.isfinite(metrics.dpo_loss)


# ---------------------------------------------------------------------------
# JIT hoisting: the compiled step must be built once, not per call
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestShardingContract:
    """Direct specs for the sharding surface (single-device host)."""

    def test_data_parallel_sharding_partitions_batch_axis(self) -> None:
        from jax.sharding import PartitionSpec

        from simulacrax.core.distributed import get_data_parallel_sharding

        mesh = create_device_mesh(DistributedConfig())
        sharding = get_data_parallel_sharding(mesh)
        assert sharding.spec == PartitionSpec("data")
        assert sharding.mesh.axis_names == ("data", "model", "pipeline")

    def test_oversized_mesh_shape_degrades_to_trivial_mesh(self) -> None:
        """On a single-device host, any mesh_shape yields the (1,1,1) mesh.

        This pins the documented contract explicitly; on multi-device
        hosts a mismatched shape would fail the reshape instead.
        """
        mesh = create_device_mesh(DistributedConfig(mesh_shape=(2, 1, 1)))
        assert mesh.devices.shape == (1, 1, 1)


class TestDistributedJitHoisting:
    """A fresh nnx.jit wrapper per call retraces every step (SRC-10)."""

    def test_trajectory_trainer_no_retrace_after_first_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-shape distributed steps after the first must not retrace."""
        trace_count = {"n": 0}
        original = TrajectoryTrainer.compute_train_step

        def counted(self: TrajectoryTrainer, *args: Any, **kwargs: Any) -> Any:
            trace_count["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(TrajectoryTrainer, "compute_train_step", counted)
        trainer = _make_trajectory_trainer()
        mesh = create_device_mesh(DistributedConfig())
        trajectories = jnp.ones((_AGENTS, _FUTURE, _STATE_DIM))
        scene_context = jnp.ones((_AGENTS, _CTX_DIM))

        # Two warm-up calls: the first update changes the optimizer
        # graphdef once (nnx warm-up), which legitimately retraces.
        for i in range(2):
            trainer.train_step_distributed(
                trajectories, scene_context, key=jax.random.key(i), mesh=mesh
            )
        traces_after_warmup = trace_count["n"]
        for i in range(2, 5):
            trainer.train_step_distributed(
                trajectories, scene_context, key=jax.random.key(i), mesh=mesh
            )

        assert trace_count["n"] == traces_after_warmup

    def test_dpo_trainer_no_retrace_after_first_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same-shape distributed DPO steps after the first must not retrace."""
        trace_count = {"n": 0}
        original = DPOAlignmentTrainer.compute_dpo_step

        def counted(self: DPOAlignmentTrainer, *args: Any, **kwargs: Any) -> Any:
            trace_count["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(DPOAlignmentTrainer, "compute_dpo_step", counted)
        trainer = _make_dpo_trainer()
        mesh = create_device_mesh(DistributedConfig())
        batch = _make_dpo_batch()

        # Two warm-up calls: the first update changes the optimizer
        # graphdef once (nnx warm-up), which legitimately retraces.
        for i in range(2):
            trainer.train_step_distributed(batch, jax.random.key(i), mesh=mesh)
        traces_after_warmup = trace_count["n"]
        for i in range(2, 5):
            trainer.train_step_distributed(batch, jax.random.key(i), mesh=mesh)

        assert trace_count["n"] == traces_after_warmup
