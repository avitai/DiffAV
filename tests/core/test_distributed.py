"""Tests for distributed execution of the trainers on a substrax device mesh.

Covers the trainer integration of ``train_step_distributed`` for both
TrajectoryTrainer and DPOAlignmentTrainer on a single-device CPU/GPU host, the
data-parallel batch placement substrax provides, and the once-only compilation
of the distributed step.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards

from diffav.alignment.dpo_trainer import (
    DPOAlignmentConfig,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from diffav.models.trainer import (
    TrainerConfig,
    TrainingMetrics,
    TrajectoryTrainer,
)
from diffav.models.trajectory_diffusion import (
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


def _data_mesh() -> jax.sharding.Mesh:
    """A one-axis data-parallel mesh over every visible device."""
    return DeviceMeshManager.create_device_mesh({"data": jax.device_count()})


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


def _param_leaves(model: nnx.Module) -> list[jax.Array]:
    """A snapshot of every parameter array of ``model``."""
    return [jnp.array(leaf) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))]


# ---------------------------------------------------------------------------
# substrax mesh and batch placement
# ---------------------------------------------------------------------------


class TestDataParallelPlacement:
    """The substrax surface the trainers are documented against."""

    def test_data_parallel_sharding_partitions_the_leading_axis(self) -> None:
        mesh = _data_mesh()
        sharding = create_data_parallel_sharding(mesh)
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == PartitionSpec("data")
        assert sharding.mesh.axis_names == ("data",)
        assert mesh.size == jax.device_count()

    def test_place_batch_on_shards_keeps_values_and_shapes(self) -> None:
        sharding = create_data_parallel_sharding(_data_mesh())
        batch = {
            "x": jnp.array([1.0, 2.0, 3.0]),
            "y": jnp.arange(8.0).reshape(_BATCH, 4),
        }
        placed = place_batch_on_shards(batch, sharding)
        assert isinstance(placed, dict)
        assert set(placed) == set(batch)
        for name, value in batch.items():
            assert placed[name].shape == value.shape
            assert jnp.array_equal(placed[name], value)
            assert placed[name].sharding == sharding

    @pytest.mark.slow
    def test_trainer_learns_from_a_batch_placed_on_shards(self) -> None:
        """A backward pass over a data-sharded batch gives a finite loss and moves the weights."""
        trainer = _make_trajectory_trainer()
        mesh = _data_mesh()
        sharding = create_data_parallel_sharding(mesh)
        placed = place_batch_on_shards(
            {
                "trajectories": jnp.ones((_AGENTS, _FUTURE, _STATE_DIM)),
                "scene_context": jnp.ones((_AGENTS, _CTX_DIM)),
            },
            sharding,
        )
        before = _param_leaves(trainer.model)

        metrics = trainer.train_step_distributed(
            placed["trajectories"],
            placed["scene_context"],
            key=jax.random.key(3),
            mesh=mesh,
        )

        assert jnp.isfinite(metrics.total_loss)
        after = _param_leaves(trainer.model)
        assert any(not jnp.array_equal(old, new) for old, new in zip(before, after, strict=True))


# ---------------------------------------------------------------------------
# Trainer integration: TrajectoryTrainer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTrajectoryTrainerDistributed:
    """TrajectoryTrainer.train_step_distributed() integration tests."""

    def test_trajectory_trainer_distributed_step(self) -> None:
        """train_step_distributed() returns TrainingMetrics."""
        trainer = _make_trajectory_trainer()
        mesh = _data_mesh()
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
        mesh = _data_mesh()
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

        dist_metrics = trainer2.train_step_distributed(
            trajectories,
            scene_context,
            key=key,
            mesh=_data_mesh(),
        )

        # Losses should be close (same model init, same input, same key)
        assert abs(single_metrics.total_loss - dist_metrics.total_loss) < 1e-4

    def test_trajectory_trainer_distributed_has_flops_field(self) -> None:
        """TrainingMetrics returned by distributed step has flops_per_step field."""
        trainer = _make_trajectory_trainer()
        mesh = _data_mesh()
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
        mesh = _data_mesh()
        batch = _make_dpo_batch()
        key = jax.random.key(1)

        metrics = trainer.train_step_distributed(batch, key, mesh=mesh)
        assert isinstance(metrics, DPOAlignmentMetrics)

    def test_dpo_trainer_distributed_returns_finite_loss(self) -> None:
        """train_step_distributed() returns finite dpo_loss."""
        trainer = _make_dpo_trainer()
        mesh = _data_mesh()
        batch = _make_dpo_batch()
        key = jax.random.key(2)

        metrics = trainer.train_step_distributed(batch, key, mesh=mesh)
        assert jnp.isfinite(metrics.dpo_loss)


# ---------------------------------------------------------------------------
# JIT hoisting: the compiled step must be built once, not per call
# ---------------------------------------------------------------------------


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
        mesh = _data_mesh()
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
        mesh = _data_mesh()
        batch = _make_dpo_batch()

        # Two warm-up calls: the first update changes the optimizer
        # graphdef once (nnx warm-up), which legitimately retraces.
        for i in range(2):
            trainer.train_step_distributed(batch, jax.random.key(i), mesh=mesh)
        traces_after_warmup = trace_count["n"]
        for i in range(2, 5):
            trainer.train_step_distributed(batch, jax.random.key(i), mesh=mesh)

        assert trace_count["n"] == traces_after_warmup
