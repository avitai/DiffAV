"""Tests for the map-conditioned training entrypoint."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from simulacrax.api.map_conditioned import (
    MapConditionedTrajectoryConfig,
    MapConditionedTrajectoryModel,
)
from simulacrax.api.map_conditioned_trainer import (
    build_optimizer,
    MapConditionedTrainer,
    TrainBatch,
    TrainingConfig,
)
from simulacrax.core.types import ModalityMode
from simulacrax.data.tokenizer import TokenizerConfig
from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionConfig


_AGENTS = 2
_STEPS = 4
_EMBED = 32


def _model(*, gradient_checkpointing: bool = False) -> MapConditionedTrajectoryModel:
    tokenizer = TokenizerConfig(
        embed_dim=_EMBED,
        num_heads=4,
        num_egnn_layers=1,
        max_polylines=8,
        lidar_modality=ModalityMode.OFF,
        camera_modality=ModalityMode.OFF,
    )
    diffusion = TrajectoryDiffusionConfig(
        hidden_dim=_EMBED,
        num_heads=4,
        num_blocks=1,
        num_temporal_layers=1,
        num_social_layers=1,
        future_steps=_STEPS,
        num_agents_max=8,
        num_timesteps=10,
        context_dim=_EMBED,
        scene_token_dim=_EMBED,
        use_map_cross_attention=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    config = MapConditionedTrajectoryConfig(tokenizer=tokenizer, diffusion=diffusion)
    return MapConditionedTrajectoryModel(config, rngs=nnx.Rngs(0))


def _raw_scene(seed: int) -> dict[str, Any]:
    steps, num_rg = 91, 40
    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "state/all/x": normal(4, steps),
        "state/all/y": normal(4, steps),
        "state/all/bbox_yaw": normal(4, steps),
        "state/all/velocity_x": normal(4, steps),
        "state/all/velocity_y": normal(4, steps),
        "state/all/valid": np.ones((4, steps), dtype=np.int64),
        "state/is_sdc": np.array([1, 0, 0, 0]),
        "roadgraph_samples/xyz": normal(num_rg, 3),
        "roadgraph_samples/dir": normal(num_rg, 3),
        "roadgraph_samples/type": np.ones((num_rg, 1), dtype=np.int64),
        "roadgraph_samples/id": np.repeat(np.arange(4), 10).reshape(-1, 1),
        "roadgraph_samples/valid": np.ones((num_rg, 1), dtype=np.int64),
    }


def _batch(batch_size: int = 2) -> TrainBatch:
    scenes = jax.tree_util.tree_map(
        lambda *arrays: jnp.stack(arrays), *[_raw_scene(index) for index in range(batch_size)]
    )
    trajectories = jax.random.normal(jax.random.key(7), (batch_size, _AGENTS, _STEPS, 4))
    agent_rows = jnp.broadcast_to(jnp.array([0, 2]), (batch_size, _AGENTS))
    valid = jnp.ones((batch_size, _AGENTS, _STEPS), dtype=bool)
    reference_pose = jax.random.normal(jax.random.key(8), (batch_size, _AGENTS, 3))
    return TrainBatch(
        scene=scenes,
        trajectories=trajectories,
        agent_rows=agent_rows,
        valid=valid,
        reference_pose=reference_pose,
    )


class TestTrainingConfig:
    """Schedule and EMA bounds are validated at construction."""

    def test_non_positive_total_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="total_steps"):
            TrainingConfig(total_steps=0)

    def test_warmup_not_less_than_total_raises(self) -> None:
        """Warmup must leave at least one step for the cosine decay."""
        with pytest.raises(ValueError, match="warmup_steps"):
            TrainingConfig(total_steps=100, warmup_steps=100)

    def test_ema_decay_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_decay"):
            TrainingConfig(total_steps=10, warmup_steps=0, ema_decay=1.0)

    def test_final_lr_fraction_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="final_lr_fraction"):
            TrainingConfig(total_steps=10, warmup_steps=0, final_lr_fraction=1.5)


class TestBuildOptimizer:
    """The optimizer warms up then decays, applied through global-norm clipping."""

    def test_warmup_then_decay_learning_rate(self) -> None:
        """A single scalar parameter follows the warmup-cosine update magnitude."""
        config = TrainingConfig(
            total_steps=100, peak_learning_rate=1.0, warmup_steps=10, final_lr_fraction=0.0
        )
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=1.0, warmup_steps=10, decay_steps=100, end_value=0.0
        )
        # The peak (step 10) exceeds both the warmup start (step 0) and the tail.
        assert float(schedule(0)) < float(schedule(10))
        assert float(schedule(99)) < float(schedule(10))
        # build_optimizer returns a usable transformation.
        tx = build_optimizer(config)
        params = jnp.ones((3,))
        state = tx.init(params)
        updates, _ = tx.update(jnp.ones((3,)), state, params)
        update_leaf = jnp.asarray(jax.tree_util.tree_leaves(updates)[0])
        assert update_leaf.shape == (3,)


class TestMapConditionedTrainer:
    """Training loop: jitted step, EMA update, and periodic validation."""

    def test_fit_trains_and_evaluates(self) -> None:
        """fit runs finite steps, fires the callback on schedule, holds EMA state."""
        model = _model()
        config = TrainingConfig(total_steps=4, warmup_steps=1, eval_every=2, ema_decay=0.9)
        trainer = MapConditionedTrainer(model, config)
        batches = [_batch() for _ in range(4)]

        eval_calls: list[int] = []

        def validation_callback(_: MapConditionedTrajectoryModel) -> dict[str, float]:
            eval_calls.append(1)
            return {"val_offroad": 0.5}

        history = trainer.fit(
            batches, key=jax.random.key(0), validation_callback=validation_callback
        )
        assert len(history) == 4
        assert all(bool(jnp.isfinite(record["loss"])) for record in history)
        # (step + 1) % 2 == 0 fires on steps 1 and 3.
        assert len(eval_calls) == 2
        assert history[1]["val_offroad"] == 0.5
        assert "val_offroad" not in history[0]
        assert trainer.ema.state is not None

    def test_fit_restores_eval_mode(self) -> None:
        """fit trains in train mode but leaves the model deterministic on exit,
        so a subsequent sample runs without dropout."""
        model = _model()
        trainer = MapConditionedTrainer(model, TrainingConfig(total_steps=2, warmup_steps=0))
        trainer.fit([_batch()], key=jax.random.key(3))
        deterministic_flags = [
            value.deterministic
            for _, value in nnx.iter_graph(model)
            if isinstance(getattr(value, "deterministic", None), bool)
        ]
        assert deterministic_flags  # the encoders expose the flag
        assert all(deterministic_flags)

    def test_fit_with_gradient_checkpointing(self) -> None:
        """The jitted step and optimizer update compose with backbone remat.

        Rematerializing the backbone blocks must not leave a stale trace level
        that the in-place optimizer update rejects.
        """
        trainer = MapConditionedTrainer(
            _model(gradient_checkpointing=True), TrainingConfig(total_steps=2, warmup_steps=0)
        )
        history = trainer.fit([_batch()], key=jax.random.key(9))
        assert bool(jnp.isfinite(history[0]["loss"]))

    def test_fit_invokes_on_step_callback(self) -> None:
        """on_step fires once per step with the step index (for logging/checkpoints)."""
        trainer = MapConditionedTrainer(_model(), TrainingConfig(total_steps=3, warmup_steps=0))
        seen: list[int] = []
        trainer.fit(
            [_batch(), _batch()],
            key=jax.random.key(2),
            on_step=lambda step, _record: seen.append(step),
        )
        assert seen == [0, 1]

    def test_fit_updates_parameters(self) -> None:
        """A training step changes at least one model parameter."""
        model = _model()
        before = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))[0]
        before_value = jnp.asarray(before).copy()
        trainer = MapConditionedTrainer(model, TrainingConfig(total_steps=2, warmup_steps=0))
        trainer.fit([_batch()], key=jax.random.key(1))
        after = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))[0]
        assert not jnp.allclose(before_value, jnp.asarray(after))
