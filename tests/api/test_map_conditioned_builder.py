"""Tests for the map-conditioned model builder and checkpoint loader.

The builder centralizes the tokenizer/diffusion/physics coupling a training
run configures (previously duplicated in ``scripts/train_wod.py``), and the
loader rebuilds that exact architecture from a checkpoint's ``run_config.json``
— which a training run records as stringified ``vars(args)`` — and restores the
saved weights, version-guarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.api.map_conditioned import (
    build_map_conditioned_model,
    load_map_conditioned_from_checkpoint,
    MapConditionedBuildSpec,
    MapConditionedTrajectoryModel,
)
from simulacrax.core.constants import (
    AGENT_LOCAL_STATE_SCALES,
    SCENE_BACKBONE_ARCHITECTURE_VERSION,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.types import PredictionType
from simulacrax.models.checkpointing import (
    CheckpointConfig,
    CheckpointCorruptError,
    SimulacraxCheckpointManager,
)


def _small_spec(**overrides: object) -> MapConditionedBuildSpec:
    """A tiny build spec — fast to construct and save/restore."""
    defaults: dict[str, object] = {
        "hidden_dim": 16,
        "num_heads": 2,
        "num_blocks": 1,
        "num_temporal_layers": 1,
        "num_social_layers": 1,
        "max_agents": 4,
        "future_steps": 4,
        "diffusion_steps": 4,
        "kinematic_weight": 0.1,
        "collision_weight": 0.1,
    }
    defaults.update(overrides)
    return MapConditionedBuildSpec(**defaults)  # type: ignore[arg-type]


def _stringified_run_config(**overrides: str) -> dict[str, str]:
    """A stringified ``vars(args)`` run config, mirroring tier-0's file."""
    config = {
        "hidden_dim": "256",
        "num_heads": "8",
        "num_blocks": "6",
        "num_temporal_layers": "2",
        "num_social_layers": "1",
        "max_agents": "32",
        "future_steps": "80",
        "diffusion_steps": "1000",
        "kinematic_weight": "0.1",
        "collision_weight": "0.1",
        "gradient_checkpointing": "True",
        # Extra keys a run records that the spec ignores.
        "learning_rate": "0.0002",
        "split": "train",
    }
    config.update(overrides)
    return config


def _param_leaves(model: MapConditionedTrajectoryModel) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))


class TestBuildSpecFromRunConfig:
    """``from_run_config`` re-types the stringified run config."""

    def test_parses_numeric_types(self) -> None:
        """Integer and float fields are parsed from their string forms."""
        spec = MapConditionedBuildSpec.from_run_config(_stringified_run_config())
        assert spec.hidden_dim == 256
        assert isinstance(spec.hidden_dim, int)
        assert spec.diffusion_steps == 1000
        assert spec.kinematic_weight == pytest.approx(0.1)
        assert isinstance(spec.kinematic_weight, float)

    def test_parses_bool_true(self) -> None:
        """``gradient_checkpointing`` "True" parses to boolean True."""
        spec = MapConditionedBuildSpec.from_run_config(_stringified_run_config())
        assert spec.gradient_checkpointing is True

    def test_parses_bool_false(self) -> None:
        """ "False" parses to boolean False (not truthy-nonempty-string True)."""
        spec = MapConditionedBuildSpec.from_run_config(
            _stringified_run_config(gradient_checkpointing="False")
        )
        assert spec.gradient_checkpointing is False

    def test_defaults_history_steps_to_canonical_constant(self) -> None:
        """History steps are not a run arg; the spec uses the WOD constant."""
        spec = MapConditionedBuildSpec.from_run_config(_stringified_run_config())
        assert spec.history_steps == WOD_HISTORY_STEPS


class TestBuildMapConditionedModel:
    """The builder produces a map-conditioned model with the locked policy."""

    def test_builds_model_instance(self) -> None:
        """A spec builds a ``MapConditionedTrajectoryModel``."""
        model = build_map_conditioned_model(_small_spec(), rngs=nnx.Rngs(0))
        assert isinstance(model, MapConditionedTrajectoryModel)

    def test_dimensions_track_the_spec(self) -> None:
        """Spec dimensions flow into the tokenizer and diffusion configs."""
        spec = _small_spec(hidden_dim=24, num_heads=3, future_steps=6, diffusion_steps=7)
        model = build_map_conditioned_model(spec, rngs=nnx.Rngs(0))
        assert model.config.tokenizer.embed_dim == 24
        assert model.config.diffusion.context_dim == 24
        assert model.config.diffusion.scene_token_dim == 24
        assert model.config.diffusion.future_steps == 6
        assert model.config.diffusion.num_timesteps == 7

    def test_locks_the_architecture_policy(self) -> None:
        """The non-varied policy matches the tier-0 training recipe."""
        model = build_map_conditioned_model(_small_spec(), rngs=nnx.Rngs(0))
        diffusion = model.config.diffusion
        assert diffusion.use_map_cross_attention is True
        assert diffusion.prediction_type is PredictionType.X0
        assert diffusion.x0_clip_bound is None
        assert diffusion.state_scales == AGENT_LOCAL_STATE_SCALES
        # Physics: kinematic + collision only, non-adaptive.
        assert model.config.physics is not None
        assert model.config.physics.road_boundary_weight == 0.0
        assert model.config.physics.adaptive_weighting is False

    def test_physics_weights_track_the_spec(self) -> None:
        """Kinematic/collision weights flow from the spec into the physics config."""
        spec = _small_spec(kinematic_weight=0.25, collision_weight=0.5)
        model = build_map_conditioned_model(spec, rngs=nnx.Rngs(0))
        assert model.config.physics is not None
        assert model.config.physics.kinematic_weight == pytest.approx(0.25)
        assert model.config.physics.collision_weight == pytest.approx(0.5)


class TestLoadFromCheckpoint:
    """The loader rebuilds from ``run_config.json`` and restores weights."""

    def _save(self, model: MapConditionedTrajectoryModel, directory: Path, *, version: int) -> None:
        config = CheckpointConfig(checkpoint_dir=str(directory), save_interval_steps=1)
        with SimulacraxCheckpointManager(config) as manager:
            manager.save(model, step=1, loss=0.0, architecture_version=version)

    def test_round_trips_weights(self, tmp_path: Path) -> None:
        """Loading restores the saved model's weights, not a fresh init."""
        spec = _small_spec()
        saved = build_map_conditioned_model(spec, rngs=nnx.Rngs(0))
        self._save(saved, tmp_path, version=SCENE_BACKBONE_ARCHITECTURE_VERSION)
        (tmp_path / "run_config.json").write_text(json.dumps(_small_run_config(spec)))

        loaded = load_map_conditioned_from_checkpoint(tmp_path, rngs=nnx.Rngs(999))

        saved_leaves = _param_leaves(saved)
        loaded_leaves = _param_leaves(loaded)
        assert len(saved_leaves) == len(loaded_leaves)
        assert all(jnp.allclose(a, b) for a, b in zip(saved_leaves, loaded_leaves, strict=True))

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """A run config with no saved checkpoint fails fast."""
        spec = _small_spec()
        (tmp_path / "run_config.json").write_text(json.dumps(_small_run_config(spec)))
        with pytest.raises(FileNotFoundError):
            load_map_conditioned_from_checkpoint(tmp_path, rngs=nnx.Rngs(0))

    def test_rejects_wrong_architecture_version(self, tmp_path: Path) -> None:
        """A checkpoint from a different backbone version is rejected."""
        spec = _small_spec()
        saved = build_map_conditioned_model(spec, rngs=nnx.Rngs(0))
        self._save(saved, tmp_path, version=SCENE_BACKBONE_ARCHITECTURE_VERSION + 99)
        (tmp_path / "run_config.json").write_text(json.dumps(_small_run_config(spec)))
        with pytest.raises(CheckpointCorruptError):
            load_map_conditioned_from_checkpoint(tmp_path, rngs=nnx.Rngs(0))


def _small_run_config(spec: MapConditionedBuildSpec) -> dict[str, str]:
    """Stringified run config matching a spec (as a training run would record)."""
    return {
        "hidden_dim": str(spec.hidden_dim),
        "num_heads": str(spec.num_heads),
        "num_blocks": str(spec.num_blocks),
        "num_temporal_layers": str(spec.num_temporal_layers),
        "num_social_layers": str(spec.num_social_layers),
        "max_agents": str(spec.max_agents),
        "future_steps": str(spec.future_steps),
        "diffusion_steps": str(spec.diffusion_steps),
        "kinematic_weight": str(spec.kinematic_weight),
        "collision_weight": str(spec.collision_weight),
        "gradient_checkpointing": str(spec.gradient_checkpointing),
    }
