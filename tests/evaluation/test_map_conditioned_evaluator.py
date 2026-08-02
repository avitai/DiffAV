"""Tests for the held-out map-conditioned validation evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from simulacrax.api.map_conditioned import (
    MapConditionedTrajectoryConfig,
    MapConditionedTrajectoryModel,
)
from simulacrax.core.geometry import RoadEdges
from simulacrax.core.types import ModalityMode, TrajectoryPrediction
from simulacrax.data.tokenizer import TokenizerConfig
from simulacrax.evaluation.map_conditioned_evaluator import (
    MapConditionedEvaluator,
    ValidationEvalConfig,
    ValidationScene,
)
from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionConfig


_AGENTS = 2
_STEPS = 4


def _square_edges() -> RoadEdges:
    """Counterclockwise 20 m square centred at the origin (interior on-road)."""
    square = np.array(
        [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0], [-10, -10, 0]],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


@dataclass(frozen=True)
class _StubSampler:
    """A sampler returning fixed trajectories regardless of the key."""

    trajectories: jax.Array

    def sample(
        self,
        scene: Mapping[str, Any],
        agent_rows: jax.Array,
        *,
        reference_pose: jax.Array,
        key: jax.Array,
    ) -> TrajectoryPrediction:
        return TrajectoryPrediction(trajectories=self.trajectories, agent_ids=())


def _scene(gt: jax.Array, *, tracks_to_predict: jax.Array | None = None) -> ValidationScene:
    return ValidationScene(
        scene={},
        trajectories=gt,
        agent_rows=jnp.arange(_AGENTS),
        valid=jnp.ones((_AGENTS, _STEPS), dtype=bool),
        tracks_to_predict=(
            jnp.ones((_AGENTS,), dtype=bool) if tracks_to_predict is None else tracks_to_predict
        ),
        reference_pose=jnp.zeros((_AGENTS, 3)),
        road_edges=_square_edges(),
    )


class TestEvaluateScene:
    """Per-scene scoring against a deterministic stub sampler."""

    def test_perfect_prediction_is_zero_ade_on_road(self) -> None:
        """Predicting the logged futures exactly gives minADE 0, off-road 0."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))  # agents at the origin, inside the square
        evaluator = MapConditionedEvaluator(
            ValidationEvalConfig(num_rollouts=3, offroad_chunk_size=None)
        )
        scene_ade, scene_offroad, scene_kinematic = evaluator.evaluate_scene(
            _StubSampler(gt), _scene(gt), key=jax.random.key(0)
        )
        assert scene_ade == pytest.approx(0.0, abs=1e-5)
        assert scene_offroad == 0.0
        assert jnp.isfinite(scene_kinematic)

    def test_offset_prediction_recovers_displacement(self) -> None:
        """A constant 3 m x-offset from the ground truth yields minADE 3."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))
        pred = gt.at[..., 0].set(3.0)
        evaluator = MapConditionedEvaluator(ValidationEvalConfig(num_rollouts=2))
        scene_ade, _, _ = evaluator.evaluate_scene(
            _StubSampler(pred), _scene(gt), key=jax.random.key(1)
        )
        assert scene_ade == pytest.approx(3.0, rel=1e-4)

    def test_off_road_prediction_is_rate_one(self) -> None:
        """Predicting far outside the square is off-road at every valid step."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))
        pred = gt.at[..., 0].set(50.0)
        evaluator = MapConditionedEvaluator()
        _, scene_offroad, _ = evaluator.evaluate_scene(
            _StubSampler(pred), _scene(gt), key=jax.random.key(2)
        )
        assert scene_offroad == 1.0

    def test_scores_only_tracks_to_predict_agents(self) -> None:
        """minADE covers the tracks_to_predict agents only: a large error on a
        non-track agent does not affect it (unlike scoring every valid agent)."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))
        # Agent 0 predicted perfectly; every other agent predicted 10 m off.
        pred = gt.at[1:, :, 0].set(10.0)
        evaluator = MapConditionedEvaluator(ValidationEvalConfig(num_rollouts=2))
        tracks = jnp.zeros((_AGENTS,), dtype=bool).at[0].set(True)
        scene_ade, _, _ = evaluator.evaluate_scene(
            _StubSampler(pred), _scene(gt, tracks_to_predict=tracks), key=jax.random.key(3)
        )
        # Only the perfectly-predicted track agent is scored.
        assert scene_ade == pytest.approx(0.0, abs=1e-5)


class TestEvaluateAggregate:
    """Aggregation across the validation set."""

    def test_aggregates_over_scenes(self) -> None:
        """Per-scene metrics are averaged and the scene count reported."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))
        evaluator = MapConditionedEvaluator()
        result = evaluator.evaluate(
            _StubSampler(gt), [_scene(gt), _scene(gt)], key=jax.random.key(3)
        )
        assert result.num_scenes == 2
        assert result.min_ade == pytest.approx(0.0, abs=1e-5)
        assert result.offroad_rate == 0.0

    def test_clustered_eval_oversamples_then_selects_modes(self) -> None:
        """With num_modes set, evaluate clusters the oversampled rollouts first."""
        gt = jnp.zeros((_AGENTS, _STEPS, 4))
        evaluator = MapConditionedEvaluator(ValidationEvalConfig(num_rollouts=8, num_modes=3))
        result = evaluator.evaluate(
            _StubSampler(gt), [_scene(gt), _scene(gt)], key=jax.random.key(7)
        )
        assert result.num_scenes == 2
        assert result.min_ade == pytest.approx(0.0, abs=1e-5)

    def test_empty_validation_set(self) -> None:
        """No scenes yields a zeroed report rather than dividing by zero."""
        evaluator = MapConditionedEvaluator()
        result = evaluator.evaluate(
            _StubSampler(jnp.zeros((_AGENTS, _STEPS, 4))), [], key=jax.random.key(4)
        )
        assert result.num_scenes == 0
        assert result.min_ade == 0.0


class TestRealModelIntegration:
    """The real map-conditioned model samples under the evaluator's vmap."""

    def _model(self) -> MapConditionedTrajectoryModel:
        embed = 32
        tokenizer = TokenizerConfig(
            embed_dim=embed,
            num_heads=4,
            num_egnn_layers=1,
            max_polylines=8,
            lidar_modality=ModalityMode.OFF,
            camera_modality=ModalityMode.OFF,
        )
        diffusion = TrajectoryDiffusionConfig(
            hidden_dim=embed,
            num_heads=4,
            num_blocks=1,
            num_temporal_layers=1,
            num_social_layers=1,
            future_steps=_STEPS,
            num_agents_max=8,
            num_timesteps=10,
            context_dim=embed,
            scene_token_dim=embed,
            use_map_cross_attention=True,
        )
        config = MapConditionedTrajectoryConfig(tokenizer=tokenizer, diffusion=diffusion)
        return MapConditionedTrajectoryModel(config, rngs=nnx.Rngs(0))

    def _raw_scene(self) -> dict[str, Any]:
        steps, num_rg = 91, 40
        rng = np.random.default_rng(0)

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

    def test_evaluate_scene_finite_metrics(self) -> None:
        """K rollouts from the real model give finite, in-range metrics."""
        model = self._model()
        scene = ValidationScene(
            scene=self._raw_scene(),
            trajectories=jax.random.normal(jax.random.key(1), (_AGENTS, _STEPS, 4)),
            agent_rows=jnp.array([0, 2]),
            valid=jnp.ones((_AGENTS, _STEPS), dtype=bool),
            tracks_to_predict=jnp.ones((_AGENTS,), dtype=bool),
            reference_pose=jax.random.normal(jax.random.key(2), (_AGENTS, 3)),
            road_edges=_square_edges(),
        )
        evaluator = MapConditionedEvaluator(ValidationEvalConfig(num_rollouts=2))
        scene_ade, scene_offroad, scene_kinematic = evaluator.evaluate_scene(
            model, scene, key=jax.random.key(5)
        )
        assert jnp.isfinite(scene_ade) and scene_ade >= 0.0
        assert 0.0 <= scene_offroad <= 1.0
        assert jnp.isfinite(scene_kinematic) and scene_kinematic >= 0.0

    def test_evaluate_batches_real_scenes(self) -> None:
        """The batched evaluate stacks multiple real scenes into one device pass.

        Uniform per-scene shapes (including road edges) let every rollout sample
        in a single ``num_scenes * K`` vmap, and the aggregates come back finite
        and in range with a single host transfer.
        """
        model = self._model()

        def _val_scene() -> ValidationScene:
            return ValidationScene(
                scene=self._raw_scene(),
                trajectories=jax.random.normal(jax.random.key(1), (_AGENTS, _STEPS, 4)),
                agent_rows=jnp.array([0, 2]),
                valid=jnp.ones((_AGENTS, _STEPS), dtype=bool),
                tracks_to_predict=jnp.ones((_AGENTS,), dtype=bool),
                reference_pose=jax.random.normal(jax.random.key(2), (_AGENTS, 3)),
                road_edges=_square_edges(),
            )

        evaluator = MapConditionedEvaluator(ValidationEvalConfig(num_rollouts=2))
        result = evaluator.evaluate(model, [_val_scene(), _val_scene()], key=jax.random.key(6))
        assert result.num_scenes == 2
        assert jnp.isfinite(result.min_ade) and result.min_ade >= 0.0
        assert 0.0 <= result.offroad_rate <= 1.0
        assert jnp.isfinite(result.kinematic_residual) and result.kinematic_residual >= 0.0
