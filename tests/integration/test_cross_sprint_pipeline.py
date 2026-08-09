"""Cross-sprint integration tests for the full DiffAV pipeline.

Exercises data types -> metrics, SDK generation -> evaluation, and
occupancy flow -> PDE loss without any real WOD TFRecord dependency.
All models use minimal configs to keep tests fast on CPU.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffav.api import create_scenario_miner, MinerConfig
from diffav.core.types import MetricsReport, TrajectoryPrediction
from diffav.evaluation.metrics import MotionMetrics, MotionMetricsConfig
from diffav.evaluation.runner import EvaluationRunner
from diffav.occupancy.flow_model import (
    OccupancyFlowConfig,
    OccupancyFlowModel,
    OccupancyGrid,
)
from diffav.occupancy.losses import FlowConsistencyLoss, FlowConsistencyLossConfig


# ---------------------------------------------------------------------------
# Shared constants for small synthetic models
# ---------------------------------------------------------------------------

_NUM_AGENTS = 2
_FUTURE_STEPS = 4
_GRID_RES = 8
_TEMPORAL_HORIZON = 2


# ---------------------------------------------------------------------------
# Minimal mock model used in Test 1
# ---------------------------------------------------------------------------


class _ConstantModel:
    """Minimal model stub that always returns zero-valued trajectories.

    Implements the interface expected by EvaluationRunner.run():
    ``sample(scene_context, *, key) -> TrajectoryPrediction``.
    """

    def __init__(self, future_steps: int) -> None:
        """Initialise with a fixed prediction horizon.

        Args:
            future_steps: Number of future timesteps to predict.
        """
        self._future_steps = future_steps

    def sample(
        self,
        scene_context: Any,
        *,
        key: jax.Array,
    ) -> TrajectoryPrediction:
        """Return zero-filled trajectory predictions.

        Args:
            scene_context: Per-agent context ``(num_agents, context_dim)``; the
                agent count is inferred from its row count.
            key: Ignored; required for interface compatibility.

        Returns:
            ``TrajectoryPrediction`` with zeroed trajectories.
        """
        num_agents = scene_context.shape[0]
        trajectories = jnp.zeros((num_agents, self._future_steps, 4))
        agent_ids = tuple(f"agent_{i}" for i in range(num_agents))
        return TrajectoryPrediction(trajectories=trajectories, agent_ids=agent_ids)


# ---------------------------------------------------------------------------
# Helper: build a minimal evaluation batch
# ---------------------------------------------------------------------------


def _make_batch(
    num_agents: int,
    future_steps: int,
    object_type: int = 0,
) -> dict[str, Any]:
    """Build a synthetic batch dict matching EvaluationRunner's expected schema.

    Args:
        num_agents: Number of agents in the batch.
        future_steps: Number of ground-truth future timesteps.
        object_type: WOD integer agent-type code (0=vehicle, 1=ped, 2=cyclist).

    Returns:
        Dict with all keys required by ``EvaluationRunner.run()``.
    """
    # scene_context shape: (num_agents, context_dim) — one row per agent
    context_dim = 8
    return {
        "scene_context": jnp.zeros((num_agents, context_dim)),
        # ground_truth: (A, TG, 7) — [x, y, len, wid, hdg, vx, vy]
        "ground_truth": jnp.ones((num_agents, future_steps, 7)),
        # All timesteps valid
        "ground_truth_is_valid": jnp.ones((num_agents, future_steps)),
        "object_type": jnp.full((num_agents,), object_type, dtype=jnp.int32),
        "key": jax.random.key(0),
    }


# ---------------------------------------------------------------------------
# Test 1: data types → MotionMetrics → MetricsReport
# ---------------------------------------------------------------------------


class TestDataToMetricsPipeline:
    """Exercise the TrajectoryPrediction → EvaluationRunner → MetricsReport path."""

    def test_data_to_metrics_pipeline(self) -> None:
        """Build synthetic data, run EvaluationRunner, assert finite ADE/FDE.

        Creates a ``_ConstantModel`` stub, feeds it a single synthetic batch
        through ``EvaluationRunner.run()``, and verifies that the resulting
        ``MetricsReport`` contains finite ADE and FDE values.
        """
        model = _ConstantModel(future_steps=_FUTURE_STEPS)
        config = MotionMetricsConfig(
            miss_rate_threshold=2.0,
            top_k=1,
            num_future_steps=_FUTURE_STEPS,
        )
        metrics = MotionMetrics(config)
        runner = EvaluationRunner(motion_metrics=metrics)

        batches = [_make_batch(num_agents=_NUM_AGENTS, future_steps=_FUTURE_STEPS)]
        report = runner.run(model, batches)

        assert isinstance(report, MetricsReport)
        # At least one per-type metric must be present
        assert len(report.metric_values) > 0

        for key, value in report.metric_values.items():
            assert math.isfinite(value), f"metric {key!r} is not finite: {value}"

        # ADE and FDE (reported as VEHICLE/minADE etc.) must be non-negative
        for key, value in report.metric_values.items():
            if "minADE" in key or "minFDE" in key:
                assert value >= 0.0, f"{key} should be non-negative, got {value}"


# ---------------------------------------------------------------------------
# Test 2: ScenarioMiner.generate() → EvaluationRunner.run()
# ---------------------------------------------------------------------------


class TestSdkGeneratesEvaluableScenarios:
    """Verify that SDK-generated scenarios can be fed directly into EvaluationRunner."""

    @pytest.fixture(scope="class")
    def miner(self):
        """Create a small ScenarioMiner with minimal config.

        Returns:
            ``ScenarioMiner`` instance configured for fast CPU testing.
        """
        config = MinerConfig(
            batch_size=2,
            max_agents=_NUM_AGENTS,
            prediction_horizon=_FUTURE_STEPS,
            context_dim=8,
            num_diffusion_steps=2,
        )
        return create_scenario_miner(config)

    def test_sdk_generates_evaluable_scenarios(self, miner) -> None:
        """Generate one scenario, evaluate with EvaluationRunner, check MetricsReport.

        Generates a single scenario via ``ScenarioMiner.generate()``, extracts the
        ``TrajectoryPrediction``, wraps it in a mock model, runs ``EvaluationRunner``,
        and asserts that the result is a valid ``MetricsReport`` with ``ade >= 0``.
        """
        scenarios = miner.generate("forward", "low", count=1, key=jax.random.key(42))
        assert len(scenarios) == 1

        scenario = scenarios[0]
        prediction = scenario.predictions
        assert isinstance(prediction, TrajectoryPrediction)

        # Wrap the generated prediction in a trivial model stub
        class _ReplayModel:
            """Model stub that replays a pre-generated TrajectoryPrediction."""

            def __init__(self, fixed_prediction: TrajectoryPrediction) -> None:
                """Store the pre-generated prediction for replay.

                Args:
                    fixed_prediction: The prediction to return on every call.
                """
                self._prediction = fixed_prediction

            def sample(
                self,
                scene_context: Any,
                *,
                key: jax.Array,
            ) -> TrajectoryPrediction:
                """Return the stored prediction regardless of input.

                Args:
                    scene_context: Ignored.
                    key: Ignored.

                Returns:
                    The fixed ``TrajectoryPrediction`` provided at construction.
                """
                return self._prediction

        replay_model = _ReplayModel(prediction)
        n_agents = prediction.trajectories.shape[0]
        n_steps = prediction.trajectories.shape[1]

        metrics_config = MotionMetricsConfig(top_k=1, num_future_steps=n_steps)
        runner = EvaluationRunner(motion_metrics=MotionMetrics(metrics_config))

        batch = _make_batch(num_agents=n_agents, future_steps=n_steps)
        report = runner.run(replay_model, [batch])

        assert isinstance(report, MetricsReport)
        # All metric values must be finite and non-negative
        for key, value in report.metric_values.items():
            assert math.isfinite(value), f"{key!r} is not finite"
            assert value >= 0.0, f"{key!r} should be non-negative, got {value}"


# ---------------------------------------------------------------------------
# Test 3: OccupancyFlowModel → FlowConsistencyLoss
# ---------------------------------------------------------------------------


class TestOccupancyIntegratesWithEvaluation:
    """Verify that OccupancyFlowModel output feeds correctly into FlowConsistencyLoss."""

    def test_occupancy_integrates_with_evaluation(self) -> None:
        """Build small OccupancyFlowModel, run forward pass, assert finite PDE loss.

        Constructs an ``OccupancyFlowModel`` with a tiny grid resolution,
        runs a forward pass with synthetic ``OccupancyGrid`` input, computes
        ``FlowConsistencyLoss``, and asserts the result is a finite scalar.
        """
        config = OccupancyFlowConfig(
            grid_resolution=_GRID_RES,
            grid_size_m=10.0,
            temporal_horizon=_TEMPORAL_HORIZON,
            num_agent_types=3,
            num_map_channels=3,
            hidden_channels=4,
            modes_per_scale=(2, 1),
            num_layers_per_scale=(1, 1),
            use_cross_scale_attention=True,
            attention_heads=2,
            use_gradient_checkpointing=False,
        )
        rngs = nnx.Rngs(params=jax.random.key(7))
        occ_model = OccupancyFlowModel(config, rngs=rngs)

        h = w = _GRID_RES
        grid = OccupancyGrid(
            occupancy=jnp.full((h, w, 3), 0.5, dtype=jnp.float32),
            flow=jnp.zeros((h, w, 2), dtype=jnp.float32),
            map_features=jnp.zeros((h, w, 3), dtype=jnp.float32),
        )

        prediction = occ_model(grid)

        # Validate output shapes before computing loss
        assert prediction.occupancy.shape == (1, _TEMPORAL_HORIZON, 3, h, w)
        assert prediction.flow.shape == (1, _TEMPORAL_HORIZON, h, w, 2)

        loss_config = FlowConsistencyLossConfig(
            weight=1.0,
            dt=0.5,
            cell_size_m=10.0 / _GRID_RES,
        )
        loss_fn = FlowConsistencyLoss(loss_config)
        loss_value = loss_fn.compute(prediction)

        assert loss_value.shape == (), "Loss must be a scalar"
        assert math.isfinite(float(loss_value)), f"Loss is not finite: {loss_value}"
        assert float(loss_value) >= 0.0, f"Loss must be non-negative, got {loss_value}"
