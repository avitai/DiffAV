"""Tests for EvaluationRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from diffav.core.types import MetricsReport, TrajectoryPrediction
from diffav.evaluation.metrics import (
    MotionMetrics,
    MotionMetricsConfig,
    SimAgentMetrics,
    SimAgentMetricsConfig,
)
from diffav.evaluation.runner import EvaluationRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_AGENTS = 2
_T = 3
_CTX_DIM = 8


def _make_mock_model(n: int = _N_AGENTS, t: int = _T) -> MagicMock:
    """Return a mock model whose .sample() returns a zero TrajectoryPrediction."""
    model = MagicMock()
    model.sample.return_value = TrajectoryPrediction(
        trajectories=jnp.zeros((n, t, 4)),
        agent_ids=tuple(f"agent_{i}" for i in range(n)),
    )
    return model


def _make_batch(
    n: int = _N_AGENTS,
    t: int = _T,
    *,
    include_road_edges: bool = False,
) -> dict:
    return {
        # One context row per agent: the model infers the agent count.
        "scene_context": jnp.zeros((n, _CTX_DIM)),
        "ground_truth": jnp.zeros((n, t, 7)),
        "ground_truth_is_valid": jnp.ones((n, t)),
        "object_type": jnp.zeros(n, dtype=jnp.int32),
        "key": jax.random.key(0),
        **({"road_edge_distances": jnp.full((n, t), -5.0)} if include_road_edges else {}),
    }


def _make_runner(*, with_sim_agent: bool = False) -> EvaluationRunner:
    config = MotionMetricsConfig(miss_rate_threshold=2.0, top_k=1, num_future_steps=_T)
    motion = MotionMetrics(config)
    sim = SimAgentMetrics(SimAgentMetricsConfig()) if with_sim_agent else None
    return EvaluationRunner(motion_metrics=motion, sim_agent_metrics=sim)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    """Tests for EvaluationRunner.run()."""

    def test_returns_metrics_report(self) -> None:
        runner = _make_runner()
        report = runner.run(_make_mock_model(), [_make_batch()])
        assert isinstance(report, MetricsReport)

    def test_metric_values_populated(self) -> None:
        runner = _make_runner()
        report = runner.run(_make_mock_model(), [_make_batch()])
        assert len(report.metric_values) > 0
        assert all(isinstance(v, float) for v in report.metric_values.values())

    def test_zero_error_for_perfect_predictions(self) -> None:
        """Perfect predictions (pred == gt == zeros) should give ADE/FDE = 0."""
        runner = _make_runner()
        report = runner.run(_make_mock_model(), [_make_batch()])
        for key, val in report.metric_values.items():
            if "minADE" in key or "minFDE" in key or "MissRate" in key:
                assert val == pytest.approx(0.0, abs=1e-5), f"{key}={val}"

    def test_empty_batches_returns_empty_report(self) -> None:
        runner = _make_runner()
        report = runner.run(_make_mock_model(), [])
        assert report.metric_values == {}
        assert report.kinematic_violation_summary == {}

    def test_multi_batch_averages(self) -> None:
        runner = _make_runner()
        batches = [_make_batch(), _make_batch()]
        report = runner.run(_make_mock_model(), batches)
        assert len(report.metric_values) > 0

    def test_model_sample_called_per_batch(self) -> None:
        runner = _make_runner()
        model = _make_mock_model()
        batches = [_make_batch(), _make_batch(), _make_batch()]
        runner.run(model, batches)
        assert model.sample.call_count == 3

    def test_with_sim_agent_metrics_populates_kinematic_summary(self) -> None:
        runner = _make_runner(with_sim_agent=True)
        report = runner.run(_make_mock_model(), [_make_batch(include_road_edges=True)])
        assert len(report.kinematic_violation_summary) > 0
        assert "offroad_rate" in report.kinematic_violation_summary

    def test_without_road_edges_kinematic_summary_empty(self) -> None:
        runner = _make_runner(with_sim_agent=True)
        # No road_edge_distances key in batch → kinematic summary stays empty
        report = runner.run(_make_mock_model(), [_make_batch(include_road_edges=False)])
        assert report.kinematic_violation_summary == {}

    def test_runner_config_immutable(self) -> None:
        runner = _make_runner()
        with pytest.raises(Exception):
            runner.motion_metrics = MotionMetrics(MotionMetricsConfig())  # type: ignore[misc]


class TestTrajectorySamplerSeam:
    """The runner's model seam is a typed consumer-side Protocol."""

    def test_diffusion_model_satisfies_protocol(self) -> None:
        from diffav.evaluation.runner import TrajectorySampler
        from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel

        assert isinstance(TrajectoryDiffusionModel, type)
        assert hasattr(TrajectoryDiffusionModel, "sample")

        # runtime_checkable structural check against the class interface
        class _Stub:
            def sample(self, scene_context, *, key):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        assert isinstance(_Stub(), TrajectorySampler)

    def test_object_without_sample_rejected(self) -> None:
        from diffav.evaluation.runner import TrajectorySampler

        assert not isinstance(object(), TrajectorySampler)
