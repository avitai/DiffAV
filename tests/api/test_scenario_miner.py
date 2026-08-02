"""Tests for ScenarioMiner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from simulacrax.alignment.scenario_steering import ScenarioSteeringConfig
from simulacrax.api.config import Scenario
from simulacrax.api.scenario_miner import ScenarioMiner
from simulacrax.core.types import TrajectoryPrediction
from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionModel


class TestScenarioMinerGenerate:
    def test_returns_list_of_scenarios(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=3)
        assert isinstance(scenarios, list)
        assert len(scenarios) == 3
        assert all(isinstance(s, Scenario) for s in scenarios)

    def test_predictions_have_correct_shape(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=2)
        for s in scenarios:
            traj = s.predictions.trajectories
            assert traj.ndim == 3
            assert traj.shape[1] == miner.config.prediction_horizon
            assert traj.shape[2] == 4

    def test_scenario_has_context(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=1)
        s = scenarios[0]
        assert s.context.ego_state is not None
        assert len(s.context.agent_states) > 0

    def test_density_affects_agent_count(self, miner: ScenarioMiner) -> None:
        low = miner.generate(scenario_type="forward", density="low", count=1)
        high = miner.generate(scenario_type="forward", density="high", count=1)
        assert len(high[0].context.agent_states) >= len(low[0].context.agent_states)

    def test_metadata_tags_include_scenario_type(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="lane_change", density="low", count=1)
        assert "lane_change" in scenarios[0].metadata.tags

    def test_unique_scenario_ids(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=5)
        ids = [s.metadata.scenario_id for s in scenarios]
        assert len(set(ids)) == len(ids)

    def test_count_zero_returns_empty(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=0)
        assert scenarios == []


class TestScenarioMinerEvaluatePlanner:
    def test_returns_metrics_report(self, miner: ScenarioMiner) -> None:
        from simulacrax.core.types import MetricsReport

        scenarios = miner.generate(scenario_type="forward", density="low", count=2)

        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        report = miner.evaluate_planner(dummy_planner, scenarios)
        assert isinstance(report, MetricsReport)

    def test_report_has_ade_and_fde(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=2)

        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        report = miner.evaluate_planner(dummy_planner, scenarios)
        assert "ade" in report.metric_values
        assert "fde" in report.metric_values

    def test_perfect_planner_has_lower_ade_than_bad_planner(self, miner: ScenarioMiner) -> None:
        scenarios = miner.generate(scenario_type="forward", density="low", count=2)

        # Perfect: return identical predictions to reference
        def perfect_planner(context):
            for s in scenarios:
                if s.context is context:
                    return s.predictions
            return scenarios[0].predictions

        def bad_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.ones((n, miner.config.prediction_horizon, 4)) * 100.0,
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        perfect_report = miner.evaluate_planner(perfect_planner, scenarios)
        bad_report = miner.evaluate_planner(bad_planner, scenarios)
        assert perfect_report.metric_values["ade"] < bad_report.metric_values["ade"]

    def test_empty_scenarios_returns_empty_report(self, miner: ScenarioMiner) -> None:
        from simulacrax.core.types import MetricsReport

        def dummy_planner(context):
            return TrajectoryPrediction(
                trajectories=jnp.zeros((1, miner.config.prediction_horizon, 4)),
                agent_ids=("agent_0",),
            )

        report = miner.evaluate_planner(dummy_planner, [])
        assert isinstance(report, MetricsReport)
        assert report.metric_values == {}


@pytest.mark.slow
class TestScenarioMinerAdversarialSearch:
    def test_returns_list_of_failure_cases(self, miner: ScenarioMiner) -> None:
        from simulacrax.api.config import FailureCase

        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        cases = miner.adversarial_search(dummy_planner, budget=2)
        assert isinstance(cases, list)
        assert all(isinstance(fc, FailureCase) for fc in cases)

    def test_budget_caps_results(self, miner: ScenarioMiner) -> None:
        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        cases = miner.adversarial_search(dummy_planner, budget=3)
        assert len(cases) <= 3

    def test_severity_is_non_negative(self, miner: ScenarioMiner) -> None:
        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        cases = miner.adversarial_search(dummy_planner, budget=2)
        for fc in cases:
            assert fc.severity >= 0.0

    def test_failure_mode_is_non_empty(self, miner: ScenarioMiner) -> None:
        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        cases = miner.adversarial_search(dummy_planner, budget=2)
        for fc in cases:
            assert fc.failure_mode != ""

    def test_results_sorted_by_severity_descending(self, miner: ScenarioMiner) -> None:
        def dummy_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        cases = miner.adversarial_search(dummy_planner, budget=4, perturbation_steps=2)
        severities = [fc.severity for fc in cases]
        assert severities == sorted(severities, reverse=True)

    def test_budget_zero_returns_empty(self, miner: ScenarioMiner) -> None:
        def dummy_planner(_):
            return TrajectoryPrediction(
                trajectories=jnp.zeros((1, miner.config.prediction_horizon, 4)),
                agent_ids=("agent_0",),
            )

        cases = miner.adversarial_search(dummy_planner, budget=0)
        assert cases == []


class TestBoundaryFailFast:
    """The SDK boundary rejects invalid inputs instead of silently defaulting."""

    def test_unknown_density_raises(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="bogus"):
            miner.generate(scenario_type="forward", density="bogus", count=1)

    def test_unknown_scenario_type_raises(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="u_turn"):
            miner.generate(scenario_type="u_turn", density="low", count=1)

    def test_negative_count_raises(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="count"):
            miner.generate(scenario_type="forward", density="low", count=-1)

    def test_steer_validates_before_work(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="bogus"):
            miner.steer(
                "forward",
                "bogus",
                steering_config=ScenarioSteeringConfig(target_scenario="lane_change"),
                count=1,
            )

    def test_adversarial_search_negative_budget_raises(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="budget"):
            miner.adversarial_search(lambda ctx: None, budget=-2)  # type: ignore[arg-type,return-value]

    def test_enum_values_accepted(self, miner: ScenarioMiner) -> None:
        from simulacrax.core.types import Density, ScenarioType

        scenarios = miner.generate(
            scenario_type=ScenarioType.LANE_CHANGE, density=Density.LOW, count=1
        )
        assert len(scenarios) == 1


def _zero_planner(miner: ScenarioMiner):
    """Planner returning all-zero trajectories (overlapping agents)."""

    def planner(context):
        n = len(context.agent_states)
        return TrajectoryPrediction(
            trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
            agent_ids=tuple(f"agent_{i}" for i in range(n)),
        )

    return planner


@pytest.mark.slow
class TestSteer:
    """steer() fine-tunes a clone on ranked pairs and tags the output."""

    def _steering_config(self) -> ScenarioSteeringConfig:
        return ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=1.0, num_candidates=4
        )

    def test_returns_count_scenarios_with_steered_metadata(self, miner: ScenarioMiner) -> None:
        scenarios = miner.steer(
            "forward",
            "low",
            steering_config=self._steering_config(),
            count=2,
            key=jax.random.key(0),
        )
        assert len(scenarios) == 2
        for i, scenario in enumerate(scenarios):
            assert scenario.metadata.scenario_id == f"steered_forward_{i:06d}"
            assert scenario.metadata.source_dataset == "simulacrax_steered"
            assert "steered_forward" in scenario.metadata.tags

    def test_miner_model_untouched(self, miner: ScenarioMiner) -> None:
        from flax import nnx

        params_before = [
            p.copy() for p in jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        ]
        miner.steer(
            "forward",
            "low",
            steering_config=self._steering_config(),
            count=1,
            key=jax.random.key(0),
        )
        params_after = jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        assert all(jnp.array_equal(a, b) for a, b in zip(params_before, params_after))

    def test_invalid_target_scenario_raises(self, miner: ScenarioMiner) -> None:
        with pytest.raises(ValueError, match="u_turn"):
            miner.steer(
                "forward",
                "low",
                steering_config=ScenarioSteeringConfig(target_scenario="u_turn"),
                count=1,
            )

    def test_count_zero_returns_empty(self, miner: ScenarioMiner) -> None:
        scenarios = miner.steer("forward", "low", steering_config=self._steering_config(), count=0)
        assert scenarios == []


@pytest.mark.slow
class TestSteerWeightSoup:
    """steer() with WEIGHT_SOUP interpolates a full-pressure expert."""

    @staticmethod
    def _soup_config(strength: float) -> ScenarioSteeringConfig:
        from simulacrax.alignment.scenario_steering import SteeringStrategy

        return ScenarioSteeringConfig(
            target_scenario="forward",
            strategy=SteeringStrategy.WEIGHT_SOUP,
            steering_strength=strength,
            num_candidates=4,
        )

    def test_returns_steered_scenarios(self, miner: ScenarioMiner) -> None:
        scenarios = miner.steer(
            "forward",
            "low",
            steering_config=self._soup_config(0.5),
            count=2,
            key=jax.random.key(0),
        )
        assert len(scenarios) == 2
        assert scenarios[0].metadata.source_dataset == "simulacrax_steered"

    def test_strength_moves_generation(self, miner: ScenarioMiner) -> None:
        """Same key, different soup strength: the interpolated weights differ."""
        weak = miner.steer(
            "forward",
            "low",
            steering_config=self._soup_config(0.0),
            count=1,
            key=jax.random.key(3),
        )
        strong = miner.steer(
            "forward",
            "low",
            steering_config=self._soup_config(1.0),
            count=1,
            key=jax.random.key(3),
        )
        assert not jnp.array_equal(
            weak[0].predictions.trajectories, strong[0].predictions.trajectories
        )

    def test_miner_model_untouched(self, miner: ScenarioMiner) -> None:
        from flax import nnx

        params_before = [
            p.copy() for p in jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        ]
        miner.steer(
            "forward",
            "low",
            steering_config=self._soup_config(0.7),
            count=1,
            key=jax.random.key(0),
        )
        params_after = jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        assert all(jnp.array_equal(a, b) for a, b in zip(params_before, params_after))


class TestBatchedGeneration:
    """generate() samples in vmapped chunks of ``config.batch_size``."""

    @staticmethod
    def _with_batch_size(miner: ScenarioMiner, batch_size: int) -> ScenarioMiner:
        return ScenarioMiner(model=miner.model, config=replace(miner.config, batch_size=batch_size))

    def test_sample_invocations_scale_with_chunks(
        self, miner: ScenarioMiner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """count=8 at batch_size=4 traces sample twice, not eight times."""
        calls = {"n": 0}
        original = TrajectoryDiffusionModel.sample

        def spy(self: TrajectoryDiffusionModel, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(TrajectoryDiffusionModel, "sample", spy)
        self._with_batch_size(miner, 4).generate("forward", "low", count=8)
        assert calls["n"] == 2

    def test_results_independent_of_chunk_layout(self, miner: ScenarioMiner) -> None:
        """Per-scenario keys make predictions invariant to batch_size.

        Tolerance: different vmap batch sizes change XLA fusion, and the ten
        reverse-diffusion steps amplify f32 rounding on ~1e3-scale untrained
        outputs (CPU exceeds rtol 1e-5). Different keys differ at order 1e3,
        so rtol 1e-3 still cleanly pins the same-key invariance.
        """
        key = jax.random.key(7)
        first = self._with_batch_size(miner, 2).generate("forward", "low", count=5, key=key)
        second = self._with_batch_size(miner, 5).generate("forward", "low", count=5, key=key)
        for a, b in zip(first, second, strict=True):
            assert jnp.allclose(
                a.predictions.trajectories,
                b.predictions.trajectories,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_ragged_final_chunk_preserves_count_and_ids(self, miner: ScenarioMiner) -> None:
        scenarios = self._with_batch_size(miner, 2).generate("forward", "low", count=5)
        assert len(scenarios) == 5
        expected_ids = [f"forward_{i:06d}" for i in range(5)]
        assert [s.metadata.scenario_id for s in scenarios] == expected_ids

    def test_generation_deterministic_for_fixed_key(self, miner: ScenarioMiner) -> None:
        key = jax.random.key(11)
        first = miner.generate("forward", "low", count=3, key=key)
        second = miner.generate("forward", "low", count=3, key=key)
        for a, b in zip(first, second, strict=True):
            assert jnp.array_equal(a.predictions.trajectories, b.predictions.trajectories)


@pytest.mark.slow
class TestAdversarialSearchSemantics:
    """budget = candidates explored; threshold filters; keys vary; predictions match."""

    def test_explores_exactly_budget_candidates(
        self, miner: ScenarioMiner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No 4x overgeneration: exactly budget candidates are generated."""
        requested: dict[str, int] = {}
        original = ScenarioMiner.generate

        def spy(self, scenario_type, density, *, count, key=None):  # type: ignore[no-untyped-def]
            requested["count"] = count
            return original(self, scenario_type, density, count=count, key=key)

        monkeypatch.setattr(ScenarioMiner, "generate", spy)
        miner.adversarial_search(_zero_planner(miner), budget=2, perturbation_steps=1)
        assert requested["count"] == 2

    def test_severity_threshold_filters_healthy_planners(self, miner: ScenarioMiner) -> None:
        """Candidates below the severity threshold are not failure cases."""
        cases = miner.adversarial_search(
            _zero_planner(miner),
            budget=2,
            severity_threshold=1e9,
            perturbation_steps=1,
        )
        assert cases == []

    def test_caller_key_varies_search(self, miner: ScenarioMiner) -> None:
        """Different keys explore different candidates."""
        first = miner.adversarial_search(
            _zero_planner(miner), budget=1, perturbation_steps=1, key=jax.random.key(1)
        )
        second = miner.adversarial_search(
            _zero_planner(miner), budget=1, perturbation_steps=1, key=jax.random.key(2)
        )
        assert len(first) == 1
        assert len(second) == 1
        assert not jnp.array_equal(
            first[0].scenario.predictions.trajectories,
            second[0].scenario.predictions.trajectories,
        )

    def test_predictions_regenerated_on_perturbed_context(self, miner: ScenarioMiner) -> None:
        """Failure cases carry predictions for the perturbed scene, not the seed.

        The candidate key is the first split of the caller key (documented
        contract), so the unperturbed seed predictions are reproducible.
        """
        key = jax.random.key(3)
        candidate_key, _ = jax.random.split(key)
        seeds = miner.generate("adversarial", "medium", count=1, key=candidate_key)

        cases = miner.adversarial_search(
            _zero_planner(miner), budget=1, perturbation_steps=3, key=key
        )
        assert len(cases) == 1
        assert not jnp.array_equal(
            cases[0].scenario.predictions.trajectories,
            seeds[0].predictions.trajectories,
        )


@pytest.mark.slow
class TestSteerGuidance:
    """steer() with GUIDANCE samples with reward-gradient ascent, no training."""

    @staticmethod
    def _guidance_config(strength: float) -> ScenarioSteeringConfig:
        from simulacrax.alignment.scenario_steering import SteeringStrategy

        return ScenarioSteeringConfig(
            target_scenario="forward",
            strategy=SteeringStrategy.GUIDANCE,
            steering_strength=strength,
        )

    def test_returns_steered_scenarios(self, miner: ScenarioMiner) -> None:
        scenarios = miner.steer(
            "forward",
            "low",
            steering_config=self._guidance_config(0.5),
            count=2,
            key=jax.random.key(0),
        )
        assert len(scenarios) == 2
        assert scenarios[0].metadata.source_dataset == "simulacrax_steered"

    def test_strength_moves_generation(self, miner: ScenarioMiner) -> None:
        weak = miner.steer(
            "forward",
            "low",
            steering_config=self._guidance_config(0.0),
            count=1,
            key=jax.random.key(3),
        )
        strong = miner.steer(
            "forward",
            "low",
            steering_config=self._guidance_config(1.0),
            count=1,
            key=jax.random.key(3),
        )
        assert not jnp.array_equal(
            weak[0].predictions.trajectories, strong[0].predictions.trajectories
        )

    def test_miner_model_untouched(self, miner: ScenarioMiner) -> None:
        from flax import nnx

        params_before = [
            p.copy() for p in jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        ]
        miner.steer(
            "forward",
            "low",
            steering_config=self._guidance_config(0.8),
            count=1,
            key=jax.random.key(0),
        )
        params_after = jax.tree_util.tree_leaves(nnx.state(miner.model, nnx.Param))
        assert all(jnp.array_equal(a, b) for a, b in zip(params_before, params_after))


class TestClassifyFailureMode:
    """_classify_failure_mode labels the dominant physics-loss component."""

    def test_kinematics_dominant(self) -> None:
        from simulacrax.api.scenario_miner import _classify_failure_mode

        components = {"kinematic_loss": jnp.array(2.0), "collision_loss": jnp.array(1.0)}
        assert _classify_failure_mode(components) == "kinematics_violation"

    def test_collision_dominant(self) -> None:
        from simulacrax.api.scenario_miner import _classify_failure_mode

        components = {"kinematic_loss": jnp.array(0.5), "collision_loss": jnp.array(3.0)}
        assert _classify_failure_mode(components) == "collision_risk"

    def test_missing_components_default_to_kinematics(self) -> None:
        from simulacrax.api.scenario_miner import _classify_failure_mode

        assert _classify_failure_mode({}) == "kinematics_violation"
