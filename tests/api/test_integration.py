"""End-to-end integration tests for the ScenarioMiner SDK."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from simulacrax.api import (
    create_scenario_miner,
    FailureCase,
    MinerConfig,
    Scenario,
    ScenarioMiner,
)
from simulacrax.core.types import MetricsReport, TrajectoryPrediction


pytestmark = [pytest.mark.slow, pytest.mark.integration]


class TestScenarioMinerIntegration:
    @pytest.fixture(scope="class")
    def miner(self) -> ScenarioMiner:
        # Use small diffusion steps so the test is fast on both CPU and GPU.
        config = MinerConfig(
            batch_size=4,
            max_agents=4,
            prediction_horizon=10,
            context_dim=32,
            num_diffusion_steps=10,
        )
        return create_scenario_miner(config)

    def test_full_sdk_workflow(self, miner: ScenarioMiner) -> None:
        """generate -> evaluate_planner -> adversarial_search end-to-end."""
        scenarios = miner.generate("unprotected_left_turn", "low", count=3)
        assert len(scenarios) == 3
        assert all(isinstance(s, Scenario) for s in scenarios)

        def mock_planner(context):
            n = len(context.agent_states)
            return TrajectoryPrediction(
                trajectories=jnp.zeros((n, miner.config.prediction_horizon, 4)),
                agent_ids=tuple(f"agent_{i}" for i in range(n)),
            )

        report = miner.evaluate_planner(mock_planner, scenarios)
        assert isinstance(report, MetricsReport)
        assert "ade" in report.metric_values
        assert "fde" in report.metric_values

        cases = miner.adversarial_search(mock_planner, budget=2, perturbation_steps=2)
        assert len(cases) <= 2
        assert all(isinstance(fc, FailureCase) for fc in cases)

    def test_create_scenario_miner_no_path(self) -> None:
        config = MinerConfig(
            max_agents=4, prediction_horizon=10, context_dim=32, num_diffusion_steps=10
        )
        miner = create_scenario_miner(config)
        assert isinstance(miner, ScenarioMiner)
        scenarios = miner.generate("forward", "low", count=1)
        assert len(scenarios) == 1

    def test_create_scenario_miner_bad_path_raises(self) -> None:
        config = MinerConfig(
            model_path="/nonexistent/path",
            max_agents=4,
            prediction_horizon=10,
            context_dim=32,
        )
        with pytest.raises(FileNotFoundError):
            create_scenario_miner(config)


class TestCheckpointLoadSuccessPath:
    """create_scenario_miner restores real checkpoint weights (TQ-3)."""

    def test_loaded_miner_carries_checkpoint_weights(self, tmp_path: Path) -> None:
        import jax
        from flax import nnx

        from simulacrax.core.constants import SCENE_BACKBONE_ARCHITECTURE_VERSION
        from simulacrax.models.checkpointing import (
            CheckpointConfig,
            SimulacraxCheckpointManager,
        )

        config = MinerConfig(
            max_agents=4, prediction_horizon=10, context_dim=32, num_diffusion_steps=10
        )
        # Non-default seed: a fresh default-init model would NOT match these
        # weights, so equality below proves the checkpoint actually loaded.
        source = create_scenario_miner(config, rngs=nnx.Rngs(params=jax.random.key(123)))

        checkpoint_dir = tmp_path / "ckpt"
        manager_config = CheckpointConfig(checkpoint_dir=str(checkpoint_dir))
        with SimulacraxCheckpointManager(manager_config) as manager:
            manager.save(
                source.model,
                step=1,
                loss=0.0,
                architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION,
            )

        loaded = create_scenario_miner(
            MinerConfig(
                model_path=str(checkpoint_dir),
                max_agents=4,
                prediction_horizon=10,
                context_dim=32,
                num_diffusion_steps=10,
            )
        )
        source_params = jax.tree_util.tree_leaves(nnx.state(source.model, nnx.Param))
        loaded_params = jax.tree_util.tree_leaves(nnx.state(loaded.model, nnx.Param))
        assert all(
            bool(jnp.array_equal(a, b)) for a, b in zip(source_params, loaded_params, strict=True)
        )
