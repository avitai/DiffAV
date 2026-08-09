"""Shared fixtures for API tests."""

from __future__ import annotations

import pytest

from diffav.api.config import MinerConfig
from diffav.api.scenario_miner import ScenarioMiner
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)
from tests import support


@pytest.fixture()
def small_config() -> MinerConfig:
    return MinerConfig(
        model_path="",
        batch_size=4,
        max_agents=4,
        prediction_horizon=10,
        context_dim=32,
    )


@pytest.fixture()
def small_model(small_config: MinerConfig) -> TrajectoryDiffusionModel:
    model_cfg = support.make_diffusion_config(
        num_agents_max=small_config.max_agents,
        future_steps=small_config.prediction_horizon,
        context_dim=small_config.context_dim,
    )
    return support.make_diffusion_model(model_cfg, seed=42)


@pytest.fixture()
def miner(small_config: MinerConfig, small_model: TrajectoryDiffusionModel) -> ScenarioMiner:
    return ScenarioMiner(model=small_model, config=small_config)
