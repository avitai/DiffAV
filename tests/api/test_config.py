"""Tests for simulacrax.api.config data containers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from simulacrax.api.config import FailureCase, MinerConfig, Scenario
from simulacrax.core.types import (
    AgentState,
    AgentType,
    ScenarioMetadata,
    SceneContext,
    TrajectoryPrediction,
)


def _make_agent(x: float = 0.0) -> AgentState:
    return AgentState(
        position=jnp.array([x, 0.0]),
        heading=0.0,
        velocity=5.0,
        acceleration=0.0,
        agent_type=AgentType.VEHICLE,
    )


def _make_scene() -> SceneContext:
    return SceneContext(
        ego_state=_make_agent(),
        agent_states=(_make_agent(10.0),),
        map_features=(),
        timestamps=jnp.zeros(11),
    )


def _make_prediction(n_agents: int = 1, steps: int = 10) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        trajectories=jnp.zeros((n_agents, steps, 4)),
        agent_ids=tuple(f"agent_{i}" for i in range(n_agents)),
    )


def _make_metadata() -> ScenarioMetadata:
    return ScenarioMetadata(scenario_id="s1", source_dataset="wod")


class TestMinerConfig:
    def test_default_values(self) -> None:
        config = MinerConfig()
        assert config.model_path == ""
        assert config.batch_size == 32
        assert config.max_agents == 32
        assert config.prediction_horizon == 80
        assert config.context_dim == 128

    def test_custom_values(self) -> None:
        config = MinerConfig(
            model_path="checkpoints/v1",
            batch_size=16,
            max_agents=16,
            prediction_horizon=40,
            context_dim=64,
        )
        assert config.model_path == "checkpoints/v1"
        assert config.batch_size == 16
        assert config.max_agents == 16
        assert config.prediction_horizon == 40
        assert config.context_dim == 64

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            MinerConfig(batch_size=0)

    def test_invalid_max_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="max_agents"):
            MinerConfig(max_agents=-1)

    def test_invalid_prediction_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_horizon"):
            MinerConfig(prediction_horizon=0)

    def test_invalid_context_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="context_dim"):
            MinerConfig(context_dim=0)

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        config = MinerConfig()
        with pytest.raises(FrozenInstanceError):
            config.batch_size = 99  # type: ignore[misc]


class TestScenario:
    def test_construction(self) -> None:
        s = Scenario(
            context=_make_scene(),
            predictions=_make_prediction(),
            metadata=_make_metadata(),
        )
        assert s.metadata.scenario_id == "s1"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        s = Scenario(
            context=_make_scene(),
            predictions=_make_prediction(),
            metadata=_make_metadata(),
        )
        with pytest.raises(FrozenInstanceError):
            s.metadata = _make_metadata()  # type: ignore[misc]


class TestFailureCase:
    def test_construction(self) -> None:
        fc = FailureCase(
            scenario=Scenario(
                context=_make_scene(),
                predictions=_make_prediction(),
                metadata=_make_metadata(),
            ),
            failure_mode="kinematics_violation",
            severity=2.5,
        )
        assert fc.failure_mode == "kinematics_violation"
        assert fc.severity == pytest.approx(2.5)

    def test_invalid_severity_raises(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            FailureCase(
                scenario=Scenario(
                    context=_make_scene(),
                    predictions=_make_prediction(),
                    metadata=_make_metadata(),
                ),
                failure_mode="x",
                severity=-1.0,
            )

    def test_empty_failure_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="failure_mode"):
            FailureCase(
                scenario=Scenario(
                    context=_make_scene(),
                    predictions=_make_prediction(),
                    metadata=_make_metadata(),
                ),
                failure_mode="",
                severity=1.0,
            )

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        fc = FailureCase(
            scenario=Scenario(
                context=_make_scene(),
                predictions=_make_prediction(),
                metadata=_make_metadata(),
            ),
            failure_mode="x",
            severity=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            fc.severity = 1.0  # type: ignore[misc]

    def test_zero_severity_allowed(self) -> None:
        fc = FailureCase(
            scenario=Scenario(
                context=_make_scene(),
                predictions=_make_prediction(),
                metadata=_make_metadata(),
            ),
            failure_mode="test",
            severity=0.0,
        )
        assert fc.severity == 0.0

    def test_failure_mode_stored(self) -> None:
        fc = FailureCase(
            scenario=Scenario(
                context=_make_scene(),
                predictions=_make_prediction(),
                metadata=_make_metadata(),
            ),
            failure_mode="collision_risk",
            severity=1.5,
        )
        assert fc.failure_mode == "collision_risk"
