"""Tests for core domain types.

Verifies construction, immutability (frozen=True), field access,
and serialization to/from dict for all domain dataclasses.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from simulacrax.core.types import (
    AgentState,
    AgentType,
    MapFeature,
    MapFeatureType,
    MetricsReport,
    ScenarioMetadata,
    SceneContext,
    TrajectoryPrediction,
)


# ---------------------------------------------------------------------------
# AgentType enum
# ---------------------------------------------------------------------------


class TestAgentType:
    """Tests for the AgentType enum."""

    def test_has_vehicle(self) -> None:
        """AgentType should include VEHICLE."""
        assert AgentType.VEHICLE.value == "vehicle"

    def test_has_pedestrian(self) -> None:
        """AgentType should include PEDESTRIAN."""
        assert AgentType.PEDESTRIAN.value == "pedestrian"

    def test_has_cyclist(self) -> None:
        """AgentType should include CYCLIST."""
        assert AgentType.CYCLIST.value == "cyclist"


# ---------------------------------------------------------------------------
# MapFeatureType enum
# ---------------------------------------------------------------------------


class TestMapFeatureType:
    """Tests for the MapFeatureType enum."""

    def test_has_lane(self) -> None:
        """MapFeatureType should include LANE."""
        assert MapFeatureType.LANE.value == "lane"

    def test_has_crosswalk(self) -> None:
        """MapFeatureType should include CROSSWALK."""
        assert MapFeatureType.CROSSWALK.value == "crosswalk"

    def test_has_traffic_signal(self) -> None:
        """MapFeatureType should include TRAFFIC_SIGNAL."""
        assert MapFeatureType.TRAFFIC_SIGNAL.value == "traffic_signal"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for AgentState frozen dataclass."""

    def test_construction(self) -> None:
        """AgentState can be constructed with required fields."""
        state = AgentState(
            position=jnp.array([1.0, 2.0]),
            heading=0.5,
            velocity=10.0,
            acceleration=0.0,
            agent_type=AgentType.VEHICLE,
        )
        assert state.heading == 0.5
        assert state.velocity == 10.0
        assert state.agent_type == AgentType.VEHICLE

    def test_frozen(self) -> None:
        """AgentState should be immutable."""
        state = AgentState(
            position=jnp.array([1.0, 2.0]),
            heading=0.5,
            velocity=10.0,
            acceleration=0.0,
            agent_type=AgentType.VEHICLE,
        )
        with pytest.raises(AttributeError):
            state.heading = 1.0  # type: ignore[misc]

    def test_position_shape(self) -> None:
        """Position should be a 2D array (x, y)."""
        state = AgentState(
            position=jnp.array([3.0, 4.0]),
            heading=0.0,
            velocity=0.0,
            acceleration=0.0,
            agent_type=AgentType.PEDESTRIAN,
        )
        assert state.position.shape == (2,)


# ---------------------------------------------------------------------------
# MapFeature
# ---------------------------------------------------------------------------


class TestMapFeature:
    """Tests for MapFeature frozen dataclass."""

    def test_construction(self) -> None:
        """MapFeature can be constructed with polyline points."""
        points = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
        feature = MapFeature(
            polyline_points=points,
            feature_type=MapFeatureType.LANE,
        )
        assert feature.feature_type == MapFeatureType.LANE
        assert feature.polyline_points.shape == (3, 2)

    def test_frozen(self) -> None:
        """MapFeature should be immutable."""
        feature = MapFeature(
            polyline_points=jnp.zeros((5, 2)),
            feature_type=MapFeatureType.CROSSWALK,
        )
        with pytest.raises(AttributeError):
            feature.feature_type = MapFeatureType.LANE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SceneContext
# ---------------------------------------------------------------------------


class TestSceneContext:
    """Tests for SceneContext frozen dataclass."""

    def test_construction(self) -> None:
        """SceneContext bundles ego, agents, map, and timestamps."""
        ego = AgentState(
            position=jnp.array([0.0, 0.0]),
            heading=0.0,
            velocity=0.0,
            acceleration=0.0,
            agent_type=AgentType.VEHICLE,
        )
        agents = [
            AgentState(
                position=jnp.array([5.0, 5.0]),
                heading=1.0,
                velocity=3.0,
                acceleration=0.0,
                agent_type=AgentType.VEHICLE,
            ),
        ]
        map_features = [
            MapFeature(
                polyline_points=jnp.zeros((4, 2)),
                feature_type=MapFeatureType.LANE,
            ),
        ]
        timestamps = jnp.arange(11, dtype=jnp.float32) * 0.1

        scene = SceneContext(
            ego_state=ego,
            agent_states=tuple(agents),
            map_features=tuple(map_features),
            timestamps=timestamps,
        )
        assert scene.ego_state is ego
        assert len(scene.agent_states) == 1
        assert len(scene.map_features) == 1
        assert scene.timestamps.shape == (11,)

    def test_frozen(self) -> None:
        """SceneContext should be immutable."""
        ego = AgentState(
            position=jnp.array([0.0, 0.0]),
            heading=0.0,
            velocity=0.0,
            acceleration=0.0,
            agent_type=AgentType.VEHICLE,
        )
        scene = SceneContext(
            ego_state=ego,
            agent_states=(),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        with pytest.raises(AttributeError):
            scene.ego_state = ego  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrajectoryPrediction
# ---------------------------------------------------------------------------


class TestTrajectoryPrediction:
    """Tests for TrajectoryPrediction frozen dataclass."""

    def test_construction(self) -> None:
        """TrajectoryPrediction wraps a [num_agents, future_steps, state_dim] tensor."""
        # 2 agents, 80 future steps, state_dim=4 (x, y, heading, v)
        trajectories = jnp.zeros((2, 80, 4), dtype=jnp.float32)
        pred = TrajectoryPrediction(
            trajectories=trajectories,
            agent_ids=("agent_0", "agent_1"),
        )
        assert pred.trajectories.shape == (2, 80, 4)
        assert len(pred.agent_ids) == 2

    def test_frozen(self) -> None:
        """TrajectoryPrediction should be immutable."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, 80, 4)),
            agent_ids=("agent_0",),
        )
        with pytest.raises(AttributeError):
            pred.agent_ids = ("new",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScenarioMetadata
# ---------------------------------------------------------------------------


class TestScenarioMetadata:
    """Tests for ScenarioMetadata frozen dataclass."""

    def test_construction(self) -> None:
        """ScenarioMetadata holds identification and tagging info."""
        meta = ScenarioMetadata(
            scenario_id="scene_001",
            source_dataset="waymo_open_dataset",
            tags=("unprotected_left_turn", "dense_traffic"),
        )
        assert meta.scenario_id == "scene_001"
        assert "unprotected_left_turn" in meta.tags

    def test_frozen(self) -> None:
        """ScenarioMetadata should be immutable."""
        meta = ScenarioMetadata(
            scenario_id="s1",
            source_dataset="wod",
        )
        with pytest.raises(AttributeError):
            meta.scenario_id = "s2"  # type: ignore[misc]

    def test_default_tags(self) -> None:
        """Tags default to empty tuple."""
        meta = ScenarioMetadata(
            scenario_id="s1",
            source_dataset="wod",
        )
        assert meta.tags == ()


# ---------------------------------------------------------------------------
# MetricsReport
# ---------------------------------------------------------------------------


class TestMetricsReport:
    """Tests for MetricsReport frozen dataclass."""

    def test_construction(self) -> None:
        """MetricsReport holds all metric values."""
        report = MetricsReport(
            metric_values={"ade": 1.5, "fde": 3.2, "collision_rate": 0.02},
            per_scenario_breakdown={"unprotected_left": {"ade": 2.0}},
        )
        assert report.metric_values["ade"] == 1.5
        assert "unprotected_left" in report.per_scenario_breakdown

    def test_frozen(self) -> None:
        """MetricsReport should be immutable."""
        report = MetricsReport(
            metric_values={"ade": 1.0},
        )
        with pytest.raises(AttributeError):
            report.metric_values = {}  # type: ignore[misc]

    def test_default_breakdown(self) -> None:
        """Per-scenario breakdown defaults to empty dict."""
        report = MetricsReport(metric_values={"ade": 1.0})
        assert report.per_scenario_breakdown == {}


class TestSdkBoundaryEnums:
    """Density and ScenarioType define the SDK's input vocabulary."""

    def test_density_values(self) -> None:
        from simulacrax.core.types import Density

        assert [d.value for d in Density] == ["low", "medium", "high"]

    def test_scenario_type_values(self) -> None:
        from simulacrax.core.types import ScenarioType

        assert [s.value for s in ScenarioType] == [
            "forward",
            "lane_change",
            "unprotected_left_turn",
            "adversarial",
        ]

    def test_density_is_str_compatible(self) -> None:
        from simulacrax.core.types import Density

        assert Density("medium") == "medium"
