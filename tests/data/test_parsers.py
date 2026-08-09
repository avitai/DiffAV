"""Tests for WOD scenario parsing functions.

Verifies that raw WOD dicts are correctly parsed into typed domain
objects (SceneContext, AgentState, MapFeature).
"""

from __future__ import annotations

import numpy as np
import pytest

from diffav.core.constants import (
    STATE_IS_SDC,
    STATE_VALID,
    WOD_CURRENT_TIME_INDEX,
    WOD_HISTORY_STEPS,
)
from diffav.core.types import (
    AgentState,
    AgentType,
    MapFeature,
    SceneContext,
)
from diffav.data.parsers import (
    parse_agent_tracks,
    parse_map_features,
    parse_scenario,
)
from tests.data.helpers import (
    NUM_OBJECTS,
    NUM_RG_POINTS,
)


# ---------------------------------------------------------------------------
# parse_scenario
# ---------------------------------------------------------------------------


class TestParseScenario:
    """Tests for parse_scenario()."""

    def test_returns_scene_context(self, raw_wod_dict: dict) -> None:
        """parse_scenario returns a SceneContext."""
        scene = parse_scenario(raw_wod_dict)
        assert isinstance(scene, SceneContext)

    def test_ego_state_is_sdc(self, raw_wod_dict: dict) -> None:
        """Ego state is the object marked as SDC (is_sdc=1)."""
        scene = parse_scenario(raw_wod_dict)
        assert scene.ego_state.agent_type == AgentType.VEHICLE

    def test_ego_position_at_current_time(self, raw_wod_dict: dict) -> None:
        """Ego position reflects the SDC at current_time_index."""
        scene = parse_scenario(raw_wod_dict)
        expected_x = raw_wod_dict["state/all/x"][0, WOD_CURRENT_TIME_INDEX]
        expected_y = raw_wod_dict["state/all/y"][0, WOD_CURRENT_TIME_INDEX]
        assert float(scene.ego_state.position[0]) == pytest.approx(expected_x, abs=1e-5)
        assert float(scene.ego_state.position[1]) == pytest.approx(expected_y, abs=1e-5)

    def test_agent_count(self, raw_wod_dict: dict) -> None:
        """Non-ego agents should be in agent_states (excluding SDC)."""
        scene = parse_scenario(raw_wod_dict)
        # 4 objects total, 1 is SDC → 3 non-ego agents
        assert len(scene.agent_states) == NUM_OBJECTS - 1

    def test_map_features_present(self, raw_wod_dict: dict) -> None:
        """Map features should be parsed from roadgraph data."""
        scene = parse_scenario(raw_wod_dict)
        assert len(scene.map_features) > 0

    def test_timestamps_shape(self, raw_wod_dict: dict) -> None:
        """Timestamps should cover history steps."""
        scene = parse_scenario(raw_wod_dict)
        assert scene.timestamps.shape == (WOD_HISTORY_STEPS,)

    def test_deterministic(self, raw_wod_dict: dict) -> None:
        """Parsing the same input twice produces identical results."""
        scene_a = parse_scenario(raw_wod_dict)
        scene_b = parse_scenario(raw_wod_dict)
        assert float(scene_a.ego_state.heading) == float(scene_b.ego_state.heading)
        assert len(scene_a.agent_states) == len(scene_b.agent_states)

    def test_minimal_scenario(self, minimal_raw_wod_dict: dict) -> None:
        """Parsing works with minimal single-object scenario."""
        scene = parse_scenario(minimal_raw_wod_dict)
        assert isinstance(scene, SceneContext)
        # Single SDC → no non-ego agents
        assert len(scene.agent_states) == 0


# ---------------------------------------------------------------------------
# parse_agent_tracks
# ---------------------------------------------------------------------------


class TestParseAgentTracks:
    """Tests for parse_agent_tracks()."""

    def test_returns_agent_states(self, raw_wod_dict: dict) -> None:
        """Returns a list of AgentState objects."""
        agents = parse_agent_tracks(raw_wod_dict)
        assert all(isinstance(a, AgentState) for a in agents)

    def test_agent_count(self, raw_wod_dict: dict) -> None:
        """Returns one AgentState per valid object."""
        agents = parse_agent_tracks(raw_wod_dict)
        assert len(agents) == NUM_OBJECTS

    def test_position_shape(self, raw_wod_dict: dict) -> None:
        """Each agent's position is a 2D array (x, y)."""
        agents = parse_agent_tracks(raw_wod_dict)
        for agent in agents:
            assert agent.position.shape == (2,)

    def test_agent_types_mapped(self, raw_wod_dict: dict) -> None:
        """WOD object types are mapped to AgentType enum values."""
        agents = parse_agent_tracks(raw_wod_dict)
        # Object types in fixture: VEHICLE, VEHICLE, PEDESTRIAN, CYCLIST
        assert agents[0].agent_type == AgentType.VEHICLE
        assert agents[2].agent_type == AgentType.PEDESTRIAN
        assert agents[3].agent_type == AgentType.CYCLIST

    def test_velocity_is_scalar(self, raw_wod_dict: dict) -> None:
        """Velocity should be scalar speed (magnitude of vx, vy)."""
        agents = parse_agent_tracks(raw_wod_dict)
        vx = raw_wod_dict["state/all/velocity_x"][0, WOD_CURRENT_TIME_INDEX]
        vy = raw_wod_dict["state/all/velocity_y"][0, WOD_CURRENT_TIME_INDEX]
        expected_speed = float(np.sqrt(vx**2 + vy**2))
        assert float(agents[0].velocity) == pytest.approx(expected_speed, abs=1e-4)


# ---------------------------------------------------------------------------
# Validity-aware parsing (no phantom agents)
# ---------------------------------------------------------------------------


class TestValidityAwareParsing:
    """Padded/invalid object rows must never become agents."""

    def test_padding_rows_excluded_from_tracks(self, padded_raw_wod_dict: dict) -> None:
        """Rows with valid=0 at the current step yield no AgentState."""
        agents = parse_agent_tracks(padded_raw_wod_dict)
        assert len(agents) == NUM_OBJECTS

    def test_no_phantom_positions(self, padded_raw_wod_dict: dict) -> None:
        """No agent sits at the (-1, -1) padding sentinel position."""
        agents = parse_agent_tracks(padded_raw_wod_dict)
        for agent in agents:
            position = (float(agent.position[0]), float(agent.position[1]))
            assert position != (-1.0, -1.0)

    def test_parse_scenario_excludes_padding(self, padded_raw_wod_dict: dict) -> None:
        """SceneContext contains only valid non-ego agents."""
        scene = parse_scenario(padded_raw_wod_dict)
        assert len(scene.agent_states) == NUM_OBJECTS - 1

    def test_parse_scenario_ego_unaffected_by_padding(self, padded_raw_wod_dict: dict) -> None:
        """The SDC is still identified correctly with padding rows present."""
        scene = parse_scenario(padded_raw_wod_dict)
        expected_x = padded_raw_wod_dict["state/all/x"][0, WOD_CURRENT_TIME_INDEX]
        assert float(scene.ego_state.position[0]) == pytest.approx(expected_x, abs=1e-5)

    def test_row_invalid_at_current_step_excluded(self, raw_wod_dict: dict) -> None:
        """An object invalid exactly at the current step is excluded."""
        raw = {key: np.copy(value) for key, value in raw_wod_dict.items()}
        raw[STATE_VALID][2, WOD_CURRENT_TIME_INDEX] = 0
        agents = parse_agent_tracks(raw)
        assert len(agents) == NUM_OBJECTS - 1

    def test_invalid_sdc_raises(self, raw_wod_dict: dict) -> None:
        """An SDC that is invalid at the current step is a hard error."""
        raw = {key: np.copy(value) for key, value in raw_wod_dict.items()}
        raw[STATE_VALID][0, WOD_CURRENT_TIME_INDEX] = 0
        with pytest.raises(ValueError, match="SDC"):
            parse_scenario(raw)

    def test_missing_sdc_raises(self, raw_wod_dict: dict) -> None:
        """A scenario with no SDC-flagged object is a hard error."""
        raw = {key: np.copy(value) for key, value in raw_wod_dict.items()}
        raw[STATE_IS_SDC][:] = 0
        with pytest.raises(ValueError, match="SDC"):
            parse_scenario(raw)


# ---------------------------------------------------------------------------
# parse_map_features
# ---------------------------------------------------------------------------


class TestParseMapFeatures:
    """Tests for parse_map_features()."""

    def test_returns_map_features(self, raw_wod_dict: dict) -> None:
        """Returns a list of MapFeature objects."""
        features = parse_map_features(raw_wod_dict)
        assert all(isinstance(f, MapFeature) for f in features)

    def test_feature_count(self, raw_wod_dict: dict) -> None:
        """Returns at least one feature (grouped by roadgraph ID)."""
        features = parse_map_features(raw_wod_dict)
        assert len(features) > 0

    def test_polyline_shape(self, raw_wod_dict: dict) -> None:
        """Each feature has polyline_points with shape (N, 2)."""
        features = parse_map_features(raw_wod_dict)
        for feature in features:
            assert feature.polyline_points.ndim == 2
            assert feature.polyline_points.shape[1] == 2

    def test_valid_feature_types(self, raw_wod_dict: dict) -> None:
        """All features have valid MapFeatureType values."""
        from diffav.core.types import MapFeatureType

        features = parse_map_features(raw_wod_dict)
        valid_types = set(MapFeatureType)
        for feature in features:
            assert feature.feature_type in valid_types

    def test_only_valid_points_included(self, raw_wod_dict: dict) -> None:
        """Invalid roadgraph points should be excluded."""
        # All points in fixture are valid, so all should be included
        features = parse_map_features(raw_wod_dict)
        total_points = sum(f.polyline_points.shape[0] for f in features)
        assert total_points == NUM_RG_POINTS
