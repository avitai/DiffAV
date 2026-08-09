"""Tests for JAX ↔ WOD format conversion functions.

Verifies that predictions can be converted to WOD submission format
and that WOD scenario dicts can be round-tripped through our types.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffav.core.types import (
    AgentType,
    SceneContext,
    TrajectoryPrediction,
)
from diffav.data.converters import (
    from_wod_scenario,
    to_wod_submission,
)
from diffav.data.parsers import parse_scenario
from tests.data.helpers import FUTURE_STEPS


# ---------------------------------------------------------------------------
# to_wod_submission
# ---------------------------------------------------------------------------


class TestToWODSubmission:
    """Tests for to_wod_submission()."""

    def test_returns_dict(self) -> None:
        """Submission should be a dict with standard WOD fields."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((2, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100", "agent_201"),
        )
        submission = to_wod_submission(pred)
        assert isinstance(submission, dict)

    def test_has_required_fields(self) -> None:
        """Submission dict contains center_x, center_y, heading, object_id."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        submission = to_wod_submission(pred)
        assert "simulated_trajectories" in submission
        traj = submission["simulated_trajectories"][0]
        assert "center_x" in traj
        assert "center_y" in traj
        assert "heading" in traj
        assert "object_id" in traj

    def test_trajectory_length(self) -> None:
        """Each simulated trajectory should have future_steps entries."""
        pred = TrajectoryPrediction(
            trajectories=jnp.ones((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        submission = to_wod_submission(pred)
        traj = submission["simulated_trajectories"][0]
        assert len(traj["center_x"]) == FUTURE_STEPS
        assert len(traj["center_y"]) == FUTURE_STEPS
        assert len(traj["heading"]) == FUTURE_STEPS

    def test_agent_count_matches(self) -> None:
        """Number of simulated trajectories matches agent count."""
        n_agents = 3
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((n_agents, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=tuple(f"agent_{i}" for i in range(n_agents)),
        )
        submission = to_wod_submission(pred)
        assert len(submission["simulated_trajectories"]) == n_agents

    def test_values_propagated(self) -> None:
        """Trajectory values (x, y, heading, v) appear in submission."""
        traj_data = jnp.array([[[1.0, 2.0, 0.5, 10.0]]], dtype=jnp.float32)
        # Pad to full future_steps
        traj_data = jnp.broadcast_to(traj_data, (1, FUTURE_STEPS, 4))
        pred = TrajectoryPrediction(
            trajectories=traj_data,
            agent_ids=("agent_100",),
        )
        submission = to_wod_submission(pred)
        traj = submission["simulated_trajectories"][0]
        assert traj["center_x"][0] == pytest.approx(1.0)
        assert traj["center_y"][0] == pytest.approx(2.0)
        assert traj["heading"][0] == pytest.approx(0.5)

    def test_has_full_proto_fields(self) -> None:
        """Each trajectory carries every SimulatedTrajectory proto field."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        submission = to_wod_submission(pred)
        traj = submission["simulated_trajectories"][0]
        assert "center_z" in traj
        assert "valid" in traj
        assert len(traj["center_z"]) == FUTURE_STEPS
        assert len(traj["valid"]) == FUTURE_STEPS

    def test_center_z_defaults_to_zeros(self) -> None:
        """Without an explicit elevation, center_z is all zeros."""
        pred = TrajectoryPrediction(
            trajectories=jnp.ones((2, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100", "agent_201"),
        )
        submission = to_wod_submission(pred)
        for traj in submission["simulated_trajectories"]:
            assert traj["center_z"] == [0.0] * FUTURE_STEPS

    def test_validity_defaults_to_all_valid(self) -> None:
        """Without explicit validity, every step is valid per the proto contract."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        submission = to_wod_submission(pred)
        assert submission["simulated_trajectories"][0]["valid"] == [True] * FUTURE_STEPS

    def test_center_z_propagated(self) -> None:
        """An explicit center_z array lands in the submission."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        center_z = np.full((1, FUTURE_STEPS), 3.25, dtype=np.float32)
        submission = to_wod_submission(pred, center_z=center_z)
        traj = submission["simulated_trajectories"][0]
        assert traj["center_z"][0] == pytest.approx(3.25)
        assert traj["center_z"][-1] == pytest.approx(3.25)

    def test_validity_propagated(self) -> None:
        """An explicit validity mask lands in the submission as booleans."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((1, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100",),
        )
        validity = np.ones((1, FUTURE_STEPS), dtype=bool)
        validity[0, FUTURE_STEPS // 2 :] = False
        submission = to_wod_submission(pred, validity=validity)
        valid = submission["simulated_trajectories"][0]["valid"]
        assert valid[0] is True
        assert valid[-1] is False

    def test_center_z_shape_mismatch_raises(self) -> None:
        """A center_z array not matching (num_agents, future_steps) is rejected."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((2, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100", "agent_201"),
        )
        with pytest.raises(ValueError, match="center_z"):
            to_wod_submission(pred, center_z=np.zeros((1, FUTURE_STEPS)))

    def test_validity_shape_mismatch_raises(self) -> None:
        """A validity array not matching (num_agents, future_steps) is rejected."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((2, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100", "agent_201"),
        )
        with pytest.raises(ValueError, match="validity"):
            to_wod_submission(pred, validity=np.ones((2, FUTURE_STEPS - 1), dtype=bool))

    def test_agent_ids_as_object_ids(self) -> None:
        """Agent IDs should be stored as object_id in submission."""
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((2, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=("agent_100", "agent_201"),
        )
        submission = to_wod_submission(pred)
        ids = [t["object_id"] for t in submission["simulated_trajectories"]]
        assert ids == ["agent_100", "agent_201"]


# ---------------------------------------------------------------------------
# from_wod_scenario
# ---------------------------------------------------------------------------


class TestFromWODScenario:
    """Tests for from_wod_scenario()."""

    def test_returns_scene_context(self, raw_wod_dict: dict) -> None:
        """from_wod_scenario returns a SceneContext."""
        scene = from_wod_scenario(raw_wod_dict)
        assert isinstance(scene, SceneContext)

    def test_ego_is_vehicle(self, raw_wod_dict: dict) -> None:
        """Ego agent should be classified as VEHICLE."""
        scene = from_wod_scenario(raw_wod_dict)
        assert scene.ego_state.agent_type == AgentType.VEHICLE

    def test_matches_parse_scenario(self, raw_wod_dict: dict) -> None:
        """from_wod_scenario should produce same result as parse_scenario."""
        scene_a = from_wod_scenario(raw_wod_dict)
        scene_b = parse_scenario(raw_wod_dict)
        assert len(scene_a.agent_states) == len(scene_b.agent_states)
        assert float(scene_a.ego_state.position[0]) == pytest.approx(
            float(scene_b.ego_state.position[0]), abs=1e-5
        )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for round-trip conversion consistency."""

    def test_parse_then_submit(self, raw_wod_dict: dict) -> None:
        """Parse scene, create prediction, convert to submission."""
        scene = parse_scenario(raw_wod_dict)
        # Create fake predictions for non-ego agents
        n_agents = len(scene.agent_states)
        pred = TrajectoryPrediction(
            trajectories=jnp.zeros((n_agents, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=tuple(f"agent_{i}" for i in range(n_agents)),
        )
        submission = to_wod_submission(pred)
        assert len(submission["simulated_trajectories"]) == n_agents

    def test_submission_values_finite(self, raw_wod_dict: dict) -> None:
        """All values in submission should be finite."""
        scene = parse_scenario(raw_wod_dict)
        n_agents = len(scene.agent_states)
        pred = TrajectoryPrediction(
            trajectories=jnp.ones((n_agents, FUTURE_STEPS, 4), dtype=jnp.float32),
            agent_ids=tuple(f"agent_{i}" for i in range(n_agents)),
        )
        submission = to_wod_submission(pred)
        for traj in submission["simulated_trajectories"]:
            assert all(np.isfinite(traj["center_x"]))
            assert all(np.isfinite(traj["center_y"]))
            assert all(np.isfinite(traj["heading"]))
