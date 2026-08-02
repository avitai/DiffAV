"""Tests for full-horizon WOD scenario preparation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from simulacrax.data import PAD_SENTINEL, prepare_full_horizon_scene, prepare_padded_scene


def _raw_scenario(
    *,
    num_agents: int = 6,
    total_steps: int = 21,
    valid_agents: int | None = None,
) -> dict[str, np.ndarray]:
    """Build a synthetic WOD-style state dict with controllable validity."""
    rng = np.random.default_rng(0)
    valid_agents = num_agents if valid_agents is None else valid_agents
    valid = np.zeros((num_agents, total_steps), dtype=np.int64)
    valid[:valid_agents] = 1
    return {
        "state/all/x": rng.normal(size=(num_agents, total_steps)) * 10.0,
        "state/all/y": rng.normal(size=(num_agents, total_steps)) * 5.0,
        "state/all/bbox_yaw": rng.normal(size=(num_agents, total_steps)),
        "state/all/velocity_x": rng.normal(size=(num_agents, total_steps)),
        "state/all/velocity_y": rng.normal(size=(num_agents, total_steps)),
        "state/all/valid": valid,
    }


class TestPrepareFullHorizonScene:
    """Validity-masked extraction of (trajectories, current states)."""

    def test_shapes_and_dtype(self) -> None:
        """Returns (A, T, 4) trajectories and (A, 4) current states."""
        raw = _raw_scenario()
        result = prepare_full_horizon_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, current = result
        assert trajectories.shape == (4, 10, 4)
        assert current.shape == (4, 4)
        assert trajectories.dtype == jnp.float32

    def test_returns_none_when_too_few_valid_agents(self) -> None:
        """Scenarios without enough full-horizon-valid agents are skipped."""
        raw = _raw_scenario(valid_agents=2)
        assert (
            prepare_full_horizon_scene(raw, num_agents=4, history_steps=11, future_steps=10) is None
        )

    def test_excludes_partial_horizon_agents(self) -> None:
        """An agent invalid at any future step is not selected."""
        raw = _raw_scenario(num_agents=5, valid_agents=5)
        raw["state/all/valid"][1, 15] = 0  # break agent 1 mid-horizon
        result = prepare_full_horizon_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, _ = result
        # Agent 1's x row must not appear among the selected trajectories.
        selected_x = np.asarray(trajectories[..., 0])
        broken_x = raw["state/all/x"][1, 11:21]
        assert not any(np.allclose(row, broken_x) for row in selected_x)

    def test_states_match_source_slices(self) -> None:
        """Trajectory channels are [x, y, heading, speed] from the future slice."""
        raw = _raw_scenario()
        result = prepare_full_horizon_scene(raw, num_agents=2, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, current = result
        np.testing.assert_allclose(
            np.asarray(trajectories[0, :, 0]), raw["state/all/x"][0, 11:21], rtol=1e-6
        )
        expected_speed = np.hypot(
            raw["state/all/velocity_x"][0, 11:21], raw["state/all/velocity_y"][0, 11:21]
        )
        np.testing.assert_allclose(np.asarray(trajectories[0, :, 3]), expected_speed, rtol=1e-6)
        # Current rows are [x, y, vx, vy] at the last history step.
        np.testing.assert_allclose(
            np.asarray(current[0]),
            [
                raw["state/all/x"][0, 10],
                raw["state/all/y"][0, 10],
                raw["state/all/velocity_x"][0, 10],
                raw["state/all/velocity_y"][0, 10],
            ],
            rtol=1e-6,
        )

    def test_rejects_nonpositive_agent_count(self) -> None:
        """num_agents must be positive."""
        with pytest.raises(ValueError, match="num_agents"):
            prepare_full_horizon_scene(
                _raw_scenario(), num_agents=0, history_steps=11, future_steps=10
            )


class TestPreparePaddedScene:
    """Padded, validity-masked extraction over all tracked agents."""

    def test_shapes_and_dtype(self) -> None:
        """Returns (A, T, 4), (A, T), (A, 3) with float32 dtype."""
        raw = _raw_scenario()
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, validity, reference_pose, _rows = result
        assert trajectories.shape == (4, 10, 4)
        assert validity.shape == (4, 10)
        assert reference_pose.shape == (4, 3)
        assert trajectories.dtype == jnp.float32
        assert validity.dtype == jnp.float32

    def test_reference_pose_is_current_step_pose(self) -> None:
        """Each slot's reference pose is its ``[x, y, heading]`` at the current step."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _, _, reference_pose, rows = result
        reference_pose = np.asarray(reference_pose)
        rows = np.asarray(rows)
        expected = np.stack(
            [
                raw["state/all/x"][rows, 10],
                raw["state/all/y"][rows, 10],
                raw["state/all/bbox_yaw"][rows, 10],
            ],
            axis=-1,
        )
        assert np.allclose(reference_pose, expected, atol=1e-5)

    def test_reference_pose_identity_for_padded_slots(self) -> None:
        """Padded slots (no source agent) keep the identity reference pose."""
        raw = _raw_scenario(num_agents=6, valid_agents=2)
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _, _, reference_pose, _ = result
        # Slots 2 and 3 are padding: their reference pose is [0, 0, 0].
        assert np.allclose(np.asarray(reference_pose)[2:], 0.0)

    def test_keeps_scenarios_below_full_agent_count(self) -> None:
        """Scenarios with fewer valid agents than slots are kept and padded."""
        raw = _raw_scenario(num_agents=6, valid_agents=2)
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _, validity, _, _ = result
        # Two agents fully valid, two padded slots fully invalid.
        per_agent_valid = np.asarray(validity).sum(axis=1)
        assert np.count_nonzero(per_agent_valid) == 2

    def test_returns_none_when_no_valid_agents(self) -> None:
        """A scenario with zero valid future steps returns None."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        raw["state/all/valid"][:, 11:21] = 0  # wipe the whole future horizon
        assert prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10) is None

    def test_excludes_agents_invalid_at_current_step(self) -> None:
        """A late-entry agent (invalid at the current step, valid in the future)
        is excluded: it has no defined reference pose, so its future cannot be
        expressed as a small local displacement and would otherwise be scored in
        the ego frame."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        # Agent 1 enters after the current step: invalid through step 10 (the
        # current step), valid across the whole future horizon.
        raw["state/all/valid"][1, :11] = 0
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _trajectories, validity, _reference_pose, rows = result
        # Only the three current-valid agents (0, 2, 3) carry a trajectory; the
        # late-entry agent 1 is dropped and its slot padded.
        assert np.count_nonzero(np.asarray(validity).sum(axis=1)) == 3
        assert 1 not in np.asarray(rows).tolist()

    def test_returns_none_when_no_agent_valid_at_current_step(self) -> None:
        """If every future-valid agent is absent at the current step, the scene is
        skipped rather than framed in the ego frame."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        raw["state/all/valid"][:, 10] = 0  # nobody is valid at the current step
        assert prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10) is None

    def test_mask_marks_partial_horizon_steps(self) -> None:
        """A step broken mid-horizon is masked 0 while other steps stay valid."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        raw["state/all/valid"][0, 15] = 0  # break one agent at future index 4
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, validity, _, _ = result
        validity_np = np.asarray(validity)
        # Exactly one (agent, step) entry is masked across all four agents.
        assert int((validity_np == 0.0).sum()) == 1
        broken_agent, broken_step = np.argwhere(validity_np == 0.0)[0]
        assert broken_step == 4
        # That masked step carries the sentinel, not real data.
        np.testing.assert_allclose(
            np.asarray(trajectories[broken_agent, broken_step]), PAD_SENTINEL
        )

    def test_sentinel_never_appears_in_valid_positions(self) -> None:
        """Valid positions carry real data, never the padding sentinel."""
        raw = _raw_scenario(num_agents=4, valid_agents=4)
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        trajectories, validity, _, _ = result
        traj_np = np.asarray(trajectories)
        valid_mask = np.asarray(validity) > 0
        # No valid (agent, step) x-channel equals the sentinel.
        assert not np.any(traj_np[valid_mask][:, 0] == PAD_SENTINEL)

    def test_ranks_agents_by_valid_step_count(self) -> None:
        """When slots are scarce, the most-tracked agents are selected."""
        raw = _raw_scenario(num_agents=5, valid_agents=5)
        # Agent 3 tracked for only 2 future steps; others fully tracked.
        raw["state/all/valid"][3, 13:21] = 0
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _, validity, _, _ = result
        # The 2-step agent must be excluded in favour of fully-tracked ones.
        per_agent_valid = sorted(np.asarray(validity).sum(axis=1).tolist(), reverse=True)
        assert per_agent_valid == [10.0, 10.0, 10.0, 10.0]

    def test_rejects_nonpositive_agent_count(self) -> None:
        """num_agents must be positive."""
        with pytest.raises(ValueError, match="num_agents"):
            prepare_padded_scene(_raw_scenario(), num_agents=0, history_steps=11, future_steps=10)

    def test_returns_ranked_agent_rows(self) -> None:
        """The fourth return value maps each output slot to its source agent."""
        raw = _raw_scenario(num_agents=5, valid_agents=5)
        raw["state/all/valid"][3, 13:21] = 0  # agent 3 tracked only 2 future steps
        result = prepare_padded_scene(raw, num_agents=4, history_steps=11, future_steps=10)
        assert result is not None
        _traj, _validity, _reference_pose, rows = result
        assert rows.shape == (4,)
        # The least-tracked agent (3) is dropped; the four full-horizon agents
        # fill the slots, so the rows index exactly those source agents.
        assert set(np.asarray(rows).tolist()) == {0, 1, 2, 4}
