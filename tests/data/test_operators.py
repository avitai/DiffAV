"""Tests for AV-specific preprocessing operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from simulacrax.data.operators import (
    AgentNormalizationConfig,
    AgentNormalizationOperator,
    MapCroppingConfig,
    MapCroppingOperator,
    TemporalStackingConfig,
    TemporalStackingOperator,
)
from tests import support


class TestAgentNormalizationOperator:
    """Tests for ego-centric coordinate normalization."""

    @pytest.fixture()
    def operator(self) -> AgentNormalizationOperator:
        """Create default normalization operator."""
        config = AgentNormalizationConfig()
        return AgentNormalizationOperator(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, np.ndarray]:
        """Scenario with ego at (10, 20) heading pi/4."""
        num_agents = 4
        num_steps = 11
        return {
            "state/all/x": np.array([[10.0] * num_steps] * num_agents),
            "state/all/y": np.array([[20.0] * num_steps] * num_agents),
            "state/all/bbox_yaw": np.array([[np.pi / 4] * num_steps] * num_agents),
            "state/all/velocity_x": np.array([[5.0] * num_steps] * num_agents),
            "state/all/velocity_y": np.array([[5.0] * num_steps] * num_agents),
            "state/all/valid": np.ones((num_agents, num_steps), dtype=np.int64),
            "state/is_sdc": np.array([1, 0, 0, 0]),
        }

    def test_ego_centered_at_origin(self, operator, sample_data) -> None:
        """Ego vehicle should be at origin after normalization."""
        result, _, _ = operator.apply(sample_data, {}, {})
        t = operator.config.current_step_idx
        sdc_idx = 0
        assert np.allclose(result["state/all/x"][sdc_idx, t], 0.0, atol=1e-5)
        assert np.allclose(result["state/all/y"][sdc_idx, t], 0.0, atol=1e-5)

    def test_ego_heading_zero(self, operator, sample_data) -> None:
        """Ego heading should be zero after normalization."""
        result, _, _ = operator.apply(sample_data, {}, {})
        t = operator.config.current_step_idx
        sdc_idx = 0
        assert np.allclose(result["state/all/bbox_yaw"][sdc_idx, t], 0.0, atol=1e-5)

    @pytest.fixture()
    def sample_data_with_map(self) -> dict[str, np.ndarray]:
        """Ego at (10,20) heading pi/4; agent 1 and a roadgraph point co-located."""
        steps = 11
        return {
            "state/all/x": np.array([[10.0] * steps, [15.0] * steps] + [[10.0] * steps] * 2),
            "state/all/y": np.array([[20.0] * steps, [25.0] * steps] + [[20.0] * steps] * 2),
            "state/all/bbox_yaw": np.array([[np.pi / 4] * steps] * 4),
            "state/all/velocity_x": np.zeros((4, steps)),
            "state/all/velocity_y": np.zeros((4, steps)),
            "state/all/valid": np.ones((4, steps), dtype=np.int64),
            "state/is_sdc": np.array([1, 0, 0, 0]),
            "roadgraph_samples/xyz": np.array([[15.0, 25.0, 3.0]]),
            "roadgraph_samples/dir": np.array([[1.0, 0.0, 0.0]]),
        }

    def test_roadgraph_shares_the_agent_frame(self, operator, sample_data_with_map) -> None:
        """A roadgraph point co-located with an agent maps to that agent's coords."""
        result, _, _ = operator.apply(sample_data_with_map, {}, {})
        t = operator.config.current_step_idx
        rg_xy = np.asarray(result["roadgraph_samples/xyz"])[0, :2]
        agent_xy = np.array(
            [
                np.asarray(result["state/all/x"])[1, t],
                np.asarray(result["state/all/y"])[1, t],
            ]
        )
        assert np.allclose(rg_xy, agent_xy, atol=1e-5)

    def test_roadgraph_z_preserved(self, operator, sample_data_with_map) -> None:
        """The roadgraph z channel is untouched by the planar transform."""
        result, _, _ = operator.apply(sample_data_with_map, {}, {})
        assert np.allclose(np.asarray(result["roadgraph_samples/xyz"])[0, 2], 3.0)

    def test_roadgraph_direction_rotated_not_translated(
        self, operator, sample_data_with_map
    ) -> None:
        """Direction vectors rotate by the ego heading and keep unit length."""
        result, _, _ = operator.apply(sample_data_with_map, {}, {})
        rg_dir = np.asarray(result["roadgraph_samples/dir"])[0, :2]
        assert np.allclose(np.linalg.norm(rg_dir), 1.0, atol=1e-5)
        assert not np.allclose(rg_dir, [1.0, 0.0])

    def test_normalization_is_idempotent(self, operator, sample_data_with_map) -> None:
        """Re-normalizing an already-normalized scene is a no-op (ego at origin)."""
        once, _, _ = operator.apply(sample_data_with_map, {}, {})
        twice, _, _ = operator.apply(once, {}, {})
        for key in ("state/all/x", "state/all/y", "roadgraph_samples/xyz"):
            assert np.allclose(np.asarray(once[key]), np.asarray(twice[key]), atol=1e-5)

    def test_preserves_relative_positions(self, operator) -> None:
        """Agents offset from ego maintain correct relative distance."""
        data = {
            "state/all/x": np.array([[0.0], [10.0]]),
            "state/all/y": np.array([[0.0], [0.0]]),
            "state/all/bbox_yaw": np.array([[0.0], [0.0]]),
            "state/all/velocity_x": np.array([[0.0], [0.0]]),
            "state/all/velocity_y": np.array([[0.0], [0.0]]),
            "state/all/valid": np.ones((2, 1), dtype=np.int64),
            "state/is_sdc": np.array([1, 0]),
        }
        config = AgentNormalizationConfig(current_step_idx=0)
        op = AgentNormalizationOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        dx = result["state/all/x"][1, 0] - result["state/all/x"][0, 0]
        dy = result["state/all/y"][1, 0] - result["state/all/y"][0, 0]
        assert np.allclose(np.sqrt(dx**2 + dy**2), 10.0, atol=1e-5)

    def test_original_data_unchanged(self, operator, sample_data) -> None:
        """Input data must not be mutated."""
        original_x = sample_data["state/all/x"].copy()
        operator.apply(sample_data, {}, {})
        assert np.array_equal(sample_data["state/all/x"], original_x)

    def test_velocity_rotation(self) -> None:
        """Velocities are rotated into ego heading frame."""
        data = {
            "state/all/x": np.array([[0.0]]),
            "state/all/y": np.array([[0.0]]),
            "state/all/bbox_yaw": np.array([[np.pi / 2]]),
            "state/all/velocity_x": np.array([[0.0]]),
            "state/all/velocity_y": np.array([[10.0]]),
            "state/all/valid": np.ones((1, 1), dtype=np.int64),
            "state/is_sdc": np.array([1]),
        }
        config = AgentNormalizationConfig(current_step_idx=0)
        op = AgentNormalizationOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        # After rotating by -pi/2: vy=10 should become vx=10, vy~=0
        assert np.allclose(result["state/all/velocity_x"][0, 0], 10.0, atol=1e-5)
        assert np.allclose(result["state/all/velocity_y"][0, 0], 0.0, atol=1e-5)

    def test_combined_translation_and_rotation(self) -> None:
        """Non-zero ego position + non-zero heading produces correct transform."""
        data = {
            "state/all/x": np.array([[5.0], [15.0]]),
            "state/all/y": np.array([[3.0], [3.0]]),
            "state/all/bbox_yaw": np.array([[np.pi / 2], [0.0]]),
            "state/all/velocity_x": np.array([[0.0], [1.0]]),
            "state/all/velocity_y": np.array([[0.0], [0.0]]),
            "state/all/valid": np.ones((2, 1), dtype=np.int64),
            "state/is_sdc": np.array([1, 0]),
        }
        config = AgentNormalizationConfig(current_step_idx=0)
        op = AgentNormalizationOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        # Ego at origin
        assert np.allclose(result["state/all/x"][0, 0], 0.0, atol=1e-5)
        assert np.allclose(result["state/all/y"][0, 0], 0.0, atol=1e-5)
        # Ego heading zero
        assert np.allclose(result["state/all/bbox_yaw"][0, 0], 0.0, atol=1e-5)
        # Agent 1 was 10m to the right of ego in world frame.
        # After rotating by -pi/2, distance should be preserved
        dx = result["state/all/x"][1, 0] - result["state/all/x"][0, 0]
        dy = result["state/all/y"][1, 0] - result["state/all/y"][0, 0]
        assert np.allclose(np.sqrt(dx**2 + dy**2), 10.0, atol=1e-5)

    def test_default_current_step_idx(self) -> None:
        """Default current_step_idx is 10."""
        config = AgentNormalizationConfig()
        assert config.current_step_idx == 10

    def test_batch_call(self) -> None:
        """Operator __call__ works with datarax Batch."""
        num_agents = 2
        num_steps = 11
        data = {
            "state/all/x": jnp.ones((num_agents, num_steps)),
            "state/all/y": jnp.ones((num_agents, num_steps)) * 2,
            "state/all/bbox_yaw": jnp.zeros((num_agents, num_steps)),
            "state/all/velocity_x": jnp.ones((num_agents, num_steps)),
            "state/all/velocity_y": jnp.zeros((num_agents, num_steps)),
            "state/all/valid": jnp.ones((num_agents, num_steps), dtype=jnp.int64),
            "state/is_sdc": jnp.array([1, 0]),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        config = AgentNormalizationConfig(current_step_idx=0)
        op = AgentNormalizationOperator(config, rngs=nnx.Rngs(0))
        result = op(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["state/all/x"].shape == (2, num_agents, num_steps)
        assert jnp.isfinite(result_data["state/all/x"]).all()

    def test_batch_call_distinct_outputs(self) -> None:
        """Batch call with different inputs produces different outputs per element."""
        data1 = {
            "state/all/x": jnp.array([[0.0], [10.0]]),
            "state/all/y": jnp.array([[0.0], [0.0]]),
            "state/all/bbox_yaw": jnp.array([[0.0], [0.0]]),
            "state/all/velocity_x": jnp.zeros((2, 1)),
            "state/all/velocity_y": jnp.zeros((2, 1)),
            "state/all/valid": jnp.ones((2, 1), dtype=jnp.int64),
            "state/is_sdc": jnp.array([1, 0]),
        }
        # Different ego heading changes rotation, producing different results
        data2 = {k: v for k, v in data1.items()}
        data2["state/all/bbox_yaw"] = jnp.array([[jnp.pi / 2], [0.0]])
        batch = Batch([Element(data=data1, state={}), Element(data=data2, state={})])
        config = AgentNormalizationConfig(current_step_idx=0)
        op = AgentNormalizationOperator(config, rngs=nnx.Rngs(0))
        result = op(batch)
        result_data = result.data.get_value()
        assert not jnp.allclose(result_data["state/all/x"][0], result_data["state/all/x"][1])


class TestMapCroppingOperator:
    """Tests for radius-based map cropping."""

    @pytest.fixture()
    def sample_data(self) -> dict[str, np.ndarray]:
        """Roadgraph with points at varying distances from origin."""
        n_points = 10
        xs = np.arange(n_points, dtype=np.float32) * 50.0
        return {
            "roadgraph_samples/xyz": np.stack(
                [xs, np.zeros(n_points), np.zeros(n_points)], axis=-1
            ),
            "roadgraph_samples/valid": np.ones((n_points, 1), dtype=np.int64),
            "roadgraph_samples/id": np.arange(n_points).reshape(-1, 1),
            "roadgraph_samples/type": np.ones((n_points, 1), dtype=np.int64),
            "roadgraph_samples/dir": np.zeros((n_points, 3), dtype=np.float32),
            "state/is_sdc": np.array([1]),
            "state/all/x": np.array([[0.0]]),
            "state/all/y": np.array([[0.0]]),
        }

    def test_crops_to_radius(self, sample_data) -> None:
        """Points beyond crop_radius are marked invalid."""
        config = MapCroppingConfig(crop_radius=150.0, current_step_idx=0)
        op = MapCroppingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(sample_data, {}, {})
        valid = result["roadgraph_samples/valid"].flatten()
        assert valid[0] == 1  # 0m
        assert valid[1] == 1  # 50m
        assert valid[2] == 1  # 100m
        assert valid[3] == 1  # 150m
        assert valid[4] == 0  # 200m

    def test_default_radius(self) -> None:
        """Default crop_radius is 150 meters."""
        config = MapCroppingConfig()
        assert config.crop_radius == 150.0

    def test_preserves_already_invalid_points(self, sample_data) -> None:
        """Points already invalid stay invalid even if within radius."""
        sample_data["roadgraph_samples/valid"][1, 0] = 0  # mark 50m point invalid
        config = MapCroppingConfig(crop_radius=150.0, current_step_idx=0)
        op = MapCroppingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(sample_data, {}, {})
        valid = result["roadgraph_samples/valid"].flatten()
        assert valid[1] == 0

    def test_original_data_unchanged(self, sample_data) -> None:
        """Input data must not be mutated."""
        original_valid = sample_data["roadgraph_samples/valid"].copy()
        config = MapCroppingConfig(crop_radius=50.0, current_step_idx=0)
        op = MapCroppingOperator(config, rngs=nnx.Rngs(0))
        op.apply(sample_data, {}, {})
        assert np.array_equal(sample_data["roadgraph_samples/valid"], original_valid)

    def test_batch_call(self) -> None:
        """Operator __call__ works with datarax Batch."""
        n_points = 10
        data = {
            "roadgraph_samples/xyz": jnp.zeros((n_points, 3)),
            "roadgraph_samples/valid": jnp.ones((n_points, 1), dtype=jnp.int64),
            "roadgraph_samples/id": jnp.arange(n_points).reshape(-1, 1),
            "roadgraph_samples/type": jnp.ones((n_points, 1), dtype=jnp.int64),
            "roadgraph_samples/dir": jnp.zeros((n_points, 3)),
            "state/is_sdc": jnp.array([1]),
            "state/all/x": jnp.array([[0.0]]),
            "state/all/y": jnp.array([[0.0]]),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        config = MapCroppingConfig(crop_radius=150.0, current_step_idx=0)
        op = MapCroppingOperator(config, rngs=nnx.Rngs(0))
        result = op(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["roadgraph_samples/valid"].shape == (2, n_points, 1)


class TestTemporalStackingOperator:
    """Tests for temporal feature stacking."""

    def test_stacks_history(self) -> None:
        """History steps are concatenated into stacked_history."""
        num_agents = 3
        total_steps = 91
        data = {
            "state/all/x": support.seeded_normal(num_agents, total_steps).astype(np.float32),
            "state/all/y": support.seeded_normal(num_agents, total_steps).astype(np.float32),
            "state/all/valid": np.ones((num_agents, total_steps), dtype=np.int64),
        }
        config = TemporalStackingConfig(
            history_steps=11,
            feature_fields=["state/all/x", "state/all/y"],
        )
        op = TemporalStackingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        assert result["stacked_history"].shape == (num_agents, 11 * 2)

    def test_preserves_future(self) -> None:
        """Future steps are extracted into separate fields."""
        num_agents = 2
        total_steps = 91
        data = {
            "state/all/x": np.arange(total_steps, dtype=np.float32)
            .reshape(1, -1)
            .repeat(num_agents, axis=0),
            "state/all/valid": np.ones((num_agents, total_steps), dtype=np.int64),
        }
        config = TemporalStackingConfig(
            history_steps=11,
            feature_fields=["state/all/x"],
        )
        op = TemporalStackingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        assert result["future_x"].shape == (num_agents, 80)

    def test_stacked_history_values_correct(self) -> None:
        """Stacked values match original history slice."""
        num_agents = 2
        total_steps = 20
        x_data = np.arange(total_steps, dtype=np.float32).reshape(1, -1).repeat(num_agents, axis=0)
        y_data = (
            (np.arange(total_steps, dtype=np.float32) * 10)
            .reshape(1, -1)
            .repeat(num_agents, axis=0)
        )
        data = {
            "state/all/x": x_data,
            "state/all/y": y_data,
        }
        config = TemporalStackingConfig(
            history_steps=5,
            feature_fields=["state/all/x", "state/all/y"],
        )
        op = TemporalStackingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        # stacked_history should be [x[:5], y[:5]] concatenated
        expected_agent0 = np.concatenate([x_data[0, :5], y_data[0, :5]])
        assert np.allclose(result["stacked_history"][0], expected_agent0)

    def test_future_values_correct(self) -> None:
        """Future field values match original future slice."""
        total_steps = 20
        x_data = np.arange(total_steps, dtype=np.float32).reshape(1, -1)
        data = {"state/all/x": x_data}
        config = TemporalStackingConfig(
            history_steps=5,
            feature_fields=["state/all/x"],
        )
        op = TemporalStackingOperator(config, rngs=nnx.Rngs(0))
        result, _, _ = op.apply(data, {}, {})
        expected_future = x_data[0, 5:]
        assert np.allclose(result["future_x"][0], expected_future)

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = TemporalStackingConfig()
        assert config.history_steps == 11
        assert config.feature_fields == ["state/all/x", "state/all/y"]

    def test_batch_call(self) -> None:
        """Operator __call__ works with datarax Batch."""
        num_agents = 3
        total_steps = 20
        data = {
            "state/all/x": jnp.ones((num_agents, total_steps)),
            "state/all/y": jnp.ones((num_agents, total_steps)) * 2,
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        config = TemporalStackingConfig(
            history_steps=5,
            feature_fields=["state/all/x", "state/all/y"],
        )
        op = TemporalStackingOperator(config, rngs=nnx.Rngs(0))
        result = op(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["stacked_history"].shape == (2, num_agents, 5 * 2)
        assert result_data["future_x"].shape == (2, num_agents, 15)
        assert result_data["future_y"].shape == (2, num_agents, 15)
