"""Tests for SceneRasterizer: official WOSAC geometry, oriented boxes, backward flow.

Geometry expectations are grounded in the waymo-open-dataset reference:
- Grid scale: 256 cells at 3.2 px/m covers 80 m
  (protos/occupancy_flow_metrics.proto:64-76).
- SDC anchor: sdc_x_in_grid=128, sdc_y_in_grid=192 of a 256 grid, i.e. the
  column anchor sits at 50% of the width and the row anchor at 75% of the
  height — 75% of the grid is ahead of the ego, 25% behind
  (protos/occupancy_flow_metrics.proto:67-73).
- Image transform flips y: points_y = round(-y * ppm) + sdc_y
  (utils/occupancy_flow_renderer.py:517-521), so forward maps to smaller
  rows ("up") and left maps to smaller columns.
- Agents render as oriented boxes (the reference samples 48x16 interior
  points of each yaw-rotated box, utils/occupancy_flow_renderer.py:698-790).
- Flow is the backward displacement (x[t-1] - x[t]) in grid-cell units,
  rendered at the later timestep (utils/occupancy_flow_renderer.py:245-256).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from simulacrax.core.types import (
    AgentState,
    AgentType,
    MapFeature,
    MapFeatureType,
    SceneContext,
)
from simulacrax.occupancy.rasterizer import RasterizerConfig, SceneRasterizer


# Default config geometry: 128 px over 80 m -> 0.625 m cells,
# SDC anchor at row 0.75*128 = 96, col 0.5*128 = 64.
_RES = 128
_CELL = 80.0 / 128
_ANCHOR_ROW = 96
_ANCHOR_COL = 64


def _agent(
    position: jax.Array,
    heading: float = 0.0,
    velocity: float = 0.0,
    agent_type: AgentType = AgentType.VEHICLE,
) -> AgentState:
    """Build a minimal AgentState."""
    return AgentState(
        position=position,
        heading=heading,
        velocity=velocity,
        acceleration=0.0,
        agent_type=agent_type,
    )


def _make_scene(
    agent_pos: jax.Array = jnp.zeros(2),
    agent_heading: float = 0.0,
    agent_velocity: float = 0.0,
    agent_type: AgentType = AgentType.VEHICLE,
    num_extra_agents: int = 0,
) -> SceneContext:
    """Create a minimal SceneContext for testing."""
    ego = _agent(agent_pos, agent_heading, agent_velocity, agent_type)
    extra = tuple(
        _agent(
            jnp.array([float(i) * 5.0, 0.0]),
            velocity=1.0,
            agent_type=AgentType.PEDESTRIAN,
        )
        for i in range(num_extra_agents)
    )
    return SceneContext(
        ego_state=ego,
        agent_states=extra,
        map_features=(),
        timestamps=jnp.array([0.0]),
    )


def _peak_row_col(channel: jax.Array) -> tuple[int, int]:
    """Return (row, col) of the argmax of a (H, W) channel."""
    idx = int(jnp.argmax(channel))
    width = channel.shape[1]
    return idx // width, idx % width


class TestRasterizerConfig:
    def test_official_grid_defaults(self) -> None:
        """Defaults follow occupancy_flow_metrics.proto:64-76 (80 m, 3:1 anchor)."""
        config = RasterizerConfig()
        assert config.grid_resolution == 128
        assert config.grid_size_m == 80.0
        assert config.sdc_x_fraction == 0.5
        assert config.sdc_y_fraction == 0.75
        assert config.default_agent_length_m == 4.7
        assert config.default_agent_width_m == 2.1
        assert config.box_edge_softness_m > 0.0
        assert config.sigma_map_m > 0.0

    def test_cell_size_derived(self) -> None:
        config = RasterizerConfig(grid_resolution=128, grid_size_m=80.0)
        assert abs(config.cell_size_m - 80.0 / 128) < 1e-9


class TestRasterizerConfigValidation:
    """RasterizerConfig validates its physical parameters."""

    def test_non_positive_resolution_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_resolution"):
            RasterizerConfig(grid_resolution=0)

    def test_non_positive_grid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="grid_size_m"):
            RasterizerConfig(grid_size_m=0.0)

    def test_sdc_fraction_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="sdc_x_fraction"):
            RasterizerConfig(sdc_x_fraction=0.0)
        with pytest.raises(ValueError, match="sdc_y_fraction"):
            RasterizerConfig(sdc_y_fraction=1.0)

    def test_non_positive_agent_dims_raise(self) -> None:
        with pytest.raises(ValueError, match="default_agent_length_m"):
            RasterizerConfig(default_agent_length_m=0.0)
        with pytest.raises(ValueError, match="default_agent_width_m"):
            RasterizerConfig(default_agent_width_m=-1.0)

    def test_non_positive_softness_raises(self) -> None:
        with pytest.raises(ValueError, match="box_edge_softness_m"):
            RasterizerConfig(box_edge_softness_m=0.0)

    def test_non_positive_sigma_map_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_map_m"):
            RasterizerConfig(sigma_map_m=0.0)


class TestOfficialGeometry:
    """Pixel transform matches occupancy_flow_renderer.py:517-521."""

    def setup_method(self) -> None:
        self.rasterizer = SceneRasterizer(RasterizerConfig())

    def test_ego_renders_at_sdc_anchor(self) -> None:
        """Ego at origin peaks at (0.75*H, 0.5*W), not the grid centre."""
        grid = self.rasterizer.rasterize(_make_scene())
        row, col = _peak_row_col(grid.occupancy[:, :, 0])
        assert abs(row - _ANCHOR_ROW) <= 1
        assert abs(col - _ANCHOR_COL) <= 1

    def test_agent_ahead_maps_to_smaller_row(self) -> None:
        """+x (forward) is 'up' in the image: rows decrease ahead of the ego."""
        scene = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(_agent(jnp.array([20.0, 0.0]), agent_type=AgentType.PEDESTRIAN),),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        grid = self.rasterizer.rasterize(scene)
        row, col = _peak_row_col(grid.occupancy[:, :, 1])
        assert abs(row - (_ANCHOR_ROW - round(20.0 / _CELL))) <= 1
        assert abs(col - _ANCHOR_COL) <= 1

    def test_agent_left_maps_to_smaller_col(self) -> None:
        """y-flip: +y (ego left) maps to smaller columns (renderer.py:520-521)."""
        scene = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(_agent(jnp.array([0.0, 10.0]), agent_type=AgentType.PEDESTRIAN),),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        grid = self.rasterizer.rasterize(scene)
        row, col = _peak_row_col(grid.occupancy[:, :, 1])
        assert abs(col - (_ANCHOR_COL - round(10.0 / _CELL))) <= 1
        assert abs(row - _ANCHOR_ROW) <= 1

    def test_map_point_ahead_renders_above_anchor(self) -> None:
        """Map points use the same anchored, y-flipped transform as agents."""
        lane = MapFeature(
            polyline_points=jnp.array([[5.0, 0.0]]),
            feature_type=MapFeatureType.LANE,
        )
        scene = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(),
            map_features=(lane,),
            timestamps=jnp.array([0.0]),
        )
        grid = self.rasterizer.rasterize(scene)
        row, col = _peak_row_col(grid.map_features[:, :, 0])
        assert abs(row - (_ANCHOR_ROW - round(5.0 / _CELL))) <= 1
        assert abs(col - _ANCHOR_COL) <= 1

    def test_frame_ego_override_recentres_grid(self) -> None:
        """rasterize(frame_ego=...) anchors the grid on the given state.

        The reference renders all waypoints w.r.t. the AV's location at the
        current time (occupancy_flow_metrics.proto:67-70).
        """
        agent = _agent(jnp.array([10.0, 0.0]), agent_type=AgentType.PEDESTRIAN)
        scene = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(agent,),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        grid = self.rasterizer.rasterize(scene, frame_ego=agent)
        row, col = _peak_row_col(grid.occupancy[:, :, 1])
        assert abs(row - _ANCHOR_ROW) <= 1
        assert abs(col - _ANCHOR_COL) <= 1


class TestOrientedBox:
    """Agents render as oriented soft boxes (renderer.py:698-790 analogue)."""

    def setup_method(self) -> None:
        self.rasterizer = SceneRasterizer(RasterizerConfig())

    def _extent_cells(self, channel: jax.Array) -> tuple[int, int]:
        """(row_extent, col_extent) of cells above 0.5."""
        mask = channel > 0.5
        rows = jnp.any(mask, axis=1)
        cols = jnp.any(mask, axis=0)
        return int(jnp.sum(rows)), int(jnp.sum(cols))

    def test_box_elongated_along_heading_zero(self) -> None:
        """Heading 0 (forward): length spans rows, width spans columns."""
        grid = self.rasterizer.rasterize(_make_scene(agent_heading=0.0))
        row_extent, col_extent = self._extent_cells(grid.occupancy[:, :, 0])
        assert row_extent > col_extent
        assert abs(row_extent - 4.7 / _CELL) <= 2
        assert abs(col_extent - 2.1 / _CELL) <= 2

    def test_box_rotates_with_heading(self) -> None:
        """A 90 degree heading swaps the row/column extents."""
        # Ego heading stays 0 so the grid frame is fixed; rotate a second agent.
        scene = SceneContext(
            ego_state=_agent(jnp.array([30.0, 0.0])),
            agent_states=(_agent(jnp.zeros(2), heading=jnp.pi / 2, agent_type=AgentType.CYCLIST),),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        # frame_ego at origin keeps the rotated agent at the anchor
        grid = self.rasterizer.rasterize(scene, frame_ego=_agent(jnp.zeros(2)))
        row_extent, col_extent = self._extent_cells(grid.occupancy[:, :, 2])
        assert col_extent > row_extent

    def test_box_area_matches_default_dims(self) -> None:
        """Soft-box mass approximates the 4.7 m x 2.1 m default footprint."""
        grid = self.rasterizer.rasterize(_make_scene())
        area_cells = float(jnp.sum(grid.occupancy[:, :, 0]))
        expected = (4.7 * 2.1) / (_CELL * _CELL)
        assert abs(area_cells - expected) / expected < 0.3

    def test_box_peak_close_to_one(self) -> None:
        grid = self.rasterizer.rasterize(_make_scene())
        assert 0.9 < float(jnp.max(grid.occupancy[:, :, 0])) <= 1.001

    def test_vehicle_in_channel_0_only(self) -> None:
        grid = self.rasterizer.rasterize(_make_scene(agent_type=AgentType.VEHICLE))
        assert float(jnp.max(grid.occupancy[:, :, 0])) > 0.1
        assert float(jnp.max(grid.occupancy[:, :, 1])) < 1e-6
        assert float(jnp.max(grid.occupancy[:, :, 2])) < 1e-6

    def test_pedestrian_in_channel_1(self) -> None:
        grid = self.rasterizer.rasterize(_make_scene(agent_type=AgentType.PEDESTRIAN))
        assert float(jnp.max(grid.occupancy[:, :, 1])) > 0.1

    def test_differentiability(self) -> None:
        """jax.grad w.r.t. agent position produces finite, non-zero gradients."""

        def total_occupancy(pos: jax.Array) -> jax.Array:
            # Differentiate a surrounding agent, not the ego: the grid is
            # anchored on the ego, so the ego's own render is translation
            # invariant by construction.
            scene = SceneContext(
                ego_state=_agent(jnp.zeros(2)),
                agent_states=(_agent(pos, agent_type=AgentType.PEDESTRIAN),),
                map_features=(),
                timestamps=jnp.array([0.0]),
            )
            grid = self.rasterizer.rasterize(scene)
            # Weight by row index so translation changes the objective.
            weights = jnp.arange(_RES, dtype=jnp.float32)[:, None]
            return jnp.sum(grid.occupancy[:, :, 1] * weights)

        grad = jax.grad(total_occupancy)(jnp.array([5.0, 0.0]))
        assert grad.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.abs(grad).max()) > 0.0


class TestSceneRasterizer:
    def test_output_shapes(self) -> None:
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))
        grid = rasterizer.rasterize(_make_scene())
        assert grid.occupancy.shape == (32, 32, 3)
        assert grid.flow.shape == (32, 32, 2)
        assert grid.map_features.shape == (32, 32, 3)

    def test_flow_input_feature_from_velocity(self) -> None:
        """rasterize() flow channels are ego-frame velocity INPUT features (m/s).

        This is distinct from the backward-displacement ground-truth flow
        (see TestBackwardFlow); an agent moving forward has positive vx here.
        """
        rasterizer = SceneRasterizer(RasterizerConfig())
        scene = _make_scene(agent_heading=0.0, agent_velocity=5.0)
        grid = rasterizer.rasterize(scene)
        assert float(grid.flow[_ANCHOR_ROW, _ANCHOR_COL, 0]) > 0.0

    def test_map_feature_lane_rendered(self) -> None:
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))
        lane = MapFeature(
            polyline_points=jnp.array([[-5.0, 0.0], [5.0, 0.0]]),
            feature_type=MapFeatureType.LANE,
        )
        scene = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(),
            map_features=(lane,),
            timestamps=jnp.array([0.0]),
        )
        grid = rasterizer.rasterize(scene)
        assert float(jnp.max(grid.map_features[:, :, 0])) > 0.1


class TestBackwardFlow:
    """rasterize_backward_flow follows occupancy_flow_renderer.py:245-256."""

    def setup_method(self) -> None:
        self.rasterizer = SceneRasterizer(RasterizerConfig())

    def _scene_pair(
        self, pos_now: jax.Array, pos_prev: jax.Array
    ) -> tuple[SceneContext, SceneContext]:
        """Static ego at origin plus one vehicle moving pos_prev -> pos_now."""

        def scene(pos: jax.Array) -> SceneContext:
            return SceneContext(
                ego_state=_agent(jnp.zeros(2)),
                agent_states=(_agent(pos),),
                map_features=(),
                timestamps=jnp.array([0.0]),
            )

        return scene(pos_now), scene(pos_prev)

    def test_output_shape(self) -> None:
        now, prev = self._scene_pair(jnp.array([10.0, 0.0]), jnp.array([8.0, 0.0]))
        flow = self.rasterizer.rasterize_backward_flow(now, prev)
        assert flow.shape == (_RES, _RES, 2)

    def test_static_scene_has_zero_flow(self) -> None:
        now, prev = self._scene_pair(jnp.array([10.0, 0.0]), jnp.array([10.0, 0.0]))
        flow = self.rasterizer.rasterize_backward_flow(now, prev)
        assert float(jnp.abs(flow).max()) < 1e-5

    def test_forward_motion_gives_positive_dy_in_cells(self) -> None:
        """Forward 2 m: dy = row_prev - row_now = +2/cell at the agent's t pose."""
        now, prev = self._scene_pair(jnp.array([10.0, 0.0]), jnp.array([8.0, 0.0]))
        flow = self.rasterizer.rasterize_backward_flow(now, prev)
        row_now = _ANCHOR_ROW - round(10.0 / _CELL)  # = 80
        dy = float(flow[row_now, _ANCHOR_COL, 1])
        dx = float(flow[row_now, _ANCHOR_COL, 0])
        assert abs(dy - 2.0 / _CELL) < 0.2
        assert abs(dx) < 1e-3

    def test_leftward_motion_gives_positive_dx_in_cells(self) -> None:
        """Left 2 m: dx = col_prev - col_now = +2/cell (columns flip with y)."""
        now, prev = self._scene_pair(jnp.array([10.0, 5.0]), jnp.array([10.0, 3.0]))
        flow = self.rasterizer.rasterize_backward_flow(now, prev)
        row_now = _ANCHOR_ROW - round(10.0 / _CELL)
        col_now = _ANCHOR_COL - round(5.0 / _CELL)
        assert abs(float(flow[row_now, col_now, 0]) - 2.0 / _CELL) < 0.2

    def test_flow_rendered_at_later_timestep(self) -> None:
        """Flow lives where the agent IS at t, not where it was at t-1
        (renderer.py:245-256 renders at the later timestep)."""
        now, prev = self._scene_pair(jnp.array([10.0, 0.0]), jnp.array([5.0, 0.0]))
        flow = self.rasterizer.rasterize_backward_flow(now, prev)
        row_prev = _ANCHOR_ROW - round(5.0 / _CELL)  # = 88, outside the t box
        assert float(jnp.abs(flow[row_prev, _ANCHOR_COL]).max()) < 0.05

    def test_agent_count_mismatch_raises(self) -> None:
        now, _ = self._scene_pair(jnp.array([10.0, 0.0]), jnp.array([8.0, 0.0]))
        prev = SceneContext(
            ego_state=_agent(jnp.zeros(2)),
            agent_states=(),
            map_features=(),
            timestamps=jnp.array([0.0]),
        )
        with pytest.raises(ValueError, match="agent"):
            self.rasterizer.rasterize_backward_flow(now, prev)

    def test_differentiable_w_r_t_current_position(self) -> None:
        def flow_mass(pos_now: jax.Array) -> jax.Array:
            now, prev = self._scene_pair(pos_now, jnp.array([8.0, 0.0]))
            return jnp.sum(self.rasterizer.rasterize_backward_flow(now, prev) ** 2)

        grad = jax.grad(flow_mass)(jnp.array([10.0, 0.0]))
        assert grad.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.abs(grad).max()) > 0.0

    def test_jit_compatible(self) -> None:
        def flow_sum(pos_now: jax.Array) -> jax.Array:
            now, prev = self._scene_pair(pos_now, jnp.array([8.0, 0.0]))
            return jnp.sum(self.rasterizer.rasterize_backward_flow(now, prev))

        pos = jnp.array([10.0, 0.0])
        assert jnp.allclose(flow_sum(pos), jax.jit(flow_sum)(pos), rtol=1e-6)


class TestRasterizerVectorized:
    """Vectorized rasterization: linearity, jit, vmap, determinism."""

    def test_many_agents_linear_superposition(self) -> None:
        """N agents rasterize to the sum of their individual grids."""
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))
        positions = [jnp.array([-8.0, -4.0]), jnp.array([0.0, 6.0]), jnp.array([7.0, 2.0])]
        combined = rasterizer.rasterize(_make_scene_with_pedestrians(positions))
        individual = [
            rasterizer.rasterize(_make_scene_with_pedestrians([pos])) for pos in positions
        ]
        # Ego (vehicle, channel 0) appears in every scene: the pedestrian
        # channel isolates the superposition property.
        expected = sum(g.occupancy[:, :, 1] for g in individual)
        assert jnp.allclose(combined.occupancy[:, :, 1], expected, atol=1e-5)

    def test_jit_compatible(self) -> None:
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))

        def occupancy_sum(pos: jax.Array) -> jax.Array:
            return jnp.sum(rasterizer.rasterize(_make_scene(agent_pos=pos)).occupancy)

        pos = jnp.array([2.0, -3.0])
        assert jnp.allclose(occupancy_sum(pos), jax.jit(occupancy_sum)(pos), rtol=1e-6)

    def test_vmap_over_positions(self) -> None:
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))

        def occupancy_sum(pos: jax.Array) -> jax.Array:
            return jnp.sum(rasterizer.rasterize(_make_scene(agent_pos=pos)).occupancy)

        batch = jnp.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 3.0]])
        results = jax.vmap(occupancy_sum)(batch)
        assert results.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(results)))

    def test_deterministic(self) -> None:
        rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=32))
        scene = _make_scene(num_extra_agents=3)
        first = rasterizer.rasterize(scene)
        second = rasterizer.rasterize(scene)
        assert jnp.array_equal(first.occupancy, second.occupancy)
        assert jnp.array_equal(first.flow, second.flow)
        assert jnp.array_equal(first.map_features, second.map_features)


def _make_scene_with_pedestrians(positions: list[jax.Array]) -> SceneContext:
    """Scene with a stationary ego and pedestrians at the given positions."""
    pedestrians = tuple(
        _agent(pos, velocity=1.0, agent_type=AgentType.PEDESTRIAN) for pos in positions
    )
    return SceneContext(
        ego_state=_agent(jnp.zeros(2)),
        agent_states=pedestrians,
        map_features=(),
        timestamps=jnp.array([0.0]),
    )
