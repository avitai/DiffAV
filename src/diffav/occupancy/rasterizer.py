"""Scene rasterization for occupancy flow prediction.

Converts SceneContext snapshots into top-down OccupancyGrid representations
using differentiable soft rasterization. All operations are pure JAX —
gradients flow through the grid construction for end-to-end training.

Geometry follows the waymo-open-dataset occupancy-flow reference:

- Grid scale: the official 256x256 grid at 3.2 px/m covers 80 m
  (protos/occupancy_flow_metrics.proto:64-76).
- SDC anchor: the ego maps to column ``sdc_x_in_grid=128`` and row
  ``sdc_y_in_grid=192`` of the 256 grid — 50% of the width, 75% of the
  height, i.e. 75% of the grid lies ahead of the ego and 25% behind
  (protos/occupancy_flow_metrics.proto:67-73).
- Image transform flips y: ``points_y = round(-y * ppm) + sdc_y``
  (utils/occupancy_flow_renderer.py:517-521). With the scene rotated so the
  SDC heads "up", ego-frame forward (+x) maps to decreasing rows and
  ego-frame left (+y) maps to decreasing columns.
- Agents render as oriented boxes: the reference samples 48x16 interior
  points of each yaw-rotated box (utils/occupancy_flow_renderer.py:698-790);
  here a smooth analytic box-interior indicator replaces point sampling so
  the render stays differentiable.
- Ground-truth flow is the backward displacement ``x[t-1] - x[t]`` in
  grid-cell units, rendered at the later timestep
  (utils/occupancy_flow_renderer.py:245-256) — see
  :meth:`SceneRasterizer.rasterize_backward_flow`.

Note (TODO): The soft rasterization utilities here are generic and should
eventually be contributed to a sister repo as a standalone differentiable
2D rasterization primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffav.core.types import (
    AgentState,
    AgentType,
    MapFeatureType,
    SceneContext,
)
from diffav.occupancy.flow_model import OccupancyGrid


@dataclass(frozen=True, slots=True, kw_only=True)
class RasterizerConfig:
    """Configuration for SceneRasterizer.

    Attributes:
        grid_resolution: Grid height and width in pixels.
        grid_size_m: Scene coverage in metres per side. The official grid
            covers 80 m (256 cells at 3.2 px/m,
            occupancy_flow_metrics.proto:64-76).
        sdc_x_fraction: Ego (SDC) column anchor as a fraction of the grid
            width. Official default 0.5 (sdc_x_in_grid=128 of 256,
            occupancy_flow_metrics.proto:73).
        sdc_y_fraction: Ego (SDC) row anchor as a fraction of the grid
            height. Official default 0.75 (sdc_y_in_grid=192 of 256,
            occupancy_flow_metrics.proto:71-72): 75% of the grid is ahead
            of the ego, 25% behind.
        default_agent_length_m: Box length used for every agent. AgentState
            carries no per-agent box dimensions, so a typical vehicle
            length stands in for all agents (the reference uses each
            agent's real ``length``, occupancy_flow_renderer.py:698-790).
        default_agent_width_m: Box width used for every agent (see
            ``default_agent_length_m`` for the shared-dimension caveat).
        box_edge_softness_m: Soft-edge width of the box-interior indicator
            in metres. Smaller values sharpen the box edges; the render
            stays differentiable for any positive value.
        sigma_map_m: Gaussian line width for map polyline points in metres.
    """

    grid_resolution: int = 128
    grid_size_m: float = 80.0
    sdc_x_fraction: float = 0.5
    sdc_y_fraction: float = 0.75
    default_agent_length_m: float = 4.7
    default_agent_width_m: float = 2.1
    box_edge_softness_m: float = 0.2
    sigma_map_m: float = 0.5

    def __post_init__(self) -> None:
        """Validate all physical parameters."""
        if self.grid_resolution <= 0:
            raise ValueError(f"grid_resolution must be positive, got {self.grid_resolution}")
        if self.grid_size_m <= 0.0:
            raise ValueError(f"grid_size_m must be positive, got {self.grid_size_m}")
        if not 0.0 < self.sdc_x_fraction < 1.0:
            raise ValueError(f"sdc_x_fraction must be in (0, 1), got {self.sdc_x_fraction}")
        if not 0.0 < self.sdc_y_fraction < 1.0:
            raise ValueError(f"sdc_y_fraction must be in (0, 1), got {self.sdc_y_fraction}")
        if self.default_agent_length_m <= 0.0:
            raise ValueError(
                f"default_agent_length_m must be positive, got {self.default_agent_length_m}"
            )
        if self.default_agent_width_m <= 0.0:
            raise ValueError(
                f"default_agent_width_m must be positive, got {self.default_agent_width_m}"
            )
        if self.box_edge_softness_m <= 0.0:
            raise ValueError(
                f"box_edge_softness_m must be positive, got {self.box_edge_softness_m}"
            )
        if self.sigma_map_m <= 0.0:
            raise ValueError(f"sigma_map_m must be positive, got {self.sigma_map_m}")

    @property
    def cell_size_m(self) -> float:
        """Grid cell size in metres (grid_size_m / grid_resolution)."""
        return self.grid_size_m / self.grid_resolution


# Channel index by agent type
_AGENT_TYPE_CHANNEL: dict[AgentType, int] = {
    AgentType.VEHICLE: 0,
    AgentType.PEDESTRIAN: 1,
    AgentType.CYCLIST: 2,
}

# Channel index by map feature type
_MAP_TYPE_CHANNEL: dict[MapFeatureType, int] = {
    MapFeatureType.LANE: 0,
    MapFeatureType.CROSSWALK: 1,
    MapFeatureType.TRAFFIC_SIGNAL: 2,
}


def _world_to_ego(pos_world: jax.Array, ego_pos: jax.Array, ego_heading: float) -> jax.Array:
    """Transform world position to ego-centric frame (+x forward, +y left).

    Args:
        pos_world: World position (2,).
        ego_pos: Ego vehicle world position (2,).
        ego_heading: Ego vehicle heading in radians.

    Returns:
        Position in ego frame (2,).
    """
    rel = pos_world - ego_pos
    cos_h = jnp.cos(-ego_heading)
    sin_h = jnp.sin(-ego_heading)
    x_ego = cos_h * rel[0] - sin_h * rel[1]
    y_ego = sin_h * rel[0] + cos_h * rel[1]
    return jnp.stack([x_ego, y_ego])


def _ego_to_pixel(pos_ego: jax.Array, config: RasterizerConfig) -> tuple[jax.Array, jax.Array]:
    """Convert ego-frame position to continuous (row, col) pixel coordinates.

    Mirrors the official image transform (occupancy_flow_renderer.py:517-521):
    ``points_x = round(x * ppm) + sdc_x`` and ``points_y = round(-y * ppm) +
    sdc_y`` with the scene rotated so the SDC heads "up". Composed with our
    ego frame (+x forward, +y left) that yields::

        row = sdc_row - x / cell_size   (forward maps "up")
        col = sdc_col - y / cell_size   (left maps to the image's left)

    Rounding is omitted to keep the transform differentiable.

    Args:
        pos_ego: Position in ego frame (2,).
        config: Rasterizer configuration providing scale and SDC anchor.

    Returns:
        (row, col) pixel coordinates as JAX scalars.
    """
    scale = config.grid_resolution / config.grid_size_m
    row = config.sdc_y_fraction * config.grid_resolution - pos_ego[0] * scale
    col = config.sdc_x_fraction * config.grid_resolution - pos_ego[1] * scale
    return row, col


def _gaussian_blob(
    grid_rows: jax.Array,
    grid_cols: jax.Array,
    row: jax.Array,
    col: jax.Array,
    sigma_pix: float,
) -> jax.Array:
    """Differentiable 2D Gaussian blob centred at (row, col).

    Args:
        grid_rows: Pixel row indices, shape (H, W).
        grid_cols: Pixel column indices, shape (H, W).
        row: Centre pixel row as JAX scalar.
        col: Centre pixel column as JAX scalar.
        sigma_pix: Gaussian std in pixels.

    Returns:
        Gaussian blob values, shape (H, W), max value 1.0 at centre.
    """
    d2 = (grid_rows - row) ** 2 + (grid_cols - col) ** 2
    return jnp.exp(-0.5 * d2 / (sigma_pix**2))


def _oriented_soft_box(
    pix_x_m: jax.Array,
    pix_y_m: jax.Array,
    center_ego: jax.Array,
    heading_ego: jax.Array,
    config: RasterizerConfig,
) -> jax.Array:
    """Differentiable oriented box-interior indicator for one agent.

    Smooth analogue of the reference box rasterization, which samples 48x16
    interior points of each yaw-rotated ``length x width`` box
    (occupancy_flow_renderer.py:698-790): pixel centres are transformed into
    the agent frame and a product of axis-aligned sigmoids approximates the
    box-interior indicator.

    Args:
        pix_x_m: Ego-frame x coordinate of each pixel centre, shape (H, W).
        pix_y_m: Ego-frame y coordinate of each pixel centre, shape (H, W).
        center_ego: Agent centre in the ego frame (2,).
        heading_ego: Agent heading relative to the ego frame (scalar).
        config: Rasterizer configuration providing box dimensions.

    Returns:
        Soft occupancy in [0, 1], shape (H, W), ~1 inside the box.
    """
    dx = pix_x_m - center_ego[0]
    dy = pix_y_m - center_ego[1]
    cos_h = jnp.cos(heading_ego)
    sin_h = jnp.sin(heading_ego)
    longitudinal = cos_h * dx + sin_h * dy
    lateral = -sin_h * dx + cos_h * dy
    softness = config.box_edge_softness_m
    inside_length = jax.nn.sigmoid(
        (config.default_agent_length_m / 2.0 - jnp.abs(longitudinal)) / softness
    )
    inside_width = jax.nn.sigmoid(
        (config.default_agent_width_m / 2.0 - jnp.abs(lateral)) / softness
    )
    return inside_length * inside_width


class SceneRasterizer:
    """Converts SceneContext to a top-down OccupancyGrid with official geometry.

    The grid is anchored on a reference ego state (75% of the grid ahead,
    25% behind by default) and oriented so the ego heads "up", matching the
    waymo-open-dataset occupancy-flow rendering (see module docstring).
    Agents render as differentiable oriented soft boxes; map points render
    as Gaussian blobs. All operations are differentiable via JAX.

    Note (TODO): This rasterizer should be factored out into a generic
    differentiable 2D rasterization utility and moved to a sister repo.
    """

    def __init__(self, config: RasterizerConfig) -> None:
        """Initialize the rasterizer.

        Args:
            config: Rasterizer configuration.
        """
        self.config = config
        H = config.grid_resolution
        indices = jnp.arange(H, dtype=jnp.float32)
        # grid_rows[i, j] = i, grid_cols[i, j] = j
        self._grid_rows, self._grid_cols = jnp.meshgrid(indices, indices, indexing="ij")
        # Ego-frame metre coordinates of each pixel centre (inverse of
        # _ego_to_pixel): x = (sdc_row - row) * cell, y = (sdc_col - col) * cell.
        cell = config.cell_size_m
        self._pix_x_m = (config.sdc_y_fraction * H - self._grid_rows) * cell
        self._pix_y_m = (config.sdc_x_fraction * H - self._grid_cols) * cell

    def rasterize(
        self, scene: SceneContext, *, frame_ego: AgentState | None = None
    ) -> OccupancyGrid:
        """Rasterize a scene snapshot to a top-down occupancy grid.

        Agent and map-point rendering are vectorized with ``jax.vmap`` over
        stacked per-agent (per-point) arrays; per-channel accumulation uses
        a one-hot matmul, so no Python loop touches the pixel grid.

        The returned ``flow`` channels are ego-frame agent velocities in m/s
        — model INPUT features only. They are NOT the occupancy-flow
        ground-truth flow, which is a backward displacement between
        waypoints; use :meth:`rasterize_backward_flow` for training targets.

        Args:
            scene: Single-timestep scene snapshot with ego, agents, and map
                features.
            frame_ego: Optional reference state that anchors the grid. The
                official pipeline renders every future waypoint w.r.t. the
                AV's location at the current time
                (occupancy_flow_metrics.proto:67-70) — pass the current ego
                state here when rasterizing future waypoints. Defaults to
                ``scene.ego_state``.

        Returns:
            OccupancyGrid with channels-last layout (H, W, C).
        """
        frame = frame_ego if frame_ego is not None else scene.ego_state
        ego_pos = frame.position
        ego_heading = frame.heading

        # Stack ego + surrounding agents into per-agent arrays
        all_agents: tuple[AgentState, ...] = (scene.ego_state,) + scene.agent_states
        positions = jnp.stack([jnp.asarray(a.position) for a in all_agents])  # (N, 2)
        headings = jnp.stack([jnp.asarray(a.heading) for a in all_agents])  # (N,)
        velocities = jnp.stack([jnp.asarray(a.velocity) for a in all_agents])  # (N,)
        channels = jnp.array([_AGENT_TYPE_CHANNEL.get(a.agent_type, 0) for a in all_agents])  # (N,)

        def render_agent(
            position: jax.Array, heading: jax.Array, velocity: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            pos_ego = _world_to_ego(position, ego_pos, ego_heading)
            occ = _oriented_soft_box(
                self._pix_x_m, self._pix_y_m, pos_ego, heading - ego_heading, self.config
            )
            # Ego-frame velocity components (input features, m/s)
            vx_world = velocity * jnp.cos(heading)
            vy_world = velocity * jnp.sin(heading)
            cos_neg_h = jnp.cos(-ego_heading)
            sin_neg_h = jnp.sin(-ego_heading)
            vx_ego = cos_neg_h * vx_world - sin_neg_h * vy_world
            vy_ego = sin_neg_h * vx_world + cos_neg_h * vy_world
            return occ, occ * vx_ego, occ * vy_ego

        occ_blobs, flow_x_blobs, flow_y_blobs = jax.vmap(render_agent)(
            positions, headings, velocities
        )  # each (N, H, H)

        agent_one_hot = jax.nn.one_hot(channels, 3)  # (N, 3)
        occupancy = jnp.einsum("nhw,nc->hwc", occ_blobs, agent_one_hot)  # (H, H, 3)
        flow = jnp.stack([flow_x_blobs.sum(axis=0), flow_y_blobs.sum(axis=0)], axis=-1)  # (H, H, 2)

        map_features = self._rasterize_map_points(scene, ego_pos, ego_heading)

        return OccupancyGrid(occupancy=occupancy, flow=flow, map_features=map_features)

    def rasterize_backward_flow(
        self,
        scene_now: SceneContext,
        scene_prev: SceneContext,
        *,
        frame_ego: AgentState | None = None,
    ) -> jax.Array:
        """Render official backward-displacement flow between two waypoints.

        Implements the occupancy-flow ground-truth flow definition
        (occupancy_flow_renderer.py:245-256): for each agent, the flow
        vector is its displacement from the earlier waypoint to the later
        one, expressed backward in image coordinates —
        ``(dx, dy) = (col_prev - col_now, row_prev - row_now)`` in
        GRID-CELL units — rendered at the agent's location at the LATER
        timestep and weighted by its soft occupancy there.

        Both scenes must contain the same agents in the same order
        (ego first, then ``agent_states`` index-aligned).

        Args:
            scene_now: Scene at waypoint t (the later timestep).
            scene_prev: Scene one waypoint earlier (t - 1 waypoint).
            frame_ego: Optional reference state anchoring the grid; the
                official pipeline uses the AV's state at the current time
                for every waypoint (occupancy_flow_metrics.proto:67-70).
                Defaults to ``scene_now.ego_state``.

        Returns:
            Backward flow field, shape (H, W, 2) with channels
            ``(dx_cols, dy_rows)`` in grid-cell units.

        Raises:
            ValueError: If the two scenes have different agent counts.
        """
        if len(scene_now.agent_states) != len(scene_prev.agent_states):
            raise ValueError(
                f"scene_now and scene_prev must contain the same agents in the same "
                f"order, got {len(scene_now.agent_states)} and "
                f"{len(scene_prev.agent_states)} surrounding agents."
            )
        frame = frame_ego if frame_ego is not None else scene_now.ego_state
        ego_pos = frame.position
        ego_heading = frame.heading

        agents_now: tuple[AgentState, ...] = (scene_now.ego_state,) + scene_now.agent_states
        agents_prev: tuple[AgentState, ...] = (scene_prev.ego_state,) + scene_prev.agent_states
        pos_now = jnp.stack([jnp.asarray(a.position) for a in agents_now])  # (N, 2)
        pos_prev = jnp.stack([jnp.asarray(a.position) for a in agents_prev])  # (N, 2)
        headings_now = jnp.stack([jnp.asarray(a.heading) for a in agents_now])  # (N,)

        def render_agent_flow(
            p_now: jax.Array, p_prev: jax.Array, heading_now: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            ego_now = _world_to_ego(p_now, ego_pos, ego_heading)
            ego_prev = _world_to_ego(p_prev, ego_pos, ego_heading)
            row_now, col_now = _ego_to_pixel(ego_now, self.config)
            row_prev, col_prev = _ego_to_pixel(ego_prev, self.config)
            # Backward displacement in grid-cell units, rendered at t
            # (occupancy_flow_renderer.py:245-256).
            occ_now = _oriented_soft_box(
                self._pix_x_m, self._pix_y_m, ego_now, heading_now - ego_heading, self.config
            )
            return occ_now * (col_prev - col_now), occ_now * (row_prev - row_now)

        dx_blobs, dy_blobs = jax.vmap(render_agent_flow)(pos_now, pos_prev, headings_now)
        return jnp.stack([dx_blobs.sum(axis=0), dy_blobs.sum(axis=0)], axis=-1)  # (H, H, 2)

    def _rasterize_map_points(
        self,
        scene: SceneContext,
        ego_pos: jax.Array,
        ego_heading: float,
    ) -> jax.Array:
        """Render all map polyline points as Gaussian blobs, one-hot pooled by channel.

        Args:
            scene: Scene whose map features to rasterize.
            ego_pos: Reference (frame) ego world position (2,).
            ego_heading: Reference (frame) ego heading in radians.

        Returns:
            Map channel grid, shape (H, H, 3).
        """
        H = self.config.grid_resolution
        if not scene.map_features:
            return jnp.zeros((H, H, 3))

        points = jnp.concatenate(
            [jnp.asarray(f.polyline_points) for f in scene.map_features]
        )  # (M, 2)
        point_channels = jnp.array(
            [
                _MAP_TYPE_CHANNEL.get(f.feature_type, 0)
                for f in scene.map_features
                for _ in range(f.polyline_points.shape[0])
            ]
        )  # (M,)

        sigma_pix = self.config.sigma_map_m / self.config.cell_size_m

        def render_point(point: jax.Array) -> jax.Array:
            pos_ego = _world_to_ego(point, ego_pos, ego_heading)
            row, col = _ego_to_pixel(pos_ego, self.config)
            return _gaussian_blob(self._grid_rows, self._grid_cols, row, col, sigma_pix)

        blobs = jax.vmap(render_point)(points)  # (M, H, H)
        map_one_hot = jax.nn.one_hot(point_channels, 3)  # (M, 3)
        return jnp.einsum("mhw,mc->hwc", blobs, map_one_hot)  # (H, H, 3)
