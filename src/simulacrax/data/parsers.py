"""Parse raw WOD scenario dicts into typed domain objects.

Converts the flat dict format produced by tf.io.parse_example()
(following Waymax's feature layout) into SceneContext, AgentState,
and MapFeature instances. Real WOD records pad state arrays to a
fixed object count; only objects valid at the current timestep are
parsed, so padding rows never become agents.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from simulacrax.core.constants import (
    ROADGRAPH_ID,
    ROADGRAPH_TYPE,
    ROADGRAPH_VALID,
    ROADGRAPH_XYZ,
    STATE_BBOX_YAW,
    STATE_IS_SDC,
    STATE_TIMESTAMP_MICROS,
    STATE_TYPE,
    STATE_VALID,
    STATE_VELOCITY_X,
    STATE_VELOCITY_Y,
    STATE_X,
    STATE_Y,
    WOD_CURRENT_TIME_INDEX,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.types import (
    AgentState,
    AgentType,
    MapFeature,
    MapFeatureType,
    SceneContext,
)


# WOD object type enum → AgentType mapping
_WOD_TYPE_MAP: dict[int, AgentType] = {
    1: AgentType.VEHICLE,
    2: AgentType.PEDESTRIAN,
    3: AgentType.CYCLIST,
}

# WOD roadgraph type → MapFeatureType mapping
# See waymo_open_dataset/protos/map.proto for full enum
_WOD_RG_LANE_TYPES = frozenset({1, 2, 3})  # LaneCenter types
_WOD_RG_CROSSWALK_TYPES = frozenset({18})  # Crosswalk
_WOD_RG_SIGNAL_TYPES = frozenset({17, 19})  # StopSign, SpeedBump (approx.)


def _valid_object_indices(
    raw: dict[str, np.ndarray],
    current_time_index: int,
) -> np.ndarray:
    """Return indices of objects valid at the current timestep.

    Args:
        raw: Raw scenario dict with a 'state/all/valid' array.
        current_time_index: Index of the current timestep.

    Returns:
        1-D array of object indices whose validity flag is set.
    """
    valid = np.asarray(raw[STATE_VALID])[:, current_time_index]
    return np.where(valid == 1)[0]


def parse_scenario(
    raw: dict[str, np.ndarray],
    current_time_index: int = WOD_CURRENT_TIME_INDEX,
) -> SceneContext:
    """Parse a raw WOD dict into a SceneContext.

    Only objects valid at ``current_time_index`` become agents; padding
    rows (valid=0) are excluded.

    Args:
        raw: Raw scenario dict with keys like 'state/all/x'.
        current_time_index: Index of the current timestep (default 10).

    Returns:
        SceneContext with ego, agents, map features, and timestamps.

    Raises:
        ValueError: If no object is flagged as the SDC, or the SDC is
            not valid at the current timestep.
    """
    valid_indices = _valid_object_indices(raw, current_time_index)
    agents = parse_agent_tracks(raw, current_time_index=current_time_index)
    map_features = parse_map_features(raw)

    # Identify SDC (ego) by is_sdc flag; it must be a valid object.
    is_sdc = np.asarray(raw[STATE_IS_SDC])
    sdc_indices = np.where(is_sdc == 1)[0]
    if len(sdc_indices) == 0:
        msg = "Scenario has no object flagged as the SDC (state/is_sdc)."
        raise ValueError(msg)
    sdc_idx = int(sdc_indices[0])
    valid_positions = np.where(valid_indices == sdc_idx)[0]
    if len(valid_positions) == 0:
        msg = f"SDC (object {sdc_idx}) is not valid at current_time_index={current_time_index}."
        raise ValueError(msg)
    sdc_position = int(valid_positions[0])

    ego_state = agents[sdc_position]
    non_ego_agents = tuple(a for i, a in enumerate(agents) if i != sdc_position)

    # Timestamps from history steps (in seconds, from microseconds)
    timestamp_micros = raw[STATE_TIMESTAMP_MICROS][sdc_idx, :WOD_HISTORY_STEPS]
    timestamps = jnp.array(timestamp_micros / 1_000_000.0, dtype=jnp.float32)

    return SceneContext(
        ego_state=ego_state,
        agent_states=non_ego_agents,
        map_features=tuple(map_features),
        timestamps=timestamps,
    )


def parse_agent_tracks(
    raw: dict[str, np.ndarray],
    current_time_index: int = WOD_CURRENT_TIME_INDEX,
) -> list[AgentState]:
    """Parse per-agent states at the current timestep.

    Only objects with ``state/all/valid`` set at ``current_time_index``
    are parsed; padded rows never become agents.

    Args:
        raw: Raw scenario dict.
        current_time_index: Index of the current timestep.

    Returns:
        List of AgentState objects, one per valid object, in object order.
    """
    xs = raw[STATE_X]
    ys = raw[STATE_Y]
    vxs = raw[STATE_VELOCITY_X]
    vys = raw[STATE_VELOCITY_Y]
    headings = raw[STATE_BBOX_YAW]
    obj_types = raw[STATE_TYPE]

    t = current_time_index
    agents = []

    for i in _valid_object_indices(raw, t):
        x = float(xs[i, t])
        y = float(ys[i, t])
        vx = float(vxs[i, t])
        vy = float(vys[i, t])
        speed = float(np.sqrt(vx**2 + vy**2))
        heading = float(headings[i, t])
        obj_type_int = int(obj_types[i])
        agent_type = _WOD_TYPE_MAP.get(obj_type_int, AgentType.VEHICLE)

        agents.append(
            AgentState(
                position=jnp.array([x, y], dtype=jnp.float32),
                heading=heading,
                velocity=speed,
                acceleration=0.0,  # Not directly in WOD current-step data
                agent_type=agent_type,
            )
        )

    return agents


def parse_map_features(raw: dict[str, np.ndarray]) -> list[MapFeature]:
    """Parse roadgraph data into MapFeature objects.

    Groups roadgraph sample points by their feature ID and classifies
    them by type (lane, crosswalk, traffic signal).

    Args:
        raw: Raw scenario dict with roadgraph_samples keys.

    Returns:
        List of MapFeature objects with polyline coordinates.
    """
    rg_xyz = raw[ROADGRAPH_XYZ]  # [N, 3]
    rg_types = raw[ROADGRAPH_TYPE]  # [N, 1]
    rg_ids = raw[ROADGRAPH_ID]  # [N, 1]
    rg_valid = raw[ROADGRAPH_VALID]  # [N, 1]

    # Flatten single-column arrays
    rg_types_flat = rg_types.reshape(-1)
    rg_ids_flat = rg_ids.reshape(-1)
    rg_valid_flat = rg_valid.reshape(-1)

    # Filter to valid points
    valid_mask = rg_valid_flat == 1
    valid_xyz = rg_xyz[valid_mask]
    valid_types = rg_types_flat[valid_mask]
    valid_ids = rg_ids_flat[valid_mask]

    # Group points by feature ID
    unique_ids = np.unique(valid_ids)
    features = []

    for fid in unique_ids:
        mask = valid_ids == fid
        points_3d = valid_xyz[mask]
        points_2d = points_3d[:, :2]  # Use x, y only
        feature_type_val = int(valid_types[mask][0])

        feature_type = _classify_roadgraph_type(feature_type_val)

        features.append(
            MapFeature(
                polyline_points=jnp.array(points_2d, dtype=jnp.float32),
                feature_type=feature_type,
            )
        )

    return features


def _classify_roadgraph_type(wod_type: int) -> MapFeatureType:
    """Map WOD roadgraph type integer to MapFeatureType enum.

    Args:
        wod_type: WOD roadgraph type integer.

    Returns:
        Corresponding MapFeatureType.
    """
    if wod_type in _WOD_RG_LANE_TYPES:
        return MapFeatureType.LANE
    if wod_type in _WOD_RG_CROSSWALK_TYPES:
        return MapFeatureType.CROSSWALK
    if wod_type in _WOD_RG_SIGNAL_TYPES:
        return MapFeatureType.TRAFFIC_SIGNAL
    # Default: classify unknown types as lanes
    return MapFeatureType.LANE
