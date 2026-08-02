"""Shared fixtures for data module tests.

Provides mock WOD scenario data in the raw dict format produced by
tf.io.parse_example(), matching Waymax's feature description layout.
No real TFRecords or waymo-open-dataset package required.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.data.helpers import (
    NUM_OBJECTS,
    NUM_PADDING_OBJECTS,
    NUM_RG_POINTS,
    NUM_TIMESTEPS,
    OBJ_TYPE_CYCLIST,
    OBJ_TYPE_PEDESTRIAN,
    OBJ_TYPE_VEHICLE,
)


@pytest.fixture()
def raw_wod_dict() -> dict[str, np.ndarray]:
    """Create a mock WOD scenario dict matching Waymax's aggregated format.

    Keys follow the pattern 'state/all/<field>' for object trajectories
    and 'roadgraph_samples/<field>' for map data.
    """
    rng = np.random.default_rng(42)

    # Object trajectories: [num_objects, num_timesteps]
    base_x = np.linspace(0.0, 50.0, NUM_TIMESTEPS, dtype=np.float32)
    base_y = np.linspace(0.0, 10.0, NUM_TIMESTEPS, dtype=np.float32)

    xs = np.stack([base_x + i * 5.0 for i in range(NUM_OBJECTS)], axis=0)
    ys = np.stack([base_y + i * 3.0 for i in range(NUM_OBJECTS)], axis=0)

    velocity_x = np.full((NUM_OBJECTS, NUM_TIMESTEPS), 10.0, dtype=np.float32)
    velocity_y = np.full((NUM_OBJECTS, NUM_TIMESTEPS), 0.5, dtype=np.float32)
    heading = rng.uniform(-np.pi, np.pi, (NUM_OBJECTS, NUM_TIMESTEPS)).astype(np.float32)
    valid = np.ones((NUM_OBJECTS, NUM_TIMESTEPS), dtype=np.int64)
    # Make one agent invalid after timestep 50
    valid[3, 50:] = 0

    length = np.full((NUM_OBJECTS, NUM_TIMESTEPS), 4.5, dtype=np.float32)
    width = np.full((NUM_OBJECTS, NUM_TIMESTEPS), 2.0, dtype=np.float32)
    height = np.full((NUM_OBJECTS, NUM_TIMESTEPS), 1.5, dtype=np.float32)
    z = np.zeros((NUM_OBJECTS, NUM_TIMESTEPS), dtype=np.float32)

    timestamp_micros = np.tile(
        np.arange(NUM_TIMESTEPS, dtype=np.int64) * 100_000,
        (NUM_OBJECTS, 1),
    )

    # Object metadata: [num_objects]
    obj_ids = np.array([100, 201, 302, 403], dtype=np.float32)
    obj_types = np.array(
        [OBJ_TYPE_VEHICLE, OBJ_TYPE_VEHICLE, OBJ_TYPE_PEDESTRIAN, OBJ_TYPE_CYCLIST],
        dtype=np.float32,
    )
    is_sdc = np.array([1, 0, 0, 0], dtype=np.int64)
    tracks_to_predict = np.array([1, 1, 1, -1], dtype=np.int64)

    # Roadgraph: [num_rg_points, dim]
    rg_xyz = rng.uniform(-50, 50, (NUM_RG_POINTS, 3)).astype(np.float32)
    rg_dir = rng.standard_normal((NUM_RG_POINTS, 3)).astype(np.float32)
    rg_type = np.zeros((NUM_RG_POINTS, 1), dtype=np.int64)
    rg_type[:10] = 1  # lanes
    rg_type[10:15] = 18  # crosswalks
    rg_type[15:] = 17  # road edges
    rg_id = np.arange(NUM_RG_POINTS, dtype=np.int64).reshape(-1, 1)
    rg_valid = np.ones((NUM_RG_POINTS, 1), dtype=np.int64)

    return {
        # Object trajectories
        "state/all/x": xs,
        "state/all/y": ys,
        "state/all/z": z,
        "state/all/velocity_x": velocity_x,
        "state/all/velocity_y": velocity_y,
        "state/all/bbox_yaw": heading,
        "state/all/valid": valid,
        "state/all/length": length,
        "state/all/width": width,
        "state/all/height": height,
        "state/all/timestamp_micros": timestamp_micros,
        # Object metadata
        "state/id": obj_ids,
        "state/type": obj_types,
        "state/is_sdc": is_sdc,
        "state/tracks_to_predict": tracks_to_predict,
        # Roadgraph
        "roadgraph_samples/xyz": rg_xyz,
        "roadgraph_samples/dir": rg_dir,
        "roadgraph_samples/type": rg_type,
        "roadgraph_samples/id": rg_id,
        "roadgraph_samples/valid": rg_valid,
        # Scenario metadata
        "scenario/id": np.array([b"test_scenario_001"]),
    }


@pytest.fixture()
def padded_raw_wod_dict(raw_wod_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """WOD dict with trailing padding rows, as real TFRecords pad to max objects.

    Real WOD Motion records pad state arrays to a fixed object count with
    -1 sentinel values and ``state/all/valid`` set to 0 everywhere.
    """
    padded: dict[str, np.ndarray] = {}
    for key, value in raw_wod_dict.items():
        if not key.startswith("state/"):
            padded[key] = value
            continue
        pad = np.full((NUM_PADDING_OBJECTS, *value.shape[1:]), -1, dtype=value.dtype)
        if key == "state/all/valid":
            pad[:] = 0
        padded[key] = np.concatenate([value, pad], axis=0)
    return padded


@pytest.fixture()
def minimal_raw_wod_dict() -> dict[str, np.ndarray]:
    """Minimal WOD dict with 1 object and 91 timesteps for edge case testing."""
    return {
        "state/all/x": np.zeros((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/y": np.zeros((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/z": np.zeros((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/velocity_x": np.ones((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/velocity_y": np.zeros((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/bbox_yaw": np.zeros((1, NUM_TIMESTEPS), dtype=np.float32),
        "state/all/valid": np.ones((1, NUM_TIMESTEPS), dtype=np.int64),
        "state/all/length": np.full((1, NUM_TIMESTEPS), 4.0, dtype=np.float32),
        "state/all/width": np.full((1, NUM_TIMESTEPS), 2.0, dtype=np.float32),
        "state/all/height": np.full((1, NUM_TIMESTEPS), 1.5, dtype=np.float32),
        "state/all/timestamp_micros": np.tile(
            np.arange(NUM_TIMESTEPS, dtype=np.int64) * 100_000, (1, 1)
        ),
        "state/id": np.array([100.0], dtype=np.float32),
        "state/type": np.array([1.0], dtype=np.float32),
        "state/is_sdc": np.array([1], dtype=np.int64),
        "state/tracks_to_predict": np.array([1], dtype=np.int64),
        "roadgraph_samples/xyz": np.zeros((5, 3), dtype=np.float32),
        "roadgraph_samples/dir": np.zeros((5, 3), dtype=np.float32),
        "roadgraph_samples/type": np.ones((5, 1), dtype=np.int64),
        "roadgraph_samples/id": np.arange(5, dtype=np.int64).reshape(-1, 1),
        "roadgraph_samples/valid": np.ones((5, 1), dtype=np.int64),
        "scenario/id": np.array([b"minimal_scenario"]),
    }
