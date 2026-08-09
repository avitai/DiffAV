"""Tests for road-edge extraction from raw WOD scenario dicts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from diffav.core.constants import (
    MAX_ROAD_EDGE_POINTS,
    MAX_ROAD_EDGE_POLYLINES,
    ROAD_EDGE_TYPES,
)
from diffav.core.geometry import RoadEdges, signed_distances_to_road_edges
from diffav.data import fixed_shape_road_edges_from_wod_dict, road_edges_from_wod_dict


def _roadgraph_dict(
    types: np.ndarray,
    ids: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble a minimal raw dict with the roadgraph sample keys."""
    return {
        "roadgraph_samples/type": types,
        "roadgraph_samples/id": ids,
        "roadgraph_samples/xyz": xyz,
        "roadgraph_samples/valid": valid,
    }


@pytest.fixture()
def typed_roadgraph_dict() -> dict[str, np.ndarray]:
    """Synthetic roadgraph with boundary, median, stop-sign, and short features.

    Column-vector ``(num_points, 1)`` layout for type/id/valid matches the
    real tf_example parsing output.

    - id 1: type 15 (ROAD_EDGE_BOUNDARY), three valid points.
    - id 2: type 16 (ROAD_EDGE_MEDIAN), three points, middle one invalid.
    - id 3: type 17 (STOP_SIGN) — must be ignored.
    - id 4: type 15 but only one valid point — dropped (< 2 points).
    """
    types = np.array([15, 15, 15, 16, 16, 16, 17, 17, 15], dtype=np.int64)
    ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4], dtype=np.int64)
    xyz = np.array(
        [
            [0.0, 0.0, 20.0],
            [1.0, 0.0, 20.5],
            [2.0, 0.0, 21.0],
            [0.0, 5.0, 20.0],
            [99.0, 99.0, 99.0],  # invalid — must be masked out
            [2.0, 5.0, 20.0],
            [10.0, 10.0, 20.0],
            [11.0, 10.0, 20.0],
            [4.0, 4.0, 20.0],
        ],
        dtype=np.float32,
    )
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int64)
    return _roadgraph_dict(types.reshape(-1, 1), ids.reshape(-1, 1), xyz, valid.reshape(-1, 1))


class TestRoadEdgeTypes:
    """The road-edge type codes match the official roadgraph mapping."""

    def test_boundary_and_median_codes(self) -> None:
        """Types 15 (ROAD_EDGE_BOUNDARY) and 16 (ROAD_EDGE_MEDIAN) only."""
        assert ROAD_EDGE_TYPES == (15, 16)


class TestRoadEdgesFromWodDict:
    """Extraction of RoadEdges polylines from raw scenario dicts."""

    def test_extracts_boundary_and_median_features(
        self, typed_roadgraph_dict: dict[str, np.ndarray]
    ) -> None:
        """Types 15 and 16 form polylines; 17 and short features do not."""
        edges = road_edges_from_wod_dict(typed_roadgraph_dict)
        assert isinstance(edges, RoadEdges)
        # id 1 (three points) and id 2 (two valid points): two polylines
        # padded to the longest length of three.
        assert edges.polylines.shape == (2, 3, 4)

    def test_masks_invalid_points(self, typed_roadgraph_dict: dict[str, np.ndarray]) -> None:
        """The invalid median point is excluded, not carried as geometry."""
        edges = road_edges_from_wod_dict(typed_roadgraph_dict)
        assert edges is not None
        median = np.asarray(edges.polylines[1])
        # Two valid points then a zero-validity pad row.
        np.testing.assert_allclose(median[0, :3], [0.0, 5.0, 20.0])
        np.testing.assert_allclose(median[1, :3], [2.0, 5.0, 20.0])
        assert median[2, 3] == 0.0
        assert not np.any(np.isclose(median[:, :3], 99.0).all(axis=-1))

    def test_preserves_elevation(self, typed_roadgraph_dict: dict[str, np.ndarray]) -> None:
        """Real z values survive into the packed polylines."""
        edges = road_edges_from_wod_dict(typed_roadgraph_dict)
        assert edges is not None
        boundary = np.asarray(edges.polylines[0])
        np.testing.assert_allclose(boundary[:3, 2], [20.0, 20.5, 21.0])

    def test_returns_none_without_road_edges(self) -> None:
        """A stop-sign-only roadgraph (type 17) yields no road edges."""
        raw = _roadgraph_dict(
            np.full((4, 1), 17, dtype=np.int64),
            np.arange(4, dtype=np.int64).reshape(-1, 1),
            np.zeros((4, 3), dtype=np.float32),
            np.ones((4, 1), dtype=np.int64),
        )
        assert road_edges_from_wod_dict(raw) is None

    def test_signed_distance_orientation(self) -> None:
        """A counterclockwise type-15 square keeps its interior on-road."""
        square = np.array(
            [
                [-10.0, -10.0, 0.0],
                [10.0, -10.0, 0.0],
                [10.0, 10.0, 0.0],
                [-10.0, 10.0, 0.0],
                [-10.0, -10.0, 0.0],
            ],
            dtype=np.float32,
        )
        n = square.shape[0]
        raw = _roadgraph_dict(
            np.full((n, 1), 15, dtype=np.int64),
            np.full((n, 1), 7, dtype=np.int64),
            square,
            np.ones((n, 1), dtype=np.int64),
        )
        edges = road_edges_from_wod_dict(raw)
        assert edges is not None
        inside = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
        outside = jnp.asarray([[15.0, 0.0]], dtype=jnp.float32)
        assert float(signed_distances_to_road_edges(inside, edges)[0]) < 0.0
        assert float(signed_distances_to_road_edges(outside, edges)[0]) > 0.0


class TestFixedShapeRoadEdges:
    """Global fixed-shape road edges for jit reuse across scenes."""

    def test_output_shape_is_fixed(self, typed_roadgraph_dict: dict[str, np.ndarray]) -> None:
        """The tensor is always (max_edges, max_points, 4) regardless of content."""
        edges = fixed_shape_road_edges_from_wod_dict(
            typed_roadgraph_dict, max_edges=8, max_points=6
        )
        assert edges.polylines.shape == (8, 6, 4)
        assert edges.is_cyclic.shape == (8,)

    def test_real_polylines_then_all_invalid_padding(
        self, typed_roadgraph_dict: dict[str, np.ndarray]
    ) -> None:
        """The two real edges fill the leading rows; the rest carry validity 0."""
        edges = fixed_shape_road_edges_from_wod_dict(
            typed_roadgraph_dict, max_edges=8, max_points=6
        )
        packed = np.asarray(edges.polylines)
        assert float(packed[0, :3, 3].sum()) == 3.0  # id 1: three valid points
        assert float(packed[1, :2, 3].sum()) == 2.0  # id 2: two valid points
        assert float(packed[2:, :, 3].sum()) == 0.0  # padding rows all invalid

    def test_edge_less_scene_is_all_invalid(self) -> None:
        """A scene with no road edge yields an all-invalid tensor, not None."""
        raw = _roadgraph_dict(
            np.full((4, 1), 17, dtype=np.int64),
            np.arange(4, dtype=np.int64).reshape(-1, 1),
            np.zeros((4, 3), dtype=np.float32),
            np.ones((4, 1), dtype=np.int64),
        )
        edges = fixed_shape_road_edges_from_wod_dict(raw, max_edges=4, max_points=5)
        assert edges.polylines.shape == (4, 5, 4)
        assert float(np.asarray(edges.polylines)[:, :, 3].sum()) == 0.0

    def test_caps_edges_and_points(self, typed_roadgraph_dict: dict[str, np.ndarray]) -> None:
        """Excess polylines and points beyond the caps are dropped."""
        edges = fixed_shape_road_edges_from_wod_dict(
            typed_roadgraph_dict, max_edges=1, max_points=2
        )
        packed = np.asarray(edges.polylines)
        assert packed.shape == (1, 2, 4)  # one polyline, two points
        assert float(packed[0, :, 3].sum()) == 2.0  # id 1 truncated to two points

    def test_matches_variable_geometry(self) -> None:
        """The fixed-shape edges reproduce the variable path's signed distance."""
        square = np.array(
            [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0], [-10, -10, 0]],
            dtype=np.float32,
        )
        n = square.shape[0]
        raw = _roadgraph_dict(
            np.full((n, 1), 15, dtype=np.int64),
            np.full((n, 1), 7, dtype=np.int64),
            square,
            np.ones((n, 1), dtype=np.int64),
        )
        variable = road_edges_from_wod_dict(raw)
        fixed = fixed_shape_road_edges_from_wod_dict(raw, max_edges=8, max_points=16)
        assert variable is not None
        points = jnp.asarray([[0.0, 0.0], [15.0, 0.0]], dtype=jnp.float32)
        assert jnp.allclose(
            signed_distances_to_road_edges(points, fixed),
            signed_distances_to_road_edges(points, variable),
            atol=1e-5,
        )

    def test_default_caps_are_positive(self) -> None:
        """The default caps cover the measured WOD road-edge distribution."""
        assert MAX_ROAD_EDGE_POLYLINES >= 117
        assert MAX_ROAD_EDGE_POINTS >= 608

    def test_chunk_size_leaves_distance_unchanged(
        self, typed_roadgraph_dict: dict[str, np.ndarray]
    ) -> None:
        """The chunk_size memory knob does not change the signed distance.

        Padding to a fixed shape adds all-invalid polylines, so a small chunk
        size genuinely scans several chunks; the result must match the
        single-pass distance.
        """
        edges = fixed_shape_road_edges_from_wod_dict(
            typed_roadgraph_dict, max_edges=8, max_points=6
        )
        points = jnp.asarray([[0.5, 0.5], [0.5, 4.5], [10.0, 10.0]], dtype=jnp.float32)
        assert jnp.allclose(
            signed_distances_to_road_edges(points, edges, chunk_size=2),
            signed_distances_to_road_edges(points, edges),
            atol=1e-6,
        )
