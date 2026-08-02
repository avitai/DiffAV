"""Road-edge extraction from raw WOD scenario dicts.

Builds :class:`~simulacrax.core.geometry.RoadEdges` from the roadgraph
samples of a raw WOD Motion dict. Mirrors the official WOSAC map metric,
which computes offroad features against road-edge features only:
``ROAD_EDGE_BOUNDARY`` (type 15) and ``ROAD_EDGE_MEDIAN`` (type 16).

Two builders share the same grouping: :func:`road_edges_from_wod_dict`
returns variable-length polylines (the accurate, uncapped path used by the
offline WOSAC metric), while :func:`fixed_shape_road_edges_from_wod_dict`
returns a global fixed shape so a jitted, batched off-road term compiles once
and reuses the trace across scenes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from simulacrax.core.constants import (
    MAX_ROAD_EDGE_POINTS,
    MAX_ROAD_EDGE_POLYLINES,
    ROAD_EDGE_TYPES,
    ROADGRAPH_ID,
    ROADGRAPH_TYPE,
    ROADGRAPH_VALID,
    ROADGRAPH_XYZ,
)
from simulacrax.core.geometry import RoadEdges, stack_polylines_fixed


def _road_edge_polylines(raw: Mapping[str, Any]) -> list[np.ndarray]:
    """Group valid road-edge samples into per-feature polylines.

    Selects the valid roadgraph samples of the road-edge types (boundary 15
    and median 16), groups them by feature id preserving sample order and real
    elevation, and drops features with fewer than two valid points.

    Args:
        raw: Raw WOD scenario dict carrying the ``roadgraph_samples/*`` keys
            (type, id, xyz, valid), in either flat or column-vector layout.

    Returns:
        A list of ``(num_points_i, 3)`` polylines (possibly empty).
    """
    rg_type = np.asarray(raw[ROADGRAPH_TYPE]).reshape(-1)
    rg_valid = np.asarray(raw[ROADGRAPH_VALID]).reshape(-1)
    rg_id = np.asarray(raw[ROADGRAPH_ID]).reshape(-1)
    rg_xyz = np.asarray(raw[ROADGRAPH_XYZ]).reshape(-1, 3)

    is_edge_point = np.isin(rg_type, ROAD_EDGE_TYPES) & (rg_valid == 1)
    polylines = [
        rg_xyz[is_edge_point & (rg_id == feature_id)]
        for feature_id in np.unique(rg_id[is_edge_point])
    ]
    return [polyline for polyline in polylines if polyline.shape[0] >= 2]


def road_edges_from_wod_dict(raw: Mapping[str, Any]) -> RoadEdges | None:
    """Build variable-length road-edge polylines from a raw WOD scenario dict.

    Packs the road-edge features into the padded :class:`RoadEdges` tensor at
    the scene's own longest length — the accurate, uncapped path the offline
    WOSAC offroad metric consumes per scene.

    Args:
        raw: Raw WOD scenario dict carrying the ``roadgraph_samples/*`` keys.

    Returns:
        The packed road edges, or ``None`` when the scenario has no road-edge
        feature with at least two valid points.
    """
    polylines = _road_edge_polylines(raw)
    if not polylines:
        return None
    return RoadEdges.from_polylines(polylines)


def fixed_shape_road_edges_from_wod_dict(
    raw: Mapping[str, Any],
    *,
    max_edges: int = MAX_ROAD_EDGE_POLYLINES,
    max_points: int = MAX_ROAD_EDGE_POINTS,
) -> RoadEdges:
    """Build fixed-shape ``(max_edges, max_points, 4)`` road edges for jit reuse.

    Unlike :func:`road_edges_from_wod_dict`, the result always has the same
    shape, so a jitted, batched per-scene off-road term compiles once and
    reuses the trace across scenes. Excess polylines and points beyond the caps
    are dropped (the defaults cover the measured WOD distribution with margin),
    and an edge-less scene yields an all-invalid tensor rather than ``None``.
    Peak memory in the consumer is bounded by the ``chunk_size`` knob of
    :func:`~simulacrax.core.geometry.signed_distance_to_polylines`, not by these
    caps, so the exact full-resolution distance stays affordable.

    Args:
        raw: Raw WOD scenario dict carrying the ``roadgraph_samples/*`` keys.
        max_edges: Fixed number of polyline rows.
        max_points: Fixed number of points per polyline.

    Returns:
        Fixed-shape road edges (all-invalid when the scene has no road edge).
    """
    polylines = _road_edge_polylines(raw)
    stacked, cyclic = stack_polylines_fixed(
        polylines, max_polylines=max_edges, max_length=max_points
    )
    return RoadEdges(polylines=stacked, is_cyclic=cyclic)
