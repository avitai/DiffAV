"""VectorNet-style hierarchical map encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from artifex.generative_models.core.layers.egnn import EGNNLayer
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from diffav.core.constants import (
    MAP_EMBEDDING,
    MAP_TOKEN_VALID,
    ROADGRAPH_DIR,
    ROADGRAPH_ID,
    ROADGRAPH_TYPE,
    ROADGRAPH_VALID,
    ROADGRAPH_XYZ,
)


# Per-point input features: xyz (3) + direction (3) + type (1).
_POINT_FEATURE_DIM = 7


@dataclass(frozen=True, slots=True, kw_only=True)
class MapEncoderConfig(OperatorConfig):
    """Configuration for VectorNet-style map encoding.

    Attributes:
        embed_dim: Output embedding dimension.
        num_egnn_layers: Number of EGNN message-passing layers.
        polyline_mlp_hidden: Hidden dim for polyline subgraph MLP.
        max_polylines: Fixed number of polyline slots (for JIT).
        edge_radius: Max distance for EGNN edge construction (meters).
        dropout_rate: Dropout rate.
    """

    embed_dim: int = 256
    num_egnn_layers: int = 3
    polyline_mlp_hidden: int = 128
    max_polylines: int = 256
    edge_radius: float = 50.0
    dropout_rate: float = 0.0


class MapEncoder(OperatorModule):
    """VectorNet-style hierarchical map encoding.

    Stage 1 (subgraph): Groups roadgraph points by polyline ID, encodes
    each via MLP, and mean-pools per polyline. Raw WOD polyline IDs are
    arbitrary, so they are densely re-indexed by rank; pooling uses a
    one-hot matmul (JIT-compatible and deterministic on GPU). When a scene
    has more unique polylines than ``max_polylines``, the smallest IDs are
    kept.

    Stage 2 (global): Builds proximity graph over polyline centroids
    and runs EGNNLayer for equivariant message passing.

    Output: Per-polyline embeddings ``[max_polylines, embed_dim]``.
    """

    config: MapEncoderConfig

    def __init__(
        self,
        config: MapEncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize map encoder.

        Args:
            config: Encoder configuration.
            rngs: Flax NNX RNGs.
            name: Module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Stage 1: Polyline subgraph MLP over per-point features
        self.polyline_proj = nnx.Linear(
            _POINT_FEATURE_DIM,
            config.polyline_mlp_hidden,
            rngs=rngs,
        )
        self.polyline_out = nnx.Linear(
            config.polyline_mlp_hidden,
            config.embed_dim,
            rngs=rngs,
        )

        # Stage 2: EGNN global graph (nnx.List for proper parameter tracking)
        self.egnn_layers = nnx.List(
            [
                EGNNLayer(
                    edge_dim=1,
                    hidden_dim=config.embed_dim,
                    dropout_rate=config.dropout_rate,
                    rngs=rngs,
                )
                for _ in range(config.num_egnn_layers)
            ]
        )
        # Dropout is disabled by default; nnx ``train()``/``eval()`` toggle this
        # flag, so EGNN dropout activates only under ``train()`` and only when
        # ``dropout_rate > 0``.
        self.deterministic = True

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode map features via VectorNet-style hierarchy.

        Stage 1: MLP per point, dense polyline-ID re-indexing, one-hot
        matmul mean-pooling per polyline.
        Stage 2: EGNN over polyline embeddings with proximity edges.

        Args:
            data: Dict with roadgraph_samples/* arrays.
            state: Passed through.
            metadata: Passed through.
            random_params: Unused.
            stats: Unused.

        Returns:
            Data dict with ``map_emb`` added.
        """
        result = dict(data)
        max_poly = self.config.max_polylines

        xyz = jnp.array(data[ROADGRAPH_XYZ])  # [N, 3]
        dirs = jnp.array(data[ROADGRAPH_DIR])  # [N, 3]
        types = jnp.array(data[ROADGRAPH_TYPE])  # [N, 1]
        ids = jnp.array(data[ROADGRAPH_ID])  # [N, 1]
        valid = jnp.array(data[ROADGRAPH_VALID])  # [N, 1]

        # Stage 1: Per-point MLP
        point_feats = jnp.concatenate(
            [xyz, dirs, types.astype(jnp.float32)],
            axis=-1,
        )
        valid_f = valid.astype(jnp.float32)
        point_feats = point_feats * valid_f

        encoded = nnx.relu(self.polyline_proj(point_feats))
        encoded = self.polyline_out(encoded)  # [N, embed_dim]
        encoded = encoded * valid_f  # mask invalid

        # Dense re-index: raw WOD polyline IDs are arbitrary and sparse, so
        # rank them into [0, max_poly) slots. Invalid points get a sentinel
        # that sorts last; when unique IDs exceed capacity the smallest
        # max_poly IDs are kept (excess ranks fall outside the one-hot range).
        flat_ids = ids.flatten().astype(jnp.int32)  # [N]
        valid_flat = valid_f.flatten()  # [N]
        sentinel = jnp.iinfo(jnp.int32).max
        masked_ids = jnp.where(valid_flat > 0.0, flat_ids, sentinel)
        _, dense_ids = jnp.unique(
            masked_ids,
            size=max_poly,
            fill_value=sentinel,
            return_inverse=True,
        )

        # Mean-pool per polyline via one-hot matmul: deterministic on GPU,
        # unlike scatter-add-based segment pooling.
        one_hot = (dense_ids[:, jnp.newaxis] == jnp.arange(max_poly)[jnp.newaxis, :]).astype(
            jnp.float32
        ) * valid_flat[:, jnp.newaxis]  # [N, max_poly]
        poly_count = jnp.sum(one_hot, axis=0)[:, jnp.newaxis]  # [max_poly, 1]
        poly_sum = one_hot.T @ encoded  # [max_poly, embed_dim]
        poly_embs = poly_sum / jnp.maximum(poly_count, 1.0)

        # Centroids for proximity graph
        centroid_sum = one_hot.T @ xyz[:, :2]
        centroids = centroid_sum / jnp.maximum(poly_count, 1.0)  # [max_poly, 2]

        # Stage 2: EGNN with proximity edges
        # EGNN expects [batch, N, 3] coordinates and [batch, N, N] adjacency
        coords_3d = jnp.concatenate(
            [centroids, jnp.zeros((max_poly, 1))],
            axis=-1,
        )
        adj_matrix = self._build_adjacency(centroids)

        # Add batch dimension for EGNN: [1, N, D] and [1, N, N]
        node_feats = poly_embs[jnp.newaxis, ...]
        coords_3d = coords_3d[jnp.newaxis, ...]
        adj_matrix = adj_matrix[jnp.newaxis, ...]
        edge_features = adj_matrix[:, :, :, jnp.newaxis]  # [1, N, N, 1]

        for egnn_layer in self.egnn_layers:
            node_feats, coords_3d, _ = egnn_layer(
                node_feats,
                coords_3d,
                adj_matrix,
                edge_features,
                deterministic=self.deterministic,
            )

        result[MAP_EMBEDDING] = node_feats[0]  # Remove batch dim
        # Per-polyline token validity: a slot is real only if at least one
        # roadgraph point pooled into it (empty padding slots have count 0).
        result[MAP_TOKEN_VALID] = poly_count[:, 0] > 0
        return result, state, metadata

    def _build_adjacency(self, centroids: jnp.ndarray) -> jnp.ndarray:
        """Build proximity-based adjacency matrix for EGNN.

        Args:
            centroids: Polyline centroids ``[num_polylines, 2]``.

        Returns:
            Adjacency matrix ``[num_polylines, num_polylines]``.
        """
        n = centroids.shape[0]
        radius = self.config.edge_radius

        diff = centroids[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :]
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)

        adj = ((dists < radius) & (jnp.eye(n) == 0)).astype(jnp.float32)
        return adj
