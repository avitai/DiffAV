"""Transformer-based LiDAR point cloud encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from artifex.generative_models.core.layers.transformers import TransformerEncoderBlock
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from diffav.core.constants import (
    LIDAR_EMBEDDING,
    LIDAR_POINTS,
)


# LiDAR returns carry xyz coordinates only.
_POINT_COORDINATE_DIM = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class LiDAREncoderConfig(OperatorConfig):
    """Configuration for LiDAR point cloud encoding.

    Attributes:
        embed_dim: Output embedding dimension.
        num_points: Number of points to subsample to.
        num_layers: Transformer layers for point processing.
        num_heads: Attention heads.
        dropout_rate: Dropout rate.
    """

    embed_dim: int = 256
    num_points: int = 1024
    num_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.0


class LiDAREncoder(OperatorModule):
    """Encode LiDAR point clouds via transformer self-attention.

    Subsamples to a fixed count using linear indexing, projects point
    coordinates through an MLP, then processes with TransformerEncoderBlock
    layers to produce per-point embeddings.
    """

    config: LiDAREncoderConfig

    def __init__(
        self,
        config: LiDAREncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize LiDAR encoder.

        Args:
            config: Encoder configuration.
            rngs: Flax NNX RNGs.
            name: Module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Point projection: 3D coordinates -> embed_dim
        self.point_proj = nnx.Linear(_POINT_COORDINATE_DIM, config.embed_dim, rngs=rngs)
        self.point_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)

        # Transformer layers (nnx.List for proper parameter tracking)
        self.transformer_layers = nnx.List(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rate,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )
        # Dropout is disabled by default; nnx ``train()``/``eval()`` toggle this
        # flag, so transformer dropout activates only under ``train()`` and only
        # when ``dropout_rate > 0``.
        self.deterministic = True

        self.output_proj = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode LiDAR point cloud.

        Reads ``lidar/points``, subsamples, and produces ``lidar_emb``.

        Args:
            data: Dict with lidar/points ``[N, 3]``.
            state: Passed through.
            metadata: Passed through.
            random_params: Unused.
            stats: Unused.

        Returns:
            Data dict with ``lidar_emb`` added.
        """
        result = dict(data)
        points = jnp.array(data[LIDAR_POINTS])  # [N, 3]

        # Subsample to fixed count via linear index
        n = points.shape[0]
        target = self.config.num_points
        indices = jnp.linspace(0, n - 1, target).astype(jnp.int32)
        points = points[indices]

        # Project: [num_points, 3] -> [num_points, embed_dim]
        tokens = self.point_proj(points)
        tokens = self.point_norm(tokens)
        tokens = nnx.relu(tokens)

        # Transformer: add batch dim [1, num_points, embed_dim]
        tokens = tokens[jnp.newaxis, ...]
        for layer in self.transformer_layers:
            tokens = layer(tokens, deterministic=self.deterministic)
        tokens = tokens[0]  # Remove batch dim

        lidar_emb = self.output_proj(tokens)
        result[LIDAR_EMBEDDING] = lidar_emb
        return result, state, metadata
