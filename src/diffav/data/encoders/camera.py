"""CNN-based multi-view camera image encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from diffav.core.constants import (
    CAMERA_EMBEDDING,
    CAMERA_IMAGES,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraEncoderConfig(OperatorConfig):
    """Configuration for camera image encoding.

    Attributes:
        embed_dim: Output embedding dimension.
        channels: Input channels (3 for RGB).
    """

    embed_dim: int = 256
    channels: int = 3


class CameraEncoder(OperatorModule):
    """Encode multi-view camera images into embeddings.

    Uses a 3-layer Conv -> ReLU -> stride-2 stack to downsample,
    followed by global average pooling and a linear projection.
    Each camera view is encoded independently via ``jax.vmap``.
    """

    def __init__(
        self,
        config: CameraEncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize camera encoder.

        Args:
            config: Encoder configuration.
            rngs: Flax NNX RNGs.
            name: Module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        hidden = config.embed_dim // 2
        self.conv1 = nnx.Conv(
            config.channels,
            hidden,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            hidden,
            hidden,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            hidden,
            config.embed_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.pool_proj = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)

    def _encode_single(self, img: jnp.ndarray) -> jnp.ndarray:
        """Encode a single camera image.

        Args:
            img: Image tensor ``[H, W, C]``.

        Returns:
            Embedding vector ``[embed_dim]``.
        """
        x = nnx.relu(self.conv1(img))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = jnp.mean(x, axis=(0, 1))  # Global average pool
        return self.pool_proj(x)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode camera images.

        Reads ``camera/images`` ``[num_cams, H, W, C]``, produces
        ``camera_emb`` ``[num_cams, embed_dim]``.

        Args:
            data: Dict with camera/images.
            state: Passed through.
            metadata: Passed through.
            key: Unused; the operator is deterministic.
            stats: Unused.

        Returns:
            Data dict with ``camera_emb`` added.
        """
        result = dict(data)
        images = jnp.array(data[CAMERA_IMAGES])  # [num_cams, H, W, C]

        cam_embs = jax.vmap(self._encode_single)(images)
        result[CAMERA_EMBEDDING] = cam_embs
        return result, state, metadata
