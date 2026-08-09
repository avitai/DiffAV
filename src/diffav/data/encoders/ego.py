"""MLP-based ego (SDC) state encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from diffav.core.constants import (
    EGO_EMBEDDING,
    STACKED_HISTORY,
    STATE_IS_SDC,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EgoEncoderConfig(OperatorConfig):
    """Configuration for ego state encoding.

    Attributes:
        embed_dim: Output embedding dimension.
        mlp_hidden: Expected input feature dimension (must match stacked_history width).
    """

    embed_dim: int = 256
    mlp_hidden: int = 128


class EgoEncoder(OperatorModule):
    """Encode ego vehicle state into an embedding.

    Extracts the SDC agent from stacked history and encodes through
    a 2-layer MLP to produce an embedding of shape ``[embed_dim]``.
    """

    def __init__(
        self,
        config: EgoEncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize ego encoder.

        Args:
            config: Encoder configuration.
            rngs: Flax NNX RNGs.
            name: Module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mlp1 = nnx.Linear(config.mlp_hidden, config.mlp_hidden, rngs=rngs)
        self.mlp2 = nnx.Linear(config.mlp_hidden, config.embed_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode ego state into embedding.

        Reads ``stacked_history`` and ``state/is_sdc``, produces
        ``ego_emb`` of shape ``[embed_dim]``.

        Args:
            data: Dict with stacked_history and state/is_sdc.
            state: Passed through.
            metadata: Passed through.
            random_params: Unused.
            stats: Unused.

        Returns:
            Data dict with ``ego_emb`` added.
        """
        result = dict(data)
        sdc_mask = jnp.array(data[STATE_IS_SDC])
        sdc_idx = jnp.argmax(sdc_mask)

        ego_history = jnp.array(data[STACKED_HISTORY])[sdc_idx]

        x = self.mlp1(ego_history)
        x = nnx.relu(x)
        x = self.mlp2(x)
        x = self.norm(x)

        result[EGO_EMBEDDING] = x
        return result, state, metadata
