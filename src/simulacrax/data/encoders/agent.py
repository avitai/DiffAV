"""Transformer-based agent state encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from artifex.generative_models.core.layers.transformers import TransformerEncoderBlock
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from simulacrax.core.constants import (
    AGENT_EMBEDDING,
    AGENT_TOKEN_VALID,
    STACKED_HISTORY,
    STATE_VALID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEncoderConfig(OperatorConfig):
    """Configuration for agent state encoding.

    Attributes:
        embed_dim: Output embedding dimension.
        num_heads: Attention heads for temporal transformer.
        mlp_hidden: Expected input feature dimension (must match stacked_history width).
        dropout_rate: Dropout rate.
    """

    embed_dim: int = 256
    num_heads: int = 8
    mlp_hidden: int = 128
    dropout_rate: float = 0.0


class AgentEncoder(OperatorModule):
    """Encode agent states into learned embeddings.

    Projects stacked history features through an MLP, applies
    self-attention via TransformerEncoderBlock, and masks invalid agents
    to produce per-agent embeddings of shape ``[num_agents, embed_dim]``.
    """

    def __init__(
        self,
        config: AgentEncoderConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize agent encoder.

        Args:
            config: Encoder configuration.
            rngs: Flax NNX RNGs.
            name: Module name.
        """
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.input_proj = nnx.Linear(config.mlp_hidden, config.embed_dim, rngs=rngs)
        self.input_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)

        self.temporal_attn = TransformerEncoderBlock(
            hidden_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        # Dropout is disabled by default; nnx ``train()``/``eval()`` toggle this
        # flag (see ``set_attributes``), so encoder dropout activates only under
        # ``train()`` and only when ``dropout_rate > 0``.
        self.deterministic = True

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Encode agent states into embeddings.

        Reads ``stacked_history`` and ``state/all/valid`` from data,
        produces ``agent_emb`` of shape ``[num_agents, embed_dim]``.

        Args:
            data: Dict with stacked_history ``[num_agents, feat_dim]``.
            state: Passed through.
            metadata: Passed through.
            random_params: Unused.
            stats: Unused.

        Returns:
            Data dict with ``agent_emb`` added.
        """
        result = dict(data)
        history = jnp.array(data[STACKED_HISTORY])
        valid = jnp.array(data[STATE_VALID])

        # Per-agent validity: agent valid if any timestep valid
        agent_valid = jnp.any(valid > 0, axis=-1).astype(jnp.float32)

        # Project: [num_agents, feat_dim] -> [num_agents, embed_dim]
        projected = self.input_proj(history)
        projected = self.input_norm(projected)
        projected = nnx.relu(projected)

        # Self-attention: add batch dim [1, num_agents, embed_dim]
        tokens = projected[jnp.newaxis, ...]
        encoded = self.temporal_attn(tokens, deterministic=self.deterministic)
        encoded = encoded[0]  # Remove batch dim

        # Mask invalid agents to zero
        result[AGENT_EMBEDDING] = encoded * agent_valid[:, jnp.newaxis]
        # Per-agent token validity, so fusion can exclude padded agents from
        # the backbone's map cross-attention.
        result[AGENT_TOKEN_VALID] = agent_valid > 0
        return result, state, metadata
