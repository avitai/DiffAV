"""Cross-modal fusion of scene modality embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from artifex.generative_models.core.layers.transformers import TransformerEncoderBlock
from artifex.generative_models.modalities.multi_modal.representations import (
    CrossModalAttention,
)
from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
from flax import nnx
from jaxtyping import PyTree

from diffav.core.constants import SCENE_TOKEN_VALID
from diffav.core.types import FusionStrategy


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneFusionConfig(CrossModalOperatorConfig):
    """Configuration for cross-modal scene fusion.

    Attributes:
        embed_dim: Embedding dimension for all modalities.
        fusion_strategy: Fusion method ("cross_attention", "early", "additive").
        num_heads: Attention heads for fusion.
        dropout_rate: Dropout rate.
        validity_fields: Optional per-modality token-validity keys, parallel to
            ``input_fields``. When set, the operator also emits
            ``scene_token_valid`` — the per-token validity concatenated in
            token order — so a downstream consumer can mask padded tokens. A
            modality without a validity key is treated as fully valid.
    """

    embed_dim: int = 256
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    num_heads: int = 8
    dropout_rate: float = 0.0
    validity_fields: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate fusion strategy and the validity-field alignment."""
        # Explicit super(): @dataclass(slots=True) rebuilds the class, breaking
        # the zero-arg form's __class__ cell.
        super(SceneFusionConfig, self).__post_init__()
        FusionStrategy(self.fusion_strategy)
        if self.validity_fields is not None and len(self.validity_fields) != len(self.input_fields):
            msg = (
                f"validity_fields ({len(self.validity_fields)}) must be parallel to "
                f"input_fields ({len(self.input_fields)})"
            )
            raise ValueError(msg)


class SceneFusionOperator(CrossModalOperator):
    """Cross-modal fusion of scene modality embeddings.

    Supports three fusion strategies:
    - cross_attention: Each modality token attends to all others via
      CrossModalAttention. Most expressive (Wayformer, DriveTransformer).
    - early: Concatenate all tokens, run joint self-attention via
      TransformerEncoderBlock. Simple but effective (Wayformer ablation).
    - additive: Mean-pool across all tokens, broadcast back.
      Works well with strong features (MoST).

    Output: ``scene_embedding`` of shape ``[total_tokens, embed_dim]``.
    """

    config: SceneFusionConfig

    def __init__(
        self,
        config: SceneFusionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize fusion operator."""
        super().__init__(config, rngs=rngs, name=name)
        if rngs is None:
            rngs = nnx.Rngs(0)

        dim = config.embed_dim

        if config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.cross_attn = CrossModalAttention(
                query_dim=dim,
                key_dim=dim,
                value_dim=dim,
                num_heads=config.num_heads,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
        elif config.fusion_strategy == FusionStrategy.EARLY:
            self.self_attn = TransformerEncoderBlock(
                hidden_dim=dim,
                num_heads=config.num_heads,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )

        # Shared projection for all strategies
        self.output_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.output_norm = nnx.LayerNorm(dim, rngs=rngs)
        # Dropout is disabled by default; nnx ``train()``/``eval()`` toggle this
        # flag, so fusion-attention dropout activates only under ``train()`` and
        # only when ``dropout_rate > 0`` (additive fusion has no dropout).
        self.deterministic = True

    def _gather_tokens(self, data: dict) -> jnp.ndarray:
        """Gather modality embeddings into a single token sequence.

        Args:
            data: Dict with ``*_emb`` fields.

        Returns:
            Concatenated tokens ``[total_tokens, embed_dim]``.
        """
        tokens = []
        for field_name in self.config.input_fields:
            emb = jnp.array(data[field_name])
            if emb.ndim == 1:
                emb = emb[jnp.newaxis, :]  # [1, dim]
            tokens.append(emb)
        return jnp.concatenate(tokens, axis=0)

    def _gather_token_validity(self, data: dict, validity_fields: list[str]) -> jnp.ndarray:
        """Concatenate per-modality token validity in ``input_fields`` order.

        A modality whose validity field is absent from ``data`` is treated as
        fully valid (one boolean per token it contributes), so modalities like
        ego that never pad need not emit validity.

        Args:
            data: Dict with ``*_emb`` fields and optional ``*_token_valid``
                fields.
            validity_fields: Per-modality validity keys, parallel to
                ``config.input_fields``.

        Returns:
            Boolean validity ``[total_tokens]`` aligned with the fused tokens.
        """
        valids: list[jnp.ndarray] = []
        for field_name, valid_field in zip(self.config.input_fields, validity_fields, strict=True):
            emb = jnp.asarray(data[field_name])
            num_tokens = 1 if emb.ndim == 1 else emb.shape[0]
            if valid_field in data:
                valids.append(jnp.asarray(data[valid_field]).astype(bool).reshape(-1))
            else:
                valids.append(jnp.ones((num_tokens,), dtype=bool))
        return jnp.concatenate(valids)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Fuse modality embeddings into unified scene embedding.

        Reads input fields from data (e.g. agent_emb, map_emb, ego_emb),
        applies fusion strategy, and produces ``scene_embedding``.

        Args:
            data: Dict with modality embeddings.
            state: Passed through.
            metadata: Passed through.
            random_params: Unused.
            stats: Unused.

        Returns:
            Data dict with ``scene_embedding`` added.
        """
        result = dict(data)
        all_tokens = self._gather_tokens(data)  # [total, dim]

        strategy = self.config.fusion_strategy

        if strategy == FusionStrategy.CROSS_ATTENTION:
            q = all_tokens[jnp.newaxis, ...]
            fused = self.cross_attn(q, q, q, deterministic=self.deterministic)
            if fused.ndim == 3:
                fused = fused[0]
        elif strategy == FusionStrategy.EARLY:
            fused = self.self_attn(
                all_tokens[jnp.newaxis, ...],
                deterministic=self.deterministic,
            )
            fused = fused[0]
        else:
            # Additive: mean pool across token dimension, broadcast back
            fused = jnp.mean(all_tokens, axis=0, keepdims=True)
            fused = jnp.broadcast_to(fused, all_tokens.shape)

        scene_emb = self.output_norm(self.output_proj(fused))
        result = self._store_outputs(result, [scene_emb])
        if self.config.validity_fields is not None:
            result[SCENE_TOKEN_VALID] = self._gather_token_validity(
                data, self.config.validity_fields
            )
        return result, state, metadata
