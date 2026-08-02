"""Tests for cross-modal scene fusion."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from simulacrax.core.types import FusionStrategy
from simulacrax.data.encoders import (
    SceneFusionConfig,
    SceneFusionOperator,
)


class TestSceneFusionOperator:
    """Tests for cross-modal scene fusion."""

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Multi-modality embeddings as fusion input."""
        return {
            "agent_emb": jnp.ones((8, 64)),
            "ego_emb": jnp.ones((64,)),
            "map_emb": jnp.ones((16, 64)),
        }

    def _make_operator(
        self,
        strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION,
    ) -> SceneFusionOperator:
        """Create fusion operator with given strategy."""
        config = SceneFusionConfig(
            input_fields=["agent_emb", "ego_emb", "map_emb"],
            output_fields=["scene_embedding"],
            embed_dim=64,
            fusion_strategy=strategy,
            num_heads=4,
        )
        return SceneFusionOperator(config, rngs=nnx.Rngs(0))

    def test_cross_attention_output_shape(self, sample_data) -> None:
        """Cross-attention produces scene_embedding with correct total tokens."""
        op = self._make_operator(FusionStrategy.CROSS_ATTENTION)
        result, _, _ = op.apply(sample_data, {}, {})
        assert "scene_embedding" in result
        # 8 agents + 1 ego + 16 map = 25 tokens
        assert result["scene_embedding"].shape == (25, 64)

    def test_early_fusion_output_shape(self, sample_data) -> None:
        """Early fusion produces scene_embedding with correct total tokens."""
        op = self._make_operator(FusionStrategy.EARLY)
        result, _, _ = op.apply(sample_data, {}, {})
        assert "scene_embedding" in result
        assert result["scene_embedding"].shape == (25, 64)

    def test_additive_fusion_output_shape(self, sample_data) -> None:
        """Additive fusion produces scene_embedding with correct total tokens."""
        op = self._make_operator(FusionStrategy.ADDITIVE)
        result, _, _ = op.apply(sample_data, {}, {})
        assert "scene_embedding" in result
        assert result["scene_embedding"].shape == (25, 64)

    def test_invalid_strategy_raises(self) -> None:
        """Invalid fusion strategy raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid FusionStrategy"):
            SceneFusionConfig(
                input_fields=["a"],
                output_fields=["b"],
                fusion_strategy=cast(Any, "invalid"),
            )

    def test_validity_fields_must_be_parallel_to_input_fields(self) -> None:
        """A validity_fields list shorter than input_fields raises."""
        with pytest.raises(ValueError, match="validity_fields"):
            SceneFusionConfig(
                input_fields=["agent_emb", "map_emb"],
                output_fields=["scene_embedding"],
                validity_fields=["agent_token_valid"],
            )

    def test_gradient_flows_cross_attention(self, sample_data) -> None:
        """Verify differentiability through cross-attention fusion."""
        op = self._make_operator(FusionStrategy.CROSS_ATTENTION)

        def loss_fn(model: SceneFusionOperator) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, {})
            return jnp.sum(result["scene_embedding"])

        grads = nnx.grad(loss_fn)(op)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_gradient_flows_early(self, sample_data) -> None:
        """Verify differentiability through early fusion."""
        op = self._make_operator(FusionStrategy.EARLY)

        def loss_fn(model: SceneFusionOperator) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, {})
            return jnp.sum(result["scene_embedding"])

        grads = nnx.grad(loss_fn)(op)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_jit_compatible(self, sample_data) -> None:
        """Fusion operator works under nnx.jit compilation."""
        op = self._make_operator(FusionStrategy.CROSS_ATTENTION)

        @nnx.jit
        def jit_apply(model: SceneFusionOperator) -> dict[str, Any]:
            result, _, _ = model.apply(sample_data, {}, {})
            return result

        result = jit_apply(op)
        assert result["scene_embedding"].shape == (25, 64)
        assert jnp.isfinite(result["scene_embedding"]).all()

    def test_deterministic_output(self, sample_data) -> None:
        """Same input produces identical output across calls."""
        op = self._make_operator(FusionStrategy.CROSS_ATTENTION)
        result1, _, _ = op.apply(sample_data, {}, {})
        result2, _, _ = op.apply(sample_data, {}, {})
        assert jnp.allclose(result1["scene_embedding"], result2["scene_embedding"])

    def test_state_metadata_passthrough(self, sample_data) -> None:
        """State and metadata are passed through unchanged."""
        op = self._make_operator(FusionStrategy.EARLY)
        state = {"counter": 42}
        metadata = {"source": "test"}
        _, out_state, out_meta = op.apply(sample_data, state, metadata)
        assert out_state == state
        assert out_meta == metadata

    def test_preserves_input_fields(self, sample_data) -> None:
        """Original modality embeddings are preserved in output."""
        op = self._make_operator(FusionStrategy.ADDITIVE)
        result, _, _ = op.apply(sample_data, {}, {})
        assert jnp.allclose(result["agent_emb"], sample_data["agent_emb"])
        assert jnp.allclose(result["ego_emb"], sample_data["ego_emb"])
        assert jnp.allclose(result["map_emb"], sample_data["map_emb"])

    def test_batch_call(self, sample_data) -> None:
        """Operator __call__ works with datarax Batch."""
        op = self._make_operator(FusionStrategy.CROSS_ATTENTION)
        batch = Batch([Element(data=sample_data, state={}) for _ in range(2)])
        result = op(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        # 8 agents + 1 ego + 16 map = 25 tokens
        assert result_data["scene_embedding"].shape == (2, 25, 64)
        assert jnp.isfinite(result_data["scene_embedding"]).all()

    def test_batch_call_all_strategies(self, sample_data) -> None:
        """Batch __call__ works for all fusion strategies."""
        for strategy in (
            FusionStrategy.CROSS_ATTENTION,
            FusionStrategy.EARLY,
            FusionStrategy.ADDITIVE,
        ):
            op = self._make_operator(strategy)
            batch = Batch([Element(data=sample_data, state={}) for _ in range(2)])
            result = op(batch)
            assert result.batch_size == 2
            result_data = result.data.get_value()
            assert result_data["scene_embedding"].shape == (2, 25, 64)
