"""Tests for the ego state encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from diffav.data.encoders import (
    EgoEncoder,
    EgoEncoderConfig,
)
from tests import support


class TestEgoEncoder:
    """Tests for ego state encoding."""

    @pytest.fixture()
    def encoder(self) -> EgoEncoder:
        """Create default ego encoder."""
        config = EgoEncoderConfig(embed_dim=64, mlp_hidden=128)
        return EgoEncoder(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Sample data with 4 agents, first is SDC."""
        return {
            "stacked_history": support.seeded_normal(4, 128).astype(np.float32),
            "state/is_sdc": np.array([1, 0, 0, 0]),
            "state/all/valid": np.ones((4, 11), dtype=np.int64),
        }

    def test_output_shape(self, encoder, sample_data) -> None:
        """Ego embedding is a 1D vector of embed_dim."""
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "ego_emb" in result
        assert result["ego_emb"].shape == (64,)

    def test_gradient_flows(self, encoder, sample_data) -> None:
        """Verify differentiability through EgoEncoder."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        def loss_fn(model: EgoEncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["ego_emb"])

        grads = nnx.grad(loss_fn)(encoder)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_jit_compatible(self, encoder, sample_data) -> None:
        """Encoder works under nnx.jit compilation."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def jit_apply(model: EgoEncoder) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        result = jit_apply(encoder)
        assert result["ego_emb"].shape == (64,)
        assert jnp.isfinite(result["ego_emb"]).all()

    def test_deterministic_output(self, encoder, sample_data) -> None:
        """Same input produces identical output across calls."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        result1, _, _ = encoder.apply(jnp_data, {}, {})
        result2, _, _ = encoder.apply(jnp_data, {}, {})
        assert jnp.allclose(result1["ego_emb"], result2["ego_emb"])

    def test_state_metadata_passthrough(self, encoder, sample_data) -> None:
        """State and metadata are passed through unchanged."""
        state = {"counter": 42}
        metadata = {"source": "test"}
        _, out_state, out_meta = encoder.apply(sample_data, state, metadata)
        assert out_state == state
        assert out_meta == metadata

    def test_batch_call(self, encoder) -> None:
        """Operator __call__ works with datarax Batch."""
        data = {
            "stacked_history": jnp.ones((4, 128)),
            "state/is_sdc": jnp.array([1, 0, 0, 0]),
            "state/all/valid": jnp.ones((4, 11), dtype=jnp.int64),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        result = encoder(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["ego_emb"].shape == (2, 64)
        assert jnp.isfinite(result_data["ego_emb"]).all()
