"""Tests for the agent state encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from diffav.data.encoders import (
    AgentEncoder,
    AgentEncoderConfig,
)
from tests import support


class TestAgentEncoder:
    """Tests for agent state encoding."""

    @pytest.fixture()
    def encoder(self) -> AgentEncoder:
        """Create default agent encoder."""
        config = AgentEncoderConfig(embed_dim=64, num_heads=4, mlp_hidden=128)
        return AgentEncoder(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Sample data with 8 agents, all valid."""
        num_agents = 8
        history = 11
        return {
            "stacked_history": support.seeded_normal(num_agents, 128).astype(np.float32),
            "state/all/valid": np.ones((num_agents, history), dtype=np.int64),
        }

    def test_output_shape(self, encoder, sample_data) -> None:
        """Agent embeddings have correct shape [num_agents, embed_dim]."""
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "agent_emb" in result
        assert result["agent_emb"].shape == (8, 64)

    def test_gradient_flows(self, encoder, sample_data) -> None:
        """Verify differentiability through AgentEncoder."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        def loss_fn(model: AgentEncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["agent_emb"])

        grads = nnx.grad(loss_fn)(encoder)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_masked_agents_zero(self, encoder, sample_data) -> None:
        """Invalid agents should produce zero embeddings."""
        sample_data["state/all/valid"] = np.zeros_like(sample_data["state/all/valid"])
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert np.allclose(result["agent_emb"], 0.0, atol=1e-6)

    def test_jit_compatible(self, encoder, sample_data) -> None:
        """Encoder works under nnx.jit compilation."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def jit_apply(model: AgentEncoder) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        result = jit_apply(encoder)
        assert result["agent_emb"].shape == (8, 64)
        assert jnp.isfinite(result["agent_emb"]).all()

    def test_deterministic_output(self, encoder, sample_data) -> None:
        """Same input produces identical output across calls."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        result1, _, _ = encoder.apply(jnp_data, {}, {})
        result2, _, _ = encoder.apply(jnp_data, {}, {})
        assert jnp.allclose(result1["agent_emb"], result2["agent_emb"])

    def test_deterministic_attribute_toggles(self, encoder) -> None:
        """``train()``/``eval()`` flip the encoder's ``deterministic`` flag."""
        assert encoder.deterministic is True
        encoder.train()
        assert encoder.deterministic is False
        encoder.eval()
        assert encoder.deterministic is True

    def test_dropout_noop_at_zero_rate(self, sample_data) -> None:
        """At the default rate 0.0, train and eval modes are identical."""
        enc = AgentEncoder(
            AgentEncoderConfig(embed_dim=64, num_heads=4, mlp_hidden=128, dropout_rate=0.0),
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        enc.train()
        train_out, _, _ = enc.apply(jnp_data, {}, {})
        enc.eval()
        eval_out, _, _ = enc.apply(jnp_data, {}, {})
        assert jnp.allclose(train_out["agent_emb"], eval_out["agent_emb"])

    def test_dropout_active_in_train_mode(self, sample_data) -> None:
        """With a dropout stream and rate > 0, train mode is stochastic and
        differs from eval mode; eval mode is deterministic."""
        enc = AgentEncoder(
            AgentEncoderConfig(embed_dim=64, num_heads=4, mlp_hidden=128, dropout_rate=0.5),
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        enc.train()
        first, _, _ = enc.apply(jnp_data, {}, {})
        second, _, _ = enc.apply(jnp_data, {}, {})
        enc.eval()
        deterministic, _, _ = enc.apply(jnp_data, {}, {})
        assert not jnp.allclose(first["agent_emb"], second["agent_emb"])
        assert not jnp.allclose(first["agent_emb"], deterministic["agent_emb"])

    def test_dropout_train_mode_differentiable_under_jit(self, sample_data) -> None:
        """Gradients flow through the encoder in train mode under ``nnx.jit``."""
        enc = AgentEncoder(
            AgentEncoderConfig(embed_dim=64, num_heads=4, mlp_hidden=128, dropout_rate=0.5),
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        enc.train()
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def loss_fn(model: AgentEncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["agent_emb"])

        grads = nnx.grad(loss_fn)(enc)
        assert any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))

    def test_state_metadata_passthrough(self, encoder, sample_data) -> None:
        """State and metadata are passed through unchanged."""
        state = {"counter": 42}
        metadata = {"source": "test"}
        _, out_state, out_meta = encoder.apply(sample_data, state, metadata)
        assert out_state == state
        assert out_meta == metadata

    def test_batch_call(self, encoder) -> None:
        """Operator __call__ works with datarax Batch."""
        num_agents = 4
        data = {
            "stacked_history": jnp.ones((num_agents, 128)),
            "state/all/valid": jnp.ones((num_agents, 11), dtype=jnp.int64),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        result = encoder(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["agent_emb"].shape == (2, num_agents, 64)
        assert jnp.isfinite(result_data["agent_emb"]).all()
