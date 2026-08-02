"""Tests for the LiDAR point cloud encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from simulacrax.data.encoders import (
    LiDAREncoder,
    LiDAREncoderConfig,
)
from tests import support


class TestLiDAREncoder:
    """Tests for LiDAR point cloud encoding."""

    @pytest.fixture()
    def encoder(self) -> LiDAREncoder:
        """Create LiDAR encoder with small dimensions."""
        config = LiDAREncoderConfig(embed_dim=64, num_points=64)
        return LiDAREncoder(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """LiDAR point cloud with 128 points."""
        return {
            "lidar/points": support.seeded_normal(128, 3).astype(np.float32),
            "lidar/features": support.seeded_normal(128, 2).astype(np.float32),
            "lidar/valid": np.ones(128, dtype=np.int64),
        }

    def test_output_shape(self, encoder, sample_data) -> None:
        """LiDAR embeddings have shape [num_points, embed_dim]."""
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "lidar_emb" in result
        assert result["lidar_emb"].shape == (64, 64)

    def test_gradient_flows(self, encoder, sample_data) -> None:
        """Verify differentiability through LiDAREncoder."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        def loss_fn(model: LiDAREncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["lidar_emb"])

        grads = nnx.grad(loss_fn)(encoder)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_jit_compatible(self, encoder, sample_data) -> None:
        """Encoder works under nnx.jit compilation."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def jit_apply(model: LiDAREncoder) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        result = jit_apply(encoder)
        assert result["lidar_emb"].shape == (64, 64)
        assert jnp.isfinite(result["lidar_emb"]).all()

    def test_deterministic_output(self, encoder, sample_data) -> None:
        """Same input produces identical output across calls."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        result1, _, _ = encoder.apply(jnp_data, {}, {})
        result2, _, _ = encoder.apply(jnp_data, {}, {})
        assert jnp.allclose(result1["lidar_emb"], result2["lidar_emb"])

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
            "lidar/points": jnp.zeros((128, 3)),
            "lidar/features": jnp.zeros((128, 2)),
            "lidar/valid": jnp.ones(128, dtype=jnp.int64),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        result = encoder(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["lidar_emb"].shape == (2, 64, 64)
        assert jnp.isfinite(result_data["lidar_emb"]).all()
