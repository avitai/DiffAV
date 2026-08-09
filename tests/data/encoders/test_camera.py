"""Tests for the camera image encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from diffav.data.encoders import (
    CameraEncoder,
    CameraEncoderConfig,
)
from tests import support


class TestCameraEncoder:
    """Tests for camera image encoding."""

    @pytest.fixture()
    def encoder(self) -> CameraEncoder:
        """Create camera encoder with small images."""
        config = CameraEncoderConfig(embed_dim=64)
        return CameraEncoder(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Multi-view camera images (5 cameras, 32x32 RGB)."""
        return {
            "camera/images": support.seeded_normal(5, 32, 32, 3).astype(np.float32),
            "camera/valid": np.ones(5, dtype=np.int64),
        }

    def test_output_shape(self, encoder, sample_data) -> None:
        """Camera embeddings have shape [num_cameras, embed_dim]."""
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "camera_emb" in result
        assert result["camera_emb"].shape == (5, 64)

    def test_gradient_flows(self, encoder, sample_data) -> None:
        """Verify differentiability through CameraEncoder."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        def loss_fn(model: CameraEncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["camera_emb"])

        grads = nnx.grad(loss_fn)(encoder)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_jit_compatible(self, encoder, sample_data) -> None:
        """Encoder works under nnx.jit compilation."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def jit_apply(model: CameraEncoder) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        result = jit_apply(encoder)
        assert result["camera_emb"].shape == (5, 64)
        assert jnp.isfinite(result["camera_emb"]).all()

    def test_deterministic_output(self, encoder, sample_data) -> None:
        """Same input produces identical output across calls."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        result1, _, _ = encoder.apply(jnp_data, {}, {})
        result2, _, _ = encoder.apply(jnp_data, {}, {})
        assert jnp.allclose(result1["camera_emb"], result2["camera_emb"])

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
            "camera/images": jnp.zeros((5, 32, 32, 3)),
            "camera/valid": jnp.ones(5, dtype=jnp.int64),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        result = encoder(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["camera_emb"].shape == (2, 5, 64)
        assert jnp.isfinite(result_data["camera_emb"]).all()
