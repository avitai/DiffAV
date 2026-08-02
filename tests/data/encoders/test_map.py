"""Tests for the VectorNet-style map encoder."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from simulacrax.data.encoders import (
    MapEncoder,
    MapEncoderConfig,
)
from tests import support


@pytest.mark.slow
class TestMapEncoder:
    """Tests for VectorNet-style map encoding."""

    @pytest.fixture()
    def encoder(self) -> MapEncoder:
        """Create map encoder with small dimensions for testing."""
        config = MapEncoderConfig(
            embed_dim=64,
            num_egnn_layers=2,
            max_polylines=16,
        )
        return MapEncoder(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def sample_data(self) -> dict[str, Any]:
        """Roadgraph with 100 points across 10 polylines."""
        n_points = 100
        return {
            "roadgraph_samples/xyz": support.seeded_normal(n_points, 3).astype(np.float32),
            "roadgraph_samples/type": np.ones((n_points, 1), dtype=np.int64),
            "roadgraph_samples/id": np.repeat(np.arange(10), 10).reshape(-1, 1),
            "roadgraph_samples/valid": np.ones((n_points, 1), dtype=np.int64),
            "roadgraph_samples/dir": support.seeded_normal(n_points, 3).astype(np.float32),
        }

    def test_output_shape(self, encoder, sample_data) -> None:
        """Map embeddings have shape [max_polylines, embed_dim]."""
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "map_emb" in result
        assert result["map_emb"].shape == (16, 64)

    def test_gradient_flows(self, encoder, sample_data) -> None:
        """Verify differentiability through MapEncoder."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        def loss_fn(model: MapEncoder) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["map_emb"])

        grads = nnx.grad(loss_fn)(encoder)
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_invalid_points_masked(self, encoder, sample_data) -> None:
        """Invalid roadgraph points should not contribute."""
        sample_data["roadgraph_samples/valid"] = np.zeros_like(
            sample_data["roadgraph_samples/valid"]
        )
        result, _, _ = encoder.apply(sample_data, {}, {})
        assert "map_emb" in result
        assert result["map_emb"].shape == (16, 64)

    def test_jit_compatible(self, encoder, sample_data) -> None:
        """Encoder works under nnx.jit compilation."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}

        @nnx.jit
        def jit_apply(model: MapEncoder) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        result = jit_apply(encoder)
        assert result["map_emb"].shape == (16, 64)
        assert jnp.isfinite(result["map_emb"]).all()

    def test_deterministic_output(self, encoder, sample_data) -> None:
        """Same input produces identical output across calls."""
        jnp_data = {k: jnp.array(v) for k, v in sample_data.items()}
        result1, _, _ = encoder.apply(jnp_data, {}, {})
        result2, _, _ = encoder.apply(jnp_data, {}, {})
        assert jnp.allclose(result1["map_emb"], result2["map_emb"])

    def test_state_metadata_passthrough(self, encoder, sample_data) -> None:
        """State and metadata are passed through unchanged."""
        state = {"counter": 42}
        metadata = {"source": "test"}
        _, out_state, out_meta = encoder.apply(sample_data, state, metadata)
        assert out_state == state
        assert out_meta == metadata

    def test_batch_call(self, encoder) -> None:
        """Operator __call__ works with datarax Batch."""
        n_points = 100
        data = {
            "roadgraph_samples/xyz": jnp.zeros((n_points, 3)),
            "roadgraph_samples/type": jnp.ones((n_points, 1), dtype=jnp.int64),
            "roadgraph_samples/id": jnp.repeat(jnp.arange(10), 10).reshape(-1, 1),
            "roadgraph_samples/valid": jnp.ones((n_points, 1), dtype=jnp.int64),
            "roadgraph_samples/dir": jnp.zeros((n_points, 3)),
        }
        batch = Batch([Element(data=data, state={}) for _ in range(2)])
        result = encoder(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["map_emb"].shape == (2, 16, 64)

    @staticmethod
    def _roadgraph_with_ids(ids: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
        """Build a deterministic roadgraph dict for the given per-point polyline IDs."""
        n_points = ids.shape[0]
        rng = np.random.default_rng(42)
        if valid is None:
            valid = np.ones((n_points, 1), dtype=np.int64)
        return {
            "roadgraph_samples/xyz": rng.normal(size=(n_points, 3)).astype(np.float32),
            "roadgraph_samples/type": np.ones((n_points, 1), dtype=np.int64),
            "roadgraph_samples/id": ids.reshape(-1, 1).astype(np.int64),
            "roadgraph_samples/valid": valid,
            "roadgraph_samples/dir": rng.normal(size=(n_points, 3)).astype(np.float32),
        }

    def test_sparse_wod_ids_match_dense_relabeling(self, encoder) -> None:
        """Arbitrary WOD polyline IDs pool identically to their dense relabeling.

        Real ``roadgraph_samples/id`` values are arbitrary (frequently far
        above ``max_polylines``); a monotonic relabeling to 0..K-1 must not
        change the embeddings.
        """
        sparse_ids = np.repeat(np.array([7, 1234, 20905, 99999, 1500011, 2000000]), 10)
        dense_ids = np.repeat(np.arange(6), 10)
        sparse_data = self._roadgraph_with_ids(sparse_ids)
        dense_data = self._roadgraph_with_ids(dense_ids)
        sparse_result, _, _ = encoder.apply(sparse_data, {}, {})
        dense_result, _, _ = encoder.apply(dense_data, {}, {})
        np.testing.assert_allclose(
            sparse_result["map_emb"],
            dense_result["map_emb"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_excess_polylines_truncate_to_smallest_ids(self) -> None:
        """More unique IDs than slots keeps the smallest IDs deterministically."""
        config = MapEncoderConfig(embed_dim=64, num_egnn_layers=2, max_polylines=4)
        encoder = MapEncoder(config, rngs=nnx.Rngs(0))
        ids = np.repeat(np.arange(6), 10)
        full_data = self._roadgraph_with_ids(ids)
        kept_valid = (ids < 4).astype(np.int64).reshape(-1, 1)
        kept_data = self._roadgraph_with_ids(ids, valid=kept_valid)
        full_result, _, _ = encoder.apply(full_data, {}, {})
        kept_result, _, _ = encoder.apply(kept_data, {}, {})
        np.testing.assert_allclose(
            full_result["map_emb"],
            kept_result["map_emb"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_padding_rows_do_not_affect_embeddings(self, encoder) -> None:
        """WOD padding rows (valid=0, id=-1) contribute nothing, whatever they hold."""
        ids = np.concatenate([np.repeat(np.arange(3), 10), np.full(20, -1)])
        valid = (ids >= 0).astype(np.int64).reshape(-1, 1)
        clean_data = self._roadgraph_with_ids(ids, valid=valid)
        garbage_ids = ids.copy()
        garbage_ids[30:] = np.array([2] * 10 + [12345] * 10)
        garbage_data = {
            **self._roadgraph_with_ids(garbage_ids, valid=valid),
            "roadgraph_samples/xyz": clean_data["roadgraph_samples/xyz"],
            "roadgraph_samples/dir": clean_data["roadgraph_samples/dir"],
        }
        clean_result, _, _ = encoder.apply(clean_data, {}, {})
        garbage_result, _, _ = encoder.apply(garbage_data, {}, {})
        np.testing.assert_allclose(
            clean_result["map_emb"],
            garbage_result["map_emb"],
            rtol=1e-6,
            atol=1e-6,
        )
