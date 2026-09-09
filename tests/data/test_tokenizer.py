"""Tests for the composite scene tokenization pipeline and modality negotiation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.element_batch import Batch, Element
from flax import nnx

from diffav.core.types import Modality, ModalityMode
from diffav.data.tokenizer import (
    resolve_active_modalities,
    SceneTokenizer,
    TokenizerConfig,
)
from tests import support


def _wod_motion_spec() -> dict[str, jax.ShapeDtypeStruct]:
    """Element spec mirroring what WODSource emits: no LiDAR, no camera."""
    num_agents, num_steps, n_rg = 4, 91, 50
    f32 = jnp.float32
    i64 = jnp.int64
    return {
        "state/all/x": jax.ShapeDtypeStruct((num_agents, num_steps), f32),
        "state/all/y": jax.ShapeDtypeStruct((num_agents, num_steps), f32),
        "state/all/bbox_yaw": jax.ShapeDtypeStruct((num_agents, num_steps), f32),
        "state/all/velocity_x": jax.ShapeDtypeStruct((num_agents, num_steps), f32),
        "state/all/velocity_y": jax.ShapeDtypeStruct((num_agents, num_steps), f32),
        "state/all/valid": jax.ShapeDtypeStruct((num_agents, num_steps), i64),
        "state/is_sdc": jax.ShapeDtypeStruct((num_agents,), i64),
        "roadgraph_samples/xyz": jax.ShapeDtypeStruct((n_rg, 3), f32),
        "roadgraph_samples/dir": jax.ShapeDtypeStruct((n_rg, 3), f32),
        "roadgraph_samples/type": jax.ShapeDtypeStruct((n_rg, 1), i64),
        "roadgraph_samples/id": jax.ShapeDtypeStruct((n_rg, 1), i64),
        "roadgraph_samples/valid": jax.ShapeDtypeStruct((n_rg, 1), i64),
    }


def _wod_motion_data() -> dict[str, Any]:
    """Synthetic WOD-Motion scenario dict: agents and map only."""
    num_agents, num_steps, n_rg = 4, 91, 50
    rng = np.random.default_rng(7)
    return {
        "state/all/x": rng.standard_normal((num_agents, num_steps)).astype(np.float32),
        "state/all/y": rng.standard_normal((num_agents, num_steps)).astype(np.float32),
        "state/all/bbox_yaw": rng.standard_normal((num_agents, num_steps)).astype(np.float32),
        "state/all/velocity_x": rng.standard_normal((num_agents, num_steps)).astype(np.float32),
        "state/all/velocity_y": rng.standard_normal((num_agents, num_steps)).astype(np.float32),
        "state/all/valid": np.ones((num_agents, num_steps), dtype=np.int64),
        "state/is_sdc": np.array([1, 0, 0, 0]),
        "roadgraph_samples/xyz": rng.standard_normal((n_rg, 3)).astype(np.float32),
        "roadgraph_samples/dir": rng.standard_normal((n_rg, 3)).astype(np.float32),
        "roadgraph_samples/type": np.ones((n_rg, 1), dtype=np.int64),
        "roadgraph_samples/id": np.repeat(np.arange(5), 10).reshape(-1, 1),
        "roadgraph_samples/valid": np.ones((n_rg, 1), dtype=np.int64),
    }


def _small_config(**overrides: Any) -> TokenizerConfig:
    """Tokenizer config with small dims for test speed."""
    defaults: dict[str, Any] = {
        "embed_dim": 64,
        "num_heads": 4,
        "num_egnn_layers": 2,
        "max_polylines": 16,
        "num_lidar_points": 64,
    }
    defaults.update(overrides)
    return TokenizerConfig(**defaults)


@pytest.mark.slow
class TestModalityNegotiation:
    """Capability negotiation of tokenizer modalities against an element spec."""

    def test_wod_motion_spec_disables_lidar_and_camera(self) -> None:
        """WOD-Motion spec (no lidar/camera keys) negotiates those modalities off."""
        active = resolve_active_modalities(_small_config(), _wod_motion_spec())
        assert Modality.AGENT in active
        assert Modality.MAP in active
        assert Modality.EGO in active
        assert Modality.LIDAR not in active
        assert Modality.CAMERA not in active

    def test_pipeline_runs_on_wod_motion_data(self) -> None:
        """The negotiated pipeline tokenizes WOD-Motion data end to end."""
        tokenizer = SceneTokenizer(
            _small_config(),
            element_spec=_wod_motion_spec(),
            rngs=nnx.Rngs(0),
        )
        result, _, _ = tokenizer.apply(_wod_motion_data(), {}, {})
        # 4 agents + 16 map polylines + 1 ego = 21 tokens
        assert result["scene_embedding"].shape == (21, 64)
        assert "lidar_emb" not in result
        assert "camera_emb" not in result

    def test_required_modality_missing_from_spec_raises(self) -> None:
        """A required modality whose keys the spec lacks fails at construction."""
        config = _small_config(lidar_modality=ModalityMode.REQUIRED)
        with pytest.raises(ValueError, match="lidar"):
            SceneTokenizer(config, element_spec=_wod_motion_spec(), rngs=nnx.Rngs(0))

    def test_off_modality_excluded_even_when_data_present(self) -> None:
        """An off modality is excluded although its data and spec keys exist."""
        config = _small_config(
            lidar_modality=ModalityMode.OFF,
            camera_modality=ModalityMode.OFF,
        )
        tokenizer = SceneTokenizer(config, rngs=nnx.Rngs(0))
        data = _wod_motion_data()
        data["lidar/points"] = np.zeros((128, 3), dtype=np.float32)
        data["camera/images"] = np.zeros((5, 32, 32, 3), dtype=np.float32)
        result, _, _ = tokenizer.apply(data, {}, {})
        assert result["scene_embedding"].shape == (21, 64)
        assert "lidar_emb" not in result
        assert "camera_emb" not in result

    def test_no_spec_auto_enables_all_modalities(self) -> None:
        """Without a spec there is nothing to negotiate against: auto stays on."""
        active = resolve_active_modalities(_small_config(), None)
        assert set(active) == set(Modality)

    def test_all_modalities_off_raises(self) -> None:
        """Disabling every modality is rejected at construction."""
        config = _small_config(
            agent_modality=ModalityMode.OFF,
            ego_modality=ModalityMode.OFF,
            map_modality=ModalityMode.OFF,
            lidar_modality=ModalityMode.OFF,
            camera_modality=ModalityMode.OFF,
        )
        with pytest.raises(ValueError, match="[Nn]o.*modalit"):
            SceneTokenizer(config, rngs=nnx.Rngs(0))

    def test_required_modality_present_in_spec_is_active(self) -> None:
        """Required modalities whose keys exist negotiate on."""
        config = _small_config(map_modality=ModalityMode.REQUIRED)
        active = resolve_active_modalities(config, _wod_motion_spec())
        assert Modality.MAP in active

    def test_active_modalities_exposed(self) -> None:
        """The tokenizer records the negotiated modality set."""
        tokenizer = SceneTokenizer(
            _small_config(),
            element_spec=_wod_motion_spec(),
            rngs=nnx.Rngs(0),
        )
        assert tokenizer.active_modalities == (Modality.AGENT, Modality.MAP, Modality.EGO)

    def test_negotiated_pipeline_gradient_flows(self) -> None:
        """Gradients flow end to end through the negotiated pipeline."""
        tokenizer = SceneTokenizer(
            _small_config(),
            element_spec=_wod_motion_spec(),
            rngs=nnx.Rngs(0),
        )
        jnp_data = {k: jnp.array(v) for k, v in _wod_motion_data().items()}

        def loss_fn(model: SceneTokenizer) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["scene_embedding"])

        grads = nnx.grad(loss_fn)(tokenizer)
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0
        assert any(jnp.any(jnp.abs(g) > 0).item() for g in grad_leaves)

    def test_negotiated_pipeline_jit_compatible(self) -> None:
        """The negotiated pipeline runs under nnx.jit, finite and deterministic."""
        tokenizer = SceneTokenizer(
            _small_config(),
            element_spec=_wod_motion_spec(),
            rngs=nnx.Rngs(0),
        )
        jnp_data = {k: jnp.array(v) for k, v in _wod_motion_data().items()}

        @nnx.jit
        def jit_apply(model: SceneTokenizer) -> dict[str, Any]:
            result, _, _ = model.apply(jnp_data, {}, {})
            return result

        first = jit_apply(tokenizer)
        second = jit_apply(tokenizer)
        assert first["scene_embedding"].shape == (21, 64)
        assert jnp.isfinite(first["scene_embedding"]).all()
        assert jnp.array_equal(first["scene_embedding"], second["scene_embedding"])

    def test_jit_matches_eager_at_double_precision(self) -> None:
        """The jitted graph is semantically identical to the eager pipeline.

        Bit-level f32 agreement between eager and jit is not a JAX
        contract: the GPU backend compiles fused attention kernels whose
        internal precision and reduction order differ per compile
        (``jax.nn.dot_product_attention`` computes at reduced internal
        precision on GPU even for f64 inputs). Run the comparison in
        float64 on CPU instead, where rounding shrinks to machine epsilon:
        the two paths must then agree tightly, while a logic divergence
        introduced by tracing would not shrink with precision.
        """
        x64_was_enabled = bool(getattr(jax.config, "jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(jax.devices("cpu")[0]):
                tokenizer = SceneTokenizer(
                    _small_config(),
                    element_spec=_wod_motion_spec(),
                    rngs=nnx.Rngs(0),
                )
                state = nnx.state(tokenizer)
                state = jax.tree.map(
                    lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                    state,
                )
                nnx.update(tokenizer, state)
                jnp_data = {
                    k: (
                        jnp.asarray(v, dtype=jnp.float64) if v.dtype.kind == "f" else jnp.asarray(v)
                    )
                    for k, v in _wod_motion_data().items()
                }
                eager, _, _ = tokenizer.apply(jnp_data, {}, {})

                @nnx.jit
                def jit_apply(model: SceneTokenizer) -> dict[str, Any]:
                    result, _, _ = model.apply(jnp_data, {}, {})
                    return result

                jitted = jit_apply(tokenizer)
                assert eager["scene_embedding"].dtype == jnp.float64
                assert jnp.allclose(
                    jitted["scene_embedding"],
                    eager["scene_embedding"],
                    rtol=1e-9,
                    atol=1e-12,
                )
        finally:
            jax.config.update("jax_enable_x64", x64_was_enabled)


@pytest.mark.slow
class TestSceneTokenizer:
    """End-to-end tests for the composite tokenization pipeline."""

    @pytest.fixture()
    def tokenizer(self) -> SceneTokenizer:
        """Create test tokenizer with small dims for speed."""
        config = TokenizerConfig(
            embed_dim=64,
            num_heads=4,
            num_egnn_layers=2,
            max_polylines=16,
            num_lidar_points=64,
        )
        return SceneTokenizer(config, rngs=nnx.Rngs(0))

    @pytest.fixture()
    def wod_data(self) -> dict[str, Any]:
        """Minimal synthetic WOD scenario dict with all modalities."""
        num_agents = 4
        num_steps = 91
        n_rg_points = 50
        return {
            "state/all/x": support.seeded_normal(num_agents, num_steps).astype(np.float32),
            "state/all/y": support.seeded_normal(num_agents, num_steps).astype(np.float32),
            "state/all/bbox_yaw": support.seeded_normal(num_agents, num_steps).astype(np.float32),
            "state/all/velocity_x": support.seeded_normal(num_agents, num_steps).astype(np.float32),
            "state/all/velocity_y": support.seeded_normal(num_agents, num_steps).astype(np.float32),
            "state/all/valid": np.ones((num_agents, num_steps), dtype=np.int64),
            "state/is_sdc": np.array([1, 0, 0, 0]),
            "roadgraph_samples/xyz": support.seeded_normal(n_rg_points, 3).astype(np.float32),
            "roadgraph_samples/dir": support.seeded_normal(n_rg_points, 3).astype(np.float32),
            "roadgraph_samples/type": np.ones((n_rg_points, 1), dtype=np.int64),
            "roadgraph_samples/id": np.repeat(np.arange(5), 10).reshape(-1, 1),
            "roadgraph_samples/valid": np.ones((n_rg_points, 1), dtype=np.int64),
            "lidar/points": support.seeded_normal(128, 3).astype(np.float32),
            "camera/images": support.seeded_normal(5, 32, 32, 3).astype(np.float32),
        }

    def test_end_to_end_produces_scene_embedding(self, tokenizer, wod_data) -> None:
        """Full pipeline produces scene_embedding with correct embed_dim."""
        result, _, _ = tokenizer.apply(wod_data, {}, {})
        assert "scene_embedding" in result
        assert result["scene_embedding"].shape[-1] == 64

    def test_all_intermediate_embeddings_present(self, tokenizer, wod_data) -> None:
        """All modality embeddings are in the output."""
        result, _, _ = tokenizer.apply(wod_data, {}, {})
        for key in ["agent_emb", "ego_emb", "map_emb", "lidar_emb", "camera_emb"]:
            assert key in result, f"Missing embedding: {key}"

    def test_embedding_shapes(self, tokenizer, wod_data) -> None:
        """Each modality embedding has expected shape."""
        result, _, _ = tokenizer.apply(wod_data, {}, {})
        assert result["agent_emb"].shape == (4, 64)
        assert result["ego_emb"].shape == (64,)
        assert result["map_emb"].shape == (16, 64)
        assert result["lidar_emb"].shape == (64, 64)
        assert result["camera_emb"].shape == (5, 64)
        # Total: 4 agents + 1 ego + 16 map + 64 lidar + 5 camera = 90
        assert result["scene_embedding"].shape == (90, 64)

    def test_gradient_flow_end_to_end(self, tokenizer, wod_data) -> None:
        """Gradients flow through the entire differentiable pipeline."""
        jnp_data = {k: jnp.array(v) for k, v in wod_data.items()}

        def loss_fn(model: SceneTokenizer) -> jax.Array:
            result, _, _ = model.apply(jnp_data, {}, {})
            return jnp.sum(result["scene_embedding"])

        grads = nnx.grad(loss_fn)(tokenizer)
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0
        has_nonzero = any(jnp.any(jnp.abs(g) > 0).item() for g in grad_leaves)
        assert has_nonzero

    def test_deterministic_output(self, tokenizer, wod_data) -> None:
        """Same input produces identical output across calls."""
        result1, _, _ = tokenizer.apply(wod_data, {}, {})
        result2, _, _ = tokenizer.apply(wod_data, {}, {})
        assert jnp.allclose(result1["scene_embedding"], result2["scene_embedding"])

    def test_state_metadata_passthrough(self, tokenizer, wod_data) -> None:
        """State and metadata are passed through unchanged."""
        state = {"counter": 42}
        metadata = {"source": "test"}
        _, out_state, out_meta = tokenizer.apply(wod_data, state, metadata)
        assert out_state == state
        assert out_meta == metadata

    def test_config_defaults(self) -> None:
        """TokenizerConfig has expected default values."""
        config = TokenizerConfig()
        assert config.embed_dim == 256
        assert config.num_heads == 8
        assert config.fusion_strategy == "cross_attention"
        assert config.feature_fields == ("state/all/x", "state/all/y")
        assert config.history_steps == 11

    def test_config_immutable(self) -> None:
        """TokenizerConfig is frozen after creation."""
        config = TokenizerConfig()
        with pytest.raises(AttributeError):
            config.embed_dim = 128  # type: ignore[misc]

    def test_batch_call(self, tokenizer, wod_data) -> None:
        """SceneTokenizer __call__ works with datarax Batch."""
        jnp_data = {k: jnp.array(v) for k, v in wod_data.items()}
        batch = Batch([Element(data=jnp_data, state={}) for _ in range(2)])
        result = tokenizer(batch)
        assert result.batch_size == 2
        result_data = result.data.get_value()
        assert result_data["scene_embedding"].shape[-1] == 64
        assert result_data["scene_embedding"].shape[0] == 2
        assert jnp.isfinite(result_data["scene_embedding"]).all()

    def test_batch_call_all_embeddings_present(self, tokenizer, wod_data) -> None:
        """All modality embeddings present in batch output."""
        jnp_data = {k: jnp.array(v) for k, v in wod_data.items()}
        batch = Batch([Element(data=jnp_data, state={}) for _ in range(2)])
        result = tokenizer(batch)
        result_data = result.data.get_value()
        for key in ["agent_emb", "ego_emb", "map_emb", "lidar_emb", "camera_emb"]:
            assert key in result_data, f"Missing embedding: {key}"
            assert result_data[key].shape[0] == 2  # batch dim

    def test_batch_call_distinct_inputs(self, tokenizer) -> None:
        """Batch with different inputs produces different outputs per element."""
        num_agents = 4
        num_steps = 91
        n_rg = 50
        key1, key2 = jax.random.split(jax.random.key(42))

        def _make_data(key: jax.Array) -> dict[str, Any]:
            keys = jax.random.split(key, 6)
            return {
                "state/all/x": jax.random.normal(keys[0], (num_agents, num_steps)),
                "state/all/y": jax.random.normal(keys[1], (num_agents, num_steps)),
                "state/all/bbox_yaw": jax.random.normal(keys[2], (num_agents, num_steps)),
                "state/all/velocity_x": jax.random.normal(keys[3], (num_agents, num_steps)),
                "state/all/velocity_y": jax.random.normal(keys[4], (num_agents, num_steps)),
                "state/all/valid": jnp.ones((num_agents, num_steps), dtype=jnp.int64),
                "state/is_sdc": jnp.array([1, 0, 0, 0]),
                "roadgraph_samples/xyz": jax.random.normal(keys[5], (n_rg, 3)),
                "roadgraph_samples/dir": jnp.zeros((n_rg, 3)),
                "roadgraph_samples/type": jnp.ones((n_rg, 1), dtype=jnp.int64),
                "roadgraph_samples/id": jnp.repeat(jnp.arange(5), 10).reshape(-1, 1),
                "roadgraph_samples/valid": jnp.ones((n_rg, 1), dtype=jnp.int64),
                "lidar/points": jnp.zeros((128, 3)),
                "camera/images": jnp.zeros((5, 32, 32, 3)),
            }

        batch = Batch(
            [
                Element(data=_make_data(key1), state={}),
                Element(data=_make_data(key2), state={}),
            ]
        )
        result = tokenizer(batch)
        result_data = result.data.get_value()
        assert not jnp.allclose(
            result_data["scene_embedding"][0],
            result_data["scene_embedding"][1],
        )


@pytest.mark.slow
class TestEncoderDropoutWiring:
    """Encoder dropout activation, and its no-op invariance at the default rate."""

    def test_dropout_is_inert_at_the_default_rate(self) -> None:
        """At the default rate of 0.0, train and eval modes tokenize identically."""
        tokenizer = SceneTokenizer(
            _small_config(dropout_rate=0.0), element_spec=_wod_motion_spec(), rngs=nnx.Rngs(0)
        )
        data = _wod_motion_data()
        tokenizer.train()
        train_out, _, _ = tokenizer.apply(data, {}, {})
        tokenizer.eval()
        eval_out, _, _ = tokenizer.apply(data, {}, {})
        assert jnp.allclose(train_out["scene_embedding"], eval_out["scene_embedding"])

    def test_dropout_active_in_train_mode(self) -> None:
        """With rate > 0 and plain rngs, tokenization is stochastic in train mode
        and deterministic in eval mode; no named dropout stream is needed."""
        tokenizer = SceneTokenizer(
            _small_config(dropout_rate=0.5),
            element_spec=_wod_motion_spec(),
            rngs=nnx.Rngs(0),
        )
        data = _wod_motion_data()
        tokenizer.train()
        first, _, _ = tokenizer.apply(data, {}, {})
        second, _, _ = tokenizer.apply(data, {}, {})
        tokenizer.eval()
        det, _, _ = tokenizer.apply(data, {}, {})
        assert not jnp.allclose(first["scene_embedding"], second["scene_embedding"])
        assert not jnp.allclose(first["scene_embedding"], det["scene_embedding"])
