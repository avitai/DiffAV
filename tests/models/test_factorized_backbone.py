"""Tests for the factorized temporal + social scene backbone.

The behavioral contract is what matters here: each agent is conditioned on its
own context row (locality), agents influence one another only through the
social attention axis (inter-agent coupling), and disabling that axis restores
strict per-agent independence.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffav.models.factorized_backbone import (
    FactorizedSceneBackbone,
    FactorizedSceneBackboneConfig,
)
from tests import support


_AGENTS = 3
_STEPS = 6
_STATE = 4
_CONTEXT = 8
_HIDDEN = 32
_SCENE_TOKENS = 5
_TOKEN_DIM = 16


def _make_backbone(
    *,
    use_social_interaction: bool = True,
    use_map_cross_attention: bool = False,
    scene_token_dim: int | None = None,
    gradient_checkpointing: bool = False,
    seed: int = 0,
) -> FactorizedSceneBackbone:
    """Build a small backbone for fast tests."""
    config = FactorizedSceneBackboneConfig(
        hidden_dim=_HIDDEN,
        num_heads=4,
        state_dim=_STATE,
        num_agents_max=8,
        future_steps=_STEPS,
        context_dim=_CONTEXT,
        num_blocks=2,
        num_temporal_layers=2,
        num_social_layers=1,
        use_social_interaction=use_social_interaction,
        use_map_cross_attention=use_map_cross_attention,
        scene_token_dim=scene_token_dim,
        gradient_checkpointing=gradient_checkpointing,
    )
    return FactorizedSceneBackbone(config, rngs=nnx.Rngs(params=jax.random.key(seed)))


def _conditioned_backbone(
    *,
    use_social_interaction: bool = True,
    use_map_cross_attention: bool = False,
    scene_token_dim: int | None = None,
    gradient_checkpointing: bool = False,
    seed: int = 0,
) -> FactorizedSceneBackbone:
    """A backbone whose adaLN projections carry signal (see randomize_adaln)."""
    backbone = _make_backbone(
        use_social_interaction=use_social_interaction,
        use_map_cross_attention=use_map_cross_attention,
        scene_token_dim=scene_token_dim,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
    )
    support.randomize_adaln(backbone, seed=seed)
    return backbone


def _scene_tokens(seed: int = 7) -> jax.Array:
    """A fused scene-token set ``(num_tokens, token_dim)`` for cross-attention."""
    return jax.random.normal(jax.random.key(seed), (_SCENE_TOKENS, _TOKEN_DIM))


def _inputs(seed: int = 1) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (noisy_trajectories, timestep, scene_context)."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    traj = jax.random.normal(k1, (_AGENTS, _STEPS, _STATE))
    context = jax.random.normal(k2, (_AGENTS, _CONTEXT))
    return traj, jnp.asarray(5), context


class TestConfig:
    """FactorizedSceneBackboneConfig validation."""

    def test_defaults(self) -> None:
        """Defaults match the CTG++ factorization and enable interaction."""
        cfg = FactorizedSceneBackboneConfig()
        assert cfg.num_blocks == 2
        assert cfg.num_temporal_layers == 2
        assert cfg.num_social_layers == 1
        assert cfg.use_social_interaction is True

    @pytest.mark.parametrize(
        "field",
        ["hidden_dim", "num_heads", "num_blocks", "num_temporal_layers", "num_social_layers"],
    )
    def test_positive_fields(self, field: str) -> None:
        """Zero-valued positive fields raise."""
        override: dict[str, Any] = {field: 0}
        with pytest.raises(ValueError, match="must be positive"):
            FactorizedSceneBackboneConfig(**override)

    def test_head_divisibility(self) -> None:
        """hidden_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            FactorizedSceneBackboneConfig(hidden_dim=30, num_heads=4)


class TestOutputShape:
    """The denoiser is shape-preserving."""

    def test_output_matches_input_shape(self) -> None:
        """Output shape equals the noisy-trajectory shape."""
        backbone = _make_backbone()
        traj, t, ctx = _inputs()
        out = backbone(traj, t, ctx)
        assert out.shape == traj.shape


class TestGradientCheckpointing:
    """Rematerializing the blocks preserves the forward and the gradients.

    Checkpointing only recomputes activations in the backward pass, so both the
    output and every parameter gradient must match the standard forward.
    """

    def test_forward_matches_standard(self) -> None:
        """The checkpointed forward equals the standard forward exactly."""
        traj, t, ctx = _inputs()
        standard = _conditioned_backbone(seed=0, gradient_checkpointing=False)
        remat = _conditioned_backbone(seed=0, gradient_checkpointing=True)
        assert jnp.allclose(standard(traj, t, ctx), remat(traj, t, ctx), atol=1e-5)

    def test_gradients_match_standard(self) -> None:
        """Checkpointing recomputes the same parameter gradients."""
        traj, t, ctx = _inputs()

        def loss(backbone: FactorizedSceneBackbone) -> jax.Array:
            return jnp.sum(backbone(traj, t, ctx) ** 2)

        grad_standard = nnx.grad(loss)(_conditioned_backbone(seed=0, gradient_checkpointing=False))
        grad_remat = nnx.grad(loss)(_conditioned_backbone(seed=0, gradient_checkpointing=True))
        leaves_standard = jax.tree_util.tree_leaves(grad_standard)
        leaves_remat = jax.tree_util.tree_leaves(grad_remat)
        assert len(leaves_standard) == len(leaves_remat)
        assert all(jnp.allclose(a, b, atol=1e-5) for a, b in zip(leaves_standard, leaves_remat))


class TestPerAgentConditioning:
    """Per-agent conditioning locality and inter-agent coupling."""

    def test_context_locality_under_independence(self) -> None:
        """With social off, perturbing ctx row j changes only agent j."""
        backbone = _conditioned_backbone(use_social_interaction=False)
        traj, t, ctx = _inputs()
        base = backbone(traj, t, ctx)

        ctx_perturbed = ctx.at[1].add(1.0)
        perturbed = backbone(traj, t, ctx_perturbed)

        assert not jnp.allclose(base[1], perturbed[1])  # agent 1 changed
        assert jnp.array_equal(base[0], perturbed[0])  # agent 0 untouched
        assert jnp.array_equal(base[2], perturbed[2])  # agent 2 untouched

    def test_social_attention_couples_agents(self) -> None:
        """With social on, perturbing agent j's input changes agent i."""
        backbone = _conditioned_backbone(use_social_interaction=True)
        traj, t, ctx = _inputs()
        base = backbone(traj, t, ctx)

        traj_perturbed = traj.at[1].add(1.0)
        perturbed = backbone(traj_perturbed, t, ctx)

        assert not jnp.allclose(base[0], perturbed[0])  # agent 0 feels agent 1

    def test_independence_control_isolates_social(self) -> None:
        """With social off, perturbing agent j's input leaves agent i exact."""
        backbone = _conditioned_backbone(use_social_interaction=False)
        traj, t, ctx = _inputs()
        base = backbone(traj, t, ctx)

        traj_perturbed = traj.at[1].add(1.0)
        perturbed = backbone(traj_perturbed, t, ctx)

        assert jnp.array_equal(base[0], perturbed[0])
        assert jnp.array_equal(base[2], perturbed[2])


class TestTimestepConditioning:
    """The diffusion timestep actually conditions the output."""

    def test_different_timesteps_differ(self) -> None:
        """Two timesteps on identical inputs give different outputs."""
        backbone = _conditioned_backbone()
        traj, _, ctx = _inputs()
        out_a = backbone(traj, jnp.asarray(0), ctx)
        out_b = backbone(traj, jnp.asarray(9), ctx)
        assert not jnp.allclose(out_a, out_b)


class TestPaddingMask:
    """Padded agents are excluded from social attention and never NaN."""

    def test_padded_agent_does_not_leak(self) -> None:
        """A padded agent's tokens do not affect any real agent's output."""
        backbone = _conditioned_backbone(use_social_interaction=True)
        traj, t, ctx = _inputs()
        valid = jnp.array([True, True, False])  # agent 2 padded

        base = backbone(traj, t, ctx, agent_valid_mask=valid)
        traj_perturbed = traj.at[2].add(5.0)  # perturb the padded agent
        perturbed = backbone(traj_perturbed, t, ctx, agent_valid_mask=valid)

        assert jnp.array_equal(base[0], perturbed[0])
        assert jnp.array_equal(base[1], perturbed[1])
        assert bool(jnp.all(jnp.isfinite(base)))

    def test_all_but_one_padded_is_finite(self) -> None:
        """A single real agent among padded ones still produces finite output."""
        backbone = _conditioned_backbone(use_social_interaction=True)
        traj, t, ctx = _inputs()
        valid = jnp.array([True, False, False])
        out = backbone(traj, t, ctx, agent_valid_mask=valid)
        assert bool(jnp.all(jnp.isfinite(out)))


class TestPermutationEquivariance:
    """Agent ordering is carried by row alignment, not a positional embedding."""

    def test_permuting_rows_permutes_outputs(self) -> None:
        """Permuting traj and ctx rows together permutes the outputs."""
        backbone = _conditioned_backbone(use_social_interaction=True)
        traj, t, ctx = _inputs()
        base = backbone(traj, t, ctx)

        perm = jnp.array([2, 0, 1])
        permuted = backbone(traj[perm], t, ctx[perm])

        assert jnp.allclose(permuted, base[perm], rtol=1e-4, atol=1e-5)

    def test_permuting_only_context_changes_outputs(self) -> None:
        """Misaligning ctx rows re-binds conditioning and changes outputs."""
        backbone = _conditioned_backbone(use_social_interaction=False)
        traj, t, ctx = _inputs()
        base = backbone(traj, t, ctx)

        perm = jnp.array([2, 0, 1])
        misaligned = backbone(traj, t, ctx[perm])

        assert not jnp.allclose(misaligned, base)


class TestFailFast:
    """The per-agent context contract is enforced fail-fast."""

    def test_wrong_row_count_raises(self) -> None:
        """A context with a row count != num_agents raises."""
        backbone = _make_backbone()
        traj, t, _ = _inputs()
        bad_ctx = jnp.ones((_AGENTS + 1, _CONTEXT))
        with pytest.raises(ValueError, match="one context row per agent"):
            backbone(traj, t, bad_ctx)

    def test_wrong_ndim_raises(self) -> None:
        """A non-2-D context raises."""
        backbone = _make_backbone()
        traj, t, _ = _inputs()
        with pytest.raises(ValueError, match="2-D"):
            backbone(traj, t, jnp.ones((_AGENTS, 1, _CONTEXT)))

    def test_wrong_context_dim_raises(self) -> None:
        """A context whose width != context_dim raises."""
        backbone = _make_backbone()
        traj, t, _ = _inputs()
        with pytest.raises(ValueError, match="context_dim"):
            backbone(traj, t, jnp.ones((_AGENTS, _CONTEXT + 1)))


class TestTransformCompat:
    """The backbone composes with grad, jit, vmap, and is deterministic."""

    def test_grad_finite_wrt_params_and_context(self) -> None:
        """nnx.grad and jax.grad give finite, nonzero gradients."""
        backbone = _conditioned_backbone()
        traj, t, ctx = _inputs()

        def param_loss(m: FactorizedSceneBackbone) -> jax.Array:
            return jnp.mean(m(traj, t, ctx) ** 2)

        grads = nnx.grad(param_loss)(backbone)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if hasattr(g, "shape")]
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)

        ctx_grad = jax.grad(lambda c: jnp.mean(backbone(traj, t, c) ** 2))(ctx)
        assert bool(jnp.all(jnp.isfinite(ctx_grad)))
        assert float(jnp.max(jnp.abs(ctx_grad))) > 0.0

    def test_jit_matches_eager(self) -> None:
        """jitted forward matches the eager forward."""
        backbone = _make_backbone()
        traj, t, ctx = _inputs()

        @nnx.jit
        def run(m: FactorizedSceneBackbone, tr: jax.Array, c: jax.Array) -> jax.Array:
            return m(tr, jnp.asarray(5), c)

        assert jnp.allclose(run(backbone, traj, ctx), backbone(traj, t, ctx), rtol=1e-5, atol=1e-6)

    def test_vmap_matches_loop(self) -> None:
        """vmap over a scene batch equals a per-scene python loop."""
        backbone = _make_backbone()
        k1, k2 = jax.random.split(jax.random.key(3))
        traj_batch = jax.random.normal(k1, (4, _AGENTS, _STEPS, _STATE))
        ctx_batch = jax.random.normal(k2, (4, _AGENTS, _CONTEXT))
        t = jnp.asarray(5)

        batched = jax.vmap(lambda tr, c: backbone(tr, t, c))(traj_batch, ctx_batch)
        looped = jnp.stack([backbone(traj_batch[i], t, ctx_batch[i]) for i in range(4)])
        assert jnp.allclose(batched, looped, rtol=1e-5, atol=1e-6)

    def test_deterministic(self) -> None:
        """Two identical calls produce identical output."""
        backbone = _make_backbone()
        traj, t, ctx = _inputs()
        assert jnp.array_equal(backbone(traj, t, ctx), backbone(traj, t, ctx))


class TestMapCrossAttentionConfig:
    """Configuration of the optional map cross-attention sublayer."""

    def test_disabled_by_default(self) -> None:
        """Map cross-attention is off unless explicitly enabled."""
        assert FactorizedSceneBackboneConfig().use_map_cross_attention is False

    def test_scene_token_dim_defaults_to_hidden_dim(self) -> None:
        """An unset scene_token_dim mirrors hidden_dim."""
        cfg = FactorizedSceneBackboneConfig(
            hidden_dim=64, num_heads=4, use_map_cross_attention=True
        )
        assert cfg.resolved_scene_token_dim == 64

    def test_scene_token_dim_positive_when_set(self) -> None:
        """A non-positive scene_token_dim raises."""
        with pytest.raises(ValueError, match="scene_token_dim"):
            FactorizedSceneBackboneConfig(use_map_cross_attention=True, scene_token_dim=0)


class TestMapCrossAttention:
    """The map cross-attention conditions denoising on a scene-token set.

    The denoiser follows the CTG++/MotionDiffuser pattern: each trajectory
    token cross-attends to a persistent set of fused scene (map + agent +
    ego) tokens, rather than a single pooled context vector.
    """

    def test_output_shape_preserved(self) -> None:
        """Map conditioning leaves the output shape unchanged."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        out = backbone(traj, t, ctx, scene_tokens=_scene_tokens())
        assert out.shape == traj.shape

    def test_map_perturbation_changes_output(self) -> None:
        """Perturbing a scene token changes the denoised output (map-sensitivity)."""
        backbone = _conditioned_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        tokens = _scene_tokens()
        base = backbone(traj, t, ctx, scene_tokens=tokens)
        perturbed = backbone(traj, t, ctx, scene_tokens=tokens.at[0].add(1.0))
        assert not jnp.allclose(base, perturbed)

    def test_masked_token_does_not_leak(self) -> None:
        """A token marked invalid never influences the output."""
        backbone = _conditioned_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        tokens = _scene_tokens()
        valid = jnp.array([True, True, True, True, False])  # last token invalid
        base = backbone(traj, t, ctx, scene_tokens=tokens, scene_token_valid=valid)
        bumped = backbone(traj, t, ctx, scene_tokens=tokens.at[4].add(9.0), scene_token_valid=valid)
        assert jnp.array_equal(base, bumped)

    def test_all_tokens_masked_stays_finite(self) -> None:
        """All-invalid scene tokens still yield finite output (NaN-safe softmax)."""
        backbone = _conditioned_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        valid = jnp.zeros((_SCENE_TOKENS,), dtype=bool)
        out = backbone(traj, t, ctx, scene_tokens=_scene_tokens(), scene_token_valid=valid)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_gradient_flows_to_params_and_tokens(self) -> None:
        """nnx.grad reaches the cross-attention params; tokens get finite grad."""
        backbone = _conditioned_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        tokens = _scene_tokens()

        def param_loss(m: FactorizedSceneBackbone) -> jax.Array:
            return jnp.mean(m(traj, t, ctx, scene_tokens=tokens) ** 2)

        grads = nnx.grad(param_loss)(backbone)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if hasattr(g, "shape")]
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)

        token_grad = jax.grad(lambda tk: jnp.mean(backbone(traj, t, ctx, scene_tokens=tk) ** 2))(
            tokens
        )
        assert bool(jnp.all(jnp.isfinite(token_grad)))
        assert float(jnp.max(jnp.abs(token_grad))) > 0.0

    def test_jit_matches_eager(self) -> None:
        """jitted map-conditioned forward matches eager."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        tokens = _scene_tokens()

        @nnx.jit
        def run(
            m: FactorizedSceneBackbone, tr: jax.Array, c: jax.Array, tk: jax.Array
        ) -> jax.Array:
            return m(tr, jnp.asarray(5), c, scene_tokens=tk)

        assert jnp.allclose(
            run(backbone, traj, ctx, tokens),
            backbone(traj, t, ctx, scene_tokens=tokens),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_vmap_over_scene_batch(self) -> None:
        """vmap over a scene batch equals a per-scene python loop."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        k1, k2, k3 = jax.random.split(jax.random.key(3), 3)
        traj_batch = jax.random.normal(k1, (4, _AGENTS, _STEPS, _STATE))
        ctx_batch = jax.random.normal(k2, (4, _AGENTS, _CONTEXT))
        tok_batch = jax.random.normal(k3, (4, _SCENE_TOKENS, _TOKEN_DIM))
        t = jnp.asarray(5)

        batched = jax.vmap(lambda tr, c, tk: backbone(tr, t, c, scene_tokens=tk))(
            traj_batch, ctx_batch, tok_batch
        )
        looped = jnp.stack(
            [backbone(traj_batch[i], t, ctx_batch[i], scene_tokens=tok_batch[i]) for i in range(4)]
        )
        assert jnp.allclose(batched, looped, rtol=1e-5, atol=1e-6)

    def test_deterministic(self) -> None:
        """Two identical map-conditioned calls produce identical output."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        tokens = _scene_tokens()
        assert jnp.array_equal(
            backbone(traj, t, ctx, scene_tokens=tokens),
            backbone(traj, t, ctx, scene_tokens=tokens),
        )

    def test_requires_scene_tokens_when_enabled(self) -> None:
        """With the sublayer enabled, omitting scene_tokens raises fail-fast."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        with pytest.raises(ValueError, match="scene_tokens"):
            backbone(traj, t, ctx)

    def test_scene_tokens_without_flag_raises(self) -> None:
        """Passing scene_tokens when the sublayer is disabled raises fail-fast."""
        backbone = _make_backbone()
        traj, t, ctx = _inputs()
        with pytest.raises(ValueError, match="map cross-attention"):
            backbone(traj, t, ctx, scene_tokens=_scene_tokens())

    def test_wrong_token_dim_raises(self) -> None:
        """scene_tokens whose width != scene_token_dim raises."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        with pytest.raises(ValueError, match="scene_token_dim"):
            backbone(traj, t, ctx, scene_tokens=jnp.ones((_SCENE_TOKENS, _TOKEN_DIM + 1)))

    def test_wrong_token_ndim_raises(self) -> None:
        """A non-2-D scene_tokens raises."""
        backbone = _make_backbone(use_map_cross_attention=True, scene_token_dim=_TOKEN_DIM)
        traj, t, ctx = _inputs()
        with pytest.raises(ValueError, match="2-D"):
            backbone(traj, t, ctx, scene_tokens=jnp.ones((1, _SCENE_TOKENS, _TOKEN_DIM)))
