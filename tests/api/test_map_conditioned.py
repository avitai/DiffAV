"""Tests for the map-conditioned trajectory model (tokenizer + diffusion).

The composed model tokenizes a raw WOD scene into a fused scene-token set plus
per-agent embeddings, gathers the trajectory agents' rows as the adaLN anchor,
and runs the diffusion backbone with map cross-attention — trained end to end
so gradients reach both the tokenizer (incl. the map encoder) and the backbone.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffav.api.map_conditioned import (
    MapConditionedTrajectoryConfig,
    MapConditionedTrajectoryModel,
)
from diffav.core.geometry import RoadEdges
from diffav.core.types import ModalityMode
from diffav.data.tokenizer import TokenizerConfig
from diffav.models.trajectory_diffusion import GuidanceSpec, TrajectoryDiffusionConfig
from diffav.physics.losses import DiffAVPhysicsConfig
from tests import support


_EMBED = 32
_RAW_AGENTS = 4
_SELECTED = 2
_FUTURE = 4
_STATE = 4


def _scene(seed: int = 3) -> dict[str, Any]:
    """A synthetic WOD-Motion scenario dict (agents + roadgraph, no sensors)."""
    steps, n_rg = 91, 40
    rng = np.random.default_rng(seed)

    def normal(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "state/all/x": normal(_RAW_AGENTS, steps),
        "state/all/y": normal(_RAW_AGENTS, steps),
        "state/all/bbox_yaw": normal(_RAW_AGENTS, steps),
        "state/all/velocity_x": normal(_RAW_AGENTS, steps),
        "state/all/velocity_y": normal(_RAW_AGENTS, steps),
        "state/all/valid": np.ones((_RAW_AGENTS, steps), dtype=np.int64),
        "state/is_sdc": np.array([1, 0, 0, 0]),
        "roadgraph_samples/xyz": normal(n_rg, 3),
        "roadgraph_samples/dir": normal(n_rg, 3),
        "roadgraph_samples/type": np.ones((n_rg, 1), dtype=np.int64),
        "roadgraph_samples/id": np.repeat(np.arange(4), 10).reshape(-1, 1),
        "roadgraph_samples/valid": np.ones((n_rg, 1), dtype=np.int64),
    }


def _config(**diffusion_overrides: Any) -> MapConditionedTrajectoryConfig:
    """A small coupled config; sensor modalities off so no spec is needed."""
    tokenizer = TokenizerConfig(
        embed_dim=_EMBED,
        num_heads=4,
        num_egnn_layers=1,
        max_polylines=8,
        lidar_modality=ModalityMode.OFF,
        camera_modality=ModalityMode.OFF,
    )
    diffusion_defaults: dict[str, Any] = {
        "hidden_dim": _EMBED,
        "num_heads": 4,
        "num_blocks": 1,
        "num_temporal_layers": 1,
        "num_social_layers": 1,
        "future_steps": _FUTURE,
        "num_agents_max": 8,
        "num_timesteps": 10,
        "context_dim": _EMBED,
        "scene_token_dim": _EMBED,
        "use_map_cross_attention": True,
    }
    diffusion_defaults.update(diffusion_overrides)
    diffusion = TrajectoryDiffusionConfig(**diffusion_defaults)
    return MapConditionedTrajectoryConfig(tokenizer=tokenizer, diffusion=diffusion)


def _model(seed: int = 0, **diffusion_overrides: Any) -> MapConditionedTrajectoryModel:
    # A default-stream Rngs serves every stream name the tokenizer's encoders
    # request at construction (params, dropout, ...).
    return MapConditionedTrajectoryModel(
        _config(**diffusion_overrides),
        rngs=nnx.Rngs(seed),
    )


def _agent_rows() -> jax.Array:
    return jnp.array([0, 2])


def _trajectories() -> jax.Array:
    return jax.random.normal(jax.random.key(11), (_SELECTED, _FUTURE, _STATE))


def _reference_pose() -> jax.Array:
    return jax.random.normal(jax.random.key(12), (_SELECTED, 3))


class TestConfigCoupling:
    """The composed config enforces the tokenizer/diffusion dimension contract."""

    def test_requires_map_cross_attention(self) -> None:
        """A diffusion config without map cross-attention is rejected."""
        with pytest.raises(ValueError, match="use_map_cross_attention"):
            _config(use_map_cross_attention=False)

    def test_context_dim_must_match_embed_dim(self) -> None:
        """context_dim must equal the tokenizer embed_dim (the anchor width)."""
        with pytest.raises(ValueError, match="context_dim"):
            _config(context_dim=_EMBED + 8)

    def test_scene_token_dim_must_match_embed_dim(self) -> None:
        """scene_token_dim must equal the tokenizer embed_dim."""
        with pytest.raises(ValueError, match="scene_token_dim"):
            _config(scene_token_dim=_EMBED + 8)


class TestMapConditionedModel:
    """Forward, sampling, sensitivity, and joint differentiability."""

    def test_compute_loss_finite(self) -> None:
        """Map-conditioned loss is a finite scalar."""
        model = _model()
        loss = model.compute_loss(
            _scene(),
            _trajectories(),
            _agent_rows(),
            reference_pose=_reference_pose(),
            key=jax.random.key(1),
        )
        assert loss.shape == ()
        assert bool(jnp.isfinite(loss))

    def test_sample_shape(self) -> None:
        """Sampling yields one trajectory per selected agent row."""
        model = _model()
        pred = model.sample(
            _scene(), _agent_rows(), reference_pose=_reference_pose(), key=jax.random.key(2)
        )
        assert pred.trajectories.shape == (_SELECTED, _FUTURE, _STATE)

    def test_sample_maps_back_to_shared_frame(self) -> None:
        """The reference pose translates the sample back to the shared frame.

        The diffuser generates in each agent's local frame; the same diffusion
        draw under a reference pose shifted 100 m in x must produce samples
        shifted 100 m in x (zero reference yaw ⇒ pure translation).
        """
        model = _model()
        scene, rows = _scene(), _agent_rows()
        pose_origin = jnp.zeros((_SELECTED, 3))
        pose_shifted = pose_origin.at[:, 0].set(100.0)
        at_origin = model.sample(
            scene, rows, reference_pose=pose_origin, key=jax.random.key(2)
        ).trajectories
        shifted = model.sample(
            scene, rows, reference_pose=pose_shifted, key=jax.random.key(2)
        ).trajectories
        assert jnp.allclose(shifted[..., 0], at_origin[..., 0] + 100.0, atol=1e-3)
        assert jnp.allclose(shifted[..., 1], at_origin[..., 1], atol=1e-3)

    def test_map_sensitivity(self) -> None:
        """Perturbing the roadgraph changes the loss (the map conditions)."""
        model = _model()
        support.randomize_adaln(model.diffusion.backbone, seed=0)
        scene = _scene()
        traj, rows = _trajectories(), _agent_rows()
        pose = _reference_pose()
        base = model.compute_loss(scene, traj, rows, reference_pose=pose, key=jax.random.key(3))

        perturbed_scene = dict(scene)
        perturbed_scene["roadgraph_samples/xyz"] = scene["roadgraph_samples/xyz"] + 5.0
        perturbed = model.compute_loss(
            perturbed_scene, traj, rows, reference_pose=pose, key=jax.random.key(3)
        )
        assert not jnp.allclose(base, perturbed)

    def test_joint_grad_reaches_tokenizer_and_backbone(self) -> None:
        """Gradients reach both the tokenizer and the backbone parameters."""
        model = _model()
        support.randomize_adaln(model.diffusion.backbone, seed=0)
        scene, traj, rows = _scene(), _trajectories(), _agent_rows()
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss_fn(p: nnx.State) -> jax.Array:
            merged = nnx.merge(graphdef, p, rest)
            return merged.compute_loss(
                scene, traj, rows, reference_pose=_reference_pose(), key=jax.random.key(4)
            )

        grads = jax.grad(loss_fn)(params)
        tokenizer_grads = jax.tree_util.tree_leaves(grads["tokenizer"])
        backbone_grads = jax.tree_util.tree_leaves(grads["diffusion"])
        assert tokenizer_grads and backbone_grads
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in tokenizer_grads + backbone_grads)
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in tokenizer_grads)
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in backbone_grads)

    def test_agent_rows_select_anchor(self) -> None:
        """Different agent-row selections condition on different anchors."""
        model = _model()
        support.randomize_adaln(model.diffusion.backbone, seed=0)
        scene, traj = _scene(), _trajectories()
        pose = _reference_pose()
        a = model.compute_loss(
            scene, traj, jnp.array([0, 1]), reference_pose=pose, key=jax.random.key(5)
        )
        b = model.compute_loss(
            scene, traj, jnp.array([2, 3]), reference_pose=pose, key=jax.random.key(5)
        )
        assert not jnp.allclose(a, b)

    def test_deterministic(self) -> None:
        """Two identical calls match (deterministic operators)."""
        model = _model()
        scene, traj, rows = _scene(), _trajectories(), _agent_rows()
        pose = _reference_pose()
        a = model.compute_loss(scene, traj, rows, reference_pose=pose, key=jax.random.key(6))
        b = model.compute_loss(scene, traj, rows, reference_pose=pose, key=jax.random.key(6))
        assert jnp.array_equal(a, b)


class TestMapConditionedGuidance:
    """Test-time x̂₀ guidance threads through the map-conditioned sample path.

    The wrapper forwards a :class:`GuidanceSpec` to the inner diffuser, which
    reward-gradient-ascends the clean estimate. Guidance acts in each agent's
    local frame; a zero reference pose makes local == shared frame, so a
    forward-progress (+x) reward must push the returned x coordinate up.
    """

    @staticmethod
    def _forward_reward(x_0: jax.Array) -> jax.Array:
        """Reward forward (+x) progress — a constant-gradient steering signal."""
        return jnp.sum(x_0[..., 0])

    def test_guidance_scale_zero_matches_unguided(self) -> None:
        """A zero guidance scale reproduces the unguided sample exactly."""
        model = _model()
        scene, rows, pose = _scene(), _agent_rows(), jnp.zeros((_SELECTED, 3))
        unguided = model.sample(scene, rows, reference_pose=pose, key=jax.random.key(2))
        zero_scale = model.sample(
            scene,
            rows,
            reference_pose=pose,
            key=jax.random.key(2),
            guidance=GuidanceSpec(reward_fn=self._forward_reward, scale=0.0),
        )
        assert jnp.allclose(unguided.trajectories, zero_scale.trajectories)

    def test_guidance_shifts_sample_toward_reward(self) -> None:
        """Positive guidance on a +x reward increases the sampled x coordinate."""
        model = _model()
        scene, rows, pose = _scene(), _agent_rows(), jnp.zeros((_SELECTED, 3))
        key = jax.random.key(2)
        unguided = model.sample(scene, rows, reference_pose=pose, key=key)
        guided = model.sample(
            scene,
            rows,
            reference_pose=pose,
            key=key,
            guidance=GuidanceSpec(reward_fn=self._forward_reward, scale=10.0),
        )
        assert float(jnp.mean(guided.trajectories[..., 0])) > float(
            jnp.mean(unguided.trajectories[..., 0])
        )

    def test_guided_sample_is_deterministic(self) -> None:
        """Identical key and guidance reproduce the same guided sample."""
        model = _model()
        scene, rows, pose = _scene(), _agent_rows(), jnp.zeros((_SELECTED, 3))
        guidance = GuidanceSpec(reward_fn=self._forward_reward, scale=5.0)
        first = model.sample(
            scene, rows, reference_pose=pose, key=jax.random.key(3), guidance=guidance
        )
        second = model.sample(
            scene, rows, reference_pose=pose, key=jax.random.key(3), guidance=guidance
        )
        assert jnp.array_equal(first.trajectories, second.trajectories)

    def test_gradient_clip_bounds_explosive_guidance(self) -> None:
        """Per-step gradient clipping tames an explosive-gradient reward.

        Without clipping a huge-magnitude reward gradient drives the sample far
        off-distribution; the norm clip caps each step so the sample stays
        bounded and finite — the stability guard the guided-sampling literature
        (CTG / MotionDiffuser) requires.
        """
        model = _model()
        scene, rows, pose = _scene(), _agent_rows(), jnp.zeros((_SELECTED, 3))

        def explosive_reward(x_0: jax.Array) -> jax.Array:
            return 1.0e6 * jnp.sum(x_0[..., 0])

        unclipped = model.sample(
            scene,
            rows,
            reference_pose=pose,
            key=jax.random.key(2),
            guidance=GuidanceSpec(reward_fn=explosive_reward, scale=1.0),
        ).trajectories
        clipped = model.sample(
            scene,
            rows,
            reference_pose=pose,
            key=jax.random.key(2),
            guidance=GuidanceSpec(reward_fn=explosive_reward, scale=1.0, grad_clip=1.0),
        ).trajectories
        assert bool(jnp.all(jnp.isfinite(clipped)))
        assert float(jnp.max(jnp.abs(clipped))) < float(jnp.max(jnp.abs(unclipped)))

    def test_guided_sample_vmaps_over_keys(self) -> None:
        """Guided sampling vmaps over per-candidate keys (the steering-pool path)."""
        model = _model()
        scene, rows, pose = _scene(), _agent_rows(), jnp.zeros((_SELECTED, 3))
        guidance = GuidanceSpec(reward_fn=self._forward_reward, scale=5.0)
        graphdef, state = nnx.split(model)

        def sample_one(sample_key: jax.Array) -> jax.Array:
            merged = nnx.merge(graphdef, state)
            return merged.sample(
                scene,
                rows,
                reference_pose=pose,
                key=sample_key,
                guidance=guidance,
            ).trajectories

        keys = jax.random.split(jax.random.key(4), 3)
        batched = jax.vmap(sample_one)(keys)
        assert batched.shape == (3, _SELECTED, _FUTURE, _STATE)
        assert bool(jnp.all(jnp.isfinite(batched)))


class TestSceneTokenValidity:
    """The composed model derives a per-token validity mask and passes it on.

    Padded scene tokens — empty polyline slots and agents with no valid
    history — must be excluded from the backbone's map cross-attention.
    """

    def test_encode_scene_returns_aligned_validity(self) -> None:
        """The validity vector has one boolean per fused scene token."""
        model = _model()
        scene_tokens, _agent_emb, scene_token_valid = model._encode_scene(_scene())
        assert scene_token_valid.shape == (scene_tokens.shape[0],)
        assert scene_token_valid.dtype == jnp.bool_

    def test_padded_polylines_are_invalid(self) -> None:
        """Empty polyline slots (max_polylines > unique polylines) are invalid."""
        # _scene has 4 unique polylines; the tokenizer caps at 8 slots, plus 4
        # (all valid) agent tokens and 1 ego token — so 9 of 13 tokens are valid.
        model = _model()
        _tokens, _agent_emb, scene_token_valid = model._encode_scene(_scene())
        assert scene_token_valid.shape == (_RAW_AGENTS + 8 + 1,)
        assert int(jnp.sum(scene_token_valid)) == _RAW_AGENTS + 4 + 1

    def test_invalid_agents_are_masked(self) -> None:
        """An agent with no valid history step yields an invalid scene token."""
        model = _model()
        scene = dict(_scene())
        valid = np.ones((_RAW_AGENTS, 91), dtype=np.int64)
        valid[1] = 0  # agent 1 has no valid step
        scene["state/all/valid"] = valid
        _tokens, _agent_emb, scene_token_valid = model._encode_scene(scene)
        assert bool(scene_token_valid[0])
        assert not bool(scene_token_valid[1])


def _physics_config(
    *, annealing: bool = True, **physics_overrides: Any
) -> MapConditionedTrajectoryConfig:
    """The default coupled config with physics enabled on the x̂₀ estimate."""
    return dataclasses.replace(
        _config(),
        physics=DiffAVPhysicsConfig(**physics_overrides),
        physics_x0_annealing=annealing,
    )


def _physics_model(
    *, annealing: bool = False, **physics_overrides: Any
) -> MapConditionedTrajectoryModel:
    """A physics-enabled model at the same seed a bare ``_model()`` uses.

    Physics adds no parameters, so its diffusion weights match ``_model()`` at
    seed 0 — every comparison below holds the diffusion path fixed and varies
    only the physics wiring.
    """
    return MapConditionedTrajectoryModel(
        _physics_config(annealing=annealing, **physics_overrides), rngs=nnx.Rngs(0)
    )


def _off_road_edges() -> RoadEdges:
    """A small counterclockwise square at the origin.

    Most standard-normal trajectory positions fall outside it, so the boundary
    term is nonzero — enough to prove ``road_edges`` reaches the physics loss.
    """
    half = 0.2
    square = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


class TestMapConditionedPhysics:
    """Physics on x̂₀: the kinematic/collision/boundary terms fold into the loss.

    The physics penalty scores the diffusion model's clean-trajectory
    reconstruction (x̂₀), so its gradients shape the denoiser rather than the
    fixed ground truth. Physics is off by default; when enabled it shares the
    physics-free model's diffusion weights at seed 0, so each test holds the
    diffusion path fixed at a shared key and varies only the physics wiring.
    """

    def test_physics_disabled_by_default(self) -> None:
        """The default config carries no physics; the loss is pure diffusion."""
        assert _config().physics is None
        loss = _model().compute_loss(
            _scene(),
            _trajectories(),
            _agent_rows(),
            reference_pose=_reference_pose(),
            key=jax.random.key(1),
        )
        assert bool(jnp.isfinite(loss))

    def test_physics_adds_annealed_positive_term(self) -> None:
        """Physics strictly raises the loss; x̂₀-annealing scales it down by ᾱ_t."""
        scene, traj, rows, key = _scene(), _trajectories(), _agent_rows(), jax.random.key(7)
        pose = _reference_pose()
        diffusion_only = _model().compute_loss(scene, traj, rows, reference_pose=pose, key=key)
        no_anneal = _physics_model(annealing=False, adaptive_weighting=False).compute_loss(
            scene, traj, rows, reference_pose=pose, key=key
        )
        annealed = _physics_model(annealing=True, adaptive_weighting=False).compute_loss(
            scene, traj, rows, reference_pose=pose, key=key
        )
        assert float(diffusion_only) < float(no_anneal)
        assert float(annealed) <= float(no_anneal) + 1e-5

    def test_road_edges_add_boundary_term(self) -> None:
        """Passing off-road ``road_edges`` raises the loss via the boundary term."""
        scene, traj, rows, key = _scene(), _trajectories(), _agent_rows(), jax.random.key(8)
        model = _physics_model(annealing=False, adaptive_weighting=False)
        pose = _reference_pose()
        without = model.compute_loss(scene, traj, rows, reference_pose=pose, key=key)
        with_edges = model.compute_loss(
            scene, traj, rows, reference_pose=pose, key=key, road_edges=_off_road_edges()
        )
        assert float(with_edges) > float(without)

    def test_valid_mask_reaches_physics(self) -> None:
        """Masking an agent shifts the physics term beyond its diffusion effect."""
        scene, traj, rows, key = _scene(), _trajectories(), _agent_rows(), jax.random.key(9)
        full = jnp.ones((_SELECTED, _FUTURE), dtype=bool)
        masked = full.at[1].set(False)
        physics_free = _model()
        physics = _physics_model(annealing=False, adaptive_weighting=False)

        pose = _reference_pose()

        def mask_delta(m: MapConditionedTrajectoryModel) -> float:
            hi = m.compute_loss(scene, traj, rows, reference_pose=pose, key=key, valid_mask=full)
            lo = m.compute_loss(scene, traj, rows, reference_pose=pose, key=key, valid_mask=masked)
            return float(hi - lo)

        # If physics ignored the mask, both models would shift by the same
        # diffusion-only amount when the mask changes.
        assert not np.isclose(mask_delta(physics_free), mask_delta(physics))

    def test_epoch_scales_adaptive_physics_weight(self) -> None:
        """A later epoch changes the adaptive physics weight, hence the loss."""
        scene, traj, rows, key = _scene(), _trajectories(), _agent_rows(), jax.random.key(10)
        model = _physics_model(annealing=False)  # adaptive weighting on by default
        pose = _reference_pose()
        early = model.compute_loss(scene, traj, rows, reference_pose=pose, key=key, epoch=0)
        late = model.compute_loss(scene, traj, rows, reference_pose=pose, key=key, epoch=300)
        assert not jnp.allclose(early, late)

    def test_physics_joint_grad_reaches_tokenizer_and_backbone(self) -> None:
        """With physics on, gradients still reach both submodules and stay finite."""
        model = _physics_model(annealing=False, adaptive_weighting=False)
        support.randomize_adaln(model.diffusion.backbone, seed=0)
        scene, traj, rows = _scene(), _trajectories(), _agent_rows()
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss_fn(p: nnx.State) -> jax.Array:
            merged = nnx.merge(graphdef, p, rest)
            return merged.compute_loss(
                scene,
                traj,
                rows,
                reference_pose=_reference_pose(),
                key=jax.random.key(4),
                road_edges=_off_road_edges(),
            )

        grads = jax.grad(loss_fn)(params)
        tokenizer_grads = jax.tree_util.tree_leaves(grads["tokenizer"])
        backbone_grads = jax.tree_util.tree_leaves(grads["diffusion"])
        assert tokenizer_grads and backbone_grads
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in tokenizer_grads + backbone_grads)

    def test_physics_compute_loss_is_jittable(self) -> None:
        """nnx.jit of the physics loss matches eager — the RoadEdges pytree, the
        ᾱ_t anneal, and the static physics module all trace cleanly."""
        model = _physics_model(annealing=True, adaptive_weighting=False)
        scene, traj, rows, key = _scene(), _trajectories(), _agent_rows(), jax.random.key(12)
        edges = _off_road_edges()
        mask = jnp.ones((_SELECTED, _FUTURE), dtype=bool)
        pose = _reference_pose()
        eager = model.compute_loss(
            scene, traj, rows, reference_pose=pose, key=key, road_edges=edges, valid_mask=mask
        )

        @nnx.jit
        def step(
            m: MapConditionedTrajectoryModel,
            sc: dict[str, Any],
            tr: jax.Array,
            rw: jax.Array,
            rp: jax.Array,
            re: RoadEdges,
            vm: jax.Array,
            k: jax.Array,
        ) -> jax.Array:
            return m.compute_loss(
                sc, tr, rw, reference_pose=rp, key=k, road_edges=re, valid_mask=vm
            )

        jitted = step(model, scene, traj, rows, pose, edges, mask, key)
        assert jnp.allclose(eager, jitted)

    def test_physics_batched_train_step_jits(self) -> None:
        """The showcase train-step shape jits: a single ``nnx.jit`` wrapping a
        ``jax.vmap`` over per-scene raw dicts, trajectories, rows, road edges,
        and validity, with the epoch a traced scalar flowing through the
        adaptive weight schedule (no per-epoch retrace)."""
        model = _physics_model(annealing=True)  # adaptive weighting on by default
        batch = 2
        scenes = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), _scene(0), _scene(1))
        traj = jnp.stack([_trajectories(), _trajectories()])
        rows = jnp.stack([_agent_rows(), _agent_rows()])
        poses = jnp.stack([_reference_pose(), _reference_pose()])
        valid = jnp.ones((batch, _SELECTED, _FUTURE), dtype=bool)
        edges = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), _off_road_edges(), _off_road_edges()
        )
        keys = jax.random.split(jax.random.key(13), batch)

        @nnx.jit
        def train_step(
            m: MapConditionedTrajectoryModel,
            sc: dict[str, Any],
            tr: jax.Array,
            rw: jax.Array,
            rp: jax.Array,
            re: RoadEdges,
            vm: jax.Array,
            ep: jax.Array,
            ks: jax.Array,
        ) -> jax.Array:
            losses = jax.vmap(
                lambda s, t, r, p, e, v, k: m.compute_loss(
                    s, t, r, reference_pose=p, key=k, epoch=ep, road_edges=e, valid_mask=v
                )
            )(sc, tr, rw, rp, re, vm, ks)
            return jnp.mean(losses)

        loss = train_step(model, scenes, traj, rows, poses, edges, valid, jnp.asarray(5), keys)
        assert bool(jnp.isfinite(loss))
