"""Tests for map-conditioned Diffusion-DPO log-prob and preference loss.

Cover the diffusion log-prob proxy (correct in the model's x̂₀ parameterization,
computed in the local frame) and the DPO loss, with a focus on the central
ablation: the ``freeze_tokenizer`` toggle must gate whether the map encoder
receives gradient — frozen adapts the backbone only; learnable flows end-to-end.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffav.alignment.dpo_trainer import DPOAlignmentConfig
from diffav.api.map_conditioned import (
    build_map_conditioned_model,
    MapConditionedBuildSpec,
    MapConditionedTrajectoryModel,
)
from diffav.api.map_conditioned_dpo import (
    _scene_pair_log_probs,
    assemble_dpo_batch,
    build_dpo_arm,
    map_conditioned_dpo_loss,
    map_conditioned_dpo_step,
)
from diffav.core.geometry import RoadEdges
from diffav.evaluation.map_conditioned_evaluator import ValidationScene
from tests import support
from tests.api.test_map_conditioned import (
    _agent_rows,
    _reference_pose,
    _scene,
    _trajectories,
)


_FUTURE = 4
_NUM_SAMPLES = 2


def _model(seed: int = 0) -> MapConditionedTrajectoryModel:
    """A small map-conditioned model with a non-trivial conditioning path.

    ``randomize_adaln`` gives the adaLN anchor and map cross-attention real
    weights, so gradients genuinely flow through the scene conditioning (a
    zero-init adaLN would zero the tokenizer gradient regardless of the toggle).
    """
    spec = MapConditionedBuildSpec(
        hidden_dim=32,
        num_heads=4,
        num_blocks=1,
        num_temporal_layers=1,
        num_social_layers=1,
        max_agents=8,
        future_steps=_FUTURE,
        diffusion_steps=10,
        kinematic_weight=0.0,
        collision_weight=0.0,
    )
    model = build_map_conditioned_model(spec, rngs=nnx.Rngs(seed))
    support.randomize_adaln(model.diffusion.backbone, seed=seed)
    return model


def _tokenizer_and_backbone_grads(
    freeze_tokenizer: bool,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Gradients of a log-prob w.r.t. tokenizer vs backbone params under the toggle."""
    model = _model()
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    scene, rows, traj, pose = _scene(), _agent_rows(), _trajectories(), _reference_pose()

    def loss_fn(p: nnx.State) -> jax.Array:
        merged = nnx.merge(graphdef, p, rest)
        return merged.denoising_log_prob(
            traj,
            rows,
            scene,
            reference_pose=pose,
            key=jax.random.key(0),
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=freeze_tokenizer,
        )

    grads = jax.grad(loss_fn)(params)
    tokenizer = jax.tree_util.tree_leaves(grads["tokenizer"])
    backbone = jax.tree_util.tree_leaves(grads["diffusion"])
    return tokenizer, backbone


class TestDenoisingLogProb:
    """The Monte-Carlo diffusion log-prob proxy."""

    def test_is_finite_nonpositive_scalar(self) -> None:
        """The proxy is a finite, non-positive scalar (negative mean loss)."""
        model = _model()
        lp = model.denoising_log_prob(
            _trajectories(),
            _agent_rows(),
            _scene(),
            reference_pose=_reference_pose(),
            key=jax.random.key(1),
            num_samples=_NUM_SAMPLES,
        )
        assert lp.shape == ()
        assert bool(jnp.isfinite(lp))
        assert float(lp) <= 0.0

    def test_deterministic(self) -> None:
        """Same key and inputs reproduce the log-prob (no dropout noise)."""
        model = _model()
        kwargs: dict[str, Any] = {
            "reference_pose": _reference_pose(),
            "key": jax.random.key(2),
            "num_samples": _NUM_SAMPLES,
        }
        first = model.denoising_log_prob(_trajectories(), _agent_rows(), _scene(), **kwargs)
        second = model.denoising_log_prob(_trajectories(), _agent_rows(), _scene(), **kwargs)
        assert jnp.array_equal(first, second)


class TestFreezeTokenizerToggle:
    """The ablation toggle gates the tokenizer gradient; the backbone always trains."""

    def test_frozen_zeros_tokenizer_gradient(self) -> None:
        """Frozen: the map encoder receives no gradient; the backbone does."""
        tokenizer, backbone = _tokenizer_and_backbone_grads(freeze_tokenizer=True)
        assert tokenizer and backbone
        assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in tokenizer)
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in backbone)

    def test_learnable_flows_into_tokenizer(self) -> None:
        """Learnable: gradient flows end-to-end into the map encoder."""
        tokenizer, backbone = _tokenizer_and_backbone_grads(freeze_tokenizer=False)
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in tokenizer)
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in backbone)


def _dpo_batch() -> dict[str, Any]:
    """A scene-grouped DPO batch: 2 scenes x 2 pairs each."""
    return assemble_dpo_batch(
        [
            (_validation_scene(0), *_pair_set(2)),
            (_validation_scene(1), *_pair_set(2)),
        ]
    )


class TestMapConditionedDpoLoss:
    """The reference-based map-conditioned DPO loss and its reward metrics."""

    def test_returns_finite_loss_and_reward_metrics(self) -> None:
        """The loss is a finite scalar with in-range reward metrics."""
        policy = _model(0)
        reference = nnx.clone(policy)
        loss, aux = map_conditioned_dpo_loss(
            policy,
            reference,
            _dpo_batch(),
            jax.random.key(3),
            config=DPOAlignmentConfig(beta=100.0),
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=True,
        )
        assert loss.shape == ()
        assert bool(jnp.isfinite(loss))
        assert 0.0 <= float(aux["reward_accuracy"]) <= 1.0
        assert bool(jnp.isfinite(aux["reward_margin"]))

    def test_identical_policy_and_reference_give_zero_margin(self) -> None:
        """A fresh reference clone yields ~zero implicit-reward margin."""
        policy = _model(0)
        reference = nnx.clone(policy)
        _, aux = map_conditioned_dpo_loss(
            policy,
            reference,
            _dpo_batch(),
            jax.random.key(3),
            config=DPOAlignmentConfig(beta=100.0),
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=True,
        )
        # policy == reference and shared keys ⇒ log-ratios cancel to zero.
        assert abs(float(aux["reward_margin"])) < 1e-3


def _dummy_road_edges() -> RoadEdges:
    """A minimal square road; unused by the batch assembler but required by the scene."""
    square = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return RoadEdges.from_polylines([square])


def _validation_scene(seed: int) -> ValidationScene:
    """A synthetic validation scene carrying one raw scene dict."""
    return ValidationScene(
        scene=_scene(seed),
        trajectories=jnp.zeros((2, _FUTURE, 4)),
        agent_rows=_agent_rows(),
        valid=jnp.ones((2, _FUTURE)),
        tracks_to_predict=jnp.array([False, True]),
        reference_pose=_reference_pose(),
        road_edges=_dummy_road_edges(),
    )


def _pair_set(num_pairs: int) -> tuple[jax.Array, jax.Array]:
    """A ``num_pairs`` chosen/rejected set (rejected offset so pairs are non-degenerate)."""
    chosen = jnp.broadcast_to(_trajectories(), (num_pairs, 2, _FUTURE, 4))
    return chosen, chosen + 1.0


def _tokenizer_leaves(model: MapConditionedTrajectoryModel) -> list[jax.Array]:
    """Flat tokenizer parameter leaves."""
    return jax.tree_util.tree_leaves(nnx.state(model.tokenizer, nnx.Param))


def _backbone_leaves(model: MapConditionedTrajectoryModel) -> list[jax.Array]:
    """Flat diffusion-backbone parameter leaves."""
    return jax.tree_util.tree_leaves(nnx.state(model.diffusion, nnx.Param))


def _any_changed(before: list[jax.Array], after: list[jax.Array]) -> bool:
    """Whether any paired leaf changed between the two snapshots."""
    return any(not bool(jnp.array_equal(a, b)) for a, b in zip(before, after, strict=True))


class TestAssembleDpoBatch:
    """Per-scene pairs stack into a scene-grouped DPO batch."""

    def test_stacks_scenes_and_pairs(self) -> None:
        """Scenes stack on a leading S axis; pairs on a per-scene P axis."""
        batch = assemble_dpo_batch(
            [
                (_validation_scene(0), *_pair_set(2)),
                (_validation_scene(1), *_pair_set(2)),
            ]
        )
        # (S, P, A, T, 4): 2 scenes, 2 pairs, 2 agents.
        assert batch["chosen"].shape == (2, 2, 2, _FUTURE, 4)
        assert batch["rejected"].shape == (2, 2, 2, _FUTURE, 4)
        assert batch["agent_rows"].shape == (2, 2)
        assert batch["reference_pose"].shape == (2, 2, 3)
        # Every raw scene leaf carries the same leading scene axis.
        leaves = jax.tree_util.tree_leaves(batch["scene"])
        assert leaves and all(leaf.shape[0] == 2 for leaf in leaves)

    def test_uneven_pair_counts_raise(self) -> None:
        """Scenes must contribute the same number of pairs (rectangular P axis)."""
        with pytest.raises(ValueError, match="same number of pairs"):
            assemble_dpo_batch(
                [
                    (_validation_scene(0), *_pair_set(2)),
                    (_validation_scene(1), *_pair_set(3)),
                ]
            )

    def test_empty_entries_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one scene entry"):
            assemble_dpo_batch([])


class TestSceneGroupedLogProbs:
    """Encoding each scene once matches per-pair denoising_log_prob (shared c)."""

    def test_matches_per_pair_denoising_log_prob(self) -> None:
        """Shared-conditioning log-probs equal independent per-pair evaluations.

        Runs at ``highest`` matmul precision. On GPU the default TF32 matmuls carry
        only ~10-bit mantissa and cuBLAS picks batch-size-dependent tiling, which
        alone shifts the batched shared-``c`` path from the size-1 per-pair path by
        ~1e-3 (verified by feeding identical inputs at batch sizes 1/2/4). With TF32
        off the two paths agree to true float32 precision (~1e-5), so the tight
        ``rtol`` genuinely tests the refactor rather than tolerating a matmul
        artifact. In training this noise cancels — policy and reference share the
        identical batch structure — so TF32 stays on there.
        """
        model = _model(0)
        batch = _dpo_batch()
        key = jax.random.key(5)
        with jax.default_matmul_precision("highest"):
            chosen_lp, _ = _scene_pair_log_probs(
                model,
                batch["scene"],
                batch["agent_rows"],
                batch["reference_pose"],
                batch["chosen"],
                batch["rejected"],
                key=key,
                num_samples=_NUM_SAMPLES,
                freeze_tokenizer=True,
            )
            for scene_index in range(2):
                scene = jax.tree_util.tree_map(lambda leaf, s=scene_index: leaf[s], batch["scene"])
                for pair_index in range(2):
                    expected = model.denoising_log_prob(
                        batch["chosen"][scene_index, pair_index],
                        batch["agent_rows"][scene_index],
                        scene,
                        reference_pose=batch["reference_pose"][scene_index],
                        key=key,
                        num_samples=_NUM_SAMPLES,
                        freeze_tokenizer=True,
                    )
                    assert jnp.allclose(chosen_lp[scene_index, pair_index], expected, rtol=1e-4)


class TestMapConditionedDpoStep:
    """The ranked-DPO update and its ablation guarantee."""

    def _batch(self) -> dict[str, Any]:
        chosen, rejected = _pair_set(2)
        return assemble_dpo_batch([(_validation_scene(0), chosen, rejected)])

    def test_frozen_arm_leaves_tokenizer_unchanged(self) -> None:
        """Frozen: the update moves the backbone but never the map encoder."""
        policy, reference, optimizer = build_dpo_arm(_model(), learning_rate=1e-2)
        batch = self._batch()
        tok_before, backbone_before = _tokenizer_leaves(policy), _backbone_leaves(policy)
        map_conditioned_dpo_step(
            policy,
            reference,
            optimizer,
            batch,
            jax.random.key(7),
            config=DPOAlignmentConfig(beta=100.0),
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=True,
        )
        assert not _any_changed(tok_before, _tokenizer_leaves(policy))
        assert _any_changed(backbone_before, _backbone_leaves(policy))

    def test_learnable_arm_updates_tokenizer(self) -> None:
        """Learnable: the update flows end-to-end into the map encoder."""
        policy, reference, optimizer = build_dpo_arm(_model(), learning_rate=1e-2)
        batch = self._batch()
        tok_before = _tokenizer_leaves(policy)
        map_conditioned_dpo_step(
            policy,
            reference,
            optimizer,
            batch,
            jax.random.key(7),
            config=DPOAlignmentConfig(beta=100.0),
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=False,
        )
        assert _any_changed(tok_before, _tokenizer_leaves(policy))

    def test_step_raises_reward_margin(self) -> None:
        """One update increases the implicit-reward margin on the trained batch."""
        policy, reference, optimizer = build_dpo_arm(_model(), learning_rate=1e-2)
        batch = self._batch()
        config = DPOAlignmentConfig(beta=50.0)
        eval_key = jax.random.key(9)

        def margin() -> float:
            _, aux = map_conditioned_dpo_loss(
                policy,
                reference,
                batch,
                eval_key,
                config=config,
                num_samples=_NUM_SAMPLES,
                freeze_tokenizer=True,
            )
            return float(aux["reward_margin"])

        before = margin()
        map_conditioned_dpo_step(
            policy,
            reference,
            optimizer,
            batch,
            jax.random.key(8),
            config=config,
            num_samples=_NUM_SAMPLES,
            freeze_tokenizer=True,
        )
        assert margin() > before

    def test_gradient_accumulation_matches_full_batch(self) -> None:
        """Per-scene accumulation reconstructs the full-batch objective exactly.

        The pair-count weighting makes the accumulated loss AND gradient equal the
        full-batch values by linearity (the DPO loss is a mean over pairs). This is
        verified at the loss/reward-margin level, where a wrong weight would shift
        the value by O(1). Parameter-level equality is deliberately NOT asserted:
        the full batch evaluates the log-probs at vmap size S while accumulation
        uses size 1, and that batch-size float floor (~1e-6) is amplified by beta
        through the saturating log-sigmoid (measured: loss/grad differences scale
        as beta*1e-6), then further by Adam's sign-normalization — so a param diff
        reflects float reassociation, not the accumulation. Small beta keeps the
        floor negligible.
        """
        batch = assemble_dpo_batch(
            [(_validation_scene(0), *_pair_set(2)), (_validation_scene(1), *_pair_set(2))]
        )
        base = _model(0)
        # A DISTINCT reference makes the two scenes' per-pair losses differ, so a
        # wrong per-chunk weight (wrong sum OR wrong distribution) shifts the
        # aggregated loss/margin — an identical policy==reference would collapse
        # every pair to ln 2 and only detect a wrong weight sum.
        reference_source = _model(1)
        key = jax.random.key(11)
        config = DPOAlignmentConfig(beta=1.0)

        def run(scene_microbatch: int) -> tuple[float, float, bool]:
            policy, _, optimizer = build_dpo_arm(base, learning_rate=1e-2)
            reference = nnx.clone(reference_source)
            before = _backbone_leaves(policy)
            loss, aux = map_conditioned_dpo_step(
                policy,
                reference,
                optimizer,
                batch,
                key,
                config=config,
                num_samples=_NUM_SAMPLES,
                freeze_tokenizer=False,
                scene_microbatch=scene_microbatch,
            )
            changed = _any_changed(before, _backbone_leaves(policy))
            return float(loss), float(aux["reward_margin"]), changed

        with jax.default_matmul_precision("highest"):
            full_loss, full_margin, _ = run(0)
            accum_loss, accum_margin, accum_changed = run(1)

        assert accum_changed  # accumulation actually applied an update
        assert full_margin != 0.0  # the distinct reference makes the weighting matter
        assert full_loss == pytest.approx(accum_loss, rel=1e-4)
        assert full_margin == pytest.approx(accum_margin, rel=1e-4, abs=1e-5)
