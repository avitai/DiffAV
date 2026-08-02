"""Offline integration test for the frozen-vs-learnable DPO ablation benchmark.

Exercises the pure ablation core on a synthetic map-carrying scene and a tiny
model — no real WOD data, checkpoint, or GPU-scale run — so the full wiring
(pair building, batched DPO steps for both arms, before/after measurement, and
the verdict) is verified end to end.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.steer_wod_dpo_ablation import (
    _decide_verdict,
    _drift_reason,
    _selection_score,
    ArmResult,
    DpoAblationConfig,
    run_dpo_ablation_on_scenes,
    SceneMeasurement,
)
from flax import nnx

from simulacrax.api.map_conditioned import build_map_conditioned_model, MapConditionedBuildSpec
from simulacrax.core.geometry import RoadEdges
from simulacrax.evaluation.map_conditioned_evaluator import ValidationScene


_RAW_AGENTS = 4
_SELECTED = 3
_FUTURE = 8


def _raw_scene(seed: int = 3) -> dict[str, Any]:
    """A synthetic WOD-Motion scenario dict the tokenizer can read."""
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


def _square_road() -> RoadEdges:
    """A counterclockwise 200x200 m square so most candidates stay feasible."""
    half = 100.0
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


def _scene() -> ValidationScene:
    """A synthetic validation scene: agent 1 is the victim, agent 0 its nearest."""
    reference_pose = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    return ValidationScene(
        scene={key: jnp.asarray(value) for key, value in _raw_scene().items()},
        trajectories=jax.random.normal(jax.random.key(1), (_SELECTED, _FUTURE, 4)),
        agent_rows=jnp.array([0, 1, 2]),
        valid=jnp.ones((_SELECTED, _FUTURE), dtype=jnp.float32),
        tracks_to_predict=jnp.array([False, True, False]),
        reference_pose=reference_pose,
        road_edges=_square_road(),
    )


def _tiny_model() -> Any:
    spec = MapConditionedBuildSpec(
        hidden_dim=16,
        num_heads=2,
        num_blocks=1,
        num_temporal_layers=1,
        num_social_layers=1,
        max_agents=_RAW_AGENTS,
        future_steps=_FUTURE,
        diffusion_steps=4,
        kinematic_weight=0.1,
        collision_weight=0.1,
    )
    return build_map_conditioned_model(spec, rngs=nnx.Rngs(0))


def _tiny_config() -> DpoAblationConfig:
    return DpoAblationConfig(
        checkpoint_dir="unused",
        num_scenes=1,
        num_rollouts=2,
        num_candidates=4,
        num_pairs_per_scene=2,
        num_dpo_steps=1,
        beta=100.0,
        num_samples=2,
        feasibility_threshold=1.0,
        max_agents=_RAW_AGENTS,
        future_steps=_FUTURE,
    )


class TestRunDpoAblationOnScenes:
    """The ablation core runs both arms end to end and reaches a verdict."""

    def test_reports_both_arms_and_a_verdict(self) -> None:
        """Both arms produce finite before/after measurements and a verdict string."""
        report = run_dpo_ablation_on_scenes(
            _tiny_model(), [_scene()], _tiny_config(), key=jax.random.key(0)
        )
        assert report.num_scenes == 1
        assert report.frozen.arm == "frozen"
        assert report.learnable.arm == "learnable"
        assert isinstance(report.verdict, str) and report.verdict
        for arm in (report.frozen, report.learnable):
            for measurement in (arm.before, arm.after):
                assert np.isfinite(measurement.victim_distance)
                assert 0.0 <= measurement.offroad_fraction <= 1.0
                assert 0.0 <= measurement.realism <= 1.0
                assert measurement.map_sensitivity >= 0.0
            assert 0.0 <= arm.reward_accuracy <= 1.0
            assert np.isfinite(arm.reward_margin)

    def test_both_arms_share_the_baseline(self) -> None:
        """The before-measurement is the shared base model, identical across arms."""
        report = run_dpo_ablation_on_scenes(
            _tiny_model(), [_scene()], _tiny_config(), key=jax.random.key(0)
        )
        assert report.frozen.before == report.learnable.before


def _measurement(*, victim: float, realism: float, map_sensitivity: float) -> SceneMeasurement:
    return SceneMeasurement(
        victim_distance=victim,
        offroad_fraction=0.2,
        realism=realism,
        map_sensitivity=map_sensitivity,
    )


def _arm(name: str, before: SceneMeasurement, after: SceneMeasurement) -> ArmResult:
    return ArmResult(
        arm=name,
        before=before,
        after=after,
        reward_accuracy=0.8,
        reward_margin=5.0,
        selected_step=1,
    )


_STEADY_BEFORE = _measurement(victim=11.0, realism=0.63, map_sensitivity=0.42)


class TestDriftGuard:
    """The two-sided over-optimization guard flags map-sensitivity swings and realism drops."""

    def test_map_sensitivity_spike_is_drift(self) -> None:
        """A 3x map-sensitivity spike (the observed frozen-arm case) counts as drift."""
        arm = _arm(
            "frozen", _STEADY_BEFORE, _measurement(victim=12.4, realism=0.47, map_sensitivity=1.31)
        )
        assert "map-sensitivity" in _drift_reason(arm)

    def test_map_sensitivity_collapse_is_drift(self) -> None:
        """A map-sensitivity collapse (map ignored) also counts as drift."""
        arm = _arm(
            "x", _STEADY_BEFORE, _measurement(victim=9.0, realism=0.62, map_sensitivity=0.04)
        )
        assert "map-sensitivity" in _drift_reason(arm)

    def test_in_band_small_realism_drop_is_not_drift(self) -> None:
        """In-band map-sensitivity with a small realism drop is genuine steering."""
        arm = _arm(
            "learn", _STEADY_BEFORE, _measurement(victim=10.6, realism=0.57, map_sensitivity=0.36)
        )
        assert _drift_reason(arm) == ""

    def test_large_realism_drop_is_drift(self) -> None:
        """A large realism drop is drift even with in-band map-sensitivity."""
        arm = _arm(
            "x", _STEADY_BEFORE, _measurement(victim=9.0, realism=0.40, map_sensitivity=0.40)
        )
        assert "realism" in _drift_reason(arm)


class TestDecideVerdict:
    """The verdict requires a positive, drift-free adversariality gain."""

    def test_clean_learnable_gain_wins(self) -> None:
        """A positive, in-band learnable gain beats a regressing frozen arm."""
        learnable = _arm(
            "learnable",
            _STEADY_BEFORE,
            _measurement(victim=10.5, realism=0.58, map_sensitivity=0.37),
        )
        frozen = _arm(
            "frozen", _STEADY_BEFORE, _measurement(victim=12.0, realism=0.55, map_sensitivity=0.40)
        )
        verdict = _decide_verdict(frozen, learnable)
        assert verdict.startswith("learnable") and "without drift" in verdict

    def test_drifted_winner_is_cautioned(self) -> None:
        """A larger gain that came with drift is flagged as over-optimization, not a win."""
        learnable = _arm(
            "learnable", _STEADY_BEFORE, _measurement(victim=8.0, realism=0.60, map_sensitivity=1.5)
        )
        frozen = _arm(
            "frozen", _STEADY_BEFORE, _measurement(victim=10.9, realism=0.62, map_sensitivity=0.41)
        )
        verdict = _decide_verdict(frozen, learnable)
        assert verdict.startswith("caution") and "over-optimization" in verdict

    def test_no_positive_gain_is_inconclusive(self) -> None:
        """If neither arm increased adversariality, the verdict is inconclusive."""
        worse = _measurement(victim=12.0, realism=0.6, map_sensitivity=0.42)
        verdict = _decide_verdict(
            _arm("frozen", _STEADY_BEFORE, worse), _arm("learnable", _STEADY_BEFORE, worse)
        )
        assert verdict.startswith("inconclusive")


class TestSelectionScore:
    """Held-out checkpoint selection: drift-free adversariality gain, drifted disqualified."""

    def test_drift_free_gain_scores_positive(self) -> None:
        """A grounded, more-adversarial checkpoint scores its victim-distance drop."""
        candidate = _measurement(victim=10.0, realism=0.60, map_sensitivity=0.40)
        assert _selection_score(_STEADY_BEFORE, candidate) == pytest.approx(1.0)  # 11.0 - 10.0

    def test_drifted_candidate_is_disqualified(self) -> None:
        """A more-adversarial but drifted checkpoint is disqualified, not selected."""
        drifted = _measurement(victim=8.0, realism=0.60, map_sensitivity=1.5)  # 3.6x spike
        assert _selection_score(_STEADY_BEFORE, drifted) == float("-inf")

    def test_baseline_scores_zero(self) -> None:
        """The un-fine-tuned baseline scores 0 (the floor a trained checkpoint must beat)."""
        assert _selection_score(_STEADY_BEFORE, _STEADY_BEFORE) == pytest.approx(0.0)
