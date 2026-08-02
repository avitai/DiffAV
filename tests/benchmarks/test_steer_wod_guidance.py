"""Offline integration tests for the adversarial-guidance WOD benchmark.

Exercise the pure scoring core on a synthetic map-carrying ``ValidationScene``
and a tiny model — no real WOD data or checkpoint — so victim/adversary
selection and the three-axis sweep are verified without the GPU-scale run.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.steer_wod_guidance import (
    GuidanceDemoConfig,
    run_guidance_demo_on_scenes,
    select_adversary_victim,
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
    """A counterclockwise 40x40 m square centred at the origin."""
    half = 20.0
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
    reference_pose = jnp.array(
        [
            [1.0, 0.0, 0.0],  # agent 0 — nearest to the victim
            [0.0, 0.0, 0.0],  # agent 1 — victim (tracks_to_predict)
            [10.0, 0.0, 0.0],  # agent 2 — farther
        ]
    )
    return ValidationScene(
        scene={key: jnp.asarray(value) for key, value in _raw_scene().items()},
        trajectories=jax.random.normal(jax.random.key(1), (_SELECTED, _FUTURE, 4)),
        agent_rows=jnp.array([0, 1, 2]),
        # Real streamed scenes carry a float32 validity mask (0.0/1.0), not bool.
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


class TestSelectAdversaryVictim:
    """Victim is the tracks_to_predict agent; adversary is its nearest neighbour."""

    def test_picks_nearest_valid_neighbour(self) -> None:
        """Agent 0 (1 m away) is chosen over agent 2 (10 m away)."""
        assert select_adversary_victim(_scene()) == (0, 1)

    def test_returns_none_without_a_tracked_agent(self) -> None:
        """A scene with no valid tracks_to_predict agent yields no pair."""
        import dataclasses

        scene = dataclasses.replace(_scene(), tracks_to_predict=jnp.array([False, False, False]))
        assert select_adversary_victim(scene) is None


class TestRunGuidanceDemoOnScenes:
    """The sweep produces finite, in-range scores for every guidance strength."""

    def test_sweep_scores_all_scales(self) -> None:
        """Cells cover the baseline (0) plus each requested scale, all finite."""
        config = GuidanceDemoConfig(
            checkpoint_dir="unused",
            num_scenes=1,
            num_rollouts=4,
            guidance_scales=(5.0,),
        )
        cells = run_guidance_demo_on_scenes(
            _tiny_model(), [_scene()], config, key=jax.random.key(0)
        )
        assert [cell.guidance_scale for cell in cells] == [0.0, 5.0]
        for cell in cells:
            assert cell.num_scenes == 1
            assert np.isfinite(cell.victim_distance)
            assert 0.0 <= cell.offroad_fraction <= 1.0
            assert 0.0 <= cell.realism <= 1.0

    def test_skips_scenes_without_a_pair(self) -> None:
        """A scene with no adversary/victim pair contributes no scores."""
        import dataclasses

        untracked = dataclasses.replace(
            _scene(), tracks_to_predict=jnp.array([False, False, False])
        )
        config = GuidanceDemoConfig(checkpoint_dir="unused", num_scenes=1, num_rollouts=4)
        cells = run_guidance_demo_on_scenes(
            _tiny_model(), [untracked], config, key=jax.random.key(0)
        )
        assert cells == []
