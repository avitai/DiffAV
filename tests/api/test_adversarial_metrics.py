"""Tests for the adversarial-scenario measurement axes.

Cover the map-integrity primitives — map-zeroing a scene and the map-sensitivity
probe — plus adversary/victim selection, on a synthetic map-carrying scene and a
tiny model.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffav.api.adversarial_metrics import (
    map_sensitivity,
    map_zeroed_scene,
    select_adversary_victim,
)
from diffav.api.map_conditioned import build_map_conditioned_model, MapConditionedBuildSpec
from diffav.core.constants import ROADGRAPH_VALID
from diffav.core.geometry import RoadEdges
from diffav.evaluation.map_conditioned_evaluator import ValidationScene


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


class TestSelectAdversaryVictim:
    def test_picks_nearest_valid_neighbour(self) -> None:
        assert select_adversary_victim(_scene()) == (0, 1)

    def test_returns_none_without_a_tracked_agent(self) -> None:
        scene = dataclasses.replace(_scene(), tracks_to_predict=jnp.array([False, False, False]))
        assert select_adversary_victim(scene) is None


class TestMapZeroedScene:
    def test_zeros_the_roadgraph_validity(self) -> None:
        """The counterfactual marks every road-graph sample invalid; the original is untouched."""
        scene = _scene()
        zeroed = map_zeroed_scene(scene)
        assert not np.any(np.asarray(zeroed.scene[ROADGRAPH_VALID]))
        assert np.any(np.asarray(scene.scene[ROADGRAPH_VALID]))

    def test_preserves_agent_state(self) -> None:
        """Only the map is blinded — the agent trajectories carry through unchanged."""
        scene = _scene()
        zeroed = map_zeroed_scene(scene)
        assert np.array_equal(
            np.asarray(zeroed.scene["state/all/x"]), np.asarray(scene.scene["state/all/x"])
        )


class TestMapSensitivity:
    def test_is_finite_and_non_negative(self) -> None:
        """The normal-vs-map-zeroed divergence is a finite, non-negative displacement."""
        sensitivity = map_sensitivity(
            _tiny_model(), _scene(), num_rollouts=2, key=jax.random.key(0)
        )
        assert np.isfinite(sensitivity)
        assert sensitivity >= 0.0
