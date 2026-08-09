"""Held-out validation evaluator for the map-conditioned trajectory model.

Samples rollouts from a map-conditioned model over prepared validation scenes
and scores them against the logged futures with the metrics that reveal whether
physics training helps: the minimum ADE over K samples, the exact WOSAC
off-road rate (memory-bounded), and the bicycle kinematic residual. Reuses the
shared metrics, geometry, and kinematics primitives; only the sampling seam is
new, and it is a structural :class:`Protocol` so this module does not depend on
the ``api`` layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from diffav.core.geometry import RoadEdges
from diffav.core.types import TrajectoryPrediction
from diffav.evaluation.clustering import select_representative_modes
from diffav.evaluation.metrics import benchmark_min_ade, offroad_rate
from diffav.physics.kinematics import BicycleModelConfig, BicycleModelConstraint


@runtime_checkable
class MapConditionedSampler(Protocol):
    """Consumer-side protocol for the evaluator's model seam.

    ``MapConditionedTrajectoryModel`` satisfies it structurally.
    """

    def sample(
        self,
        scene: Mapping[str, Any],
        agent_rows: jax.Array,
        *,
        reference_pose: jax.Array,
        key: jax.Array,
    ) -> TrajectoryPrediction:
        """Sample trajectories for the selected agents conditioned on the map."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationScene:
    """One prepared held-out validation scene.

    Attributes:
        scene: Raw WOD scenario dict carrying the tokenizer keys.
        trajectories: Logged futures, shape ``(num_agents, future_steps, 4)``.
        agent_rows: Anchor-row indices for the agents to generate, ``(num_agents,)``.
        valid: Per-step validity, shape ``(num_agents, future_steps)``.
        tracks_to_predict: Per-agent benchmark flag, ``(num_agents,)`` bool — the
            agents scored for minADE (the WOMD ``tracks_to_predict`` subset, all
            valid at the current step). Off-road and kinematic metrics still cover
            every valid agent.
        reference_pose: Per-agent current pose, ``(num_agents, 3)``, defining each
            agent's local frame (passed to the sampler so its output returns in
            the shared frame).
        road_edges: Full-resolution per-scene road edges (uncapped).
    """

    scene: Mapping[str, Any]
    trajectories: jax.Array
    agent_rows: jax.Array
    valid: jax.Array
    tracks_to_predict: jax.Array
    reference_pose: jax.Array
    road_edges: RoadEdges


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationEvalConfig:
    """Configuration for the held-out validation evaluation.

    Attributes:
        num_rollouts: Number of rollouts sampled per scene. With ``num_modes``
            set this is the oversample count ``N`` (SOTA draws 32-64); otherwise
            it is the ``K`` scored directly.
        num_modes: When set, reduce the ``num_rollouts`` samples to this many
            density-weighted representative modes before scoring, so the metrics
            are the SOTA oversample-and-cluster ``minADE_K`` rather than a raw
            best-of-K. ``None`` (default) scores all rollouts.
        offroad_chunk_size: Polyline-chunk size bounding the off-road distance
            memory (see :func:`~diffav.evaluation.metrics.offroad_rate`).
    """

    num_rollouts: int = 6
    num_modes: int | None = None
    offroad_chunk_size: int | None = 8


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationMetrics:
    """Aggregated held-out validation metrics.

    Attributes:
        min_ade: Mean over scenes of the per-scene minimum ADE (over K samples,
            averaged across the tracks_to_predict agents).
        offroad_rate: Mean off-road rate over scenes.
        kinematic_residual: Mean bicycle kinematic residual over scenes.
        num_scenes: Number of scenes scored.
    """

    min_ade: float
    offroad_rate: float
    kinematic_residual: float
    num_scenes: int


def _scene_metrics(
    rollouts: jax.Array,
    trajectories: jax.Array,
    valid: jax.Array,
    tracks_to_predict: jax.Array,
    road_edges: RoadEdges,
    bicycle: BicycleModelConstraint,
    chunk_size: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score one scene's rollouts, keeping every metric on-device.

    Returns device scalars (no host sync) so the batched evaluator can ``vmap``
    this over the whole validation set and realize the aggregates in a single
    transfer. The WOMD-style minADE (2 Hz subsample, mean over the 3/5/8 s
    horizons) is taken per agent over the K rollouts, then averaged across the
    ``tracks_to_predict`` agents (all valid at the current step).

    Args:
        rollouts: Sampled rollouts, ``(num_rollouts, num_agents, future_steps, >=2)``.
        trajectories: Logged futures, ``(num_agents, future_steps, >=2)``.
        valid: Per-step validity, ``(num_agents, future_steps)``.
        tracks_to_predict: Per-agent minADE scoring mask, ``(num_agents,)`` bool.
        road_edges: Per-scene road edges.
        bicycle: Bicycle kinematic constraint.
        chunk_size: Polyline-chunk size bounding the off-road distance memory.

    Returns:
        ``(min_ade, offroad_rate, kinematic_residual)`` as device scalars.
    """
    per_agent_min_ade = jax.vmap(benchmark_min_ade, in_axes=(1, 0, 0))(
        rollouts[..., :2], trajectories[..., :2], valid
    )
    # minADE is a marginal motion-prediction metric scored over the benchmark's
    # tracks_to_predict agents (all valid at the current step); the off-road and
    # kinematic residuals stay realism proxies over every valid agent.
    score_mask = jnp.any(valid, axis=-1) & tracks_to_predict
    scene_min_ade = jnp.sum(per_agent_min_ade * score_mask) / jnp.maximum(jnp.sum(score_mask), 1.0)
    scene_offroad = offroad_rate(rollouts, road_edges, valid, chunk_size=chunk_size)
    scene_kinematic = jnp.mean(
        jax.vmap(lambda rollout: bicycle.compute_residuals(rollout, valid))(rollouts)
    )
    return scene_min_ade, scene_offroad, scene_kinematic


class MapConditionedEvaluator:
    """Scores a map-conditioned model's rollouts against logged validation scenes."""

    def __init__(self, config: ValidationEvalConfig | None = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration. Uses defaults if ``None``.
        """
        self._config = config or ValidationEvalConfig()
        self._bicycle = BicycleModelConstraint(BicycleModelConfig())

    def evaluate_scene(
        self,
        model: MapConditionedSampler,
        scene: ValidationScene,
        *,
        key: jax.Array,
    ) -> tuple[float, float, float]:
        """Sample K rollouts and score one scene.

        Args:
            model: Map-conditioned sampler.
            scene: Prepared validation scene.
            key: JAX random key; split into one per rollout.

        Returns:
            ``(min_ade, offroad_rate, kinematic_residual)`` for the scene.
        """
        keys = jax.random.split(key, self._config.num_rollouts)
        rollouts = jax.vmap(
            lambda rollout_key: (
                model.sample(
                    scene.scene,
                    scene.agent_rows,
                    reference_pose=scene.reference_pose,
                    key=rollout_key,
                ).trajectories
            )
        )(keys)  # (num_rollouts, num_agents, future_steps, 4)
        if self._config.num_modes is not None:
            rollouts = select_representative_modes(rollouts, self._config.num_modes)
        scene_min_ade, scene_offroad, scene_kinematic = _scene_metrics(
            rollouts,
            scene.trajectories,
            scene.valid,
            scene.tracks_to_predict,
            scene.road_edges,
            self._bicycle,
            self._config.offroad_chunk_size,
        )
        return float(scene_min_ade), float(scene_offroad), float(scene_kinematic)

    def evaluate(
        self,
        model: MapConditionedSampler,
        scenes: Iterable[ValidationScene],
        *,
        key: jax.Array,
    ) -> ValidationMetrics:
        """Aggregate per-scene metrics over the validation set in one device pass.

        The scenes are stacked along a leading axis (this requires uniform
        shapes, including fixed-shape road edges) so the ``K`` rollouts sample in
        a single ``num_scenes * K``-batched reverse-diffusion pass rather than a
        per-scene Python loop, and every metric reduces on-device with a single
        host transfer at the end. This keeps the accelerator saturated during
        evaluation instead of stalling on a per-scene host sync each iteration.

        Args:
            model: Map-conditioned sampler.
            scenes: Iterable of prepared validation scenes with uniform shapes.
            key: JAX random key; split into one per (scene, rollout).

        Returns:
            The scene-averaged :class:`ValidationMetrics`.
        """
        scene_list = list(scenes)
        num_scenes = len(scene_list)
        if num_scenes == 0:
            return ValidationMetrics(
                min_ade=0.0, offroad_rate=0.0, kinematic_residual=0.0, num_scenes=0
            )

        stack = lambda *arrays: jnp.stack(arrays)  # noqa: E731
        scene_batch = jax.tree_util.tree_map(stack, *[dict(scene.scene) for scene in scene_list])
        road_edges = jax.tree_util.tree_map(stack, *[scene.road_edges for scene in scene_list])
        trajectories = jnp.stack([scene.trajectories for scene in scene_list])
        agent_rows = jnp.stack([scene.agent_rows for scene in scene_list])
        valid = jnp.stack([scene.valid for scene in scene_list])
        tracks_to_predict = jnp.stack([scene.tracks_to_predict for scene in scene_list])
        reference_pose = jnp.stack([scene.reference_pose for scene in scene_list])

        num_rollouts = self._config.num_rollouts
        keys = jax.random.split(key, num_scenes * num_rollouts).reshape(num_scenes, num_rollouts)

        def sample_scene(
            one_scene: Mapping[str, Any],
            rows: jax.Array,
            pose: jax.Array,
            rollout_keys: jax.Array,
        ) -> jax.Array:
            return jax.vmap(
                lambda rollout_key: (
                    model.sample(one_scene, rows, reference_pose=pose, key=rollout_key).trajectories
                )
            )(rollout_keys)

        rollouts = jax.vmap(sample_scene)(scene_batch, agent_rows, reference_pose, keys)
        if self._config.num_modes is not None:
            num_modes = self._config.num_modes
            rollouts = jax.vmap(
                lambda scene_rollouts: select_representative_modes(scene_rollouts, num_modes)
            )(rollouts)

        bicycle = self._bicycle
        chunk_size = self._config.offroad_chunk_size

        def score(
            scene_rollouts: jax.Array,
            scene_traj: jax.Array,
            scene_valid: jax.Array,
            scene_tracks: jax.Array,
            scene_edges: RoadEdges,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _scene_metrics(
                scene_rollouts,
                scene_traj,
                scene_valid,
                scene_tracks,
                scene_edges,
                bicycle,
                chunk_size,
            )

        min_ades, offroads, kinematics = jax.vmap(score)(
            rollouts, trajectories, valid, tracks_to_predict, road_edges
        )

        means = jax.device_get(
            jnp.stack([jnp.mean(min_ades), jnp.mean(offroads), jnp.mean(kinematics)])
        )
        return ValidationMetrics(
            min_ade=float(means[0]),
            offroad_rate=float(means[1]),
            kinematic_residual=float(means[2]),
            num_scenes=num_scenes,
        )
