"""Measurement axes for adversarial scenarios over map-conditioned scenes.

The shared, library-grade metrics behind the Pillar B demonstrations: pick an
adversary/victim pair, then score a set of rollouts along the axes that define a
useful safety-critical scenario — how close the adversary comes to the victim
(adversariality), whether it stays drivable (feasibility), how realistic the
rollouts remain against the logged scenario (the WOSAC metametric), and how much
the model still depends on the map (the map-sensitivity probe that guards the
freeze-vs-learnable ablation against reward-hacking).

Both the guidance sweep and the DPO ablation import these so the two
demonstrations report the same quantities computed the same way.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from simulacrax.alignment.steering_spine import candidate_offroad_fractions
from simulacrax.api.map_conditioned import MapConditionedTrajectoryModel
from simulacrax.api.map_conditioned_steering import sample_scene_candidates
from simulacrax.core.constants import ROADGRAPH_VALID
from simulacrax.evaluation.map_conditioned_evaluator import ValidationScene
from simulacrax.evaluation.wosac_metametric import compute_metametric_features, WosacMetametric


def select_adversary_victim(scene: ValidationScene) -> tuple[int, int] | None:
    """Choose the victim and its nearest adversary for a scene.

    The victim is the first ``tracks_to_predict`` agent with a valid future; the
    adversary is the nearest other agent (by current-pose distance) that also has
    a valid future.

    Args:
        scene: The validation scene.

    Returns:
        ``(adversary_index, victim_index)`` or ``None`` when no such pair exists.
    """
    usable = np.asarray(jnp.any(scene.valid.astype(bool), axis=1))
    tracks = np.asarray(scene.tracks_to_predict) & usable
    if not bool(tracks.any()):
        return None
    victim_index = int(np.argmax(tracks))
    positions = np.asarray(scene.reference_pose)[:, :2]
    distances = np.linalg.norm(positions - positions[victim_index], axis=1)
    distances[victim_index] = np.inf
    distances[~usable] = np.inf
    if not np.isfinite(distances).any():
        return None
    return int(np.argmin(distances)), victim_index


def victim_min_distance(rollouts: jax.Array, adversary_index: int, victim_index: int) -> float:
    """Mean-over-rollouts minimum-over-time adversary-to-victim distance (m)."""
    adversary = rollouts[:, adversary_index, :, :2]
    victim = rollouts[:, victim_index, :, :2]
    per_step = jnp.linalg.norm(adversary - victim, axis=-1)
    return float(jnp.mean(jnp.min(per_step, axis=1)))


def adversary_offroad_fraction(
    rollouts: jax.Array, adversary_index: int, scene: ValidationScene
) -> float:
    """Mean adversary off-road fraction against the scene's real road edges."""
    adversary = rollouts[:, adversary_index : adversary_index + 1, :, :]
    fractions = candidate_offroad_fractions(adversary, scene.road_edges)
    return float(jnp.mean(fractions))


def rollout_realism(
    rollouts: jax.Array, scene: ValidationScene, *, offroad_chunk_size: int
) -> float:
    """WOSAC normalized metametric of the rollouts against the logged scenario."""
    valid = scene.valid.astype(bool)
    log_features = compute_metametric_features(
        scene.trajectories[jnp.newaxis],
        valid,
        scene.road_edges,
        offroad_chunk_size=offroad_chunk_size,
    )
    sim_features = compute_metametric_features(
        rollouts, valid, scene.road_edges, offroad_chunk_size=offroad_chunk_size
    )
    return WosacMetametric().compute(log_features, sim_features).normalized_metametric


def map_zeroed_scene(scene: ValidationScene) -> ValidationScene:
    """Return a copy of ``scene`` whose road-graph is marked entirely invalid.

    Zeroing ``roadgraph_samples/valid`` masks every map token out of the
    tokenizer, so the diffuser conditions on the agents alone — the "no map"
    counterfactual for the map-sensitivity probe.

    Args:
        scene: The validation scene to blind to the map.

    Returns:
        A new :class:`ValidationScene` sharing everything but a map-zeroed scene
        dict.
    """
    zeroed = dict(scene.scene)
    zeroed[ROADGRAPH_VALID] = np.zeros_like(np.asarray(zeroed[ROADGRAPH_VALID]))
    return replace(scene, scene=zeroed)


def map_sensitivity(
    model: MapConditionedTrajectoryModel,
    scene: ValidationScene,
    *,
    num_rollouts: int,
    key: jax.Array,
) -> float:
    """Divergence between normal and map-zeroed rollouts — the map-integrity guard.

    Samples the same rollouts (shared keys) from the real scene and from its
    map-zeroed counterfactual and returns their mean per-position displacement. A
    model that genuinely conditions on the map diverges when the map is removed; a
    collapse toward zero after fine-tuning signals the map grounding was corrupted
    to satisfy the adversarial reward (reward-hacking).

    Args:
        model: The map-conditioned model.
        scene: The validation scene.
        num_rollouts: Rollouts to average over.
        key: JAX random key (shared across both scenes so only the map differs).

    Returns:
        Mean per-position displacement (m) between normal and map-zeroed rollouts.
    """
    normal = sample_scene_candidates(model, scene, num_candidates=num_rollouts, key=key)
    zeroed = sample_scene_candidates(
        model, map_zeroed_scene(scene), num_candidates=num_rollouts, key=key
    )
    displacement = jnp.linalg.norm(normal[..., :2] - zeroed[..., :2], axis=-1)
    return float(jnp.mean(displacement))
