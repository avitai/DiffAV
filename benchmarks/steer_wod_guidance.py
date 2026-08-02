#!/usr/bin/env python
"""Adversarial guidance steering on real WOD, certified by the WOSAC metametric.

Demonstrates the Pillar B steering loop end to end on real, map-carrying Waymo
Open Dataset validation scenes with the frozen tier-0 checkpoint. For each
scene one *adversary* agent is steered toward one *victim* (a
``tracks_to_predict`` agent) via test-time x̂₀ guidance
(:func:`~simulacrax.alignment.steering_spine.make_adversarial_guidance`), and
three axes are reported across a guidance-strength sweep:

- **adversariality** — the minimum adversary-to-victim distance over the
  horizon (smaller means a more critical scenario);
- **feasibility** — the adversary's off-road fraction against the scene's real
  road edges (the feasibility signal, non-vacuous now that the map is real);
- **realism** — the faithful WOSAC normalized metametric of the steered rollouts
  against the logged scenario (guided vs. the unguided prior quantifies the
  realism cost of steering).

Steering perturbs only the adversary; every other agent follows the unguided
prior, so realism degrades gracefully rather than collapsing the whole scene.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dotenv import load_dotenv
from flax import nnx

from simulacrax.alignment.steering_spine import make_adversarial_guidance
from simulacrax.api.adversarial_metrics import (
    adversary_offroad_fraction,
    rollout_realism,
    select_adversary_victim,
    victim_min_distance,
)
from simulacrax.api.map_conditioned import (
    load_map_conditioned_from_checkpoint,
    MapConditionedTrajectoryModel,
)
from simulacrax.api.map_conditioned_steering import sample_scene_candidates
from simulacrax.api.wod_validation import stream_validation_scenes
from simulacrax.core.constants import WOD_CURRENT_TIME_INDEX
from simulacrax.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from simulacrax.evaluation.map_conditioned_evaluator import ValidationScene
from simulacrax.models.trajectory_diffusion import GuidanceSpec


logger = logging.getLogger("steer_wod_guidance")


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceDemoConfig:
    """Configuration for the adversarial-guidance demonstration.

    Attributes:
        checkpoint_dir: Directory of the frozen reference checkpoint.
        num_scenes: Held-out validation scenes to stream and score.
        num_rollouts: Rollouts sampled per (scene, guidance strength).
        guidance_scales: Non-zero guidance strengths to sweep; the unguided
            prior (scale 0) is always scored as the baseline.
        offroad_weight: Weight of the adversary off-road penalty in the reward.
        other_collision_weight: Weight of the non-victim collision penalty.
        other_collision_margin: Distance (m) below which the adversary is
            penalized for crowding a non-victim agent.
        guidance_grad_clip: Per-step global-L2-norm clip on the guidance
            gradient — the stability guard that keeps a far-off-road adversary's
            large penalty gradient from driving the sample off-distribution.
        offroad_chunk_size: Polyline-chunk size bounding the WOSAC/feasibility
            road-edge memory.
        max_agents: Agent slots per scene.
        future_steps: Predicted horizon (matches the checkpoint).
        seed: Base random seed.
    """

    checkpoint_dir: str
    num_scenes: int = 8
    num_rollouts: int = 8
    guidance_scales: tuple[float, ...] = (1.0, 5.0)
    offroad_weight: float = 1.0
    other_collision_weight: float = 1.0
    other_collision_margin: float = 2.0
    guidance_grad_clip: float = 1.0
    offroad_chunk_size: int = 8
    max_agents: int = 32
    future_steps: int = 80
    seed: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceCell:
    """Mean scores at one guidance strength across the scored scenes.

    Attributes:
        guidance_scale: The guidance strength (0 is the unguided prior).
        victim_distance: Mean minimum adversary-to-victim distance (m); lower
            is more adversarial.
        offroad_fraction: Mean adversary off-road fraction in [0, 1]; lower is
            more feasible.
        realism: Mean WOSAC normalized metametric in [0, 1]; higher is more
            realistic.
        num_scenes: Scenes contributing to the means.
    """

    guidance_scale: float
    victim_distance: float
    offroad_fraction: float
    realism: float
    num_scenes: int


def _score_scene_at_scale(
    model: MapConditionedTrajectoryModel,
    scene: ValidationScene,
    *,
    adversary_index: int,
    victim_index: int,
    guidance_scale: float,
    config: GuidanceDemoConfig,
    key: jax.Array,
) -> tuple[float, float, float]:
    """Score one scene at one guidance strength: (victim_distance, offroad, realism)."""
    guidance = None
    if guidance_scale != 0.0:
        usable = jnp.any(scene.valid, axis=1)
        reward_fn = make_adversarial_guidance(
            adversary_index=adversary_index,
            victim_index=victim_index,
            reference_pose=scene.reference_pose,
            road_edges=scene.road_edges,
            valid_agents=usable,
            offroad_weight=config.offroad_weight,
            other_collision_weight=config.other_collision_weight,
            other_collision_margin=config.other_collision_margin,
        )
        guidance = GuidanceSpec(
            reward_fn=reward_fn, scale=guidance_scale, grad_clip=config.guidance_grad_clip
        )
    rollouts = sample_scene_candidates(
        model,
        scene,
        num_candidates=config.num_rollouts,
        guidance=guidance,
        key=key,
    )
    return (
        victim_min_distance(rollouts, adversary_index, victim_index),
        adversary_offroad_fraction(rollouts, adversary_index, scene),
        rollout_realism(rollouts, scene, offroad_chunk_size=config.offroad_chunk_size),
    )


def run_guidance_demo_on_scenes(
    model: MapConditionedTrajectoryModel,
    scenes: list[ValidationScene],
    config: GuidanceDemoConfig,
    *,
    key: jax.Array,
) -> list[GuidanceCell]:
    """Score the guidance sweep over pre-loaded scenes (the pure, testable core).

    Scenes without a valid adversary/victim pair are skipped (and logged). The
    unguided prior (scale 0) is always scored first as the baseline.

    Args:
        model: The frozen reference model to steer.
        scenes: Pre-loaded validation scenes.
        config: Demonstration configuration.
        key: Base random key.

    Returns:
        One :class:`GuidanceCell` per guidance strength (0 first).
    """
    scales = (0.0, *config.guidance_scales)
    accumulators: dict[float, list[tuple[float, float, float]]] = {scale: [] for scale in scales}

    for scene_index, scene in enumerate(scenes):
        pair = select_adversary_victim(scene)
        if pair is None:
            logger.info("scene %d: no adversary/victim pair; skipping", scene_index)
            continue
        adversary_index, victim_index = pair
        for scale in scales:
            scale_key = jax.random.fold_in(jax.random.fold_in(key, scene_index), int(scale * 1000))
            scores = _score_scene_at_scale(
                model,
                scene,
                adversary_index=adversary_index,
                victim_index=victim_index,
                guidance_scale=scale,
                config=config,
                key=scale_key,
            )
            accumulators[scale].append(scores)
            logger.info(
                "scene %d scale %.2f: victim_dist=%.2f offroad=%.3f realism=%.3f",
                scene_index,
                scale,
                *scores,
            )

    cells: list[GuidanceCell] = []
    for scale in scales:
        rows = accumulators[scale]
        if not rows:
            continue
        stacked = np.asarray(rows)  # (num_scored, 3)
        means = stacked.mean(axis=0)
        cells.append(
            GuidanceCell(
                guidance_scale=scale,
                victim_distance=float(means[0]),
                offroad_fraction=float(means[1]),
                realism=float(means[2]),
                num_scenes=len(rows),
            )
        )
    return cells


def _write_report(cells: list[GuidanceCell], output_dir: Path) -> Path:
    """Write the sweep table to CSV and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "steer_wod_guidance.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["guidance_scale", "victim_distance", "offroad_fraction", "realism", "num_scenes"]
        )
        for cell in cells:
            writer.writerow(
                [
                    cell.guidance_scale,
                    cell.victim_distance,
                    cell.offroad_fraction,
                    cell.realism,
                    cell.num_scenes,
                ]
            )
    return report_path


def run_guidance_demo(config: GuidanceDemoConfig, output_dir: Path) -> list[GuidanceCell]:
    """Load the checkpoint, stream real scenes, and score the guidance sweep."""
    load_dotenv(".env.data")
    model = load_map_conditioned_from_checkpoint(config.checkpoint_dir, rngs=nnx.Rngs(config.seed))
    norm = AgentNormalizationOperator(
        AgentNormalizationConfig(current_step_idx=WOD_CURRENT_TIME_INDEX), rngs=nnx.Rngs(0)
    )
    scenes = stream_validation_scenes(
        norm,
        num_scenes=config.num_scenes,
        max_agents=config.max_agents,
        future_steps=config.future_steps,
    )
    if not scenes:
        msg = "No validation scenes streamed; check the WOD data path."
        raise RuntimeError(msg)
    logger.info("Scoring %d scenes over scales %s", len(scenes), (0.0, *config.guidance_scales))
    cells = run_guidance_demo_on_scenes(model, scenes, config, key=jax.random.key(config.seed))
    report_path = _write_report(cells, output_dir)
    logger.info("Wrote report to %s", report_path)
    for cell in cells:
        logger.info(
            "scale %.2f | victim_dist %.2f m | offroad %.3f | realism %.3f | scenes %d",
            cell.guidance_scale,
            cell.victim_distance,
            cell.offroad_fraction,
            cell.realism,
            cell.num_scenes,
        )
    return cells


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the adversarial-guidance demonstration."""
    # force=True: JAX/absl configure the root logger at import, so a plain
    # basicConfig would be a no-op and drop the per-scene progress lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Adversarial guidance steering on real WOD.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/wod-physics-tier0")
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument("--output-dir", default="temp/steering")
    args = parser.parse_args(argv)

    config = GuidanceDemoConfig(
        checkpoint_dir=args.checkpoint_dir,
        num_scenes=args.num_scenes,
        num_rollouts=args.num_rollouts,
        guidance_scales=tuple(args.guidance_scales),
    )
    run_guidance_demo(config, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
