#!/usr/bin/env python
"""Frozen-vs-learnable tokenizer DPO ablation on real WOD.

The Pillar B RL/DPO demonstration, framed as an ablation on the project's core
thesis: is the raw-data→result pipeline better *optimized end-to-end* (a
trainable map encoder) or with a frozen scene encoder? Both arms fine-tune the
frozen tier-0 backbone with reference-based map-conditioned Diffusion-DPO toward
the adversarial victim-approach preference, on the *same* reward-ranked pairs,
differing only in whether the tokenizer trains (:func:`map_conditioned_dpo_step`
``freeze_tokenizer``).

Each arm is scored along four axes — adversariality (victim min-distance),
feasibility (adversary off-road), realism (WOSAC metametric) and the
map-sensitivity probe. To avoid crediting DPO over-optimization (the Goodhart
failure where the implicit reward climbs while the real metrics degrade), each
arm's checkpoint is selected by early stopping on the real metrics over HELD-OUT
scenes — never the implicit reward — and drift is guarded two-sided: a
map-sensitivity swing in either direction, or a realism drop, disqualifies a
checkpoint. The verdict compares the arms' selected, drift-free adversariality
gains.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from dotenv import load_dotenv
from flax import nnx

from diffav.alignment.dpo_trainer import DPOAlignmentConfig
from diffav.alignment.steering_spine import AdversarialRewardConfig
from diffav.api.adversarial_metrics import (
    adversary_offroad_fraction,
    map_sensitivity,
    rollout_realism,
    select_adversary_victim,
    victim_min_distance,
)
from diffav.api.map_conditioned import (
    load_map_conditioned_from_checkpoint,
    MapConditionedTrajectoryModel,
)
from diffav.api.map_conditioned_dpo import (
    assemble_dpo_batch,
    build_dpo_arm,
    map_conditioned_dpo_step,
)
from diffav.api.map_conditioned_steering import build_scene_pairs, sample_scene_candidates
from diffav.api.wod_validation import stream_validation_scenes
from diffav.core.constants import WOD_CURRENT_TIME_INDEX
from diffav.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from diffav.evaluation.map_conditioned_evaluator import ValidationScene


logger = logging.getLogger("steer_wod_dpo_ablation")

_MAP_DRIFT_BAND = 2.0
"""Map-sensitivity must stay within ``[1/band, band]`` x baseline. Drift in EITHER direction
signals over-optimization: a collapse means the map is ignored; a large spike means the model
became hyper-reactive to the conditioning to chase the reward. A one-sided (collapse-only) test
would miss the latter (an arm whose map-sensitivity tripled while its real metrics degraded)."""

_REALISM_DROP_LIMIT = 0.1
"""Absolute WOSAC-realism drop above which an arm's steering is treated as drift, not steering."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DpoAblationConfig:
    """Configuration for the frozen-vs-learnable DPO ablation.

    Attributes:
        checkpoint_dir: Directory of the frozen tier-0 checkpoint.
        num_scenes: Held-out validation scenes to stream.
        num_rollouts: Rollouts per scene for the before/after measurements.
        num_candidates: Candidate pool size per scene when building pairs.
        num_pairs_per_scene: Preference pairs emitted per scene.
        num_dpo_steps: Gradient steps per arm over the fixed preference batch.
        num_holdout_scenes: Scenes held out (not used for preference pairs) on which
            the baseline and each arm's checkpoints are measured — so selection
            reflects generalization, not memorization of the training pairs.
        eval_interval: Measure the held-out metric every this many steps; the best
            drift-free checkpoint is selected (early stopping on a real metric, never
            the implicit reward — the standard guard against DPO over-optimization).
        selection_pressure: α tightening the chosen set toward the reward extreme.
        learning_rate: Adam learning rate for the DPO update.
        beta: Diffusion-DPO temperature (MSE-scale; large for trajectory losses).
        num_samples: Timestep/noise draws per DPO log-prob; 1 is the standard
            Diffusion-DPO recipe (a single random timestep per pair per step).
        scene_microbatch: Scenes per backward pass in the DPO update; 1 keeps peak
            memory flat in the number of scenes (gradient accumulation).
        feasibility_threshold: Maximum allowed adversary off-road fraction.
        offroad_weight: Weight of the adversary off-road penalty in the reward.
        other_collision_weight: Weight of the non-victim collision penalty.
        other_collision_margin: Distance (m) below which the adversary is
            penalized for crowding a non-victim agent.
        softmin_temperature: Temperature of the reward's soft-min over time.
        offroad_chunk_size: Polyline-chunk size bounding road-edge memory.
        max_agents: Agent slots per scene.
        future_steps: Predicted horizon (matches the checkpoint).
        seed: Base random seed.
    """

    checkpoint_dir: str
    num_scenes: int = 6
    num_rollouts: int = 4
    num_candidates: int = 8
    num_pairs_per_scene: int = 4
    num_dpo_steps: int = 5
    num_holdout_scenes: int = 3
    eval_interval: int = 4
    selection_pressure: float = 0.75
    learning_rate: float = 1e-6
    beta: float = 5000.0
    num_samples: int = 1
    scene_microbatch: int = 1
    feasibility_threshold: float = 0.25
    offroad_weight: float = 1.0
    other_collision_weight: float = 1.0
    other_collision_margin: float = 2.0
    softmin_temperature: float = 1.0
    offroad_chunk_size: int = 8
    max_agents: int = 32
    future_steps: int = 80
    seed: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneMeasurement:
    """Mean before/after-agnostic scores over the scored scenes.

    Attributes:
        victim_distance: Mean minimum adversary-to-victim distance (m); lower is
            more adversarial.
        offroad_fraction: Mean adversary off-road fraction; lower is more feasible.
        realism: Mean WOSAC normalized metametric; higher is more realistic.
        map_sensitivity: Mean normal-vs-map-zeroed rollout divergence (m); a drop
            after fine-tuning signals corrupted map grounding.
    """

    victim_distance: float
    offroad_fraction: float
    realism: float
    map_sensitivity: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmResult:
    """One ablation arm's before/after measurements and DPO signal.

    Attributes:
        arm: ``"frozen"`` or ``"learnable"``.
        before: Baseline measurement (the frozen base model, shared by both arms).
        after: Measurement of the SELECTED checkpoint — the best drift-free
            held-out checkpoint, which may be the un-fine-tuned baseline (step 0)
            if training only degraded or drifted.
        reward_accuracy: Mean DPO implicit-reward accuracy over the steps.
        reward_margin: DPO implicit-reward margin at the selected step.
        selected_step: Step whose checkpoint was selected (0 = the baseline / no
            fine-tuning was better than any trained checkpoint).
    """

    arm: str
    before: SceneMeasurement
    after: SceneMeasurement
    reward_accuracy: float
    reward_margin: float
    selected_step: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationReport:
    """The full ablation outcome.

    Attributes:
        frozen: The frozen-tokenizer arm result.
        learnable: The learnable-tokenizer arm result.
        verdict: The freeze-vs-learnable decision with its rationale.
        num_scenes: Scenes contributing an adversary/victim pair.
    """

    frozen: ArmResult
    learnable: ArmResult
    verdict: str
    num_scenes: int


def _reward_config(config: DpoAblationConfig) -> AdversarialRewardConfig:
    """Build the adversarial reward weights from the ablation config."""
    return AdversarialRewardConfig(
        offroad_weight=config.offroad_weight,
        other_collision_weight=config.other_collision_weight,
        other_collision_margin=config.other_collision_margin,
        softmin_temperature=config.softmin_temperature,
    )


def _prepare_scenes(scenes: list[ValidationScene]) -> list[tuple[ValidationScene, int, int]]:
    """Keep scenes with a valid adversary/victim pair, tagged with their indices."""
    prepared: list[tuple[ValidationScene, int, int]] = []
    for scene_index, scene in enumerate(scenes):
        pair = select_adversary_victim(scene)
        if pair is None:
            logger.info("scene %d: no adversary/victim pair; skipping", scene_index)
            continue
        prepared.append((scene, pair[0], pair[1]))
    return prepared


def _build_preference_batch(
    model: MapConditionedTrajectoryModel,
    prepared: list[tuple[ValidationScene, int, int]],
    config: DpoAblationConfig,
    *,
    key: jax.Array,
) -> dict[str, object]:
    """Sample and rank one shared preference batch from the base model.

    Both arms train on the same reward-ranked pairs, so the ablation isolates the
    tokenizer toggle.

    Raises:
        RuntimeError: If no scene yields a feasible preference pair.
    """
    reward_config = _reward_config(config)
    entries: list[tuple[ValidationScene, jax.Array, jax.Array]] = []
    for index, (scene, adversary_index, victim_index) in enumerate(prepared):
        pairs = build_scene_pairs(
            model,
            scene,
            adversary_index=adversary_index,
            victim_index=victim_index,
            num_candidates=config.num_candidates,
            selection_pressure=config.selection_pressure,
            num_pairs=config.num_pairs_per_scene,
            config=reward_config,
            feasibility_threshold=config.feasibility_threshold,
            key=jax.random.fold_in(key, index),
        )
        if pairs is None:
            logger.info("scene %d: no feasible preference pair; skipping", index)
            continue
        chosen, rejected, _ = pairs
        entries.append((scene, chosen, rejected))
    if not entries:
        msg = "No feasible preference pairs across the prepared scenes."
        raise RuntimeError(msg)
    return assemble_dpo_batch(entries)


def _measure(
    model: MapConditionedTrajectoryModel,
    prepared: list[tuple[ValidationScene, int, int]],
    config: DpoAblationConfig,
    *,
    key: jax.Array,
) -> SceneMeasurement:
    """Score the four axes over the prepared scenes with unguided rollouts."""
    victim, offroad, realism, sensitivity = [], [], [], []
    for index, (scene, adversary_index, victim_index) in enumerate(prepared):
        scene_key = jax.random.fold_in(key, index)
        rollouts = sample_scene_candidates(
            model, scene, num_candidates=config.num_rollouts, key=scene_key
        )
        victim.append(victim_min_distance(rollouts, adversary_index, victim_index))
        offroad.append(adversary_offroad_fraction(rollouts, adversary_index, scene))
        realism.append(
            rollout_realism(rollouts, scene, offroad_chunk_size=config.offroad_chunk_size)
        )
        sensitivity.append(
            map_sensitivity(
                model,
                scene,
                num_rollouts=config.num_rollouts,
                key=jax.random.fold_in(scene_key, 7),
            )
        )
    return SceneMeasurement(
        victim_distance=float(np.mean(victim)),
        offroad_fraction=float(np.mean(offroad)),
        realism=float(np.mean(realism)),
        map_sensitivity=float(np.mean(sensitivity)),
    )


def _run_arm(
    base_model: MapConditionedTrajectoryModel,
    batch: dict[str, object],
    holdout_scenes: list[tuple[ValidationScene, int, int]],
    config: DpoAblationConfig,
    baseline: SceneMeasurement,
    *,
    freeze_tokenizer: bool,
    key: jax.Array,
) -> ArmResult:
    """Fine-tune one arm, selecting the best drift-free HELD-OUT checkpoint.

    Trains on the shared preference batch and, every ``eval_interval`` steps,
    measures the real metrics on held-out scenes and scores the checkpoint
    (:func:`_selection_score`). The best drift-free checkpoint is kept — early
    stopping on a real metric, never the implicit reward — defaulting to the
    baseline (step 0) if no trained checkpoint beats it without drifting.
    """
    policy, reference, optimizer = build_dpo_arm(base_model, learning_rate=config.learning_rate)
    dpo_config = DPOAlignmentConfig(beta=config.beta)
    arm = "frozen" if freeze_tokenizer else "learnable"
    accuracies: list[float] = []
    # The baseline (un-fine-tuned) checkpoint is always a candidate: gain 0, drift-free.
    best_score, best_after, best_step, best_margin = 0.0, baseline, 0, 0.0
    for step in range(config.num_dpo_steps):
        _, aux = map_conditioned_dpo_step(
            policy,
            reference,
            optimizer,
            batch,
            jax.random.fold_in(key, step),
            config=dpo_config,
            num_samples=config.num_samples,
            freeze_tokenizer=freeze_tokenizer,
            scene_microbatch=config.scene_microbatch,
        )
        accuracies.append(float(aux["reward_accuracy"]))
        margin = float(aux["reward_margin"])
        logger.info(
            "arm %s step %d: reward_acc=%.3f reward_margin=%.4f", arm, step, accuracies[-1], margin
        )
        if step % config.eval_interval != 0 and step != config.num_dpo_steps - 1:
            continue
        candidate = _measure(
            policy, holdout_scenes, config, key=jax.random.fold_in(key, 9973 + step)
        )
        score = _selection_score(baseline, candidate)
        logger.info(
            "arm %s step %d holdout: victim=%.2f offroad=%.3f realism=%.3f map=%.3f score=%.2f",
            arm,
            step,
            candidate.victim_distance,
            candidate.offroad_fraction,
            candidate.realism,
            candidate.map_sensitivity,
            score,
        )
        if score > best_score:
            best_score, best_after, best_step, best_margin = score, candidate, step + 1, margin
    return ArmResult(
        arm=arm,
        before=baseline,
        after=best_after,
        reward_accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        reward_margin=best_margin,
        selected_step=best_step,
    )


def _adversariality_gain(arm: ArmResult) -> float:
    """Drop in victim distance after fine-tuning — positive means more adversarial."""
    return arm.before.victim_distance - arm.after.victim_distance


def _measurement_drift(before: SceneMeasurement, after: SceneMeasurement) -> str:
    """Two-sided over-optimization check between two measurements.

    A checkpoint's adversariality gain is only trustworthy if it did not drift off
    the base model's manifold while chasing the implicit reward (the DPO
    over-optimization / Goodhart failure: reward margin up, real metrics down).
    Drift is flagged when map-sensitivity leaves the ``[1/band, band]`` reference
    band in EITHER direction, or when WOSAC realism drops beyond the limit.
    """
    ratio = after.map_sensitivity / (before.map_sensitivity + 1e-9)
    if ratio < 1.0 / _MAP_DRIFT_BAND or ratio > _MAP_DRIFT_BAND:
        return (
            f"map-sensitivity {before.map_sensitivity:.3f}→"
            f"{after.map_sensitivity:.3f} ({ratio:.1f}x baseline)"
        )
    if before.realism - after.realism > _REALISM_DROP_LIMIT:
        return f"realism {before.realism:.3f}→{after.realism:.3f}"
    return ""


def _drift_reason(arm: ArmResult) -> str:
    """Drift between an arm's baseline and its selected checkpoint (safety net)."""
    return _measurement_drift(arm.before, arm.after)


def _selection_score(baseline: SceneMeasurement, candidate: SceneMeasurement) -> float:
    """Held-out selection score: drift-free adversariality gain, else disqualified.

    The metric checkpoints are selected on — a REAL held-out quantity, never the
    implicit DPO reward (which keeps rising as the policy over-optimizes). A
    drifted candidate is disqualified (``-inf``); an on-manifold candidate scores
    its victim-distance drop, so higher = more adversarial while staying grounded.
    """
    if _measurement_drift(baseline, candidate):
        return float("-inf")
    return baseline.victim_distance - candidate.victim_distance


def _decide_verdict(frozen: ArmResult, learnable: ArmResult) -> str:
    """Turn the two arms' adversariality gains and drift into a verdict.

    Only a genuine *positive* gain (the adversary moved closer to the victim)
    counts as steering; when neither arm achieves that the verdict is honestly
    inconclusive rather than crowning the least-bad arm. The winning arm must
    also not have drifted (two-sided map-sensitivity band + realism), or its gain
    reads as over-optimization, not map-grounded steering.
    """
    frozen_gain, learnable_gain = _adversariality_gain(frozen), _adversariality_gain(learnable)
    if max(frozen_gain, learnable_gain) <= 0.0:
        return (
            "inconclusive — neither arm increased adversariality (victim distance rose: "
            f"frozen {frozen_gain:.2f} m, learnable {learnable_gain:.2f} m); needs more DPO "
            "steps/scenes or reward tuning"
        )

    winner, winner_gain, loser_gain = (
        ("learnable", learnable_gain, frozen_gain)
        if learnable_gain > frozen_gain
        else ("frozen", frozen_gain, learnable_gain)
    )
    winning_arm = learnable if winner == "learnable" else frozen
    drift = _drift_reason(winning_arm)
    if drift:
        return (
            f"caution — the {winner} arm had the larger adversariality gain "
            f"({winner_gain:.2f} m vs {loser_gain:.2f} m) but drifted ({drift}); reads as "
            "over-optimization, not map-grounded steering"
        )
    if winner == "learnable":
        return (
            "learnable — end-to-end optimization wins: larger adversariality gain "
            f"({learnable_gain:.2f} m vs {frozen_gain:.2f} m) without drift"
        )
    return (
        "freeze — the frozen backbone matches or beats end-to-end tuning "
        f"({frozen_gain:.2f} m vs {learnable_gain:.2f} m) without drift; "
        "Occam and stability favor freezing"
    )


def run_dpo_ablation_on_scenes(
    model: MapConditionedTrajectoryModel,
    scenes: list[ValidationScene],
    config: DpoAblationConfig,
    *,
    key: jax.Array,
) -> AblationReport:
    """Run the frozen-vs-learnable ablation over pre-loaded scenes (testable core).

    Args:
        model: The frozen base model to fine-tune (left untouched; each arm clones).
        scenes: Pre-loaded validation scenes.
        config: Ablation configuration.
        key: Base random key.

    Returns:
        The :class:`AblationReport` with both arms and the verdict.

    Raises:
        RuntimeError: If no scene yields an adversary/victim pair.
    """
    prepared = _prepare_scenes(scenes)
    if not prepared:
        msg = "No scene yielded an adversary/victim pair."
        raise RuntimeError(msg)

    # Hold out scenes for selection/measurement so it reflects generalization, not
    # memorization of the training pairs. Leave at least one scene to train on;
    # with too few prepared scenes for a real split, fall back to measuring on the
    # training scenes (degenerate, logged).
    holdout_count = min(config.num_holdout_scenes, max(0, len(prepared) - 1))
    holdout_scenes = prepared[:holdout_count] if holdout_count > 0 else prepared
    train_scenes = prepared[holdout_count:]
    if holdout_count == 0:
        logger.warning(
            "only %d scene(s) with a pair; no held-out split — selection is on training scenes",
            len(prepared),
        )
    logger.info("%d train scene(s), %d held-out scene(s)", len(train_scenes), len(holdout_scenes))

    batch_key, measure_key, frozen_key, learnable_key = jax.random.split(key, 4)
    batch = _build_preference_batch(model, train_scenes, config, key=batch_key)
    baseline = _measure(model, holdout_scenes, config, key=measure_key)
    logger.info(
        "baseline: victim_dist=%.2f offroad=%.3f realism=%.3f map_sens=%.3f",
        baseline.victim_distance,
        baseline.offroad_fraction,
        baseline.realism,
        baseline.map_sensitivity,
    )

    frozen = _run_arm(
        model, batch, holdout_scenes, config, baseline, freeze_tokenizer=True, key=frozen_key
    )
    learnable = _run_arm(
        model, batch, holdout_scenes, config, baseline, freeze_tokenizer=False, key=learnable_key
    )
    verdict = _decide_verdict(frozen, learnable)
    logger.info("verdict: %s", verdict)
    return AblationReport(
        frozen=frozen, learnable=learnable, verdict=verdict, num_scenes=len(prepared)
    )


def _write_report(report: AblationReport, output_dir: Path) -> Path:
    """Write the ablation table to CSV and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "steer_wod_dpo_ablation.csv"
    columns = [
        "arm",
        "victim_distance_before",
        "victim_distance_after",
        "offroad_before",
        "offroad_after",
        "realism_before",
        "realism_after",
        "map_sensitivity_before",
        "map_sensitivity_after",
        "reward_accuracy",
        "reward_margin",
        "selected_step",
    ]
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for arm in (report.frozen, report.learnable):
            writer.writerow(
                [
                    arm.arm,
                    arm.before.victim_distance,
                    arm.after.victim_distance,
                    arm.before.offroad_fraction,
                    arm.after.offroad_fraction,
                    arm.before.realism,
                    arm.after.realism,
                    arm.before.map_sensitivity,
                    arm.after.map_sensitivity,
                    arm.reward_accuracy,
                    arm.reward_margin,
                    arm.selected_step,
                ]
            )
    return report_path


def run_dpo_ablation(config: DpoAblationConfig, output_dir: Path) -> AblationReport:
    """Load the checkpoint, stream real scenes, and run the DPO ablation."""
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
    report = run_dpo_ablation_on_scenes(model, scenes, config, key=jax.random.key(config.seed))
    report_path = _write_report(report, output_dir)
    logger.info("Wrote report to %s", report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the frozen-vs-learnable DPO ablation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description="Frozen-vs-learnable DPO ablation on real WOD.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/wod-physics-tier0")
    parser.add_argument("--num-scenes", type=int, default=6)
    parser.add_argument("--num-dpo-steps", type=int, default=5)
    parser.add_argument(
        "--num-holdout-scenes",
        type=int,
        default=3,
        help="scenes held out for checkpoint selection (not used for training pairs)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=4,
        help="measure the held-out metric every N steps for early-stopping selection",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5000.0,
        help="Diffusion-DPO KL strength; higher = tighter to the reference (DiffusionDPO 5000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="DPO fine-tune LR; keep tiny (DiffusionDPO ~1e-8..2e-5, Rafailov 5e-7) to avoid drift",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=8,
        help="candidate pool per scene; raise to find feasible pairs on an off-road-prone model",
    )
    parser.add_argument(
        "--feasibility-threshold",
        type=float,
        default=0.25,
        help="max adversary off-road fraction for a candidate; raise to keep more scenes",
    )
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument(
        "--num-pairs-per-scene",
        type=int,
        default=4,
        help="preference pairs per scene; the DPO batch is scenes x this, so lower it if the "
        "training step OOMs",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="timestep draws per DPO log-prob; 1 is the standard Diffusion-DPO recipe",
    )
    parser.add_argument(
        "--scene-microbatch",
        type=int,
        default=1,
        help="scenes per backward pass; 1 keeps peak memory flat in scene count (accumulation)",
    )
    parser.add_argument("--output-dir", default="temp/steering")
    args = parser.parse_args(argv)

    config = DpoAblationConfig(
        checkpoint_dir=args.checkpoint_dir,
        num_scenes=args.num_scenes,
        num_dpo_steps=args.num_dpo_steps,
        num_holdout_scenes=args.num_holdout_scenes,
        eval_interval=args.eval_interval,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_candidates=args.num_candidates,
        feasibility_threshold=args.feasibility_threshold,
        num_rollouts=args.num_rollouts,
        num_pairs_per_scene=args.num_pairs_per_scene,
        num_samples=args.num_samples,
        scene_microbatch=args.scene_microbatch,
    )
    run_dpo_ablation(config, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
