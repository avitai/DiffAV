#!/usr/bin/env python
"""Train the map-conditioned diffusion model on local WOD Motion data.

Builds a ``MapConditionedTrajectoryModel`` (SceneTokenizer + factorized
backbone) and trains the tokenizer and backbone jointly under the diffusion
plus physics loss on real WOD scenarios, so the denoiser conditions on the map
via cross-attention rather than a hand-rolled ``[x, y, vx, vy]`` context. Each
scenario is ego-normalized once (agents and roadgraph share the frame); the
model tokenizes it per step, gathers each trajectory agent's embedding as the
adaLN anchor, and cross-attends over the fused scene tokens.

The training recipe is the showcase configuration: a warmup-cosine AdamW
schedule, an exponential moving average of the parameters, and periodic
held-out validation (minADE, off-road rate, and kinematic residual on streamed
``val`` scenes) run against the EMA weights, all driven by
:class:`~diffav.api.map_conditioned_trainer.MapConditionedTrainer`.

Usage::

    source ./activate.sh
    python scripts/train_wod.py --steps 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from dotenv import load_dotenv
from flax import nnx

from diffav.api.map_conditioned import (
    build_map_conditioned_model,
    MapConditionedBuildSpec,
    MapConditionedTrajectoryModel,
)
from diffav.api.map_conditioned_trainer import (
    MapConditionedTrainer,
    TrainBatch,
    TrainingConfig,
)
from diffav.api.wod_validation import (
    prepare_scene,
    stream_validation_scenes,
    TOKENIZER_KEYS,
)
from diffav.core.constants import (
    SCENE_BACKBONE_ARCHITECTURE_VERSION,
    WOD_CURRENT_TIME_INDEX,
    WOD_HISTORY_STEPS,
)
from diffav.core.types import DatasetMode
from diffav.data import (
    resolve_wod_tfrecord_path,
)
from diffav.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.evaluation.map_conditioned_evaluator import (
    MapConditionedEvaluator,
    ValidationEvalConfig,
)
from diffav.models.checkpointing import CheckpointConfig, DiffAVCheckpointManager


logger = logging.getLogger("train_wod")


def _build_model(args: argparse.Namespace, rngs: nnx.Rngs) -> MapConditionedTrajectoryModel:
    """Build the coupled map-conditioned model from CLI arguments.

    Delegates to :func:`~diffav.api.map_conditioned.build_map_conditioned_model`
    so the training script and the checkpoint loader construct byte-identical
    architectures; this adapter only maps the CLI namespace onto the build spec.
    """
    spec = MapConditionedBuildSpec(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_temporal_layers=args.num_temporal_layers,
        num_social_layers=args.num_social_layers,
        max_agents=args.max_agents,
        future_steps=args.future_steps,
        diffusion_steps=args.diffusion_steps,
        kinematic_weight=args.kinematic_weight,
        collision_weight=args.collision_weight,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return build_map_conditioned_model(spec, rngs=rngs)


def _load_scenarios(
    source: WODSource, norm: AgentNormalizationOperator, args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream, ego-normalize, and extract usable scenarios into host arrays.

    WOD scenes are roadgraph-heavy (~1.5 MB each), so they are streamed from
    the TFRecords and kept as host ``numpy`` — never pre-stacked onto the
    device — and each training batch is moved to the GPU per step. The scenario
    count is capped by ``--max-scenarios`` to bound host memory.

    Returns ``(batched_scene, trajectories, validity, reference_pose, agent_rows)``
    stacked along a leading scenario axis.
    """
    limit = args.max_scenarios
    scenes: list[dict[str, np.ndarray]] = []
    trajectories: list[np.ndarray] = []
    validities: list[np.ndarray] = []
    reference_poses: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for index, element in enumerate(source):
        if 0 < limit <= index:
            break
        normalized, _, _ = norm.apply(dict(element.data), {}, {})
        prepared = prepare_scene(normalized, args.max_agents, args.future_steps)
        if prepared is None:
            continue
        scene, traj, valid, reference_pose, agent_rows = prepared
        scenes.append(scene)
        trajectories.append(np.asarray(traj))
        validities.append(np.asarray(valid))
        reference_poses.append(np.asarray(reference_pose))
        rows.append(np.asarray(agent_rows))
    if not scenes:
        msg = "No usable scenarios found."
        raise RuntimeError(msg)

    batched_scene = {key: np.stack([s[key] for s in scenes]) for key in TOKENIZER_KEYS}
    return (
        batched_scene,
        np.stack(trajectories),
        np.stack(validities),
        np.stack(reference_poses),
        np.stack(rows),
    )


def _train_batch_stream(
    host: tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    batch_size: int,
    total_steps: int,
    key: jax.Array,
) -> Iterator[TrainBatch]:
    """Yield one random-permutation training batch per step.

    The working set already lives on the device, so each batch is a fast
    on-device gather rather than a per-step host->device copy — which otherwise
    starves the GPU at the small batch sizes a large model forces.
    """
    batched_scene, stacked_traj, stacked_valid, stacked_reference_pose, stacked_rows = host
    num_scenes = stacked_traj.shape[0]
    for step in range(total_steps):
        step_key = jax.random.fold_in(key, step)
        perm = jax.random.permutation(step_key, num_scenes)[:batch_size]
        scene: dict[str, jax.Array] = {name: value[perm] for name, value in batched_scene.items()}
        yield TrainBatch(
            scene=scene,
            trajectories=stacked_traj[perm],
            agent_rows=stacked_rows[perm],
            valid=stacked_valid[perm],
            reference_pose=stacked_reference_pose[perm],
        )


def main(argv: list[str] | None = None) -> int:
    """Train the map-conditioned model on WOD with warmup-cosine, EMA, and val-eval."""
    parser = argparse.ArgumentParser(description="Train the map-conditioned model on WOD data.")
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--max-agents", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=80)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-temporal-layers", type=int, default=2)
    parser.add_argument("--num-social-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rematerialize backbone blocks to save activation memory (bigger batch)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="peak learning rate after warmup"
    )
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument(
        "--kinematic-weight",
        type=float,
        default=0.1,
        help="bicycle-model kinematic physics weight on x-hat-0 (0 disables)",
    )
    parser.add_argument(
        "--collision-weight",
        type=float,
        default=0.1,
        help="pairwise collision physics weight on x-hat-0 (0 disables)",
    )
    parser.add_argument("--split", default="train", help="WOD split to train on")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=8_000,
        help="host-RAM cap on streamed scenarios (0 = all; bounded by RAM)",
    )
    parser.add_argument(
        "--val-scenes", type=int, default=32, help="held-out validation scenes (0 disables)"
    )
    parser.add_argument(
        "--val-rollouts", type=int, default=32, help="rollouts sampled per validation scene"
    )
    parser.add_argument(
        "--val-modes",
        type=int,
        default=6,
        help="representative modes clustered from the rollouts before scoring (minADE_K)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=2_000, help="validation interval in steps"
    )
    parser.add_argument(
        "--offroad-chunk-size",
        type=int,
        default=8,
        help="polyline-chunk size bounding the off-road metric memory",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/wod-physics-showcase")
    )
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, force=True)
    load_dotenv()
    load_dotenv(".env.data")

    source = WODSource(
        WODSourceConfig(
            wod_path=resolve_wod_tfrecord_path(),
            split=args.split,
            mode=DatasetMode.STREAMING,
            max_agents=args.max_agents,
            history_steps=WOD_HISTORY_STEPS,
            future_steps=args.future_steps,
        )
    )
    norm = AgentNormalizationOperator(
        AgentNormalizationConfig(current_step_idx=WOD_CURRENT_TIME_INDEX), rngs=nnx.Rngs(0)
    )

    host_arrays = _load_scenarios(source, norm, args)
    # Move the working set onto the device once so per-step batching is a fast
    # on-device gather rather than a host->device copy each step. The set must
    # fit alongside the model; reduce --max-scenarios if it runs out of memory.
    host: tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array, jax.Array] = (
        {key: jnp.asarray(value) for key, value in host_arrays[0].items()},
        jnp.asarray(host_arrays[1]),
        jnp.asarray(host_arrays[2]),
        jnp.asarray(host_arrays[3]),
        jnp.asarray(host_arrays[4]),
    )
    num_scenes = int(host[1].shape[0])
    val_scenes = stream_validation_scenes(
        norm,
        num_scenes=args.val_scenes,
        max_agents=args.max_agents,
        future_steps=args.future_steps,
    )
    logger.info("Prepared %d train scenarios | %d validation scenes", num_scenes, len(val_scenes))

    model = _build_model(args, nnx.Rngs(0))
    trainer = MapConditionedTrainer(
        model,
        TrainingConfig(
            total_steps=args.steps,
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            ema_decay=args.ema_decay,
            eval_every=args.eval_every,
            gradient_clip=1.0,
        ),
    )
    evaluator = MapConditionedEvaluator(
        ValidationEvalConfig(
            num_rollouts=args.val_rollouts,
            num_modes=args.val_modes,
            offroad_chunk_size=args.offroad_chunk_size,
        )
    )

    def validation_callback(eval_model: MapConditionedTrajectoryModel) -> dict[str, float]:
        result = evaluator.evaluate(eval_model, val_scenes, key=jax.random.key(1))
        return {
            "val_minADE": result.min_ade,
            "val_offroad": result.offroad_rate,
            "val_kinematic": result.kinematic_residual,
        }

    manager_config = CheckpointConfig(
        checkpoint_dir=str(args.checkpoint_dir),
        save_interval_steps=args.checkpoint_every,
        max_to_keep=3,
    )
    # Record the run configuration beside the checkpoints so a restored model can
    # be rebuilt with the exact architecture/normalization it was trained under
    # (the earlier checkpoints carried no config and became unrestorable).
    run_config_path = Path(args.checkpoint_dir) / "run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(
        json.dumps({key: str(value) for key, value in vars(args).items()}, indent=2)
    )
    logger.info("Recorded run config at %s", run_config_path)
    with DiffAVCheckpointManager(manager_config) as manager:

        def on_step(step: int, record: dict[str, Any]) -> None:
            # Validation fires at ``eval_every - 1`` (not a log_every multiple),
            # so log whenever the record carries val metrics, else they are lost.
            has_validation = any(name.startswith("val_") for name in record)
            log_step = step % args.log_every == 0
            checkpoint_step = step > 0 and step % args.checkpoint_every == 0
            if not (log_step or checkpoint_step or has_validation):
                return
            # Realize the on-device loss only here (the loop's periodic sync).
            loss_value = float(record["loss"])
            if log_step or has_validation:
                extra = "".join(
                    f" | {name} {value:.4f}"
                    for name, value in record.items()
                    if name.startswith("val_")
                )
                logger.info("step %6d | loss %.4f%s", step, loss_value, extra)
            if checkpoint_step:
                # Persist the EMA-smoothed weights (what validation reports and
                # deployment uses), not the raw training weights.
                with trainer.ema.swap_in(model):
                    manager.save(
                        model,
                        step=step,
                        loss=loss_value,
                        architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION,
                    )

        history = trainer.fit(
            _train_batch_stream(
                host, batch_size=args.batch_size, total_steps=args.steps, key=jax.random.key(0)
            ),
            key=jax.random.key(0),
            validation_callback=validation_callback if val_scenes else None,
            on_step=on_step,
        )
        final_loss = float(history[-1]["loss"]) if history else 0.0
        with trainer.ema.swap_in(model):
            manager.save(
                model,
                step=args.steps,
                loss=final_loss,
                architecture_version=SCENE_BACKBONE_ARCHITECTURE_VERSION,
            )
    logger.info("Done. Checkpoints in %s", args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
