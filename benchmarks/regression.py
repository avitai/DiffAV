#!/usr/bin/env python
"""Performance regression benchmarks for Simulacrax core training paths.

Measures wall-clock throughput of ``TrajectoryTrainer.compute_train_step``
and ``DPOAlignmentTrainer.compute_dpo_step`` using
``calibrax.profiling.TimingCollector``.

On runs with ``--fail-on-regression``, current timings are compared against
the stored baseline and the run exits non-zero if any benchmark regresses
beyond the configured threshold (default: 20%). The baseline at
``temp/benchmark_results.json`` is only overwritten when the gate passes —
a regressing run never installs its own timings as the next baseline.

Usage::

    # Establish a new baseline:
    python -m benchmarks.regression

    # Fail CI if timings regress more than 20%:
    python -m benchmarks.regression --fail-on-regression

    # Custom output path and regression threshold:
    python -m benchmarks.regression --output temp/bench.json --threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from calibrax.profiling import TimingCollector
from flax import nnx

from simulacrax.alignment.dpo_trainer import DPOAlignmentConfig, DPOAlignmentTrainer
from simulacrax.models.trainer import TrainerConfig, TrajectoryTrainer
from simulacrax.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)


logger = logging.getLogger(__name__)

# ── Benchmark model dimensions (small for fast CI) ────────────────────────────
_HIDDEN_DIM = 32
_NUM_BLOCKS = 2
_NUM_TEMPORAL_LAYERS = 1
_NUM_SOCIAL_LAYERS = 1
_NUM_HEADS = 2
_FUTURE_STEPS = 8
_NUM_AGENTS = 4
_CONTEXT_DIM = 16
_STATE_DIM = 4
_BATCH_SIZE = 2
_NUM_TIMESTEPS = 10

# ── Benchmark run parameters ──────────────────────────────────────────────────
_WARMUP_STEPS = 2
_MEASURE_STEPS = 20
_DEFAULT_THRESHOLD = 0.20  # 20% regression tolerance
_DEFAULT_OUTPUT = Path("temp/benchmark_results.json")


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
    """Timing result for a single benchmark.

    Attributes:
        name: Benchmark identifier.
        mean_ms: Mean step latency in milliseconds.
        steps: Number of measured steps.
        timestamp: Unix timestamp when the benchmark was run.
    """

    name: str
    mean_ms: float
    steps: int
    timestamp: float


def _build_diffusion_config() -> TrajectoryDiffusionConfig:
    """Return a small TrajectoryDiffusionConfig for benchmark use."""
    return TrajectoryDiffusionConfig(
        hidden_dim=_HIDDEN_DIM,
        num_blocks=_NUM_BLOCKS,
        num_temporal_layers=_NUM_TEMPORAL_LAYERS,
        num_social_layers=_NUM_SOCIAL_LAYERS,
        num_heads=_NUM_HEADS,
        future_steps=_FUTURE_STEPS,
        num_agents_max=_NUM_AGENTS + 1,
        num_timesteps=_NUM_TIMESTEPS,
        context_dim=_CONTEXT_DIM,
        state_dim=_STATE_DIM,
    )


def _measure_mean_ms(step_fn, key: jax.Array) -> float:
    """Mean per-step latency in milliseconds via calibrax's TimingCollector.

    Runs ``step_fn`` on ``_WARMUP_STEPS + _MEASURE_STEPS`` derived keys;
    warm-up iterations execute (JIT compilation) but are excluded from the
    timing statistics, and each result is synchronized before the timestamp.

    Args:
        step_fn: Callable executing one benchmark step for a given key and
            returning a value with a ``block_until_ready``-capable leading
            element.
        key: Base random key; per-step keys derive via ``fold_in``.

    Returns:
        Mean measured step latency in milliseconds.
    """
    collector = TimingCollector(
        sync_fn=lambda result: result[0].block_until_ready(),
        warmup_iterations=_WARMUP_STEPS,
    )
    keys = [jax.random.fold_in(key, i) for i in range(_WARMUP_STEPS + _MEASURE_STEPS)]
    sample = collector.measure_iteration(iter(keys), process_fn=step_fn)
    return (sum(sample.per_batch_times) / len(sample.per_batch_times)) * 1000.0


def _bench_trajectory_trainer() -> BenchmarkResult:
    """Benchmark ``TrajectoryTrainer.compute_train_step``.

    Returns:
        ``BenchmarkResult`` for the trajectory trainer.
    """
    config = _build_diffusion_config()
    model = TrajectoryDiffusionModel(config, rngs=nnx.Rngs(params=jax.random.key(0)))
    trainer = TrajectoryTrainer(model, TrainerConfig(num_epochs=1, log_interval=50))
    jitted = nnx.jit(trainer.compute_train_step)

    trajectories = jnp.ones((_NUM_AGENTS, _FUTURE_STEPS, _STATE_DIM))
    # The factorized backbone requires exactly one context row per agent.
    scene_ctx = jnp.ones((_NUM_AGENTS, _CONTEXT_DIM))

    def step(step_key: jax.Array):
        return jitted(trainer.model, trainer.optimizer, trajectories, scene_ctx, step_key, 0)

    mean_ms = _measure_mean_ms(step, jax.random.key(1))
    logger.info("TrajectoryTrainer.compute_train_step: %.2f ms/step", mean_ms)
    return BenchmarkResult(
        name="trajectory_trainer_compute_step",
        mean_ms=mean_ms,
        steps=_MEASURE_STEPS,
        timestamp=time.time(),
    )


def _bench_dpo_trainer() -> BenchmarkResult:
    """Benchmark ``DPOAlignmentTrainer.compute_dpo_step``.

    Returns:
        ``BenchmarkResult`` for the DPO alignment trainer.
    """
    config = _build_diffusion_config()
    model = TrajectoryDiffusionModel(config, rngs=nnx.Rngs(params=jax.random.key(0)))
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
    trainer = DPOAlignmentTrainer(model, optimizer, config=DPOAlignmentConfig(reference_free=True))
    jitted = nnx.jit(trainer.compute_dpo_step)

    batch = {
        "chosen": jnp.ones((_BATCH_SIZE, _NUM_AGENTS, _FUTURE_STEPS, _STATE_DIM)),
        "rejected": jnp.zeros((_BATCH_SIZE, _NUM_AGENTS, _FUTURE_STEPS, _STATE_DIM)),
        "scene_contexts": jnp.ones((_BATCH_SIZE, _NUM_AGENTS, _CONTEXT_DIM)),
    }

    def step(step_key: jax.Array):
        return jitted(trainer.model, trainer.optimizer, batch, step_key)

    mean_ms = _measure_mean_ms(step, jax.random.key(2))
    logger.info("DPOAlignmentTrainer.compute_dpo_step: %.2f ms/step", mean_ms)
    return BenchmarkResult(
        name="dpo_trainer_compute_step",
        mean_ms=mean_ms,
        steps=_MEASURE_STEPS,
        timestamp=time.time(),
    )


def run_benchmarks() -> list[BenchmarkResult]:
    """Run all registered benchmarks and return results.

    Returns:
        List of ``BenchmarkResult`` instances, one per benchmark.
    """
    logger.info(
        "Running benchmarks (warmup=%d, measure=%d steps each) ...", _WARMUP_STEPS, _MEASURE_STEPS
    )
    return [
        _bench_trajectory_trainer(),
        _bench_dpo_trainer(),
    ]


def check_regression(
    current: list[BenchmarkResult],
    baseline: list[BenchmarkResult],
    threshold: float,
) -> bool:
    """Compare current results against a baseline for regressions.

    Args:
        current: Results from the current run.
        baseline: Results from the stored baseline.
        threshold: Maximum allowed relative slowdown (e.g. ``0.20`` for 20%).

    Returns:
        ``True`` if all benchmarks are within threshold; ``False`` if any
        regressed beyond it.
    """
    baseline_map = {r.name: r.mean_ms for r in baseline}
    all_pass = True

    for result in current:
        if result.name not in baseline_map:
            logger.warning("No baseline for %r — skipping regression check", result.name)
            continue

        base_ms = baseline_map[result.name]
        ratio = result.mean_ms / base_ms
        change_pct = (ratio - 1.0) * 100.0

        if ratio > 1.0 + threshold:
            logger.error(
                "REGRESSION: %s — %.2f ms vs baseline %.2f ms (+%.1f%% > threshold %.0f%%)",
                result.name,
                result.mean_ms,
                base_ms,
                change_pct,
                threshold * 100,
            )
            all_pass = False
        else:
            logger.info(
                "OK: %s — %.2f ms vs baseline %.2f ms (%+.1f%%)",
                result.name,
                result.mean_ms,
                base_ms,
                change_pct,
            )

    return all_pass


def main(argv: list[str] | None = None) -> int:
    """Entry point for the benchmark runner.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: ``0`` on success, ``1`` on regression failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run Simulacrax performance benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Path to write/read benchmark results JSON.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=False,
        help="Exit non-zero if any benchmark regresses beyond --threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help="Maximum allowed relative slowdown before regression is flagged.",
    )
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    results = run_benchmarks()

    # Load existing baseline (if any) for regression comparison
    baseline: list[BenchmarkResult] | None = None
    if output_path.exists():
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            baseline = [BenchmarkResult(**r) for r in raw]
            logger.info("Loaded baseline from %s", output_path)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Could not load baseline from %s: %s", output_path, exc)

    # Regression check runs BEFORE the baseline is touched: a regressing
    # run must never install its own timings as the next baseline.
    if args.fail_on_regression and baseline is not None:
        passed = check_regression(results, baseline, args.threshold)
        if not passed:
            print("Benchmark regression detected — see logs above.", file=sys.stderr)
            print(f"Baseline at {output_path} left unchanged.", file=sys.stderr)
            return 1

    # Save current results as the new baseline (gate passed or not requested)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": r.name,
            "mean_ms": r.mean_ms,
            "steps": r.steps,
            "timestamp": r.timestamp,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)

    print(f"Benchmarks complete. Results: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
