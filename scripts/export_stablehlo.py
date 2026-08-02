#!/usr/bin/env python
"""Export the SimulacraxScenarioMiner trajectory diffusion model to StableHLO.

Exports the ``TrajectoryDiffusionModel.predict_noise`` method using
``jax.export`` and serialises the result as a StableHLO ``.mlir`` artifact.

Usage::

    python scripts/export_stablehlo.py
    python scripts/export_stablehlo.py --output-dir /tmp/artifacts

Requirements:
    JAX >= 0.4.24 for ``jax.export`` support.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from simulacrax.models.trajectory_diffusion import TrajectoryDiffusionModel


logger = logging.getLogger(__name__)


# ── Model dimensions used for the export ──────────────────────────────────────
_NUM_AGENTS = 8
_FUTURE_STEPS = 80
_STATE_DIM = 4
# Scene context is strictly one row per agent, so its length equals num_agents.
_CONTEXT_LEN = _NUM_AGENTS
_CONTEXT_DIM = 128
_HIDDEN_DIM = 128
_NUM_BLOCKS = 2
_NUM_TEMPORAL_LAYERS = 2
_NUM_SOCIAL_LAYERS = 1
_NUM_HEADS = 4
_NUM_DIFFUSION_STEPS = 10  # small value — only shapes matter for export

_DEFAULT_OUTPUT_DIR = "artifacts"
_ARTIFACT_FILENAME = "simulacrax_scenario_miner.mlir"


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for this script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Export SimulacraxScenarioMiner to StableHLO .mlir format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write the .mlir artifact into.",
    )
    return parser


def _build_model() -> TrajectoryDiffusionModel:
    """Construct a small default TrajectoryDiffusionModel for export.

    Uses reduced hidden dimensions and diffusion steps so that export
    completes quickly regardless of available compute.

    Returns:
        Freshly initialised ``TrajectoryDiffusionModel``.
    """
    from simulacrax.models.trajectory_diffusion import (
        TrajectoryDiffusionConfig,
        TrajectoryDiffusionModel,
    )

    config = TrajectoryDiffusionConfig(
        hidden_dim=_HIDDEN_DIM,
        num_blocks=_NUM_BLOCKS,
        num_temporal_layers=_NUM_TEMPORAL_LAYERS,
        num_social_layers=_NUM_SOCIAL_LAYERS,
        num_heads=_NUM_HEADS,
        num_agents_max=_NUM_AGENTS,
        future_steps=_FUTURE_STEPS,
        context_dim=_CONTEXT_DIM,
        num_timesteps=_NUM_DIFFUSION_STEPS,
    )
    rngs = nnx.Rngs(params=jax.random.key(0))
    model = TrajectoryDiffusionModel(config, rngs=rngs)
    logger.info(
        "Built TrajectoryDiffusionModel: hidden_dim=%d, num_blocks=%d, num_agents=%d",
        _HIDDEN_DIM,
        _NUM_BLOCKS,
        _NUM_AGENTS,
    )
    return model


def _make_abstract_inputs() -> tuple[
    jax.ShapeDtypeStruct, jax.ShapeDtypeStruct, jax.ShapeDtypeStruct
]:
    """Create abstract (shape + dtype) inputs for ``predict_noise``.

    Shapes:
        noisy_traj: ``(num_agents, future_steps, state_dim)``
        t:          ``()``  scalar float32
        scene_context: ``(num_agents, context_dim)`` — one row per agent

    Returns:
        Tuple of ``(noisy_traj_spec, t_spec, scene_context_spec)``.
    """
    noisy_traj_spec = jax.ShapeDtypeStruct(
        shape=(_NUM_AGENTS, _FUTURE_STEPS, _STATE_DIM),
        dtype=jnp.float32,
    )
    t_spec = jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)
    scene_context_spec = jax.ShapeDtypeStruct(
        shape=(_CONTEXT_LEN, _CONTEXT_DIM),
        dtype=jnp.float32,
    )
    return noisy_traj_spec, t_spec, scene_context_spec


def export_to_stablehlo(output_dir: Path) -> Path:
    """Export ``predict_noise`` to StableHLO and write the .mlir file.

    Uses ``jax.export.export`` to trace the JIT-compiled ``predict_noise``
    method against abstract inputs and serialises the result via
    ``Exported.mlir_module()``.

    Args:
        output_dir: Directory where the ``.mlir`` artifact is written.
            Created automatically if it does not exist.

    Returns:
        Absolute path of the written ``.mlir`` file.

    Raises:
        ImportError: If ``jax.export`` is unavailable (JAX < 0.4.24).
        RuntimeError: If StableHLO serialisation fails.
    """
    try:
        import jax.export  # noqa: F401 — presence check
    except ImportError as exc:
        raise ImportError(
            "jax.export is not available. Upgrade to JAX >= 0.4.24: uv pip install 'jax>=0.4.24'"
        ) from exc

    model = _build_model()

    # Freeze model state via nnx.split so the closure captures a stable pytree
    # of concrete arrays rather than live NNX variables.
    graphdef, model_state = nnx.split(model)

    def predict_noise_fn(
        noisy_traj: jax.Array,
        t: jax.Array,
        scene_context: jax.Array,
    ) -> jax.Array:
        """Stateless wrapper around ``predict_noise`` for JAX export."""
        frozen_model = nnx.merge(graphdef, model_state)
        return frozen_model.predict_noise(noisy_traj, t, scene_context, deterministic=True)

    noisy_traj_spec, t_spec, scene_context_spec = _make_abstract_inputs()

    logger.info("Tracing predict_noise for StableHLO export ...")
    exported = jax.export.export(
        jax.jit(predict_noise_fn),
        platforms=["cuda", "cpu"],
    )(noisy_traj_spec, t_spec, scene_context_spec)

    logger.info("Serialising to StableHLO MLIR ...")
    stablehlo_text = exported.mlir_module()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / _ARTIFACT_FILENAME

    with artifact_path.open("w", encoding="utf-8") as mlir_file:
        mlir_file.write(stablehlo_text)

    return artifact_path.resolve()


def main(argv: list[str] | None = None) -> None:
    """Entry point for the StableHLO export script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    artifact_path = export_to_stablehlo(output_dir)
    print(f"StableHLO export written to: {artifact_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
