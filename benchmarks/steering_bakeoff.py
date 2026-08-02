"""Steering bake-off: compare steering strategies on a fixed probe grid.

Runs every configured (strategy, steering strength) cell on the same probe
scenes with deterministic keys and scores three axes:

- **adversariality** — mean physics-violation loss of the steered scene
  trajectories themselves (scene-intrinsic violation intensity; the
  planner-specific severity variant lives in
  ``ScenarioMiner.adversarial_search``).
- **realism** — mean normalized WOSAC metametric of each steered rollout
  scored under the unsteered model's rollout distribution (how typical the
  steered behaviour remains).
- **feasibility** — mean off-road fraction of the steered trajectories
  against the probe road edges.

Results are written as CSV + HTML via calibrax's ``PublicationGenerator``
under ``<output_dir>/steering/`` and returned as ``BakeoffCell`` rows. Every
cell derives its keys as ``fold_in(base_key, cell_index)``, so a report is
reproducible from one seed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric, Point, Run
from calibrax.exporters import PublicationGenerator

from simulacrax.alignment.scenario_steering import ScenarioSteeringConfig, SteeringStrategy
from simulacrax.alignment.steering_spine import candidate_offroad_fractions
from simulacrax.api.config import MinerConfig, Scenario
from simulacrax.api.scenario_miner import create_scenario_miner
from simulacrax.core.geometry import RoadEdges
from simulacrax.evaluation.wosac_metametric import (
    compute_metametric_features,
    WosacMetametric,
)
from simulacrax.physics.losses import SimulacraxPhysicsLoss


logger = logging.getLogger(__name__)

_PROBE_ROAD_HALF_WIDTH_M = 100.0


@dataclass(frozen=True, slots=True, kw_only=True)
class BakeoffConfig:
    """Configuration for the steering bake-off grid.

    Attributes:
        strategies: Steering strategies to compare.
        strengths: Steering strengths swept per strategy.
        target_scenario: Scenario label steered toward.
        num_rollouts: Steered rollouts per cell (also the unsteered
            reference count).
        num_candidates: Spine candidates for the training strategies.
        num_steering_steps: Gradient steps for the training strategies.
        miner_config: SDK configuration (model size, diffusion steps).
        seed: Base random seed; every cell folds its index into it.
        output_dir: Report root; tables land in ``<output_dir>/steering/``.
    """

    strategies: tuple[SteeringStrategy, ...] = (
        SteeringStrategy.RANKED_DPO,
        SteeringStrategy.WEIGHT_SOUP,
        SteeringStrategy.GUIDANCE,
    )
    strengths: tuple[float, ...] = (0.1, 0.3, 0.5, 0.8)
    target_scenario: str = "forward"
    num_rollouts: int = 4
    num_candidates: int = 8
    num_steering_steps: int = 1
    miner_config: MinerConfig = MinerConfig(
        max_agents=8, prediction_horizon=20, context_dim=64, num_diffusion_steps=10
    )
    seed: int = 0
    output_dir: Path = Path("benchmarks/output")


@dataclass(frozen=True, slots=True, kw_only=True)
class BakeoffCell:
    """Scores for one (strategy, strength) grid cell.

    Attributes:
        strategy: Steering strategy of the cell.
        strength: Steering strength of the cell.
        adversariality: Mean physics-violation loss of the steered scenes.
        realism: Mean normalized metametric of steered rollouts under the
            unsteered distribution.
        feasibility: Mean off-road fraction (lower is more feasible).
    """

    strategy: SteeringStrategy
    strength: float
    adversariality: float
    realism: float
    feasibility: float


def _probe_road_edges() -> RoadEdges:
    """Large counterclockwise square as the probe drivable area."""
    half = _PROBE_ROAD_HALF_WIDTH_M
    square = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.0],
        ]
    )
    return RoadEdges.from_polylines([square])


def _stack_rollouts(scenarios: list[Scenario]) -> jax.Array:
    """Stack scenario predictions into a ``(K, N, T, 4)`` rollout tensor."""
    return jnp.stack([s.predictions.trajectories for s in scenarios])


def _score_cell(
    steered: jax.Array,
    reference: jax.Array,
    road_edges: RoadEdges,
    physics_loss: SimulacraxPhysicsLoss,
    metametric: WosacMetametric,
) -> tuple[float, float, float]:
    """Score one cell's steered rollouts on the three bake-off axes."""
    num_rollouts, num_agents, num_steps = steered.shape[:3]
    valid = jnp.ones((num_agents, num_steps), dtype=bool)

    losses = [
        physics_loss.compute_loss(steered[k], 0, road_edges=road_edges)[0]
        for k in range(num_rollouts)
    ]
    adversariality = float(jnp.mean(jnp.stack(losses)))

    sim_features = compute_metametric_features(reference, valid, road_edges)
    realism_scores = [
        metametric.compute(
            compute_metametric_features(steered[k : k + 1], valid, road_edges),
            sim_features,
        ).normalized_metametric
        for k in range(num_rollouts)
    ]
    realism = float(jnp.mean(jnp.stack(realism_scores)))

    feasibility = float(jnp.mean(candidate_offroad_fractions(steered, road_edges)))
    return adversariality, realism, feasibility


def run_steering_bakeoff(config: BakeoffConfig) -> list[BakeoffCell]:
    """Run the full bake-off grid and write the calibrax report.

    Args:
        config: Grid configuration.

    Returns:
        One ``BakeoffCell`` per (strategy, strength) pair, in grid order.
    """
    miner = create_scenario_miner(config.miner_config)
    road_edges = _probe_road_edges()
    physics_loss = SimulacraxPhysicsLoss()
    metametric = WosacMetametric()
    base_key = jax.random.key(config.seed)

    reference = _stack_rollouts(
        miner.generate(
            config.target_scenario,
            "medium",
            count=config.num_rollouts,
            key=jax.random.fold_in(base_key, 0),
        )
    )

    cells: list[BakeoffCell] = []
    for cell_index, (strategy, strength) in enumerate(
        (s, w) for s in config.strategies for w in config.strengths
    ):
        steering_config = ScenarioSteeringConfig(
            target_scenario=config.target_scenario,
            strategy=strategy,
            steering_strength=strength,
            num_candidates=config.num_candidates,
            num_steering_steps=config.num_steering_steps,
        )
        steered_scenarios = miner.steer(
            config.target_scenario,
            "medium",
            steering_config=steering_config,
            count=config.num_rollouts,
            key=jax.random.fold_in(base_key, cell_index + 1),
        )
        adversariality, realism, feasibility = _score_cell(
            _stack_rollouts(steered_scenarios),
            reference,
            road_edges,
            physics_loss,
            metametric,
        )
        cells.append(
            BakeoffCell(
                strategy=strategy,
                strength=strength,
                adversariality=adversariality,
                realism=realism,
                feasibility=feasibility,
            )
        )
        logger.info(
            "bakeoff cell %s @ %.2f: adversariality=%.4f realism=%.4f feasibility=%.4f",
            strategy,
            strength,
            adversariality,
            realism,
            feasibility,
        )

    _write_report(cells, config.output_dir)
    return cells


def _write_report(cells: list[BakeoffCell], output_dir: Path) -> Path:
    """Write the bake-off table as CSV + HTML via calibrax."""
    points = tuple(
        Point(
            name=f"{cell.strategy.value}@{cell.strength:.2f}",
            scenario="steering_bakeoff",
            tags={"cell": f"{cell.strategy.value}@{cell.strength:.2f}"},
            metrics={
                "adversariality": Metric(value=cell.adversariality),
                "realism": Metric(value=cell.realism),
                "feasibility": Metric(value=cell.feasibility),
                "strength": Metric(value=cell.strength),
            },
        )
        for cell in cells
    )
    run = Run(points=points)
    report_dir = output_dir / "steering"
    report_dir.mkdir(parents=True, exist_ok=True)
    generator = PublicationGenerator(report_dir)
    generator.generate_table(run, output_format="csv", group_by_tag="cell")
    generator.generate_table(run, output_format="html", group_by_tag="cell")
    logger.info("Steering bake-off report written to %s.", report_dir)
    return report_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the bake-off grid and write the report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Run the steering bake-off grid.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[member.value for member in SteeringStrategy],
        choices=[member.value for member in SteeringStrategy],
        help="Steering strategies to compare.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.8],
        help="Steering strengths swept per strategy.",
    )
    parser.add_argument("--rollouts", type=int, default=4, help="Rollouts per cell.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/output"),
        help="Report root directory.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = BakeoffConfig(
        strategies=tuple(SteeringStrategy(s) for s in args.strategies),
        strengths=tuple(args.strengths),
        num_rollouts=args.rollouts,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    cells = run_steering_bakeoff(config)
    print(f"Bake-off complete: {len(cells)} cells -> {args.output_dir / 'steering'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
