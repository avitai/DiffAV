"""EvaluationRunner: orchestrates model sampling and WOD metric computation.

Bridges the TrajectoryDiffusionModel output format with the WOD-compatible
MotionMetrics and SimAgentMetrics evaluators, accumulating per-batch scores
into a single aggregated MetricsReport.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from simulacrax.core.constants import SCENE_CONTEXT
from simulacrax.core.types import MetricsReport, TrajectoryPrediction
from simulacrax.evaluation.metrics import MotionMetrics, SimAgentMetrics


logger = logging.getLogger(__name__)


@runtime_checkable
class TrajectorySampler(Protocol):
    """Consumer-side protocol for the runner's model seam.

    Anything with a ``sample`` method of this shape can be evaluated —
    ``TrajectoryDiffusionModel`` satisfies it structurally.
    """

    def sample(
        self,
        scene_context: jax.Array,
        *,
        key: jax.Array,
    ) -> TrajectoryPrediction:
        """Sample trajectory predictions for one scene.

        Args:
            scene_context: Per-agent scene context, shape
                ``(num_agents, context_dim)`` — one row per agent. The agent
                count is inferred from ``scene_context.shape[0]``.
            key: JAX random key.

        Returns:
            ``TrajectoryPrediction`` with one agent per context row.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRunner:
    """Orchestrates model evaluation over a sequence of WOD batches.

    Runs ``model.sample()`` per batch, wraps predictions into WOD metric
    input shapes, and accumulates metric averages into a ``MetricsReport``.

    Attributes:
        motion_metrics: Configured MotionMetrics evaluator.
        sim_agent_metrics: Optional sim-agent realism evaluator.
    """

    motion_metrics: MotionMetrics
    sim_agent_metrics: SimAgentMetrics | None = None

    def run(
        self,
        model: TrajectorySampler,
        batches: Iterable[dict[str, Any]],
    ) -> MetricsReport:
        """Run evaluation over all batches and return aggregated metrics.

        Each batch dict must contain:

        - ``"scene_context"``: JAX array ``(A, context_dim)`` — one row per agent
        - ``"ground_truth"``: JAX array ``(A, TG, 7)``
        - ``"ground_truth_is_valid"``: JAX array ``(A, TG)``
        - ``"object_type"``: JAX array ``(A,)`` — 0=vehicle, 1=ped, 2=cyclist
        - ``"key"``: JAX random key

        Optionally for sim-agent metrics:

        - ``"road_edge_distances"``: JAX array ``(A, T)`` — signed distance
          to the nearest road edge, positive off-road

        Args:
            model: Model with a ``.sample(scene_context, *, key)`` method
                returning a ``TrajectoryPrediction``.
            batches: Iterable of batch dicts.

        Returns:
            ``MetricsReport`` with aggregated motion and sim-agent metrics.
        """
        accumulated: dict[str, list[float]] = {}
        kinematic_summary: dict[str, list[float]] = {}
        n_batches = 0

        for batch in batches:
            motion, kinematic = self._process_batch(model, batch)
            for key, val in motion.items():
                accumulated.setdefault(key, []).append(val)
            for key, val in kinematic.items():
                kinematic_summary.setdefault(key, []).append(val)
            n_batches += 1

        if n_batches == 0:
            logger.warning("EvaluationRunner received empty batches; returning empty report.")

        metric_values = {k: sum(vs) / len(vs) for k, vs in accumulated.items()}
        kin_summary = {k: sum(vs) / len(vs) for k, vs in kinematic_summary.items()}

        return MetricsReport(
            metric_values=metric_values,
            kinematic_violation_summary=kin_summary,
        )

    def _process_batch(
        self, model: TrajectorySampler, batch: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Sample from model and compute metrics for one batch.

        Args:
            model: Model with a ``.sample()`` method.
            batch: Batch dict with required keys.

        Returns:
            Tuple of (motion_metrics_dict, kinematic_summary_dict).
        """
        prediction = model.sample(
            batch[SCENE_CONTEXT],
            key=batch["key"],
        )

        # prediction.trajectories: (N, T, 4) — take xy only → (N, T, 2)
        pred_xy = prediction.trajectories[..., :2]

        # Wrap into WOD shape: (B=1, M=1, K=1, N, T, 2)
        predictions_wod = pred_xy[jnp.newaxis, jnp.newaxis, jnp.newaxis]
        scores_wod = jnp.ones((1, 1, 1))

        gt: Any = batch["ground_truth"]
        valid: Any = batch["ground_truth_is_valid"]
        obj_type: Any = batch["object_type"]

        motion = self.motion_metrics.compute(
            predictions_wod,
            scores_wod,
            gt[jnp.newaxis],
            valid[jnp.newaxis],
            obj_type[jnp.newaxis],
        )

        kinematic: dict[str, float] = {}
        if self.sim_agent_metrics is not None and "road_edge_distances" in batch:
            sim_traj = prediction.trajectories[jnp.newaxis]  # (K=1, N, T, 4)
            result = self.sim_agent_metrics.compute(sim_traj, batch["road_edge_distances"])
            kinematic = {
                "speed_violation_rate": 1.0 - result.per_feature["linear_speed_ok"],
                "accel_violation_rate": 1.0 - result.per_feature["linear_acceleration_ok"],
                "offroad_rate": 1.0 - result.per_feature["offroad_free"],
                "collision_rate": 1.0 - result.per_feature["collision_free"],
            }

        return motion, kinematic
