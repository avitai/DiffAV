"""JAX-native motion prediction and sim-agent evaluation metrics.

Implements motion prediction metrics (ADE, FDE, minADE, minFDE, miss rate,
mAP) and sim-agent realism proxies in pure JAX. Input shapes follow the
``waymo_open_dataset`` ``get_motion_metric_ops`` convention, but the
definitions are deliberate simplifications: miss rate and mAP use a single
Euclidean final-displacement threshold rather than WOD's speed-scaled
lateral/longitudinal criteria, and the sim-agent scores are
envelope-feasibility rates rather than WOSAC's histogram log-likelihoods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from simulacrax.core.constants import WOSAC_2025_KINEMATIC_ENVELOPE
from simulacrax.core.geometry import RoadEdges, signed_distances_to_road_edges
from simulacrax.physics.kinematics import compute_kinematic_features


logger = logging.getLogger(__name__)

# Object type integer codes matching WOD convention.
_VEHICLE_TYPE = 0
_PEDESTRIAN_TYPE = 1
_CYCLIST_TYPE = 2
_TYPE_NAMES: dict[int, str] = {
    _VEHICLE_TYPE: "VEHICLE",
    _PEDESTRIAN_TYPE: "PEDESTRIAN",
    _CYCLIST_TYPE: "CYCLIST",
}


def ade(pred: jax.Array, gt: jax.Array, valid: jax.Array) -> jax.Array:
    """Average Displacement Error over valid timesteps.

    Args:
        pred: Predicted trajectory, shape (T, 2).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).

    Returns:
        Scalar ADE value.
    """
    dists = jnp.linalg.norm(pred - gt, axis=-1)  # (T,)
    return jnp.sum(dists * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def offroad_rate(
    rollouts: jax.Array,
    road_edges: RoadEdges,
    valid: jax.Array,
    *,
    chunk_size: int | None = None,
) -> jax.Array:
    """Fraction of valid agent-steps that are off-road (WOSAC definition).

    A position is off-road when its signed distance to the nearest oriented
    road edge is positive; the rate averages that indicator over the valid
    (rollout, agent, step) entries. Uses the exact WOSAC signed distance, so it
    matches the offline metric; ``chunk_size`` only bounds its memory.

    Args:
        rollouts: Rollouts, shape ``(num_rollouts, num_agents, num_steps,
            >=2)``; only the leading x, y are read.
        road_edges: Oriented road-edge polylines.
        valid: Per-step validity, shape ``(num_agents, num_steps)``, broadcast
            over rollouts.
        chunk_size: Optional polyline-chunk size bounding the closest-segment
            search memory (see
            :func:`~simulacrax.core.geometry.signed_distances_to_road_edges`).
            The rate is independent of it.

    Returns:
        Scalar off-road rate in ``[0, 1]``.
    """
    num_rollouts, num_agents, num_steps = rollouts.shape[:3]
    signed = signed_distances_to_road_edges(
        rollouts[..., :2], road_edges, chunk_size=chunk_size
    ).reshape(num_rollouts, num_agents, num_steps)
    offroad = signed > 0.0
    mask = jnp.broadcast_to(valid.astype(bool), offroad.shape)
    return jnp.sum(offroad & mask) / jnp.maximum(jnp.sum(mask), 1.0)


def fde(pred: jax.Array, gt: jax.Array, valid: jax.Array) -> jax.Array:
    """Final Displacement Error at the last valid timestep.

    The last valid index is located directly, so occlusion gaps in the
    validity mask are handled correctly.

    Args:
        pred: Predicted trajectory, shape (T, 2).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).

    Returns:
        Scalar FDE value. Returns 0.0 if no valid timesteps.
    """
    num_steps = valid.shape[0]
    last_t = (num_steps - 1) - jnp.argmax(valid[::-1] > 0)
    has_valid = jnp.any(valid > 0).astype(jnp.float32)
    return jnp.linalg.norm(pred[last_t] - gt[last_t]) * has_valid


def min_ade(pred_k: jax.Array, gt: jax.Array, valid: jax.Array) -> jax.Array:
    """Minimum ADE over K trajectory hypotheses.

    Args:
        pred_k: K predicted trajectories, shape (K, T, 2).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).

    Returns:
        Scalar minADE value.
    """
    ade_per_k = jax.vmap(lambda p: ade(p, gt, valid))(pred_k)  # (K,)
    return jnp.min(ade_per_k)


def benchmark_min_ade(
    pred_k: jax.Array,
    gt: jax.Array,
    valid: jax.Array,
    *,
    sample_interval: int = 5,
    measurement_steps: tuple[int, ...] = (30, 50, 80),
) -> jax.Array:
    """WOMD-style minADE over K hypotheses.

    Matches the Waymo Open Motion metric aggregation rather than a dense
    single-horizon average: the trajectory is subsampled from the 10 Hz track
    rate to the 2 Hz prediction rate (``sample_interval=5``), the average
    displacement is measured up to each of the 3 s / 5 s / 8 s horizons
    (``measurement_steps`` in 10 Hz steps), the minimum over the K hypotheses is
    taken per horizon, and the three per-horizon minima are averaged. This is the
    number reported on the WOMD leaderboard (marginal). If the trajectory is
    shorter than ``sample_interval`` (e.g. a small test fixture), it falls back to
    a dense stride so the metric stays defined.

    Args:
        pred_k: K predicted trajectories, shape ``(K, T, 2)``.
        gt: Ground-truth trajectory, shape ``(T, 2)``.
        valid: Boolean per-step validity, shape ``(T,)``.
        sample_interval: 10 Hz-to-prediction-rate stride (5 gives 2 Hz).
        measurement_steps: Horizons in 10 Hz steps to average over (3/5/8 s).

    Returns:
        Scalar WOMD-style minADE.
    """
    num_steps = gt.shape[0]
    stride = sample_interval if sample_interval <= num_steps else 1
    sub = jnp.arange(stride - 1, num_steps, stride)  # subsampled step indices
    distances = jnp.linalg.norm(pred_k[:, sub, :] - gt[sub, :][None], axis=-1)  # (K, S)
    sub_valid = valid[sub].astype(jnp.float32)  # (S,)
    horizon_min_ades = []
    for step in measurement_steps:
        # Static horizon mask (no dynamic slicing): subsampled points at or before
        # the measurement step. Clamps naturally when the horizon exceeds T.
        weights = sub_valid * (sub < step).astype(jnp.float32)  # (S,)
        ade_per_k = jnp.sum(distances * weights[None], axis=-1) / jnp.maximum(
            jnp.sum(weights), 1.0
        )  # (K,)
        horizon_min_ades.append(jnp.min(ade_per_k))
    return jnp.mean(jnp.stack(horizon_min_ades))


def min_fde(pred_k: jax.Array, gt: jax.Array, valid: jax.Array) -> jax.Array:
    """Minimum FDE over K trajectory hypotheses.

    Args:
        pred_k: K predicted trajectories, shape (K, T, 2).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).

    Returns:
        Scalar minFDE value.
    """
    fde_per_k = jax.vmap(lambda p: fde(p, gt, valid))(pred_k)  # (K,)
    return jnp.min(fde_per_k)


def miss_rate(
    pred_k: jax.Array,
    gt: jax.Array,
    valid: jax.Array,
    *,
    threshold: float,
) -> jax.Array:
    """Miss rate: 1.0 if minFDE exceeds threshold, else 0.0.

    Args:
        pred_k: K predicted trajectories, shape (K, T, 2).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).
        threshold: Miss threshold in metres (WOD default: 2.0).

    Returns:
        Scalar 0.0 (hit) or 1.0 (miss).
    """
    return (min_fde(pred_k, gt, valid) > threshold).astype(jnp.float32)


def mean_average_precision(
    pred_k: jax.Array,
    scores_k: jax.Array,
    gt: jax.Array,
    valid: jax.Array,
    *,
    threshold: float = 2.0,
) -> jax.Array:
    """Mean Average Precision over K ranked hypotheses.

    Ranks hypotheses by score, computes precision at each rank,
    then averages precision at hit positions.

    Args:
        pred_k: K predicted trajectories, shape (K, T, 2).
        scores_k: Confidence scores per hypothesis, shape (K,).
        gt: Ground truth trajectory, shape (T, 2).
        valid: Boolean validity mask, shape (T,).
        threshold: Hit threshold in metres (default 2.0).

    Returns:
        Scalar mAP value in [0, 1].
    """
    fde_per_k = jax.vmap(lambda p: fde(p, gt, valid))(pred_k)  # (K,)
    hits = (fde_per_k <= threshold).astype(jnp.float32)  # (K,)
    sorted_idx = jnp.argsort(-scores_k)
    sorted_hits = hits[sorted_idx]  # (K,)
    k = jnp.arange(1, pred_k.shape[0] + 1, dtype=jnp.float32)
    precision = jnp.cumsum(sorted_hits) / k  # (K,)
    n_hits = jnp.maximum(jnp.sum(hits), 1.0)
    return jnp.sum(precision * sorted_hits) / n_hits


# ---------------------------------------------------------------------------
# MotionMetricsConfig + MotionMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionMetricsConfig:
    """Configuration for WOD-compatible motion prediction metrics.

    Attributes:
        miss_rate_threshold: Displacement threshold for miss-rate (metres).
            WOD default is 2.0 m.
        top_k: Number of trajectory hypotheses per prediction group.
        num_future_steps: Prediction horizon length (WOD: 80 steps = 8 s).
    """

    miss_rate_threshold: float = 2.0
    top_k: int = 6
    num_future_steps: int = 80


class MotionMetrics:
    """Motion prediction evaluation metrics with WOD input shapes.

    Implements minADE, minFDE, miss rate, and mAP per agent type using the
    input shape convention of ``waymo_open_dataset``'s
    ``get_motion_metric_ops``. All computation is pure JAX. Miss rate and
    mAP use a single Euclidean final-displacement threshold
    (``miss_rate_threshold``) — a simplification of WOD's speed-scaled
    lateral/longitudinal criteria.

    Args:
        config: Metric configuration.
    """

    def __init__(self, config: MotionMetricsConfig) -> None:
        """Initialise MotionMetrics."""
        self._config = config

    def compute(
        self,
        predictions: jax.Array,
        scores: jax.Array,
        ground_truth: jax.Array,
        ground_truth_is_valid: jax.Array,
        object_type: jax.Array,
    ) -> dict[str, float]:
        """Compute motion metrics over a batch of scenarios.

        Input shapes mirror ``get_motion_metric_ops``:

        - predictions: ``(B, M, K, N, T, 2)``
        - scores: ``(B, M, K)``
        - ground_truth: ``(B, A, TG, 7)`` — last dim ``[x,y,len,wid,hdg,vx,vy]``
        - ground_truth_is_valid: ``(B, A, TG)``
        - object_type: ``(B, A)`` — 0=vehicle, 1=pedestrian, 2=cyclist

        Prediction groups map to ground-truth agents in row-major order:
        group ``m``, in-group agent ``n`` corresponds to ground-truth agent
        ``m * N + n``, so ``A`` must equal ``M * N``.

        Output keys match WOD breakdown names, e.g. ``"VEHICLE/minADE"``.

        Args:
            predictions: Multi-hypothesis trajectory predictions.
            scores: Confidence scores per hypothesis.
            ground_truth: Ground truth trajectories with full state.
            ground_truth_is_valid: Per-timestep validity mask.
            object_type: Integer agent type codes.

        Returns:
            Dict mapping ``"TYPE/metric"`` keys to float values.

        Raises:
            ValueError: If the ground-truth agent count does not equal
                ``M * N``.
        """
        b, m, k, n, t = predictions.shape[:5]
        num_ground_truth_agents = ground_truth.shape[1]
        if num_ground_truth_agents != m * n:
            msg = (
                f"ground_truth carries {num_ground_truth_agents} agents but "
                f"predictions imply {m} groups x {n} agents = {m * n}; the "
                "row-major group-to-agent correspondence requires A == M * N"
            )
            raise ValueError(msg)

        # Flatten (B, M, N) → single agent axis in row-major group order;
        # each agent has k hypotheses of t timesteps.
        num_agents = b * m * n
        tg = ground_truth.shape[2]
        use_t = min(t, tg)
        pred_flat = predictions.transpose(0, 1, 3, 2, 4, 5).reshape(num_agents, k, t, 2)
        pred_flat = pred_flat[:, :, :use_t, :]
        gt_flat = ground_truth[..., :2].reshape(num_agents, tg, 2)[:, :use_t, :]
        valid_flat = ground_truth_is_valid.reshape(num_agents, tg)[:, :use_t]
        obj_flat = object_type.reshape(num_agents)
        # Each agent in a group shares the group's hypothesis scores.
        scores_flat = jnp.broadcast_to(scores[:, :, jnp.newaxis, :], (b, m, n, k)).reshape(
            num_agents, k
        )

        threshold = self._config.miss_rate_threshold
        results: dict[str, float] = {}
        for type_id, type_name in _TYPE_NAMES.items():
            mask = obj_flat == type_id
            if not bool(jnp.any(mask)):
                continue
            pred_t = pred_flat[mask]
            gt_t = gt_flat[mask]
            valid_t = valid_flat[mask]
            scores_t = scores_flat[mask]

            results[f"{type_name}/minADE"] = float(
                jnp.mean(jax.vmap(lambda p, g, v: min_ade(p, g, v))(pred_t, gt_t, valid_t))
            )
            results[f"{type_name}/minFDE"] = float(
                jnp.mean(jax.vmap(lambda p, g, v: min_fde(p, g, v))(pred_t, gt_t, valid_t))
            )
            results[f"{type_name}/MissRate"] = float(
                jnp.mean(
                    jax.vmap(lambda p, g, v: miss_rate(p, g, v, threshold=threshold))(
                        pred_t, gt_t, valid_t
                    )
                )
            )
            results[f"{type_name}/mAP"] = float(
                jnp.mean(
                    jax.vmap(
                        lambda p, s, g, v: mean_average_precision(p, s, g, v, threshold=threshold)
                    )(pred_t, scores_t, gt_t, valid_t)
                )
            )
        return results


# ---------------------------------------------------------------------------
# SimAgentMetricsConfig + SimAgentMetrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SimAgentMetricsConfig:
    """Configuration for JAX-native sim agent realism proxies.

    Attributes:
        kinematic_weight: Weight for the kinematic bucket in the combined
            score.
        interactive_weight: Weight for the interactive bucket.
        map_weight: Weight for the map-based bucket.
        collision_threshold: Agent centre-to-centre collision distance
            (metres).
    """

    kinematic_weight: float = 0.4
    interactive_weight: float = 0.4
    map_weight: float = 0.2
    collision_threshold: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SimAgentMetricsResult:
    """Immutable sim agent evaluation results.

    Attributes:
        kinematic_score: Score in [0, 1] for speed/acceleration realism.
        interactive_score: Score in [0, 1] for collision/TTC/proximity realism.
        map_score: Score in [0, 1] for offroad/road-edge realism.
        metametric: Weighted combination of the three bucket scores.
        per_feature: Individual feature scores keyed by feature name.
    """

    kinematic_score: float
    interactive_score: float
    map_score: float
    metametric: float
    per_feature: dict[str, float]


def _fraction_within(values: jax.Array, lower: float, upper: float) -> jax.Array:
    """Fraction of finite entries within ``[lower, upper]``.

    The NaN boundary pads produced by the central-difference kinematic
    features are excluded from both numerator and denominator.
    """
    finite = jnp.isfinite(values)
    ok = finite & (values >= lower) & (values <= upper)
    return jnp.sum(ok) / jnp.maximum(jnp.sum(finite), 1)


class SimAgentMetrics:
    """JAX-native sim agent realism proxies aligned with the WOSAC buckets.

    Computes kinematic, interactive (collision), and map-based (offroad)
    scores and combines them into a weighted summary. Kinematic features
    are derived from positions and headings via the WOSAC central
    differences at the scenario step rate and checked against the 2025
    challenge kinematic envelope; each rollout is scored individually.
    These are envelope-feasibility rates — a proxy, not the WOSAC
    histogram log-likelihood metametric.

    Args:
        config: Metric configuration with bucket weights.
    """

    def __init__(self, config: SimAgentMetricsConfig) -> None:
        """Initialise SimAgentMetrics."""
        self._config = config

    def compute(
        self,
        trajectories: jax.Array,
        road_edge_distances: jax.Array,
    ) -> SimAgentMetricsResult:
        """Compute sim agent realism scores.

        Args:
            trajectories: Simulated trajectories, shape ``(K, N, T, 4)``
                where the last dim is ``[x, y, heading, speed]``. Speeds
                and accelerations are derived from the positions and
                headings, not from the speed channel.
            road_edge_distances: Precomputed signed distance to the nearest
                road edge per agent per timestep, shape ``(N, T)``;
                positive is off-road.

        Returns:
            ``SimAgentMetricsResult`` with per-bucket and combined scores.
        """
        num_agents = trajectories.shape[1]

        # ---- Kinematic bucket: central-difference features at 10 Hz ----
        x = trajectories[..., 0]
        y = trajectories[..., 1]
        z = jnp.zeros_like(x)
        heading = trajectories[..., 2]
        linear_speed, linear_accel, angular_speed, angular_accel = compute_kinematic_features(
            x, y, z, heading
        )
        envelope = WOSAC_2025_KINEMATIC_ENVELOPE
        speed_ok = _fraction_within(
            linear_speed, envelope.linear_speed_min, envelope.linear_speed_max
        )
        accel_ok = _fraction_within(
            linear_accel,
            envelope.linear_acceleration_min,
            envelope.linear_acceleration_max,
        )
        angular_speed_ok = _fraction_within(
            angular_speed, envelope.angular_speed_min, envelope.angular_speed_max
        )
        angular_accel_ok = _fraction_within(
            angular_accel,
            envelope.angular_acceleration_min,
            envelope.angular_acceleration_max,
        )
        kinematic_score = float((speed_ok + accel_ok + angular_speed_ok + angular_accel_ok) / 4.0)

        # ---- Interactive bucket: per-rollout collision rate ----
        def rollout_collision_rate(rollout_xy: jax.Array) -> jax.Array:
            """Fraction of agent-steps closer than the collision threshold."""
            pos_i = rollout_xy[:, jnp.newaxis, :, :]  # (N, 1, T, 2)
            pos_j = rollout_xy[jnp.newaxis, :, :, :]  # (1, N, T, 2)
            pairwise = jnp.linalg.norm(pos_i - pos_j, axis=-1)  # (N, N, T)
            eye = jnp.eye(num_agents, dtype=bool)[:, :, jnp.newaxis]
            inf_fill = jnp.full_like(pairwise, jnp.inf)
            # cast: the jax 0.9 stubs annotate three-argument jnp.where with
            # the one-argument tuple return included.
            pairwise_masked = cast(jax.Array, jnp.where(eye, inf_fill, pairwise))
            nearest = jnp.min(pairwise_masked, axis=1)  # (N, T)
            collision = nearest < self._config.collision_threshold
            return jnp.mean(collision.astype(jnp.float32))

        collision_rate = jnp.mean(jax.vmap(rollout_collision_rate)(trajectories[..., :2]))
        interactive_score = float(1.0 - collision_rate)

        # ---- Map-based bucket ----
        offroad = road_edge_distances > 0.0
        map_score = float(1.0 - jnp.mean(offroad.astype(jnp.float32)))

        # ---- Combined score ----
        cfg = self._config
        metametric = (
            cfg.kinematic_weight * kinematic_score
            + cfg.interactive_weight * interactive_score
            + cfg.map_weight * map_score
        )

        per_feature = {
            "linear_speed_ok": float(speed_ok),
            "linear_acceleration_ok": float(accel_ok),
            "angular_speed_ok": float(angular_speed_ok),
            "angular_acceleration_ok": float(angular_accel_ok),
            "collision_free": interactive_score,
            "distance_to_road_edge_mean": float(jnp.mean(road_edge_distances)),
            "offroad_free": map_score,
        }

        return SimAgentMetricsResult(
            kinematic_score=kinematic_score,
            interactive_score=interactive_score,
            map_score=map_score,
            metametric=float(metametric),
            per_feature=per_feature,
        )
