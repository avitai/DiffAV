# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
"""
# Evaluation Pipeline Tutorial

| | |
|---|---|
| **Level** | Intermediate |
| **Runtime (CPU)** | ~15 min |
| **Runtime (GPU)** | ~3 min |
| **Prerequisites** | `01_wod_metrics_quickref.py` |
| **Key concepts** | EvaluationRunner, MotionMetrics, WosacMetametric, MetricsDashboard |
| **Sister repos** | calibrax (PublicationGenerator) |

## Overview

This tutorial walks through the complete evaluation pipeline for trajectory
prediction on real Waymo Open Dataset (WOD) Motion scenarios — from building
evaluation batches for the challenge-designated `tracks_to_predict` agents,
through motion metrics and the WOSAC-style realism metametric with 32
rollouts, to publication-ready CSV and HTML reports.

Every batch comes from a real WOD scenario: ground truth, validity masks,
agent types, and signed road-edge distances computed from the scenario's own
road-edge polylines (roadgraph types 15/16).

```
WODSource → tracks_to_predict ground truth + road edges (types 15/16)
              ↓
MotionMetrics.compute(predictions, scores, ground_truth, ...)
    → {"VEHICLE/minADE": ..., "CYCLIST/MissRate": ...}
              ↓
SimAgentMetrics.compute(trajectories, road_edge_distances)
    → envelope-feasibility proxy (kinematic / interactive / map)
              ↓
compute_metametric_features + WosacMetametric  (K=32 rollouts)
    → histogram log-likelihood realism scores (official scoring flow)
              ↓
EvaluationRunner.run(model, batches) → MetricsReport
              ↓
MetricsDashboard.generate(report) → motion/table.csv + .html
```

## What You'll Learn

1. Build evaluation batches from real scenarios with correct road edges
2. Run the `EvaluationRunner` over multiple scenarios
3. Break down results per agent type: VEHICLE / PEDESTRIAN / CYCLIST
4. Compute the `SimAgentMetrics` feasibility proxy with official bucket weights
5. Score K=32 rollouts with the repo's `WosacMetametric` (histogram
   log-likelihood, the official 2025 scoring flow)
6. Generate publication-ready reports with `MetricsDashboard`
7. Inspect physics violation summaries
"""

# %% [markdown]
"""
## 1. Overview

The evaluation stack in DiffAV has these layers:

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| Functions | `ade`, `fde`, `min_ade`, `min_fde`, `miss_rate` | Per-trajectory scalars |
| Aggregator | `MotionMetrics` | Per-agent-type WOD breakdown |
| Realism proxy | `SimAgentMetrics` | Envelope-feasibility rates per bucket |
| Realism metric | `WosacMetametric` | Histogram log-likelihood metametric |
| Orchestrator | `EvaluationRunner` | Batch iteration, metric accumulation |
| Export | `MetricsDashboard` | CSV + HTML tables via calibrax |

The `EvaluationRunner` calls `model.sample()` per batch, wraps the prediction
into the WOD `(B, M, K, N, T, 2)` input shape expected by `MotionMetrics`
(as a single K=1 hypothesis), and accumulates per-batch averages into a
single `MetricsReport`.
"""

# %%
# Imports — standard library, JAX, numpy, and diffav
import dataclasses
import os
import tempfile
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from dotenv import load_dotenv

from diffav.core.constants import (
    ROAD_EDGE_Z_STRETCH,
    WOSAC_2025_METAMETRIC_CONFIG,
    WOSAC_N_ROLLOUTS,
)
from diffav.core.geometry import RoadEdges, signed_distance_to_polylines
from diffav.core.types import MetricsReport, TrajectoryPrediction
from diffav.data import resolve_wod_tfrecord_path, road_edges_from_wod_dict
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.evaluation import (
    compute_metametric_features,
    EvaluationRunner,
    log_likelihood_estimate_timeseries,
    MetametricFeatures,
    MetricsDashboard,
    MotionMetrics,
    MotionMetricsConfig,
    SimAgentMetrics,
    SimAgentMetricsConfig,
    WosacMetametric,
)


# Constants matching WOD Motion format
HIST_STEPS = 11  # 10 past steps + 1 current
FUTURE_STEPS = 80  # 8 s at 10 Hz
MAX_AGENTS = 128  # tf_example native width — keeps every tracks_to_predict row
N_SCENARIOS = 20  # scenarios to evaluate

print(f"JAX devices: {jax.devices()}")
# Smoke mode (set by the example execution tests) shrinks the WOSAC rollout
# count; real runs keep the official K = 32.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
N_EVAL_ROLLOUTS = 4 if _SMOKE else WOSAC_N_ROLLOUTS
print(f"Scenarios: {N_SCENARIOS}, future steps: {FUTURE_STEPS}, rollouts: {WOSAC_N_ROLLOUTS}")

# %% [markdown]
"""
## 2. Setup: EvaluationRunner with MotionMetrics

`MotionMetricsConfig` mirrors the configuration surface of the official WOD
`get_motion_metric_ops` call. Note the simplification: miss rate and mAP use
a single 2.0 m Euclidean final-displacement threshold, whereas the official
metrics use speed-scaled lateral/longitudinal thresholds evaluated at 2 Hz.

`EvaluationRunner` wraps each sampled prediction as a single hypothesis
(`top_k=1`); Section 7 evaluates full K=32 rollout sets directly.
"""

# %%
motion_config = MotionMetricsConfig(
    miss_rate_threshold=2.0,  # metres — simplified Euclidean criterion
    top_k=1,  # EvaluationRunner wraps one sample per scenario as K=1
    num_future_steps=FUTURE_STEPS,
)
motion_metrics = MotionMetrics(motion_config)

runner = EvaluationRunner(motion_metrics=motion_metrics)

print("MotionMetricsConfig:")
print(f"  miss_rate_threshold = {motion_config.miss_rate_threshold} m")
print(f"  top_k               = {motion_config.top_k}")
print(f"  num_future_steps    = {motion_config.num_future_steps}")
# Expected output:
# MotionMetricsConfig:
#   miss_rate_threshold = 2.0 m
#   top_k               = 1
#   num_future_steps    = 80

# %% [markdown]
"""
## 3. Building Evaluation Batches from Real Scenarios

Each batch evaluates only the challenge-designated `state/tracks_to_predict`
agents — padding rows (WOD fills invalid agents with -1 sentinels) never
enter the metrics, and object types come from the real `state/type` codes.

Road edges are extracted with `road_edges_from_wod_dict`, which follows the
official map metric: roadgraph types 15 (`ROAD_EDGE_BOUNDARY`) and 16
(`ROAD_EDGE_MEDIAN`) only. The signed-distance query passes the agents'
real `[x, y, z]` positions — WOD maps sit at real elevations (often tens of
metres), and the closest-segment association scales vertical offsets by
`ROAD_EDGE_Z_STRETCH = 3`, so querying at z=0 would mis-associate segments.
"""

# %%
load_dotenv()
load_dotenv(".env.data")
WOD_PATH = resolve_wod_tfrecord_path()
wod_source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
print(f"Loaded {len(wod_source)} scenarios")
# Expected output:
# Loaded 333 scenarios


def _extract_gt(raw: dict[str, Any]) -> tuple[np.ndarray, jax.Array, jax.Array, jax.Array]:
    """Extract future ground truth for the tracks_to_predict agents.

    Only the challenge-designated ``state/tracks_to_predict`` rows are
    evaluated; padding rows (-1 sentinels) never enter the metrics, and
    object types come from the real ``state/type`` codes without clipping.

    Returns:
        rows:         (N,) agent row indices with tracks_to_predict == 1.
        ground_truth: (N, 80, 7) — [x, y, length, width, heading, vx, vy].
        valid:        (N, 80) float32 validity mask.
        object_type:  (N,) int32 — 0=vehicle, 1=pedestrian, 2=cyclist.
    """
    rows = np.where(np.asarray(raw["state/tracks_to_predict"]).reshape(-1) > 0)[0]
    future = slice(HIST_STEPS, HIST_STEPS + FUTURE_STEPS)

    def _field(name: str) -> jax.Array:
        return jnp.array(np.asarray(raw[name])[rows][:, future])

    ground_truth = jnp.stack(
        [
            _field("state/all/x"),
            _field("state/all/y"),
            _field("state/all/length"),
            _field("state/all/width"),
            _field("state/all/bbox_yaw"),
            _field("state/all/velocity_x"),
            _field("state/all/velocity_y"),
        ],
        axis=-1,
    )  # (N, 80, 7)
    valid = jnp.array(np.asarray(raw["state/all/valid"])[rows][:, future], dtype=jnp.float32)
    object_type = jnp.array(np.asarray(raw["state/type"]).reshape(-1)[rows].astype(np.int32) - 1)
    return rows, ground_truth, valid, object_type


def _make_wod_batch(scenario_idx: int) -> dict[str, Any] | None:
    """Build one evaluation batch from a real WOD scenario.

    Args:
        scenario_idx: Index into the loaded WOD split.

    Returns:
        Batch dict compatible with EvaluationRunner.run(), or None when the
        scenario has no tracks_to_predict rows or no road edges.
    """
    raw = wod_source[scenario_idx].data
    rows, ground_truth, valid, object_type = _extract_gt(raw)
    road_edges = road_edges_from_wod_dict(raw)
    if rows.size == 0 or road_edges is None:
        return None
    num_agents = int(rows.size)

    # Scene context: flattened xy history (simplified embedding)
    scene_context = ground_truth[..., :2].reshape(num_agents, -1)[:, :64]

    # Signed road-edge distances at the agents' real [x, y, z] positions
    future = slice(HIST_STEPS, HIST_STEPS + FUTURE_STEPS)
    z = jnp.array(np.asarray(raw["state/all/z"])[rows][:, future])
    query_xyz = jnp.concatenate([ground_truth[..., :2], z[..., jnp.newaxis]], axis=-1)
    road_edge_distances = signed_distance_to_polylines(
        query_xyz.reshape(-1, 3),
        road_edges.polylines,
        road_edges.is_cyclic,
        z_stretch=ROAD_EDGE_Z_STRETCH,
    ).reshape(num_agents, FUTURE_STEPS)

    return {
        "scene_context": scene_context,  # (N, 64)
        "ground_truth": ground_truth,  # (N, 80, 7)
        "ground_truth_is_valid": valid,  # (N, 80)
        "object_type": object_type,  # (N,)
        "key": jax.random.key(scenario_idx),
        "road_edge_distances": road_edge_distances,  # (N, 80), positive = off-road
    }


# Inspect the first real batch
sample_batch = _make_wod_batch(0)
assert sample_batch is not None
print("WOD batch shapes (tracks_to_predict agents only):")
for key in ("ground_truth", "ground_truth_is_valid", "object_type", "scene_context"):
    print(f"  {key}: {sample_batch[key].shape}")

for label, code in [("VEHICLE", 0), ("PEDESTRIAN", 1), ("CYCLIST", 2)]:
    n = int(jnp.sum(sample_batch["object_type"] == code))
    print(f"  {label}: {n} agents")
on_road = float(jnp.mean((sample_batch["road_edge_distances"] < 0.0).astype(jnp.float32)))
print(f"  on-road (signed distance < 0): {on_road:.1%} of agent-steps")
# Expected output (scenario 0 has tracks_to_predict rows {0, 1, 2}):
# WOD batch shapes (tracks_to_predict agents only):
#   ground_truth: (3, 80, 7)
#   ground_truth_is_valid: (3, 80)
#   object_type: (3,)
#   scene_context: (3, 64)
#   VEHICLE: 2 agents
#   PEDESTRIAN: 0 agents
#   CYCLIST: 1 agents
#   on-road (signed distance < 0): xx.x% of agent-steps

# %% [markdown]
"""
## 4. Running the EvaluationRunner

The model passed to `EvaluationRunner.run()` must implement:

```python
def sample(
    self,
    scene_context: jax.Array,   # (N, context_dim) — one row per agent
    *,
    key: jax.Array,
) -> TrajectoryPrediction:
    ...
```

`_NoisyGtPredictor` below simulates a model whose predicted positions are
the ground truth plus time-growing lateral noise, while heading and speed
come from the logged state — a calibrated but non-trivial baseline. Replace
with a trained `TrajectoryDiffusionModel` for production benchmarks.
"""


# %%
class _NoisyGtPredictor:
    """Baseline predictor: ground-truth motion plus time-growing lateral noise.

    Implements the EvaluationRunner model interface:
        sample(scene_context, *, key) -> TrajectoryPrediction

    Positions are the ground truth plus Gaussian noise whose scale ramps
    from 0 m at t=0 to ``noise_max_m`` at the final step; heading and speed
    come from the logged state so downstream kinematic features stay
    meaningful. Replace with a trained TrajectoryDiffusionModel for real
    evaluation.

    Args:
        ground_truth: Ground truth state, shape (A, T, 7) —
            [x, y, length, width, heading, vx, vy].
        noise_max_m: Noise standard deviation (metres) at the final step.
    """

    def __init__(self, ground_truth: jax.Array, noise_max_m: float = 1.5) -> None:
        """Initialise predictor with ground truth and noise level."""
        self._xy = ground_truth[..., :2]  # (A, T, 2)
        self._heading = ground_truth[..., 4:5]  # (A, T, 1)
        self._speed = jnp.linalg.norm(ground_truth[..., 5:7], axis=-1, keepdims=True)
        self._ramp = jnp.linspace(0.0, noise_max_m, ground_truth.shape[1])  # (T,)

    def sample(
        self,
        scene_context: jax.Array,
        *,
        key: jax.Array,
    ) -> TrajectoryPrediction:
        """Return one noisy rollout, one agent per scene-context row."""
        num_agents = scene_context.shape[0]
        xy = self._xy[:num_agents]
        noise = jax.random.normal(key, xy.shape) * self._ramp[jnp.newaxis, :, jnp.newaxis]
        trajectories = jnp.concatenate(
            [xy + noise, self._heading[:num_agents], self._speed[:num_agents]], axis=-1
        )  # (A, T, 4) — [x, y, heading, speed]
        return TrajectoryPrediction(
            trajectories=trajectories,
            agent_ids=tuple(f"agent_{i}" for i in range(num_agents)),
        )


# Build batches over the first N_SCENARIOS scenarios with usable data
batches = []
predictors = []
for i in range(len(wod_source)):
    if len(batches) >= N_SCENARIOS:
        break
    batch = _make_wod_batch(i)
    if batch is None:
        continue
    predictors.append(_NoisyGtPredictor(batch["ground_truth"]))
    batches.append(batch)
print(f"Built {len(batches)} evaluation batches")

# Run evaluation per scenario (each predictor is tied to its scenario's GT)
all_reports = []
for i, (batch, predictor) in enumerate(zip(batches, predictors)):
    report = runner.run(predictor, [batch])
    all_reports.append(report)
    if i < 3:
        min_ade_val = report.metric_values.get("VEHICLE/minADE", float("nan"))
        print(f"Scenario {i}: VEHICLE/minADE = {min_ade_val:.4f} m")

# Expected output (approximate — noise is random):
# Scenario 0: VEHICLE/minADE = x.xxxx m
# Scenario 1: VEHICLE/minADE = x.xxxx m
# Scenario 2: VEHICLE/minADE = x.xxxx m

# %% [markdown]
"""
## 5. Per-Agent-Type Breakdown (VEHICLE / PEDESTRIAN / CYCLIST)

`MotionMetrics` automatically stratifies results by agent type.
Each metric key has the form `"TYPE/metricName"` where TYPE is one of
`VEHICLE`, `PEDESTRIAN`, or `CYCLIST`. Types outside these codes (e.g. the
-1 padding sentinel) are never counted.

Here we aggregate the per-scenario reports into a single average report.
"""

# %%
agg: dict[str, list[float]] = defaultdict(list)
for rep in all_reports:
    for key, val in rep.metric_values.items():
        agg[key].append(val)

avg_metric_values = {k: sum(v) / len(v) for k, v in agg.items()}
avg_report = MetricsReport(metric_values=avg_metric_values)

print(f"{len(all_reports)}-scenario average metrics (per agent type):")
for agent_type in ("VEHICLE", "PEDESTRIAN", "CYCLIST"):
    type_keys = sorted(k for k in avg_metric_values if k.startswith(f"{agent_type}/"))
    if not type_keys:
        continue
    print(f"  {agent_type}:")
    for key in type_keys:
        print(f"    {key.split('/')[1]}: {avg_metric_values[key]:.4f}")
# Expected output (approximate):
# 20-scenario average metrics (per agent type):
#   VEHICLE:
#     MissRate: x.xxxx
#     mAP: x.xxxx
#     minADE: x.xxxx
#     minFDE: x.xxxx
#   PEDESTRIAN:
#     ...
#   CYCLIST:
#     ...

# %% [markdown]
"""
## 6. SimAgentMetrics: Envelope-Feasibility Proxy

`SimAgentMetrics` computes fast realism *proxies* aligned with the three
WOSAC buckets — envelope-feasibility rates, not the official histogram
log-likelihoods (Section 7 computes those). The bucket weights below are
the official 2025 challenge weights (per-feature weights summed per
bucket in `challenge_2025_sim_agents_config`):

| Bucket | Weight | Proxy check |
|--------|--------|-------------|
| Kinematic | 0.2 | Central-difference speed/accel/angular rates inside the WOSAC 2025 envelope |
| Interactive | 0.45 | Nearest-agent centre distance ≥ collision threshold |
| Map | 0.35 | Signed road-edge distance ≤ 0 (agent is on-road) |

Input shape: `trajectories (K, N, T, 4)` where the last dim is
`[x, y, heading, speed]`. A single-hypothesis prediction uses `K=1`.

Plug `sim_agent_metrics` into `EvaluationRunner` to get the
`kinematic_violation_summary` field in `MetricsReport`.
"""

# %%
sim_config = SimAgentMetricsConfig(
    kinematic_weight=0.2,  # official 2025 bucket weights
    interactive_weight=0.45,
    map_weight=0.35,
    collision_threshold=1.0,  # metres, centre-to-centre
)
sim_metrics = SimAgentMetrics(sim_config)

runner_with_sim = EvaluationRunner(
    motion_metrics=motion_metrics,
    sim_agent_metrics=sim_metrics,
)

# Re-run scenario 0 with sim-agent metrics enabled
report_0 = runner_with_sim.run(predictors[0], [batches[0]])

print("SimAgentMetrics results (scenario 0):")
if report_0.kinematic_violation_summary:
    for key, val in sorted(report_0.kinematic_violation_summary.items()):
        print(f"  {key}: {val:.4f}")
else:
    print("  (no road_edge_distances in batch — sim-agent metrics skipped)")
# Expected output:
# SimAgentMetrics results (scenario 0):
#   accel_violation_rate: x.xxxx
#   collision_rate: x.xxxx
#   offroad_rate: x.xxxx
#   speed_violation_rate: x.xxxx

# Demonstrate SimAgentMetrics.compute() directly on the logged trajectories
gt_0 = batches[0]["ground_truth"]
sample_traj = jnp.concatenate(
    [
        gt_0[..., :2],  # (N, T, 2) xy
        gt_0[..., 4:5],  # (N, T, 1) heading
        jnp.linalg.norm(gt_0[..., 5:7], axis=-1, keepdims=True),  # (N, T, 1) speed
    ],
    axis=-1,
)  # (N, T, 4)
sim_result = sim_metrics.compute(
    trajectories=sample_traj[jnp.newaxis],  # (K=1, N, T, 4)
    road_edge_distances=batches[0]["road_edge_distances"],  # (N, T)
)
print()
print("Direct SimAgentMetrics.compute() result (logged trajectories):")
print(f"  kinematic_score   = {sim_result.kinematic_score:.4f}")
print(f"  interactive_score = {sim_result.interactive_score:.4f}")
print(f"  map_score         = {sim_result.map_score:.4f}")
print(f"  metametric        = {sim_result.metametric:.4f}")
# Expected output (logged data is feasible almost everywhere):
# Direct SimAgentMetrics.compute() result (logged trajectories):
#   kinematic_score   = ~1.0
#   interactive_score = ~1.0
#   map_score         = ~1.0
#   metametric        = ~1.0

# %% [markdown]
"""
## 7. WosacMetametric: Histogram Log-Likelihood over K=32 Rollouts

The official 2025 scoring flow estimates each feature's distribution across
`WOSAC_N_ROLLOUTS = 32` simulated rollouts, evaluates the logged scenario
under it, and combines per-feature likelihoods with the challenge weights.
The repo implements this as:

1. `compute_metametric_features` — kinematic, interactive, and map features
   per rollout (`(K, N, T)` arrays)
2. `log_likelihood_estimate_timeseries` / `..._scenario_level` — JAX ports
   of the official histogram/bernoulli estimators
3. `WosacMetametric.compute(log_features, sim_features)` — validity-masked
   likelihoods, bucket scores, and the weighted metametric

We score the noisy predictor on scenarios that carry at least two fully
valid `tracks_to_predict` agents (partially observed logged tracks would
leak -1 sentinel positions into the simulated feature distributions).
Features are computed in rollout chunks to bound the memory of the
road-edge distance query (points x segments).
"""


# %%
def _sim_features_chunked(
    rollouts: jax.Array,
    valid: jax.Array,
    road_edges: RoadEdges,
    chunk_size: int = 4,
) -> MetametricFeatures:
    """Run compute_metametric_features over rollout chunks and concatenate.

    Bounds the peak memory of the road-edge query, which materialises a
    (points x polylines x segments) tensor.
    """
    parts = [
        compute_metametric_features(rollouts[start : start + chunk_size], valid, road_edges)
        for start in range(0, rollouts.shape[0], chunk_size)
    ]
    if len(parts) == 1:
        return parts[0]
    merged = {
        field.name: jnp.concatenate([getattr(p, field.name) for p in parts], axis=0)
        for field in dataclasses.fields(MetametricFeatures)
        if field.name != "valid"
    }
    return MetametricFeatures(valid=valid, **merged)


metametric = WosacMetametric()  # defaults to WOSAC_2025_METAMETRIC_CONFIG
bucket_totals: dict[str, list[float]] = defaultdict(list)
scored_scenarios = 0
last_result = None

for scenario_idx in range(len(wod_source)):
    if scored_scenarios >= N_SCENARIOS:
        break
    raw = wod_source[scenario_idx].data
    rows, ground_truth, valid, _ = _extract_gt(raw)
    road_edges = road_edges_from_wod_dict(raw)
    fully_valid = np.where(np.asarray(valid).sum(axis=1) == FUTURE_STEPS)[0]
    if fully_valid.size < 2 or road_edges is None:
        continue

    gt = ground_truth[fully_valid]  # (N, 80, 7)
    log_traj = jnp.concatenate(
        [gt[..., :2], gt[..., 4:5], jnp.linalg.norm(gt[..., 5:7], axis=-1, keepdims=True)],
        axis=-1,
    )[jnp.newaxis]  # (1, N, 80, 4)
    log_valid = jnp.ones(log_traj.shape[1:3], dtype=bool)

    predictor = _NoisyGtPredictor(gt)
    base_key = jax.random.key(1000 + scenario_idx)
    rollouts = jnp.stack(
        [
            predictor.sample(
                jnp.zeros((gt.shape[0], 1)), key=jax.random.fold_in(base_key, k)
            ).trajectories
            for k in range(N_EVAL_ROLLOUTS)
        ]
    )  # (K, N, 80, 4)

    log_features = compute_metametric_features(log_traj, log_valid, road_edges)
    sim_features = _sim_features_chunked(rollouts, log_valid, road_edges)
    result = metametric.compute(log_features, sim_features)

    bucket_totals["kinematic"].append(result.kinematic_score)
    bucket_totals["interactive"].append(result.interactive_score)
    bucket_totals["map"].append(result.map_score)
    bucket_totals["normalized_metametric"].append(result.normalized_metametric)
    scored_scenarios += 1
    last_result = result

print(f"WosacMetametric over {scored_scenarios} scenarios, K={N_EVAL_ROLLOUTS} rollouts:")
for name in ("kinematic", "interactive", "map", "normalized_metametric"):
    values = bucket_totals[name]
    print(f"  {name:<22} {sum(values) / len(values):.4f}")
# Expected output (approximate — noisy rollouts around the logged scenario):
# WosacMetametric over 20 scenarios, K=32 rollouts:
#   kinematic              x.xxxx
#   interactive            x.xxxx
#   map                    x.xxxx
#   normalized_metametric  x.xxxx

assert last_result is not None
print()
print("Per-feature likelihoods (last scenario):")
for name, score in sorted(last_result.per_feature.items()):
    print(f"  {name:<28} {score:.4f}")
print(
    f"  active_weight_sum = {last_result.active_weight_sum:.2f} "
    "(time-to-collision and traffic-light features need box/light data)"
)

# The estimator layer is directly accessible — here the linear-speed
# histogram estimator scores the logged speeds under the 32-rollout pool.
speed_config = WOSAC_2025_METAMETRIC_CONFIG.linear_speed
speed_log_likelihood = log_likelihood_estimate_timeseries(
    speed_config, log_features.linear_speed[0], sim_features.linear_speed
)
print()
print(f"linear_speed log-likelihood shape: {speed_log_likelihood.shape}  [agents, steps]")
print(f"mean likelihood: {float(jnp.exp(jnp.nanmean(speed_log_likelihood))):.4f}")

# %% [markdown]
"""
## 8. MetricsDashboard: Generating Reports

`MetricsDashboard` calls `calibrax.exporters.PublicationGenerator` to write:

- `output_dir/motion/table.csv` — per-agent-type metrics
- `output_dir/motion/table.html` — same data as HTML
- `output_dir/violations/table.csv` — if kinematic violations are present

Pass a `MetricsReport` with populated `metric_values`. The dashboard groups
rows by agent type automatically from the `"TYPE/metricName"` key structure.
"""

# %%
# Aggregate all scenarios with sim-agent metrics enabled
sim_reports = []
for batch, predictor in zip(batches, predictors):
    sim_reports.append(runner_with_sim.run(predictor, [batch]))

all_motion_vals: dict[str, list[float]] = defaultdict(list)
all_kin_vals: dict[str, list[float]] = defaultdict(list)
for rep in sim_reports:
    for key, val in rep.metric_values.items():
        all_motion_vals[key].append(val)
    for key, val in rep.kinematic_violation_summary.items():
        all_kin_vals[key].append(val)

full_report = MetricsReport(
    metric_values={k: sum(v) / len(v) for k, v in all_motion_vals.items()},
    kinematic_violation_summary={k: sum(v) / len(v) for k, v in all_kin_vals.items()},
)

with tempfile.TemporaryDirectory() as tmp_dir:
    dashboard = MetricsDashboard(tmp_dir)
    output_path = dashboard.generate(full_report)

    motion_csv = output_path / "motion" / "table.csv"
    print("motion/table.csv:")
    print(motion_csv.read_text())

    violations_csv = output_path / "violations" / "table.csv"
    if violations_csv.exists():
        print("violations/table.csv:")
        print(violations_csv.read_text())
# Expected output:
# motion/table.csv:
# Framework,MissRate,mAP,minADE,minFDE
# VEHICLE,x.xx,x.xx,x.xx,x.xx
# PEDESTRIAN,x.xx,x.xx,x.xx,x.xx
# CYCLIST,x.xx,x.xx,x.xx,x.xx

# %% [markdown]
"""
## 9. Physics Violation Summary

`EvaluationRunner` populates `MetricsReport.kinematic_violation_summary` when
`sim_agent_metrics` is configured and batches include `"road_edge_distances"`.

The four violation rate keys (all over finite (agent, timestep) pairs; the
central-difference boundary pads are excluded):

| Key | Description |
|-----|-------------|
| `speed_violation_rate` | Fraction with speed outside [0, 25] m/s (WOSAC 2025 envelope) |
| `accel_violation_rate` | Fraction with linear acceleration outside ±12 m/s² |
| `offroad_rate` | Fraction with signed road-edge distance > 0 (positive = off-road) |
| `collision_rate` | Fraction with nearest-agent centre distance < threshold |

A value of 0.0 means no violations; 1.0 means all agent-timestep pairs violated.
"""

# %%
print(f"Physics violation summary ({len(sim_reports)}-scenario average):")
for key, val in sorted(full_report.kinematic_violation_summary.items()):
    pct = val * 100
    status = "OK" if val < 0.05 else "WARNING"
    print(f"  {key:<28} {val:.4f}  ({pct:.1f}%)  [{status}]")
# Expected output (real scenarios, noisy but calibrated predictor):
# Physics violation summary (20-scenario average):
#   accel_violation_rate         x.xxxx  (x.x%)   [OK/WARNING]
#   collision_rate               x.xxxx  (x.x%)   [OK/WARNING]
#   offroad_rate                 x.xxxx  (x.x%)   [OK/WARNING]
#   speed_violation_rate         x.xxxx  (x.x%)   [OK/WARNING]

# %% [markdown]
"""
## 10. Next Steps

### Experiments to Try

1. Replace `_NoisyGtPredictor` with a trained `TrajectoryDiffusionModel`
   and compare minADE / MissRate against the noise baseline
2. Set `noise_max_m=0` on `_NoisyGtPredictor` to verify oracle behaviour:
   ADE=0, MissRate=0, and per-feature metametric likelihoods driven by the
   degenerate single-bin rollout distributions
3. Sweep `noise_max_m` in {0.5, 1.5, 3.0} and watch `normalized_metametric`
   fall as the rollout distributions widen away from the logged scenario
4. Increase `N_SCENARIOS` toward the full 333-scenario split for tighter
   averages

### Related Examples

- [WOD Metrics Quick Reference](../evaluation/wod-metrics-quickref.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
- [DPO Fine-Tuning Quick Reference](../alignment/dpo-finetuning-quickref.md)
- [Occupancy Flow Tutorial](../advanced/occupancy-flow-tutorial.md)

### API Reference

- [evaluation.metrics](../../api/evaluation/metrics.md)
- [evaluation.wosac_metametric](../../api/evaluation/wosac_metametric.md)
- [evaluation.estimators](../../api/evaluation/estimators.md)
- [evaluation.runner](../../api/evaluation/runner.md)
- [evaluation.dashboard](../../api/evaluation/dashboard.md)
"""
