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
# WOD Evaluation Metrics Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min (CPU) |
| **Prerequisites** | JAX arrays, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This quick reference evaluates trajectory predictions against real Waymo Open
Dataset (WOD) Motion ground truth. All metric functions are pure JAX —
differentiable, JIT-compatible, and usable directly as training objectives.

Evaluation is restricted to the challenge-designated `state/tracks_to_predict`
agents, so padding rows (WOD fills invalid agents with -1 sentinels) never
enter the metrics and agent types come from the real `state/type` codes.

Input shapes follow the `waymo_open_dataset.metrics.python.motion_metrics
.get_motion_metric_ops` convention. One deliberate simplification: miss rate
and mAP here use a single 2.0 m Euclidean final-displacement threshold,
whereas the official metrics evaluate at 2 Hz (3 s / 5 s / 8 s horizons)
with lateral/longitudinal thresholds that scale with the agent's speed.
Numbers from this quickref are therefore not directly comparable with
leaderboard values.

```
WODSource → tracks_to_predict ground truth + agent types + validity masks
                          ↓
ade / fde / min_ade / min_fde / miss_rate  →  per-agent scalar metrics
                          ↓
MotionMetrics.compute(predictions, scores, ground_truth, ...)
    → {"VEHICLE/minADE": ..., "PEDESTRIAN/MissRate": ..., ...}
                          ↓
EvaluationRunner.run(model, batches) → MetricsReport
                          ↓
MetricsDashboard.generate(report) → motion/table.csv + .html
```

## What You'll Learn

1. Extract WOD ground truth into evaluation-ready JAX arrays
2. Compute ADE, FDE, minADE, minFDE, and miss rate on real trajectories
3. Run `MotionMetrics` for per-agent-type WOD breakdown across real scenarios
4. Accumulate metrics over the full 333-scenario split with `EvaluationRunner`
5. Export per-type tables with `MetricsDashboard`

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | diffav.data | Real WOD TFRecord loading |
| `PublicationGenerator` | calibrax | Table export (CSV, HTML) |
"""

# %% [markdown]
r"""
## Setup

### Required Data

Download at least one WOD Motion validation shard:

```bash
WOD_BUCKET="gs://waymo_open_dataset_motion_v_1_2_1"
WOD_SHARD="uncompressed/tf_example/validation"
SHARD_FILE="validation_tfexample.tfrecord-00000-of-00150"
gsutil -m cp "$WOD_BUCKET/$WOD_SHARD/$SHARD_FILE" \
    /path/to/waymo/motion_v1.2.1/tf_example/validation/
```

### Environment

Create a `.env` file in the project root:

```bash
WOD_MOTION_TFRECORD_PATH=/path/to/waymo/motion_v1.2.1/tf_example
```

### Installation

```bash
uv sync
```
"""

# %%
# Imports
import tempfile
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from dotenv import load_dotenv

from diffav.core.types import MetricsReport, TrajectoryPrediction
from diffav.data import resolve_wod_tfrecord_path
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.evaluation import (
    ade,
    EvaluationRunner,
    fde,
    MetricsDashboard,
    min_ade,
    min_fde,
    miss_rate,
    MotionMetrics,
    MotionMetricsConfig,
)


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 80  # 8 s at 10 Hz
MAX_AGENTS = 128  # tf_example native width — keeps every tracks_to_predict row
K_HYPOTHESES = 6  # trajectory hypotheses per agent

# %% [markdown]
"""
## 1. Load Real WOD Data

`WODSource` loads Waymo Open Dataset Motion TFRecords, aggregates past/current/future
timesteps into `state/all/*` arrays, and exposes each scenario as a datarax `Element`.
The future slice (timesteps `HIST_STEPS:`—90) provides the 80-step ground truth.

The benchmark below runs over the full split; the local single-shard setup
holds 333 validation scenarios.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
N_SCENARIOS = len(source)
print(f"Loaded {N_SCENARIOS} validation scenarios")
print(f"Evaluating all {N_SCENARIOS} scenarios, K={K_HYPOTHESES} hypotheses each")
# Expected output:
# Loaded 333 validation scenarios
# Evaluating all 333 scenarios, K=6 hypotheses each


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


# %% [markdown]
"""
## 2. Pure JAX Metric Functions on Real Trajectories

We compute oracle metrics (prediction = ground truth) to verify correctness,
then demonstrate K=6 hypotheses with *distinct* noise scales — each
hypothesis ramps its lateral noise from 0 m at t=0 to its own maximum at
t=8 s, and hypothesis scores are the softmax of the negative noise scale
(lower noise → higher confidence), not a uniform 1/K.
"""

# %%
element = source[0]
raw = element.data

ttp_rows, ground_truth, gt_valid, object_type = _extract_gt(raw)
print(f"tracks_to_predict rows: {ttp_rows.tolist()}")
# Expected output (scenario 0):
# tracks_to_predict rows: [0, 1, 2]

# First tracks_to_predict agent as the single-agent demo
gt_xy_agent = ground_truth[0, :, :2]  # (80, 2)
valid_agent = gt_valid[0]  # (80,)

# Oracle: prediction = ground truth → ADE/FDE must be exactly 0
oracle_ade = ade(gt_xy_agent, gt_xy_agent, valid_agent)
oracle_fde = fde(gt_xy_agent, gt_xy_agent, valid_agent)
print(f"Oracle ADE  (pred = GT): {float(oracle_ade):.4f} m")
print(f"Oracle FDE  (pred = GT): {float(oracle_fde):.4f} m")

# K=6 hypotheses with distinct final-step noise scales (metres)
noise_scales = jnp.linspace(0.5, 3.0, K_HYPOTHESES)  # (K,)
scores_k = jax.nn.softmax(-noise_scales)  # (K,) — low noise = high confidence
ramp = jnp.linspace(0.0, 1.0, FUTURE_STEPS)  # (80,)
noise_k = (
    jax.random.normal(jax.random.key(0), (K_HYPOTHESES, FUTURE_STEPS, 2))
    * noise_scales[:, jnp.newaxis, jnp.newaxis]
    * ramp[jnp.newaxis, :, jnp.newaxis]
)
pred_k = gt_xy_agent[jnp.newaxis] + noise_k  # (K, 80, 2)

print(f"Hypothesis scores: {np.round(np.asarray(scores_k), 4).tolist()}")
print(f"minADE  (K={K_HYPOTHESES}):    {float(min_ade(pred_k, gt_xy_agent, valid_agent)):.4f} m")
print(f"minFDE  (K={K_HYPOTHESES}):    {float(min_fde(pred_k, gt_xy_agent, valid_agent)):.4f} m")
print(f"MissRate (2 m): {float(miss_rate(pred_k, gt_xy_agent, valid_agent, threshold=2.0)):.4f}")
# Expected output:
# Oracle ADE  (pred = GT): 0.0000 m
# Oracle FDE  (pred = GT): 0.0000 m
# Hypothesis scores: [descending, sums to 1]
# minADE  (K=6):    0.xxxx m
# minFDE  (K=6):    x.xxxx m
# MissRate (2 m): x.xxxx

# %% [markdown]
"""
## 3. MotionMetrics — Per-Agent-Type WOD Breakdown

`MotionMetrics.compute()` accepts the same shapes as
`waymo_open_dataset.metrics.python.motion_metrics.get_motion_metric_ops`
and stratifies results by VEHICLE / PEDESTRIAN / CYCLIST automatically.
Remember the simplification: the 2.0 m Euclidean threshold stands in for
the official speed-scaled 2 Hz criteria.
"""

# %%
m_config = MotionMetricsConfig(
    miss_rate_threshold=2.0,
    top_k=K_HYPOTHESES,
    num_future_steps=FUTURE_STEPS,
)
motion_metrics = MotionMetrics(m_config)

# K hypotheses for all tracks_to_predict agents — per-hypothesis noise scale
num_agents = ground_truth.shape[0]
gt_xy_all = ground_truth[..., :2]  # (N, 80, 2)
noise_all = (
    jax.random.normal(jax.random.key(1), (K_HYPOTHESES, num_agents, FUTURE_STEPS, 2))
    * noise_scales[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    * ramp[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
)
pred_all = gt_xy_all[jnp.newaxis] + noise_all  # (K, N, 80, 2)

# Wrap for MotionMetrics: (B=1, M=1, K, N, T, 2); scores (B=1, M=1, K)
result = motion_metrics.compute(
    predictions=pred_all[jnp.newaxis, jnp.newaxis],
    scores=scores_k[jnp.newaxis, jnp.newaxis],
    ground_truth=ground_truth[jnp.newaxis],
    ground_truth_is_valid=gt_valid[jnp.newaxis],
    object_type=object_type[jnp.newaxis],
)

# Count real agent types among the evaluated rows
for label, code in [("VEHICLE", 0), ("PEDESTRIAN", 1), ("CYCLIST", 2)]:
    n = int(jnp.sum(object_type == code))
    print(f"  {label}: {n} agents")
print("MotionMetrics (single scenario):")
for k, v in sorted(result.items()):
    print(f"  {k}: {v:.4f}")
# Expected output (scenario 0 carries vehicles and a cyclist):
#   VEHICLE: 2 agents
#   PEDESTRIAN: 0 agents
#   CYCLIST: 1 agents
# MotionMetrics (single scenario):
#   CYCLIST/MissRate: x.xxxx
#   CYCLIST/mAP: x.xxxx
#   CYCLIST/minADE: x.xxxx
#   CYCLIST/minFDE: x.xxxx
#   VEHICLE/...

# %% [markdown]
"""
## 4. EvaluationRunner — Full-Split Batch Accumulation

`EvaluationRunner` iterates batches, calls `model.sample()`, and accumulates
metric averages into a single `MetricsReport`. Here it runs over all 333
scenarios, one batch per scenario, restricted to the `tracks_to_predict`
agents of each.

The predictor below simulates model predictions with the same calibrated
noise strategy. Replace with a trained `TrajectoryDiffusionModel` for
production benchmarks.
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


runner = EvaluationRunner(motion_metrics=motion_metrics)
agg: dict[str, list[float]] = defaultdict(list)
n_evaluated = 0

for i, elem in enumerate(source):
    rows_i, gt_i, valid_i, obj_i = _extract_gt(elem.data)
    if rows_i.size == 0:
        continue
    predictor = _NoisyGtPredictor(gt_i)
    batch = {
        "scene_context": gt_i[..., :2].reshape(rows_i.size, -1)[:, :64],
        "ground_truth": gt_i,
        "ground_truth_is_valid": valid_i,
        "object_type": obj_i,
        "key": jax.random.key(i),
    }
    report = runner.run(predictor, [batch])
    for key_str, val in report.metric_values.items():
        agg[key_str].append(val)
    n_evaluated += 1

print(f"\n{n_evaluated}-scenario average metrics (tracks_to_predict agents):")
for key_str in sorted(agg):
    avg = sum(agg[key_str]) / len(agg[key_str])
    print(f"  {key_str}: {avg:.4f}")
# Expected output (approximate — real WOD scenarios, single noisy hypothesis):
# 333-scenario average metrics (tracks_to_predict agents):
#   CYCLIST/MissRate: x.xxxx
#   CYCLIST/mAP: x.xxxx
#   CYCLIST/minADE: x.xxxx
#   ...
#   VEHICLE/minADE: ~0.6-1.2 m (noise_max=1.5 m, K=1)

# %% [markdown]
"""
## 5. MetricsDashboard — Export Per-Type Tables

`MetricsDashboard` writes per-agent-type tables (CSV + HTML) under
`output_dir/motion/` using `calibrax.exporters.PublicationGenerator`.
"""

# %%
avg_metrics = {k: sum(v) / len(v) for k, v in agg.items()}
final_report = MetricsReport(metric_values=avg_metrics)

with tempfile.TemporaryDirectory() as tmp_dir:
    dashboard = MetricsDashboard(tmp_dir)
    output_path = dashboard.generate(final_report)
    motion_csv = output_path / "motion" / "table.csv"
    print("motion/table.csv:")
    print(motion_csv.read_text())
# Expected output (approximate):
# motion/table.csv:
# Framework,MissRate,mAP,minADE,minFDE
# VEHICLE,x.xx,x.xx,x.xx,x.xx
# PEDESTRIAN,x.xx,x.xx,x.xx,x.xx
# CYCLIST,x.xx,x.xx,x.xx,x.xx

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Replace `_NoisyGtPredictor` with a trained `TrajectoryDiffusionModel`
   and observe significantly lower ADE/FDE values
2. Set `noise_max_m=0` to verify oracle metrics: ADE=0, MissRate=0
3. Sweep the miss-rate threshold (1 m / 2 m / 4 m) and watch MissRate and
   mAP move — a reminder that the single Euclidean threshold is a
   simplification of the official speed-scaled criteria

### Related Examples

- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)
- [Trajectory Diffusion Quick Reference](../models/trajectory-diffusion-quickref.md)
- [DPO Fine-Tuning Quick Reference](../alignment/dpo-finetuning-quickref.md)

### API Reference

- [evaluation.metrics](../../api/evaluation/metrics.md)
- [evaluation.runner](../../api/evaluation/runner.md)
- [evaluation.dashboard](../../api/evaluation/dashboard.md)
"""
