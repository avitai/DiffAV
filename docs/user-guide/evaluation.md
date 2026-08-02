# Evaluation

Simulacrax provides a full evaluation stack for trajectory prediction: pure JAX
metric functions, a WOD-compatible aggregator (`MotionMetrics`), a simulation
realism evaluator (`SimAgentMetrics`), a batch orchestrator (`EvaluationRunner`),
and a publication-ready table exporter (`MetricsDashboard`). All metric functions
are differentiable, JIT-compatible, and directly usable as training objectives.

## Architecture Overview

```
WODSource → real WOD scenarios
                  ↓
ade / fde / min_ade / min_fde / miss_rate / mean_average_precision
          (pure JAX, per-trajectory scalar metrics)
                  ↓
MotionMetrics.compute(predictions, scores, ground_truth, ...)
          → {"VEHICLE/minADE": ..., "PEDESTRIAN/MissRate": ..., ...}
                  ↓
SimAgentMetrics.compute(trajectories, road_edge_distances)
          → SimAgentMetricsResult(kinematic_score, interactive_score, map_score, metametric)
                  ↓
EvaluationRunner.run(model, batches)
          → MetricsReport(metric_values, kinematic_violation_summary)
                  ↓
MetricsDashboard.generate(report)
          → motion/table.csv + motion/table.html + violations/table.csv
```

## Pure JAX Metric Functions

The low-level functions operate on single-agent trajectory arrays and return
scalar metrics. They are differentiable and can be used as training losses.

### Input Shapes

| Symbol | Shape | Description |
|--------|-------|-------------|
| `pred` | `(T, 2)` | Predicted xy positions |
| `gt` | `(T, 2)` | Ground truth xy positions |
| `valid` | `(T,)` | Float validity mask (1=valid, 0=invalid) |
| `pred_k` | `(K, T, 2)` | K trajectory hypotheses |

### Functions

```python
from simulacrax.evaluation import ade, fde, min_ade, min_fde, miss_rate

# Average Displacement Error — mean L2 over valid steps
ade_value = ade(pred, gt, valid)               # → scalar

# Final Displacement Error — L2 at last valid step
fde_value = fde(pred, gt, valid)               # → scalar

# Oracle ADE over K hypotheses (best-of-K)
min_ade_value = min_ade(pred_k, gt, valid)     # → scalar

# Oracle FDE over K hypotheses
min_fde_value = min_fde(pred_k, gt, valid)     # → scalar

# Miss rate — 1.0 if all K hypotheses exceed threshold at last valid step
mr_value = miss_rate(pred_k, gt, valid, threshold=2.0)  # → scalar
```

### Using as Training Losses

Because all functions are pure JAX, they compose naturally with Flax NNX:

```python
import jax
from flax import nnx

def loss_fn(model: nnx.Module, batch: dict) -> jax.Array:
    pred = model.sample(batch["scene_context"], key=batch["key"])
    pred_xy = pred.trajectories[..., :2]  # (A, T, 2)
    return jax.vmap(lambda p, g, v: ade(p, g, v))(
        pred_xy, batch["gt_xy"], batch["valid"]
    ).mean()
```

## MotionMetrics — WOD-Compatible Aggregator

`MotionMetrics` computes per-agent-type (VEHICLE / PEDESTRIAN / CYCLIST) metric
averages using the same input shapes as
`waymo_open_dataset.metrics.python.motion_metrics.get_motion_metric_ops`.

### Input Convention

Following the WOD convention, `predictions` has shape `(B, M, K, N, T, 2)`:

| Dimension | Symbol | Description |
|-----------|--------|-------------|
| B | batch size | Scenarios per call |
| M | model groups | Prediction groups per scenario |
| K | hypotheses | Trajectory hypotheses per group |
| N | agents | Agents per group |
| T | time steps | Future steps (80 for 8 s at 10 Hz) |
| 2 | xy | Predicted positions |

```python
from simulacrax.evaluation import MotionMetrics, MotionMetricsConfig

config  = MotionMetricsConfig(miss_rate_threshold=2.0, top_k=6, num_future_steps=80)
metrics = MotionMetrics(config)

result = metrics.compute(
    predictions=pred_all[jnp.newaxis, jnp.newaxis],   # (1, 1, K, A, T, 2)
    scores=jnp.ones((1, 1, K)) / K,                   # (B, M, K)
    ground_truth=ground_truth[jnp.newaxis],            # (B, A, T, 7)
    ground_truth_is_valid=gt_valid[jnp.newaxis],       # (B, A, T)
    object_type=object_type[jnp.newaxis],              # (B, A)
)
# → {"VEHICLE/minADE": ..., "VEHICLE/minFDE": ..., "VEHICLE/MissRate": ...,
#    "VEHICLE/mAP": ..., "PEDESTRIAN/...", "CYCLIST/..."}
```

### Relationship to the Official WOD Metrics

Input shapes follow `waymo_open_dataset`'s `get_motion_metric_ops`
convention, but miss rate and mAP use a single Euclidean
final-displacement threshold (`miss_rate_threshold`) — a simplification
of WOD's speed-scaled lateral/longitudinal criteria. For official
leaderboard numbers, use `waymo_open_dataset` directly.

## SimAgentMetrics — Simulation Realism Evaluator

`SimAgentMetrics` scores simulation realism with envelope-feasibility
proxies aligned with the WOSAC buckets (kinematic features are derived
from positions and headings via central differences at 10 Hz and checked
against the 2025 challenge envelope; each rollout is scored
individually). These are feasibility rates, not the WOSAC histogram
log-likelihood metametric:

| Bucket | Weight | Features |
|--------|--------|---------|
| Kinematic | 0.4 | Speed/acceleration and angular rates within the 2025 envelope |
| Interactive | 0.4 | Agent-to-agent collision avoidance |
| Map | 0.2 | Offroad and road-edge distance |

```python
from simulacrax.evaluation import SimAgentMetrics, SimAgentMetricsConfig

config  = SimAgentMetricsConfig(
    kinematic_weight=0.4,
    interactive_weight=0.4,
    map_weight=0.2,
    collision_threshold=1.0,
)
sim_metrics = SimAgentMetrics(config)

result = sim_metrics.compute(
    trajectories=pred_traj,                  # (K, N, T, 4) [x, y, heading, speed]
    road_edge_distances=road_edge_dists,     # (A, T)
)

print(f"Kinematic: {result.kinematic_score:.4f}")
print(f"Interactive: {result.interactive_score:.4f}")
print(f"Map: {result.map_score:.4f}")
print(f"Metametric: {result.metametric:.4f}")
```

## EvaluationRunner — Batch Orchestration

`EvaluationRunner` iterates over evaluation batches, calls `model.sample()`, and
accumulates metric averages into a `MetricsReport`.

### Model Interface

The model seam is the `TrajectorySampler` protocol (exported from
`simulacrax.evaluation`): any object with a `sample()` method of this shape
can be evaluated — `TrajectoryDiffusionModel` satisfies it structurally.

```python
def sample(
    self,
    scene_context: jax.Array,   # (num_agents, context_dim) — one row per agent
    *,
    key: jax.Array,
) -> TrajectoryPrediction:
    ...
```

### Running Evaluation

```python
from simulacrax.evaluation import EvaluationRunner, MotionMetrics, SimAgentMetrics

runner = EvaluationRunner(
    motion_metrics=MotionMetrics(MotionMetricsConfig()),
    sim_agent_metrics=SimAgentMetrics(SimAgentMetricsConfig()),  # optional
)

report = runner.run(model, batches)

print(report.metric_values)
# → {"VEHICLE/minADE": 0.82, "VEHICLE/MissRate": 0.12, ...}

print(report.kinematic_violation_summary)
# → {"speed_violation_rate": 0.03, "collision_rate": 0.07, ...}
```

### Batch Format

Each batch is a dict with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `scene_context` | `(A, context_dim)` | Per-agent scene embedding from `WODSource` — one row per agent |
| `ground_truth` | `(A, T, 7)` | WOD state tensor |
| `ground_truth_is_valid` | `(A, T)` | Float validity mask |
| `object_type` | `(A,)` | Int agent types (0=vehicle, 1=ped, 2=cyclist) |
| `key` | `jax.Array` | JAX PRNG key |
| `road_edge_distances` | `(A, T)` | *(optional)* For `SimAgentMetrics` |

## MetricsDashboard — Publication-Ready Export

`MetricsDashboard` writes per-group tables (CSV + HTML) and an optional
physics violation summary under a configurable output directory. Metric keys
prefixed `VEHICLE/`, `PEDESTRIAN/`, or `CYCLIST/` form per-agent-type groups;
un-prefixed keys — the shape `ScenarioMiner.evaluate_planner` reports
(`"ade"`, `"fde"`) — render as an `ALL` group. The dashboard fails fast: a
report with nothing to render, or a metric key with an unrecognised
agent-type prefix, raises `ValueError` instead of silently writing nothing.

```python
from simulacrax.evaluation import MetricsDashboard
from simulacrax.core.types import MetricsReport

dashboard = MetricsDashboard("output/evaluation")
output_path = dashboard.generate(report)

# Written files:
#   output/evaluation/motion/table.csv
#   output/evaluation/motion/table.html
#   output/evaluation/violations/table.csv  (if violations present)
```

### CSV Format

```
Framework,MissRate,mAP,minADE,minFDE
VEHICLE,0.12,0.85,0.82,1.94
PEDESTRIAN,0.08,0.91,0.54,1.22
CYCLIST,0.15,0.79,0.95,2.10
```

## End-to-End Example

See the [WOD Metrics Quick Reference](../examples/evaluation/wod-metrics-quickref.md)
for a complete, runnable example using real WOD TFRecord data.

## Related

- [WOD Metrics Quick Reference](../examples/evaluation/wod-metrics-quickref.md)
- [EvaluationRunner API](../api/evaluation/runner.md)
- [MotionMetrics API](../api/evaluation/metrics.md)
- [MetricsDashboard API](../api/evaluation/dashboard.md)
- [TrajectoryDiffusionModel](../user-guide/models.md)
