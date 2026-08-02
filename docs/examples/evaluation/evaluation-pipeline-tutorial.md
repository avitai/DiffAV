# Evaluation Pipeline Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5 min (CPU), ~1 min (GPU) |
| **Prerequisites** | WOD Metrics Quick Reference, JAX arrays |
| **Format** | Python + Jupyter |

A Tier 2 tutorial walking through the full evaluation pipeline for trajectory
prediction — from building WOD-format data and configuring metric evaluators,
through running multi-scenario batch evaluation, to generating publication-ready
CSV and HTML reports. All steps use mock JAX arrays and run without WOD TFRecord
files.

## Files

- **Python Script**: [`examples/evaluation/02_evaluation_pipeline_tutorial.py`](https://github.com/avitai/simulacrax/blob/main/examples/evaluation/02_evaluation_pipeline_tutorial.py)
- **Jupyter Notebook**: [`examples/evaluation/02_evaluation_pipeline_tutorial.ipynb`](https://github.com/avitai/simulacrax/blob/main/examples/evaluation/02_evaluation_pipeline_tutorial.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/evaluation/02_evaluation_pipeline_tutorial.py`

## What You'll Learn

1. Configure `EvaluationRunner` with `MotionMetrics` and run it over mock batches
2. Build WOD-format mock predictions and ground truth arrays without TFRecord files
3. Aggregate per-scenario results into a single `MetricsReport`
4. Break down metrics per agent type: VEHICLE / PEDESTRIAN / CYCLIST
5. Compute `SimAgentMetrics`: kinematic, interactive, and map realism scores
6. Generate publication-ready reports with `MetricsDashboard`
7. Interpret the physics violation summary from `kinematic_violation_summary`

## Prerequisites

- Simulacrax installed (`uv sync`)
- Familiarity with [WOD Metrics Quick Reference](wod-metrics-quickref.md)
- JAX arrays and Flax NNX basics

## Pipeline Overview

```
mock predictions (numpy/JAX arrays)
              ↓
MotionMetrics.compute(predictions, scores, ground_truth, ...)
    → {"VEHICLE/minADE": ..., "PEDESTRIAN/MissRate": ..., ...}
              ↓
SimAgentMetrics.compute(trajectories, road_edge_distances)
    → SimAgentMetricsResult(kinematic, interactive, map, metametric)
              ↓
EvaluationRunner.run(model, batches) → MetricsReport
              ↓
MetricsDashboard.generate(report) → motion/table.csv + .html
```

## Section Overview

| Section | Topic |
|---------|-------|
| 1. Overview | Evaluation stack layers and data flow |
| 2. Setup | `EvaluationRunner` + `MotionMetrics` configuration |
| 3. Mock Data | Building WOD-format batches without TFRecord files |
| 4. Running | `EvaluationRunner.run()` over N scenarios |
| 5. Per-Agent-Type | VEHICLE / PEDESTRIAN / CYCLIST breakdowns |
| 6. SimAgentMetrics | Kinematic, interactive, and map scores |
| 7. MetricsDashboard | CSV + HTML report generation |
| 8. Physics Violations | `kinematic_violation_summary` interpretation |
| 9. Next Steps | Experiments and related examples |

## Quick Usage

```python
import jax
import jax.numpy as jnp
from simulacrax.evaluation import (
    EvaluationRunner,
    MetricsDashboard,
    MotionMetrics,
    MotionMetricsConfig,
    SimAgentMetrics,
    SimAgentMetricsConfig,
)

# Configure
motion_config = MotionMetricsConfig(miss_rate_threshold=2.0, top_k=6)
sim_config    = SimAgentMetricsConfig(kinematic_weight=0.4, interactive_weight=0.4)

runner = EvaluationRunner(
    motion_metrics=MotionMetrics(motion_config),
    sim_agent_metrics=SimAgentMetrics(sim_config),
)

# Run (replace model + batches with your WODSource-based data)
report = runner.run(model, batches)

# Inspect results
print(report.metric_values)
# → {"VEHICLE/minADE": 0.82, "VEHICLE/MissRate": 0.12, ...}
print(report.kinematic_violation_summary)
# → {"speed_violation_rate": 0.03, "collision_rate": 0.07, ...}

# Export
dashboard = MetricsDashboard("output/evaluation")
output_path = dashboard.generate(report)
# Writes: output/evaluation/motion/table.csv + table.html
#         output/evaluation/violations/table.csv
```

## Output Reference

### MotionMetrics Output Keys

Each key follows the pattern `"TYPE/metricName"`:

| Key | Description |
|-----|-------------|
| `VEHICLE/minADE` | Best-of-K average displacement error for vehicles (m) |
| `VEHICLE/minFDE` | Best-of-K final displacement error for vehicles (m) |
| `VEHICLE/MissRate` | Fraction of vehicles where all K hypotheses exceed 2 m at final step |
| `VEHICLE/mAP` | Mean average precision for vehicles (ranked hypotheses) |
| `PEDESTRIAN/minADE` | Same metrics for pedestrian agents |
| `PEDESTRIAN/minFDE` | |
| `PEDESTRIAN/MissRate` | |
| `PEDESTRIAN/mAP` | |
| `CYCLIST/minADE` | Same metrics for cyclist agents |
| `CYCLIST/minFDE` | |
| `CYCLIST/MissRate` | |
| `CYCLIST/mAP` | |

### SimAgentMetrics Violation Summary Keys

| Key | Description |
|-----|-------------|
| `speed_violation_rate` | Fraction of (agent, timestep) pairs with speed outside [0, 50 m/s] |
| `accel_violation_rate` | Fraction of (agent, timestep) pairs with \|Δspeed\| > 6 m/s² |
| `offroad_rate` | Fraction of (agent, timestep) pairs with road-edge distance ≤ 0 |
| `collision_rate` | Fraction of (agent, timestep) pairs within collision threshold |

### MetricsDashboard CSV Format

```
Framework,MissRate,mAP,minADE,minFDE
VEHICLE,0.12,0.85,0.82,1.94
PEDESTRIAN,0.08,0.91,0.54,1.22
CYCLIST,0.15,0.79,0.95,2.10
```

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | simulacrax.data | Real WOD TFRecord loading for production batches |
| `PublicationGenerator` | calibrax | Table export to CSV and HTML |
| `TrajectoryDiffusionModel` | simulacrax.models | Production model for `runner.run()` |

## Related

- [WOD Metrics Quick Reference](wod-metrics-quickref.md)
- [Evaluation User Guide](../../user-guide/evaluation.md)
- [EvaluationRunner API](../../api/evaluation/runner.md)
- [MotionMetrics API](../../api/evaluation/metrics.md)
- [MetricsDashboard API](../../api/evaluation/dashboard.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
