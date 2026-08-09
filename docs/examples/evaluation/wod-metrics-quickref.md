# WOD Evaluation Metrics Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min (CPU) |
| **Prerequisites** | JAX arrays, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

A Tier 1 quick reference demonstrating JAX-native WOD-compatible evaluation metrics on real
Waymo Open Dataset trajectories. All functions are differentiable and JIT-compatible, following
the input shape convention of
`waymo_open_dataset.metrics.python.motion_metrics.get_motion_metric_ops`.

## Files

- **Python Script**: [`examples/evaluation/01_wod_metrics_quickref.py`](https://github.com/avitai/DiffAV/blob/main/examples/evaluation/01_wod_metrics_quickref.py)
- **Jupyter Notebook**: [`examples/evaluation/01_wod_metrics_quickref.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/evaluation/01_wod_metrics_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/evaluation/01_wod_metrics_quickref.py`

## What You'll Learn

1. Extract WOD ground truth into evaluation-ready JAX arrays
2. Compute ADE, FDE, minADE, minFDE, and miss rate on real trajectories
3. Run `MotionMetrics` for per-agent-type WOD breakdown across real scenarios
4. Accumulate multi-scenario metrics with `EvaluationRunner`
5. Export per-type tables with `MetricsDashboard`

## Prerequisites

- DiffAV installed (`uv sync`)
- At least one WOD Motion validation TFRecord shard (see Setup in the example)
- `WOD_MOTION_TFRECORD_PATH` env var pointing to the TFRecord directory

## Pipeline Overview

```
WODSource → real ground truth positions + agent types + validity masks
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

## Quick Usage

```python
from diffav.evaluation import (
    ade, fde, min_ade, min_fde, miss_rate,
    MotionMetrics, MotionMetricsConfig,
    EvaluationRunner, MetricsDashboard,
)

# Pure JAX functions — differentiable, JIT-able
ade_value = ade(pred, gt, valid)       # (T,2),(T,2),(T,) → scalar
fde_value = fde(pred, gt, valid)
best_k    = min_ade(pred_k, gt, valid) # (K,T,2) → scalar

# WOD-compatible per-type breakdown
config  = MotionMetricsConfig(miss_rate_threshold=2.0, top_k=6)
metrics = MotionMetrics(config)
result  = metrics.compute(predictions, scores, ground_truth, gt_valid, object_type)
# → {"VEHICLE/minADE": 0.0, "PEDESTRIAN/MissRate": 0.0, ...}

# Batch evaluation loop
runner = EvaluationRunner(motion_metrics=metrics)
report = runner.run(model, batches)
```

## Configuration Options

### MotionMetricsConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `miss_rate_threshold` | 2.0 | Displacement threshold for miss-rate (metres) |
| `top_k` | 6 | Number of trajectory hypotheses per prediction group |
| `num_future_steps` | 80 | Prediction horizon (WOD: 80 steps = 8 s at 10 Hz) |

## Terminal Output

```
Loaded 333 validation scenarios
Evaluating first 5 scenarios, K=6 hypotheses each
Oracle ADE  (pred = GT): 0.0000 m
Oracle FDE  (pred = GT): 0.0000 m
minADE  (K=6):    0.xxxx m
minFDE  (K=6):    x.xxxx m
MissRate (2 m): x.xxxx
  VEHICLE: N agents
  PEDESTRIAN: N agents
  CYCLIST: N agents
MotionMetrics (single scenario):
  CYCLIST/MissRate: x.xxxx
  CYCLIST/mAP: x.xxxx
  CYCLIST/minADE: x.xxxx
  CYCLIST/minFDE: x.xxxx
  PEDESTRIAN/MissRate: x.xxxx
  ...
  VEHICLE/minADE: x.xxxx
  ...

5-scenario average metrics:
  CYCLIST/MissRate: x.xxxx
  ...
  VEHICLE/minADE: ~0.6-1.2 m (noise_max=1.5 m, K=1)
motion/table.csv:
Framework,MissRate,mAP,minADE,minFDE
VEHICLE,x.xx,x.xx,x.xx,x.xx
PEDESTRIAN,x.xx,x.xx,x.xx,x.xx
CYCLIST,x.xx,x.xx,x.xx,x.xx
```

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | diffav.data | Real WOD TFRecord loading |
| `PublicationGenerator` | calibrax | Table export (CSV, HTML) |

## Related

- [EvaluationRunner API](../../api/evaluation/runner.md)
- [MotionMetrics API](../../api/evaluation/metrics.md)
- [MetricsDashboard API](../../api/evaluation/dashboard.md)
- [Evaluation User Guide](../../user-guide/evaluation.md)
- [DPO Fine-Tuning Quick Reference](../alignment/dpo-finetuning-quickref.md)
