# DPO Fine-Tuning Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~10 min (CPU) |
| **Prerequisites** | JAX arrays, reward functions, DPO basics, Preference Construction example |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~10 min CPU) demonstrating how to fine-tune a
trajectory diffusion model using Direct Preference Optimization (DPO).
The trainer implements the Diffusion-DPO approach (Wallace et al. 2023),
estimating trajectory log-probabilities via Monte Carlo noise prediction
error and aligning the model toward safer trajectories.

## Files

- **Python Script**: [`examples/alignment/02_dpo_finetuning_quickref.py`](https://github.com/avitai/simulacrax/blob/main/examples/alignment/02_dpo_finetuning_quickref.py)
- **Jupyter Notebook**: [`examples/alignment/02_dpo_finetuning_quickref.ipynb`](https://github.com/avitai/simulacrax/blob/main/examples/alignment/02_dpo_finetuning_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/alignment/02_dpo_finetuning_quickref.py`

## What You'll Learn

1. Create a trajectory diffusion model and optimizer for DPO training
2. Build preference pairs from synthetic candidates using SafetyReward
3. Configure the DPO trainer with Diffusion-DPO log-probability estimation
4. Run training steps and interpret alignment metrics
5. Understand standard DPO vs reference-free (SimPO) modes

## Prerequisites

- Simulacrax installed (`uv sync`)
- JAX arrays, reward functions, DPO basics
- Familiarity with [Preference Construction](preference-construction-quickref.md)

## Pipeline Overview

```
TrajectoryDiffusionModel      PreferenceBatch.to_dpo_batch()
    |                              |
    +-- create_reference_model()   +-- {"chosen", "rejected", "scene_contexts"}
    |   (frozen copy via           |
    |    nnx.clone)                |
    v                              v
DPOAlignmentTrainer
    |-- compute_trajectory_log_prob()  (Monte Carlo Diffusion-DPO)
    |-- compute_dpo_loss()             (DPO objective + optional physics)
    +-- train_step()                   (gradient update)
                |
                v
        DPOAlignmentMetrics
```

## Quick Usage

```python
from simulacrax.alignment import (
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    create_reference_model,
)
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

# Create optimizer and reference model
tx = create_optimizer(OptimizerConfig(learning_rate=1e-3, gradient_clip=1.0))
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
reference = create_reference_model(model)

# Train
trainer = DPOAlignmentTrainer(
    model=model, optimizer=optimizer,
    config=DPOAlignmentConfig(beta=0.1),
    reference_model=reference,
)
metrics = trainer.train_step(dpo_batch, jax.random.key(0))
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.1 | DPO temperature (preference strength) |
| `num_log_prob_samples` | 4 | Monte Carlo samples for log-prob estimation |
| `label_smoothing` | 0.0 | Robust DPO label smoothing |
| `reference_free` | False | SimPO mode (no reference model) |
| `physics_weight` | 0.0 | Physics regularisation weight |

## Terminal Output

```
Model created: 32d, 2 layers
Candidates shape: (8, 3, 8, 4)
DPO batch keys: ['chosen', 'rejected', 'scene_contexts']
Chosen shape:   (2, 3, 8, 4)
Rejected shape: (2, 3, 8, 4)
Reference model created (frozen copy via nnx.clone)
DPO config: beta=0.1, K=4
DPO trainer ready
Step 0: loss=0.6698 acc=100.00% margin=0.4942 grad_norm=0.168476
Step 1: loss=0.6767 acc=100.00% margin=0.4957 grad_norm=0.140427
Step 2: loss=0.6861 acc=100.00% margin=0.3341 grad_norm=0.121241
--- Final Metrics ---
DPO loss:              0.6861
Policy chosen log-p:   -1.3181
Policy rejected log-p: -1.6522
Reward accuracy:       100.00%
Reward margin:         0.3341
Gradient norm:         0.121241
```

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `NoiseSchedule` | artifex | Forward diffusion for log-prob estimation |
| `create_optimizer` | opifex | Gradient-clipped optimizer creation |
| `nnx.clone` | Flax NNX | Deep-copy for frozen reference model |

## Related

- [Preference Construction Quick Reference](preference-construction-quickref.md)
- [DPOAlignmentTrainer API](../../api/alignment/dpo_trainer.md)
- [Alignment User Guide](../../user-guide/alignment.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
