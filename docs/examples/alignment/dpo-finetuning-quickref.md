# DPO Fine-Tuning Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~10 min (GPU with checkpoint; small-model CPU fallback) |
| **Prerequisites** | JAX arrays, reward functions, DPO basics, Preference Construction example |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~10 min CPU) demonstrating how to fine-tune a
trajectory diffusion model using Direct Preference Optimization (DPO).
The trainer implements the Diffusion-DPO approach (Wallace et al. 2023),
estimating trajectory log-probabilities via Monte Carlo noise prediction
error and aligning the model toward safer trajectories.

## Files

- **Python Script**: [`examples/alignment/02_dpo_finetuning_quickref.py`](https://github.com/avitai/DiffAV/blob/main/examples/alignment/02_dpo_finetuning_quickref.py)
- **Jupyter Notebook**: [`examples/alignment/02_dpo_finetuning_quickref.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/alignment/02_dpo_finetuning_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/alignment/02_dpo_finetuning_quickref.py`

## What You'll Learn

1. Create a trajectory diffusion model and optimizer for DPO training
2. Build preference pairs from real WOD trajectories (GT + noise candidates) using SafetyReward
3. Configure the DPO trainer with Diffusion-DPO log-probability estimation
4. Run training steps and interpret alignment metrics
5. Understand standard DPO vs reference-free (SimPO) modes

## Prerequisites

- DiffAV installed (`uv sync`)
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
from diffav.alignment import (
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    create_reference_model,
)
from substrax.optim import create_optimizer, OptimizerConfig

# Create optimizer and reference model
optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3, gradient_clip_norm=1.0))
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

The output below is from a run on a Modal L4 GPU (`modal run deploy/modal_app.py --task examples`), with deterministic GPU kernels and WOD read from the data volume. The image carries no `checkpoints/wod-mini`, so the example ran its untrained fallback, as it does anywhere that checkpoint is absent.

```text
WARNING: no checkpoint at checkpoints/wod-mini — using an untrained model.
A random reference makes DPO's implicit reward meaningless; train one with
`python scripts/train_wod.py` for real results.
Loaded 4815 validation scenarios
Ground truth: (8, 80, 4), scene context: (8, 128)
GT score (noise=0):           -10.4618  ← should be highest
Corrupted score (noise=3.5m): -59.9994  ← should be lowest
GT outperforms corrupted:     True
DPO batch keys: ['chosen', 'rejected', 'scene_contexts']
Chosen shape:   (2, 8, 80, 4)
Rejected shape: (2, 8, 80, 4)
Reference model created (frozen copy of the restored policy)
DPO beta=1000.0, K=8 MC samples
DPO trainer ready
Step 0: loss=0.6931 acc=0.00% margin=0.0000 grad=5.763288
Step 1: loss=0.6909 acc=100.00% margin=0.0044 grad=4.763657
Step 2: loss=0.6933 acc=50.00% margin=-0.0003 grad=5.041406
Step 3: loss=0.6793 acc=100.00% margin=0.0280 grad=5.373095
Step 4: loss=0.6834 acc=100.00% margin=0.0195 grad=5.409656
--- Final Metrics ---
DPO loss:               0.6834
Policy chosen log-p:    -1.9310
Policy rejected log-p:  -1.9310
Implicit-reward acc:    100.00%
Implicit-reward margin: 0.0195
Gradient norm:          5.409656
```

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `NoiseSchedule` | artifex | Forward diffusion for log-prob estimation |
| `create_optimizer` | substrax | Gradient-clipped optimizer creation |
| `nnx.clone` | Flax NNX | Deep-copy for frozen reference model |

## Related

- [Preference Construction Quick Reference](preference-construction-quickref.md)
- [DPOAlignmentTrainer API](../../api/alignment/dpo_trainer.md)
- [Alignment User Guide](../../user-guide/alignment.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
