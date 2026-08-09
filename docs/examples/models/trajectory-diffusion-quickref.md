# Trajectory Diffusion Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min (CPU) |
| **Prerequisites** | JAX arrays, Flax NNX basics, diffusion models |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~3 min CPU) demonstrating how to build, train, and
sample from the trajectory diffusion model at WOD (Waymo Open Dataset) scale.
The model generates 8-second multi-agent future trajectories (80 steps at 10 Hz)
conditioned on scene context embeddings, using a denoising diffusion
probabilistic model (DDPM) with a FactorizedSceneBackbone.

## Files

- **Python Script**: [`examples/models/01_trajectory_diffusion_quickref.py`](https://github.com/avitai/DiffAV/blob/main/examples/models/01_trajectory_diffusion_quickref.py)
- **Jupyter Notebook**: [`examples/models/01_trajectory_diffusion_quickref.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/models/01_trajectory_diffusion_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/models/01_trajectory_diffusion_quickref.py`

## What You'll Learn

1. Configure and instantiate a trajectory diffusion model at WOD scale
2. Run forward diffusion to visualize the noising process
3. Compute training loss with gradient flow for backpropagation
4. Generate multi-agent trajectories via full reverse diffusion

## Prerequisites

- DiffAV installed (`uv sync`)
- JAX arrays, Flax NNX basics, diffusion models

## Pipeline Overview

```
SceneTokenizer.apply() -> scene_context (num_agents, 64)
    |
TrajectoryDiffusionModel
    |-- q_sample()         -> forward diffusion (add noise)
    |-- predict_noise()    -> FactorizedSceneBackbone
    |-- compute_loss()     -> training loss (MSE/L1)
    +-- sample()           -> reverse diffusion -> TrajectoryPrediction
```

## Quick Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx

from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
    create_trajectory_model,
)

# WOD-scale configuration
config = TrajectoryDiffusionConfig(
    hidden_dim=128,
    num_blocks=2,
    num_temporal_layers=2,
    num_social_layers=1,
    num_heads=4,
    future_steps=80,          # 8 seconds at 10 Hz
    num_agents_max=32,        # Top-32 agents per scene
    state_dim=4,              # [x, y, heading, velocity]
    context_dim=64,           # Scene embedding from tokenizer
    noise_schedule_type="cosine",
    num_timesteps=1000,
)

# Build model
rngs = nnx.Rngs(params=jax.random.key(42))
model = create_trajectory_model(config, rngs=rngs)

# Simulate per-agent scene context from tokenizer (one row per agent)
scene_context = jax.random.normal(jax.random.key(0), (32, 64))
trajectories = jax.random.normal(jax.random.key(1), (32, 80, 4)) * 0.1

# Training loss
loss = model.compute_loss(trajectories, scene_context, key=jax.random.key(2))

# Generate trajectories via reverse diffusion (agent count inferred from context)
prediction = model.sample(scene_context, key=jax.random.key(3))
assert prediction.trajectories.shape == (32, 80, 4)
```

## Configuration Variants

| Variant | Hidden | Blocks | Heads | Parameters |
|---------|--------|--------|-------|------------|
| Small | 64 | 1 | 2 | ~167K |
| Base | 128 | 2 | 4 | ~1.15M |
| Large | 256 | 4 | 8 | ~8.7M |

## Related

- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)
- [TrajectoryDiffusionModel API](../../api/models/trajectory_diffusion.md)
- [FactorizedSceneBackbone API](../../api/models/factorized_backbone.md)
- [TrajectoryPrediction](../../api/core/types.md)
