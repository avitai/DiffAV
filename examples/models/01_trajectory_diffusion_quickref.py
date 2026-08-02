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
# Trajectory Diffusion Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min (CPU) |
| **Prerequisites** | JAX arrays, Flax NNX basics, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates how to build, train, and sample from the
trajectory diffusion model at WOD (Waymo Open Dataset) scale. The model
generates 8-second multi-agent future trajectories (80 steps at 10 Hz)
conditioned on scene context embeddings, using a denoising diffusion
probabilistic model (DDPM) with a factorized scene backbone.

The pipeline follows the architecture:

```
SceneTokenizer.apply() → scene_context (num_agents, 64)
    ↓
TrajectoryDiffusionModel
    ├── q_sample()         → forward diffusion (add noise)
    ├── predict_noise()    → FactorizedSceneBackbone
    ├── compute_loss()     → training loss (MSE/L1)
    └── sample()           → reverse diffusion → TrajectoryPrediction
```

## Learning Goals

By the end of this example, you will be able to:

1. Configure and instantiate a trajectory diffusion model at WOD scale
2. Run forward diffusion to visualize the noising process
3. Compute training loss with gradient flow for backpropagation
4. Generate multi-agent trajectories via full reverse diffusion
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
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from dotenv import load_dotenv
from flax import nnx

from simulacrax.api import create_scenario_miner, MinerConfig
from simulacrax.core.constants import MINER_STATE_OFFSETS, MINER_STATE_SCALES
from simulacrax.core.types import TrajectoryPrediction
from simulacrax.data import prepare_full_horizon_scene, resolve_wod_tfrecord_path
from simulacrax.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from simulacrax.data.wod_source import WODSource, WODSourceConfig
from simulacrax.models.trajectory_diffusion import (
    create_trajectory_model,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)


load_dotenv()
load_dotenv(".env.data")


# %% [markdown]
"""
## Implementation

### Step 1: Configure the Model at WOD Scale

WOD Motion scenarios contain up to 128 agents tracked for 9.1 seconds
(11 history + 80 future steps at 10 Hz). The model predicts future
trajectories for up to 32 agents (the most relevant ones) with a
4-dimensional state: [x, y, heading, velocity].

The factorized scene backbone uses 4 temporal+social blocks with adaptive
layer-norm conditioning on per-agent scene context from the tokenizer.
"""

# %%
# WOD-scale configuration
config = TrajectoryDiffusionConfig(
    # Architecture
    hidden_dim=128,  # Transformer hidden dimension
    num_blocks=4,  # Factorized temporal+social blocks
    num_heads=4,  # Attention heads (128 / 4 = 32 per head)
    mlp_ratio=4.0,  # FFN hidden = 128 * 4 = 512
    # Trajectory dimensions (WOD scale)
    future_steps=80,  # 8 seconds at 10 Hz
    num_agents_max=32,  # Top-32 agents per scene
    state_dim=4,  # [x, y, heading, velocity]
    context_dim=64,  # Scene embedding from tokenizer
    # Diffusion schedule
    noise_schedule_type="cosine",  # Improved diffusion schedule
    num_timesteps=100,  # Reference trajectory-diffusion schedule length
    beta_start=1e-4,  # Noise schedule bounds
    beta_end=2e-2,
    loss_type="mse",  # Epsilon-matching MSE loss
    dropout_rate=0.0,  # No dropout for deterministic inference
    # Diffuse in normalized space (reference add/div-coefficient scheme):
    # raw WOD coordinates span hundreds of metres and would swamp unit
    # Gaussian noise; boundaries stay in metres, samples stay bounded.
    state_offsets=MINER_STATE_OFFSETS,
    state_scales=MINER_STATE_SCALES,
    x0_clip_bound=1.0,
)

print(f"Model type: {config.model_type}")
print(f"Architecture: {config.num_blocks}B / {config.num_heads}H / {config.hidden_dim}D")
print(f"Trajectory shape: ({config.num_agents_max}, {config.future_steps}, {config.state_dim})")
print(f"Sequence length: {config.num_agents_max * config.future_steps} tokens")
print(f"Diffusion: {config.num_timesteps} steps, {config.noise_schedule_type} schedule")

# %% [markdown]
"""
### Step 2: Build the Model

The factory function creates a `TrajectoryDiffusionModel` with:
- A `FactorizedSceneBackbone` (factorized temporal+social blocks)
- A `NoiseSchedule` from artifex (cosine beta schedule)

Parameter count scales with hidden_dim, num_blocks, and the number of
attention heads.
"""

# %%
# Build model
rngs = nnx.Rngs(params=jax.random.key(42))
model = create_trajectory_model(config, rngs=rngs)

# Count parameters
param_count = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Total parameters: {param_count:,}")
print(f"Parameter memory: ~{param_count * 4 / 1024 / 1024:.1f} MB (float32)")

# %% [markdown]
"""
### Step 3: Load Real WOD Trajectories

Real WOD ground-truth trajectories are used as training inputs to demonstrate
forward diffusion and loss computation on actual vehicle motion data.

Scene context comes from `SceneTokenizer.apply()` in production; here it is
built from the agents' real current states — one 64-dim row per agent — to
keep this example focused on the diffusion model API.
"""

# %%
# Load real WOD trajectories
HIST_STEPS = 11  # 10 past + 1 current
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=config.num_agents_max,
        history_steps=HIST_STEPS,
        future_steps=config.future_steps,
    )
)
print(f"Loaded {len(source)} validation scenarios")
# Expected output (single validation shard):
# Loaded 333 validation scenarios

# Re-centre on the SDC and keep only agents valid across the full horizon —
# WOD pads missing agents/steps with -1 sentinels that must never reach
# the model. `prepare_full_horizon_scene` is the same helper the training
# script uses.
NUM_AGENTS = 8
norm_op = AgentNormalizationOperator(
    AgentNormalizationConfig(current_step_idx=HIST_STEPS - 1), rngs=nnx.Rngs(0)
)
recentred, _, _ = norm_op.apply(dict(source[0].data), {}, {})
prepared = prepare_full_horizon_scene(
    recentred,
    num_agents=NUM_AGENTS,
    history_steps=HIST_STEPS,
    future_steps=config.future_steps,
)
assert prepared is not None, "scenario lacks full-horizon-valid agents"
trajectories, current_states = prepared
num_agents = NUM_AGENTS

# Scene context rows: the same [x, y, vx, vy] encoding the miner uses,
# zero-padded to context_dim — real conditioning, not random noise.
scene_context = jnp.pad(current_states, ((0, 0), (0, config.context_dim - current_states.shape[1])))

print(f"Scene context: {scene_context.shape}  (agents × embed_dim, real current states)")
print(f"Trajectories:  {trajectories.shape}  (agents × steps × state, real WOD data)")
# Expected output:
# Scene context: (8, 64)  (agents × embed_dim, real current states)
# Trajectories:  (8, 80, 4)  (agents × steps × state, real WOD data)

# %% [markdown]
r"""
### Step 4: Forward Diffusion — Visualize the Noising Process

The forward process $q(x_t | x_0)$ adds noise according to the cosine
schedule. At $t=0$ the trajectory is nearly clean; at $t=99$ it is
almost pure Gaussian noise. Diffusion runs in normalized space, so the
table below normalizes the clean trajectory the same way before
differencing.

$$x_t = \sqrt{\bar\alpha_t} \, x_0 + \sqrt{1 - \bar\alpha_t} \, \epsilon$$
"""

# %%
# Add noise at different timesteps to see the schedule's effect
normalized_clean = (trajectories - jnp.asarray(MINER_STATE_OFFSETS)) / jnp.asarray(
    MINER_STATE_SCALES
)
noise = jax.random.normal(jax.random.key(2), trajectories.shape)

timesteps_to_show = [0, 10, 50, 99]
print("Forward diffusion — mean absolute difference from clean trajectory:")
print(f"{'Timestep':>10} {'Mean |x_t - x_0|':>20} {'Signal preserved':>18}")
print("-" * 52)

for t in timesteps_to_show:
    noisy = model.q_sample(normalized_clean, t, noise)
    diff = jnp.mean(jnp.abs(noisy - normalized_clean))
    signal = jnp.mean(jnp.abs(noisy)) / (jnp.mean(jnp.abs(normalized_clean)) + 1e-8)
    print(f"{t:>10} {float(diff):>20.4f} {float(signal):>18.2f}x")

# %% [markdown]
r"""
### Step 5: Compute Training Loss

`compute_loss()` implements the DDPM training objective:

1. Sample a random timestep $t \sim \mathcal{U}(0, T)$
2. Sample noise $\epsilon \sim \mathcal{N}(0, I)$
3. Create noisy trajectory: $x_t = q(x_0, t, \epsilon)$
4. Predict noise: $\hat\epsilon = f_\theta(x_t, t, \text{scene\_ctx})$
5. Return $\| \hat\epsilon - \epsilon \|^2$

The loss is differentiable w.r.t. all backbone parameters for gradient
descent training.
"""

# %%
# Compute loss (single sample for demonstration)
key = jax.random.key(100)
loss = model.compute_loss(trajectories, scene_context, key=key)
print(f"Training loss (MSE): {float(loss):.6f}")


# Verify gradient flow
def loss_fn(m: TrajectoryDiffusionModel) -> jax.Array:
    return m.compute_loss(trajectories, scene_context, key=key)


start = time.perf_counter()
loss_val, grads = nnx.value_and_grad(loss_fn)(model)
grad_time = time.perf_counter() - start

grad_leaves = jax.tree_util.tree_leaves(grads)
grad_norms = [float(jnp.linalg.norm(g)) for g in grad_leaves if hasattr(g, "shape")]
nonzero_grads = sum(1 for n in grad_norms if n > 0)
total_grads = len(grad_norms)

print(f"Loss value: {float(loss_val):.6f}")
print(f"Gradient computation: {grad_time:.2f}s")
print(f"Parameter groups with gradients: {nonzero_grads}/{total_grads}")
print(f"Max gradient norm: {max(grad_norms):.6f}")

# %% [markdown]
r"""
### Step 6: Generate Trajectories via Reverse Diffusion

`sample()` runs the full DDPM reverse process: starting from pure
Gaussian noise $x_T \sim \mathcal{N}(0, I)$, iteratively denoise
through all 1000 timesteps to produce trajectory predictions.

The output is a `TrajectoryPrediction` containing:
- `trajectories`: shape `(num_agents, future_steps, state_dim)`
- `agent_ids`: tuple of string identifiers

When the WOD-trained checkpoint exists (produce one with
`python scripts/train_wod.py`), sampling uses it and the statistics below
are physically meaningful. Without it, a freshly initialised model is
used and the output is structured noise — labeled honestly either way.
"""

# %%
CHECKPOINT_DIR = Path("checkpoints/wod-mini")
USE_CHECKPOINT = CHECKPOINT_DIR.exists()
if USE_CHECKPOINT:
    miner = create_scenario_miner(
        MinerConfig(
            model_path=str(CHECKPOINT_DIR),
            max_agents=NUM_AGENTS,
            num_diffusion_steps=100,
            hidden_dim=256,
            num_blocks=6,
            num_heads=8,
        )
    )
    sample_model = miner.model
    sample_context = jnp.pad(
        current_states,
        ((0, 0), (0, miner.config.context_dim - current_states.shape[1])),
    )
else:
    sample_model = model  # untrained fallback — output is structured noise
    sample_context = scene_context

start = time.perf_counter()
prediction = sample_model.sample(sample_context, key=jax.random.key(99))
sample_time = time.perf_counter() - start

print(f"Generated trajectories: {prediction.trajectories.shape}")
print(f"Agent IDs: {prediction.agent_ids[:3]}... ({len(prediction.agent_ids)} total)")
print(f"Sampling time (100 steps): {sample_time:.2f}s")
print(f"Output type: {type(prediction).__name__}")
assert isinstance(prediction, TrajectoryPrediction)

# Trajectory statistics + displacement error against the real futures
traj = prediction.trajectories
label = "trained checkpoint" if USE_CHECKPOINT else "untrained model"
print()
print(f"Per-dimension statistics ({label}):")
dim_names = ["x", "y", "heading", "velocity"]
for d, name in enumerate(dim_names):
    vals = traj[:, :, d]
    print(
        f"  {name:>8}: mean={float(jnp.mean(vals)):+.3f}, "
        f"std={float(jnp.std(vals)):.3f}, "
        f"range=[{float(jnp.min(vals)):.2f}, {float(jnp.max(vals)):.2f}]"
    )
displacement = jnp.hypot(traj[..., 0] - trajectories[..., 0], traj[..., 1] - trajectories[..., 1])
print(f"ADE vs ground truth: {float(jnp.mean(displacement)):.2f} m ({label})")

# %% [markdown]
"""
> **Reading the ADE.** When `checkpoints/wod-mini` is present, this displacement
> is measured against the same scenes the checkpoint was trained on, so a low
> value (~1-2 m) reflects in-training-set memorization, not generalization. The
> honest held-out figure is the `wod-physics-tier0` baseline: minADE_6 ≈ 5.6 m on
> unseen WOD scenes. Treat this quickref number as a wiring check, not a benchmark.
"""

# %% [markdown]
"""
## Results & Evaluation

### What We Demonstrated

| Aspect | Result |
|--------|--------|
| Model scale | 128-dim, 4-block factorized backbone at WOD dimensions |
| Trajectory shape | (8 agents, 80 steps, 4 state) — full-horizon-valid agents |
| Sampling model | `checkpoints/wod-mini` (256-dim, 6-block) when present |
| Forward diffusion | Cosine schedule, 100 timesteps |
| Training loss | MSE epsilon-matching with full gradient flow |
| Sampling | Reverse diffusion producing TrajectoryPrediction |

### Key Takeaways

- The model cross-attends trajectory tokens to per-agent context rows
  encoding real current states ([x, y, vx, vy])
- Cosine noise schedule provides smooth degradation from clean to noisy
- All backbone parameters receive gradients through the diffusion loss
- Sampling runs full reverse diffusion to produce structured output
"""

# %%
print("Trajectory diffusion quick reference complete!")

# %% [markdown]
"""
## Next Steps & Resources

### Training at Scale

To train the model on real WOD data:

1. Load WOD scenarios via `WODSource` and tokenize with `SceneTokenizer`
2. Use the tokenizer's `fused_context` as `scene_context` input
3. Extract ground-truth future trajectories from parsed scenarios
4. Optimize with `optax` using `compute_loss()` and `nnx.value_and_grad()`

Reference trajectory-diffusion practice this quickref omits for brevity:
weight EMA (decay 0.995) evaluated instead of the raw weights,
classifier-free guidance via conditioning dropout, and x0-prediction as an
alternative to the epsilon-matching objective used here.

### Configuration Variants

| Variant | Hidden | Blocks | Heads | Parameters |
|---------|--------|--------|-------|------------|
| Small | 64 | 2 | 2 | ~50K |
| Base | 128 | 4 | 4 | ~400K |
| Large | 256 | 6 | 8 | ~3M (matches `checkpoints/wod-mini`) |

### Related Examples

- [WOD Loading Quick Reference](../core/01_wod_loading_quickref.py)

### API Reference

- [TrajectoryDiffusionModel](../../docs/api/models/trajectory_diffusion.md)
- [FactorizedSceneBackbone](../../docs/api/models/factorized_backbone.md)
- [TrajectoryPrediction](../../docs/api/core/types.md)
"""


# %%
def main() -> None:
    """Main entry point for command-line execution."""
    print("Running trajectory diffusion quick reference...")

    cfg = TrajectoryDiffusionConfig(
        hidden_dim=128,
        num_blocks=4,
        num_heads=4,
        future_steps=80,
        num_agents_max=32,
        context_dim=64,
        num_timesteps=100,
    )
    mdl = create_trajectory_model(cfg, rngs=nnx.Rngs(params=jax.random.key(42)))

    ctx = jax.random.normal(jax.random.key(0), (32, 64))
    traj = jax.random.normal(jax.random.key(1), (32, 80, 4)) * 0.1

    loss = mdl.compute_loss(traj, ctx, key=jax.random.key(2))
    print(f"Training loss: {float(loss):.6f}")

    pred = mdl.sample(ctx, key=jax.random.key(3))
    print(f"Generated: {pred.trajectories.shape}")
    print("Done!")


if __name__ == "__main__":
    main()
