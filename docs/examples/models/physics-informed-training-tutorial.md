# Physics-Informed Training Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5-8 min (GPU) |
| **Prerequisites** | Trajectory diffusion model basics, bicycle model, JAX/Flax NNX |
| **Format** | Python + Jupyter |

A Tier 2 tutorial demonstrating the physics-informed training loop that
combines diffusion loss with kinematic and collision penalties. The trainer
uses adaptive weight scheduling to gradually increase the physics loss
contribution during training, Min-SNR-γ weighting on the diffusion term, and
an ᾱ_t anneal on the physics term. The model is conditioned on per-agent
scene context rows standardized to unit scale.
Training milestones sample a held-out scene so the loss decrease is tied to
visible sample improvement.

## Files

- **Python Script**: [`examples/models/02_physics_informed_training_tutorial.py`](https://github.com/avitai/DiffAV/blob/main/examples/models/02_physics_informed_training_tutorial.py)
- **Jupyter Notebook**: [`examples/models/02_physics_informed_training_tutorial.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/models/02_physics_informed_training_tutorial.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/models/02_physics_informed_training_tutorial.py`

## What You'll Learn

1. Configure a `TrajectoryTrainer` with physics-informed training
2. Set up adaptive weight scheduling for physics loss
3. Run training steps and observe loss and physics weight evolution
4. Connect the loss decrease to sample quality at training milestones
5. Use JIT-compiled training for maximum throughput
6. Run epoch-based training with data iterators

## Prerequisites

- DiffAV installed (`uv sync`)
- Trajectory diffusion model basics, bicycle model, JAX/Flax NNX

## Pipeline Overview

```
TrajectoryDiffusionModel + DiffAVPhysicsLoss
    |                         |
    +-- compute_loss()        +-- kinematic + collision penalties
    |                         |
    v                         v
TrajectoryTrainer
    |-- train_step()        -> TrainingStepMetrics
    |-- train_epoch()       -> [TrainingStepMetrics, ...]
    +-- compute_train_step() -> (loss, aux)  [JIT-compatible]
```

## Quick Usage

```python
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig, TrajectoryDiffusionModel,
)
from diffav.physics.losses import DiffAVPhysicsConfig

trainer_config = TrainerConfig(
    physics_config=DiffAVPhysicsConfig(
        kinematic_weight=1.0, collision_weight=1.0,
        adaptive_weighting=True, initial_physics_weight=0.01,
    ),
    num_epochs=100,
)
trainer = TrajectoryTrainer(model, trainer_config)
metrics = trainer.train_step(trajectories, scene_context, key=key)
```

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `OptimizerConfig` | substrax | Optimizer + gradient clipping config; a schedule is an optax schedule |
| `create_optimizer` | substrax | Build the NNX optimizer from config |
| `ErrorRecoveryManager` | opifex | NaN/instability detection |
| `TimingCollector` | calibrax | Wall-clock timing |
| `DiffAVPhysicsLoss` | diffav | Bicycle model + collision penalties |

## Coming from standard PyTorch training?

If you're familiar with standard PyTorch training loops, here's how DiffAV physics-informed training compares:

| PyTorch | DiffAV |
|---------|------------|
| Manual `loss.backward()` + `optimizer.step()` | `TrajectoryTrainer.train_step(traj, ctx, key)` |
| Physics penalty as separate loss term | `DiffAVPhysicsLoss` integrated via `TrainerConfig` |
| Manual gradient clipping | Pre-configured via `OptimizerConfig(gradient_clip_norm=...)` |
| Custom NaN handling | `nan_safe_gradients()` utility in training loop |
| `torch.compile(model)` | `nnx.jit(trainer.compute_train_step)` |

**Key differences:**

1. **Adaptive weighting**: Physics loss weight ramps from `initial_physics_weight` to 1.0 over training — no manual tuning needed
2. **NaN-safe by default**: Gradients are zeroed if NaN detected, preventing silent divergence
3. **Functional state**: Optimizer state flows explicitly; no hidden `optimizer.zero_grad()` calls needed

## Related

- [Trajectory Diffusion Quick Reference](trajectory-diffusion-quickref.md)
- [Bicycle Model Quick Reference](../physics/bicycle-model-quickref.md)
- [TrajectoryTrainer API](../../api/models/trainer.md)
- [DiffAVPhysicsLoss API](../../api/physics/losses.md)

## Terminal Output

The output below is from a run on a Modal L4 GPU (`modal run deploy/modal_app.py --task examples`), with deterministic GPU kernels and WOD read from the data volume.

```text
Loaded 4815 training scenarios
Batch trajectories: (8, 8, 16, 4)  [scenes, agents, steps, state_dim]
Held-out eval scene: index 8, (8, 16, 4)
Model: 64d hidden, 2 blocks
Diffusion timesteps: 20
Physics enabled: True
Recovery manager: ErrorRecoveryManager
Milestone step   0 | sample ADE=  151.13 m | kinematic residual=  40314.73 m^2
Step   0 | total=95.5913 | diff=1.6010 | phys=93.9903 | weight=0.0100 | grad=148.6740
Step 100 | total=3.6913 | diff=2.3133 | phys=1.3780 | weight=0.0100 | grad=9.4860
Step 200 | total=0.6709 | diff=0.3374 | phys=0.3335 | weight=0.0100 | grad=25.7440
Milestone step 250 | sample ADE=   26.47 m | kinematic residual=     95.60 m^2
Step 300 | total=0.2683 | diff=0.1330 | phys=0.1352 | weight=0.0100 | grad=6.2605
Step 400 | total=0.1413 | diff=0.0796 | phys=0.0618 | weight=0.0100 | grad=10.8598
Milestone step 500 | sample ADE=   22.41 m | kinematic residual=     20.01 m^2
Saved $AVITAI_OUTPUT_DIR/examples/physics_training_sample_evolution.png
Saved $AVITAI_OUTPUT_DIR/examples/physics_training_curves.png
JIT step 0 | loss=86.5709 | grad=200.5102 | nan=False
JIT step 1 | loss=87.4272 | grad=150.7269 | nan=False
JIT step 2 | loss=73.1568 | grad=130.7695 | nan=False
JIT step 3 | loss=81.4873 | grad=143.6379 | nan=False
JIT step 4 | loss=83.4870 | grad=145.3188 | nan=False
Epoch metrics: 3 steps
  Step 500 | loss=0.2063 | physics_weight=0.0251
  Step 501 | loss=0.3120 | physics_weight=0.0251
  Step 502 | loss=0.4381 | physics_weight=0.0251
  Epoch    0: physics_weight = 0.0100
  Epoch   10: physics_weight = 0.0158
  Epoch   25: physics_weight = 0.0316
  Epoch   50: physics_weight = 0.1000
  Epoch  100: physics_weight = 1.0000
  Epoch  200: physics_weight = 1.0000
```

## Example Output

![Ground truth versus model samples on a held-out scene at three training milestones.](../../assets/images/examples/physics_training_sample_evolution.png)

*Held-out scene, fixed sampling key: at step 0 the samples are data-scale
noise crossing the whole frame; by step 250 they contract to the data's
spatial scale, and by step 500 they gather in a tight cluster near the scene
centre, with ADE falling from 151 m to 22 m and the kinematic residual by three
orders of magnitude. They do not yet follow the individual agents'
ground-truth tracks. Every change between panels comes from the trained
weights.*

![Training losses, the annealed physics loss, and sample-quality metrics at milestones.](../../assets/images/examples/physics_training_curves.png)

*Left: Min-SNR-weighted diffusion and total losses. Middle: ᾱ_t-annealed
physics loss on the x̂₀ reconstruction. Right: what the loss decrease buys —
displacement error and kinematic residual of samples at the milestones.*
