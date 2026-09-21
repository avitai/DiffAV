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
# Physics-Informed Training Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5-8 min (GPU) |
| **Prerequisites** | Trajectory diffusion, bicycle model, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This tutorial demonstrates the physics-informed training loop that trains a
`TrajectoryDiffusionModel` on real Waymo Open Dataset trajectories, combining
diffusion loss with kinematic and collision penalties from the bicycle model.

Four reference-grounded training techniques keep the objective well-behaved:

- **Min-SNR-γ loss weighting** (`snr_gamma=5.0`): the ε-prediction loss at each
  drawn timestep is scaled by `min(SNR_t, γ)/SNR_t`, so easy low-noise timesteps
  stop dominating the objective (Hang et al. 2023; the value 5.0 is the
  reference trainers' recommendation).
- **ᾱ_t-annealed physics** (`physics_x0_annealing=True`): the physics penalty is
  scored on the model's clean-trajectory reconstruction x̂₀, which is divided by
  `sqrt(ᾱ_t)` — at high noise levels it is mostly amplified model error. Scaling
  the penalty by ᾱ_t trusts x̂₀ exactly where it is trustworthy, mirroring the
  reconstruction-guidance anneal the sampling path applies to x̂₀ guidance.
- **Learning-rate warmup, then a flat schedule**: linear warmup over the first
  10% of the schedule horizon, after which the rate stays near peak
  (score-based diffusion references warm up and then hold the learning rate
  constant; the cosine horizon here is 3x the run length so the run sits in
  the flat region).
- **Standardized scene conditioning**: the per-agent `[x, y, vx, vy]` context
  rows are standardized with the same dataset coefficients as the diffusion
  space before they meet the model, following the reference practice of
  feeding conditioning features to learned layers at unit scale (CTG
  standardizes all history conditioning with dataset add/div coefficients;
  cross-attention conditioning in image diffusers arrives LayerNorm-ed).

```
WODSource → real vehicle trajectories (x, y, heading, speed)
                        ↓
TrajectoryDiffusionModel.compute_loss()  →  diffusion_loss (Min-SNR-weighted)
DiffAVPhysicsLoss.compute()          →  ᾱ_t × (kinematic + collision)
                        ↓
total_loss = diffusion_loss + physics_weight × physics_loss
                        ↓
nnx.value_and_grad → gradients → optax.apply_updates
                        ↓
TrainerConfig.adaptive_weighting: physics_weight ↑ over epochs
```

Loss curves alone do not show what training buys, so the loop pauses at three
milestones (untrained, halfway, final), samples trajectories for a held-out
scene with a fixed key, and reports displacement error and kinematic residual
of the *samples* — the outcome the loss is a proxy for.

## What You'll Learn

1. Load real WOD trajectories for training batches
2. Configure `TrainerConfig` with physics-informed loss weights
3. Run `TrajectoryTrainer.train_step()` and monitor loss components
4. Connect the loss decrease to sample quality at training milestones
5. Apply `nnx.jit` to the training step for maximum throughput
6. Observe adaptive physics weight scheduling over epochs

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `OptimizerConfig` / `create_optimizer` | substrax | Optimizer over an optax warmup-cosine |
| `GaussianNormalizer` | opifex | Add/div standardization of context rows |
| `ErrorRecoveryManager` | opifex | Stability checks + stable-state rollback |
| `TimingCollector` | calibrax | Wall-clock timing |
| `DiffAVPhysicsLoss` | diffav | Bicycle model + collision penalties |
"""

# %% [markdown]
r"""
## Setup

### Required Data

Download at least one WOD Motion training shard:

```bash
WOD_BUCKET="gs://waymo_open_dataset_motion_v_1_2_1"
WOD_SHARD="uncompressed/tf_example/training"
SHARD_FILE="training_tfexample.tfrecord-00000-of-01000"
gsutil -m cp "$WOD_BUCKET/$WOD_SHARD/$SHARD_FILE" \
    /path/to/waymo/motion_v1.2.1/tf_example/training/
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

import os

import jax
import jax.numpy as jnp
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from substrax.artifacts import resolve_output_dir


PLOT_DIR = resolve_output_dir("examples").path

import optax
from dotenv import load_dotenv
from flax import nnx
from opifex.core.normalization import GaussianNormalizer
from substrax.optim import OptimizerConfig

from diffav.core.constants import MINER_STATE_OFFSETS, MINER_STATE_SCALES
from diffav.data import prepare_full_horizon_scene, resolve_wod_tfrecord_path
from diffav.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)
from diffav.physics.kinematics import BicycleModelConstraint
from diffav.physics.losses import DiffAVPhysicsConfig, DiffAVPhysicsLoss


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 16  # Truncated to 16 for fast CPU demo (production: 80)
NUM_AGENTS = 8  # Fully-valid agents selected per scene (production: 32)
MAX_RAW_AGENTS = 64  # Raw agent pool per scenario to select from
CONTEXT_DIM = 32  # Scene context embedding dimension
BATCH_SCENES = 8  # Scenes stacked per training step (reference practice)

# Smoke mode (set by the example execution tests) shrinks the loop and the
# milestone sampling.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
NUM_TRAIN_STEPS = 60 if _SMOKE else 500

# %% [markdown]
"""
## 1. Load Real WOD Training Trajectories

`WODSource` loads real vehicle trajectories from WOD Motion TFRecords.
`prepare_full_horizon_scene` — the same helper the training script uses —
selects agents that are valid across the whole horizon and extracts the 4D
state vector `[x, y, heading, speed]` used by the bicycle model constraint.
Validity handling matters: WOD fills untracked agent-steps with `-1`, which
an SDC-recentred frame turns into kilometre-scale garbage coordinates.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=MAX_RAW_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
print(f"Loaded {len(source)} training scenarios")
# Expected output (training split):
# Loaded 486995 training scenarios

# Re-centre every scenario on the SDC at the current step with the
# pipeline's own normalization operator — diffusion training on raw global
# WOD coordinates (thousands of metres) is numerically hostile.
norm_op = AgentNormalizationOperator(
    AgentNormalizationConfig(current_step_idx=HIST_STEPS - 1), rngs=nnx.Rngs(0)
)

# Conditioning features must reach the model at roughly unit scale, exactly
# like the trajectories the model diffuses. Reference conditional diffusers
# standardize every conditioning feature with dataset add/div coefficients
# before it meets a learned layer (CTG's history encoders), and the
# production `SceneTokenizer` LayerNorms its embeddings for the same reason.
# Recentred metres fed raw (|x| up to ~150) enter the cross-attention
# keys/values orders of magnitude hotter than the token stream, which
# destabilizes training. The context rows are [x, y, vx, vy], so x/y reuse
# the position scales of the diffusion space and vx/vy the speed scale.
CONTEXT_NORMALIZER = GaussianNormalizer(
    mean=jnp.array([-MINER_STATE_OFFSETS[0], -MINER_STATE_OFFSETS[1], 0.0, 0.0]),
    std=jnp.array(
        [
            MINER_STATE_SCALES[0],
            MINER_STATE_SCALES[1],
            MINER_STATE_SCALES[3],
            MINER_STATE_SCALES[3],
        ]
    ),
)


def _load_normalized_scene(index: int) -> tuple | None:
    """Load, SDC-recentre, and select fully-valid agents for one scenario.

    Returns:
        ``(trajectories, scene_context)`` with shapes
        ``(NUM_AGENTS, FUTURE_STEPS, 4)`` / ``(NUM_AGENTS, CONTEXT_DIM)``,
        or ``None`` when the scenario lacks NUM_AGENTS full-horizon-valid
        agents. Scene context rows carry each agent's current
        ``[x, y, vx, vy]``, standardized by ``CONTEXT_NORMALIZER`` and
        zero-padded to CONTEXT_DIM. Without real conditioning the model can
        only learn the unconditional marginal (a blob at the data mean);
        with it, samples anchor to the actual agent positions.
    """
    elem = source[index]
    normed_scene, _, _ = norm_op.apply(dict(elem.data), {}, {})
    prepared = prepare_full_horizon_scene(
        normed_scene,
        num_agents=NUM_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
    if prepared is None:
        return None
    scene_trajectories, current = prepared
    current = CONTEXT_NORMALIZER.normalize(current)
    context = jnp.pad(current, ((0, 0), (0, CONTEXT_DIM - current.shape[1])))
    return scene_trajectories, context


# Stack BATCH_SCENES distinct real scenes into one (B, A, T, 4) batch, plus
# one extra scene held out for milestone evaluation. The reference
# implementations (e.g. CTG) train on multi-scene batches, and the trainer
# averages the per-scene diffusion loss over the batch. Crucially, each
# scene draws its diffusion timestep from its own schedule stratum, so
# every step spans early, middle, and late noise levels instead of cycling
# through a single random timestep — the batch-mean loss is then smooth
# across steps rather than jumping with each draw.
scenes: list[tuple] = []
scene_index = 0
while len(scenes) < BATCH_SCENES + 1:
    scene = _load_normalized_scene(scene_index)
    if scene is not None:
        scenes.append(scene)
    scene_index += 1

batch_trajectories = jnp.stack([scene[0] for scene in scenes[:BATCH_SCENES]])
batch_scene_context = jnp.stack([scene[1] for scene in scenes[:BATCH_SCENES]])
print(f"Batch trajectories: {batch_trajectories.shape}  [scenes, agents, steps, state_dim]")
# Expected output:
# Batch trajectories: (8, 8, 16, 4)  [scenes, agents, steps, state_dim]

# Held-out scene for milestone evaluation: NOT one of the training scenes,
# so the sample metrics measure learned realism rather than memorization.
eval_trajectories, eval_context = scenes[BATCH_SCENES]
print(f"Held-out eval scene: index {scene_index - 1}, {eval_trajectories.shape}")
# Expected output:
# Held-out eval scene: index 8, (8, 16, 4)

# %% [markdown]
"""
## 2. Create Model and Trainer

Set up a compact model for fast demonstration.
In production use `hidden_dim=256`, `num_blocks=8`, `future_steps=80`.

Two loss-shaping options are enabled here (both default to off in the
library, keeping the plain objective bit-identical):

- `snr_gamma=5.0` — Min-SNR-γ weighting of the ε-prediction loss.
- `physics_x0_annealing=True` — the physics penalty on x̂₀ is scaled by ᾱ_t.

The optimizer follows score-based diffusion reference practice: Adam at
2e-4 with linear warmup from zero, then a near-constant rate (the optax
schedule is the learning rate itself, peaking at 2e-4; the cosine horizon
is 3x the run so the rate barely decays in-run).
"""

# %%
model_config = TrajectoryDiffusionConfig(
    hidden_dim=64,
    num_blocks=2,
    num_temporal_layers=1,
    num_social_layers=1,
    num_heads=2,
    future_steps=FUTURE_STEPS,
    num_agents_max=NUM_AGENTS,
    num_timesteps=20,
    context_dim=CONTEXT_DIM,
    # Diffuse in normalized space (reference add/div-coefficient scheme):
    # every boundary — including the physics loss on predictions — stays in
    # metres, and the unit clamp keeps early-training predictions bounded.
    state_offsets=MINER_STATE_OFFSETS,
    state_scales=MINER_STATE_SCALES,
    x0_clip_bound=1.0,
    # Min-SNR-γ (Hang et al. 2023): clamp the per-timestep loss weight at
    # min(SNR, 5)/SNR so easy high-SNR timesteps stop dominating.
    snr_gamma=5.0,
)
model = TrajectoryDiffusionModel(model_config, rngs=nnx.Rngs(params=jax.random.key(0)))
print(f"Model: {model_config.hidden_dim}d hidden, {model_config.num_blocks} blocks")
print(f"Diffusion timesteps: {model_config.num_timesteps}")
# Expected output:
# Model: 64d hidden, 2 blocks
# Diffusion timesteps: 20

trainer_config = TrainerConfig(
    optimizer_config=OptimizerConfig(
        optimizer_type="adam",
        # The rate is a schedule over a horizon 3x the run: linear warmup
        # from zero for 10% of the horizon to the 2e-4 peak, then a cosine
        # so shallow the run sits in its flat region. Score-based references
        # warm up and then hold the rate constant (diffusionjax's optimizer
        # builds the warmup schedule with end value == peak); decaying to
        # zero within the run measurably halts conditioning learning — the
        # cross-attention needs the late steps to anchor samples to the
        # scene context.
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=2e-4,
            warmup_steps=max(3 * NUM_TRAIN_STEPS // 10, 1),
            decay_steps=3 * NUM_TRAIN_STEPS,
        ),
        gradient_clip_norm=1.0,
    ),
    physics_config=DiffAVPhysicsConfig(
        kinematic_weight=1.0,
        collision_weight=1.0,
        adaptive_weighting=True,
        schedule_type="exponential",
        initial_physics_weight=0.01,
        final_physics_weight=1.0,
        transition_epochs=50,
    ),
    # Trust the physics score of x̂₀ in proportion to ᾱ_t (the sampling
    # path anneals x̂₀ guidance with the same factor).
    physics_x0_annealing=True,
    num_epochs=5,
    log_interval=10,
    nan_detection=True,
)
trainer = TrajectoryTrainer(model, trainer_config)
print(f"Physics enabled: {trainer.physics_enabled}")
print(f"Recovery manager: {type(trainer.recovery_manager).__name__}")
# Expected output:
# Physics enabled: True
# Recovery manager: ErrorRecoveryManager

# %% [markdown]
"""
## 3. Run Training Steps on Real Data — with Outcome Milestones

Execute the training loop on a batch of real WOD scenarios. Observe:
- `total_loss` = diffusion_loss + physics_weight × physics_loss
- `physics_weight` starts near 0 (epoch 0) and grows over training
- The batch spreads its diffusion timesteps across the schedule (one
  stratum per scene), so the loss is smooth across steps rather than
  jumping with each single random timestep draw

At three milestones (before training, halfway, final) the loop samples the
held-out scene via full reverse diffusion with a fixed key and scores the
*samples*:

- **ADE** — mean displacement between sampled and ground-truth positions.
  The model is conditioned on the held-out scene's standardized agent-state
  rows, so this measures scene-conditioned prediction: how well samples
  anchor to where the eval scene's agents actually are.
- **Kinematic residual** — the bicycle-model residual of the samples
  (mean squared infeasible displacement, m²): the physics-violation
  measure the physics loss exists to reduce.
"""

# %%
key = jax.random.key(42)
metrics_history = []

SAMPLE_KEY = jax.random.key(1234)  # fixed key: milestones differ only by weights
bicycle = BicycleModelConstraint()

# ~3 milestones x 1 scene x 1 fixed key keeps sampling cost small; smoke
# mode shrinks to the first and final milestones only.
MILESTONE_STEPS = {NUM_TRAIN_STEPS} if _SMOKE else {NUM_TRAIN_STEPS // 2, NUM_TRAIN_STEPS}


def evaluate_milestone(step: int) -> dict:
    """Sample the held-out scene and score realism + physics of the samples."""
    prediction = model.sample(eval_context, key=SAMPLE_KEY)
    samples = prediction.trajectories
    displacement = jnp.linalg.norm(samples[..., :2] - eval_trajectories[..., :2], axis=-1)
    milestone = {
        "step": step,
        "samples": np.asarray(samples),
        "ade": float(jnp.mean(displacement)),
        "kinematic_residual": float(bicycle.compute_residuals(samples)),
    }
    print(
        f"Milestone step {step:3d} | sample ADE={milestone['ade']:8.2f} m | "
        f"kinematic residual={milestone['kinematic_residual']:10.2f} m^2"
    )
    return milestone


milestones = [evaluate_milestone(0)]
for i in range(NUM_TRAIN_STEPS):
    step_key = jax.random.fold_in(key, i)
    metrics = trainer.train_step(batch_trajectories, batch_scene_context, key=step_key)
    metrics_history.append(metrics)
    if i % 100 == 0:
        print(
            f"Step {metrics.step:3d} | "
            f"total={metrics.total_loss:.4f} | "
            f"diff={metrics.diffusion_loss:.4f} | "
            f"phys={metrics.physics_loss:.4f} | "
            f"weight={metrics.physics_weight:.4f} | "
            f"grad={metrics.grad_norm:.4f}"
        )
    if (i + 1) in MILESTONE_STEPS:
        milestones.append(evaluate_milestone(i + 1))
# Expected output (approximate — loss depends on random diffusion timestep):
# Milestone step   0 | sample ADE=  153.57 m | kinematic residual=  23832.19 m^2
# Step   0 | total=78.8704 | diff=1.4514 | phys=77.4190 | weight=0.0100 | grad=169.1920
# Step 100 | total=0.6021 | diff=0.4353 | phys=0.1668 | weight=0.0100 | grad=4.3065
# Step 200 | total=0.3000 | diff=0.2337 | phys=0.0662 | weight=0.0100 | grad=4.0677
# Milestone step 250 | sample ADE=   37.01 m | kinematic residual=     80.96 m^2
# Step 300 | total=0.2352 | diff=0.1735 | phys=0.0617 | weight=0.0100 | grad=3.8218
# Step 400 | total=0.2813 | diff=0.1365 | phys=0.1448 | weight=0.0100 | grad=13.1791
# Milestone step 500 | sample ADE=   32.63 m | kinematic residual=     25.40 m^2

# %% [markdown]
"""
### What the Loss Decrease Buys

The overlay below is the direct visual of training progress: at step 0 the
"trajectories" are unit Gaussian noise stretched to data scale — scribbles
crossing the whole frame, with a kinematic residual in the tens of
thousands. By the midpoint the samples have contracted to the data's
spatial scale and the residual has fallen by two orders of magnitude; the
final milestone tightens both further, with samples anchored near the
conditioning agents' locations. A 64-dim model trained for 500 steps does
not yet produce smooth, lane-following trajectories — closing that gap is
what the production configuration (256d/8L, 80-step horizon, full scene
embeddings, longer training) buys. The fixed sampling key means every
change between panels comes from the trained weights, not sampling luck.
"""

# %%
gt_xy = np.asarray(eval_trajectories[..., :2])
# Frame the union of ground truth and the FINAL milestone's samples: the
# trained samples always stay in view, while step-0 noise overflows the
# frame — which is exactly the point.
final_xy = milestones[-1]["samples"][..., :2]
frame_xy = np.concatenate([gt_xy.reshape(-1, 2), final_xy.reshape(-1, 2)])
pad = 0.15 * max(np.ptp(frame_xy[:, 0]), np.ptp(frame_xy[:, 1]), 20.0)
x_lim = (frame_xy[:, 0].min() - pad, frame_xy[:, 0].max() + pad)
y_lim = (frame_xy[:, 1].min() - pad, frame_xy[:, 1].max() + pad)

fig, axes = plt.subplots(
    1, len(milestones), figsize=(4.2 * len(milestones), 4.2), sharex=True, sharey=True
)
for ax, milestone in zip(np.atleast_1d(axes), milestones, strict=True):
    for agent in range(NUM_AGENTS):
        ax.plot(gt_xy[agent, :, 0], gt_xy[agent, :, 1], color="black", lw=1.8, alpha=0.8)
        ax.plot(
            milestone["samples"][agent, :, 0],
            milestone["samples"][agent, :, 1],
            color="tab:orange",
            lw=1.2,
            alpha=0.85,
        )
    ax.plot([], [], color="black", lw=1.8, label="ground truth")
    ax.plot([], [], color="tab:orange", lw=1.2, label="model samples")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal")
    ax.set_title(
        f"step {milestone['step']}\n"
        f"ADE {milestone['ade']:.1f} m | residual {milestone['kinematic_residual']:.2g} m²"
    )
    ax.set_xlabel("x (m)")
np.atleast_1d(axes)[0].set_ylabel("y (m)")
np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Held-out scene: samples at training milestones (fixed key)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "physics_training_sample_evolution.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'physics_training_sample_evolution.png'}")

# %% [markdown]
"""
### Loss Curves and Outcome Metrics

The left panels show what the optimizer sees; the right panel shows what it
buys. Read them together: as the Min-SNR-weighted diffusion loss and the
ᾱ_t-annealed physics loss fall, the sample ADE and the kinematic residual of
the samples fall with them — the loss decrease is real model improvement,
not reweighting cosmetics.
"""

# %%
steps = [m.step for m in metrics_history]


def _rolling_mean(values: list[float], window: int = 25) -> np.ndarray:
    """Windowed mean that keeps array length, unbiased at the edges.

    Zero-padded convolution alone would drag the boundary values toward
    zero (fewer real samples divided by the full window); normalizing by
    the per-position sample coverage removes that artifact.
    """
    kernel = np.ones(window)
    array = np.asarray(values)
    summed = np.convolve(array, kernel, mode="same")
    coverage = np.convolve(np.ones_like(array), kernel, mode="same")
    return summed / coverage


total = [m.total_loss for m in metrics_history]
diffusion = [m.diffusion_loss for m in metrics_history]
physics_vals = [m.physics_loss for m in metrics_history]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].semilogy(steps, total, alpha=0.25, color="tab:blue")
axes[0].semilogy(steps, diffusion, alpha=0.25, color="tab:orange")
axes[0].semilogy(steps, _rolling_mean(total), color="tab:blue", label="total (mean-25)")
axes[0].semilogy(steps, _rolling_mean(diffusion), color="tab:orange", label="diffusion (mean-25)")
axes[0].set_xlabel("step")
axes[0].set_title("Training losses (log scale)")
axes[0].legend()
axes[1].semilogy(steps, physics_vals, alpha=0.25, color="tab:red")
axes[1].semilogy(steps, _rolling_mean(physics_vals), color="tab:red")
axes[1].set_xlabel("step")
axes[1].set_title("Physics loss on x̂₀ (ᾱ_t-annealed)")

milestone_steps = [m["step"] for m in milestones]
ade_values = [m["ade"] for m in milestones]
residual_values = [m["kinematic_residual"] for m in milestones]
ax_ade = axes[2]
ax_ade.semilogy(milestone_steps, ade_values, "o-", color="tab:green", label="sample ADE (m)")
ax_res = ax_ade.twinx()
ax_res.semilogy(
    milestone_steps,
    residual_values,
    "s--",
    color="tab:purple",
    label="kinematic residual (m²)",
)
ax_ade.set_xlabel("step")
ax_ade.set_ylabel("ADE (m)", color="tab:green")
ax_res.set_ylabel("residual (m²)", color="tab:purple")
ax_ade.set_title("Sample quality at milestones")
handles, labels = ax_ade.get_legend_handles_labels()
handles_res, labels_res = ax_res.get_legend_handles_labels()
ax_ade.legend(handles + handles_res, labels + labels_res, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "physics_training_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'physics_training_curves.png'}")

# %% [markdown]
"""
## 4. JIT-Compiled Training Step

`nnx.jit` wraps `compute_train_step` for XLA compilation. The first call
pays compilation cost; subsequent calls run at hardware-native speed.
JIT is not applied internally so you retain full control of execution mode.
The compiled step takes the same stacked scene batch as the main loop.
"""

# %%
jit_model = TrajectoryDiffusionModel(model_config, rngs=nnx.Rngs(params=jax.random.key(7)))
jit_trainer = TrajectoryTrainer(jit_model, trainer_config)
jit_step = nnx.jit(jit_trainer.compute_train_step)

for i in range(5):
    step_key = jax.random.fold_in(key, 1000 + i)
    total_loss, aux = jit_step(
        jit_trainer.model,
        jit_trainer.optimizer,
        batch_trajectories,
        batch_scene_context,
        step_key,
        0,  # epoch (traced): the physics weight advances without retracing
    )
    print(
        f"JIT step {i} | "
        f"loss={float(total_loss):.4f} | "
        f"grad={float(aux['grad_norm']):.4f} | "
        f"nan={bool(aux['has_nan'])}"
    )
# Expected output (fresh untrained model — losses start high and descend):
# JIT step 0 | loss=58.5793 | grad=428.3538 | nan=False
# JIT step 1 | loss=48.1298 | grad=352.2870 | nan=False
# ...
# JIT step 4 | loss=30.6189 | grad=206.0480 | nan=False

# %% [markdown]
"""
## 5. Epoch-Based Training with Multiple Real Scenarios

`train_epoch` iterates a data iterator of `(trajectories, scene_context)` batches.
Here we build 3 batches from different WOD scenarios (SDC-recentred like the
main loop — the diffusion space must match) for a realistic demo.
"""

# %%
# Reuse 3 different prepared WOD scenarios as separate batches
data = scenes[:3]

trainer.set_epoch(10)  # Simulate later epoch for higher physics weight
epoch_metrics = trainer.train_epoch(iter(data), key=jax.random.key(99))
print(f"Epoch metrics: {len(epoch_metrics)} steps")
for m in epoch_metrics:
    print(f"  Step {m.step} | loss={m.total_loss:.4f} | physics_weight={m.physics_weight:.4f}")
# Expected output (single-scene steps draw one uniform timestep each,
# so per-step losses vary more than the batched main loop):
# Epoch metrics: 3 steps
#   Step 500 | loss=0.3937 | physics_weight=0.0251
#   Step 501 | loss=1.0638 | physics_weight=0.0251
#   Step 502 | loss=0.2136 | physics_weight=0.0251

# %% [markdown]
"""
## 6. Adaptive Physics Weight Schedule

The physics weight follows an exponential schedule from `initial_physics_weight`
(0.01) to `final_physics_weight` (1.0) over `transition_epochs` (50) epochs.
Early training focuses on the diffusion objective; later training enforces
kinematic plausibility on the real WOD trajectories.
"""

# %%
phys = DiffAVPhysicsLoss(
    DiffAVPhysicsConfig(
        initial_physics_weight=0.01,
        final_physics_weight=1.0,
        transition_epochs=100,
    )
)
for epoch in [0, 10, 25, 50, 100, 200]:
    w = float(phys.get_current_weight(epoch))
    print(f"  Epoch {epoch:4d}: physics_weight = {w:.4f}")
# Expected output:
#   Epoch    0: physics_weight = 0.0100
#   Epoch   10: physics_weight = ~0.02-0.05
#   Epoch   25: physics_weight = ~0.08-0.15
#   Epoch   50: physics_weight = ~0.30-0.50
#   Epoch  100: physics_weight = ~0.90-1.00
#   Epoch  200: physics_weight = 1.0000

# %% [markdown]
"""
## Production Notes

Techniques that pay off at production scale (256d/6-layer/80-step,
`scripts/train_wod.py`) but are out of scope for a minutes-long tutorial:

- **EMA of weights**: reference trajectory diffusers (CTG) maintain an
  exponential moving average of the model (decay 0.995, engaged after a few
  thousand steps) and sample/evaluate from the EMA copy — their primary
  stability mechanism alongside large batches.
- **Warmup length**: score-based diffusion references warm up for ~5000 steps
  on full training runs; keep the ~10% ratio used here.
- **x₀ parameterization**: CTG predicts the clean trajectory directly
  (`predict_epsilon=False`) instead of the noise. That removes the
  `1/sqrt(ᾱ_t)` error amplification in the x̂₀ reconstruction at the source —
  an architecture-level alternative to the ᾱ_t anneal used here.

## Next Steps

### Try These Experiments

1. Increase `NUM_AGENTS = 32` and `FUTURE_STEPS = 80` for full WOD-scale training
2. Replace the current-state context rows with full `SceneTokenizer` embeddings
   (agent history + map + ego, LayerNorm-ed by the tokenizer itself)
3. Train for 100+ epochs and observe the adaptive physics weight driving
   kinematic loss to zero

### Related Examples

- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)
- [Trajectory Diffusion Quick Reference](../models/trajectory-diffusion-quickref.md)
- [Bicycle Model Quick Reference](../physics/bicycle-model-quickref.md)

### API Reference

- [trainer](../../api/models/trainer.md)
- [trajectory_diffusion](../../api/models/trajectory_diffusion.md)
- [losses](../../api/physics/losses.md)
"""
