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
# DPO Fine-Tuning Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~10 min (GPU with checkpoint; small-model CPU fallback) |
| **Prerequisites** | JAX arrays, WOD data, preference construction quickref |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates DPO fine-tuning of a trajectory diffusion
model on real Waymo Open Dataset preference pairs. The policy is restored
from the WOD-trained checkpoint **before** the frozen reference is cloned:
DPO's implicit reward is `beta * (log p_policy - log p_ref)` (Rafailov et
al. 2023), so a randomly initialised reference makes both the reward and
the KL anchor meaningless.

The Diffusion-DPO trainer (Wallace et al. 2023) estimates trajectory
log-probabilities via Monte Carlo noise prediction error — no explicit
likelihood computation required:

```
checkpoints/wod-mini → restored SFT policy model
                    ↓
create_reference_model(model) → frozen SFT reference copy
                    ↓
WODSource → real GT trajectories → GT + noise candidates (K=8)
                    ↓
PreferencePairBuilder → chosen (GT-like) / rejected (corrupted)
                    ↓
DPOAlignmentTrainer.train_step(batch, key) → implicit-reward metrics
```

## What You'll Learn

1. Restore the WOD-trained checkpoint as the SFT anchor for DPO
2. Create a frozen reference model with `create_reference_model` *after* restoring
3. Build preference pairs from real WOD trajectories using `SafetyReward`
4. Run jitted DPO steps and read the implicit-reward margin and accuracy
5. Why diffusion DPO uses a large `beta` (full derivation in the tutorial)

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | simulacrax.data | Real WOD TFRecord loading |
| `create_scenario_miner` | simulacrax.api | Checkpoint restore into the SDK model |
| `NoiseSchedule` | artifex | Forward diffusion for log-prob estimation |
| `create_optimizer` | opifex | Gradient-clipped optimizer creation |
| `nnx.clone` | Flax NNX | Deep-copy for frozen reference model |
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
from pathlib import Path

import jax
import jax.numpy as jnp
from dotenv import load_dotenv
from flax import nnx
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

from simulacrax.alignment import (
    create_reference_model,
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    PreferencePairBuilder,
    PreferencePairConfig,
    SafetyReward,
)
from simulacrax.api import create_scenario_miner, MinerConfig
from simulacrax.core.constants import MINER_STATE_OFFSETS, MINER_STATE_SCALES
from simulacrax.data import prepare_full_horizon_scene, resolve_wod_tfrecord_path
from simulacrax.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from simulacrax.data.wod_source import WODSource, WODSourceConfig


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 80  # Full WOD horizon — matches the checkpoint
MAX_AGENTS = 8  # Agents per scene — matches the checkpoint
CONTEXT_DIM = 128  # Scene embedding dimension — matches the checkpoint
NUM_CANDIDATES = 8  # Trajectory hypotheses per scene


def build_gt_noise_candidates(
    gt_scene: jax.Array,
    key: jax.Array,
    *,
    num_candidates: int = 8,
    max_noise_metres: float = 3.5,
) -> jax.Array:
    """Stack the clean scene with increasingly position-noised variants.

    Candidate 0 is the untouched ground truth; the last candidate carries
    ``max_noise_metres`` of Gaussian x/y noise. Heading and speed channels
    stay clean so the corruption is purely positional. Returns
    ``(num_candidates, num_agents, future_steps, state_dim)``.
    """
    noise_scales = jnp.linspace(0.0, max_noise_metres, num_candidates)
    variants = []
    for index in range(num_candidates):
        noise = jax.random.normal(jax.random.fold_in(key, index), gt_scene.shape)
        variants.append(gt_scene + (noise * noise_scales[index]).at[..., 2:].set(0))
    return jnp.stack(variants)


def to_model_space(trajectories: jax.Array) -> jax.Array:
    """Map metre-space trajectories into the model's normalized diffusion space.

    Applies the same ``(x + offsets) / scales`` transform the model uses
    internally for training losses, so the DPO log-prob estimator sees the
    space the checkpoint was trained in.
    """
    return (trajectories + jnp.asarray(MINER_STATE_OFFSETS)) / jnp.asarray(MINER_STATE_SCALES)


# %% [markdown]
"""
## 1. Restore the SFT Policy Model

DPO fine-tunes a model that already generates plausible trajectories.
`create_scenario_miner` restores `checkpoints/wod-mini` (train it with
`scripts/train_wod.py`); without the checkpoint a small fresh model is used
and every DPO metric below is meaningless — the fallback exists only so the
file executes end-to-end.
"""

# %%
CHECKPOINT_DIR = Path("checkpoints/wod-mini")
USE_CHECKPOINT = CHECKPOINT_DIR.exists()
miner = create_scenario_miner(
    MinerConfig(
        model_path=str(CHECKPOINT_DIR) if USE_CHECKPOINT else "",
        max_agents=MAX_AGENTS,
        prediction_horizon=FUTURE_STEPS,
        context_dim=CONTEXT_DIM,
        num_diffusion_steps=100,
        hidden_dim=256 if USE_CHECKPOINT else 32,
        num_blocks=6 if USE_CHECKPOINT else 2,
        num_heads=8 if USE_CHECKPOINT else 2,
    )
)
model = miner.model
if USE_CHECKPOINT:
    print(f"Restored WOD-trained policy from {CHECKPOINT_DIR}")
else:
    print(
        "WARNING: no checkpoint at checkpoints/wod-mini — using an untrained model.\n"
        "A random reference makes DPO's implicit reward meaningless; train one with\n"
        "`python scripts/train_wod.py` for real results."
    )
# Expected output (with checkpoint):
# Restored WOD-trained policy from checkpoints/wod-mini

# %% [markdown]
"""
## 2. Build Preference Pairs from Real WOD Data

Real WOD trajectories are re-centred on the SDC and validity-masked with
`prepare_full_horizon_scene` — the same helper the training script uses —
then noised into `K=8` candidates: near-perfect GT is `chosen`, the most
corrupted is `rejected`. Scene context rows are the per-agent
`[x, y, vx, vy]` encoding the checkpoint was conditioned on.

Scoring happens in raw metre space (`SafetyReward` thresholds are metres);
the DPO batch is then mapped into the model's normalized diffusion space,
because the trainer's log-prob estimator calls `q_sample`/`predict_noise`
directly — the space `compute_loss` trained the checkpoint in.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=32,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
print(f"Loaded {len(source)} validation scenarios")
# Expected output (333 scenarios for a single validation shard):
# Loaded 333 validation scenarios

norm_op = AgentNormalizationOperator(
    AgentNormalizationConfig(current_step_idx=HIST_STEPS - 1), rngs=nnx.Rngs(0)
)
prepared = None
scene_index = 0
while prepared is None:
    recentred, _, _ = norm_op.apply(dict(source[scene_index].data), {}, {})
    prepared = prepare_full_horizon_scene(
        recentred,
        num_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
    scene_index += 1
gt_traj, current_states = prepared
scene_ctx = jnp.pad(current_states, ((0, 0), (0, CONTEXT_DIM - current_states.shape[1])))
print(f"Ground truth: {gt_traj.shape}, scene context: {scene_ctx.shape}")
# Expected output:
# Ground truth: (8, 80, 4), scene context: (8, 128)

candidates = build_gt_noise_candidates(
    gt_traj, jax.random.key(42), num_candidates=NUM_CANDIDATES
)  # (K, A, T, 4)

reward = SafetyReward()
scores = reward(candidates)
print(f"GT score (noise=0):           {float(scores[0]):.4f}  ← should be highest")
print(f"Corrupted score (noise=3.5m): {float(scores[-1]):.4f}  ← should be lowest")
print(f"GT outperforms corrupted:     {bool(scores[0] > scores[-1])}")
# Expected output:
# GT score (noise=0):           <highest of the eight>
# Corrupted score (noise=3.5m): <lowest of the eight>
# GT outperforms corrupted:     True

# Build preference batch, then map trajectories into the model's diffusion space
builder = PreferencePairBuilder(
    reward_fn=reward,
    config=PreferencePairConfig(top_k=2, num_candidates=NUM_CANDIDATES),
)
batch = builder.build_batch(candidates, conditions=None, scene_context=scene_ctx)
dpo_batch = batch.to_dpo_batch()
dpo_batch["chosen"] = to_model_space(dpo_batch["chosen"])
dpo_batch["rejected"] = to_model_space(dpo_batch["rejected"])
print(f"DPO batch keys: {list(dpo_batch.keys())}")
print(f"Chosen shape:   {dpo_batch['chosen'].shape}")
print(f"Rejected shape: {dpo_batch['rejected'].shape}")
# Expected output:
# DPO batch keys: ['chosen', 'rejected', 'scene_contexts']
# Chosen shape:   (2, 8, 80, 4)
# Rejected shape: (2, 8, 80, 4)

# %% [markdown]
"""
## 3. Configure DPO Trainer

The frozen reference is cloned from the **restored** policy via
`create_reference_model` (uses `nnx.clone`) — the order matters: clone
after loading the checkpoint, never before.

`beta=1000` looks enormous next to token-DPO's 0.1–0.5, but the diffusion
log-prob proxy is a negative denoising MSE whose chosen/rejected differences
are tiny in normalized space; Diffusion-DPO uses beta 2000–5000 for the same
reason (Wallace et al. 2023). The fine-tuning learning rate is small (1e-5)
so the policy stays close to its SFT anchor.
"""

# %%
reference = create_reference_model(model)
print("Reference model created (frozen copy of the restored policy)")
# Expected output:
# Reference model created (frozen copy of the restored policy)

tx = create_optimizer(OptimizerConfig(optimizer_type="adam", learning_rate=1e-5, gradient_clip=1.0))
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

dpo_config = DPOAlignmentConfig(
    beta=1000.0,  # MSE-scale log-prob proxy needs a large temperature
    num_log_prob_samples=8,  # Monte Carlo samples for Diffusion-DPO
    physics_weight=0.0,  # No physics regularisation in this demo
)
trainer = DPOAlignmentTrainer(
    model=model,
    optimizer=optimizer,
    config=dpo_config,
    reference_model=reference,
)
print(f"DPO beta={dpo_config.beta}, K={dpo_config.num_log_prob_samples} MC samples")
print("DPO trainer ready")
# Expected output:
# DPO beta=1000.0, K=8 MC samples
# DPO trainer ready

# %% [markdown]
"""
## 4. Run Training Steps on Real Preference Data

`train_step` runs the jitted Diffusion-DPO update on the WOD preference
pairs; the reference model stays frozen. The reported margin and accuracy
are computed on the **implicit reward** `beta * (log p_policy - log p_ref)`
— at step 0 policy and reference are identical, so the margin starts at
exactly zero and grows as the policy shifts mass toward chosen trajectories.
"""

# %%
for step in range(5):
    step_key = jax.random.key(step)
    metrics = trainer.train_step(dpo_batch, step_key)
    print(
        f"Step {step}: loss={metrics.dpo_loss:.4f} "
        f"acc={metrics.reward_accuracy:.2%} "
        f"margin={metrics.reward_margin:.4f} "
        f"grad={metrics.grad_norm:.6f}"
    )
# Expected (with checkpoint): step-0 margin is 0.0 and loss ~0.6931 (= log 2);
# the implicit-reward margin then increases over the steps.

# %% [markdown]
"""
## 5. Detailed Alignment Metrics

`DPOAlignmentMetrics` provides complete diagnostics for training monitoring:

- **dpo_loss**: DPO objective (lower = better alignment; log 2 at start)
- **reward_accuracy**: % of pairs where the implicit reward ranks chosen >
  rejected (target: 100%)
- **reward_margin**: mean implicit-reward gap
  `beta * [(policy - ref) chosen - (policy - ref) rejected]`
  (larger = more confident preference)
- **policy_chosen/rejected_log_prob**: raw MC log-prob proxies (diagnostic
  only — content-biased, which is exactly why the reward is
  reference-anchored)
- **grad_norm**: gradient magnitude (monitors stability)
"""

# %%
print("--- Final Metrics ---")
print(f"DPO loss:               {metrics.dpo_loss:.4f}")
print(f"Policy chosen log-p:    {metrics.policy_chosen_log_prob:.4f}")
print(f"Policy rejected log-p:  {metrics.policy_rejected_log_prob:.4f}")
print(f"Implicit-reward acc:    {metrics.reward_accuracy:.2%}")
print(f"Implicit-reward margin: {metrics.reward_margin:.4f}")
print(f"Gradient norm:          {metrics.grad_norm:.6f}")
# Expected: margin > 0 after training on GT-vs-corrupted pairs; accuracy
# reaches 100% once the policy separates chosen from rejected.

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Scale to many scenes with `build_batch_from_scenes` — the tutorial runs
   200 minibatched DPO steps over 32 scenes
2. Add `physics_weight=0.5` to penalise kinematic violations during alignment
3. Set `reference_free=True` with `simpo_gamma=0.5` for the SimPO-style
   objective (see the tutorial for honest labeling of that mode)

### Related Examples

- [Preference Construction Quick Reference](01_preference_construction_quickref.py)
- [DPO Fine-Tuning Tutorial](03_dpo_finetuning_tutorial.py)
- [Physics-Informed Training Tutorial](../models/02_physics_informed_training_tutorial.py)

### API Reference

- [dpo_trainer](../../api/alignment/dpo_trainer.md)
- [preferences](../../api/alignment/preferences.md)
- [rewards](../../api/alignment/rewards.md)
"""
