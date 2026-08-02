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
# Preference Construction Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~10 min (CPU) |
| **Prerequisites** | JAX arrays, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates how to build preference pairs from real
Waymo Open Dataset trajectories for DPO fine-tuning of the trajectory
diffusion model. Real WOD ground-truth trajectories naturally exhibit safe
driving behaviour — the safety reward consistently ranks GT-like trajectories
above physically impossible (corrupted) alternatives.

```
WODSource → real WOD trajectories (ground truth)
                    ↓
K=8 candidates = GT + noise at increasing levels
                    ↓
SafetyReward(candidates) → scores (GT-like = high, noisy = low)
                    ↓
PreferencePairBuilder.build_batch() → PreferenceBatch
                    ↓
PreferenceBatch.to_dpo_batch() → {"chosen", "rejected", "scene_contexts"}
```

## What You'll Learn

1. Extract real WOD agent trajectories as ground-truth candidates
2. Generate K=8 candidates with varying noise levels to span the safety spectrum
3. Score with `SafetyReward` and verify that GT trajectories rank highest
4. Build preference pairs and convert to DPO training format
5. Stack preference batches across multiple real driving scenarios

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | simulacrax.data | Real WOD TFRecord loading |
| `BicycleModelConstraint` | simulacrax.physics | Kinematic reward computation |
| `RewardFunction` protocol | artifex | Interface contract for reward functions |
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

import jax
import jax.numpy as jnp
from dotenv import load_dotenv

from simulacrax.alignment import (
    PreferencePairBuilder,
    PreferencePairConfig,
    RankingStrategy,
    SafetyReward,
    SafetyRewardConfig,
)
from simulacrax.data import resolve_wod_tfrecord_path
from simulacrax.data.wod_source import WODSource, WODSourceConfig


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 20  # Truncated to 20 for fast demo (production: 80)
MAX_AGENTS = 4  # Agents per scene
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


# %% [markdown]
"""
## 1. Load Real WOD Trajectories

`WODSource` provides real agent trajectories from WOD Motion TFRecords.
We extract 4D state vectors `[x, y, heading, speed]` for the future horizon.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
print(f"Loaded {len(source)} validation scenarios")
# Expected output (333 scenarios for a single validation shard):
# Loaded 333 validation scenarios


def _extract_traj(raw: dict, num_agents: int, future_steps: int) -> jax.Array:
    """Extract future (x, y, heading, speed) from raw WOD dict.

    Returns:
        trajectory: (num_agents, future_steps, 4) — [x, y, heading, speed]
    """
    sl = (slice(None, num_agents), slice(HIST_STEPS, HIST_STEPS + future_steps))
    vx = jnp.array(raw["state/all/velocity_x"][sl])
    vy = jnp.array(raw["state/all/velocity_y"][sl])
    return jnp.stack(
        [
            jnp.array(raw["state/all/x"][sl]),
            jnp.array(raw["state/all/y"][sl]),
            jnp.array(raw["state/all/bbox_yaw"][sl]),
            jnp.sqrt(vx**2 + vy**2),
        ],
        axis=-1,
    )  # (A, T, 4)


gt_traj = _extract_traj(source[0].data, MAX_AGENTS, FUTURE_STEPS)
print(f"Ground truth shape: {gt_traj.shape}  [agents, future_steps, state_dim]")
# Expected output:
# Ground truth shape: (4, 20, 4)  [agents, future_steps, state_dim]

# %% [markdown]
"""
## 2. Configure Safety Reward

The composite `SafetyReward` combines collision avoidance, kinematic
plausibility (bicycle model), boundary compliance, and comfort metrics.
Real WOD vehicles driving on real roads should score consistently high
on all components.
"""

# %%
config = SafetyRewardConfig(
    collision_threshold=2.0,
    collision_weight=1.0,
    kinematic_weight=1.0,
    boundary_weight=0.0,  # no road-edge data in this demo
    comfort_weight=0.5,
    dt=0.1,
)
reward = SafetyReward(config)
print(f"Safety reward components: {list(reward.component_rewards.keys())}")
# Expected output:
# Safety reward components: ['collision', 'kinematic', 'boundary', 'comfort']

# %% [markdown]
"""
## 3. Generate Candidate Trajectories from Real GT

We create `NUM_CANDIDATES=8` trajectory hypotheses by adding increasing
Gaussian noise to the real WOD ground truth. Candidate 0 = nearly perfect
(noise_scale≈0); Candidate 7 = heavily corrupted (noise_scale=3.5 m).

This models the range of outputs a diffusion model might produce:
from near-optimal samples to physically implausible rollouts.
"""

# %%
candidates_array = build_gt_noise_candidates(
    gt_traj, jax.random.key(42), num_candidates=NUM_CANDIDATES
)  # (K, A, T, 4)
noise_scales = jnp.linspace(0.0, 3.5, NUM_CANDIDATES)  # 0 → 3.5 m position noise
print(f"Candidates: {candidates_array.shape}  [{NUM_CANDIDATES} candidates, ...]")
print(f"Noise levels: {[f'{float(s):.2f}m' for s in noise_scales]}")
# Expected output:
# Candidates: (8, 4, 20, 4)  [8 candidates, ...]
# Noise levels: ['0.00m', '0.50m', '1.00m', '1.50m', '2.00m', '2.50m', '3.00m', '3.50m']

# %% [markdown]
"""
## 4. Score Trajectories

`SafetyReward` scores each candidate. Real GT (noise=0) should score
highest; progressively noisier candidates score lower — demonstrating
that WOD ground truth represents genuinely safe driving behaviour.
"""

# %%
scores = reward(candidates_array)
print(f"Scores: {[f'{float(s):.4f}' for s in scores]}")
print(f"  GT (noise=0.00m) score:       {float(scores[0]):.4f}  ← should be highest")
print(f"  Corrupted (noise=3.50m) score: {float(scores[-1]):.4f}  ← should be lowest")
print(f"GT outperforms corrupted: {bool(scores[0] > scores[-1])}")
# Expected output:
# Scores: ['x.xxxx', 'x.xxxx', ..., 'x.xxxx']
# GT (noise=0.00m) score:       x.xxxx  ← should be highest
# Corrupted (noise=3.50m) score: x.xxxx  ← should be lowest
# GT outperforms corrupted: True

# %% [markdown]
"""
## 5. Inspect Component Rewards

Break down the composite score to understand which safety dimension
most distinguishes real WOD trajectories from corrupted alternatives.
"""

# %%
components = reward.component_rewards
for name, comp_fn in components.items():
    comp_scores = comp_fn(candidates_array)
    print(
        f"{name:>12}: mean={float(jnp.mean(comp_scores)):.4f}  "
        f"min={float(jnp.min(comp_scores)):.4f}  "
        f"max={float(jnp.max(comp_scores)):.4f}"
    )
# Expected output (collision + kinematic most sensitive to noise):
# collision: mean=x.xxxx  min=x.xxxx  max=x.xxxx
# kinematic: mean=x.xxxx  min=x.xxxx  max=x.xxxx
# boundary: mean=0.0000  min=0.0000  max=0.0000
#   comfort: mean=x.xxxx  min=x.xxxx  max=x.xxxx

# %% [markdown]
"""
## 6. Build Preference Pairs

Use `PreferencePairBuilder` to rank candidates and construct chosen/rejected
pairs. Two strategies are available:

- **BEST_VS_WORST**: pair top-k with bottom-k (maximum reward margin, strong signal)
- **ADJACENT**: pair consecutive ranks (more pairs, smaller margins)
"""

# %%
# Strategy 1: Best vs Worst — pairs GT-like (chosen) vs corrupted (rejected)
bvw_config = PreferencePairConfig(
    ranking_strategy=RankingStrategy.BEST_VS_WORST,
    num_candidates=NUM_CANDIDATES,
    top_k=2,
    min_reward_margin=0.0,
)
builder = PreferencePairBuilder(reward_fn=reward, config=bvw_config)
batch = builder.build_batch(candidates_array, conditions=None)

print("Best vs Worst strategy:")
print(f"  Chosen shape:   {batch.chosen.shape}  ← GT-like (low noise)")
print(f"  Rejected shape: {batch.rejected.shape}  ← Corrupted (high noise)")
print(f"  Margins: {[f'{float(m):.4f}' for m in batch.margins]}")
# Expected output:
# Best vs Worst strategy:
#   Chosen shape:   (2, 4, 20, 4)  ← GT-like (low noise)
#   Rejected shape: (2, 4, 20, 4)  ← Corrupted (high noise)
#   Margins: ['x.xxxx', 'x.xxxx']

# %%
# Strategy 2: Adjacent — pairs each consecutive rank
adj_config = PreferencePairConfig(
    ranking_strategy=RankingStrategy.ADJACENT,
    num_candidates=NUM_CANDIDATES,
)
adj_builder = PreferencePairBuilder(reward_fn=reward, config=adj_config)
adj_batch = adj_builder.build_batch(candidates_array, conditions=None)

print("Adjacent strategy:")
print(f"  Chosen shape:   {adj_batch.chosen.shape}")
print(f"  Rejected shape: {adj_batch.rejected.shape}")
print(f"  Num pairs:      {adj_batch.chosen.shape[0]}  (= {NUM_CANDIDATES}-1 adjacent pairs)")
# Expected output:
# Adjacent strategy:
#   Chosen shape:   (7, 4, 20, 4)
#   Rejected shape: (7, 4, 20, 4)
#   Num pairs:      7  (= 8-1 adjacent pairs)

# %% [markdown]
"""
## 7. Convert to DPO Format

`to_dpo_batch()` produces the dict format expected by the DPO trainer.
"""

# %%
dpo_batch = batch.to_dpo_batch()
print(f"DPO batch keys: {list(dpo_batch.keys())}")
print(f"  chosen shape:   {dpo_batch['chosen'].shape}")
print(f"  rejected shape: {dpo_batch['rejected'].shape}")
# Expected output (no scene_context passed, so no "scene_contexts" key):
# DPO batch keys: ['chosen', 'rejected']
#   chosen shape:   (2, 4, 20, 4)
#   rejected shape: (2, 4, 20, 4)

# %% [markdown]
"""
## 8. Multi-Scene Stacking

Stack preference batches across multiple real WOD scenarios to build
a training dataset. Each scenario contributes independently ranked pairs.
"""

# %%
# Load 3 different real WOD scenarios and noise each with the shared helper
scene_candidates = [
    build_gt_noise_candidates(
        _extract_traj(source[idx].data, MAX_AGENTS, FUTURE_STEPS),
        jax.random.key(idx),
        num_candidates=NUM_CANDIDATES,
    )
    for idx in range(3)
]

scene_conditions = [None, None, None]
multi_batch = builder.build_batch_from_scenes(
    candidates_per_scene=scene_candidates,
    conditions_per_scene=scene_conditions,
)
print(f"Multi-scene batch: {multi_batch.chosen.shape[0]} pairs from 3 real scenarios")
# Expected output:
# Multi-scene batch: 6 pairs from 3 real scenarios

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Increase `FUTURE_STEPS = 80` for full WOD-scale preference construction
2. Set `min_reward_margin=0.1` to filter low-quality pairs
3. Replace the noisy candidates with actual `TrajectoryDiffusionModel.sample()` outputs

### Related Examples

- [DPO Fine-Tuning Quick Reference](02_dpo_finetuning_quickref.py)
- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)

### API Reference

- [rewards](../../api/alignment/rewards.md)
- [preferences](../../api/alignment/preferences.md)
"""
