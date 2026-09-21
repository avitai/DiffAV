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
# DPO Fine-Tuning with Scenario Steering — Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~30 min (GPU recommended) |
| **Prerequisites** | DPO Fine-Tuning Quick Reference, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This tutorial runs the full standard-DPO alignment pipeline on real Waymo
Open Dataset preference pairs, then extends it with scenario steering.

The main loop is **standard DPO** (Rafailov et al. 2023): the WOD-trained
checkpoint is the SFT anchor, a frozen reference copy is cloned from it,
and the policy is optimised on the implicit reward
`beta * (log p_policy - log p_ref)` over shuffled minibatches drawn from
preference pairs across many WOD scenes. A short, honestly-labeled
reference-free contrast (SimPO-style target margin) follows, and the final
sections steer generation toward a target scenario type using preference
pairs the model itself sampled.

```
checkpoints/wod-mini → SFT policy  +  create_reference_model → frozen reference
        |
WODSource (32 scenes) → GT + noise candidates → SafetyReward ranking
        |
build_batch_from_scenes → 64 preference pairs (+ real scene contexts)
        |
200 × DPOAlignmentTrainer.train_step on 16-pair shuffled minibatches
        |
margin / win-rate curves → the resolved output directory
        |
reference-free contrast (simpo_gamma)      steering: sample_and_score
        |                                     → build_steering_pairs
        v                                     → steer_step / ScenarioMiner.steer()
```

## What You'll Learn

1. Restore the WOD-trained checkpoint as the SFT anchor for standard DPO
2. Build preference pairs across many WOD scenes with `build_batch_from_scenes`
3. Run a 200-step standard DPO loop on fresh shuffled minibatches
4. Why diffusion DPO needs `beta ~ 1000` while token DPO uses 0.1–0.5,
   and how batch size and stratified timesteps tame estimator variance
5. Plot implicit-reward margin and win-rate curves
6. Contrast with a reference-free (SimPO-style) objective, labeled honestly
7. Steer generation with `steer_step()` on reward-ranked model samples
8. Use `ScenarioMiner.steer()` for high-level steered generation
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
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from substrax.artifacts import resolve_output_dir


PLOT_DIR = resolve_output_dir("examples").path
import numpy as np
from dotenv import load_dotenv
from flax import nnx
from substrax.optim import create_optimizer, OptimizerConfig

from diffav.alignment import (
    compute_scenario_reward,
    create_reference_model,
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    PreferencePairBuilder,
    PreferencePairConfig,
    SafetyReward,
    ScenarioSteeringConfig,
    ScenarioSteeringTrainer,
)
from diffav.api import create_scenario_miner, MinerConfig
from diffav.core.constants import MINER_STATE_OFFSETS, MINER_STATE_SCALES
from diffav.data import prepare_full_horizon_scene, resolve_wod_tfrecord_path
from diffav.data.operators import AgentNormalizationConfig, AgentNormalizationOperator
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer
from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 80  # Full WOD horizon — matches the checkpoint
MAX_AGENTS = 8  # Agents per scene — matches the checkpoint
CONTEXT_DIM = 128  # Scene embedding dimension — matches the checkpoint
NUM_CANDIDATES = 8  # Trajectory hypotheses per scene
# Smoke mode (set by the example execution tests) shrinks the loops so the
# CPU tier finishes quickly; real runs keep the full showcase schedule.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
NUM_SCENES = 8 if _SMOKE else 32  # WOD scenes contributing preference pairs
# Pairs per optimizer update. Reference DPO recipes update on large
# effective batches, not the 1-2 pairs a single device holds: Diffusion-DPO
# trains 1 pair/GPU but accumulates to an effective batch of 2048 pairs,
# and the original DPO release trains Anthropic-HH with batch_size 32-64.
# 16 pairs (a quarter of the pool) is the largest reference-scale batch
# that fits a 24 GB device with the checkpointed estimator.
MINIBATCH_PAIRS = 4 if _SMOKE else 16
NUM_DPO_STEPS = 20 if _SMOKE else 200  # Standard-DPO training steps
NUM_CONTRAST_STEPS = 10 if _SMOKE else 50  # Reference-free contrast steps
NUM_STEER_STEPS = 3 if _SMOKE else 12  # Scenario-steering steps
DPO_BETA = 1000.0  # Diffusion-DPO temperature (see Section 3)
DPO_LEARNING_RATE = 1e-6  # Full-model diffusion-DPO stays near TRL's 1e-6 default;
# 1e-5 measurably degrades sample quality (policy drifts off the SFT anchor)
# Monte Carlo samples for the log-prob proxy. Timesteps are stratified
# across the schedule (one draw per bin), so K=16 covers the schedule
# densely; activation checkpointing makes the cost linear in compute only.
NUM_MC_SAMPLES = 2 if _SMOKE else 16

print(f"JAX devices: {jax.devices()}")
print(f"Future steps: {FUTURE_STEPS}, Agents: {MAX_AGENTS}")
# Expected output:
# JAX devices: [<available devices>]
# Future steps: 80, Agents: 8


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

DPO fine-tunes a model that already generates plausible trajectories — the
"SFT" stage of the alignment recipe. `create_scenario_miner` restores
`checkpoints/wod-mini` (train it with `scripts/train_wod.py`). Without the
checkpoint, a short **jitted** synthetic warm-up pulls a small fallback
model onto the data scale so the tutorial still executes; its metrics are
demo-grade only.
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
        hidden_dim=256 if USE_CHECKPOINT else 64,
        num_blocks=6 if USE_CHECKPOINT else 2,
        num_heads=8 if USE_CHECKPOINT else 2,
    )
)
model = miner.model


def _warmup_batch(
    key: jax.Array, num_agents: int, horizon: int, context_dim: int
) -> tuple[jax.Array, jax.Array]:
    """Synthetic constant-velocity trajectories + miner-style context rows."""
    speed_key, offset_key = jax.random.split(key)
    speeds = jax.random.uniform(speed_key, (num_agents, 1), minval=5.0, maxval=15.0)
    offsets = jax.random.uniform(offset_key, (num_agents, 1), minval=-20.0, maxval=20.0)
    t_axis = jnp.arange(horizon)[None, :] * 0.1
    x = speeds * t_axis
    y = jnp.broadcast_to(offsets, (num_agents, horizon))
    heading = jnp.zeros_like(x)
    velocity = jnp.broadcast_to(speeds, (num_agents, horizon))
    trajectories = jnp.stack([x, y, heading, velocity], axis=-1)
    # [x0, y0, vx, vy] rows (heading 0 → vx = speed), zero-padded to context_dim
    rows = jnp.concatenate([x[:, :1], y[:, :1], speeds, jnp.zeros_like(speeds)], axis=-1)
    context = jnp.pad(rows, ((0, 0), (0, context_dim - rows.shape[1])))
    return trajectories, context


def warm_up(target_model: TrajectoryDiffusionModel, seed: int, steps: int = 300) -> float:
    """Short jitted warm-up on synthetic trajectories (SFT stand-in)."""
    trainer = TrajectoryTrainer(target_model, TrainerConfig(num_epochs=1, log_interval=100_000))
    jit_step = nnx.jit(trainer.compute_train_step)
    loss = jnp.array(0.0)
    for step in range(steps):
        step_key = jax.random.fold_in(jax.random.key(seed), step)
        trajectories, context = _warmup_batch(step_key, MAX_AGENTS, FUTURE_STEPS, CONTEXT_DIM)
        loss, _ = jit_step(trainer.model, trainer.optimizer, trajectories, context, step_key, 0)
    return float(loss)


if USE_CHECKPOINT:
    print(f"Restored WOD-trained policy from {CHECKPOINT_DIR}")
else:
    print("No checkpoint found — running a 300-step jitted synthetic warm-up instead.")
    warmup_loss = warm_up(model, seed=100)
    print(f"Warm-up complete — diffusion loss: {warmup_loss:.3f}")
# Expected output (with checkpoint):
# Restored WOD-trained policy from checkpoints/wod-mini

# %% [markdown]
"""
## 2. Build Preference Pairs from Many WOD Scenes

A single static batch invites memorisation; standard DPO shuffles a pool
of pairs drawn from many scenes. Each scene is re-centred on the SDC and
validity-masked with `prepare_full_horizon_scene` (the training-script
helper), noised into `K=8` candidates, ranked by `SafetyReward`, and the
top-2 / bottom-2 form two pairs per scene via `build_batch_from_scenes`.
Scene context rows are the per-agent `[x, y, vx, vy]` encoding the
checkpoint was conditioned on.

Scoring happens in raw metre space (`SafetyReward` thresholds are metres);
the stacked pairs are then mapped into the model's normalized diffusion
space, because the trainer's log-prob estimator calls
`q_sample`/`predict_noise` directly — the space the checkpoint was
trained in.
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
norm_op = AgentNormalizationOperator(
    AgentNormalizationConfig(current_step_idx=HIST_STEPS - 1), rngs=nnx.Rngs(0)
)

candidates_per_scene: list[jax.Array] = []
contexts_per_scene: list[jax.Array] = []
scene_index = 0
while len(candidates_per_scene) < NUM_SCENES and scene_index < len(source):
    recentred, _, _ = norm_op.apply(dict(source[scene_index].data), {}, {})
    prepared = prepare_full_horizon_scene(
        recentred,
        num_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
    if prepared is not None:
        trajectories, current_states = prepared
        candidates_per_scene.append(
            build_gt_noise_candidates(
                trajectories, jax.random.key(scene_index), num_candidates=NUM_CANDIDATES
            )
        )
        contexts_per_scene.append(
            jnp.pad(current_states, ((0, 0), (0, CONTEXT_DIM - current_states.shape[1])))
        )
    scene_index += 1
print(f"Prepared {len(candidates_per_scene)} scenes from {scene_index} records")
# Expected output:
# Prepared 32 scenes from <scanned record count> records

builder = PreferencePairBuilder(
    reward_fn=SafetyReward(),
    config=PreferencePairConfig(top_k=2, num_candidates=NUM_CANDIDATES),
)
preference_batch = builder.build_batch_from_scenes(
    candidates_per_scene=candidates_per_scene,
    conditions_per_scene=[None] * len(candidates_per_scene),
    scene_contexts=contexts_per_scene,
)
dpo_data = preference_batch.to_dpo_batch()
dpo_data["chosen"] = to_model_space(dpo_data["chosen"])
dpo_data["rejected"] = to_model_space(dpo_data["rejected"])
num_pairs = dpo_data["chosen"].shape[0]
print(f"Preference pool: {num_pairs} pairs, chosen shape {dpo_data['chosen'].shape}")
print(f"Scene contexts:  {dpo_data['scene_contexts'].shape}")
# Expected output:
# Preference pool: 64 pairs, chosen shape (64, 8, 80, 4)
# Scene contexts:  (64, 8, 128)

# %% [markdown]
r"""
## 3. Standard DPO Training Loop

The frozen reference is cloned from the restored policy with
`create_reference_model` (`nnx.clone`), and the trainer evaluates it on
every pair with **no gradients** — exactly the reference pass of the
original DPO recipe (Rafailov et al. 2023). Each step draws a fresh
shuffled 16-pair minibatch from the 64-pair pool; the update itself is the
prebuilt `nnx.jit`-compiled step inside `train_step`.

### Keeping the Monte Carlo variance down

Every log-prob is a K-sample estimate, so the per-step curves carry
sampling noise on top of learning. Three reference-grounded choices keep
that variance in check rather than smoothing it away:

1. **Reference-scale batches.** DPO updates average the pairwise logits
   over large effective batches — 32–64 pairs in the original DPO release,
   2048 pairs (1/GPU × 128 accumulation × 16 GPUs) in Diffusion-DPO. With
   activation checkpointing inside the estimator, 16 pairs fit a 24 GB
   device outright — minibatch gradient variance falls as 1/batch.
2. **Stratified timesteps.** The estimator draws one timestep per
   schedule bin instead of K independent uniforms (the discrete analog of
   the low-discrepancy sampler of Kingma et al. 2021, motivated by the
   timestep-noise analysis of Nichol & Dhariwal 2021): every estimate
   covers early, middle, and late noise levels, so it cannot cluster by
   chance on an unrepresentative slice of the schedule.
3. **Shared draws.** Policy/reference and chosen/rejected are always
   scored on the same (t, eps) draws — Wallace et al. (2023) — so
   per-draw noise cancels in the log-ratios.

### Why `beta ~ 1000` here but 0.1–0.5 in LLM DPO?

Token-level DPO works with exact sequence log-likelihoods: chosen and
rejected responses typically differ by $O(1)$ nats, so a temperature of
0.1–0.5 puts $\beta \Delta$ in the log-sigmoid's active range. The
Diffusion-DPO log-prob proxy is a **negative denoising MSE**
($\log p_\theta(x) \approx -\mathbb{E}_{t,\epsilon}\,
\|\epsilon - \epsilon_\theta(x_t, t)\|^2$), and in normalized diffusion
space two trajectories that differ by a few metres change that MSE only in
the third or fourth decimal. Multiplying by $\beta \sim 10^3$ rescales those
tiny differences into a usable learning signal — Wallace et al. (2023) use
$\beta$ = 2000–5000 for image diffusion for exactly this reason. The same
paper's practice of evaluating policy/reference and chosen/rejected on the
**same** $(t, \epsilon)$ draws (so Monte Carlo noise cancels in the
log-ratios) is built into the trainer.
"""

# %%
reference = create_reference_model(model)
optimizer = create_optimizer(
    model,
    OptimizerConfig(optimizer_type="adam", learning_rate=DPO_LEARNING_RATE, gradient_clip_norm=1.0),
)

dpo_config = DPOAlignmentConfig(
    beta=DPO_BETA,
    num_log_prob_samples=NUM_MC_SAMPLES,
    reference_free=False,  # standard DPO — frozen reference required
)
trainer = DPOAlignmentTrainer(
    model=model,
    optimizer=optimizer,
    config=dpo_config,
    reference_model=reference,
)

dpo_losses: list[float] = []
dpo_win_rates: list[float] = []
dpo_margins: list[float] = []

# Per-step metrics mix learning with Monte Carlo and 16-pair minibatch
# noise. The honest learning curve is a periodic evaluation over the FULL
# preference pool with FIXED keys: deterministic draws, so consecutive
# points differ only through the policy update. The eval is jitted with
# the same pattern as the trainer's compiled step (policy explicit and
# traced, frozen reference in the closure) and walks the pool in
# minibatch-sized chunks: identical shapes to the training step, so eval
# peak memory stays at train-step scale, and equal-sized chunks make the
# mean of chunk means exactly the pool mean.
EVAL_EVERY = 2 if _SMOKE else 10
EVAL_KEY = jax.random.key(2_024)
eval_steps: list[int] = []
eval_losses: list[float] = []
eval_margins: list[float] = []
eval_win_rates: list[float] = []


@nnx.jit
def _eval_dpo_loss(
    policy: TrajectoryDiffusionModel,
    batch: dict[str, jax.Array],
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    return trainer.compute_dpo_loss(batch, key, policy_model=policy)


def _evaluate_full_pool() -> tuple[float, float, float]:
    losses: list[float] = []
    margins: list[float] = []
    win_rates: list[float] = []
    for start in range(0, num_pairs, MINIBATCH_PAIRS):
        chunk = {name: array[start : start + MINIBATCH_PAIRS] for name, array in dpo_data.items()}
        loss, aux = _eval_dpo_loss(trainer.model, chunk, jax.random.fold_in(EVAL_KEY, start))
        losses.append(float(loss))
        margins.append(float(aux["reward_margin"]))
        win_rates.append(float(aux["reward_accuracy"]))
    return float(np.mean(losses)), float(np.mean(margins)), float(np.mean(win_rates))


for step in range(NUM_DPO_STEPS):
    perm_key, loss_key = jax.random.split(jax.random.key(step))
    take = jax.random.permutation(perm_key, num_pairs)[:MINIBATCH_PAIRS]
    minibatch = {name: array[take] for name, array in dpo_data.items()}
    metrics = trainer.train_step(minibatch, loss_key)
    dpo_losses.append(metrics.dpo_loss)
    dpo_win_rates.append(metrics.reward_accuracy)
    dpo_margins.append(metrics.reward_margin)
    if step % EVAL_EVERY == 0 or step == NUM_DPO_STEPS - 1:
        eval_loss, eval_margin, eval_win = _evaluate_full_pool()
        eval_steps.append(step)
        eval_losses.append(eval_loss)
        eval_margins.append(eval_margin)
        eval_win_rates.append(eval_win)
    if step % 20 == 0 or step == NUM_DPO_STEPS - 1:
        print(
            f"DPO step {step:3d}: "
            f"loss={metrics.dpo_loss:.4f}  "
            f"win-rate={metrics.reward_accuracy:.2%}  "
            f"margin={metrics.reward_margin:.4f}"
        )
# Expected: loss starts at ~0.6931 (= log 2, policy == reference) and
# decreases; the implicit-reward margin increases over training and the
# win rate climbs toward 100%.

# %% [markdown]
"""
## 4. Alignment Curves

DPO loss, implicit-reward win rate, and implicit-reward margin. Faint
traces are raw per-step values on 16-pair minibatches with fresh Monte
Carlo draws — noisy by construction. The solid curves evaluate the full
preference pool under a fixed key every few steps, so they isolate actual
learning. The margin is `beta * [(policy − ref)_chosen −
(policy − ref)_rejected]` — reference-anchored, so it starts at exactly
zero.
"""

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
steps_range = np.arange(NUM_DPO_STEPS)

axes[0].plot(steps_range, dpo_losses, color="steelblue", alpha=0.25, linewidth=1)
axes[0].plot(eval_steps, eval_losses, color="steelblue", linewidth=2, label="full-pool eval")
axes[0].axhline(float(np.log(2.0)), color="gray", linestyle="--", linewidth=1, label="log 2")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("DPO Loss")
axes[0].set_title("DPO Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(steps_range, dpo_win_rates, color="seagreen", alpha=0.25, linewidth=1)
axes[1].plot(eval_steps, eval_win_rates, color="seagreen", linewidth=2)
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Win Rate")
axes[1].set_title("Implicit-Reward Win Rate")
axes[1].grid(True, alpha=0.3)

axes[2].plot(steps_range, dpo_margins, color="darkorange", alpha=0.25, linewidth=1)
axes[2].plot(eval_steps, eval_margins, color="darkorange", linewidth=2)
axes[2].set_xlabel("Training Step")
axes[2].set_ylabel("Reward Margin")
axes[2].set_title("Implicit-Reward Margin (chosen − rejected)")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Standard DPO on WOD Preference Pairs", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "dpo_alignment_curves.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Alignment curves saved to {PLOT_DIR / 'dpo_alignment_curves.png'}")
# Expected output:
# Alignment curves saved to <output dir>/dpo_alignment_curves.png

# %% [markdown]
"""
## 5. Reference-Free Contrast (SimPO-Style Target Margin)

Dropping the reference model gives the CPO/SimPO family of objectives.
Honest labeling: true SimPO (Meng et al. 2024) length-normalizes sequence
log-probabilities; our Monte Carlo proxy is already a per-trajectory mean,
so what runs here is **reference-free DPO with SimPO's target reward
margin** `gamma` — the same construction as the `simpo` loss in TRL's CPO
trainer, where `gamma` shifts the logits before the log-sigmoid.

The contrast trains a *fresh clone of the SFT weights* (not the
DPO-trained policy) so the two runs are comparable. Note the margins are
not on the same scale as Section 3: without the reference subtraction the
implicit reward is just `beta * log p_policy`, which is biased by
trajectory content.
"""

# %%
contrast_model = create_reference_model(reference)  # fresh SFT copy, untouched by the DPO run
contrast_optimizer = create_optimizer(
    contrast_model,
    OptimizerConfig(optimizer_type="adam", learning_rate=DPO_LEARNING_RATE, gradient_clip_norm=1.0),
)
contrast_trainer = DPOAlignmentTrainer(
    model=contrast_model,
    optimizer=contrast_optimizer,
    config=DPOAlignmentConfig(
        beta=DPO_BETA,
        reference_free=True,
        simpo_gamma=0.5,  # target margin, already beta-scaled (TRL default 0.5)
        num_log_prob_samples=NUM_MC_SAMPLES,
    ),
)

for step in range(NUM_CONTRAST_STEPS):
    perm_key, loss_key = jax.random.split(jax.random.key(10_000 + step))
    take = jax.random.permutation(perm_key, num_pairs)[:MINIBATCH_PAIRS]
    minibatch = {name: array[take] for name, array in dpo_data.items()}
    contrast_metrics = contrast_trainer.train_step(minibatch, loss_key)
    if step % 10 == 0 or step == NUM_CONTRAST_STEPS - 1:
        print(
            f"Reference-free step {step:2d}: "
            f"loss={contrast_metrics.dpo_loss:.4f}  "
            f"win-rate={contrast_metrics.reward_accuracy:.2%}  "
            f"margin={contrast_metrics.reward_margin:.4f}"
        )
print(f"\nStandard-DPO final margin (reference-anchored): {dpo_margins[-1]:.4f}")
print(f"Reference-free final margin (policy-only):      {contrast_metrics.reward_margin:.4f}")
# Expected: the reference-free loss also decreases, but its margin is on a
# different (policy-only) scale than the reference-anchored one above.

# %% [markdown]
"""
## 6. ScenarioSteeringConfig

``ScenarioSteeringConfig`` adds these parameters on top of DPO:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_scenario` | (required) | Scenario type to steer toward |
| `strategy` | `ranked_dpo` | Steering strategy |
| `steering_strength` | 0.3 | Selection pressure α ∈ [0.0, 1.0] |
| `num_steering_steps` | 1 | Gradient steps per ``ScenarioMiner.steer()`` |
| `num_candidates` | 8 | Candidates sampled per scene context |
| `feasibility_threshold` | 0.0 | Max off-road fraction for "chosen" |

Chosen/rejected pools span the top/bottom ``ceil((1 − α) · num_feasible)``
of the reward ranking, so higher α widens the reward gap between pairs.
``steering_strength=0.0`` pairs candidates randomly — plain DPO, no
steering pressure.
"""

# %%
# Demonstrate validation
steering_config = ScenarioSteeringConfig(
    target_scenario="forward",
    steering_strength=0.3,
    num_steering_steps=1,
)
print(f"Target scenario:    {steering_config.target_scenario}")
print(f"Steering strength:  {steering_config.steering_strength}")
print(f"Steering steps:     {steering_config.num_steering_steps}")
# Expected output:
# Target scenario:    forward
# Steering strength:  0.3
# Steering steps:     1

# Demonstrate that frozen config raises on mutation
try:
    steering_config.steering_strength = 0.9  # type: ignore[misc]
except AttributeError as exc:
    print(f"\nMutation blocked: {exc}")
# Expected output:
# Mutation blocked: cannot assign to field 'steering_strength'

# Demonstrate validation
try:
    ScenarioSteeringConfig(target_scenario="forward", steering_strength=1.5)
except ValueError as exc:
    print(f"Invalid strength: {exc}")
# Expected output:
# Invalid strength: steering_strength must be in [0.0, 1.0]; got 1.5

# %% [markdown]
"""
## 7. compute_scenario_reward

``compute_scenario_reward`` scores trajectories against the kinematic
signature of the target scenario.

Reward rules:

- **forward**: Rewards high forward (x) displacement normalised by expected
  progress at ego speed.
- **lane_change**: Rewards lateral (y) displacement relative to a lane width
  of 3.5 m.
- **other**: Rewards trajectory finiteness and boundedness (1.0 if valid,
  −1.0 otherwise).
"""

# %%
t = jnp.arange(FUTURE_STEPS) * 0.1

# Forward trajectory (high x-velocity)
forward_traj = jnp.stack(
    [t * 10.0, jnp.zeros(FUTURE_STEPS), jnp.zeros(FUTURE_STEPS), jnp.full(FUTURE_STEPS, 10.0)],
    axis=-1,
)[None].repeat(MAX_AGENTS, axis=0)

# Lane-change trajectory (lateral drift)
lane_change_traj = jnp.stack(
    [t * 5.0, t * 4.0, jnp.zeros(FUTURE_STEPS), jnp.full(FUTURE_STEPS, 6.0)],
    axis=-1,
)[None].repeat(MAX_AGENTS, axis=0)

r_forward_on_forward = compute_scenario_reward(forward_traj, "forward", reference_speed=10.0)
r_lateral_on_forward = compute_scenario_reward(lane_change_traj, "forward", reference_speed=10.0)
r_forward_on_lane = compute_scenario_reward(forward_traj, "lane_change", reference_speed=10.0)
r_lateral_on_lane = compute_scenario_reward(lane_change_traj, "lane_change", reference_speed=10.0)

print("Scenario reward matrix:")
print(f"  forward traj   → 'forward'     reward: {float(r_forward_on_forward):+.4f}")
print(f"  forward traj   → 'lane_change' reward: {float(r_forward_on_lane):+.4f}")
print(f"  lane_change traj → 'forward'     reward: {float(r_lateral_on_forward):+.4f}")
print(f"  lane_change traj → 'lane_change' reward: {float(r_lateral_on_lane):+.4f}")
all_rewards = [r_forward_on_forward, r_lateral_on_forward, r_forward_on_lane, r_lateral_on_lane]
in_range = all(-1.0 <= float(r) <= 1.0 for r in all_rewards)
print(f"\nAll rewards in [-1, 1]: {in_range}")
# Expected output:
# Scenario reward matrix:
#   forward traj   → 'forward'     reward: positive
#   forward traj   → 'lane_change' reward: negative (no lateral displacement)
#   lane_change traj → 'forward'     reward: positive (still some x-progress)
#   lane_change traj → 'lane_change' reward: positive (strong lateral)
# All rewards in [-1, 1]: True

# %% [markdown]
"""
## 8. steer_step() — Ranked-Pair DPO Training

``ScenarioSteeringTrainer.steer_step()`` samples candidates from the current
policy for each probe scene context, ranks them by scenario + safety reward,
and runs one standard DPO update on the resulting chosen/rejected pairs.
Steering starts from a fresh copy of the SFT weights so its effect is
isolated from the Section 3 run, and the probe contexts are real WOD scene
encodings.

The steering pairs come from ``model.sample()`` outputs in raw metre space,
where the log-prob proxy differences are orders of magnitude larger than
for the normalized-space WOD pairs — hence the default (small) ``beta``
for this trainer, and reference-free mode because the pairs are
self-generated rather than anchored to demonstrations.
"""

# %%
steer_model = create_reference_model(reference)  # fresh SFT copy for the steering demo
steer_optimizer = create_optimizer(
    steer_model,
    OptimizerConfig(optimizer_type="adam", learning_rate=DPO_LEARNING_RATE, gradient_clip_norm=1.0),
)
steer_dpo_trainer = DPOAlignmentTrainer(
    model=steer_model,
    optimizer=steer_optimizer,
    config=DPOAlignmentConfig(reference_free=True, num_log_prob_samples=NUM_MC_SAMPLES),
)

# Steering config: target "forward" with 30% steering weight
steer_config = ScenarioSteeringConfig(
    target_scenario="forward",
    steering_strength=0.3,
    num_candidates=4,  # small candidate pool keeps the demo fast
)
steer_trainer = ScenarioSteeringTrainer(dpo_trainer=steer_dpo_trainer, config=steer_config)

probe_contexts = jnp.stack(contexts_per_scene[:2])  # (2, agents, context_dim) real scenes

for step in range(NUM_STEER_STEPS):
    step_key = jax.random.key(step + 100)
    steer_metrics = steer_trainer.steer_step(probe_contexts, step_key, num_agents=MAX_AGENTS)
    if step % 3 == 0 or step == NUM_STEER_STEPS - 1:
        print(
            f"Steer step {step:2d}: "
            f"loss={steer_metrics.dpo_loss:.4f}  "
            f"win-rate={steer_metrics.reward_accuracy:.2%}  "
            f"margin={steer_metrics.reward_margin:.4f}"
        )
# Expected: loss stays finite and the reward-ranked pairs provide a
# steering gradient; per-step values vary with the sampled candidates.

# %% [markdown]
"""
## 9. Steered vs. Unsteered Trajectory Comparison

We compare trajectory samples from the DPO-aligned model (Section 3) and
the steered model on the same real WOD scene context. A well-steered model
should produce higher forward displacement.
"""

# %%
context_arr = contexts_per_scene[0]  # real WOD scene encoding

dpo_pred = model.sample(context_arr, key=jax.random.key(99))
steer_pred = steer_model.sample(context_arr, key=jax.random.key(99))

dpo_x_disp = float(jnp.mean(dpo_pred.trajectories[..., -1, 0] - dpo_pred.trajectories[..., 0, 0]))
steer_x_disp = float(
    jnp.mean(steer_pred.trajectories[..., -1, 0] - steer_pred.trajectories[..., 0, 0])
)

print(f"DPO-aligned mean x-displacement: {dpo_x_disp:.4f}")
print(f"Steered     mean x-displacement: {steer_x_disp:.4f}")
print(f"\nSteered model trajectories shape: {steer_pred.trajectories.shape}")
# Expected: steered x-displacement exceeds the unsteered one on average;
# trajectories shape (8, 80, 4).

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for agent_idx in range(MAX_AGENTS):
    axes[0].plot(
        dpo_pred.trajectories[agent_idx, :, 0],
        dpo_pred.trajectories[agent_idx, :, 1],
        label=f"Agent {agent_idx}",
        linewidth=2,
    )
    axes[1].plot(
        steer_pred.trajectories[agent_idx, :, 0],
        steer_pred.trajectories[agent_idx, :, 1],
        label=f"Agent {agent_idx}",
        linewidth=2,
    )

axes[0].set_title("DPO-Aligned Trajectories")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Steered Trajectories (forward, α=0.3)")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Y (m)")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "steered_trajectory_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"Trajectory comparison saved to {PLOT_DIR / 'steered_trajectory_comparison.png'}")
# Expected output:
# Trajectory comparison saved to <output dir>/steered_trajectory_comparison.png

# %% [markdown]
"""
## 10. ScenarioMiner.steer() — High-Level SDK

``ScenarioMiner.steer()`` wraps the full steering pipeline into a single
method call. It:

1. Generates seed scenarios from the base ``scenario_type``
2. Samples and reward-ranks candidate trajectories per probe context
3. Runs ``ScenarioSteeringTrainer`` for ``num_steering_steps`` gradient steps
   on a cloned copy of the miner's model
4. Generates ``count`` scenarios using the steered weights

This is the recommended entry point for production use. The miner here
carries the Section 3 DPO-aligned weights, so this call steers the aligned
policy.
"""

# %%
steering_cfg = ScenarioSteeringConfig(
    target_scenario="forward",
    steering_strength=0.3,
    num_steering_steps=2,
    num_candidates=4,
)
steered_scenarios = miner.steer(
    scenario_type="forward",
    density="low",
    steering_config=steering_cfg,
    count=3,
    key=jax.random.key(0),
)
print(f"Generated {len(steered_scenarios)} steered scenarios")
for i, s in enumerate(steered_scenarios):
    traj_shape = s.predictions.trajectories.shape
    tags = s.metadata.tags
    print(f"  Scenario {i}: trajectories={traj_shape}, tags={tags}")
# Expected output:
# Generated 3 steered scenarios
#   Scenario 0: trajectories=(2, 80, 4), tags=('forward', 'low', 'steered_forward')
#   Scenario 1: trajectories=(2, 80, 4), tags=('forward', 'low', 'steered_forward')
#   Scenario 2: trajectories=(2, 80, 4), tags=('forward', 'low', 'steered_forward')

# %% [markdown]
"""
## Next Steps

### Experiments to Try

1. Raise `NUM_SCENES` and `NUM_DPO_STEPS` for a longer alignment run
2. Add `label_smoothing=0.1` for robustness to preference-label noise
3. Combine `physics_weight=0.5` in `DPOAlignmentConfig` for
   physics-regularised alignment
4. Set `steering_strength=0.0` to confirm steering reduces to plain DPO
5. Try `target_scenario="lane_change"` and observe lateral displacement

### Related Examples

- [Preference Construction Quick Reference](01_preference_construction_quickref.py)
- [DPO Fine-Tuning Quick Reference](02_dpo_finetuning_quickref.py)
- [ScenarioMiner Guide](../advanced/02_scenario_miner_guide.py)

### API Reference

- [scenario_steering](../../api/alignment/scenario_steering.md)
- [dpo_trainer](../../api/alignment/dpo_trainer.md)
- [ScenarioMiner](../../api/api/scenario_miner.md)
"""
