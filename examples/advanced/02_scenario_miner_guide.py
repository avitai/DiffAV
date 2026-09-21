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
# ScenarioMiner Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~30 min (CPU), ~5 min (GPU) |
| **Prerequisites** | End-to-End Quick Reference, physics losses |
| **Format** | Python + Jupyter |

## What You'll Learn

1. Configure `MinerConfig` and create a miner
2. Generate scenarios across types and densities
3. Evaluate planners and interpret ADE/FDE
4. Run and tune gradient-based adversarial search

## Overview

A deep dive into the `ScenarioMiner` SDK covering configuration, scenario
synthesis, planner evaluation, and gradient-based adversarial search with
physics-informed failure scoring.

## Architecture

```
MinerConfig ─── max_agents, prediction_horizon, context_dim
     │           model_path (optional checkpoint)
     ▼
create_scenario_miner()
     │
     ▼
ScenarioMiner (frozen dataclass)
  ├── model: TrajectoryDiffusionModel  ← FactorizedSceneBackbone
  └── config: MinerConfig

     │
     ├── generate(scenario_type, density, count)
     │     → list[Scenario]  ← SceneContext + TrajectoryPrediction + metadata
     │
     ├── evaluate_planner(planner_fn, scenarios)
     │     → MetricsReport  ← ADE + FDE averaged over agents and scenarios
     │
     └── adversarial_search(planner_fn, budget, severity_threshold, ...)
           → list[FailureCase]  ← at most budget, sorted by severity descending
```

## Sister Repo Components Used

| Component | Source | Role |
|-----------|--------|------|
| `optax.adam` | optax | Adam ascent over the scenario perturbation |
| `nan_safe_gradients` | `diffav.core.training_utils` | NaN-safe gradient filtering |
| `ade`, `fde` | `diffav.evaluation.metrics` | Per-agent displacement errors |
| `DiffAVPhysicsLoss` | `diffav.physics.losses` | Physics violation scoring |

## Coming from Other Tools?

| Tool | DiffAV ScenarioMiner |
|------|--------------------------|
| Static scenario databases | Physics-informed synthesis from diffusion model |
| Hand-crafted edge cases | Gradient-based adversarial search |
| Per-vehicle metric scripts | `evaluate_planner()` with ADE/FDE in 2 lines |
| PyTorch-only frameworks | Pure JAX: `jax.jit`, `jax.grad`, `jax.vmap` throughout |
"""

# %% [markdown]
"""
## Step 1: Configuration and Factory
"""

# %%
import os
from pathlib import Path

import jax.numpy as jnp

from diffav.api import create_scenario_miner, MinerConfig
from diffav.core.types import TrajectoryPrediction


# Full config with all fields documented. When the WOD-trained checkpoint
# exists (see `scripts/train_wod.py`), load it; the fields shown are the
# fresh-model fallback so this guide stays runnable without training.
CHECKPOINT_DIR = Path("checkpoints/wod-mini")
USE_CHECKPOINT = CHECKPOINT_DIR.exists()
if USE_CHECKPOINT:
    config = MinerConfig(
        model_path=str(CHECKPOINT_DIR),
        max_agents=8,
        prediction_horizon=80,
        context_dim=128,
        num_diffusion_steps=100,
        hidden_dim=256,
        num_blocks=6,
        num_heads=8,
    )
else:
    config = MinerConfig(
        model_path="",  # empty = fresh model; set to checkpoint path for production
        batch_size=32,  # scenarios sampled per vmapped generation chunk
        max_agents=8,  # cap on agents per scene
        prediction_horizon=20,  # future timesteps to predict
        context_dim=64,  # scene embedding dimension
        num_diffusion_steps=10,  # diffusion timesteps (small for the fallback demo)
    )
miner = create_scenario_miner(config)
print("Model type:", type(miner.model).__name__)
print("Config:", config)
# Expected:
# Model type: TrajectoryDiffusionModel
# Config: MinerConfig(model_path='', batch_size=32, ...)

# %% [markdown]
"""
With a trained checkpoint, pass `model_path`:

```python
config = MinerConfig(
    model_path="checkpoints/wod-mini",
    ...
)
miner = create_scenario_miner(config)
```

`create_scenario_miner` raises `FileNotFoundError` immediately if the path
does not exist — no silent fallback.
"""

# %% [markdown]
"""
## Step 2: Generating Diverse Scenario Types

Four built-in templates cover the main autonomous driving challenges.
"""

# %%
scenario_types = ["forward", "lane_change", "unprotected_left_turn", "adversarial"]
densities = ["low", "medium", "high"]

print("Scenario type × density agent counts (capped at max_agents=8):\n")
print(f"{'Type':<30} {'low':>5} {'medium':>7} {'high':>6}")
print("-" * 52)
# Agent count depends only on density, so sample each density once and
# reuse the counts across types — 3 reverse-diffusion runs instead of 12.
# Smoke mode (set by the example execution tests) shrinks sampling counts;
# real runs keep the showcase scale.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
density_counts = [
    len(miner.generate("forward", density=d, count=1)[0].context.agent_states) for d in densities
]
for stype in scenario_types:
    row = density_counts
    print(f"{stype:<30} {row[0]:>5} {row[1]:>7} {row[2]:>6}")
# Expected:
# Scenario type × density agent counts (capped at max_agents=8):
#
# Type                            low  medium    high
# ----------------------------------------------------
# forward                           2       8       8
# lane_change                       2       8       8
# unprotected_left_turn             2       8       8
# adversarial                       2       8       8

# %%
# Inspect a SceneContext in detail
scene = miner.generate("unprotected_left_turn", density="medium", count=1)[0]
ctx = scene.context

print("SceneContext fields:")
print(f"  ego_state.position   : {ctx.ego_state.position}")
print(f"  ego_state.velocity   : {ctx.ego_state.velocity}")
print(f"  agent_states count   : {len(ctx.agent_states)}")
print(f"  agent_states[0].pos  : {ctx.agent_states[0].position}")
print(f"  timestamps shape     : {ctx.timestamps.shape}")
print(f"\nPredictions shape    : {scene.predictions.trajectories.shape}")
print("  state dims           : [x, y, heading, velocity]")
# Expected:
# SceneContext fields:
#   ego_state.position   : [0. 0.]
#   ego_state.velocity   : 10.0
#   agent_states count   : 8
#   agent_states[0].pos  : [8. 0.]
#   timestamps shape     : (11,)
#
# Predictions shape    : (8, 80, 4)
#   state dims           : [x, y, heading, velocity]

# %% [markdown]
"""
## Step 3: Evaluating a Realistic Planner

`evaluate_planner()` maps your planner over all scenarios and computes
per-agent ADE and FDE using `jax.vmap`. The result is a `MetricsReport`
with values averaged over all scenarios and agents.

Here we compare a zero-velocity planner against a slightly better
constant-forward-velocity planner to show that evaluation is sensitive
to planner quality.
"""


# %%
def zero_planner(context):
    """All agents frozen at origin."""
    n = len(context.agent_states)
    return TrajectoryPrediction(
        trajectories=jnp.zeros((n, config.prediction_horizon, 4)),
        agent_ids=tuple(f"agent_{i}" for i in range(n)),
    )


def forward_planner(context):
    """Constant forward velocity (ego speed 10 m/s) for all agents."""
    n = len(context.agent_states)
    # x increases linearly; y, heading, velocity constant
    t = jnp.linspace(0.0, config.prediction_horizon * 0.1, config.prediction_horizon)
    x = 10.0 * t
    trajs = jnp.stack(
        [x, jnp.zeros_like(x), jnp.zeros_like(x), jnp.full_like(x, 10.0)], axis=-1
    )  # (T, 4)
    trajs = jnp.broadcast_to(trajs[None], (n, config.prediction_horizon, 4))
    return TrajectoryPrediction(
        trajectories=trajs,
        agent_ids=tuple(f"agent_{i}" for i in range(n)),
    )


scenarios = miner.generate("unprotected_left_turn", density="medium", count=2 if _SMOKE else 10)

zero_report = miner.evaluate_planner(zero_planner, scenarios)
fwd_report = miner.evaluate_planner(forward_planner, scenarios)

print(
    f"Zero-velocity planner — ADE: {zero_report.metric_values['ade']:.4f}  "
    f"FDE: {zero_report.metric_values['fde']:.4f}"
)
print(
    f"Forward-velocity planner — ADE: {fwd_report.metric_values['ade']:.4f}  "
    f"FDE: {fwd_report.metric_values['fde']:.4f}"
)
# Expected (forward planner will generally have lower ADE/FDE for forward scenarios):
# Zero-velocity planner — ADE: X.XXXX  FDE: X.XXXX
# Forward-velocity planner — ADE: X.XXXX  FDE: X.XXXX

# %% [markdown]
"""
## Step 4: Adversarial Search Deep Dive

`adversarial_search()` uses Adam gradient ascent on the scene context
embedding to find perturbations that maximise `DiffAVPhysicsLoss`
in the model's predictions. The perturbed scenes are then passed to
your planner and the physics loss scores the failure severity.

Key parameters:
- `budget`: candidate scenes explored — at most this many failure cases return
- `severity_threshold`: exclusive minimum planner severity for a candidate to
  count as a failure (default 0.0); healthy planners may yield zero cases
- `perturbation_steps`: Adam steps per candidate (more = stronger adversarial)
- `step_size`: Adam learning rate for perturbation
- `key`: JAX random key driving candidate seeds and perturbations; the seed
  scenarios are reproducible from its first split

Internally uses the `nnx.split`/`nnx.merge` pattern from opifex's
`NNXHybridOptimizer` to correctly handle NNX model state inside
`jax.value_and_grad`.
"""

# %%
# Compare budgets — all results sorted by severity descending
for budget in [1] if _SMOKE else [2, 4]:
    cases = miner.adversarial_search(
        zero_planner,
        budget=budget,
        perturbation_steps=2 if _SMOKE else 5,
    )
    print(f"\nbudget={budget}: found {len(cases)} failure cases")
    for i, fc in enumerate(cases):
        print(f"  [{i}] mode={fc.failure_mode:25s}  severity={fc.severity:.4f}")
# Expected (sorted descending by severity, up to budget cases each):
# budget=2: found 2 failure cases
#   [0] mode=...  severity=X.XXXX
#   [1] mode=...  severity=X.XXXX
#
# budget=4: found 4 failure cases
#   [0] mode=...  severity=X.XXXX
#   ...

# %%
# Inspect a failure case scenario
cases = miner.adversarial_search(
    zero_planner, budget=1 if _SMOKE else 3, perturbation_steps=2 if _SMOKE else 5
)
fc = cases[0]  # highest severity

print("FailureCase fields:")
print(f"  failure_mode : {fc.failure_mode}")
print(f"  severity     : {fc.severity:.4f}")
print(f"  scenario_id  : {fc.scenario.metadata.scenario_id}")
print(f"  source       : {fc.scenario.metadata.source_dataset}")
print(f"  tags         : {fc.scenario.metadata.tags}")
print(f"\n  Adversarial context agents: {len(fc.scenario.context.agent_states)}")
print(f"  Adversarial predictions:    {fc.scenario.predictions.trajectories.shape}")
# Expected:
# FailureCase fields:
#   failure_mode : kinematics_violation (or collision_risk)
#   severity     : X.XXXX
#   scenario_id  : adv_adversarial_000000
#   source       : diffav_adversarial
#   tags         : ('adversarial',)

# %%
# Verify severity ordering is always descending
severities = [fc.severity for fc in cases]
assert severities == sorted(severities, reverse=True), "Should be sorted descending"
print("Severity ordering verified:", [f"{s:.4f}" for s in severities])
# Expected:
# Severity ordering verified: ['X.XXXX', 'X.XXXX', 'X.XXXX']

# %% [markdown]
"""
## Next Steps and Experiments

**Experiments to try:**

1. **Production-quality model**: Set `num_diffusion_steps=100` and provide
   a trained `model_path` checkpoint.

2. **Stronger adversarial search**: Increase `perturbation_steps` to 50-100
   for deeper exploration of the failure landscape.

3. **Custom planner**: Replace `zero_planner` with your real planning system.
   The planner does not need to be JAX-differentiable.

4. **High-density scenes**: Use `density="high"` with `max_agents=16` for
   complex multi-agent scenarios.

5. **Batch evaluation**: Generate thousands of scenarios and use
   `evaluate_planner()` to build an extensive benchmark.

See the [ScenarioMiner SDK user guide](../../docs/user-guide/sdk.md) for
configuration details and API reference.
"""
