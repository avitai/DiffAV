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
# End-to-End Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Core |
| **Runtime** | ~2 min (CPU) |
| **Prerequisites** | DiffAV installed (`uv sync`) |
| **Format** | Python + Jupyter |

The "hello world" of DiffAV: generate synthetic driving scenarios,
evaluate a planner with ADE/FDE metrics, and search for adversarial
failure cases — all in under 15 lines of Python.

## Overview

```
MinerConfig
    ↓
create_scenario_miner()
    ↓
generate()  →  list[Scenario]
    ↓
evaluate_planner()  →  MetricsReport
    ↓
adversarial_search()  →  list[FailureCase]
```

## What You'll Learn

1. Create a `ScenarioMiner` from a `MinerConfig`
2. Generate synthetic scenarios with `generate()`
3. Score a planner with `evaluate_planner()`
4. Mine failure cases with `adversarial_search()`
"""

# %%
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from substrax.artifacts import resolve_output_dir


PLOT_DIR = resolve_output_dir("examples").path


from diffav.api import create_scenario_miner, MinerConfig
from diffav.core.types import TrajectoryPrediction


# Load the WOD-trained checkpoint when available (produce one with
# `python scripts/train_wod.py`); otherwise fall back to a small model that
# a brief warm-up below keeps presentable. Production runs always use a
# trained checkpoint.
CHECKPOINT_DIR = Path("checkpoints/wod-mini")
USE_CHECKPOINT = CHECKPOINT_DIR.exists()
if USE_CHECKPOINT:
    config = MinerConfig(
        model_path=str(CHECKPOINT_DIR),
        max_agents=8,
        num_diffusion_steps=100,
        hidden_dim=256,
        num_blocks=6,
        num_heads=8,
    )
else:
    config = MinerConfig(
        max_agents=8,
        prediction_horizon=20,
        context_dim=64,
        num_diffusion_steps=10,
    )
miner = create_scenario_miner(config)
print("ScenarioMiner ready:", "trained checkpoint" if USE_CHECKPOINT else "fresh model")
# Expected:
# ScenarioMiner ready: trained checkpoint

# %% [markdown]
"""
## 1. Warm-Up Training

A freshly initialised diffusion model samples pure noise — a poor showcase.
A short warm-up on synthetic constant-velocity convoy trajectories (the
same family the scene templates use) pulls samples onto the data scale in
seconds. Production runs skip this and load a trained checkpoint via
`MinerConfig(model_path=...)`.
"""

# %%
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer


def _convoy_batch(key, num_agents, horizon):
    """Synthetic constant-velocity convoy trajectories + matching context.

    Starting positions and speeds span the same ranges the scene templates
    use, so the conditioning the model learns here transfers to
    ``generate()``'s contexts.
    """
    speed_key, x0_key, y0_key = jax.random.split(key, 3)
    speeds = jax.random.uniform(speed_key, (num_agents, 1), minval=5.0, maxval=15.0)
    x_starts = jax.random.uniform(x0_key, (num_agents, 1), minval=0.0, maxval=60.0)
    y_starts = jax.random.uniform(y0_key, (num_agents, 1), minval=-2.0, maxval=10.0)
    t = jnp.arange(horizon)[None, :] * 0.1
    x = x_starts + speeds * t
    y = jnp.broadcast_to(y_starts, (num_agents, horizon))
    heading = jnp.zeros_like(x)
    velocity = jnp.broadcast_to(speeds, (num_agents, horizon))
    trajectories = jnp.stack([x, y, heading, velocity], axis=-1)
    # Context rows mirror generate()'s encoding: [x0, y0, vx, vy, 0, ...]
    rows = jnp.concatenate([x_starts, y_starts, speeds, jnp.zeros_like(speeds)], axis=-1)
    context = jnp.pad(rows, ((0, 0), (0, config.context_dim - rows.shape[1])))
    return trajectories, context


if USE_CHECKPOINT:
    print("Trained checkpoint loaded — warm-up not needed.")
else:
    warmup_trainer = TrajectoryTrainer(
        miner.model, TrainerConfig(num_epochs=1, log_interval=10_000)
    )
    for step in range(1500):
        step_key = jax.random.fold_in(jax.random.key(7), step)
        trajectories, context = _convoy_batch(
            step_key, num_agents=4, horizon=config.prediction_horizon
        )
        metrics = warmup_trainer.train_step(trajectories, context, key=step_key)
    print(f"Warm-up complete — final diffusion loss: {metrics.total_loss:.3f}")
# Expected:
# Warm-up complete — final diffusion loss: <about 0.1-1.0>

# %% [markdown]
"""
## 2. Generate Scenarios

`generate()` synthesises a batch of driving scenes from a named template.
Each `Scenario` contains a `SceneContext` (ego state, agent states, map
features) and a reference `TrajectoryPrediction` from the diffusion model.
"""

# %%
# Smoke mode (set by the example execution tests) shrinks sampling counts.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
scenarios = miner.generate("unprotected_left_turn", density="medium", count=2 if _SMOKE else 5)
s = scenarios[0]

print(f"Generated: {len(scenarios)} scenarios")
print(f"  Agents per scene : {len(s.context.agent_states)}")
print(f"  Prediction shape : {s.predictions.trajectories.shape}")
print(f"  Scenario ID      : {s.metadata.scenario_id}")
print(f"  Tags             : {s.metadata.tags}")
# Expected:
# Generated: 5 scenarios
#   Agents per scene : 8
#   Prediction shape : (8, 80, 4)
#   Scenario ID      : unprotected_left_turn_000000
#   Tags             : (<ScenarioType.UNPROTECTED_LEFT_TURN: ...>, <Density.MEDIUM: ...>)

# %%
# Scenes span ~150 m in x but only a few metres in y: equal-aspect axes
# would collapse into thin strips, so panels are stacked with a shared
# x-axis and an expanded y range that keeps lane-level detail readable.
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, layout="constrained")
scenario_title = "Unprotected left turn"
for panel, (ax, scenario) in enumerate(zip(axes, scenarios[:3]), start=1):
    trajs = scenario.predictions.trajectories
    for agent in range(trajs.shape[0]):
        ax.plot(
            trajs[agent, :, 0],
            trajs[agent, :, 1],
            alpha=0.8,
            label=f"agent {agent}" if panel == 1 else None,
        )
        ax.scatter(trajs[agent, 0, 0], trajs[agent, 0, 1], s=14, marker="o")
    y_mid = float(trajs[..., 1].mean())
    half_span = max(6.0, float(jnp.abs(trajs[..., 1] - y_mid).max()) * 1.2)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_title(f"{scenario_title} — sample {panel}", fontsize=10)
    ax.set_ylabel("y (m)")
axes[-1].set_xlabel("x (m)")
fig.legend(loc="outside right center", fontsize=8, frameon=False)
fig.suptitle("Generated reference trajectories (dots = start)")
fig.savefig(PLOT_DIR / "end_to_end_generated_scenarios.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'end_to_end_generated_scenarios.png'}")

# %% [markdown]
"""
## 3. Evaluate a Planner

`evaluate_planner()` calls your planner on each scenario's `SceneContext`
and computes ADE (Average Displacement Error) and FDE (Final Displacement
Error) against the reference predictions.

`planner_fn` just needs to return a `TrajectoryPrediction` — it does not
need to be JAX-differentiable.
"""


# %%
def constant_velocity_planner(context):
    """Extrapolate each agent's current state at constant velocity."""
    steps = jnp.arange(config.prediction_horizon) * 0.1  # 10 Hz
    rows = []
    for state in context.agent_states:
        vx = state.velocity * jnp.cos(jnp.asarray(state.heading))
        vy = state.velocity * jnp.sin(jnp.asarray(state.heading))
        x = state.position[0] + vx * steps
        y = state.position[1] + vy * steps
        heading = jnp.full_like(steps, state.heading)
        speed = jnp.full_like(steps, state.velocity)
        rows.append(jnp.stack([x, y, heading, speed], axis=-1))
    return TrajectoryPrediction(
        trajectories=jnp.stack(rows),
        agent_ids=tuple(f"agent_{i}" for i in range(len(rows))),
    )


report = miner.evaluate_planner(constant_velocity_planner, scenarios)
print("Metrics:", {k: f"{v:.4f}" for k, v in report.metric_values.items()})
# Expected (non-zero because reference predictions are non-trivial):
# Metrics: {'ade': '10.7635', 'fde': '19.1762'}

# %% [markdown]
"""
## 4. Adversarial Search

`adversarial_search()` explores exactly `budget` candidate scenes, applying
Adam gradient ascent on each scene context embedding to find perturbations
that maximise physics violations in the model's output. The perturbed scenes
are then scored against your planner; only candidates whose severity exceeds
`severity_threshold` (default 0.0) become failure cases, so a healthy planner
can return fewer than `budget` — even zero.

Results are sorted by `severity` descending — worst failure cases first.
"""


# %%
def reckless_planner(context):
    """Swerving high-jerk planner: every agent oscillates hard laterally."""
    steps = jnp.arange(config.prediction_horizon) * 0.1
    rows = []
    for state in context.agent_states:
        x = state.position[0] + state.velocity * steps
        y = state.position[1] + 4.0 * jnp.sin(6.0 * steps)  # violent weaving
        heading = jnp.full_like(steps, state.heading)
        speed = jnp.full_like(steps, state.velocity)
        rows.append(jnp.stack([x, y, heading, speed], axis=-1))
    return TrajectoryPrediction(
        trajectories=jnp.stack(rows),
        agent_ids=tuple(f"agent_{i}" for i in range(len(rows))),
    )


healthy_cases = miner.adversarial_search(
    constant_velocity_planner,
    budget=1 if _SMOKE else 3,
    severity_threshold=0.01,
    perturbation_steps=2 if _SMOKE else 5,
)
reckless_cases = miner.adversarial_search(
    reckless_planner,
    budget=1 if _SMOKE else 3,
    severity_threshold=0.01,
    perturbation_steps=2 if _SMOKE else 5,
)
print(f"Healthy planner failure cases:  {len(healthy_cases)}")
print(f"Reckless planner failure cases: {len(reckless_cases)}")
for fc in reckless_cases:
    print(f"  mode={fc.failure_mode:25s}  severity={fc.severity:.4f}")
# Expected: the smooth constant-velocity planner survives the search while
# the weaving planner racks up kinematic violations:
# Healthy planner failure cases:  0
# Reckless planner failure cases: 3
#   mode=kinematics_violation       severity=0.0281
#   ...

# %% [markdown]
"""
## Next Steps

- Load a trained checkpoint: `MinerConfig(model_path="checkpoints/...")`
- Try different scenario types: `"forward"`, `"lane_change"`, `"adversarial"`
- Increase density for more agents: `density="high"` (up to `max_agents`)
- Tune `perturbation_steps` and `step_size` in `adversarial_search()`
- See the [ScenarioMiner Guide](../advanced/02_scenario_miner_guide.ipynb) for a deep dive
"""
