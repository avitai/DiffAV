# ScenarioMiner Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~30 min (CPU), ~5 min (GPU) |
| **Prerequisites** | End-to-End Quick Reference, physics losses |
| **Format** | Python + Jupyter |

A complete guide to the `ScenarioMiner` SDK: configuration options,
multi-type scenario synthesis, planner evaluation strategy, and gradient-based
adversarial failure-case search with physics-informed scoring.

## Files

- **Python Script**: [`examples/advanced/02_scenario_miner_guide.py`](https://github.com/avitai/DiffAV/blob/main/examples/advanced/02_scenario_miner_guide.py)
- **Jupyter Notebook**: [`examples/advanced/02_scenario_miner_guide.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/advanced/02_scenario_miner_guide.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Run: `python examples/advanced/02_scenario_miner_guide.py` — no dataset required

## What You'll Learn

1. All `MinerConfig` fields and when to set each one
2. How to generate scenarios across all four types and three density levels
3. How to interpret `MetricsReport` ADE/FDE values from `evaluate_planner()`
4. How gradient-based adversarial search works internally
5. How to analyse and compare `FailureCase` severity distributions
6. How to plug in a custom planner for benchmark evaluation

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

## Scenario Types

| Label | Description |
|-------|-------------|
| `"forward"` | Straight-ahead convoy with lead vehicles |
| `"lane_change"` | Lateral agent cuts across ego path |
| `"unprotected_left_turn"` | Cross-traffic intersection |
| `"adversarial"` | Gradient-perturbed high-violation template |

## Density Labels

| Label | Agents |
|-------|--------|
| `"low"` | 2 |
| `"medium"` | 8 |
| `"high"` | 16 |

All counts are capped at `MinerConfig.max_agents`.

## Coming from Other Tools?

| Tool | DiffAV ScenarioMiner |
|------|--------------------------|
| Static scenario databases | Physics-informed synthesis from diffusion model |
| Hand-crafted edge cases | Gradient-based adversarial search |
| Per-vehicle metric scripts | `evaluate_planner()` with ADE/FDE in 2 lines |
| PyTorch-only frameworks | Pure JAX: `jax.jit`, `jax.grad`, `jax.vmap` throughout |

## Sister Repo Components

| Component | Source | Role |
|-----------|--------|------|
| `create_optimizer`, `OptimizerConfig` | `opifex.core.training.optimizers` | Adam for gradient ascent |
| `nan_safe_gradients` | `diffav.core.training_utils` | NaN-safe gradient filtering |
| `ade`, `fde` | `diffav.evaluation.metrics` | Per-agent displacement errors |
| `DiffAVPhysicsLoss` | `diffav.physics.losses` | Physics violation scoring |

## API References

- [`ScenarioMiner`, `create_scenario_miner`](../../api/api/scenario_miner.md)
- [`MinerConfig`, `Scenario`, `FailureCase`](../../api/api/config.md)
- [ScenarioMiner User Guide](../../user-guide/sdk.md)
- [Physics Losses](../../api/physics/losses.md)
- [Evaluation Metrics](../../api/evaluation/metrics.md)
