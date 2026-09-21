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

## Terminal Output

The output below is from a run on a Modal L4 GPU (`modal run deploy/modal_app.py --task examples`), with deterministic GPU kernels and WOD read from the data volume. The image carries no `checkpoints/wod-mini`, so the example ran its untrained fallback, as it does anywhere that checkpoint is absent.

```text
Model type: TrajectoryDiffusionModel
Config: MinerConfig(hidden_dim=128, num_heads=4, num_blocks=2, num_temporal_layers=2, num_social_layers=1, use_social_interaction=True, use_map_cross_attention=False, scene_token_dim=None, gradient_checkpointing=False, model_path='', batch_size=32, max_agents=8, prediction_horizon=20, context_dim=64, num_diffusion_steps=10)
Scenario type × density agent counts (capped at max_agents=8):
Type                             low  medium   high
----------------------------------------------------
forward                            2       8      8
lane_change                        2       8      8
unprotected_left_turn              2       8      8
adversarial                        2       8      8
SceneContext fields:
  ego_state.position   : [0. 0.]
  ego_state.velocity   : 10.0
  agent_states count   : 8
  agent_states[0].pos  : [8. 0.]
  timestamps shape     : (11,)
Predictions shape    : (8, 20, 4)
  state dims           : [x, y, heading, velocity]
Zero-velocity planner — ADE: 157.1409  FDE: 162.7078
Forward-velocity planner — ADE: 153.7227  FDE: 151.9520
budget=2: found 2 failure cases
  [0] mode=collision_risk             severity=0.0400
  [1] mode=collision_risk             severity=0.0400
budget=4: found 4 failure cases
  [0] mode=collision_risk             severity=0.0400
  [1] mode=collision_risk             severity=0.0400
  [2] mode=collision_risk             severity=0.0400
  [3] mode=collision_risk             severity=0.0400
FailureCase fields:
  failure_mode : collision_risk
  severity     : 0.0400
  scenario_id  : adv_adversarial_000000
  source       : diffav_adversarial
  tags         : ('adversarial',)
  Adversarial context agents: 8
  Adversarial predictions:    (8, 20, 4)
Severity ordering verified: ['0.0400', '0.0400', '0.0400']
```

## Sister Repo Components

| Component | Source | Role |
|-----------|--------|------|
| `optax.adam` | optax | Adam for gradient ascent over the scenario perturbation |
| `nan_safe_gradients` | `diffav.core.training_utils` | NaN-safe gradient filtering |
| `ade`, `fde` | `diffav.evaluation.metrics` | Per-agent displacement errors |
| `DiffAVPhysicsLoss` | `diffav.physics.losses` | Physics violation scoring |

## API References

- [`ScenarioMiner`, `create_scenario_miner`](../../api/api/scenario_miner.md)
- [`MinerConfig`, `Scenario`, `FailureCase`](../../api/api/config.md)
- [ScenarioMiner User Guide](../../user-guide/sdk.md)
- [Physics Losses](../../api/physics/losses.md)
- [Evaluation Metrics](../../api/evaluation/metrics.md)
