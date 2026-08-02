# ScenarioMiner SDK

`ScenarioMiner` is the primary entry point for the Simulacrax public API. It wraps
a `TrajectoryDiffusionModel` and exposes three operations: scenario generation,
planner evaluation, and adversarial failure-case search.

## Quick Usage

```python
from simulacrax.api import MinerConfig, create_scenario_miner

config = MinerConfig(model_path="checkpoints/wod-mini")
miner = create_scenario_miner(config)

# 1. Generate scenarios
scenarios = miner.generate("unprotected_left_turn", density="high", count=1000)

# 2. Evaluate an external planner
report = miner.evaluate_planner(my_planner_fn, scenarios)
print(report.metric_values)  # {"ade": 1.2, "fde": 2.4}

# 3. Find adversarial failure cases
cases = miner.adversarial_search(my_planner_fn, budget=50)
```

The boundary fails fast: an unknown `scenario_type` or `density`, or a
negative `count`/`budget`, raises `ValueError` naming the valid choices —
nothing silently maps to a default. The vocabularies are the
`ScenarioType` and `Density` enums in `simulacrax.core.types` (plain
strings with those values are accepted too).

## Pipeline

```
MinerConfig
    ↓
create_scenario_miner()  ← loads TrajectoryDiffusionModel from checkpoint
    ↓
ScenarioMiner.generate(scenario_type, density, count)
    → list[Scenario]  (SceneContext + reference TrajectoryPrediction + metadata)
    scenario_type ∈ ScenarioType (forward | lane_change | unprotected_left_turn | adversarial)
    density       ∈ Density      (low: 2 | medium: 8 | high: 16 agents)
    ↓
ScenarioMiner.evaluate_planner(planner_fn, scenarios)
    → MetricsReport   (ADE, FDE averaged over scenarios and agents)
    ↓
ScenarioMiner.adversarial_search(planner_fn, budget)
    → list[FailureCase]  at most `budget`, sorted by severity (highest first)
```

## Scenario Types

| Label | Description |
|-------|-------------|
| `"forward"` | Straight-ahead convoy with lead vehicles |
| `"lane_change"` | Lateral agent cuts across ego path |
| `"unprotected_left_turn"` | Cross-traffic intersection |
| `"adversarial"` | Internally generated high-violation template |

## Density Labels

| Label | Agents |
|-------|--------|
| `"low"` | 2 |
| `"medium"` | 8 |
| `"high"` | 16 |

Values are capped at `MinerConfig.max_agents`.

Generation samples the diffusion model in `jax.vmap` chunks of
`MinerConfig.batch_size`; each scenario draws its own subkey, so results are
independent of the chunk layout.

## Planner Interface

`planner_fn` must accept a `SceneContext` and return a `TrajectoryPrediction`:

```python
from simulacrax.core.types import SceneContext, TrajectoryPrediction
import jax.numpy as jnp

def my_planner(context: SceneContext) -> TrajectoryPrediction:
    n = len(context.agent_states)
    return TrajectoryPrediction(
        trajectories=jnp.zeros((n, 80, 4)),  # (agents, steps, [x, y, heading, vel])
        agent_ids=tuple(f"agent_{i}" for i in range(n)),
    )
```

`planner_fn` does **not** need to be JAX-differentiable — only the internal
diffusion model is differentiated in `adversarial_search`.

## Adversarial Search

`adversarial_search()` explores exactly `budget` candidate scenes. For each
candidate, Adam gradient ascent on the scene context embedding finds
perturbations that maximise physics violations in the internal model's
trajectory predictions. The perturbed scene is then passed to the external
planner and scored by `SimulacraxPhysicsLoss`; it becomes a `FailureCase`
only when that severity exceeds `severity_threshold`. A healthy planner
therefore yields fewer than `budget` cases — possibly none.

```python
import jax

cases = miner.adversarial_search(
    my_planner_fn,
    budget=50,
    severity_threshold=0.0,  # exclusive minimum severity to count as a failure
    perturbation_steps=20,   # gradient-ascent steps per candidate
    step_size=0.05,          # Adam learning rate
    key=jax.random.key(0),   # optional; drives candidate seeds + perturbations
)

for fc in cases:
    print(fc.failure_mode, fc.severity)
```

Results are sorted by `severity` descending so the most challenging failure
cases appear first. Each `FailureCase.scenario` carries reference predictions
regenerated on the *perturbed* context, and the unperturbed seed scenarios are
reproducible: the candidate key is the first split of `key`, so
`miner.generate("adversarial", "medium", count=budget, key=jax.random.split(key)[0])`
recreates them.

## API Reference

- [`ScenarioMiner`, `create_scenario_miner`](../api/api/scenario_miner.md)
- [`MinerConfig`, `Scenario`, `FailureCase`](../api/api/config.md)
