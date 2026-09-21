# DPO Fine-Tuning with Scenario Steering — Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~30 min (GPU recommended) |
| **Prerequisites** | DPO Fine-Tuning Quick Reference, JAX arrays, preference basics |
| **Format** | Python + Jupyter |

A Tier 2 tutorial covering the complete DPO fine-tuning and scenario steering
workflow end-to-end. Extends the DPO quick reference with guided generation:
a technique for steering trajectory generation toward a target scenario type
(e.g. `"forward"`, `"lane_change"`) using a kinematic reward signal.

This tutorial runs on real Waymo Open Dataset preference pairs, so a WOD Motion
validation shard is required. Only the checkpoint-absent warm-up model is
synthetic; the preference pairs are built from real WOD trajectories.

## Files

- **Python Script**: [`examples/alignment/03_dpo_finetuning_tutorial.py`](https://github.com/avitai/DiffAV/blob/main/examples/alignment/03_dpo_finetuning_tutorial.py)
- **Jupyter Notebook**: [`examples/alignment/03_dpo_finetuning_tutorial.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/alignment/03_dpo_finetuning_tutorial.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/alignment/03_dpo_finetuning_tutorial.py`

## What You'll Learn

1. Build preference pairs from synthetic trajectory candidates using `SafetyReward`
2. Run a plain DPO training loop and record alignment curves
3. Configure `ScenarioSteeringConfig` with target scenario and steering strength
4. Understand `compute_scenario_reward` for rule-based kinematic scoring
5. Execute `steer_step()` to blend DPO loss with scenario reward
6. Compare steered vs. unsteered trajectory distributions
7. Call `ScenarioMiner.steer()` for high-level steered generation

## Pipeline Overview

```
PreferenceBatch                ScenarioSteeringConfig
    |                                    |
    +-- to_dpo_batch()      target="forward", α=0.3
    |                                    |
    v                                    v
DPOAlignmentTrainer <──── ScenarioSteeringTrainer
    |                           |
    +-- compute_dpo_loss()      +-- compute_scenario_reward()
    |   (Monte Carlo)           |   (rule-based kinematic reward)
    v                           v
        steer_step() → DPOAlignmentMetrics
                              |
                              v
                    ScenarioMiner.steer()
```

## Steering Objective

Each steering step samples candidates from the current policy, ranks them by
scenario + safety reward, gates feasibility on road edges, and applies a
standard DPO update on the ranked chosen/rejected pairs. The selection
pressure α sets how tight the top/bottom pools are
(``ceil((1 − α) · num_feasible)``): higher α widens the reward gap between
pairs, and α = 0 pairs randomly (no steering).

| Parameter | Effect |
|-----------|--------|
| `α = 0.0` | Random pairing — plain DPO, no steering pressure |
| `α = 0.3` | Moderate selection pressure — recommended default |
| `α = 1.0` | Best-vs-worst pairing — maximum steering pressure |

## Steering Configuration

```python
from diffav.alignment import ScenarioSteeringConfig

config = ScenarioSteeringConfig(
    target_scenario="forward",   # kinematic target label
    steering_strength=0.3,        # selection pressure α ∈ [0.0, 1.0]
    num_steering_steps=1,         # gradient steps per ScenarioMiner.steer()
    num_candidates=8,             # candidates sampled per scene context
    feasibility_threshold=0.0,    # max off-road fraction for "chosen"
)
```

## Quick Usage

```python
from diffav.alignment import (
    DPOAlignmentConfig,
    DPOAlignmentTrainer,
    ScenarioSteeringConfig,
    ScenarioSteeringTrainer,
    create_reference_model,
)
from substrax.optim import create_optimizer, OptimizerConfig

# Build DPO trainer
optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3, gradient_clip_norm=1.0))
dpo_trainer = DPOAlignmentTrainer(
    model=model,
    optimizer=optimizer,
    config=DPOAlignmentConfig(reference_free=True),
)

# Wrap with steering
steering_config = ScenarioSteeringConfig(
    target_scenario="forward",
    steering_strength=0.3,
)
steer_trainer = ScenarioSteeringTrainer(
    dpo_trainer=dpo_trainer,
    config=steering_config,
)

# Run steered training step
metrics = steer_trainer.steer_step(dpo_batch, jax.random.key(0))
```

## High-Level SDK

```python
from diffav.api.scenario_miner import create_scenario_miner
from diffav.alignment import ScenarioSteeringConfig
from diffav.api.config import MinerConfig

miner = create_scenario_miner(MinerConfig(...))
steered = miner.steer(
    scenario_type="forward",
    density="medium",
    steering_config=ScenarioSteeringConfig(
        target_scenario="forward",
        steering_strength=0.3,
        num_steering_steps=5,
    ),
    count=10,
)
```

## Prerequisites

- DiffAV installed (`uv sync`)
- Familiarity with [DPO Fine-Tuning Quick Reference](dpo-finetuning-quickref.md)
- JAX arrays and basic understanding of preference-based alignment

## Related

- [DPO Fine-Tuning Quick Reference](dpo-finetuning-quickref.md)
- [Preference Construction Quick Reference](preference-construction-quickref.md)
- [ScenarioSteeringConfig API](../../api/alignment/scenario_steering.md)
- [DPOAlignmentTrainer API](../../api/alignment/dpo_trainer.md)
- [ScenarioMiner API](../../api/api/scenario_miner.md)
- [Alignment User Guide](../../user-guide/alignment.md)

## Terminal Output

The output below is from a run on a Modal L4 GPU (`modal run deploy/modal_app.py --task examples`), with deterministic GPU kernels and WOD read from the data volume. Without a checkpoint the tutorial runs a 300-step synthetic warm-up instead. The steering spine's warning that no road edges were passed repeats 30 times in the log and is shown once.

```text
JAX devices: [CudaDevice(id=0)]
Future steps: 80, Agents: 8
No checkpoint found — running a 300-step jitted synthetic warm-up instead.
Warm-up complete — diffusion loss: 0.017
Prepared 32 scenes from 45 records
Preference pool: 64 pairs, chosen shape (64, 8, 80, 4)
Scene contexts:  (64, 8, 128)
DPO step   0: loss=0.6931  win-rate=50.00%  margin=0.0000
DPO step  20: loss=0.6633  win-rate=100.00%  margin=0.0611
DPO step  40: loss=0.6042  win-rate=100.00%  margin=0.1910
DPO step  60: loss=0.5892  win-rate=100.00%  margin=0.2240
DPO step  80: loss=0.5599  win-rate=100.00%  margin=0.2975
DPO step 100: loss=0.5217  win-rate=100.00%  margin=0.3963
DPO step 120: loss=0.5216  win-rate=100.00%  margin=0.3925
DPO step 140: loss=0.4529  win-rate=100.00%  margin=0.6117
DPO step 160: loss=0.4016  win-rate=100.00%  margin=0.7790
DPO step 180: loss=0.4080  win-rate=100.00%  margin=0.7418
DPO step 199: loss=0.4008  win-rate=100.00%  margin=0.7783
Alignment curves saved to $AVITAI_OUTPUT_DIR/examples/dpo_alignment_curves.png
Reference-free step  0: loss=0.3193  win-rate=100.00%  margin=1.7123
Reference-free step 10: loss=0.3345  win-rate=100.00%  margin=1.5647
Reference-free step 20: loss=0.3144  win-rate=100.00%  margin=1.7788
Reference-free step 30: loss=0.3763  win-rate=100.00%  margin=1.3598
Reference-free step 40: loss=0.3405  win-rate=100.00%  margin=1.5037
Reference-free step 49: loss=0.3632  win-rate=100.00%  margin=1.5208
Standard-DPO final margin (reference-anchored): 0.7783
Reference-free final margin (policy-only):      1.5208
Target scenario:    forward
Steering strength:  0.3
Steering steps:     1
Mutation blocked: cannot assign to field 'steering_strength'
Invalid strength: steering_strength must be in [0.0, 1.0]; got 1.5
Scenario reward matrix:
  forward traj   → 'forward'     reward: +0.9875
  forward traj   → 'lane_change' reward: -1.0000
  lane_change traj → 'forward'     reward: +0.4938
  lane_change traj → 'lane_change' reward: +1.0000
All rewards in [-1, 1]: True
Steer step  0: loss=0.6930  win-rate=100.00%  margin=0.0003
Steer step  3: loss=0.6921  win-rate=100.00%  margin=0.0022
Steer step  6: loss=0.6938  win-rate=0.00%  margin=-0.0013
Steer step  9: loss=0.6932  win-rate=50.00%  margin=-0.0001
Steer step 11: loss=0.6932  win-rate=50.00%  margin=-0.0000
DPO-aligned mean x-displacement: 33.2633
Steered     mean x-displacement: 33.7977
Steered model trajectories shape: (8, 80, 4)
Trajectory comparison saved to $AVITAI_OUTPUT_DIR/examples/steered_trajectory_comparison.png
Generated 3 steered scenarios
  Scenario 0: trajectories=(2, 80, 4), tags=(<ScenarioType.FORWARD: 'forward'>, <Density.LOW: 'low'>, 'steered_forward')
  Scenario 1: trajectories=(2, 80, 4), tags=(<ScenarioType.FORWARD: 'forward'>, <Density.LOW: 'low'>, 'steered_forward')
  Scenario 2: trajectories=(2, 80, 4), tags=(<ScenarioType.FORWARD: 'forward'>, <Density.LOW: 'low'>, 'steered_forward')
WARNING:diffav.alignment.steering_spine:sample_and_score called without road_edges; feasibility gate is inactive.
```

## Example Output

![DPO loss, reward accuracy, and reward margin over training.](../../assets/images/examples/dpo_alignment_curves.png)

*DPO loss, reward accuracy, and reward margin over training.*
![Sampled trajectories before and after scenario steering.](../../assets/images/examples/steered_trajectory_comparison.png)

*Sampled trajectories before and after scenario steering.*
