# Preference Construction Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~10 min (CPU) |
| **Prerequisites** | JAX arrays, reward functions, DPO basics |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~10 min CPU) demonstrating how to build preference
pairs for DPO fine-tuning of the trajectory diffusion model. The pipeline
scores candidate trajectories with safety rewards, ranks them, and constructs
chosen/rejected pairs for the DPO trainer.

## Files

- **Python Script**: [`examples/alignment/01_preference_construction_quickref.py`](https://github.com/avitai/simulacrax/blob/main/examples/alignment/01_preference_construction_quickref.py)
- **Jupyter Notebook**: [`examples/alignment/01_preference_construction_quickref.ipynb`](https://github.com/avitai/simulacrax/blob/main/examples/alignment/01_preference_construction_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/alignment/01_preference_construction_quickref.py`

## What You'll Learn

1. Configure the composite SafetyReward with collision, kinematic, boundary, and comfort components
2. Score candidate trajectories and inspect per-component rewards
3. Build preference pairs with best-vs-worst and adjacent ranking strategies
4. Convert preference batches to DPO training format
5. Stack pairs across multiple driving scenes

## Prerequisites

- Simulacrax installed (`uv sync`)
- JAX arrays, reward functions, DPO basics

## Pipeline Overview

```
DiffusionModel.sample() -> candidates (N, A, T, 4)
    |
SafetyReward(candidates) -> scores (N,)
    |
PreferencePairBuilder
    |-- score_candidates() -> (N,)
    |-- rank_candidates()  -> sorted by reward
    |-- build_pairs()      -> list[PreferencePair]
    +-- build_batch()      -> PreferenceBatch
                                |
                                +-- to_dpo_batch() -> {"chosen", "rejected"}
```

## Quick Usage

```python
from simulacrax.alignment import (
    PreferencePairBuilder,
    PreferencePairConfig,
    SafetyReward,
)

reward = SafetyReward()
builder = PreferencePairBuilder(
    reward_fn=reward,
    config=PreferencePairConfig(top_k=2, num_candidates=8),
)
batch = builder.build_batch(candidates, conditions=None)
dpo_batch = batch.to_dpo_batch()
```

**Terminal Output:**
```
Safety reward components: ['collision', 'kinematic', 'boundary', 'comfort']
Candidates shape: (8, 4, 20, 4)
  = (8 candidates, 4 agents, 20 steps, 4 state)
Best vs Worst strategy:
  Chosen shape:  (2, 4, 20, 4)
  Rejected shape: (2, 4, 20, 4)
  Margins: ['69.4622', '53.1470']
DPO batch keys: ['chosen', 'rejected']
Multi-scene batch: 6 pairs from 3 scenes
```

## Ranking Strategies

| Strategy | Pairs per scene | Margin | Use case |
|----------|----------------|--------|----------|
| `BEST_VS_WORST` | `top_k` | Maximum | Strong training signal |
| `ADJACENT` | `N // 2` | Smaller | Fine-grained preference |

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `BicycleModelConstraint` | simulacrax.physics | Kinematic reward computation |
| `RewardFunction` protocol | artifex | Interface contract for reward functions |

## Related

- [DPO Fine-Tuning Quick Reference](dpo-finetuning-quickref.md)
- [SafetyReward API](../../api/alignment/rewards.md)
- [PreferencePairBuilder API](../../api/alignment/preferences.md)
- [Alignment User Guide](../../user-guide/alignment.md)
- [Physics-Informed Training](../models/physics-informed-training-tutorial.md)
