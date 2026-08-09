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
# WOD Loading Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min (CPU) |
| **Prerequisites** | Basic Python, JAX arrays |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates how to load real Waymo Open Dataset (WOD)
Motion scenarios from TFRecord files using `WODSource`, iterate over datarax
`Element` objects, and parse raw scenario dicts into typed DiffAV domain
objects.

`WODSource` extends datarax's `DataSourceModule`, making WOD data
compatible with the datarax operator ecosystem. Under the hood, TensorFlow
is used CPU-only for TFRecord parsing while all downstream processing uses
JAX arrays.

## Learning Goals

By the end of this example, you will be able to:

1. Create a `WODSourceConfig` pointing to real WOD TFRecord data
2. Build a `WODSource` that loads and parses TFRecords automatically
3. Iterate over datarax `Element` objects and inspect metadata
4. Parse raw WOD dicts into `SceneContext`, `AgentState`, and `MapFeature`
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

Create a `.env` file in the project root:

```bash
WOD_MOTION_TFRECORD_PATH=/path/to/waymo/motion_v1.2.1/tf_example
```

### Installation

```bash
uv sync
```

**Estimated Time:** ~2 minutes
"""

# %%
# Imports

import numpy as np
from dotenv import load_dotenv

from diffav.core.types import AgentType  # noqa: F401
from diffav.data import resolve_wod_tfrecord_path
from diffav.data.parsers import parse_scenario
from diffav.data.wod_source import WODSource, WODSourceConfig


load_dotenv()
load_dotenv(".env.data")


# %% [markdown]
"""
## Implementation

### Step 1: Configure WODSource

Point `WODSourceConfig` at your local WOD TFRecord directory. The config
validates all fields on construction and `WODSource` handles TFRecord
parsing, temporal aggregation (past/current/future to state/all/*), and
agent limiting automatically.

`max_agents=128` is the native tf_example agent width. Requesting fewer
truncates rows in storage order — the SDC or `tracks_to_predict` rows can
sit anywhere in the array, so truncation can silently drop them. Keep the
full width for loading and select agents downstream instead.
"""

# %%
# Configure the source — set WOD_MOTION_TFRECORD_PATH in your .env file
WOD_MOTION_TFRECORD_PATH = resolve_wod_tfrecord_path()

config = WODSourceConfig(
    wod_path=WOD_MOTION_TFRECORD_PATH,
    split="val",
    max_agents=128,  # native tf_example width — no truncation
    history_steps=11,
    future_steps=80,
)
print(f"Config: split={config.split}, total_steps={config.total_steps}")
print(f"Max agents: {config.max_agents}")

# Build the source — this loads and parses all TFRecords in the split directory
source = WODSource(config)
print(f"Source length: {len(source)} scenarios")

# %% [markdown]
"""
### Step 2: Iterate Elements

`WODSource` yields datarax `Element` objects containing the raw
scenario dict in `.data` and metadata in `.metadata`.
"""

# %%
# Iterate over first 5 elements
for i, element in enumerate(source):
    if i >= 5:
        break
    info = element.metadata.source_info or {}
    sid = info["scenario_id"]
    split = info["split"]
    num_keys = len(element.data)
    print(f"  Scenario {i}: id={sid}, split={split}, features={num_keys}")

# Random access by index
element_0 = source[0]
print()
info_0 = element_0.metadata.source_info or {}
print(f"First element scenario: {info_0['scenario_id']}")

# Access by scenario ID
first_id = source.scenario_ids[0].decode("utf-8")
found = source.get_by_scenario_id(first_id)
print(f"Found by ID: {(found.metadata.source_info or {})['scenario_id']}" if found else "Not found")

# %% [markdown]
"""
### Step 3: Parse to Domain Types

Use `parse_scenario` to convert raw WOD dicts into typed
`SceneContext` objects with `AgentState` and `MapFeature` instances.
Only objects valid at the current timestep become agents — the
padding rows real TFRecords carry are excluded.
"""

# %%
# Parse the first scenario
element = source[0]
scene = parse_scenario(element.data)

# Inspect the SceneContext
print(f"Ego position: ({scene.ego_state.position[0]:.2f}, {scene.ego_state.position[1]:.2f})")
print(f"Ego heading: {scene.ego_state.heading:.3f} rad")
print(f"Ego type: {scene.ego_state.agent_type}")
print(f"Ego velocity: {scene.ego_state.velocity:.2f} m/s")
print(f"Agents: {len(scene.agent_states)}")
print(f"Map features: {len(scene.map_features)}")
print(f"Timestamps shape: {scene.timestamps.shape}")

# %% [markdown]
"""
### Step 4: Inspect Agent and Map Details

Examine individual agent states and map feature types from a real scenario.
"""

# %%
# Inspect agent types — real WOD has vehicles, pedestrians, cyclists
print("Agent breakdown:")
type_counts: dict[str, int] = {}
for agent in scene.agent_states:
    name = agent.agent_type.value
    type_counts[name] = type_counts.get(name, 0) + 1
for agent_type, count in sorted(type_counts.items()):
    print(f"  {agent_type}: {count}")

# Show first 5 agents with details
print()
print("First 5 agents:")
for i, agent in enumerate(scene.agent_states[:5]):
    print(
        f"  Agent {i}: type={agent.agent_type.value}, "
        f"pos=({agent.position[0]:.1f}, {agent.position[1]:.1f}), "
        f"vel={agent.velocity:.2f} m/s"
    )

# Inspect map features
print()
print("Map feature breakdown:")
feat_counts: dict[str, int] = {}
for feat in scene.map_features:
    name = feat.feature_type.value
    feat_counts[name] = feat_counts.get(name, 0) + 1
for feat_type, count in sorted(feat_counts.items()):
    print(f"  {feat_type}: {count}")

# %% [markdown]
"""
### Step 5: Examine Trajectory Data

Access the raw temporal data to see full agent trajectories across
91 timesteps (10 past + 1 current + 80 future).
"""

# %%
# Access raw trajectory arrays from the Element data
raw = element.data
xs = raw["state/all/x"]  # shape: [max_agents, 91]
ys = raw["state/all/y"]
valid = raw["state/all/valid"]

print(f"Trajectory array shape: {xs.shape}")
print(f"Valid mask shape: {valid.shape}")

# Find SDC (ego) index
sdc_rows = np.where(raw["state/is_sdc"] == 1)[0]
if sdc_rows.size == 0:
    raise RuntimeError("SDC not found — it may have been truncated out by max_agents")
sdc_idx = int(sdc_rows[0])
print(f"SDC (ego) index: {sdc_idx}")

# Challenge-designated prediction targets for this scenario
ttp_rows = np.where(np.asarray(raw["state/tracks_to_predict"]).reshape(-1) > 0)[0]
print(f"tracks_to_predict rows: {ttp_rows.tolist()}")

# Show ego trajectory at key timesteps
for t, label in [(0, "t=0 (oldest past)"), (10, "t=10 (current)"), (90, "t=90 (latest future)")]:
    is_valid = "valid" if valid[sdc_idx, t] == 1 else "invalid"
    print(f"  {label}: pos=({xs[sdc_idx, t]:.2f}, {ys[sdc_idx, t]:.2f}) [{is_valid}]")

# Count valid timesteps per agent
valid_counts = valid.sum(axis=1)
print()
print(f"Agents with full validity (91/91): {(valid_counts == 91).sum()}")
print(f"Agents with partial validity: {((valid_counts > 0) & (valid_counts < 91)).sum()}")
print(f"Fully invalid agents (padded): {(valid_counts == 0).sum()}")


# %% [markdown]
"""
## Results & Evaluation

### What We Achieved

| Aspect | Result |
|--------|--------|
| Data source | Real WOD Motion v1.2.1 TFRecords |
| Scenarios loaded | All scenarios from validation shard |
| Element iteration | With metadata (scenario_id, split, index) |
| Scene parsing | SceneContext with real ego, agents, map |
| Trajectory access | 91 timesteps with validity tracking |

### Key Points

- `WODSourceConfig` validates all fields on construction
- `WODSource` extends `DataSourceModule` for datarax compatibility
- TFRecord parsing aggregates past/current/future into state/all/* arrays
- `parse_scenario` produces typed domain objects from real driving data
- Real scenarios contain vehicles, pedestrians, and cyclists
"""

# %%
print("WOD loading quick reference complete!")

# %% [markdown]
"""
## Next Steps & Resources

### Try These Experiments

1. Load multiple shards by pointing `wod_path` to a directory with more files
2. Filter agents by type using `AgentType` enum values
3. Visualize ego trajectory by plotting the `state/all/x` and `state/all/y` arrays
4. Compare scenarios by iterating through `source` and examining agent counts

### Related Examples

- See the `models/` and `alignment/` examples for trajectory diffusion and DPO fine-tuning

### API Reference

- [WODSource](../../docs/api/data/wod_source.md)
- [parse_scenario](../../docs/api/data/parsers.md)
- [SceneContext](../../docs/api/core/types.md)
"""


# %%
def main() -> None:
    """Main entry point for command-line execution."""
    load_dotenv()
    load_dotenv(".env.data")
    print("Running WOD loading quick reference...")

    wod_path = resolve_wod_tfrecord_path()
    cfg = WODSourceConfig(wod_path=wod_path, split="val", max_agents=128)
    src = WODSource(cfg)

    total_agents = 0
    for elem in src:
        sc = parse_scenario(elem.data)
        total_agents += len(sc.agent_states)

    print(f"Processed {len(src)} scenarios, {total_agents} total agents")
    print("Done!")


if __name__ == "__main__":
    main()
