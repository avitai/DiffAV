# WOD Loading Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min (CPU) |
| **Prerequisites** | Simulacrax installed, WOD TFRecord data |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~5 min) demonstrating how to load real Waymo Open Dataset
Motion scenarios from TFRecord files using `WODSource`, iterate over datarax
`Element` objects, and parse raw scenario dicts into typed `SceneContext` domain
objects.

## Files

- **Python Script**: [`examples/core/01_wod_loading_quickref.py`](https://github.com/avitai/simulacrax/blob/main/examples/core/01_wod_loading_quickref.py)
- **Jupyter Notebook**: [`examples/core/01_wod_loading_quickref.ipynb`](https://github.com/avitai/simulacrax/blob/main/examples/core/01_wod_loading_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/core/01_wod_loading_quickref.py`

## What You'll Learn

1. Create a `WODSourceConfig` pointing to real WOD TFRecord data
2. Build a `WODSource` that loads and parses TFRecords automatically
3. Iterate over `Element` objects and inspect metadata
4. Parse raw WOD dicts into `SceneContext`, `AgentState`, and `MapFeature`
5. Access raw trajectory arrays with validity tracking

## Prerequisites

- Simulacrax installed (`uv sync`)
- At least one WOD Motion TFRecord shard downloaded
- A `.env` file with `WOD_MOTION_TFRECORD_PATH` set (see [Installation](../../getting-started/installation.md))

## Quick Usage

```python
import os

from dotenv import load_dotenv

from simulacrax.data.wod_source import WODSource, WODSourceConfig

load_dotenv()

# Configure and load (eager mode loads all at init, then releases TF)
config = WODSourceConfig(
    wod_path=os.environ["WOD_MOTION_TFRECORD_PATH"],
    split="val",
    mode="eager",
    max_agents=32,
)
source = WODSource(config)

# Iterate Elements
for element in source:
    print(element.metadata["scenario_id"])
```

## Loading Modes

`WODSource` supports two modes via the `mode` config parameter:

- **`"eager"`** (default): Loads all scenarios to memory at init. TF resources
  are released after loading. Best for validation splits and development.
- **`"streaming"`**: Lazily iterates TFRecords with fixed prefetch. Handles
  training splits too large for memory.

```python
# Streaming mode for large training sets
config = WODSourceConfig(
    wod_path=os.environ["WOD_MOTION_TFRECORD_PATH"],
    split="train",
    mode="streaming",
)
source = WODSource(config)

for element in source:
    process(element.data)
```

## Parse to Domain Types

```python
from simulacrax.data.parsers import parse_scenario

element = source[0]
scene = parse_scenario(element.data)

print(f"Ego: {scene.ego_state.position}")
print(f"Agents: {len(scene.agent_states)}")
print(f"Map features: {len(scene.map_features)}")
```

## Related

- [Quick Start Guide](../../getting-started/quickstart.md)
- [Core Concepts](../../getting-started/concepts.md)
- [WODSource API](../../api/data/wod_source.md)
- [Parsers API](../../api/data/parsers.md)
