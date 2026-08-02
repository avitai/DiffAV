# Quick Start

This guide walks through loading Waymo Open Dataset scenarios from real TFRecord
files, parsing them into Simulacrax domain types, and inspecting the result.

## Prerequisites

1. Download at least one WOD Motion validation shard:

    ```bash
    WOD_BUCKET="gs://waymo_open_dataset_motion_v_1_2_1"
    WOD_SHARD="uncompressed/tf_example/validation"
    SHARD_FILE="validation_tfexample.tfrecord-00000-of-00150"
    gsutil -m cp "$WOD_BUCKET/$WOD_SHARD/$SHARD_FILE" \
        /path/to/waymo/motion_v1.2.1/tf_example/validation/
    ```

2. Create a `.env` file in the project root (see `.env.example`):

    ```bash
    WOD_MOTION_TFRECORD_PATH=/path/to/waymo/motion_v1.2.1/tf_example
    ```

## Load WOD Scenarios

```python
import os

from dotenv import load_dotenv

from simulacrax.data.wod_source import WODSource, WODSourceConfig

load_dotenv()

# Configure the data source
config = WODSourceConfig(
    wod_path=os.environ["WOD_MOTION_TFRECORD_PATH"],
    split="val",
    mode="eager",       # or "streaming" for large splits
    max_agents=32,
    history_steps=11,
    future_steps=80,
)

# Create the source (loads TFRecords using CPU-only TensorFlow)
source = WODSource(config)
print(f"Loaded {len(source)} scenarios")
```

`WODSource` supports two loading modes:

- **Eager** (default): Loads all scenarios to memory at init. TensorFlow resources
  are released after loading. Best for validation sets and development.
- **Streaming**: Lazily iterates TFRecords on-the-fly with fixed prefetch.
  Handles training splits too large for memory.

## Iterate Over Elements

`WODSource` yields datarax `Element` objects. Each element wraps a raw scenario
dict with metadata:

```python
for element in source:
    print(f"Scenario: {element.metadata['scenario_id']}")
    print(f"Split: {element.metadata['split']}")
    print(f"Keys: {list(element.data.keys())[:5]}...")
    break  # Just show the first one
```

## Parse into Domain Types

Use the parser to convert raw WOD dicts into typed Simulacrax objects.
Only objects valid at the current timestep become agents, so the padding
rows real WOD records carry never appear as phantom agents:

```python
from simulacrax.data.parsers import parse_scenario

# Get the raw scenario dict from the first Element
element = source[0]
scene = parse_scenario(element.data)

print(f"Ego position: {scene.ego_state.position}")
print(f"Ego heading: {scene.ego_state.heading:.3f} rad")
print(f"Agents: {len(scene.agent_states)}")
print(f"Map features: {len(scene.map_features)}")
print(f"Timestamps: {scene.timestamps.shape}")
```

## Inspect Agent States

Each `AgentState` is a frozen dataclass with position, heading, velocity,
acceleration, and agent type:

```python
for i, agent in enumerate(scene.agent_states[:3]):
    print(f"Agent {i}: type={agent.agent_type}, "
          f"pos={agent.position}, vel={agent.velocity:.2f} m/s")
```

## Convert to Submission Format

Use the converter to produce WOD-compatible submission dicts. Each
entry carries the full `SimulatedTrajectory` proto field set —
`center_x`, `center_y`, `center_z`, `heading`, `valid`, and
`object_id` (elevation and validity accept optional overrides):

```python
from simulacrax.data.converters import to_wod_submission
from simulacrax.core.types import TrajectoryPrediction
import jax.numpy as jnp

# Create a dummy prediction (2 agents, 80 future steps, 4 state dims)
prediction = TrajectoryPrediction(
    trajectories=jnp.zeros((2, 80, 4)),
    agent_ids=("agent_0", "agent_1"),
)

submission = to_wod_submission(prediction)
print(f"Submission agents: {len(submission['simulated_trajectories'])}")
```

## Next Steps

- [Core Concepts](concepts.md) -- understand the type system and protocol architecture
- [WOD Loading Example](../examples/core/wod-loading-quickref.md) -- detailed walkthrough
- [API Reference](../api/index.md) -- complete API documentation
