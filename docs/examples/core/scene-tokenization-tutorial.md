# Scene Tokenization Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~8 min (CPU) |
| **Prerequisites** | WOD data loading, JAX/Flax NNX basics |
| **Format** | Python + Jupyter |

A Tier 2 tutorial (~8 min CPU) demonstrating the differentiable scene
tokenization pipeline that converts raw Waymo Open Dataset (WOD) scenario
dicts into multi-modal embeddings via a sequential datarax operator DAG.

## Files

- **Python Script**: [`examples/core/02_scene_tokenization_tutorial.py`](https://github.com/avitai/DiffAV/blob/main/examples/core/02_scene_tokenization_tutorial.py)
- **Jupyter Notebook**: [`examples/core/02_scene_tokenization_tutorial.ipynb`](https://github.com/avitai/DiffAV/blob/main/examples/core/02_scene_tokenization_tutorial.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/core/02_scene_tokenization_tutorial.py`

## What You'll Learn

1. Configure and build a `SceneTokenizer` at WOD scale
2. Construct mock WOD data in the `state/all/<field>` dict format
3. Run preprocessing operators independently (normalization, cropping, stacking)
4. Encode individual modalities — agents, map, ego, lidar, camera
5. Run the full end-to-end pipeline from raw dict to `scene_embedding`
6. Switch fusion strategies and compare their output shapes
7. Verify gradient flow through the entire differentiable pipeline
8. JIT-compile the pipeline for production throughput

## Prerequisites

- DiffAV installed (`uv sync`)
- WOD data loading basics (see [WOD Loading Quick Reference](wod-loading-quickref.md))
- JAX arrays, Flax NNX basics

## Pipeline Overview

```
Raw WOD dict
    ↓ AgentNormalizationOperator  — ego-centric coordinates, no params
    ↓ MapCroppingOperator         — 150 m radius validity crop, no params
    ↓ TemporalStackingOperator    — stack 11 history steps, no params
    ↓ AgentEncoder                — [num_agents, embed_dim]
    ↓ MapEncoder                  — [max_polylines, embed_dim]  (VectorNet + EGNN)
    ↓ EgoEncoder                  — [embed_dim]
    ↓ LiDAREncoder                — [num_lidar_points, embed_dim]
    ↓ CameraEncoder               — [num_cameras, embed_dim]
    ↓ SceneFusionOperator         — [total_tokens, embed_dim]
```

`total_tokens = num_agents + 1 (ego) + max_polylines + num_lidar_points + num_cameras`

## Quick Usage

```python
from diffav.data.tokenizer import SceneTokenizer, TokenizerConfig
from diffav.core.types import FusionStrategy

config = TokenizerConfig(
    embed_dim=256,
    num_heads=8,
    fusion_strategy=FusionStrategy.CROSS_ATTENTION,
    max_polylines=256,
    num_lidar_points=1024,
)
tokenizer = SceneTokenizer(config)
result, _, _ = tokenizer.apply(wod_dict, {}, {})
scene_embedding = result["scene_embedding"]  # [total_tokens, 256]
```

## Output Shape Reference

| Modality | Key | Shape |
|----------|-----|-------|
| Agents | `agent_emb` | `[num_agents, embed_dim]` |
| Map | `map_emb` | `[max_polylines, embed_dim]` |
| Ego | `ego_emb` | `[embed_dim]` |
| LiDAR | `lidar_emb` | `[num_lidar_points, embed_dim]` |
| Camera | `camera_emb` | `[num_cameras, embed_dim]` |
| Fused | `scene_embedding` | `[total_tokens, embed_dim]` |

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `EGNNLayer` | artifex | MapEncoder — equivariant map graph |
| `TransformerEncoderBlock` | artifex | AgentEncoder, LiDAREncoder, early fusion |
| `CrossModalAttention` | artifex | SceneFusionOperator default fusion |
| `OperatorModule` | datarax | Base class for all 9 pipeline operators |
| `CompositeOperatorModule` | datarax | SceneTokenizer sequential DAG |

## Related

- [WOD Loading Quick Reference](wod-loading-quickref.md)
- [Scene Tokenization User Guide](../../user-guide/tokenization.md)
- [Trajectory Diffusion Quick Reference](../models/trajectory-diffusion-quickref.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
- [Tokenizer API](../../api/data/tokenizer.md)
- [Operators API](../../api/data/operators.md)
