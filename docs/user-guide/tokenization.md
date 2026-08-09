# Scene Tokenization

DiffAV converts raw Waymo Open Dataset (WOD) scenarios into
differentiable embeddings through a hierarchical multi-modal
tokenization pipeline. This guide explains the architecture and
how it integrates with the datarax operator abstraction.

## Pipeline Overview

The `SceneTokenizer` is a `CompositeOperatorModule` that chains up to
nine operators sequentially via the datarax DAG:

```
Raw WOD dict
    ↓ AgentNormalizationOperator  — ego-centric coordinates
    ↓ MapCroppingOperator         — 150 m radius crop
    ↓ TemporalStackingOperator    — stack 11 history steps
    ↓ AgentEncoder                — [num_agents, embed_dim]
    ↓ MapEncoder                  — [max_polylines, embed_dim]
    ↓ EgoEncoder                  — [embed_dim]
    ↓ LiDAREncoder                — [num_lidar_points, embed_dim]
    ↓ CameraEncoder               — [num_cameras, embed_dim]
    ↓ SceneFusionOperator         — [total_tokens, embed_dim]
```

All operators use JAX operations and are vmap/JIT-compatible inside
the datarax DAG. Modality legs (and the preprocessing operators that
serve them) are included only when the modality is active — see
[Modality Negotiation](#modality-negotiation).

## Modality Negotiation

Each modality (`agent`, `map`, `ego`, `lidar`, `camera`) carries a
negotiation mode in `TokenizerConfig`:

| Mode | Behaviour |
|------|-----------|
| `auto` (default) | Enabled when the source's element spec provides every key the modality reads |
| `required` | Construction raises `ValueError` naming the missing keys |
| `off` | Excluded unconditionally, even when the data is present |

Negotiation happens at construction against the data source's
`element_spec()`:

```python
source = WODSource(WODSourceConfig(split="train"))
tokenizer = SceneTokenizer(
    TokenizerConfig(),
    element_spec=source.element_spec(),
)
tokenizer.active_modalities  # (agent, map, ego) — WOD Motion has no sensor data
```

WOD Motion scenarios carry no LiDAR or camera arrays, so `auto`
negotiates those encoders off and the fused embedding contains only
agent, map, and ego tokens. A sensor-simulation dataset whose spec
includes `lidar/points` or `camera/images` re-enables them without any
config change. Passing no `element_spec` skips negotiation (there is
nothing to negotiate against) and every non-`off` modality stays
enabled.

`SceneFusionOperator` derives its input fields from the active set, so
`total_tokens` counts only active modalities.

## Preprocessing Operators

### Agent Normalization

`AgentNormalizationOperator` centers all coordinates on the SDC
(self-driving car) position at the current timestep and rotates
so the ego vehicle faces the +x direction:

$$
x'_i = R_{-\theta_{sdc}} \begin{bmatrix} x_i - x_{sdc} \\ y_i - y_{sdc} \end{bmatrix}
$$

where $R_{-\theta}$ is the 2D rotation matrix that negates the ego
heading. After normalization, the SDC is at the origin facing right.

### Map Cropping

`MapCroppingOperator` marks roadgraph points beyond `crop_radius`
(default 150 m) from the SDC as invalid, reducing map complexity
to only the locally relevant region.

### Temporal Stacking

`TemporalStackingOperator` extracts the first `history_steps` (default
11) from each temporal feature field and concatenates them into a flat
`stacked_history` tensor `[num_agents, history_steps × num_features]`.

## Modality Encoders

### Agent Encoder (VectorNet-style)

Encodes each agent's stacked history through:
1. Linear projection → LayerNorm → ReLU
2. `TransformerEncoderBlock` for temporal self-attention
3. Masking invalid agents to zero

Output: `agent_emb` of shape `[num_agents, embed_dim]`

### Map Encoder (VectorNet Hierarchical)

Two-stage encoding following VectorNet (CVPR 2020):

**Stage 1 — Polyline subgraph:** Each roadgraph point (xyz + direction
+ type = 7 features) is encoded with an MLP. Raw polyline IDs (arbitrary
in WOD data) are densely re-indexed by rank, then points are mean-pooled
per polyline via a one-hot matmul (JIT-safe and deterministic on GPU;
scenes with more unique polylines than `max_polylines` keep the smallest
IDs):

$$
\text{poly}_k = \frac{\sum_{i \in S_k} \text{MLP}(p_i) \cdot v_i}{\sum_{i \in S_k} v_i}
$$

where $S_k$ is the set of points belonging to polyline $k$ and $v_i$
is the validity flag.

**Stage 2 — Global graph:** Polyline centroids form a proximity graph
(edges between polylines within the map encoder's fixed `edge_radius`,
50 m default). `EGNNLayer`
from artifex runs equivariant message passing to capture spatial
relationships between polylines.

Output: `map_emb` of shape `[max_polylines, embed_dim]`

### Ego Encoder

Extracts the SDC agent's stacked history and encodes through a 2-layer
MLP + LayerNorm. Output: `ego_emb` of shape `[embed_dim]`.

### LiDAR Encoder

Subsamples to `num_lidar_points` via linear indexing (JIT-safe),
projects 3D coordinates through an MLP, then processes with stacked
`TransformerEncoderBlock` layers. Output: `lidar_emb` of shape
`[num_lidar_points, embed_dim]`.

### Camera Encoder

Encodes each camera view independently via `jax.vmap` over a
3-layer Conv → ReLU → stride-2 stack followed by global average
pooling. Output: `camera_emb` of shape `[num_cameras, embed_dim]`.

## Cross-Modal Fusion

`SceneFusionOperator` concatenates all modality tokens and applies
one of three fusion strategies:

| Strategy | Method | Reference |
|----------|--------|-----------|
| `cross_attention` | Each token attends to all others via `CrossModalAttention` | Wayformer, DriveTransformer |
| `early` | Concat all tokens, joint self-attention | Wayformer ablation (matches hierarchical) |
| `additive` | Mean-pool across tokens, broadcast back | MoST |

**Default is `cross_attention`** — DriveTransformer ablations showed
an 86% score drop without cross-modal attention.

Output: `scene_embedding` of shape `[total_tokens, embed_dim]` where
`total_tokens = num_agents + 1 (ego) + max_polylines + num_lidar_points + num_cameras`.

## Output Shape Reference

| Modality | Array key | Shape |
|----------|-----------|-------|
| Agents | `agent_emb` | `[num_agents, embed_dim]` |
| Map | `map_emb` | `[max_polylines, embed_dim]` |
| Ego | `ego_emb` | `[embed_dim]` |
| LiDAR | `lidar_emb` | `[num_lidar_points, embed_dim]` |
| Camera | `camera_emb` | `[num_cameras, embed_dim]` |
| Fused | `scene_embedding` | `[total_tokens, embed_dim]` |

## datarax Integration

`SceneTokenizer` extends `CompositeOperatorModule` with
`CompositionStrategy.SEQUENTIAL`. Each operator in the pipeline
receives the full data dict, adds its output key, and passes
the enriched dict forward.

The pipeline is invoked via:

```python
result, state, metadata = tokenizer.apply(wod_dict, {}, {})
scene_embedding = result["scene_embedding"]  # [total_tokens, embed_dim]
```

Because every operator uses JAX operations, the entire pipeline is
differentiable and JIT-compatible.

## Sister Repo Components

| Component | From | Usage |
|-----------|------|-------|
| `EGNNLayer` | artifex | MapEncoder global graph |
| `TransformerEncoderBlock` | artifex | AgentEncoder, LiDAREncoder, early fusion |
| `CrossModalAttention` | artifex | SceneFusionOperator cross-attention |
| `OperatorModule` | datarax | All 9 operators inherit from this |
| `CrossModalOperator` | datarax | SceneFusionOperator base class |
| `CompositeOperatorModule` | datarax | SceneTokenizer base class |

## Configuration

```python
from diffav.data.tokenizer import SceneTokenizer, TokenizerConfig
from diffav.core.types import FusionStrategy

config = TokenizerConfig(
    embed_dim=256,          # Embedding dimension for all modalities
    num_heads=8,            # Attention heads
    num_egnn_layers=3,      # EGNN layers in MapEncoder
    fusion_strategy=FusionStrategy.CROSS_ATTENTION,
    history_steps=11,       # 1.1 s at 10 Hz
    crop_radius=150.0,      # Map cropping radius (metres)
    max_polylines=256,      # Fixed JIT-compatible polyline count
    num_lidar_points=1024,  # LiDAR subsampling target
)
tokenizer = SceneTokenizer(config)
```

## Related

- [Scene Tokenization Tutorial](../examples/core/scene-tokenization-tutorial.md)
- [WOD Loading Quick Reference](../examples/core/wod-loading-quickref.md)
- [Tokenizer API](../api/data/tokenizer.md)
- [Operators API](../api/data/operators.md)
- [Trajectory Generation Guide](models.md)
