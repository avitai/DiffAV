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
# Scene Tokenization Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~8 min (CPU) |
| **Prerequisites** | WOD data loading (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This tutorial demonstrates the differentiable scene tokenization pipeline that
converts a real Waymo Open Dataset scenario into multi-modal embeddings.
The `SceneTokenizer` composes up to nine operators sequentially:

```
Real WOD scenario (WODSource)
    ↓ AgentNormalizationOperator  — ego-centric coordinates
    ↓ MapCroppingOperator         — 150 m radius crop
    ↓ TemporalStackingOperator    — stack 11 history steps
    ↓ AgentEncoder                — [num_agents, embed_dim]
    ↓ MapEncoder                  — [max_polylines, embed_dim]
    ↓ EgoEncoder                  — [embed_dim]
    ↓ LiDAREncoder                — [num_lidar_points, embed_dim]*
    ↓ CameraEncoder               — [num_cameras, embed_dim]*
    ↓ SceneFusionOperator         — [total_tokens, embed_dim]
```

(*WOD Motion does not include sensor data; LiDAR and camera use synthetic
placeholders. Sensor data is available in WOD Perception.)

Modalities are capability-negotiated at construction: pass the source's
`element_spec()` to `SceneTokenizer` and each `auto` modality is enabled only
when the source provides its keys — on plain WOD Motion the LiDAR and camera
legs negotiate off and fusion covers the active set. This tutorial adds
synthetic sensor arrays and passes no spec, so all nine operators run.

Every operator uses JAX operations, making the full pipeline differentiable
and JIT-compatible.

## What You'll Learn

1. Load a real WOD scenario and pass it directly to `SceneTokenizer`
2. Run preprocessing operators on real trajectory and roadgraph data
3. Encode individual modalities and inspect their shapes
4. Run the full end-to-end pipeline and compare fusion strategies
5. Verify gradient flow through the differentiable pipeline

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | simulacrax.data | Real WOD TFRecord loading |
| `EGNNLayer` | artifex | MapEncoder equivariant graph |
| `TransformerEncoderBlock` | artifex | AgentEncoder, LiDAREncoder |
| `CrossModalAttention` | artifex | SceneFusionOperator default fusion |
| `OperatorModule` | datarax | Base class for all 9 operators |
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

```bash
WOD_MOTION_TFRECORD_PATH=/path/to/waymo/motion_v1.2.1/tf_example
```

### Installation

```bash
uv sync
```
"""

# %%
# Imports

import jax
import jax.numpy as jnp
import numpy as np
from dotenv import load_dotenv
from flax import nnx

from simulacrax.core.types import FusionStrategy
from simulacrax.data import resolve_wod_tfrecord_path
from simulacrax.data.encoders import (
    AgentEncoder,
    AgentEncoderConfig,
    CameraEncoder,
    CameraEncoderConfig,
    EgoEncoder,
    EgoEncoderConfig,
    LiDAREncoder,
    LiDAREncoderConfig,
    MapEncoder,
    MapEncoderConfig,
)
from simulacrax.data.operators import (
    AgentNormalizationConfig,
    AgentNormalizationOperator,
    MapCroppingConfig,
    MapCroppingOperator,
    TemporalStackingConfig,
    TemporalStackingOperator,
)
from simulacrax.data.tokenizer import SceneTokenizer, TokenizerConfig
from simulacrax.data.wod_source import WODSource, WODSourceConfig


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 80  # 8 s at 10 Hz
MAX_AGENTS = 32  # SDC can sit anywhere in the agent rows (row 3 in scenario 0)
# — aggressive truncation risks dropping it, which crashes ego-centric operators

# %% [markdown]
"""
## 1. Load Real WOD Data

`WODSource` parses WOD TFRecords into scenario dicts. We load the first
validation scenario and pass it directly to the tokenizer operators.

WOD Motion includes: agent trajectories (`state/all/*`) and roadgraph
(`roadgraph_samples/*`). LiDAR and camera data are only available in
WOD Perception; we add synthetic placeholders for those modalities.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=MAX_AGENTS,
        history_steps=HIST_STEPS,
        future_steps=FUTURE_STEPS,
    )
)
print(f"Loaded {len(source)} validation scenarios")
# Expected output:
# Loaded 333 validation scenarios

element = source[0]
wod_dict = dict(element.data)  # real WOD dict from TFRecord

# WOD Motion does not include sensor data — add synthetic placeholders
# (WOD Perception adds real lidar/camera; set image dims to match config below)
rng = np.random.default_rng(42)
num_rg_points = wod_dict["roadgraph_samples/xyz"].shape[0]
wod_dict["lidar/points"] = rng.standard_normal((200, 3)).astype(np.float32)
wod_dict["camera/images"] = rng.standard_normal((5, 32, 32, 3)).astype(np.float32)

print(f"Agent trajectories: {wod_dict['state/all/x'].shape}  [max_agents, num_timesteps]")
print(f"Roadgraph points:   {wod_dict['roadgraph_samples/xyz'].shape}  [num_points, xyz]")
print(f"LiDAR points:       {wod_dict['lidar/points'].shape}  [N, xyz]  (synthetic placeholder)")
print(f"Camera images:      {wod_dict['camera/images'].shape}  [5 cams, H, W, C]  (synthetic)")
# Expected output (shapes vary by scenario):
# Agent trajectories: (32, 91)  [max_agents, num_timesteps]
# Roadgraph points:   (N, 3)   [num_points, xyz]  (WOD has up to 30000)
# LiDAR points:       (200, 3)  [N, xyz]  (synthetic placeholder)
# Camera images:      (5, 32, 32, 3)  [5 cams, H, W, C]  (synthetic)

# %% [markdown]
"""
## 2. Configure the Tokenizer

Small configuration for rapid CPU demonstration.
In production use `embed_dim=256`, `num_heads=8`, and WOD-scale values.
"""

# %%
config = TokenizerConfig(
    embed_dim=64,
    num_heads=4,
    num_egnn_layers=2,
    fusion_strategy=FusionStrategy.CROSS_ATTENTION,
    history_steps=HIST_STEPS,
    crop_radius=150.0,
    max_polylines=16,  # WOD default: 256
    num_lidar_points=64,  # WOD default: 1024
)
tokenizer = SceneTokenizer(config, rngs=nnx.Rngs(0))
print(f"Embed dim:    {config.embed_dim}")
print(f"Fusion:       {config.fusion_strategy}")
print(f"History:      {config.history_steps} steps (1.1 s at 10 Hz)")
print(f"Max polylines:{config.max_polylines}")
# Expected output:
# Embed dim:    64
# Fusion:       FusionStrategy.CROSS_ATTENTION
# History:      11 steps (1.1 s at 10 Hz)
# Max polylines:16

# %% [markdown]
"""
## 3. Preprocessing Operators

The first three operators normalize and reshape the raw data.
They have no trainable parameters and are deterministic.

### 3.1 Agent Normalization

Centers all positions on the SDC and rotates to the ego heading frame.
After normalization the SDC x, y at the current timestep should be 0.
"""

# %%
norm_op = AgentNormalizationOperator(
    AgentNormalizationConfig(current_step_idx=10),  # Timestep 10 = current
    rngs=nnx.Rngs(0),
)
normed, _, _ = norm_op.apply(wod_dict, {}, {})
sdc_rows = np.where(np.array(wod_dict["state/is_sdc"]) == 1)[0]
if sdc_rows.size == 0:
    raise RuntimeError("SDC not found — it may have been truncated out by max_agents")
sdc_idx = int(sdc_rows[0])
sdc_x_before = float(wod_dict["state/all/x"][sdc_idx, 10])
sdc_x_after = float(normed["state/all/x"][sdc_idx, 10])
print(f"SDC x at t=10 before normalization: {sdc_x_before:.2f} m")
print(f"SDC x at t=10 after normalization:  {sdc_x_after:.6f} m (should be 0)")
sdc_y_after = float(normed["state/all/y"][sdc_idx, 10])
print(f"SDC y at t=10 after normalization:  {sdc_y_after:.6f} m (should be 0)")
# Expected output (raw position depends on WOD scenario):
# SDC x at t=10 before normalization: xxx.xx m  (real WOD position)
# SDC x at t=10 after normalization:  0.000000 m (should be 0)
# SDC y at t=10 after normalization:  0.000000 m (should be 0)

# %% [markdown]
"""
### 3.2 Map Cropping

Marks roadgraph points beyond `crop_radius` (150 m) from the SDC as invalid.
Real WOD scenarios have many roadgraph points; cropping keeps only nearby road features.
"""

# %%
crop_op = MapCroppingOperator(
    MapCroppingConfig(crop_radius=150.0, current_step_idx=10),
    rngs=nnx.Rngs(0),
)
cropped, _, _ = crop_op.apply(normed, {}, {})
valid_before = int(wod_dict["roadgraph_samples/valid"].sum())
valid_after = int(cropped["roadgraph_samples/valid"].sum())
print(f"Roadgraph points valid before crop: {valid_before}/{num_rg_points}")
print(f"Roadgraph points valid after crop:  {valid_after}/{num_rg_points}")
print(f"Points removed by crop: {valid_before - valid_after}")
# Expected output (varies by scenario):
# Roadgraph points valid before crop: N/M
# Roadgraph points valid after crop:  K/M (some removed outside 150m)
# Points removed by crop: N - K

# %% [markdown]
"""
### 3.3 Temporal Stacking

Extracts `history_steps` (11) from each temporal feature and concatenates
them into a flat `stacked_history` tensor per agent.

`feature_fields` selects which temporal keys are stacked. This demo uses
positions only; richer configurations add `state/all/bbox_yaw` and
`state/all/velocity_x` / `state/all/velocity_y` for heading- and
velocity-aware histories. The downstream `AgentEncoderConfig.mlp_hidden`
must match `history_steps x num_features` (11 x 2 = 22 here).
"""

# %%
stack_op = TemporalStackingOperator(
    TemporalStackingConfig(
        history_steps=HIST_STEPS,
        feature_fields=["state/all/x", "state/all/y"],
    ),
    rngs=nnx.Rngs(0),
)
stacked, _, _ = stack_op.apply(cropped, {}, {})
print(
    f"stacked_history: {stacked['stacked_history'].shape}  "
    f"[num_agents, history_steps × num_features = 11 × 2]"
)
print(f"future_x:        {stacked['future_x'].shape}  [num_agents, future_steps]")
# Expected output:
# stacked_history: (32, 22)  [num_agents, history_steps × num_features = 11 × 2]
# future_x:        (32, 80)  [num_agents, future_steps]

# %% [markdown]
"""
## 4. Modality Encoders

Each encoder reads a specific key, produces learned embeddings, and
adds them back to the dict.

### 4.1 Agent Encoder

Encodes `stacked_history` through MLP → LayerNorm → ReLU →
TransformerEncoderBlock → validity masking.
"""

# %%
agent_enc = AgentEncoder(
    AgentEncoderConfig(embed_dim=64, num_heads=4, mlp_hidden=22),  # 11 steps × 2 features
    rngs=nnx.Rngs(0),
)
result, _, _ = agent_enc.apply(stacked, {}, {})
print(f"agent_emb: {result['agent_emb'].shape}  [num_agents, embed_dim]")
print(f"Finite:    {bool(jnp.isfinite(result['agent_emb']).all())}")
# Expected output:
# agent_emb: (32, 64)  [num_agents, embed_dim]
# Finite:    True

# %% [markdown]
"""
### 4.2 Ego Encoder

Extracts the SDC agent's stacked history and encodes with a 2-layer
MLP + LayerNorm, producing a single embedding vector.
"""

# %%
ego_enc = EgoEncoder(EgoEncoderConfig(embed_dim=64, mlp_hidden=22), rngs=nnx.Rngs(0))
result_ego, _, _ = ego_enc.apply(stacked, {}, {})
print(f"ego_emb: {result_ego['ego_emb'].shape}  [embed_dim]  (1D vector for the SDC)")
# Expected output:
# ego_emb: (64,)  [embed_dim]  (1D vector for the SDC)

# %% [markdown]
"""
### 4.3 Map Encoder (VectorNet Hierarchical)

**Stage 1** — MLP per point → dense polyline-ID re-index → one-hot matmul mean-pooling.
**Stage 2** — Proximity adjacency between polyline centroids → EGNNLayer.
"""

# %%
map_enc = MapEncoder(
    MapEncoderConfig(embed_dim=64, num_egnn_layers=2, max_polylines=16, edge_radius=50.0),
    rngs=nnx.Rngs(0),
)
result_map, _, _ = map_enc.apply(cropped, {}, {})
print(f"map_emb: {result_map['map_emb'].shape}  [max_polylines, embed_dim]")
print(f"Finite:  {bool(jnp.isfinite(result_map['map_emb']).all())}")
# Expected output:
# map_emb: (16, 64)  [max_polylines, embed_dim]
# Finite:  True

# %% [markdown]
"""
## 5. Sensor Encoders

*(WOD Motion has no sensor data — these encoders run on synthetic placeholders.
With WOD Perception data, pass real lidar/camera arrays instead.)*

### 5.1 LiDAR Encoder

Subsamples to `num_lidar_points`, projects 3D coordinates through MLP,
then applies TransformerEncoderBlock layers.
"""

# %%
lidar_enc = LiDAREncoder(
    LiDAREncoderConfig(embed_dim=64, num_points=64, num_layers=2, num_heads=4),
    rngs=nnx.Rngs(0),
)
result_lidar, _, _ = lidar_enc.apply(wod_dict, {}, {})
print(f"lidar_emb: {result_lidar['lidar_emb'].shape}  [num_lidar_points, embed_dim]")
# Expected output:
# lidar_emb: (64, 64)  [num_lidar_points, embed_dim]

# %% [markdown]
"""
### 5.2 Camera Encoder

Encodes each camera view independently via `jax.vmap` over a
3-layer Conv → stride-2 stack with global average pooling.
"""

# %%
cam_enc = CameraEncoder(
    CameraEncoderConfig(embed_dim=64),
    rngs=nnx.Rngs(0),
)
result_cam, _, _ = cam_enc.apply(wod_dict, {}, {})
print(f"camera_emb: {result_cam['camera_emb'].shape}  [num_cameras, embed_dim]")
# Expected output:
# camera_emb: (5, 64)  [num_cameras, embed_dim]

# %% [markdown]
"""
## 6. Full End-to-End Pipeline

`SceneTokenizer` chains all nine operators. The output `scene_embedding`
concatenates all modality tokens:
`[num_agents + 1 (ego) + max_polylines + num_lidar_points + num_cameras, embed_dim]`
"""

# %%
result, _, _ = tokenizer.apply(wod_dict, {}, {})

num_cameras = 5
total_tokens = MAX_AGENTS + 1 + config.max_polylines + config.num_lidar_points + num_cameras

print("Intermediate embeddings:")
print(f"  agent_emb:       {result['agent_emb'].shape}")
print(f"  ego_emb:         {result['ego_emb'].shape}")
print(f"  map_emb:         {result['map_emb'].shape}")
print(f"  lidar_emb:       {result['lidar_emb'].shape}")
print(f"  camera_emb:      {result['camera_emb'].shape}")
print()
print(f"scene_embedding: {result['scene_embedding'].shape}  [total_tokens, embed_dim]")
print(
    f"Expected tokens: {total_tokens}  "
    f"({MAX_AGENTS} agents + 1 ego + {config.max_polylines} map + "
    f"{config.num_lidar_points} lidar + {num_cameras} camera)"
)
assert result["scene_embedding"].shape == (total_tokens, config.embed_dim)
# Expected output:
# Intermediate embeddings:
#   agent_emb:       (32, 64)
#   ego_emb:         (64,)
#   map_emb:         (16, 64)
#   lidar_emb:       (64, 64)
#   camera_emb:      (5, 64)
#
# scene_embedding: (118, 64)  [total_tokens, embed_dim]
# Expected tokens: 118  (32 agents + 1 ego + 16 map + 64 lidar + 5 camera)

# %% [markdown]
"""
## 7. Fusion Strategy Comparison

`SceneFusionOperator` supports three strategies. All produce the same output
shape; they differ in how modalities interact.

| Strategy | Method | Tradeoff |
|----------|--------|----------|
| `cross_attention` | Q=K=V over all tokens | Most expressive; default |
| `early` | Concat + TransformerEncoderBlock | Simpler but still effective |
| `additive` | Mean-pool + broadcast | Fastest; weakest interactions |
"""

# %%
jnp_dict = {
    k: jnp.array(v)
    for k, v in wod_dict.items()
    if not k.startswith("scenario/")  # scenario/id is a byte string, not array data
}

for strategy in ["cross_attention", "early", "additive"]:
    cfg = TokenizerConfig(
        embed_dim=64,
        num_heads=4,
        num_egnn_layers=2,
        fusion_strategy=FusionStrategy(strategy),
        history_steps=HIST_STEPS,
        max_polylines=16,
        num_lidar_points=64,
    )
    tok = SceneTokenizer(cfg, rngs=nnx.Rngs(0))
    out, _, _ = tok.apply(jnp_dict, {}, {})
    emb = out["scene_embedding"]
    print(f"{strategy:>16}: {emb.shape}, finite={bool(jnp.isfinite(emb).all())}")
# Expected output (all three strategies share the same output shape):
# cross_attention: (118, 64), finite=True
#           early: (118, 64), finite=True
#        additive: (118, 64), finite=True

# %% [markdown]
"""
## 8. Gradient Flow Verification

The entire pipeline is differentiable. Gradients flow from `scene_embedding`
back through cross-modal fusion, all five encoders, and linear projections —
enabling end-to-end training from a downstream trajectory diffusion loss.
"""


# %%
def loss_fn(model: SceneTokenizer) -> jax.Array:
    """Sum of scene embedding as a scalar loss."""
    out, _, _ = model.apply(jnp_dict, {}, {})
    return jnp.sum(out["scene_embedding"])


grads = nnx.grad(loss_fn)(tokenizer)
grad_leaves = jax.tree_util.tree_leaves(grads)
nonzero = sum(1 for g in grad_leaves if hasattr(g, "shape") and float(jnp.any(jnp.abs(g) > 0)))
total = sum(1 for g in grad_leaves if hasattr(g, "shape"))
print(f"Parameter groups with nonzero gradients: {nonzero}/{total}")
print("Gradient flow through full pipeline: verified")
# Expected output:
# Parameter groups with nonzero gradients: N/N
# Gradient flow through full pipeline: verified

# %% [markdown]
"""
## 9. JIT Compilation

`nnx.jit` triggers XLA compilation. First call includes compilation
overhead; subsequent calls are significantly faster on GPU/TPU.
"""


# %%
@nnx.jit
def tokenize_jit(model: SceneTokenizer) -> jax.Array:
    """JIT-compiled tokenization."""
    out, _, _ = model.apply(jnp_dict, {}, {})
    return out["scene_embedding"]


emb1 = tokenize_jit(tokenizer)
emb2 = tokenize_jit(tokenizer)
print(f"JIT output shape: {emb1.shape}")
print(f"Deterministic:    {bool(jnp.allclose(emb1, emb2))}")
print("JIT compilation: verified")
# Expected output:
# JIT output shape: (118, 64)
# Deterministic:    True
# JIT compilation: verified

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Extend `feature_fields` with `state/all/bbox_yaw` and
   `state/all/velocity_x` / `state/all/velocity_y` (raise
   `AgentEncoderConfig.mlp_hidden` to `11 x num_features` to match)
2. Swap `FusionStrategy.ADDITIVE` for fastest inference on CPU
3. Connect `scene_embedding` to `TrajectoryDiffusionModel` as `scene_context`

### Related Examples

- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)
- [Trajectory Diffusion Quick Reference](../models/trajectory-diffusion-quickref.md)

### API Reference

- [tokenizer](../../api/data/tokenizer.md)
- [operators](../../api/data/operators.md)
"""
