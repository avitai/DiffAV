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
# Occupancy Flow Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~15 min (GPU) |
| **Prerequisites** | WOD data, SceneContext types, JAX basics |
| **Format** | Python + Jupyter |

## Overview

This tutorial demonstrates occupancy flow prediction for autonomous driving,
following the Waymo occupancy-flow challenge setup: rasterize real Waymo Open
Dataset scenes to top-down grids with the official geometry, render
ground-truth future occupancy and backward-displacement flow, and train a
Multi-Scale Fourier Neural Operator with the official loss composition.

```
WODSource → SceneContext (per waypoint)
                ↓
SceneRasterizer.rasterize()               → input OccupancyGrid (128×128, 8 ch)
SceneRasterizer.rasterize(t_k)            → GT occupancy per waypoint
SceneRasterizer.rasterize_backward_flow() → GT backward flow per waypoint
                ↓
OccupancyFlowModel(grid) → OccupancyGridPrediction
    .occupancy_logits (1, 8, 3, 128, 128) ← sigmoid XE target
    .flow             (1, 8, 128, 128, 2) ← masked L1 target
                ↓
sigmoid XE + masked L1 flow + FlowConsistencyLoss (regulariser)
```

### Grid Geometry (official)

The rasterizer follows the waymo-open-dataset occupancy-flow reference:
an 80 m × 80 m field of view (the official 256×256 grid at 3.2 px/m),
the ego anchored at 50% of the width and 75% of the height — three
quarters of the grid ahead of the ego, one quarter behind — and a
y-flipped image transform so forward is "up" and the ego's left is the
image's left.

### Two Kinds of Flow

- `rasterize().flow` — ego-frame agent velocities in m/s. Model INPUT
  features only.
- `rasterize_backward_flow()` — the official ground-truth flow: per-agent
  backward displacement `(x[t-1s] - x[t])` in grid-cell units, rendered at
  the later waypoint. This is the training target for the flow head.

## What You'll Learn

1. Build `SceneContext` snapshots for every waypoint of real WOD scenarios
2. Rasterize with the official grid geometry — 8-channel layout
3. Render ground-truth occupancy and backward flow targets
4. Train `OccupancyFlowModel` with sigmoid XE + masked L1 + continuity
5. Verify FNO resolution independence: same call at 64×64 and 128×128
6. (Bonus) Compare predictions against a classical diffusion-advection solver

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `MultiScaleFourierNeuralOperator` | opifex.neural.operators.fno.multiscale | FNO backbone |
| `WODSource` | diffav.data | Real WOD TFRecord loading |
| `solve_diffusion_advection_2d` | opifex.physics.solvers | Classical solver (optional) |
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
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from substrax.artifacts import resolve_output_dir


PLOT_DIR = resolve_output_dir("examples").path

import numpy as np
import optax
from dotenv import load_dotenv
from flax import nnx

from diffav.core.types import (
    AgentState,
    AgentType,
    MapFeature,
    MapFeatureType,
    SceneContext,
)
from diffav.data import prepare_full_horizon_scene, resolve_wod_tfrecord_path
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.occupancy import (
    create_occupancy_flow_model,
    FlowConsistencyLoss,
    FlowConsistencyLossConfig,
    OccupancyFlowConfig,
    OccupancyGrid,
    RasterizerConfig,
    SceneRasterizer,
)


load_dotenv()
load_dotenv(".env.data")

# WOD timeline: 10 past + 1 current + 80 future steps at 10 Hz.
CURRENT_STEP = 10  # absolute index of the current timestep
NUM_STEPS = 81  # current + 80 future steps (8 s)
WAYPOINT_STRIDE = 10  # waypoints are 1 s apart (8 waypoints over 8 s)
NUM_AGENTS = 16  # full-horizon-valid agents per scenario
NUM_SCENARIOS = 8
TRAIN_STEPS = 800  # 100 passes over the 8 scenarios
GRID_RES = 128
GRID_SIZE_M = 80.0  # official field of view (256 cells at 3.2 px/m)

# %% [markdown]
"""
## 1. Load Real WOD Scenes and Build Waypoint SceneContexts

We load real WOD scenarios and keep the first `NUM_SCENARIOS` that contain at
least `NUM_AGENTS` agents valid over the whole horizon
(`prepare_full_horizon_scene` performs the validity masking). For each
scenario we build one `SceneContext` per timestep of interest: the current
step plus the eight 1 s waypoints and the step one waypoint before each — the
backward-flow targets need both endpoints.

The map comes from the full valid roadgraph (`roadgraph_samples/valid`
masked, real `roadgraph_samples/type` codes) rather than a token handful of
points. WOD roadgraph type codes group into our three map channels:
lane centres (1-3) plus road lines/edges (6-16) → LANE, crosswalks (18) →
CROSSWALK, stop signs (17) → TRAFFIC_SIGNAL.
"""

# %%
WOD_PATH = resolve_wod_tfrecord_path()
source = WODSource(
    WODSourceConfig(
        wod_path=WOD_PATH,
        split="val",
        max_agents=128,  # keep every modeled agent; avoids SDC truncation
        history_steps=CURRENT_STEP + 1,
        future_steps=NUM_STEPS - 1,
    )
)
print(f"Loaded {len(source)} validation scenarios")
# Expected output:
# Loaded NNN validation scenarios  (one shard holds a few hundred)

# WOD object type codes: 1=vehicle, 2=pedestrian, 3=cyclist
_TYPE_MAP = {1: AgentType.VEHICLE, 2: AgentType.PEDESTRIAN, 3: AgentType.CYCLIST}

# WOD roadgraph type codes → map channels
_ROADGRAPH_CHANNELS = {
    MapFeatureType.LANE: (1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16),
    MapFeatureType.CROSSWALK: (18,),
    MapFeatureType.TRAFFIC_SIGNAL: (17,),  # stop signs; WOD has no signal points
}


def _build_map_features(raw: dict, ego_xy: np.ndarray) -> tuple[MapFeature, ...]:
    """Map features from the full valid roadgraph near the ego.

    Masks ``roadgraph_samples/valid`` and groups real type codes into the
    three map channels. Points farther than 75 m from the current ego cannot
    fall inside the 80 m grid (max ego-to-corner distance ~72 m) and are
    dropped to keep rasterization cheap.
    """
    xy = np.asarray(raw["roadgraph_samples/xyz"])[:, :2]
    types = np.asarray(raw["roadgraph_samples/type"]).reshape(-1)
    valid = np.asarray(raw["roadgraph_samples/valid"]).reshape(-1) > 0
    near = np.linalg.norm(xy - ego_xy[None, :], axis=1) < 75.0
    keep = valid & near
    features = []
    for channel_type, codes in _ROADGRAPH_CHANNELS.items():
        mask = keep & np.isin(types, codes)
        if mask.any():
            features.append(
                MapFeature(
                    polyline_points=jnp.asarray(xy[mask], dtype=jnp.float32),
                    feature_type=channel_type,
                )
            )
    return tuple(features)


def _agent_state(track: np.ndarray, agent_type: AgentType) -> AgentState:
    """AgentState from one [x, y, heading, speed] row."""
    return AgentState(
        position=jnp.asarray(track[:2], dtype=jnp.float32),
        heading=float(track[2]),
        velocity=float(track[3]),
        acceleration=0.0,
        agent_type=agent_type,
    )


def _prepare_scenario(raw: dict) -> dict | None:
    """Waypoint SceneContexts + map for one scenario, or None to skip.

    Uses prepare_full_horizon_scene for the validity-masked trajectory
    extraction over steps [CURRENT_STEP, CURRENT_STEP + NUM_STEPS), then
    recovers the matching agent types and the SDC row (the helper's row
    selection is mirrored: first NUM_AGENTS rows valid over the window).
    """
    result = prepare_full_horizon_scene(
        raw, num_agents=NUM_AGENTS, history_steps=CURRENT_STEP, future_steps=NUM_STEPS
    )
    if result is None:
        return None
    trajectories, _ = result  # (NUM_AGENTS, NUM_STEPS, [x, y, heading, speed])

    window = slice(CURRENT_STEP, CURRENT_STEP + NUM_STEPS)
    valid = np.asarray(raw["state/all/valid"])[:, window] > 0
    rows = np.flatnonzero(valid.all(axis=1))[:NUM_AGENTS]
    types = np.asarray(raw["state/type"]).reshape(-1)[rows]
    is_sdc = np.asarray(raw["state/is_sdc"]).reshape(-1)[rows] == 1
    if not is_sdc.any():
        return None  # SDC truncated or not full-horizon valid: skip

    sdc_slot = int(np.flatnonzero(is_sdc)[0])
    tracks = np.asarray(trajectories)  # (NUM_AGENTS, NUM_STEPS, 4)
    agent_slots = [i for i in range(NUM_AGENTS) if i != sdc_slot]
    map_features = _build_map_features(raw, tracks[sdc_slot, 0, :2])

    def scene_at(step: int) -> SceneContext:
        """SceneContext at a trajectory index (0 = current timestep)."""
        agents = tuple(
            _agent_state(tracks[i, step], _TYPE_MAP.get(int(types[i]), AgentType.VEHICLE))
            for i in agent_slots
        )
        return SceneContext(
            ego_state=_agent_state(tracks[sdc_slot, step], AgentType.VEHICLE),
            agent_states=agents,
            map_features=map_features,
            timestamps=jnp.array([0.0]),
        )

    return {"scene_at": scene_at, "current_ego": scene_at(0).ego_state}


scenarios = []
for index in range(len(source)):
    element = source[index]
    if element is None:
        continue
    prepared = _prepare_scenario(element.data)
    if prepared is not None:
        scenarios.append(prepared)
    if len(scenarios) == NUM_SCENARIOS:
        break
print(f"Prepared {len(scenarios)} scenarios with {NUM_AGENTS} full-horizon-valid agents each")
# Expected output:
# Prepared 8 scenarios with 16 full-horizon-valid agents each

# %% [markdown]
"""
## 2. Rasterize Inputs and Ground-Truth Targets

`SceneRasterizer` converts each scene to a 128×128 grid with 8 channels:

- Channels 0-2: per-type occupancy (vehicle, pedestrian, cyclist) — agents
  render as oriented soft boxes (default 4.7 m × 2.1 m; `AgentState`
  carries no per-agent dimensions, so a typical vehicle footprint stands in
  for all agents)
- Channels 3-4: ego-frame velocity field (vx, vy) in m/s — input features
- Channels 5-7: map features (lane/road, crosswalk, signal)

Ground truth per scenario, all rendered w.r.t. the ego's CURRENT pose
(`frame_ego=`), as in the official pipeline:

- occupancy at each of the 8 waypoints (t = +1 s ... +8 s)
- backward flow between consecutive waypoints, in grid-cell units
"""

# %%
HORIZON = 8
rasterizer = SceneRasterizer(RasterizerConfig(grid_resolution=GRID_RES, grid_size_m=GRID_SIZE_M))


def _rasterize_scenario(scenario: dict) -> dict:
    """Input grid + stacked occupancy / backward-flow targets for training."""
    scene_at = scenario["scene_at"]
    frame = scenario["current_ego"]
    input_grid = rasterizer.rasterize(scene_at(0))

    occupancy_frames = []
    flow_frames = []
    for k in range(HORIZON):
        step_now = (k + 1) * WAYPOINT_STRIDE
        step_prev = k * WAYPOINT_STRIDE
        grid_now = rasterizer.rasterize(scene_at(step_now), frame_ego=frame)
        # Overlapping soft boxes can sum above 1; clip for valid XE targets.
        occupancy_frames.append(jnp.minimum(grid_now.occupancy, 1.0))
        flow_frames.append(
            rasterizer.rasterize_backward_flow(
                scene_at(step_now), scene_at(step_prev), frame_ego=frame
            )
        )

    target_occupancy = jnp.transpose(jnp.stack(occupancy_frames), (0, 3, 1, 2))[None]
    return {
        "occupancy": input_grid.occupancy,
        "flow": input_grid.flow,
        "map_features": input_grid.map_features,
        "target_occupancy": target_occupancy,  # (1, T, types, H, W)
        "target_flow": jnp.stack(flow_frames)[None],  # (1, T, H, W, 2)
    }


training_data = [_rasterize_scenario(s) for s in scenarios]
grid = OccupancyGrid(
    occupancy=training_data[0]["occupancy"],
    flow=training_data[0]["flow"],
    map_features=training_data[0]["map_features"],
)
print("OccupancyGrid:")
print(f"  occupancy:    {grid.occupancy.shape}  (H, W, types)")
print(f"  flow:         {grid.flow.shape}  (H, W, 2)")
print(f"  map_features: {grid.map_features.shape}  (H, W, map_channels)")
print(f"  target occupancy: {training_data[0]['target_occupancy'].shape}  (1, T, types, H, W)")
print(f"  target flow:      {training_data[0]['target_flow'].shape}  (1, T, H, W, 2)")
# Expected output:
# OccupancyGrid:
#   occupancy:    (128, 128, 3)  (H, W, types)
#   flow:         (128, 128, 2)  (H, W, 2)
#   map_features: (128, 128, 3)  (H, W, map_channels)
#   target occupancy: (1, 8, 3, 128, 128)  (1, T, types, H, W)
#   target flow:      (1, 8, 128, 128, 2)  (1, T, H, W, 2)

# %% [markdown]
"""
## 3. Create OccupancyFlowModel

The model wraps `MultiScaleFourierNeuralOperator` from opifex and predicts
all 8 future waypoints in a single forward pass (resolution-independent).
`grid_size_m` is set explicitly so the model's cell size matches the
rasterizer's 80 m field of view.
"""

# %%
model_config = OccupancyFlowConfig(
    grid_resolution=GRID_RES,
    grid_size_m=GRID_SIZE_M,
    temporal_horizon=HORIZON,
    hidden_channels=32,
    modes_per_scale=(12, 6, 3),
    num_layers_per_scale=(2, 2, 2),
)
model = create_occupancy_flow_model(model_config, nnx.Rngs(params=jax.random.key(0)))

param_count = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
print(f"Model parameters: {param_count:,}")

prediction = model(grid)
print("Prediction shapes:")
print(f"  occupancy: {prediction.occupancy.shape}  (B, T, types, H, W)")
print(f"  flow:      {prediction.flow.shape}  (B, T, H, W, 2)")
# Expected output:
# Model parameters: N,NNN,NNN
# Prediction shapes:
#   occupancy: (1, 8, 3, 128, 128)  (B, T, types, H, W)
#   flow:      (1, 8, 128, 128, 2)  (B, T, H, W, 2)

# %% [markdown]
"""
## 4. Train with the Official Loss Composition

Following the occupancy-flow challenge tutorial:

- **Occupancy**: sigmoid cross-entropy on `prediction.occupancy_logits`
  against the rasterized future occupancy, summed over cells and scaled by
  1000 / num_cells.
- **Flow**: L1 against the backward-displacement ground truth, masked to
  cells where GT flow exists (soft rendering leaves near-zero values
  everywhere, so "exists" means magnitude above a small threshold).
- **Continuity** (`FlowConsistencyLoss`, dt=1.0 — waypoints are 1 s apart):
  an EXTRA physics regulariser on top of the official terms, not a
  replacement for supervised flow.

One scenario per step, cycling through all of them; adam at 1e-3 (the
official tutorial's optimizer).
"""

# %%
_FLOW_EXISTS_THRESHOLD = 0.05  # grid cells; soft-box floor for "GT flow exists"

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
continuity = FlowConsistencyLoss(
    FlowConsistencyLossConfig(weight=0.1, dt=1.0, cell_size_m=model_config.cell_size_m)
)


@nnx.jit
def train_step(model_, optimizer_, occ, flow, map_feats, target_occ, target_flow):
    """One step: sigmoid XE + masked L1 flow + continuity regulariser."""

    def loss_fn(m):
        pred = m(OccupancyGrid(occupancy=occ, flow=flow, map_features=map_feats))
        xe = optax.sigmoid_binary_cross_entropy(pred.occupancy_logits, target_occ)
        occupancy_loss = jnp.sum(xe) * 1000.0 / (GRID_RES * GRID_RES) / xe.shape[1] / xe.shape[2]
        flow_exists = (
            jnp.linalg.norm(target_flow, axis=-1, keepdims=True) > _FLOW_EXISTS_THRESHOLD
        ).astype(jnp.float32)
        flow_loss = jnp.sum(jnp.abs(pred.flow - target_flow) * flow_exists) / (
            jnp.sum(flow_exists) + 1.0
        )
        total = occupancy_loss + flow_loss + continuity.compute(pred)
        return total, (occupancy_loss, flow_loss)

    (loss, (occ_loss, flow_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model_)
    optimizer_.update(model_, grads)
    return loss, occ_loss, flow_loss


for step in range(TRAIN_STEPS):
    data = training_data[step % len(training_data)]
    loss, occ_loss, flow_loss = train_step(
        model,
        optimizer,
        data["occupancy"],
        data["flow"],
        data["map_features"],
        data["target_occupancy"],
        data["target_flow"],
    )
    if step % 100 == 0 or step == TRAIN_STEPS - 1:
        print(
            f"step {step:3d} | total {float(loss):.4f} | "
            f"occupancy XE {float(occ_loss):.4f} | flow L1 {float(flow_loss):.4f}"
        )
prediction = model(grid)
# Expected: all three terms decrease; occupancy XE falls by an order of
# magnitude as the model learns to move the observed agents forward.

# %%
data0 = training_data[0]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(jnp.sum(grid.occupancy, axis=-1), cmap="magma")
axes[0].set_title("Rasterized input (t=0)")
axes[1].imshow(data0["target_occupancy"][0, 0].sum(axis=0), cmap="magma")
axes[1].set_title("Ground truth (t=+1s)")
axes[2].imshow(prediction.occupancy[0, 0].sum(axis=0), cmap="magma")
axes[2].set_title("Predicted (t=+1s)")
axes[3].imshow(prediction.occupancy[0, -1].sum(axis=0), cmap="magma")
axes[3].set_title(f"Predicted (t=+{HORIZON}s)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(PLOT_DIR / "occupancy_flow_grids.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'occupancy_flow_grids.png'}")
# The ego sits at 75% of the grid height (three quarters of the view ahead
# of it), and forward motion moves agents UP the image — official layout.

# %% [markdown]
"""
## 5. Per-Waypoint Occupancy Statistics

After training, per-type occupancy mass should track the ground truth:
agents drift across the grid instead of dissolving into noise.
"""

# %%
print("Per-waypoint occupancy statistics (trained model, scenario 0):")
print(f"{'t (s)':>5} {'Vehicle mean':>14} {'Ped mean':>10} {'Cyclist mean':>13} {'GT vehicle':>12}")
print("-" * 58)
for t in range(HORIZON):
    occ_t = prediction.occupancy[0, t]  # (3, H, W)
    gt_t = data0["target_occupancy"][0, t]
    print(
        f"{t + 1:>5} {float(jnp.mean(occ_t[0])):>14.6f} {float(jnp.mean(occ_t[1])):>10.6f} "
        f"{float(jnp.mean(occ_t[2])):>13.6f} {float(jnp.mean(gt_t[0])):>12.6f}"
    )
# Expected output (values vary by scenario):
# t (s)   Vehicle mean   Ped mean  Cyclist mean   GT vehicle
# ----------------------------------------------------------
#     1       x.xxxxxx   x.xxxxxx      x.xxxxxx     x.xxxxxx
#     ...

# %% [markdown]
"""
## 6. FlowConsistencyLoss

The continuity loss penalises predictions where occupancy changes without
matching flow: ∂ρ/∂t + ∇·(ρv) = 0. With dt=1.0 the temporal derivative
matches the 1 s waypoint spacing. During training above it acted as a soft
physics regulariser alongside the supervised occupancy and flow terms.
"""

# %%
loss_fn = FlowConsistencyLoss(
    FlowConsistencyLossConfig(
        weight=0.1,
        dt=1.0,  # occupancy-flow waypoints are 1 s apart
        cell_size_m=model_config.cell_size_m,  # derived from the grid config
    )
)
phys_loss = loss_fn.compute(prediction)
print(f"FlowConsistencyLoss: {float(phys_loss):.6f}")
# Expected output:
# FlowConsistencyLoss: x.xxxxxx  (small after training)

# %% [markdown]
"""
## 7. Resolution Independence

The FNO produces valid predictions at any grid resolution.
Useful for fast CPU testing (64×64) vs. production (256×256, the official
challenge resolution).
"""

# %%
print("Resolution independence test:")
scene_for_demo = scenarios[0]["scene_at"](0)
for res in (64, 128):
    cfg = OccupancyFlowConfig(
        grid_resolution=res,
        grid_size_m=GRID_SIZE_M,
        modes_per_scale=(8, 4, 2),
        num_layers_per_scale=(1, 1, 1),
        hidden_channels=16,
        use_gradient_checkpointing=False,  # faster for demo
    )
    m = create_occupancy_flow_model(cfg, nnx.Rngs(params=jax.random.key(0)))
    rast = SceneRasterizer(RasterizerConfig(grid_resolution=res, grid_size_m=GRID_SIZE_M))
    g = rast.rasterize(scene_for_demo)
    pred = m(g)
    print(f"  Resolution {res}x{res}: occupancy {pred.occupancy.shape}, flow {pred.flow.shape}")
# Expected output:
# Resolution independence test:
#   Resolution 64x64:  occupancy (1, 8, 3, 64, 64),  flow (1, 8, 64, 64, 2)
#   Resolution 128x128: occupancy (1, 8, 3, 128, 128), flow (1, 8, 128, 128, 2)

# %% [markdown]
"""
## 8. (Bonus) Classical Solver Comparison

`opifex.physics.solvers.diffusion_advection.solve_diffusion_advection_2d`
provides a classical reference solution for a 2D density advected by a known
velocity field. Comparing the FNO occupancy predictions against this
classical solver offers a sanity check: both should agree when the flow
field is simple and known.
"""

# %%
# Bonus: Classical solver comparison for sanity check
try:
    from opifex.physics.solvers.diffusion_advection import solve_diffusion_advection_2d

    H = 64
    # Uniform rightward flow: vx=1 m/s, vy=0
    initial_density = jnp.zeros((H, H)).at[H // 2, H // 2].set(1.0)
    classical_pred = solve_diffusion_advection_2d(
        initial_condition=initial_density,
        diffusion_coeff=0.1,
        advection_vel=(1.0, 0.0),
        dt=1.0,  # match the 1 s waypoint spacing
        n_steps=8,
    )
    print(f"Classical solver output shape: {classical_pred.shape}")
    print("Classical solver comparison available — FNO predictions can be validated against it.")
except (ImportError, AttributeError) as exc:
    print(f"Classical solver not available: {exc}")
# Expected: classical_pred shape (64, 64) — the final advected density field

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Increase `hidden_channels=64` and `grid_resolution=256` for the official
   challenge scale
2. Train on more scenarios (raise `NUM_SCENARIOS`) and watch the flow L1
   generalise instead of memorising
3. Sweep the continuity weight — the regulariser trades occupancy sharpness
   for physically consistent flow
4. Run the classical solver comparison and verify FNO agreement on simple
   flows

### Related Examples

- [Physics-Informed Training Tutorial](../models/02_physics_informed_training_tutorial.py)
- [WOD Metrics Quick Reference](../evaluation/01_wod_metrics_quickref.py)

### API Reference

- [flow_model](../../docs/api/occupancy/flow_model.md)
- [rasterizer](../../docs/api/occupancy/rasterizer.md)
- [losses](../../docs/api/occupancy/losses.md)
"""
