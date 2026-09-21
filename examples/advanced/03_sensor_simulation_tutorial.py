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
# Sensor Simulation Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~5-10 min (GPU) |
| **Prerequisites** | JAX basics, ScenarioMiner Quick Reference |
| **Format** | Python + Jupyter |

## What You'll Learn

1. Render RGB + depth with the NeRF renderer
2. Cast differentiable LiDAR rays over a voxel grid
3. Apply rain, fog, and snow weather augmentation
4. Chain sensor operators into a datarax pipeline

## Overview

DiffAV provides three differentiable sensor simulation modules under
`diffav.sensor`:

1. **NeRF Renderer** — scene-conditioned RGB + depth rendering via Neural
   Radiance Fields; uses `SinusoidalEmbedding` (opifex) and `StandardMLP`
   (opifex, `snake` activation)
2. **Weather Augmentations** — physics-inspired image degradations (rain,
   fog, glare) built on `ModalityOperator` (datarax), composable with
   `CompositeOperatorModule`
3. **LiDAR Ray Caster** — differentiable ray marching through a voxel density
   grid; density grids can be produced by `VoxelModel` (artifex)

All three are pure JAX, fully differentiable, and JIT-compatible.

## Architecture

```
scene_context (ctx_len, context_dim)
         │
         ▼
NeRFRenderer.render_scene(scene_context, camera_pose)
         │
         ▼
RenderedImage
  ├── pixels (H, W, 3)  ← RGB in [0, 1]
  └── depth  (H, W)     ← per-pixel depth in world units

         │
         ▼
RainAugmentation / FogAugmentation / GlareAugmentation
         │    (datarax ModalityOperator stack)
         ▼
CompositeOperatorModule (SEQUENTIAL strategy)
         │
         ▼
augmented image (H, W, 3)

voxel_density (R, R, R)   ← from jnp, manual, or artifex.VoxelModel
         │
         ▼
LiDARRayCaster.cast_rays(voxel_density, position, rotation, key)
         │
         ▼
PointCloud
  ├── points      (num_beams, 3)  ← world-space hit positions
  └── intensities (num_beams,)    ← return intensity in [0, 1]
```

## Sister Repo Components Used

| Component | Source | Role |
|-----------|--------|------|
| `SinusoidalEmbedding` | `opifex.neural.operators.common.embeddings` | NeRF positional encoding |
| `StandardMLP` | `opifex.neural.base` | Scene projector + field MLP (snake activation) |
| `ModalityOperator` | `datarax.core.modality` | Base for weather augmentation operators |
| `CompositeOperatorModule` | `datarax.operators.composite_operator` | Weather pipeline compose |
| `VoxelModel` | `artifex...geometric.voxel` | 3-D occupancy grid (optional) |
"""

# %% [markdown]
"""
## Part 1: NeRF-Based Scene Rendering

The `NeRFRenderer` accepts a scene context embedding (e.g., from
`SceneTransformer`) and a camera pose, and synthesises an RGB image with
per-pixel depth via differentiable volume rendering.

Under the hood:

- `SinusoidalEmbedding(in_channels=3, embedding_type="nerf")` from opifex
  encodes 3-D sample positions along each ray.
- Two `StandardMLP` networks (opifex, `snake` activation) act as a scene
  projector and a density + colour field.
- Alpha compositing along each ray produces the final pixels and expected depth.
"""

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from substrax.artifacts import resolve_output_dir


PLOT_DIR = resolve_output_dir("examples").path
from flax import nnx

from diffav.sensor.nerf_renderer import (
    CameraPose,
    NeRFRenderer,
    NeRFRendererConfig,
    RenderedImage,
)


# Renderer sized after the jaxnerf reference implementation: 64 coarse
# samples per ray (configs/demo.yaml num_coarse_samples) and 10 positional
# encoding frequencies (nerf/utils.py max_deg_point=10).
config = NeRFRendererConfig(
    height=64,
    width=64,
    near_plane=0.1,
    far_plane=20.0,
    num_samples=64,
    context_dim=64,
    hidden_dim=128,
    num_frequencies=10,
)
renderer = NeRFRenderer(config, rngs=nnx.Rngs(params=jax.random.key(0)))

# Synthetic scene context: (ctx_len=8, context_dim=64)
scene_ctx = jax.random.normal(jax.random.key(1), (8, config.context_dim))

# Identity camera pose — sensor at origin, looking along -Z.
pose = CameraPose(
    position=jnp.array([0.0, 0.0, 5.0]),
    rotation=jnp.eye(3),
)

result: RenderedImage = renderer.render_scene(scene_ctx, pose)
print("pixels shape:", result.pixels.shape)  # (64, 64, 3)
print("depth  shape:", result.depth.shape)  # (64, 64)
print(f"pixel range: [{float(jnp.min(result.pixels)):.3f}, {float(jnp.max(result.pixels)):.3f}]")
print(f"depth range: [{float(jnp.min(result.depth)):.3f}, {float(jnp.max(result.depth)):.3f}]")
# Expected:
# pixels shape: (64, 64, 3)
# depth  shape: (64, 64)
# pixel range: sub-interval of [0, 1] (untrained field, exact values seed-dependent)
# depth range: within [0, 20] (exact values seed-dependent)

# %% [markdown]
"""
With freshly initialised weights the radiance field is smooth and
featureless — the RGB panel below is exactly what an untrained NeRF looks
like, and the depth panel only reflects the encoding structure. The
photometric-fit section that follows shows what the same renderer produces
after optimisation.
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(result.pixels)
axes[0].set_title("Untrained NeRF RGB (64x64)")
axes[1].imshow(result.depth, cmap="viridis")
axes[1].set_title("Untrained NeRF depth (m)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(PLOT_DIR / "sensor_nerf_render.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'sensor_nerf_render.png'}")

# %% [markdown]
"""
### JIT compilation

`NeRFRenderer.render_scene` is compatible with `nnx.jit`:
"""

# %%
jitted_render = nnx.jit(NeRFRenderer.render_scene)
result_jit = jitted_render(renderer, scene_ctx, pose)
assert result_jit.pixels.shape == (64, 64, 3)
print("JIT render succeeded.")

# %% [markdown]
"""
### Fitting the field to a target

The cell below overfits the field to a single synthetic image (horizon
gradient plus a bright disc). To be clear about what this demonstrates:

- **This is a single-view photometric fit, not novel-view synthesis.** A real
  NeRF trains on rays from many camera poses and is evaluated on held-out
  views; here one fixed pose is memorised, which any image-parameterised
  model could do. What the fit does demonstrate is stable end-to-end
  gradient flow through positional encoding, the field MLP, and volume
  rendering.
- **Hyperparameters follow the jaxnerf reference.** 64 samples per ray
  (`configs/demo.yaml`), 10 positional-encoding frequencies
  (`nerf/utils.py` `max_deg_point=10`), and Adam with learning rate decayed
  from 5e-4 to 5e-6 (`nerf/utils.py:361-385` log-lerp schedule, reproduced
  here as `optax.exponential_decay` with `decay_rate=1e-2`).
- **Each step renders the full 64x64 image = 4096 rays**, which at this
  resolution exactly equals jaxnerf's Blender batch size
  (`configs/blender.yaml` `batch_size: 4096`).
- **Renderer simplifications vs jaxnerf:** no view-direction conditioning
  (jaxnerf feeds encoded view dirs, `deg_view=4`), no white-background
  compositing (`nerf/model_utils.py:204-205` adds `1 - acc` to the RGB;
  this renderer composites over black and does not expose `acc`), and
  samples sit at fixed uniform depths rather than being stratified-jittered
  per step (`nerf/model_utils.py:116-127`).

Progress is reported as PSNR = -10 log10(MSE), the metric jaxnerf logs
(`nerf/utils.py:266-274`). Roughly 30 dB is the usual calibration point at
which a NeRF reconstruction looks clean; a single-view fit should reach it
comfortably, whereas multi-view models earn it on unseen poses.
"""

# %%
import math

import optax


_FIT_H, _FIT_W = 64, 64
rows = jnp.linspace(0.0, 1.0, _FIT_H)[:, None, None]
disc_y, disc_x = jnp.meshgrid(jnp.arange(_FIT_H), jnp.arange(_FIT_W), indexing="ij")
disc = ((disc_x - 44.0) ** 2 + (disc_y - 16.0) ** 2 < 144.0)[..., None]
fit_target = jnp.where(
    disc,
    jnp.array([1.0, 0.9, 0.6]),
    rows * jnp.array([0.35, 0.25, 0.2]) + (1 - rows) * jnp.array([0.5, 0.7, 0.95]),
)

# Smoke mode (set by the example execution tests) shrinks the fit so the
# CPU tier finishes quickly; real runs keep the full showcase schedule.
_SMOKE = os.environ.get("DIFFAV_EXAMPLES_SMOKE") == "1"
max_fit_steps = 200 if _SMOKE else 2500
# jaxnerf decays lr from 5e-4 to 5e-6 over training (nerf/utils.py:142-143);
# exponential_decay with decay_rate=1e-2 over max_fit_steps is the same
# log-linear schedule: lr(t) = 5e-4 * 0.01^(t / max_fit_steps).
fit_schedule = optax.exponential_decay(
    init_value=5e-4,
    transition_steps=max_fit_steps,
    decay_rate=1e-2,
)
fit_optimizer = nnx.Optimizer(renderer, optax.adam(fit_schedule), wrt=nnx.Param)


@nnx.jit
def fit_step(renderer_, optimizer_):
    """One photometric fitting step."""

    def loss_fn(m):
        return jnp.mean((m.render_scene(scene_ctx, pose).pixels - fit_target) ** 2)

    loss, grads_ = nnx.value_and_grad(loss_fn)(renderer_)
    optimizer_.update(renderer_, grads_)
    return loss


for fit_iteration in range(max_fit_steps):
    fit_mse = fit_step(renderer, fit_optimizer)
    if fit_iteration % 250 == 0 or fit_iteration == max_fit_steps - 1:
        fit_psnr = -10.0 * math.log10(max(float(fit_mse), 1e-12))
        print(f"fit step {fit_iteration:4d} | MSE {float(fit_mse):.6f} | PSNR {fit_psnr:6.2f} dB")
fitted = renderer.render_scene(scene_ctx, pose)
# Expected: MSE decreases monotonically apart from small plateaus; PSNR
# should climb well past the ~30 dB calibration point for this single-view
# fit (exact trajectory depends on hardware and seed).

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].imshow(result.pixels)
axes[0].set_title("Before fitting")
axes[1].imshow(jnp.clip(fitted.pixels, 0.0, 1.0))
axes[1].set_title(f"After {max_fit_steps} steps")
axes[2].imshow(fit_target)
axes[2].set_title("Target")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(PLOT_DIR / "sensor_nerf_fit.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'sensor_nerf_fit.png'}")

# %% [markdown]
"""
## Part 2: Weather Augmentation Pipeline

The three weather operators extend `datarax.core.modality.ModalityOperator`
and follow the standard `apply(data, state, metadata) → (data, state, metadata)`
protocol:

- **`RainAugmentation`** — sinusoidal streak mask alpha-blended over the image
- **`FogAugmentation`** — Beer-Lambert atmospheric scattering
  (`transmission = exp(-intensity * 3)`)
- **`GlareAugmentation`** — Gaussian brightness bloom at a configurable centre

All three compose into a single pipeline via
`datarax.operators.composite_operator.CompositeOperatorModule`.
"""

# %%
from diffav.sensor.weather import (
    FogAugmentation,
    FogConfig,
    GlareAugmentation,
    GlareConfig,
    RainAugmentation,
    RainConfig,
)


_H, _W = 16, 16
image = jnp.full((_H, _W, 3), 0.4, dtype=jnp.float32)  # dark grey scene

# Individual operators.
rain = RainAugmentation(
    RainConfig(field_key="image", intensity=0.3, streak_density=12.0),
    rngs=nnx.Rngs(0),
)
fog = FogAugmentation(
    FogConfig(field_key="image", intensity=0.5, fog_color=(0.85, 0.85, 0.9)),
    rngs=nnx.Rngs(1),
)
glare = GlareAugmentation(
    GlareConfig(field_key="image", intensity=0.4, glare_position=(0.5, 0.15)),
    rngs=nnx.Rngs(2),
)

# Apply each step individually.
data = {"image": image}
data_rain, _, _ = rain.apply(data, {}, {})
data_fog, _, _ = fog.apply(data_rain, {}, {})
data_glare, _, _ = glare.apply(data_fog, {}, {})

print("After rain  — mean pixel:", float(jnp.mean(data_rain["image"])))
print("After fog   — mean pixel:", float(jnp.mean(data_fog["image"])))
print("After glare — mean pixel:", float(jnp.mean(data_glare["image"])))
lo = float(jnp.min(data_glare["image"]))
hi = float(jnp.max(data_glare["image"]))
print(f"Output range: [{lo:.3f}, {hi:.3f}]")

# %%
stages = [
    ("clean", image),
    ("rain", data_rain["image"]),
    ("fog", data_fog["image"]),
    ("glare", data_glare["image"]),
]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, (label, img) in zip(axes, stages):
    ax.imshow(jnp.clip(img, 0.0, 1.0))
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(PLOT_DIR / "sensor_weather_stages.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'sensor_weather_stages.png'}")

# %% [markdown]
"""
### Composing into a pipeline with `CompositeOperatorModule`

`datarax.operators.composite_operator.CompositeOperatorModule` with the
`SEQUENTIAL` strategy applies operators in order, chaining outputs:
"""

# %%
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)


pipeline = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[rain, fog, glare],
    ),
)

out_data, _, _ = pipeline.apply({"image": image}, {}, {})
print("Pipeline output shape:", out_data["image"].shape)  # (16, 16, 3)
lo = float(jnp.min(out_data["image"]))
hi = float(jnp.max(out_data["image"]))
print(f"Pipeline output range: [{lo:.3f}, {hi:.3f}]")
# Values should match the step-by-step run above.

# %% [markdown]
"""
## Part 3: LiDAR Ray Casting

`LiDARRayCaster` performs ray marching through a voxel density grid and
returns a `PointCloud`.  The density grid can be:

- A hand-crafted array (as below)
- Output of `VoxelModel.generate()` from **artifex** — a 3-D generative model
  that decodes latent vectors into occupancy grids

The computation is pure JAX and fully differentiable with respect to the
density field, enabling gradient-based scene optimisation.
"""

# %%
from diffav.sensor.lidar import LiDARConfig, LiDARRayCaster, PointCloud


lidar_cfg = LiDARConfig(
    num_beams=256,
    max_range=20.0,
    num_steps=32,
    density_threshold=0.3,
    noise_std=0.02,
)
caster = LiDARRayCaster(lidar_cfg)

# A structured demo scene in the [-20, 20] m voxel volume: near-empty air,
# a solid wall along one side, and a vehicle-sized box ahead of the sensor.
# Scanning geometry (rather than uniform fog) makes the returns trace
# recognizable surfaces in the point cloud.
voxel_density = jnp.full((32, 32, 32), 0.02, dtype=jnp.float32)
voxel_density = voxel_density.at[26:28, :, :].set(0.9)  # wall at x ≈ +13 m
voxel_density = voxel_density.at[20:24, 14:18, 15:18].set(0.9)  # box ahead

cloud: PointCloud = caster.cast_rays(
    voxel_density,
    sensor_position=jnp.zeros(3),
    sensor_rotation=jnp.eye(3),
    key=jax.random.key(42),
)

print("PointCloud points shape:     ", cloud.points.shape)  # (256, 3)
print("PointCloud intensities shape:", cloud.intensities.shape)  # (256,)
dists = jnp.linalg.norm(cloud.points, axis=-1)
print(f"Distance range: [{float(jnp.min(dists)):.3f}, {float(jnp.max(dists)):.3f}]")
lo = float(jnp.min(cloud.intensities))
hi = float(jnp.max(cloud.intensities))
print(f"Intensity range: [{lo:.3f}, {hi:.3f}]")

# %% [markdown]
"""
### Masking no-hit beams

Beams that never accumulate enough density are clipped to `max_range`
(20 m here). Plotting them alongside real returns paints a phantom
spherical shell at 20 m, so they are masked out of the scatter below.

Two differences from real Waymo LiDAR data are worth keeping in mind:

- Waymo range images encode **no-return as -1**, not as a max-range
  distance (`waymo_open_dataset/utils/range_image_utils.py:264-291`
  masks `range_image_mask = range_image[..., 0] > 0` before scattering
  to a point cloud). Downstream code consuming this module's output
  must apply the equivalent `distance < max_range` mask.
- A real top-lidar range image is **64 beam channels x 2650 azimuth
  columns** (rows = laser inclinations, columns = a full azimuth sweep),
  whereas this module spreads `num_beams` over an approximately square
  azimuth x elevation grid (16 x 16 for the 256 beams above).
"""

# %%
hit_mask = dists < 0.995 * lidar_cfg.max_range
num_hits = int(jnp.sum(hit_mask))
print(f"Hit beams: {num_hits} / {lidar_cfg.num_beams} (rest clipped to max_range)")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
pts = cloud.points[hit_mask]
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cloud.intensities[hit_mask], s=12, cmap="plasma")
ax.set_title("LiDAR returns, no-hit beams masked (colour = intensity)")
fig.savefig(PLOT_DIR / "sensor_lidar_cloud.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved {PLOT_DIR / 'sensor_lidar_cloud.png'}")

# %% [markdown]
"""
### Empty-space behaviour

When the density grid is zero, no beam records a hit and every distance is
clipped to `max_range`. Note this is a **module encoding convention, not
sensor realism**: a real spinning LiDAR simply reports no return for such
beams, and Waymo range images store that as -1
(`range_image_utils.py:264-291`). Treat `distance ~= max_range` as
"no return" and mask it, exactly as done for the scatter plot above:
"""

# %%
empty = jnp.zeros((16, 16, 16))
cloud_empty = caster.cast_rays(
    empty,
    sensor_position=jnp.zeros(3),
    sensor_rotation=jnp.eye(3),
    key=jax.random.key(0),
)
dists_empty = jnp.linalg.norm(cloud_empty.points, axis=-1)
mean_d = float(jnp.mean(dists_empty))
empty_hits = int(jnp.sum(dists_empty < 0.995 * lidar_cfg.max_range))
print(f"Empty grid — mean distance: {mean_d:.3f} (max_range={lidar_cfg.max_range})")
print(f"Empty grid — hit beams after masking: {empty_hits}")
# Expected: mean distance close to max_range (20.0); 0 hit beams after masking

# %% [markdown]
"""
### Closing the loop: encoding the point cloud

`diffav.data.encoders.lidar.LiDAREncoder` consumes exactly this kind of
point cloud. It reads `lidar/points` `(N, 3)` from the data dict,
subsamples to a fixed `num_points` count, projects the coordinates, runs
transformer self-attention, and adds a per-point `lidar_emb` embedding of
shape `(num_points, embed_dim)`:
"""

# %%
from diffav.core.constants import LIDAR_EMBEDDING, LIDAR_POINTS
from diffav.data.encoders.lidar import LiDAREncoder, LiDAREncoderConfig


lidar_encoder = LiDAREncoder(
    LiDAREncoderConfig(embed_dim=64, num_points=128, num_layers=1, num_heads=4),
    rngs=nnx.Rngs(0),
)
encoded, _, _ = lidar_encoder.apply({LIDAR_POINTS: cloud.points[hit_mask]}, {}, {})
print("lidar_emb shape:", encoded[LIDAR_EMBEDDING].shape)  # (128, 64)
# Expected: lidar_emb shape: (128, 64)

# %% [markdown]
"""
### Connecting to artifex `VoxelModel`

For realistic density grids, use `VoxelModel` from artifex.  Its `generate()`
method produces an array of shape ``(n_samples, R, R, R)`` in ``[0, 1]``:

```python
from artifex.generative_models.models.geometric.voxel import VoxelModel
from artifex.generative_models.core.configuration.geometric_config import (
    VoxelConfig, VoxelNetworkConfig
)
from flax import nnx

voxel_model = VoxelModel(
    VoxelConfig(voxel_size=16, network=VoxelNetworkConfig(base_channels=32)),
    rngs=nnx.Rngs(0),
)
samples = voxel_model.generate(n_samples=1, rngs=nnx.Rngs(1))  # (1, 16, 16, 16)
cloud = caster.cast_rays(
    samples[0],  # (16, 16, 16)
    sensor_position=jnp.zeros(3),
    sensor_rotation=jnp.eye(3),
    key=jax.random.key(0),
)
```

The density field from `VoxelModel` is differentiable, so gradients can flow
from LiDAR metrics back through the occupancy representation.
"""

# %% [markdown]
"""
## Summary

| Module | Key class | Input | Output |
|--------|-----------|-------|--------|
| `sensor.nerf_renderer` | `NeRFRenderer` | context + pose | `RenderedImage` |
| `sensor.weather` | `Rain/Fog/GlareAugmentation` | image dict | augmented dict |
| `sensor.lidar` | `LiDARRayCaster` | voxel + pose | `PointCloud` |
| `data.encoders.lidar` | `LiDAREncoder` | `lidar/points` (N, 3) | `lidar_emb` |

All modules:
- Are pure JAX — no Python loops in the forward pass
- Support `nnx.jit` and `jax.grad`
- Use sister-repo building blocks: opifex (positional encoding, MLP), datarax
  (operator protocol, composition), artifex (voxel generation)
"""
