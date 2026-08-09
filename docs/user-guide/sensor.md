# Sensor Simulation

DiffAV provides three differentiable sensor simulation modules under
`diffav.sensor`:

| Module | Key class | Purpose |
|--------|-----------|---------|
| `nerf_renderer` | `NeRFRenderer` | RGB + depth image synthesis |
| `weather` | `Rain/Fog/GlareAugmentation` | Physics-inspired image degradation |
| `lidar` | `LiDARRayCaster` | Point cloud generation from voxel grids |

All modules are pure JAX, fully differentiable, and `nnx.jit`-compatible.

## NeRF Renderer

`NeRFRenderer` synthesises RGB images and depth maps from a scene context
embedding and a camera pose using a Neural-Radiance-Field-style volumetric
renderer. It is a differentiable rendering skeleton for perception-in-the-loop
testing — **not** a photorealistic sensor model; there is no trained radiance
field, so outputs are structurally plausible rather than photoreal.

### Architecture

```
scene_context (ctx_len, context_dim)
        │  mean-pool → scene_mlp (StandardMLP, snake)
        ▼
scene_latent (hidden_dim,)

camera_pose (position + rotation)
        │  _generate_rays()
        ▼
(H*W rays) × (num_samples points per ray)
        │  camera-relative coords / far_plane → [-1, 1]
        │  SinusoidalEmbedding (opifex, "nerf") → encoded (enc_dim,)
        │  field_mlp([encoded | scene_latent]) → (density, r, g, b)
        ▼
alpha compositing → RenderedImage (pixels H×W×3, depth H×W)
```

### Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from diffav.sensor import CameraPose, NeRFRenderer, NeRFRendererConfig

cfg = NeRFRendererConfig(
    height=64, width=64,
    near_plane=0.1, far_plane=50.0,
    num_samples=64, context_dim=128,
    hidden_dim=64, num_frequencies=8,
)
renderer = NeRFRenderer(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))

scene_ctx = jax.random.normal(jax.random.key(1), (16, 128))
pose = CameraPose(
    position=jnp.array([0.0, 0.0, 5.0]),
    rotation=jnp.eye(3),
)
result = renderer.render_scene(scene_ctx, pose)
# result.pixels: (64, 64, 3)
# result.depth:  (64, 64)
```

Gradients flow through `render_scene` back to both the renderer parameters and
the `scene_context` embedding, enabling end-to-end optimisation.

## Weather Augmentations

Three augmentation operators extend
`datarax.core.modality.ModalityOperator` and apply physics-inspired
transformations to float32 images stored in a data dict.

### Operators

| Class | Effect | Key parameter |
|-------|--------|--------------|
| `RainAugmentation` | Sinusoidal streak mask | `streak_density` (streak bands across the image; higher → denser), `streak_angle_deg` |
| `FogAugmentation` | Beer-Lambert atmospheric scattering | `intensity`, `fog_color` |
| `GlareAugmentation` | Gaussian brightness bloom | `glare_position`, `glare_radius` |

All clip output to `[0, 1]` by default via the `clip_range` config field.

### Individual usage

```python
from flax import nnx
from diffav.sensor import FogAugmentation, FogConfig

fog = FogAugmentation(
    FogConfig(field_key="image", intensity=0.5),
    rngs=nnx.Rngs(0),
)
out_data, _, _ = fog.apply({"image": image}, {}, {})
foggy_image = out_data["image"]  # (H, W, 3)
```

### Composing into a pipeline

```python
from datarax.operators.composite_operator import (
    CompositeOperatorConfig, CompositeOperatorModule, CompositionStrategy,
)
from diffav.sensor import (
    RainAugmentation, RainConfig,
    FogAugmentation,  FogConfig,
    GlareAugmentation, GlareConfig,
)
from flax import nnx

rain  = RainAugmentation(RainConfig(field_key="image"),  rngs=nnx.Rngs(0))
fog   = FogAugmentation(FogConfig(field_key="image"),    rngs=nnx.Rngs(1))
glare = GlareAugmentation(GlareConfig(field_key="image"), rngs=nnx.Rngs(2))

pipeline = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[rain, fog, glare],
    ),
)
out_data, _, _ = pipeline.apply({"image": image}, {}, {})
```

## LiDAR Ray Caster

`LiDARRayCaster` casts spherical beams through a 3-D voxel density grid and
returns a `PointCloud` at the first high-density intersection.

### Key properties

- Beams arranged in a configurable spherical grid (azimuth × elevation)
- Trilinear interpolation samples the density at each step; the grid spans
  `[grid_center − max_range, grid_center + max_range]` per axis and samples
  outside it contribute zero density (no phantom returns for off-grid sensors)
- Soft first-hit via transmittance weighting with Beer-Lambert opacity
  scaled by the marching step size, so results are stable under `num_steps`
  refinement (differentiable)
- Gaussian range noise optional via `noise_std`

### Usage

```python
import jax
import jax.numpy as jnp
from diffav.sensor import LiDARConfig, LiDARRayCaster

caster = LiDARRayCaster(LiDARConfig(
    num_beams=64,
    max_range=50.0,
    num_steps=64,
    noise_std=0.02,
))

# Supply a voxel density grid directly or from artifex.VoxelModel.generate()
voxel = jnp.full((16, 16, 16), 0.3)  # (R, R, R) in [0, 1]

cloud = caster.cast_rays(
    voxel,
    sensor_position=jnp.zeros(3),
    sensor_rotation=jnp.eye(3),
    key=jax.random.key(0),
)
# cloud.points:      (64, 3)  — world-space hit positions
# cloud.intensities: (64,)    — return intensity in [0, 1]
```

### Using artifex `VoxelModel` as density source

```python
from artifex.generative_models.models.geometric.voxel import VoxelModel
from artifex.generative_models.core.configuration.geometric_config import (
    VoxelConfig, VoxelNetworkConfig,
)
from flax import nnx

voxel_model = VoxelModel(
    VoxelConfig(voxel_size=16, network=VoxelNetworkConfig(base_channels=32)),
    rngs=nnx.Rngs(0),
)
grids = voxel_model.generate(n_samples=1, rngs=nnx.Rngs(1))  # (1, 16, 16, 16)
cloud = caster.cast_rays(grids[0], jnp.zeros(3), jnp.eye(3), key=jax.random.key(0))
```

## API Reference

- [`diffav.sensor.nerf_renderer`](../api/sensor/nerf_renderer.md)
- [`diffav.sensor.weather`](../api/sensor/weather.md)
- [`diffav.sensor.lidar`](../api/sensor/lidar.md)
