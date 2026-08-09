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
# Bicycle Model Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min (CPU) |
| **Prerequisites** | JAX arrays, WOD data (see WOD Loading quickref) |
| **Format** | Python + Jupyter |

## Overview

This quick reference validates real Waymo Open Dataset trajectories against
the bicycle model kinematic constraints. The bicycle model derives steering
angle and acceleration from finite-differenced positions and headings — no
explicit control inputs required.

```
WODSource → moving, fully observed WOD vehicles (x, y, heading, speed)
                          ↓
BicycleModelConstraint.compute_residuals(traj)  → residual percentiles
                          ↓
BicycleModelConstraint.validate(traj)           → KinematicValidationResult
                          ↓
AckermanSteeringConstraint.compute_violation()  → steering violations
```

The bicycle model only applies to vehicles, so the population is filtered
before validation: vehicles only (`state/type == 1`), full 80-step future
validity (finite differences across occlusion gaps produce teleport
artefacts), and mean speed above 0.6 m/s. The speed cutoff mirrors waymax's
`InvertibleBicycleModel`, which disables curvature-based steering below its
`_SPEED_LIMIT = 0.6 m/s` guard — at near-zero speed the yaw signal is
noise-dominated and steering inversion is ill-conditioned.

Real vehicle trajectories satisfy the bicycle model with very low residuals —
the constraint is tight enough to detect physically implausible motion
(e.g., teleportation or impossibly fast turns) while accepting normal
driving behaviour.

## What You'll Learn

1. Filter WOD agents to the population the bicycle model describes
2. Compare residual percentiles for real vehicles vs a corrupted control
3. Interpret position, heading, and acceleration residuals
4. Check Ackerman steering geometry with `AckermanSteeringConstraint`
5. Compute gradients through the bicycle model loss

## Sister Repo Components

| Component | Source | Purpose |
|-----------|--------|---------|
| `WODSource` | diffav.data | Real WOD TFRecord loading |
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

from diffav.data import resolve_wod_tfrecord_path
from diffav.data.wod_source import WODSource, WODSourceConfig
from diffav.physics.kinematics import (
    AckermanSteeringConstraint,
    BicycleModelConfig,
    BicycleModelConstraint,
)


load_dotenv()
load_dotenv(".env.data")

HIST_STEPS = 11  # 10 past + 1 current
FUTURE_STEPS = 80  # 8 s at 10 Hz
MAX_AGENTS = 128  # tf_example native width
DT = 0.1  # WOD: 10 Hz
N_SCENARIOS = 50  # scenarios pooled for the residual statistics
MIN_MEAN_SPEED = 0.6  # m/s — waymax InvertibleBicycleModel _SPEED_LIMIT
VEHICLE_TYPE = 1  # WOD state/type code for vehicles

# %% [markdown]
"""
## 1. Load Real WOD Vehicle Trajectories

`WODSource` loads Waymo Open Dataset Motion TFRecords. We pool the 80-step
future trajectories of every qualifying vehicle across `N_SCENARIOS`
scenarios and convert to the bicycle model state format:
`[x, y, heading, speed]`.

Filters applied per agent:

- `state/type == 1` — vehicles only (pedestrians and cyclists do not
  follow bicycle kinematics)
- all 80 future steps valid — finite differences across occlusion gaps
  look like teleports
- mean speed > 0.6 m/s — the waymax low-speed steering guard (see Overview)
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

vehicle_trajectories = []
n_vehicles_total = 0
for i in range(min(N_SCENARIOS, len(source))):
    raw = source[i].data
    sl = (slice(None), slice(HIST_STEPS, HIST_STEPS + FUTURE_STEPS))

    x = np.asarray(raw["state/all/x"])[sl]  # (A, 80)
    y = np.asarray(raw["state/all/y"])[sl]
    heading = np.asarray(raw["state/all/bbox_yaw"])[sl]
    vx = np.asarray(raw["state/all/velocity_x"])[sl]
    vy = np.asarray(raw["state/all/velocity_y"])[sl]
    speed = np.sqrt(vx**2 + vy**2)
    valid = np.asarray(raw["state/all/valid"])[sl]
    obj_type = np.asarray(raw["state/type"]).reshape(-1)

    is_vehicle = obj_type == VEHICLE_TYPE
    n_vehicles_total += int(is_vehicle.sum())
    keep = is_vehicle & (valid.sum(axis=1) == FUTURE_STEPS) & (speed.mean(axis=1) > MIN_MEAN_SPEED)
    if keep.any():
        traj = np.stack([x[keep], y[keep], heading[keep], speed[keep]], axis=-1)
        vehicle_trajectories.append(traj)

trajectories = jnp.array(np.concatenate(vehicle_trajectories, axis=0))  # (V, 80, 4)
print(f"Vehicles across {N_SCENARIOS} scenarios: {n_vehicles_total}")
print(f"Moving, fully observed vehicles kept:  {trajectories.shape[0]}")
print(f"Trajectory array: {trajectories.shape}  [num_vehicles, future_steps, state_dim]")
# Expected output:
# Vehicles across 50 scenarios: ~2000-3000
# Moving, fully observed vehicles kept:  ~250-300
# Trajectory array: (V, 80, 4)  [num_vehicles, future_steps, state_dim]

# %% [markdown]
"""
## 2. Configure Bicycle Model

Default geometry matches a typical passenger car: 2.7 m wheelbase and
±0.7 rad steering range cover most WOD vehicles. The acceleration bound of
6.0 m/s² follows waymax's `InvertibleBicycleModel` (`max_accel = 6.0`),
which inverts WOD trajectories with the same central-difference controls
used here.
"""

# %%
config = BicycleModelConfig(
    wheelbase=2.7,  # metres — standard passenger car
    dt=DT,  # 0.1 s = 10 Hz (WOD rate)
    max_acceleration=6.0,  # m/s² — waymax InvertibleBicycleModel max_accel
    max_steering_angle=0.7,  # radians (~40°) — covers lane changes
    max_velocity=40.0,  # m/s (~144 km/h) — covers highway driving
)
constraint = BicycleModelConstraint(config)
print(f"Wheelbase: {config.wheelbase} m")
print(f"Max acceleration: {config.max_acceleration} m/s²")
deg = config.max_steering_angle * 180 / 3.14159
print(f"Max steering: {config.max_steering_angle:.1f} rad ({deg:.0f}°)")
print(f"Min turning radius: {config.wheelbase / jnp.tan(config.max_steering_angle):.2f} m")
# Expected output:
# Wheelbase: 2.7 m
# Max acceleration: 6.0 m/s²
# Max steering: 0.7 rad (40°)
# Min turning radius: ~3.2 m

# %% [markdown]
"""
## 3. Residual Percentiles: Real Vehicles vs Corrupted Control

Real WOD vehicle trajectories satisfy the bicycle model constraints — the
per-vehicle position residual (mean squared deviation between observed
displacements and the bicycle-model prediction) should sit near zero
across the whole population.

As a control, the same vehicles are corrupted with 5 m random position
jumps — physically impossible motion the constraint must flag. Comparing
percentiles shows the separation between the two populations.
"""

# %%
key = jax.random.PRNGKey(0)
corrupted = trajectories.at[:, :, :2].set(
    trajectories[:, :, :2] + jax.random.normal(key, trajectories[:, :, :2].shape) * 5.0
)

real_result = constraint.validate(trajectories)
corrupted_result = constraint.validate(corrupted)

real_residual = constraint.compute_residuals(trajectories)
corrupted_residual = constraint.compute_residuals(corrupted)
print(f"Real mean residual:      {float(real_residual):.4f}")
print(f"Corrupted mean residual: {float(corrupted_residual):.4f}")
ratio = float(corrupted_residual) / (float(real_residual) + 1e-6)
print(f"Corrupted / Real ratio:  {ratio:.1f}x")

print()
print("Per-vehicle position residual percentiles (m² per step):")
print(f"{'percentile':>12} {'real':>12} {'corrupted':>12}")
for pct in (50, 90, 99):
    real_p = float(jnp.percentile(real_result.position_residual, pct))
    corr_p = float(jnp.percentile(corrupted_result.position_residual, pct))
    print(f"{pct:>11}% {real_p:>12.4f} {corr_p:>12.4f}")
# Expected output (real WOD vehicles have near-zero residuals):
# Real mean residual:      ~0.0x
# Corrupted mean residual: >> 1.0
# Corrupted / Real ratio:  >> 1.0x
# Percentile table: real column orders of magnitude below corrupted

# %% [markdown]
"""
## 4. Detailed Violation Breakdown

`BicycleModelConstraint.validate()` returns per-violation components,
letting you identify which constraint (position, heading, acceleration)
is most violated. Each component is reported in its native unit — no
mixed-unit total.
"""

# %%
result = real_result
print(f"Real vehicle population valid: {bool(result.is_valid)}")
print(f"  Position residual (mean):   {float(jnp.mean(result.position_residual)):.6f}")
print(f"  Heading residual (mean):    {float(jnp.mean(result.heading_residual)):.6f}")
print(f"  Accel violation (mean):     {float(jnp.mean(result.acceleration_violation)):.6f}")
print(f"  Steering violation (mean):  {float(jnp.mean(result.steering_violation)):.6f}")
print(f"  Velocity violation (mean):  {float(jnp.mean(result.velocity_violation)):.6f}")
# Expected output:
# Real vehicle population valid: True (or low violations for real data)
# Position/heading residuals near 0 for moving, fully observed vehicles

# %% [markdown]
"""
## 5. Ackerman Steering Constraint

The Ackerman constraint checks the minimum turning radius implied by the
trajectory curvature. Real WOD vehicles obey their mechanical steering
limits; note this is exactly the regime where the stationary-agent filter
matters — near-zero speed makes the implied curvature blow up, which is
why waymax freezes steering below 0.6 m/s.
"""

# %%
ackerman = AckermanSteeringConstraint(config)
min_radius = ackerman.compute_min_turning_radius()
print(f"Minimum turning radius at max steering: {float(min_radius):.2f} m")

real_violation = ackerman.compute_violation(trajectories)
corrupted_violation = ackerman.compute_violation(corrupted)
print(f"Real Ackerman violation:      {float(real_violation):.6f}")
print(f"Corrupted Ackerman violation: {float(corrupted_violation):.6f}")
# Expected output:
# Minimum turning radius at max steering: ~3.2 m
# Real Ackerman violation: near 0 for normal driving
# Corrupted Ackerman violation: higher

# %% [markdown]
"""
## 6. Differentiability — Use as Training Loss

The residual is fully differentiable. Gradients flow back to the trajectory,
enabling physics-informed training: penalise violations during optimisation.
"""


# %%
def kinematic_loss(traj: jax.Array) -> jax.Array:
    """Sum bicycle model residuals — use as physics-informed training penalty."""
    return constraint.compute_residuals(traj)


grad = jax.grad(kinematic_loss)(trajectories)
print(f"Gradient shape:     {grad.shape}")
print(f"All gradients finite: {bool(jnp.all(jnp.isfinite(grad)))}")
print(f"Gradient norm:      {float(jnp.linalg.norm(grad)):.4f}")
# Expected output:
# Gradient shape:     (V, 80, 4)
# All gradients finite: True
# Gradient norm:      x.xxxx

# %% [markdown]
"""
## Next Steps

### Try These Experiments

1. Drop the `MIN_MEAN_SPEED` filter and watch the heading residual tail
   grow — stationary agents' yaw noise violates the steering-rate bound
2. Loosen `max_acceleration` back to 8.0 m/s² and compare the acceleration
   violation percentiles
3. Replace `constraint.compute_residuals` with `DiffAVPhysicsLoss` for
   full training integration

### Related Examples

- [WOD Loading Quick Reference](../core/wod-loading-quickref.md)
- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)

### API Reference

- [kinematics](../../api/physics/kinematics.md)
- [losses](../../api/physics/losses.md)
"""
