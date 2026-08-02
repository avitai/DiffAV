# Bicycle Model Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min (CPU) |
| **Prerequisites** | JAX arrays, vehicle kinematics basics |
| **Format** | Python + Jupyter |

A Tier 1 quick reference (~1 min CPU) demonstrating the bicycle model kinematic
constraints for trajectory validation and physics-informed loss computation.
The constraint enforces realistic vehicle dynamics by checking position,
heading, acceleration, steering angle, and velocity constraints.

## Files

- **Python Script**: [`examples/physics/01_bicycle_model_quickref.py`](https://github.com/avitai/simulacrax/blob/main/examples/physics/01_bicycle_model_quickref.py)
- **Jupyter Notebook**: [`examples/physics/01_bicycle_model_quickref.ipynb`](https://github.com/avitai/simulacrax/blob/main/examples/physics/01_bicycle_model_quickref.ipynb)

## Requirements & Run

1. Install dependencies: `uv sync`
2. Activate the managed environment: `source ./activate.sh`
3. Point at your WOD Motion TFRecords (once): add
   `WOD_MOTION_TFRECORD_PATH=/path/to/motion_v1.2.1/tf_example` to `.env.data`
4. Run: `python examples/physics/01_bicycle_model_quickref.py`

## What You'll Learn

1. Configure a `BicycleModelConstraint` with vehicle dynamics parameters
2. Compute kinematic residuals for trajectory plausibility scoring
3. Validate trajectories against kinematic limits
4. Use the Ackerman steering constraint for turning radius checks
5. Compute differentiable gradients through kinematic constraints

## Prerequisites

- Simulacrax installed (`uv sync`)
- JAX arrays, vehicle kinematics basics

## Quick Usage

```python
from simulacrax.physics.kinematics import (
    BicycleModelConfig, BicycleModelConstraint,
)

config = BicycleModelConfig(wheelbase=2.7, dt=0.1)
constraint = BicycleModelConstraint(config)

# Compute kinematic residual (lower = more plausible)
residual = constraint.compute_residuals(trajectory)

# Validate against kinematic limits
result = constraint.validate(trajectory)
print(f"Valid: {result.is_valid}")
```

## Summary

| Feature | API |
|---------|-----|
| Kinematic residual | `BicycleModelConstraint.compute_residuals(traj)` |
| Trajectory validation | `BicycleModelConstraint.validate(traj)` |
| Turning radius | `AckermanSteeringConstraint.compute_min_turning_radius()` |
| Steering violation | `AckermanSteeringConstraint.compute_violation(traj)` |

All operations are vectorized JAX, JIT-compatible, and differentiable.

## Related

- [Physics-Informed Training Tutorial](../models/physics-informed-training-tutorial.md)
- [BicycleModelConstraint API](../../api/physics/kinematics.md)
- [SimulacraxPhysicsLoss API](../../api/physics/losses.md)
