# Physics-Informed Training

Simulacrax enforces physically plausible trajectory generation by combining
diffusion model training with vehicle kinematics constraints, collision
penalties, and road boundary enforcement. This guide explains the physics
components and how they integrate with the training loop.

## Bicycle Model Kinematics

The bicycle model is a simplified vehicle dynamics model that represents
a car as a single-track vehicle with front-wheel steering. It enforces
the relationship between position, heading, velocity, and steering angle.

### State Representation

Trajectories use the state vector `[x, y, heading, velocity]` at each
timestep:

| Dimension | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| 0 | $x$ | metres | Longitudinal position |
| 1 | $y$ | metres | Lateral position |
| 2 | $\theta$ | radians | Heading angle |
| 3 | $v$ | m/s | Forward velocity |

### Kinematic Equations

The bicycle model predicts the next state from controls (acceleration $a$,
steering angle $\delta$):

$$
\begin{aligned}
x_{t+1} &= x_t + v_t \cos(\theta_t) \cdot \Delta t \\
y_{t+1} &= y_t + v_t \sin(\theta_t) \cdot \Delta t \\
\theta_{t+1} &= \theta_t + \frac{v_t}{L} \tan(\delta_t) \cdot \Delta t \\
v_{t+1} &= v_t + a_t \cdot \Delta t
\end{aligned}
$$

where $L$ is the wheelbase (distance between front and rear axles).

### Finite-Difference Controls

Since trajectories do not include explicit control inputs, the constraint
derives controls from finite differences:

- **Acceleration**: $a_t = (v_{t+1} - v_t) / \Delta t$
- **Steering angle**: $\delta_t = \arctan\left(\frac{\Delta\theta_t \cdot L}{v_t \cdot \Delta t}\right)$
- **Heading change**: $\Delta\theta_t = \text{arctan2}(\sin(\theta_{t+1} - \theta_t), \cos(\theta_{t+1} - \theta_t))$

The `arctan2` wrapping ensures correct handling of heading angles crossing
the $\pm\pi$ boundary.

### Kinematic Residual

The residual quantifies how well a trajectory satisfies the bicycle model:

$$
\mathcal{L}_{\text{kin}} = \text{MSE}(\Delta x_{\text{actual}} - \Delta x_{\text{model}})
  + \text{mean}(\text{heading\_excess}^2)
$$

The heading term is a hinge on the *excess* turning beyond the steering-feasible
rate, not on all heading change:

$$
\text{heading\_excess} = \max\left(0,\;
  |\Delta\theta_{\text{wrapped}}| - |v| \cdot \frac{\Delta t}{L} \tan(\delta_{\max})
\right)
$$

so a trajectory that turns within the bicycle model's feasible steering rate
contributes zero heading residual; only turning faster than $\delta_{\max}$
allows is penalized.

Lower residuals indicate more physically plausible trajectories.

## Collision Avoidance

The collision penalty uses pairwise soft distance enforcement between all
agents at each timestep:

$$
\mathcal{L}_{\text{collision}} = \frac{1}{N(N-1)T}
  \sum_{i \neq j} \sum_t \max(0, d_{\text{safe}} - \|p_i^t - p_j^t\|)^2
$$

where $d_{\text{safe}}$ is the minimum safe distance (default 2.0 m) and
$p_i^t$ is agent $i$'s position at timestep $t$.

This quadratic penalty is smooth and differentiable, producing zero loss
when agents maintain safe distances and increasing loss as agents approach
each other.

## Road Boundary Enforcement

When oriented road edges are provided (as a
`simulacrax.core.RoadEdges`, built with `RoadEdges.from_polylines`), the
boundary penalty hinges on the WOSAC signed distance to the road edges —
negative on-road, positive off-road:

$$
\mathcal{L}_{\text{boundary}} = \text{mean}\left(\max(0, d_{\text{signed}}(p))^2\right)
$$

where $d_{\text{signed}}(p)$ is the signed distance from agent position
$p$ to the nearest road edge. On-road positions contribute zero loss;
off-road positions are penalized by their squared distance to the
boundary, so the gradient pushes trajectories back inside the drivable
area.

## Momentum Variation Penalty

The momentum-variation penalty tracks the scene's total momentum across
agents and penalizes its variance over time:

$$
\mathcal{L}_{\text{momentum}} = \text{mean}\left(\text{Var}_t\left(
  \sum_i v_i^t \begin{bmatrix} \cos\theta_i^t \\ \sin\theta_i^t \end{bmatrix}
\right)\right)
$$

This is a soft smoothness prior that discourages large collective momentum
swings — **not** a physical conservation law. Traffic momentum is not
conserved under tire and road forces, so the term is a mild regularizer,
disabled by default in the map-conditioned iteration.

## Adaptive Weight Scheduling

Physics losses are combined with the diffusion training loss using an
adaptive weight that increases during training:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}}
  + w(e) \cdot \left(
    \lambda_{\text{kin}} \mathcal{L}_{\text{kin}}
    + \lambda_{\text{col}} \mathcal{L}_{\text{collision}}
    + \lambda_{\text{bd}} \mathcal{L}_{\text{boundary}}
    + \lambda_{\text{mom}} \mathcal{L}_{\text{momentum}}
  \right)
$$

The weight $w(e)$ transitions from `initial_physics_weight` (e.g., 0.01) to
`final_physics_weight` (e.g., 1.0) over `transition_epochs` using either
exponential or linear scheduling.

**Rationale**: Early in training, the diffusion model needs to learn the
data distribution without strong physics constraints interfering. As
training progresses, the physics weight increases to enforce plausibility
in the generated trajectories.

## Opifex Integration

The physics module composes building blocks from the opifex sister
repository:

| Component | From | Purpose |
|-----------|------|---------|
| `AdaptiveWeightScheduler` | `opifex.core.physics.losses` | Weight scheduling |
| `create_optimizer` | `opifex.core.training.optimizers` | Optimizer creation |
| `ErrorRecoveryManager` | `opifex.core.training.components.recovery` | NaN detection |

The trainer can also profile step FLOPs via calibrax's `FlopsCounter`
when `TrainerConfig.profile_flops` is enabled.

## Configuration

### Physics Loss

```python
from simulacrax.physics.losses import SimulacraxPhysicsConfig

config = SimulacraxPhysicsConfig(
    kinematic_weight=1.0,        # Bicycle model residual weight
    collision_weight=1.0,        # Collision penalty weight
    collision_threshold=2.0,     # Safe distance (metres)
    road_boundary_weight=1.0,    # Boundary violation weight
    adaptive_weighting=True,     # Enable adaptive scheduling
    schedule_type="exponential", # "exponential" or "linear"
    initial_physics_weight=0.01,
    final_physics_weight=1.0,
    transition_epochs=100,
    momentum_variation_weight=0.1,
)
```

### Bicycle Model

```python
from simulacrax.physics.kinematics import BicycleModelConfig

config = BicycleModelConfig(
    wheelbase=2.7,          # metres (typical sedan)
    dt=0.1,                 # 10 Hz timestep
    max_acceleration=8.0,   # m/s^2
    max_steering_angle=0.7, # radians (~40 degrees)
    max_velocity=40.0,      # m/s (~144 km/h)
)
```

## Related

- [Bicycle Model Quick Reference](../examples/physics/bicycle-model-quickref.md)
- [Physics-Informed Training Tutorial](../examples/models/physics-informed-training-tutorial.md)
- [Kinematics API](../api/physics/kinematics.md)
- [Physics Losses API](../api/physics/losses.md)
- [Trainer API](../api/models/trainer.md)
- [Trajectory Generation Guide](models.md)
