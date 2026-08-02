# Occupancy Flow Prediction

Occupancy flow prediction answers: *where will each type of agent be at the next N timesteps?*
The answer is a top-down grid (`OccupancyGrid`) over a 60m × 60m region anchored to the ego vehicle
(75% of the extent ahead, matching the official occupancy-flow grid conventions),
with one channel per agent type and a 2D flow vector field. The model's
`OccupancyFlowConfig.grid_size_m` defaults to 60 m; the standalone rasterizer default is 80 m.

## Pipeline

```
SceneContext ──► SceneRasterizer ──► OccupancyGrid ──► OccupancyFlowModel ──► OccupancyGridPrediction
                  (differentiable)    (H, W, 8)         (MultiScaleFNO)         (B, T, types, H, W)
```

## Rasterization

`SceneRasterizer` renders each agent as a differentiable oriented soft box (default 4.7m × 2.1m
vehicle footprint at the agent's heading); map polyline points render as Gaussian blobs:

```python
from simulacrax.occupancy import RasterizerConfig, SceneRasterizer

config = RasterizerConfig(grid_resolution=128, grid_size_m=80.0)
rasterizer = SceneRasterizer(config)
grid = rasterizer.rasterize(scene)   # SceneContext → OccupancyGrid
```

The grid is ego-centric and aligned with the ego heading, making it invariant to absolute
world coordinates. Rasterization is fully differentiable — `jax.grad` flows through agent
positions and velocities.

## Model

`OccupancyFlowModel` wraps `opifex.neural.operators.fno.multiscale.MultiScaleFourierNeuralOperator`,
which operates on the top-down grid and predicts all T=8 future steps in one forward pass:

```python
from simulacrax.occupancy import OccupancyFlowConfig, create_occupancy_flow_model
from flax import nnx
import jax

config = OccupancyFlowConfig(grid_resolution=128)
model = create_occupancy_flow_model(config, nnx.Rngs(params=jax.random.key(0)))
prediction = model(grid)
# prediction.occupancy: (1, 8, 3, 128, 128) — sigmoid, ∈ [0, 1]
# prediction.flow:      (1, 8, 128, 128, 2) — vx, vy in m/s
```

The FNO is **resolution-independent**: the same model configuration works at 64×64
(fast CPU testing) and 256×256 (production).

## Physics-Informed Loss

`FlowConsistencyLoss` enforces the continuity equation as a soft constraint:

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$

```python
from simulacrax.occupancy import FlowConsistencyLoss, FlowConsistencyLossConfig

loss_fn = FlowConsistencyLoss(
    FlowConsistencyLossConfig(weight=0.1, dt=1.0, cell_size_m=config.cell_size_m)
)
physics_loss = loss_fn.compute(prediction)
total_loss = reconstruction_loss + physics_loss
```

The spatial derivatives follow the rasterizer's axis convention (ego-x on
grid rows, ego-y on grid columns) and use interior central differences —
the ego-centric grid is not periodic, so boundary cells are excluded from
the residual. `cell_size_m` has no default: derive it from the grid config
(`OccupancyFlowConfig.cell_size_m` or `RasterizerConfig.cell_size_m`) so
the normalisation matches the actual grid.

## Channel Layout

| Grid component | Shape | Content |
|---|---|---|
| `occupancy` | `(H, W, 3)` | Per-type soft-box occupancy: vehicle(0), pedestrian(1), cyclist(2) |
| `flow` | `(H, W, 2)` | Aggregate velocity field: vx(0), vy(1) in ego frame (m/s) |
| `map_features` | `(H, W, 3)` | Road(0), crosswalk(1), traffic_signal(2) |
