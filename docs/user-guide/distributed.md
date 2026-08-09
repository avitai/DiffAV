# Distributed Training

DiffAV supports data-parallel training across multiple JAX devices via a
thin mesh-sharding layer in `diffav.core.distributed`.  The same code runs
on a single CPU, a single GPU, or a multi-GPU pod — single-device behaviour is
always the silent default.

---

## When to use distributed training

Use `train_step_distributed()` when:

- You have **multiple GPUs or TPU cores** and want to shard the batch across
  them with data-parallel replication (DDP).
- You want single-program multi-data (SPMD) execution with XLA handling
  parameter replication and gradient all-reduce automatically.

On a single device `train_step_distributed()` is functionally identical to
`train_step()` — the mesh collapses to a trivial `(1, 1, 1)` configuration.

---

## DistributedConfig options

```python
from diffav.core import DistributedConfig

# Default — single-device, data-parallel
config = DistributedConfig()

# Explicit 8-device data-parallel across a single node
config = DistributedConfig(mesh_shape=(8, 1, 1))

# 2 nodes × 4 GPUs — shard batch across 8 devices in data dimension
config = DistributedConfig(mesh_shape=(8, 1, 1))
```

| Field | Default | Description |
|-------|---------|-------------|
| `mesh_shape` | `(1, 1, 1)` | `(data, model, pipeline)` device counts.  Product must equal `jax.device_count()`. |
| `sharding_strategy` | `"ddp"` | Only `"ddp"` is supported.  `"fsdp"` and `"mp"` raise `NotImplementedError`. |
| `axis_names` | `("data", "model", "pipeline")` | Labels for the three mesh axes. |

---

## Data-parallel (DDP) vs. model-parallel trade-offs

| Strategy | When to use | Limitation |
|----------|-------------|------------|
| **DDP** (default) | Batch is large; model fits on one device | Each device holds a full model replica |
| **FSDP** (future) | Model is too large for one device | Higher communication overhead |
| **MP / tensor-parallel** (future) | Very deep models requiring layer splitting | Complex partitioning logic |

The distributed layer implements **DDP only**.  The `"fsdp"` and `"mp"` identifiers are
reserved for future sprints.

---

## Usage with TrajectoryTrainer

```python
import jax
import jax.numpy as jnp
from flax import nnx

from diffav.core import DistributedConfig, create_device_mesh
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer
from diffav.models.trajectory_diffusion import (
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)

# Build model and trainer as usual
model_config = TrajectoryDiffusionConfig(
    hidden_dim=256,
    num_blocks=4,
    num_temporal_layers=2,
    num_social_layers=1,
    num_heads=8,
    future_steps=80,
    num_agents_max=32,
    num_timesteps=1000,
    context_dim=128,
    state_dim=4,
)
model = TrajectoryDiffusionModel(model_config, rngs=nnx.Rngs(params=jax.random.key(0)))
trainer = TrajectoryTrainer(model, TrainerConfig())

# Create mesh once — reuse across steps
dist_config = DistributedConfig()  # or DistributedConfig(mesh_shape=(8, 1, 1))
mesh = create_device_mesh(dist_config)

# Training loop
key = jax.random.key(42)
for step, (trajectories, scene_context) in enumerate(dataloader):
    step_key = jax.random.fold_in(key, step)
    metrics = trainer.train_step_distributed(
        trajectories,
        scene_context,
        key=step_key,
        mesh=mesh,
    )
    print(f"step={metrics.step}  loss={metrics.total_loss:.4f}  "
          f"flops={metrics.flops_per_step:.2e}")
```

---

## Usage with DPOAlignmentTrainer

```python
import jax
import optax
from flax import nnx

from diffav.core import DistributedConfig, create_device_mesh
from diffav.alignment.dpo_trainer import DPOAlignmentConfig, DPOAlignmentTrainer

optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)
dpo_trainer = DPOAlignmentTrainer(
    model,
    optimizer,
    config=DPOAlignmentConfig(reference_free=True),
)

dist_config = DistributedConfig()
mesh = create_device_mesh(dist_config)

key = jax.random.key(0)
for step, batch in enumerate(preference_dataloader):
    step_key = jax.random.fold_in(key, step)
    metrics = dpo_trainer.train_step_distributed(batch, step_key, mesh=mesh)
    print(f"dpo_loss={metrics.dpo_loss:.4f}  accuracy={metrics.reward_accuracy:.2%}")
```

---

## Single-device fallback behaviour

When `jax.device_count() == 1`, `create_device_mesh()` always returns a
`(1, 1, 1)` trivial mesh regardless of `mesh_shape`.  All `shard_batch()`
calls become `jax.device_put()` calls with a single-device placement — the
values are unchanged and no data is moved.

This means the exact same training code works in:

- Local CPU development (`jax.device_count() == 1`)
- Single-GPU cloud instances
- Multi-GPU pods (sharding takes effect)

No conditional branching is needed in application code.

---

## FLOP counting with calibrax

Enable `TrainerConfig(profile_flops=True)` to measure per-step FLOPs with
`calibrax.profiling.FlopsCounter`. The step function is traced once on the
first step, the count is cached, and every `TrainingMetrics.flops_per_step`
reports it. With profiling off (the default) the field is an honest `0.0`.

```python
trainer = TrajectoryTrainer(model, TrainerConfig(profile_flops=True))

metrics = trainer.train_step_distributed(trajectories, scene_context, key=key, mesh=mesh)
print(f"FLOPs this step: {metrics.flops_per_step:.2e}")
```
