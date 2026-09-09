# Distributed Training

DiffAV trains data-parallel across every visible JAX device through
[substrax](https://pypi.org/project/substrax/), the device, mesh and sharding
layer the Avitai packages share. The same code runs on a single CPU, a single
GPU, or a multi-GPU host: a one-device mesh makes every placement a no-op.

---

## When to use distributed training

Use `train_step_distributed()` when:

- You have **multiple GPUs or TPU cores** and want to shard the batch across
  them with data-parallel replication.
- You want single-program multi-data (SPMD) execution with XLA handling
  parameter replication and gradient all-reduce automatically.

On a single device `train_step_distributed()` is functionally identical to
`train_step()`.

---

## Building the mesh

`DeviceMeshManager.create_device_mesh` takes the mesh shape as a mapping from
axis name to size; the product must equal the number of devices it is built
over (every visible device by default). Its axes are `Auto`, which is what lets
XLA infer the data-parallel sharding of the backward pass on jax 0.11.

```python
import jax
from substrax.mesh import DeviceMeshManager

# Every visible device on the data axis
mesh = DeviceMeshManager.create_device_mesh({"data": jax.device_count()})
```

`substrax.spmd.create_data_parallel_sharding(mesh)` is the `NamedSharding`
that partitions a leading batch axis over `"data"`, and
`substrax.spmd.place_batch_on_shards(batch, sharding)` places a pytree of
arrays with it. The trainers accept batches placed this way, and equally a
batch on the default device: inside `jax.set_mesh(mesh)`, XLA's partitioner
moves the data as the mesh requires.

---

## Usage with TrajectoryTrainer

```python
import jax
import jax.numpy as jnp
from flax import nnx
from substrax.mesh import DeviceMeshManager

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

# Create the mesh once and reuse it across steps
mesh = DeviceMeshManager.create_device_mesh({"data": jax.device_count()})

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
from substrax.mesh import DeviceMeshManager

from diffav.alignment.dpo_trainer import DPOAlignmentConfig, DPOAlignmentTrainer

optimizer = nnx.Optimizer(model, optax.adam(1e-5), wrt=nnx.Param)
dpo_trainer = DPOAlignmentTrainer(
    model,
    optimizer,
    config=DPOAlignmentConfig(reference_free=True),
)

mesh = DeviceMeshManager.create_device_mesh({"data": jax.device_count()})

key = jax.random.key(0)
for step, batch in enumerate(preference_dataloader):
    step_key = jax.random.fold_in(key, step)
    metrics = dpo_trainer.train_step_distributed(batch, step_key, mesh=mesh)
    print(f"dpo_loss={metrics.dpo_loss:.4f}  accuracy={metrics.reward_accuracy:.2%}")
```

---

## Beyond data parallelism

substrax also carries the FSDP, tensor-parallel and pipeline-parallel
strategies (`substrax.mesh.strategies`) and the cross-device collectives
(`substrax.spmd`). The DiffAV trainers use data parallelism only; a model too
large for one device is a modelling change, not a mesh change.

---

## FLOP counting with calibrax

Enable `TrainerConfig(profile_flops=True)` to measure per-step FLOPs with
`calibrax.profiling.FlopsCounter`. The step function is lowered once on the
first step, XLA's cost analysis of it is cached, and every
`TrainingMetrics.flops_per_step` reports it. With profiling off (the default)
the field is an honest `0.0`.

```python
trainer = TrajectoryTrainer(model, TrainerConfig(profile_flops=True))

metrics = trainer.train_step_distributed(trajectories, scene_context, key=key, mesh=mesh)
print(f"FLOPs this step: {metrics.flops_per_step:.2e}")
```
