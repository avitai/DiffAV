# Production Deployment

## Overview

DiffAV supports these deployment paths:

- **Docker** — ship the complete Python runtime in a container for training, batch
  evaluation, and benchmarks. This is the supported production path today.
- **Direct JAX** — run from a Python process with GPU/TPU acceleration when a Python
  runtime is acceptable.
- **StableHLO export (experimental)** — lower a trajectory-diffusion step to a portable
  `.mlir` artifact via `jax.export`, as the foundation for future C++/on-device serving
  (IREE, OpenXLA). Today this exports a single denoise step of a default-configured model
  for shape/lowering validation; checkpoint-loaded export of the full `ScenarioMiner`
  pipeline is planned, not yet implemented.

---

## StableHLO Export (experimental)

`scripts/export_stablehlo.py` lowers the trajectory-diffusion denoise step
(`TrajectoryDiffusionModel.predict_noise`) to [StableHLO](https://openxla.org/stablehlo)
MLIR via `jax.export`, writing a `.mlir` artifact that any OpenXLA-compatible runtime
can consume.

!!! note "Scope"
    This is a lowering/portability scaffold, not a serving path yet. It exports a
    **single reverse-diffusion step** of a **freshly-initialised, default-configured**
    model — no checkpoint is loaded — which is enough to validate the StableHLO lowering.
    Exporting a trained checkpoint and the full `ScenarioMiner` sampling loop is planned
    future work; the C++ (nanobind) binding path is not implemented.

### Prerequisites

`jax.export` requires JAX >= 0.4.24, already satisfied by the pinned environment.

### Export command

```bash
# Writes artifacts/diffav_scenario_miner.mlir
uv run python scripts/export_stablehlo.py

# Choose a different output directory
uv run python scripts/export_stablehlo.py --output-dir /tmp/artifacts
```

The only flag is `--output-dir` (default `artifacts`). The exported model uses fixed
demonstration dimensions: 8 agents, 80 future steps, `hidden_dim=128`, `context_dim=128`.

### Expected output

```
... INFO ... Built TrajectoryDiffusionModel: hidden_dim=128, num_blocks=2, num_agents=8
... INFO ... Tracing predict_noise for StableHLO export ...
... INFO ... Serialising to StableHLO MLIR ...
StableHLO export written to: /.../artifacts/diffav_scenario_miner.mlir
```

The resulting `.mlir` can be compiled for a target device with `iree-compile`:

```bash
iree-compile artifacts/diffav_scenario_miner.mlir \
    --iree-input-type=stablehlo \
    --iree-hal-target-backends=cuda \
    -o artifacts/diffav_scenario_miner_cuda.vmfb
```

---

## Docker Deployment

The repository ships a GPU-capable (CPU-fallback) `Dockerfile` that installs JAX with CUDA
support (the `[gpu]` extra) and the project itself. It is a single runtime image for
training, evaluation, benchmarks, and tests — not a standalone inference server; the
container runs whatever entrypoint you pass it.

### Build

```bash
docker build -t diffav:latest .
```

The build installs dependencies from the pinned `uv.lock`, copies `src/`, `tests/`,
`scripts/`, `benchmarks/`, and `examples/`, installs the project, and verifies that
`diffav` imports on CPU so a broken image fails the build.

### Run

The default command runs the fast test suite; override it to run any entrypoint.

```bash
# Verify the GPU runtime
docker run --rm --gpus all diffav:latest \
    python -c "import diffav, jax; print(jax.devices())"

# Run the fast tests on CPU
docker run --rm -e JAX_PLATFORMS=cpu diffav:latest \
    python -m pytest tests/ -x -q -m "not slow"

# Train against a mounted checkpoint directory
docker run --gpus all \
    -e JAX_PLATFORMS=cuda \
    -v /local/checkpoints:/checkpoints \
    diffav:latest \
    python scripts/train_wod.py --help
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `JAX_PLATFORMS` | `cuda`, `tpu`, or `cpu`. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Preallocate GPU memory (image default: `false`). |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory to allocate (image default: `0.75`). |
| `DIFFAV_BACKEND` | Backend policy reported by `scripts/verify_diffav_gpu.py`. |

For managed GPU runs without a local card, `deploy/modal_app.py` launches the same
entrypoints on a Modal A100/H100 — see `deploy/README.md` in the repository.

---

## Performance Profiling

`TrainingMetrics` (returned by `TrajectoryTrainer.train_step()`) exposes a
`flops_per_step` field measuring one forward + backward pass. Enable it with
`TrainerConfig(profile_flops=True)` — the count is measured once via
calibrax's `FlopsCounter` and cached; with profiling off the field is `0.0`.

```python
from diffav.models.trainer import TrainerConfig, TrajectoryTrainer

trainer = TrajectoryTrainer(model, TrainerConfig(profile_flops=True))
metrics = trainer.train_step(trajectories, scene_context, key=jax.random.key(0))

print(f"Loss:          {metrics.total_loss:.4f}")
print(f"FLOPs / step:  {metrics.flops_per_step / 1e9:.1f} GFLOPs")
print(f"Grad norm:     {metrics.grad_norm:.4f}")
```

To profile end-to-end throughput use JAX's built-in profiler:

```python
import jax

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for batch in dataloader:
        metrics = trainer.train_step(batch, key=key)
```

Open the resulting Perfetto trace at <https://ui.perfetto.dev> to inspect
per-operation timings, memory usage, and XLA HLO graphs.

For multi-GPU / multi-host profiling see the
[Distributed Training guide](distributed.md).

---

## Next Steps

- [API Reference — ScenarioMiner](../api/api/scenario_miner.md) — full SDK documentation.
- [API Reference — EvaluationRunner](../api/evaluation/runner.md) — batch evaluation pipeline.
- [Distributed Training](distributed.md) — scale to multi-GPU and multi-host setups.
- [ScenarioMiner SDK guide](sdk.md) — generate, evaluate, and adversarially search scenarios.
