# Production Deployment

## Overview

Simulacrax supports three primary deployment paths:

- **StableHLO export** — compile the `ScenarioMiner` model to a portable `.mlir` artifact
  for integration into C++/on-device inference stacks (TensorFlow Serving, IREE, OpenXLA).
- **Docker** — ship the complete Python runtime in a container for training, batch
  evaluation, and benchmarks.
- **Direct JAX** — run from a Python process with GPU/TPU acceleration when a Python
  runtime is acceptable.

Choose StableHLO when you need maximum portability or sub-millisecond C++ serving.
Choose Docker when you want a fully managed runtime with all GPU dependencies pre-configured.

---

## StableHLO Export

`scripts/export_stablehlo.py` compiles the `ScenarioMiner` diffusion backbone to
[StableHLO](https://openxla.org/stablehlo) MLIR, producing a `.mlir` file that can be
ingested by any OpenXLA-compatible runtime.

### Prerequisites

```bash
# StableHLO export requires jaxlib >= 0.4.25
uv pip install "jaxlib>=0.4.25"
```

### Export command

```bash
uv run python scripts/export_stablehlo.py \
    --checkpoint path/to/checkpoint \
    --output     artifacts/scenario_miner.mlir \
    --max-agents 32 \
    --horizon    80 \
    --context-dim 128
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | _(required)_ | Path to a `SimulacraxCheckpointManager` directory. |
| `--output` | `scenario_miner.mlir` | Output `.mlir` path. |
| `--max-agents` | `32` | Maximum agents the exported model supports. |
| `--horizon` | `80` | Future timesteps (must match training config). |
| `--context-dim` | `128` | Scene context embedding dimension. |
| `--diffusion-steps` | `10` | Reverse-diffusion steps baked into the export. |

### Expected output

```
[INFO] Loading checkpoint: path/to/checkpoint (step=25000)
[INFO] Tracing ScenarioMiner.sample() with max_agents=32, horizon=80
[INFO] Lowering to StableHLO...
[INFO] Written 4.2 MB → artifacts/scenario_miner.mlir
```

The resulting `.mlir` can be compiled for a target device with `iree-compile` or fed
directly to a TensorFlow Serving instance:

```bash
iree-compile artifacts/scenario_miner.mlir \
    --iree-input-type=stablehlo \
    --iree-hal-target-backends=cuda \
    -o artifacts/scenario_miner_cuda.vmfb
```

---

## Docker Deployment

The repository ships a GPU-capable (CPU-fallback) `Dockerfile` that installs JAX with CUDA
support (the `[gpu]` extra) and the project itself. It is a single runtime image for
training, evaluation, benchmarks, and tests — not a standalone inference server; the
container runs whatever entrypoint you pass it.

### Build

```bash
docker build -t simulacrax:latest .
```

The build installs dependencies from the pinned `uv.lock`, copies `src/`, `tests/`,
`scripts/`, `benchmarks/`, and `examples/`, installs the project, and verifies that
`simulacrax` imports on CPU so a broken image fails the build.

### Run

The default command runs the fast test suite; override it to run any entrypoint.

```bash
# Verify the GPU runtime
docker run --rm --gpus all simulacrax:latest \
    python -c "import simulacrax, jax; print(jax.devices())"

# Run the fast tests on CPU
docker run --rm -e JAX_PLATFORMS=cpu simulacrax:latest \
    python -m pytest tests/ -x -q -m "not slow"

# Train against a mounted checkpoint directory
docker run --gpus all \
    -e JAX_PLATFORMS=cuda \
    -v /local/checkpoints:/checkpoints \
    simulacrax:latest \
    python scripts/train_wod.py --help
```

Environment variables:

| Variable | Description |
|----------|-------------|
| `JAX_PLATFORMS` | `cuda`, `tpu`, or `cpu`. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Preallocate GPU memory (image default: `false`). |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory to allocate (image default: `0.75`). |
| `SIMULACRAX_BACKEND` | Backend policy reported by `scripts/verify_simulacrax_gpu.py`. |

For managed GPU runs without a local card, `deploy/modal_app.py` launches the same
entrypoints on a Modal A100/H100 — see `deploy/README.md` in the repository.

---

## Performance Profiling

`TrainingMetrics` (returned by `TrajectoryTrainer.train_step()`) exposes a
`flops_per_step` field measuring one forward + backward pass. Enable it with
`TrainerConfig(profile_flops=True)` — the count is measured once via
calibrax's `FlopsCounter` and cached; with profiling off the field is `0.0`.

```python
from simulacrax.models.trainer import TrainerConfig, TrajectoryTrainer

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
