# Installation

## Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- Git

## Standard Installation

The setup script auto-detects the host (NVIDIA GPU, Apple Silicon, or CPU),
creates the uv-managed environment, and writes the generated backend file
`.diffav.env` consumed by `source ./activate.sh`:

=== "Linux (CUDA)"

    ```bash
    # Clone the repository
    git clone https://github.com/avitai/DiffAV.git
    cd DiffAV

    # Auto-detects the NVIDIA GPU and installs the CUDA 12 backend
    ./setup.sh
    ```

=== "Linux (CPU only)"

    ```bash
    git clone https://github.com/avitai/DiffAV.git
    cd DiffAV

    ./setup.sh --backend cpu
    ```

=== "macOS (Metal)"

    ```bash
    git clone https://github.com/avitai/DiffAV.git
    cd DiffAV

    ./setup.sh --backend metal
    ```

## Manual Installation with uv

`uv sync` installs the runtime dependencies plus the default `dev` dependency
group (lint, test, and docs tooling). Hardware backends are extras:

```bash
# CPU-only development environment
uv sync

# Linux with NVIDIA GPU — JAX's pip-managed CUDA 12 runtime
uv sync --extra gpu

# macOS with Metal
uv sync --extra metal
```

## GPU Setup

DiffAV uses JAX for GPU computation and TensorFlow (CPU-only) for WOD proto parsing.
TensorFlow's GPU visibility is disabled at import time to prevent memory conflicts.

### Verify JAX GPU Access

```python
import jax
print(jax.devices())
# [CudaDevice(id=0)]
```

### Verify TensorFlow CPU Confinement

```python
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
print(tf.config.get_visible_devices("GPU"))
# []  (empty -- GPU hidden from TF)
```

## Troubleshooting

### CUDA Version Mismatch

If JAX fails to detect your GPU, ensure your CUDA version matches the installed
`jaxlib`. Check with:

```bash
nvidia-smi          # Shows driver CUDA version
python -c "import jax; print(jax.devices())"
```

### TensorFlow GPU Conflict

If you see TensorFlow allocating GPU memory, ensure `tf.config.set_visible_devices([], "GPU")`
is called **before** any TF operations. The `WODSource` class handles this automatically.

### Environment Configuration

DiffAV uses a `.env` file for dataset paths. Copy the template and edit:

```bash
cp .env.example .env
```

Set `WOD_MOTION_TFRECORD_PATH` to your local Waymo Open Dataset TFRecord directory:

```bash
# .env
WOD_MOTION_TFRECORD_PATH=/path/to/waymo/motion_v1.2.1/tf_example
```

The directory should contain `training/`, `validation/`, and/or `testing/` subdirectories
with `.tfrecord` files. See the [Quick Start](quickstart.md) for download instructions.

### Sister Repo Installation

DiffAV depends on four sister repositories, pinned to their latest
release tags via `[tool.uv.sources]` in `pyproject.toml` (the artifex
distribution is published as `avitai-artifex`; imports stay `artifex`):

```bash
# These are installed automatically by uv sync. Manual install if needed:
uv pip install "datarax @ git+https://github.com/avitai/datarax@v0.1.4"
uv pip install "avitai-artifex @ git+https://github.com/avitai/artifex@v0.1.2"
uv pip install "opifex @ git+https://github.com/avitai/opifex@v0.2.0"
uv pip install "calibrax @ git+https://github.com/avitai/calibrax@v0.1.1"
```
