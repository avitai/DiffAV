# =============================================================================
# DiffAV — GPU-capable, CPU-fallback runtime image
# =============================================================================
# One image for training, evaluation, benchmarks, and tests. GPU support comes
# from JAX's pip-managed CUDA wheels (the [gpu] extra); on hosts without a GPU
# the same image runs on CPU (set JAX_PLATFORMS=cpu to force it).
#
# Build:  docker build -t diffav:latest .
# Run:    docker run --rm --gpus all diffav:latest \
#           python -c "import diffav, jax; print(jax.devices())"
# Test:   docker run --rm -e JAX_PLATFORMS=cpu diffav:latest \
#           python -m pytest tests/ -x -q -m "not slow"
# =============================================================================

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# JAX runtime defaults — prevent full GPU memory preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Ubuntu 24.04 ships Python 3.12 natively; git is required for the
# git-sourced dependencies in the lockfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-dev \
  git \
  curl \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install uv — pinned for reproducible builds. Update intentionally.
RUN curl -LsSf https://astral.sh/uv/0.9.28/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Use the system interpreter: a uv-managed Python would live under /root,
# unreadable by the non-root runtime user.
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON=/usr/bin/python3

# The uv cache dies with each RUN layer anyway; disabling it halves the
# peak disk usage of the dependency layer (the CUDA wheels are large).
ENV UV_NO_CACHE=1

WORKDIR /app

# --- Layer 1: dependencies (cached unless pyproject.toml / uv.lock change) ---
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --extra gpu --no-install-project

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# --- Layer 2: source (changes frequently, invalidates only these layers) ---
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY benchmarks ./benchmarks
COPY examples ./examples

# Install the project itself now that the source is present
RUN uv sync --frozen --extra gpu

# Verify the package imports (CPU works on GPU-less build hosts)
RUN JAX_PLATFORMS=cpu python -c "import diffav, jax; print(f'JAX {jax.__version__} OK')"

# Non-root runtime user
RUN useradd --create-home diffav && chown -R diffav:diffav /app
USER diffav

# Default command — overridable at runtime
CMD ["python", "-m", "pytest", "tests/", "-x", "-q", "-m", "not slow"]
