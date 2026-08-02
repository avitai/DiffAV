"""Root conftest: TF CPU isolation and shared fixtures.

JAX selects the best available backend automatically: GPU when the ``gpu``
extra is installed (``./setup.sh`` handles this on CUDA hosts) and CPU
otherwise. ``[tool.pytest-env]`` in pyproject.toml only supplies defaults;
``source ./activate.sh`` layers the generated ``.simulacrax.env`` plus any
user-owned ``.env``/``.env.data`` overrides on top.
"""

from __future__ import annotations

import os


# Force TF to CPU-only BEFORE any TF import to prevent GPU memory conflicts with JAX.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# JAX device detection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jax_backend() -> str:
    """Return the active JAX backend name (cpu, gpu, tpu)."""
    return jax.default_backend()


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Return True if a GPU device is available."""
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng_key() -> jax.Array:
    """Provide a deterministic JAX PRNG key for reproducible tests."""
    return jax.random.key(42)


# ---------------------------------------------------------------------------
# TF GPU isolation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _isolate_tf_to_cpu() -> None:
    """Ensure TensorFlow only uses CPU devices.

    WOD proto parsing requires TF but must not compete with JAX for GPU memory.
    """
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
    except (ImportError, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Sample domain fixtures (used across test modules)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_position() -> jax.Array:
    """Sample 2D position array (x, y)."""
    return jnp.array([10.0, 20.0], dtype=jnp.float32)


@pytest.fixture()
def sample_velocity() -> float:
    """Sample scalar velocity (m/s)."""
    return 5.0


@pytest.fixture()
def sample_heading() -> float:
    """Sample heading angle (radians)."""
    return 1.57  # ~90 degrees
