"""Tests for representative-mode selection over oversampled rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simulacrax.evaluation.clustering import select_representative_modes


def _rollouts_with_endpoints(endpoints_x: jax.Array) -> jax.Array:
    """One agent, ``len(endpoints_x)`` rollouts, each ending at the given x."""
    count = endpoints_x.shape[0]
    return jnp.zeros((count, 1, 2, 4)).at[:, 0, -1, 0].set(endpoints_x)


def test_reduces_rollout_count_to_num_modes() -> None:
    """Selection returns exactly ``num_modes`` representatives per agent."""
    rollouts = jax.random.normal(jax.random.key(0), (32, 3, 5, 4))
    modes = select_representative_modes(rollouts, num_modes=6)
    assert modes.shape == (6, 3, 5, 4)


def test_selected_modes_are_input_rollouts() -> None:
    """Every selected mode is one of the input rollouts (selection, not synthesis)."""
    rollouts = jax.random.normal(jax.random.key(1), (16, 2, 4, 4))
    modes = select_representative_modes(rollouts, num_modes=4)
    for agent in range(rollouts.shape[1]):
        for mode in range(modes.shape[0]):
            matches = jnp.all(jnp.isclose(rollouts[:, agent], modes[mode, agent]), axis=(-1, -2))
            assert bool(jnp.any(matches))


def test_keeps_dense_modes_over_sparse_outliers() -> None:
    """Density weighting keeps the populated basins, unlike farthest-point
    sampling which would retain a sparse outlier for its spread.

    Two dense modes at x=0 and x=10 (10 rollouts each) plus a sparse outlier at
    x=50 (2 rollouts): selecting two modes must pick the two dense basins, never
    the outlier.
    """
    endpoints_x = jnp.concatenate([jnp.zeros(10), jnp.full(10, 10.0), jnp.full(2, 50.0)])
    modes = select_representative_modes(_rollouts_with_endpoints(endpoints_x), num_modes=2)
    selected_x = modes[:, 0, -1, 0]
    # Both representatives are the dense modes (|x| <= 10), not the x=50 outlier.
    assert float(jnp.max(jnp.abs(selected_x))) < 20.0


def test_covers_distinct_basins() -> None:
    """Two well-separated basins are each represented (near-duplicates collapse).

    Endpoints at x = 0, 0.01, 10, 10.01: the two modes straddle the gap.
    """
    endpoints_x = jnp.array([0.0, 0.01, 10.0, 10.01])
    modes = select_representative_modes(_rollouts_with_endpoints(endpoints_x), num_modes=2)
    selected_x = jnp.sort(modes[:, 0, -1, 0])
    assert float(selected_x[0]) < 1.0
    assert float(selected_x[1]) > 9.0


def test_more_modes_than_rollouts_is_clamped() -> None:
    """Requesting more modes than rollouts returns at most the rollout count."""
    rollouts = jax.random.normal(jax.random.key(2), (3, 2, 4, 4))
    modes = select_representative_modes(rollouts, num_modes=6)
    assert modes.shape[0] == 3


def test_jit_and_vmap_compatible() -> None:
    """Selection composes with jit."""
    rollouts = jax.random.normal(jax.random.key(3), (8, 2, 4, 4))
    jitted = jax.jit(lambda r: select_representative_modes(r, num_modes=3))(rollouts)
    assert jitted.shape == (3, 2, 4, 4)
