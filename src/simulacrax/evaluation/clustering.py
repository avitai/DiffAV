"""Representative-mode selection over oversampled trajectory rollouts.

SOTA trajectory models report ``minADE_K`` by drawing many rollouts and reducing
them to ``K`` representative modes before scoring (MotionDiffuser draws 64 and
greedily clusters to 6 weighted by assigned-sample count; MotionLM uses NMS plus
trajectory k-means). Those representatives approximate the model's *most probable*
modes, so the reducer must be **density-weighted** — keep the populated basins,
not the sparse extremes. Farthest-point sampling does the opposite (it maximises
spread, so it favours outliers and under-represents the dominant mode), which
makes ``minADE_K`` unrepresentative of what the benchmark rewards. This module
instead selects each mode by endpoint density with non-maximum suppression of
already-covered basins.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Endpoints closer than this (metres) count as the same behavioural mode when
# estimating density and suppressing already-selected basins — the trajectory-NMS
# distance threshold used by the reference reducers (e.g. MTR's NMS_DIST_THRESH).
_DEFAULT_NMS_RADIUS_M = 2.5


def select_representative_modes(
    rollouts: jax.Array, num_modes: int, *, nms_radius: float = _DEFAULT_NMS_RADIUS_M
) -> jax.Array:
    """Select ``num_modes`` representative rollouts per agent by endpoint density.

    Greedy density-with-suppression (the MotionLM NMS spirit): score each rollout
    by how many rollouts have an endpoint within ``nms_radius`` (a local density
    estimate), repeatedly take the densest not-yet-selected rollout, then
    suppress the rollouts within ``nms_radius`` of it so the next pick lands on a
    different basin. Unlike farthest-point sampling this keeps the most-populated
    (most probable) modes rather than the sparse extremes, so the retained set is
    representative of the model's distribution.

    Args:
        rollouts: Sampled rollouts, shape ``(num_rollouts, num_agents,
            future_steps, state_dim)``.
        num_modes: Number of representative modes ``K`` to keep. Clamped to the
            available rollout count.
        nms_radius: Endpoint distance (metres) within which rollouts count as the
            same mode for density and suppression.

    Returns:
        The selected rollouts, shape ``(min(num_modes, num_rollouts), num_agents,
        future_steps, state_dim)`` (a subset of the inputs, per agent).
    """
    num_rollouts = rollouts.shape[0]
    num_agents = rollouts.shape[1]
    modes = min(num_modes, num_rollouts)
    endpoints = jnp.swapaxes(rollouts[:, :, -1, :2], 0, 1)  # (num_agents, num_rollouts, 2)

    def representative_indices(points: jax.Array) -> jax.Array:
        """Indices of ``modes`` density-selected rollouts ``(modes,)``."""
        distances = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        within = (distances < nms_radius).astype(jnp.float32)  # (num_rollouts, num_rollouts)
        density = within.sum(axis=-1)  # (num_rollouts,)
        # A suppression larger than any density pushes a covered basin below every
        # un-covered rollout, so the next pick prefers a fresh mode; an exhausted
        # set still yields a distinct index because picks are excluded outright.
        suppression = density.max() + 1.0
        available = density
        selected = []
        for _ in range(modes):
            index = jnp.argmax(available)
            selected.append(index)
            available = available - suppression * within[index]
            available = available.at[index].set(-jnp.inf)
        return jnp.stack(selected)

    indices = jax.vmap(representative_indices)(endpoints)  # (num_agents, modes)
    gathered = jax.vmap(lambda agent: rollouts[indices[agent], agent])(jnp.arange(num_agents))
    return jnp.swapaxes(gathered, 0, 1)  # (modes, num_agents, future_steps, state_dim)
