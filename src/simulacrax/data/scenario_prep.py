"""Scenario preparation shared by training and examples.

Extracts fixed-size ``(trajectories, current-state)`` arrays from a raw
WOD-style state dict. Two extraction policies live here:

* :func:`prepare_full_horizon_scene` keeps only agents valid across the
  entire future horizon (strict, drops partially-tracked scenarios).
* :func:`prepare_padded_scene` keeps the top agents that are valid at the
  current step (so each has a defined reference pose), ranked by valid-step
  count, and zero-pads the rest, returning a per-step validity mask so the loss
  can be masked-averaged. Per-step masking mirrors the official WOD motion
  tutorial (weighting by ``gt_future_is_valid`` rather than filtering to
  fully-valid agents); the current-step requirement additionally gives every
  kept agent a well-defined local frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# Sentinel written into trajectory/current channels at padded (invalid)
# agent-step positions. Chosen to be an obviously out-of-range coordinate so
# a leak into a valid slot is detectable; the validity mask — never these
# values — is what the masked loss consumes.
PAD_SENTINEL: float = -1.0


def prepare_full_horizon_scene(
    raw: dict,
    *,
    num_agents: int,
    history_steps: int,
    future_steps: int,
) -> tuple[jax.Array, jax.Array] | None:
    """Extract (trajectories, current states) for fully-valid agents.

    Args:
        raw: WOD-style state dict with ``state/all/{x, y, bbox_yaw,
            velocity_x, velocity_y, valid}`` arrays of shape
            ``(num_raw_agents, total_steps)``.
        num_agents: Exact number of agents to select. Scenarios with fewer
            full-horizon-valid agents are skipped.
        history_steps: Number of history steps preceding the future slice;
            the current step is ``history_steps - 1``.
        future_steps: Length of the future horizon to extract.

    Returns:
        ``(trajectories, current)`` where ``trajectories`` has shape
        ``(num_agents, future_steps, 4)`` with channels
        ``[x, y, heading, speed]`` and ``current`` has shape
        ``(num_agents, 4)`` with ``[x, y, vx, vy]`` at the current step —
        or ``None`` if the scenario lacks enough full-horizon-valid agents.

    Raises:
        ValueError: If ``num_agents`` is not positive.
    """
    if num_agents <= 0:
        msg = f"num_agents must be positive, got {num_agents}"
        raise ValueError(msg)

    future = (slice(None, None), slice(history_steps, history_steps + future_steps))
    valid = np.asarray(raw["state/all/valid"])[future] > 0
    rows = np.flatnonzero(valid.all(axis=1))[:num_agents]
    if rows.size < num_agents:
        return None

    x = np.asarray(raw["state/all/x"])[future][rows]
    y = np.asarray(raw["state/all/y"])[future][rows]
    heading = np.asarray(raw["state/all/bbox_yaw"])[future][rows]
    speed = np.hypot(
        np.asarray(raw["state/all/velocity_x"])[future][rows],
        np.asarray(raw["state/all/velocity_y"])[future][rows],
    )
    trajectories = jnp.asarray(np.stack([x, y, heading, speed], axis=-1), dtype=jnp.float32)

    current_step = history_steps - 1
    current = jnp.asarray(
        np.stack(
            [
                np.asarray(raw["state/all/x"])[rows, current_step],
                np.asarray(raw["state/all/y"])[rows, current_step],
                np.asarray(raw["state/all/velocity_x"])[rows, current_step],
                np.asarray(raw["state/all/velocity_y"])[rows, current_step],
            ],
            axis=-1,
        ),
        dtype=jnp.float32,
    )
    return trajectories, current


def prepare_padded_scene(
    raw: dict,
    *,
    num_agents: int,
    history_steps: int,
    future_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None:
    """Extract padded ``(trajectories, validity, current)`` for masked loss.

    Selects the ``num_agents`` agents that are valid at the current step (each
    with at least one valid future step), ranked by valid-step count, and
    zero-pads remaining slots so the output is fixed-shape for every scenario.
    Agents absent at the current step have no defined reference pose — their
    future would be a large ego-frame offset rather than a small local
    displacement — so they are excluded, not framed at the identity pose.
    Invalid agent-step positions are filled
    with :data:`PAD_SENTINEL`; the returned validity mask — not the sentinel —
    is what the masked diffusion loss consumes. This is the WOD-tutorial
    ``sample_weight`` policy: keep partially-tracked agents and mask per step,
    rather than discarding scenarios that lack ``num_agents`` fully-valid
    agents.

    Args:
        raw: WOD-style state dict with ``state/all/{x, y, bbox_yaw,
            velocity_x, velocity_y, valid}`` arrays of shape
            ``(num_raw_agents, total_steps)``.
        num_agents: Number of agent slots in the fixed-shape output.
        history_steps: Number of history steps preceding the future slice;
            the current step is ``history_steps - 1``.
        future_steps: Length of the future horizon to extract.

    Returns:
        ``(trajectories, validity, reference_pose, agent_rows)`` where
        ``trajectories`` has shape ``(num_agents, future_steps, 4)`` with
        channels ``[x, y, heading, speed]``, ``validity`` has shape
        ``(num_agents, future_steps)`` (1.0 valid, 0.0 padded), ``reference_pose``
        has shape ``(num_agents, 3)`` with ``[x, y, heading]`` at the current step
        (the per-agent local-frame origin for
        :func:`~simulacrax.core.geometry.to_agent_frame`; only padded slots keep
        the identity pose ``[0, 0, 0]`` — every selected agent is valid at the
        current step), and
        ``agent_rows`` has shape ``(num_agents,)`` int32 — the source-agent index
        selected for each output slot (padded slots repeat index 0, whose
        trajectory is masked). ``agent_rows`` lets a caller gather the matching
        per-agent embedding from a scene encoder. Returns ``None`` if no agent
        has any valid future step.

    Raises:
        ValueError: If ``num_agents`` is not positive.
    """
    if num_agents <= 0:
        msg = f"num_agents must be positive, got {num_agents}"
        raise ValueError(msg)

    current_step = history_steps - 1
    future = (slice(None, None), slice(history_steps, history_steps + future_steps))
    valid_all = np.asarray(raw["state/all/valid"]) > 0
    valid = valid_all[future]  # (num_raw, future_steps)
    valid_counts = valid.sum(axis=1)
    # Keep only agents valid at the current step -- so each has a defined
    # reference pose and its future is a small local displacement rather than a
    # large ego-frame offset -- that also have at least one valid future step.
    # Rank the eligible agents by valid-step count (stable top-k), without
    # requiring the scenario to fill every slot.
    eligible = (valid_counts > 0) & valid_all[:, current_step]
    ranked = np.argsort(-valid_counts, kind="stable")
    rows = ranked[eligible[ranked]][:num_agents]
    if rows.size == 0:
        return None

    trajectories = np.full((num_agents, future_steps, 4), PAD_SENTINEL, dtype=np.float32)
    validity = np.zeros((num_agents, future_steps), dtype=np.float32)
    reference_pose = np.zeros((num_agents, 3), dtype=np.float32)

    row_valid = valid[rows]  # (selected, future_steps)
    speed = np.hypot(
        np.asarray(raw["state/all/velocity_x"])[future][rows],
        np.asarray(raw["state/all/velocity_y"])[future][rows],
    )
    stacked = np.stack(
        [
            np.asarray(raw["state/all/x"])[future][rows],
            np.asarray(raw["state/all/y"])[future][rows],
            np.asarray(raw["state/all/bbox_yaw"])[future][rows],
            speed,
        ],
        axis=-1,
    ).astype(np.float32)
    selected = rows.size
    # Write valid entries only; padded steps keep the sentinel and mask 0.
    mask_3d = row_valid[..., None]
    trajectories[:selected] = np.where(mask_3d, stacked, PAD_SENTINEL)
    validity[:selected] = row_valid.astype(np.float32)

    # Every selected agent is valid at the current step (enforced by ``eligible``
    # above), so its reference pose is always defined; padded slots keep the
    # zero-initialized identity pose.
    reference_pose[:selected] = np.stack(
        [
            np.asarray(raw["state/all/x"])[rows, current_step],
            np.asarray(raw["state/all/y"])[rows, current_step],
            np.asarray(raw["state/all/bbox_yaw"])[rows, current_step],
        ],
        axis=-1,
    ).astype(np.float32)

    # Source-agent index per output slot; padded slots repeat 0 (masked).
    agent_rows = np.zeros(num_agents, dtype=np.int32)
    agent_rows[:selected] = rows

    return (
        jnp.asarray(trajectories),
        jnp.asarray(validity),
        jnp.asarray(reference_pose),
        jnp.asarray(agent_rows),
    )
