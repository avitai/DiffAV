"""JAX ↔ WOD format conversion utilities.

Converts between Simulacrax domain types and the dict-based WOD
submission format matching the ``SimulatedTrajectory`` message in
``waymo_open_dataset/protos/sim_agents_submission.proto``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulacrax.core.types import SceneContext, TrajectoryPrediction
from simulacrax.data.parsers import parse_scenario


def _validate_per_step_field(
    name: str,
    value: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Check that a per-agent per-step field matches the trajectory shape.

    Args:
        name: Field name for the error message.
        value: Candidate array.
        expected_shape: Required (num_agents, future_steps) shape.

    Returns:
        The value as a numpy array.

    Raises:
        ValueError: If the shape does not match.
    """
    array = np.asarray(value)
    if array.shape != expected_shape:
        msg = f"{name} must have shape {expected_shape}, got {array.shape}"
        raise ValueError(msg)
    return array


def to_wod_submission(
    prediction: TrajectoryPrediction,
    *,
    center_z: np.ndarray | None = None,
    validity: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert a TrajectoryPrediction to WOD submission format.

    Produces a dict matching the JointScene proto structure with
    SimulatedTrajectory entries carrying the full per-step field set:
    ``center_x``, ``center_y``, ``center_z``, ``heading``, ``valid``,
    plus ``object_id`` per agent. The proto carries no velocity fields;
    the prediction's velocity channel does not appear in submissions.

    The state_dim layout is [x, y, heading, velocity].

    Args:
        prediction: Predicted trajectories, shape (num_agents, future_steps, 4).
        center_z: Optional per-step elevation, shape (num_agents, future_steps).
            Defaults to zeros — the trajectory models predict planar motion.
        validity: Optional per-step validity mask, shape
            (num_agents, future_steps). Defaults to all-valid, matching the
            proto contract that submitted objects are simulated for the whole
            horizon.

    Returns:
        Dict with 'simulated_trajectories' list of per-agent dicts.

    Raises:
        ValueError: If ``center_z`` or ``validity`` does not match the
            trajectory's (num_agents, future_steps) shape.
    """
    trajectories_np = np.asarray(prediction.trajectories)
    num_agents, future_steps = trajectories_np.shape[0], trajectories_np.shape[1]
    per_step_shape = (num_agents, future_steps)

    if center_z is None:
        center_z_np = np.zeros(per_step_shape, dtype=np.float32)
    else:
        center_z_np = _validate_per_step_field("center_z", center_z, per_step_shape)

    if validity is None:
        validity_np = np.ones(per_step_shape, dtype=bool)
    else:
        validity_np = _validate_per_step_field("validity", validity, per_step_shape).astype(bool)

    simulated_trajectories = []
    for i in range(num_agents):
        agent_traj = trajectories_np[i]  # [future_steps, 4]
        simulated_trajectories.append(
            {
                "center_x": agent_traj[:, 0].tolist(),
                "center_y": agent_traj[:, 1].tolist(),
                "center_z": center_z_np[i].tolist(),
                "heading": agent_traj[:, 2].tolist(),
                "valid": validity_np[i].tolist(),
                "object_id": prediction.agent_ids[i],
            }
        )

    return {"simulated_trajectories": simulated_trajectories}


def from_wod_scenario(raw: dict) -> SceneContext:
    """Convert a raw WOD scenario dict to a SceneContext.

    This is a convenience wrapper around parse_scenario() providing
    a symmetric API alongside to_wod_submission().

    Args:
        raw: Raw WOD scenario dict with Waymax-format keys.

    Returns:
        Parsed SceneContext with ego, agents, map, and timestamps.
    """
    return parse_scenario(raw)
