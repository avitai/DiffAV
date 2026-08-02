"""AV-specific preprocessing operators for scene tokenization.

Provides coordinate normalization, map cropping, and temporal stacking
operators that extend datarax OperatorModule for use in the tokenization
pipeline.

All operators use JAX operations for vmap/JIT compatibility inside the
datarax CompositeOperatorModule DAG. Config values accessed via
``self.config.*`` are static (not traced), so they work safely under JIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from jaxtyping import PyTree

from simulacrax.core.constants import (
    ROADGRAPH_DIR,
    ROADGRAPH_VALID,
    ROADGRAPH_XYZ,
    STACKED_HISTORY,
    STATE_BBOX_YAW,
    STATE_IS_SDC,
    STATE_VELOCITY_X,
    STATE_VELOCITY_Y,
    STATE_X,
    STATE_Y,
    WOD_CURRENT_TIME_INDEX,
    WOD_HISTORY_STEPS,
)


def _get_ego_position(
    data: dict,
    step_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract SDC index and position at a given timestep.

    Uses JAX operations for vmap/JIT compatibility.

    Args:
        data: WOD scenario dict with state/is_sdc and state/all/* arrays.
        step_idx: Timestep index to read position from.

    Returns:
        Tuple of (sdc_idx, ego_x, ego_y) as JAX arrays.
    """
    sdc_idx = jnp.argmax(jnp.asarray(data[STATE_IS_SDC]))
    ego_x = jnp.asarray(data[STATE_X])[sdc_idx, step_idx]
    ego_y = jnp.asarray(data[STATE_Y])[sdc_idx, step_idx]
    return sdc_idx, ego_x, ego_y


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentNormalizationConfig(OperatorConfig):
    """Configuration for ego-centric normalization.

    Attributes:
        current_step_idx: Index of current timestep in temporal arrays.
    """

    current_step_idx: int = WOD_CURRENT_TIME_INDEX


class AgentNormalizationOperator(OperatorModule):
    """Center coordinates on ego vehicle and rotate to ego heading frame.

    Translates all positions so ego is at origin, rotates so ego faces +x,
    and adjusts headings and velocities accordingly. Operates on WOD
    scenario dicts with state/all/* arrays, and on the roadgraph geometry
    (roadgraph_samples/{xyz,dir}) when present so the map shares the agent
    frame. The transform is idempotent: re-normalizing a normalized scene
    (ego already at the origin with zero heading) is a no-op.
    """

    config: AgentNormalizationConfig

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply ego-centric normalization.

        Centers all coordinates on the SDC position at current_step_idx and
        rotates into the SDC heading frame so that ego faces +x direction.

        Args:
            data: Raw WOD scenario dict with state/all/* arrays.
            state: Pass-through state dict.
            metadata: Pass-through metadata dict.
            random_params: Unused (deterministic operator).
            stats: Unused.

        Returns:
            Tuple of (normalized_data, state, metadata).
        """
        result = dict(data)
        t = self.config.current_step_idx

        sdc_idx, ego_x, ego_y = _get_ego_position(data, t)
        ego_yaw = jnp.asarray(data[STATE_BBOX_YAW])[sdc_idx, t]

        cos_yaw = jnp.cos(-ego_yaw)
        sin_yaw = jnp.sin(-ego_yaw)

        x_centered = jnp.asarray(data[STATE_X]) - ego_x
        y_centered = jnp.asarray(data[STATE_Y]) - ego_y

        result[STATE_X] = x_centered * cos_yaw - y_centered * sin_yaw
        result[STATE_Y] = x_centered * sin_yaw + y_centered * cos_yaw

        result[STATE_BBOX_YAW] = jnp.asarray(data[STATE_BBOX_YAW]) - ego_yaw

        vx = jnp.asarray(data[STATE_VELOCITY_X])
        vy = jnp.asarray(data[STATE_VELOCITY_Y])
        result[STATE_VELOCITY_X] = vx * cos_yaw - vy * sin_yaw
        result[STATE_VELOCITY_Y] = vx * sin_yaw + vy * cos_yaw

        # Roadgraph geometry shares the state frame, so recentre and rotate it
        # too when present; otherwise the map stays in world coordinates while
        # the agents are ego-centred, and map cropping/encoding break.
        if ROADGRAPH_XYZ in data:
            xyz = jnp.asarray(data[ROADGRAPH_XYZ])
            rg_x = xyz[:, 0] - ego_x
            rg_y = xyz[:, 1] - ego_y
            result[ROADGRAPH_XYZ] = (
                xyz.at[:, 0]
                .set(rg_x * cos_yaw - rg_y * sin_yaw)
                .at[:, 1]
                .set(rg_x * sin_yaw + rg_y * cos_yaw)
            )
        if ROADGRAPH_DIR in data:
            dirs = jnp.asarray(data[ROADGRAPH_DIR])
            d_x = dirs[:, 0]
            d_y = dirs[:, 1]
            result[ROADGRAPH_DIR] = (
                dirs.at[:, 0]
                .set(d_x * cos_yaw - d_y * sin_yaw)
                .at[:, 1]
                .set(d_x * sin_yaw + d_y * cos_yaw)
            )

        return result, state, metadata


@dataclass(frozen=True, slots=True, kw_only=True)
class MapCroppingConfig(OperatorConfig):
    """Configuration for map radius cropping.

    Attributes:
        crop_radius: Maximum distance from ego to keep (meters).
        current_step_idx: Index of current timestep.
    """

    crop_radius: float = 150.0
    current_step_idx: int = WOD_CURRENT_TIME_INDEX


class MapCroppingOperator(OperatorModule):
    """Crop roadgraph samples to a radius around the ego vehicle.

    Marks roadgraph points beyond crop_radius from the SDC position
    at current_step_idx as invalid (valid=0).
    """

    config: MapCroppingConfig

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Crop map to radius around ego.

        Computes Euclidean distance from each roadgraph point to the SDC
        position and marks points beyond crop_radius as invalid.

        Args:
            data: Raw WOD scenario dict with roadgraph_samples/* and state/* arrays.
            state: Pass-through state dict.
            metadata: Pass-through metadata dict.
            random_params: Unused (deterministic operator).
            stats: Unused.

        Returns:
            Tuple of (cropped_data, state, metadata).
        """
        result = dict(data)
        t = self.config.current_step_idx

        _, ego_x, ego_y = _get_ego_position(data, t)

        xyz = jnp.asarray(data[ROADGRAPH_XYZ])
        dx = xyz[:, 0] - ego_x
        dy = xyz[:, 1] - ego_y
        dist = jnp.sqrt(dx**2 + dy**2)

        within_radius = (dist <= self.config.crop_radius).astype(jnp.int64)
        existing_valid = jnp.asarray(data[ROADGRAPH_VALID]).flatten()
        result[ROADGRAPH_VALID] = (existing_valid * within_radius).reshape(-1, 1)

        return result, state, metadata


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalStackingConfig(OperatorConfig):
    """Configuration for temporal feature stacking.

    Attributes:
        history_steps: Number of history timesteps to stack.
        feature_fields: Field names to stack along temporal dimension.
    """

    history_steps: int = WOD_HISTORY_STEPS
    feature_fields: list[str] = field(default_factory=lambda: [STATE_X, STATE_Y])


class TemporalStackingOperator(OperatorModule):
    """Stack historical timesteps into a feature vector.

    Extracts the first history_steps from temporal arrays and concatenates
    them into a single feature dimension. Also separates future steps into
    dedicated output fields.
    """

    config: TemporalStackingConfig

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Stack history and separate future.

        For each feature field, extracts history_steps into a concatenated
        stacked_history array and remaining steps into future_* fields.

        Args:
            data: Raw WOD scenario dict with temporal feature arrays.
            state: Pass-through state dict.
            metadata: Pass-through metadata dict.
            random_params: Unused (deterministic operator).
            stats: Unused.

        Returns:
            Tuple of (stacked_data, state, metadata).
        """
        result = dict(data)
        h = self.config.history_steps

        history_parts: list[jnp.ndarray] = []
        for field_name in self.config.feature_fields:
            arr = jnp.asarray(data[field_name])
            history_parts.append(arr[:, :h])
            future_key = f"future_{field_name.split('/')[-1]}"
            result[future_key] = arr[:, h:]

        result[STACKED_HISTORY] = jnp.concatenate(history_parts, axis=-1)

        return result, state, metadata
