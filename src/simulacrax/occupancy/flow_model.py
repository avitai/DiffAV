"""Occupancy flow prediction via Fourier Neural Operators.

This module provides the OccupancyFlowModel, which wraps
opifex.neural.operators.fno.multiscale.MultiScaleFourierNeuralOperator
to predict future occupancy grids and flow vectors from rasterized scenes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.operators.fno.multiscale import MultiScaleFourierNeuralOperator


@dataclass(frozen=True, slots=True, kw_only=True)
class OccupancyFlowConfig:
    """Configuration for the occupancy flow prediction model.

    Attributes:
        grid_resolution: Height and width of the occupancy grid in pixels.
            Use 128 for CPU demo; 256 for production (WOD standard).
        grid_size_m: Scene coverage in metres per side (H x W region).
        temporal_horizon: Number of future timesteps to predict (4s at 2Hz).
        num_agent_types: Number of agent classes (vehicle, pedestrian, cyclist).
        num_map_channels: Number of map feature channels (road, crosswalk, signal).
        hidden_channels: FNO hidden dimension.
        modes_per_scale: Fourier modes at each scale level. Must satisfy
            modes_per_scale[i] <= grid_resolution // 2 at all levels.
        num_layers_per_scale: Number of FNO layers at each scale.
        use_cross_scale_attention: Enable cross-scale attention in FNO backbone.
        attention_heads: Number of attention heads for cross-scale attention.
            Must evenly divide hidden_channels when use_cross_scale_attention=True.
        use_gradient_checkpointing: Enable gradient checkpointing in FNO backbone
            via artifex. Saves memory at the cost of recomputation. Set False in
            tests for speed.
    """

    grid_resolution: int = 128
    grid_size_m: float = 60.0
    temporal_horizon: int = 8
    num_agent_types: int = 3
    num_map_channels: int = 3
    hidden_channels: int = 32
    modes_per_scale: tuple[int, ...] = (12, 6, 3)
    num_layers_per_scale: tuple[int, ...] = (2, 2, 2)
    use_cross_scale_attention: bool = True
    attention_heads: int = 8
    use_gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        """Validate positivity and the coupling between grid and FNO scales."""
        if self.grid_resolution <= 0:
            raise ValueError(f"grid_resolution must be positive, got {self.grid_resolution}")
        if self.grid_size_m <= 0.0:
            raise ValueError(f"grid_size_m must be positive, got {self.grid_size_m}")
        if self.temporal_horizon <= 0:
            raise ValueError(f"temporal_horizon must be positive, got {self.temporal_horizon}")
        if len(self.modes_per_scale) != len(self.num_layers_per_scale):
            raise ValueError(
                f"modes_per_scale and num_layers_per_scale must have equal lengths, "
                f"got {len(self.modes_per_scale)} and {len(self.num_layers_per_scale)} "
                f"(the FNO backbone would silently truncate the longer list)."
            )
        num_scales = len(self.modes_per_scale)
        coarsest_factor = 2 ** (num_scales - 1)
        if self.grid_resolution % coarsest_factor != 0:
            raise ValueError(
                f"grid_resolution ({self.grid_resolution}) must be divisible by "
                f"2**(num_scales - 1) = {coarsest_factor} so every scale downsamples evenly."
            )
        for scale_index, modes in enumerate(self.modes_per_scale):
            scale_resolution = self.grid_resolution // (2**scale_index)
            if not 0 < modes <= scale_resolution // 2:
                raise ValueError(
                    f"modes_per_scale[{scale_index}] = {modes} must be in "
                    f"[1, {scale_resolution // 2}] for the scale's "
                    f"{scale_resolution}-pixel resolution."
                )
        if self.use_cross_scale_attention and self.hidden_channels % self.attention_heads != 0:
            raise ValueError(
                f"attention_heads ({self.attention_heads}) must evenly divide "
                f"hidden_channels ({self.hidden_channels}) when "
                f"use_cross_scale_attention=True."
            )

    @property
    def cell_size_m(self) -> float:
        """Grid cell size in metres (grid_size_m / grid_resolution)."""
        return self.grid_size_m / self.grid_resolution


@dataclass(frozen=True, slots=True, kw_only=True)
class OccupancyGrid:
    """Rasterized top-down occupancy grid for a single scene snapshot.

    All arrays are channels-last for natural numpy/WOD indexing.
    The model's __call__ handles transposition to FNO's channels-first format.

    Attributes:
        occupancy: Per-type occupancy probability, shape (H, W, num_agent_types).
            Values in [0, 1] via Gaussian soft rasterization.
        flow: Aggregate velocity field, shape (H, W, 2) as (vx, vy) in m/s.
        map_features: Map channel probabilities, shape (H, W, num_map_channels).
            Channels: road (0), crosswalk (1), traffic_signal (2).
    """

    occupancy: jax.Array
    flow: jax.Array
    map_features: jax.Array


@dataclass(frozen=True, slots=True, kw_only=True)
class OccupancyGridPrediction:
    """Predicted future occupancy grids and flow vectors.

    Attributes:
        occupancy: Predicted occupancy, shape (B, T, num_agent_types, H, W).
            Values in [0, 1] via sigmoid.
        flow: Predicted flow vectors, shape (B, T, H, W, 2) as (vx, vy) in m/s.
        occupancy_logits: Pre-sigmoid occupancy, same shape as ``occupancy``.
            Use for numerically stable cross-entropy training (the official
            occupancy-flow loss is sigmoid XE on logits). ``None`` only for
            hand-built predictions in tests.
    """

    occupancy: jax.Array
    flow: jax.Array
    occupancy_logits: jax.Array | None = None


class OccupancyFlowModel(nnx.Module):
    """Occupancy flow prediction model using Multi-Scale Fourier Neural Operator.

    Predicts future occupancy grids and flow vectors from a rasterized top-down
    scene. Uses opifex.neural.operators.fno.multiscale.MultiScaleFourierNeuralOperator
    as the backbone, which provides resolution-independent predictions via spectral
    convolution across multiple resolution scales.

    Input format: OccupancyGrid (H, W, 8) — 3 occupancy + 2 flow + 3 map channels.
    Output format: OccupancyGridPrediction with shapes:
        occupancy: (B=1, T, num_agent_types, H, W)
        flow:      (B=1, T, H, W, 2)
    """

    def __init__(self, config: OccupancyFlowConfig, rngs: nnx.Rngs) -> None:
        """Initialize OccupancyFlowModel.

        Args:
            config: Model configuration.
            rngs: Random number generators for parameter initialization.
        """
        super().__init__()
        self.config = config

        in_channels = config.num_agent_types + 2 + config.num_map_channels  # = 8
        # out = T * types * 3 (occ + vx + vy per type)
        out_channels = config.temporal_horizon * config.num_agent_types * 3  # = 72

        self.backbone = MultiScaleFourierNeuralOperator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=config.hidden_channels,
            modes_per_scale=list(config.modes_per_scale),
            num_layers_per_scale=list(config.num_layers_per_scale),
            use_cross_scale_attention=config.use_cross_scale_attention,
            attention_heads=config.attention_heads,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            rngs=rngs,
        )

    def __call__(self, grid: OccupancyGrid) -> OccupancyGridPrediction:
        """Predict future occupancy and flow from current scene grid.

        Args:
            grid: Rasterized scene as OccupancyGrid with (H, W, C) channels-last.

        Returns:
            OccupancyGridPrediction with B=1 batch dimension added.

        Raises:
            ValueError: If the grid's spatial size does not match the
                configured ``grid_resolution`` (the mode counts were
                validated against that resolution).
        """
        H, W = grid.occupancy.shape[:2]
        expected = self.config.grid_resolution
        if (H, W) != (expected, expected):
            raise ValueError(
                f"grid spatial size {(H, W)} does not match config.grid_resolution "
                f"{expected}; construct the model with the resolution it will consume."
            )

        # Stack channels: (H, W, 8) channels-last
        channels = jnp.concatenate(
            [grid.occupancy, grid.flow, grid.map_features], axis=-1
        )  # (H, W, 8)

        # FNO expects channels-first: (1, 8, H, W)
        x = channels.transpose(2, 0, 1)[jnp.newaxis]  # (1, 8, H, W)

        # Forward pass: (1, 8, H, W) → (1, 72, H, W)
        raw = self.backbone(x)  # (1, 72, H, W)

        # Reshape to (1, T, types, 3, H, W) — 3 = (occ, vx, vy)
        T = self.config.temporal_horizon
        types = self.config.num_agent_types
        raw_reshaped = raw.reshape(1, T, types, 3, H, W)

        # Split: occupancy via sigmoid (logits kept for stable XE training),
        # flow as average over agent types
        occupancy_logits = raw_reshaped[:, :, :, 0, :, :]  # (1, T, types, H, W)
        occupancy = jax.nn.sigmoid(occupancy_logits)
        flow_x = raw_reshaped[:, :, :, 1, :, :].mean(axis=2)  # (1, T, H, W)
        flow_y = raw_reshaped[:, :, :, 2, :, :].mean(axis=2)  # (1, T, H, W)
        flow = jnp.stack([flow_x, flow_y], axis=-1)  # (1, T, H, W, 2)

        return OccupancyGridPrediction(
            occupancy=occupancy, flow=flow, occupancy_logits=occupancy_logits
        )


def create_occupancy_flow_model(config: OccupancyFlowConfig, rngs: nnx.Rngs) -> OccupancyFlowModel:
    """Create an OccupancyFlowModel from config.

    Args:
        config: Model configuration specifying grid resolution, FNO architecture.
        rngs: Random number generators for parameter initialization.

    Returns:
        Initialized OccupancyFlowModel.
    """
    return OccupancyFlowModel(config, rngs)
