"""PDE-informed losses for occupancy flow training.

This module provides FlowConsistencyLoss, which enforces the continuity
equation ∂ρ/∂t + ∇·(ρv) = 0 as a soft physics constraint during training.

Note (TODO): This loss implements a generic PDE continuity constraint for
any 2D density field. It should be contributed back to
opifex.core.physics.losses as a proper ContinuityEquationLoss, replacing
the placeholder in ConservationLawEnforcer._compute_mass_conservation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simulacrax.occupancy.flow_model import OccupancyGridPrediction


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowConsistencyLossConfig:
    """Configuration for FlowConsistencyLoss.

    Attributes:
        weight: Loss weight multiplier for combining with other losses.
        dt: Time interval between predicted frames in seconds.
            Occupancy-flow waypoints are 1 s apart (8 waypoints over the
            8 s horizon), so the default is 1.0.
        cell_size_m: Grid cell size in metres. No default — derive it from
            the grid configuration (``OccupancyFlowConfig.cell_size_m`` or
            ``RasterizerConfig.cell_size_m``) so the spatial derivative
            normalisation matches the actual grid.
    """

    weight: float = 1.0
    dt: float = 1.0
    cell_size_m: float

    def __post_init__(self) -> None:
        """Validate physical parameters are strictly positive."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.cell_size_m <= 0.0:
            raise ValueError(f"cell_size_m must be positive, got {self.cell_size_m}")


class FlowConsistencyLoss:
    """Continuity equation loss: ∂ρ/∂t + ∇·(ρv) = 0.

    Enforces mass conservation for predicted occupancy flow: agents cannot
    appear or disappear, they can only move. The residual is the violation
    of the continuity PDE in its discrete finite-difference form.

    Operates on per-type occupancy averaged to a scalar density field,
    which simplifies the divergence computation while preserving the
    physical constraint.

    Note (TODO): Contribute this loss to opifex.core.physics.losses as
    ContinuityEquationLoss — the existing ConservationLawEnforcer only
    implements a placeholder for integral conservation, not the local PDE.
    """

    def __init__(self, config: FlowConsistencyLossConfig) -> None:
        """Initialise loss function.

        Args:
            config: Loss configuration.
        """
        self.config = config

    def compute(self, prediction: OccupancyGridPrediction) -> jax.Array:
        """Compute continuity equation violation.

        Implements the discrete form:
            residual = (ρ[t+1] - ρ[t]) / dt + ∂(ρvx)/∂x + ∂(ρvy)/∂y

        The rasterizer maps ego-x to grid rows (axis -2, H) and ego-y to
        grid columns (axis -1, W), so ∂/∂x differentiates along rows and
        ∂/∂y along columns. Spatial derivatives use central differences on
        interior cells only: the ego-centric grid is not periodic — mass
        genuinely enters and leaves through its edges — so boundary cells
        carry no defined stencil and are excluded from the residual.
        Occupancy is summed over agent types before computing the PDE
        residual.

        Args:
            prediction: OccupancyGridPrediction from OccupancyFlowModel.

        Returns:
            Scalar loss value (non-negative).
        """
        # Sum over agent types for a scalar density field: (B, T, H, W)
        rho = prediction.occupancy.sum(axis=2)
        vx = prediction.flow[..., 0]  # (B, T, H, W)
        vy = prediction.flow[..., 1]  # (B, T, H, W)

        # Time derivative via forward difference: (B, T-1, H, W)
        drho_dt = jnp.diff(rho, axis=1) / self.config.dt

        # Spatial flux at midpoint timesteps: (B, T-1, H, W)
        rho_t = rho[:, :-1]
        vx_t = vx[:, :-1]
        vy_t = vy[:, :-1]
        flux_x = rho_t * vx_t  # transported along ego-x → grid rows
        flux_y = rho_t * vy_t  # transported along ego-y → grid columns

        dx = dy = self.config.cell_size_m

        # Interior central differences: ∂f/∂x ≈ (f[i+1] - f[i-1]) / (2·dx)
        div_x = (flux_x[..., 2:, :] - flux_x[..., :-2, :]) / (2.0 * dx)  # (…, H-2, W)
        div_y = (flux_y[..., :, 2:] - flux_y[..., :, :-2]) / (2.0 * dy)  # (…, H, W-2)

        residual = (
            drho_dt[..., 1:-1, 1:-1] + div_x[..., :, 1:-1] + div_y[..., 1:-1, :]
        )  # (B, T-1, H-2, W-2)
        return self.config.weight * jnp.mean(residual**2)
