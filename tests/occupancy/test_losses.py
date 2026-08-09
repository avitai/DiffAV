"""Tests for FlowConsistencyLoss: continuity equation enforcement."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from diffav.occupancy.flow_model import OccupancyGridPrediction
from diffav.occupancy.losses import FlowConsistencyLoss, FlowConsistencyLossConfig


def _make_pred(
    occupancy: jax.Array,
    flow: jax.Array,
) -> OccupancyGridPrediction:
    return OccupancyGridPrediction(occupancy=occupancy, flow=flow)


_CELL_SIZE_M = 1.0


def _loss_config(
    *,
    weight: float = 1.0,
    dt: float = 0.5,
    cell_size_m: float = _CELL_SIZE_M,
) -> FlowConsistencyLossConfig:
    """Loss config with an explicit unit cell size unless overridden."""
    return FlowConsistencyLossConfig(weight=weight, dt=dt, cell_size_m=cell_size_m)


class TestFlowConsistencyLoss:
    """Tests for FlowConsistencyLoss continuity equation loss."""

    def test_static_occupancy_zero_flow_zero_loss(self) -> None:
        """Constant occupancy + zero flow → residual = 0."""
        config = _loss_config()
        loss_fn = FlowConsistencyLoss(config)

        B, T, types, H, W = 1, 8, 3, 16, 16
        # Constant occupancy over time → ∂ρ/∂t = 0
        rho = jnp.ones((B, T, types, H, W)) * 0.5
        flow = jnp.zeros((B, T, H, W, 2))

        pred = _make_pred(rho, flow)
        loss = loss_fn.compute(pred)
        assert float(loss) < 1e-6

    def test_diverging_flow_nonzero_loss(self) -> None:
        """Expanding occupancy without matching flow → positive loss."""
        config = _loss_config()
        loss_fn = FlowConsistencyLoss(config)

        B, T, types, H, W = 1, 8, 3, 16, 16
        # Linearly growing occupancy over time
        t_vals = jnp.linspace(0.5, 1.5, T)
        rho = jnp.ones((B, T, types, H, W)) * t_vals[None, :, None, None, None]
        # Zero flow — can't explain the growing occupancy
        flow = jnp.zeros((B, T, H, W, 2))

        pred = _make_pred(rho, flow)
        loss = loss_fn.compute(pred)
        assert float(loss) > 1e-4

    def test_weight_scaling(self) -> None:
        """Loss with weight=2.0 is exactly 2x loss with weight=1.0."""
        B, T, types, H, W = 1, 8, 3, 16, 16
        t_vals = jnp.linspace(0.5, 1.5, T)
        rho = jnp.ones((B, T, types, H, W)) * t_vals[None, :, None, None, None]
        flow = jnp.zeros((B, T, H, W, 2))
        pred = _make_pred(rho, flow)

        loss_1 = FlowConsistencyLoss(_loss_config(weight=1.0)).compute(pred)
        loss_2 = FlowConsistencyLoss(_loss_config(weight=2.0)).compute(pred)
        assert abs(float(loss_2) - 2.0 * float(loss_1)) < 1e-5

    def test_loss_is_scalar(self) -> None:
        B, T, types, H, W = 1, 8, 3, 16, 16
        rho = jnp.ones((B, T, types, H, W))
        flow = jnp.zeros((B, T, H, W, 2))
        pred = _make_pred(rho, flow)
        loss = FlowConsistencyLoss(_loss_config()).compute(pred)
        assert loss.shape == ()

    def test_loss_is_nonnegative(self) -> None:
        B, T, types, H, W = 1, 8, 3, 16, 16
        rho = jax.random.uniform(jax.random.key(0), (B, T, types, H, W))
        flow = jax.random.normal(jax.random.key(1), (B, T, H, W, 2))
        pred = _make_pred(rho, flow)
        loss = FlowConsistencyLoss(_loss_config()).compute(pred)
        assert float(loss) >= 0.0

    def test_invalid_dt_raises(self) -> None:
        """dt=0 must raise ValueError immediately at config construction."""
        with pytest.raises(ValueError, match="dt must be positive"):
            _loss_config(dt=0.0)

    def test_invalid_cell_size_raises(self) -> None:
        """cell_size_m=0 must raise ValueError immediately at config construction."""
        with pytest.raises(ValueError, match="cell_size_m must be positive"):
            _loss_config(cell_size_m=0.0)


# ---------------------------------------------------------------------------
# PDE correctness: grid axes and boundary handling
# ---------------------------------------------------------------------------


class TestContinuityAxesAndBoundaries:
    """The residual must use the rasterizer's axis convention, non-periodically.

    The rasterizer maps ego-x to grid rows (axis -2, H) and ego-y to grid
    columns (axis -1, W), and the ego-centric grid is not periodic — mass
    crosses its edges, so boundary cells carry no defined stencil.
    """

    def test_uniform_density_linear_vx_matches_analytic_residual(self) -> None:
        """rho=1, vx = x  →  residual = ∂(ρvx)/∂x = 1 exactly on the interior.

        vx varies along ego-x, which the rasterizer stores on grid rows
        (axis -2). Differentiating along the wrong axis yields 0; keeping
        periodic wrap-around blows the mean up by orders of magnitude.
        """
        B, T, types, H, W = 1, 2, 1, 16, 16
        cell = 0.5
        rho = jnp.ones((B, T, types, H, W))
        x_coords = jnp.arange(H, dtype=jnp.float32) * cell  # metres along ego-x
        vx = jnp.broadcast_to(x_coords[None, None, :, None], (B, T, H, W))
        flow = jnp.stack([vx, jnp.zeros_like(vx)], axis=-1)

        loss = FlowConsistencyLoss(_loss_config(cell_size_m=cell)).compute(_make_pred(rho, flow))
        assert float(loss) == pytest.approx(1.0, rel=1e-5)

    def test_uniform_density_linear_vy_matches_analytic_residual(self) -> None:
        """rho=1, vy = y  →  residual = ∂(ρvy)/∂y = 1 exactly on the interior."""
        B, T, types, H, W = 1, 2, 1, 16, 16
        cell = 0.5
        rho = jnp.ones((B, T, types, H, W))
        y_coords = jnp.arange(W, dtype=jnp.float32) * cell  # metres along ego-y
        vy = jnp.broadcast_to(y_coords[None, None, None, :], (B, T, H, W))
        flow = jnp.stack([jnp.zeros_like(vy), vy], axis=-1)

        loss = FlowConsistencyLoss(_loss_config(cell_size_m=cell)).compute(_make_pred(rho, flow))
        assert float(loss) == pytest.approx(1.0, rel=1e-5)

    def test_translating_blob_consistent_flow_beats_inconsistent(self) -> None:
        """A blob moving along ego-x with matching vx scores below mismatched vy."""
        B, T, _types, H, W = 1, 4, 1, 32, 32
        cell = 1.0
        speed = 1.0  # m/s; with dt=0.5 the blob moves 0.5 cells per frame
        dt = 0.5
        rows = jnp.arange(H, dtype=jnp.float32)[:, None]
        cols = jnp.arange(W, dtype=jnp.float32)[None, :]

        frames = []
        for t in range(T):
            centre_row = 10.0 + speed * dt * t / cell
            blob = jnp.exp(-0.5 * ((rows - centre_row) ** 2 + (cols - 16.0) ** 2) / 9.0)
            frames.append(blob)
        rho = jnp.stack(frames)[None, :, None]  # (B, T, types, H, W)

        vx_field = jnp.full((B, T, H, W), speed)
        zeros = jnp.zeros_like(vx_field)
        consistent = jnp.stack([vx_field, zeros], axis=-1)
        inconsistent = jnp.stack([zeros, vx_field], axis=-1)

        config = _loss_config(cell_size_m=cell, dt=dt)
        loss_consistent = FlowConsistencyLoss(config).compute(_make_pred(rho, consistent))
        loss_inconsistent = FlowConsistencyLoss(config).compute(_make_pred(rho, inconsistent))
        assert float(loss_consistent) < float(loss_inconsistent)


class TestLossTransformCompatibility:
    """The loss is a JAX path: jit, grad, and determinism must hold."""

    @staticmethod
    def _random_pred(seed: int = 0) -> OccupancyGridPrediction:
        B, T, types, H, W = 1, 4, 3, 16, 16
        rho = jax.random.uniform(jax.random.key(seed), (B, T, types, H, W))
        flow = jax.random.normal(jax.random.key(seed + 1), (B, T, H, W, 2))
        return _make_pred(rho, flow)

    def test_jit_matches_eager(self) -> None:
        """jax.jit over the compute path reproduces the eager value."""
        loss_fn = FlowConsistencyLoss(_loss_config())

        def compute(occupancy: jax.Array, flow: jax.Array) -> jax.Array:
            return loss_fn.compute(_make_pred(occupancy, flow))

        pred = self._random_pred()
        eager = compute(pred.occupancy, pred.flow)
        jitted = jax.jit(compute)(pred.occupancy, pred.flow)
        assert jnp.allclose(eager, jitted, rtol=1e-6)

    def test_gradient_flows_to_flow_field(self) -> None:
        """d(loss)/d(flow) is finite and non-zero for a generic prediction."""
        loss_fn = FlowConsistencyLoss(_loss_config())
        pred = self._random_pred()

        def compute(flow: jax.Array) -> jax.Array:
            return loss_fn.compute(_make_pred(pred.occupancy, flow))

        grad = jax.grad(compute)(pred.flow)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0

    def test_deterministic(self) -> None:
        """Two identical computations produce bit-identical scalars."""
        loss_fn = FlowConsistencyLoss(_loss_config())
        pred = self._random_pred()
        first = loss_fn.compute(pred)
        second = loss_fn.compute(pred)
        assert jnp.array_equal(first, second)
