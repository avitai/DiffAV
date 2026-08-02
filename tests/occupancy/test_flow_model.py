"""Tests for OccupancyFlowConfig and data types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.occupancy.flow_model import (
    create_occupancy_flow_model,
    OccupancyFlowConfig,
    OccupancyFlowModel,
    OccupancyGrid,
    OccupancyGridPrediction,
)


class TestOccupancyFlowConfig:
    """Tests for OccupancyFlowConfig defaults and derived values."""

    def test_default_grid_resolution(self) -> None:
        """Default grid resolution is 128."""
        config = OccupancyFlowConfig()
        assert config.grid_resolution == 128

    def test_default_temporal_horizon(self) -> None:
        """Default temporal horizon is 8 timesteps."""
        config = OccupancyFlowConfig()
        assert config.temporal_horizon == 8

    def test_default_num_agent_types(self) -> None:
        """Default number of agent types is 3."""
        config = OccupancyFlowConfig()
        assert config.num_agent_types == 3

    def test_default_modes_per_scale(self) -> None:
        """Default modes_per_scale has 3 scale levels."""
        config = OccupancyFlowConfig()
        assert len(config.modes_per_scale) == 3

    def test_immutability(self) -> None:
        """Config is frozen and raises on mutation attempt."""
        config = OccupancyFlowConfig()
        with pytest.raises(FrozenInstanceError):
            config.grid_resolution = 64  # type: ignore[misc]

    def test_in_channels_derived(self) -> None:
        """Input channels = num_agent_types + 2 (flow) + num_map_channels = 8."""
        config = OccupancyFlowConfig()
        # 3 occ + 2 flow + 3 map = 8
        expected = config.num_agent_types + 2 + config.num_map_channels
        assert expected == 8

    def test_out_channels_derived(self) -> None:
        """Output channels = temporal_horizon * num_agent_types * 3 = 72."""
        config = OccupancyFlowConfig()
        # T=8 x types=3 x 3 = 72
        expected = config.temporal_horizon * config.num_agent_types * 3
        assert expected == 72

    def test_gradient_checkpointing_default_true(self) -> None:
        """Gradient checkpointing is enabled by default."""
        config = OccupancyFlowConfig()
        assert config.use_gradient_checkpointing is True

    def test_gradient_checkpointing_can_be_disabled(self) -> None:
        """Gradient checkpointing can be explicitly disabled."""
        config = OccupancyFlowConfig(use_gradient_checkpointing=False)
        assert config.use_gradient_checkpointing is False

    def test_attention_heads_must_divide_hidden_channels(self) -> None:
        """attention_heads that does not divide hidden_channels raises ValueError."""
        with pytest.raises(ValueError, match="attention_heads"):
            OccupancyFlowConfig(hidden_channels=8, attention_heads=3)

    def test_attention_heads_divisibility_ok(self) -> None:
        """Valid attention_heads/hidden_channels combination constructs without error."""
        config = OccupancyFlowConfig(hidden_channels=8, attention_heads=2)
        assert config.attention_heads == 2


class TestOccupancyGrid:
    """Tests for OccupancyGrid data container."""

    def test_shape_consistency(self) -> None:
        """All arrays have matching H, W spatial dimensions."""
        H, W = 32, 32
        grid = OccupancyGrid(
            occupancy=jnp.zeros((H, W, 3)),
            flow=jnp.zeros((H, W, 2)),
            map_features=jnp.zeros((H, W, 3)),
        )
        assert grid.occupancy.shape == (H, W, 3)
        assert grid.flow.shape == (H, W, 2)
        assert grid.map_features.shape == (H, W, 3)

    def test_immutability(self) -> None:
        """OccupancyGrid is frozen and raises on mutation attempt."""
        grid = OccupancyGrid(
            occupancy=jnp.zeros((32, 32, 3)),
            flow=jnp.zeros((32, 32, 2)),
            map_features=jnp.zeros((32, 32, 3)),
        )
        with pytest.raises(FrozenInstanceError):
            grid.occupancy = jnp.ones((32, 32, 3))  # type: ignore[misc]


class TestOccupancyGridPrediction:
    """Tests for OccupancyGridPrediction data container."""

    def test_shapes(self) -> None:
        """Prediction arrays carry the correct batch and temporal dimensions."""
        B, T, types, H, W = 1, 8, 3, 64, 64
        pred = OccupancyGridPrediction(
            occupancy=jnp.zeros((B, T, types, H, W)),
            flow=jnp.zeros((B, T, H, W, 2)),
        )
        assert pred.occupancy.shape == (B, T, types, H, W)
        assert pred.flow.shape == (B, T, H, W, 2)


@pytest.mark.slow
class TestOccupancyFlowModel:
    """Tests for OccupancyFlowModel forward pass."""

    def _make_grid(self, H: int) -> OccupancyGrid:
        return OccupancyGrid(
            occupancy=jnp.zeros((H, H, 3)),
            flow=jnp.zeros((H, H, 2)),
            map_features=jnp.zeros((H, H, 3)),
        )

    def _make_model(self, H: int) -> OccupancyFlowModel:
        config = OccupancyFlowConfig(
            grid_resolution=H,
            modes_per_scale=(4, 2, 1),
            num_layers_per_scale=(1, 1, 1),
            hidden_channels=8,
            attention_heads=2,  # 8 / 2 = 4 — explicitly valid
            use_gradient_checkpointing=False,
        )
        return create_occupancy_flow_model(config, nnx.Rngs(params=jax.random.key(0)))

    def test_forward_shape_64(self) -> None:
        """Forward pass at 64x64 produces correctly shaped occupancy and flow."""
        H = 64
        model = self._make_model(H)
        grid = self._make_grid(H)
        pred = model(grid)
        assert pred.occupancy.shape == (1, 8, 3, H, H)
        assert pred.flow.shape == (1, 8, H, H, 2)

    def test_forward_shape_128(self) -> None:
        """Forward pass at 128x128 produces correctly shaped occupancy and flow."""
        H = 128
        model = self._make_model(H)
        grid = self._make_grid(H)
        pred = model(grid)
        assert pred.occupancy.shape == (1, 8, 3, H, H)
        assert pred.flow.shape == (1, 8, H, H, 2)

    def test_resolution_independence(self) -> None:
        """Same model architecture at 64 and 128 — both produce valid output."""
        for H in (64, 128):
            model = self._make_model(H)
            pred = model(self._make_grid(H))
            assert pred.occupancy.shape[-2:] == (H, H)
            assert pred.flow.shape[-3:-1] == (H, H)

    def test_occupancy_range(self) -> None:
        """Occupancy output must be in [0, 1] (sigmoid applied)."""
        model = self._make_model(32)
        pred = model(self._make_grid(32))
        assert float(jnp.min(pred.occupancy)) >= 0.0
        assert float(jnp.max(pred.occupancy)) <= 1.0

    def test_gradient_flow(self) -> None:
        """All model parameters receive finite gradients."""
        model = self._make_model(32)
        grid = self._make_grid(32)

        def loss_fn(m: OccupancyFlowModel) -> jax.Array:
            pred = m(grid)
            return jnp.mean(pred.occupancy)

        loss_val, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss_val)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        finite_count = sum(1 for g in grad_leaves if jnp.all(jnp.isfinite(g)))
        assert finite_count == len(grad_leaves)

    def test_factory_returns_correct_type(self) -> None:
        """create_occupancy_flow_model returns an OccupancyFlowModel instance."""
        config = OccupancyFlowConfig(
            grid_resolution=32,
            modes_per_scale=(2, 1, 1),
            num_layers_per_scale=(1, 1, 1),
            hidden_channels=4,
            attention_heads=4,
            use_gradient_checkpointing=False,
        )
        model = create_occupancy_flow_model(config, nnx.Rngs(params=jax.random.key(42)))
        assert isinstance(model, OccupancyFlowModel)


class TestConfigCouplingValidation:
    """Coupled config fields must be validated at construction."""

    def test_mismatched_scale_list_lengths_raise(self) -> None:
        """modes_per_scale and num_layers_per_scale lengths must match."""
        with pytest.raises(ValueError, match="num_layers_per_scale"):
            OccupancyFlowConfig(
                modes_per_scale=(4, 2, 1),
                num_layers_per_scale=(1, 1),
            )

    def test_modes_exceeding_scale_nyquist_raise(self) -> None:
        """modes_per_scale[i] must fit the downsampled resolution at scale i."""
        # Scale 2 runs at 32 / 4 = 8 pixels → at most 4 modes.
        with pytest.raises(ValueError, match="modes_per_scale"):
            OccupancyFlowConfig(
                grid_resolution=32,
                modes_per_scale=(4, 2, 5),
                num_layers_per_scale=(1, 1, 1),
            )

    def test_modes_exceeding_top_scale_nyquist_raise(self) -> None:
        """Even the finest scale bounds its mode count at resolution // 2."""
        with pytest.raises(ValueError, match="modes_per_scale"):
            OccupancyFlowConfig(
                grid_resolution=32,
                modes_per_scale=(20, 2, 1),
                num_layers_per_scale=(1, 1, 1),
            )

    def test_non_positive_modes_raise(self) -> None:
        """Zero modes at any scale is invalid."""
        with pytest.raises(ValueError, match="modes_per_scale"):
            OccupancyFlowConfig(
                grid_resolution=32,
                modes_per_scale=(4, 0, 1),
                num_layers_per_scale=(1, 1, 1),
            )

    def test_resolution_not_divisible_by_scale_factor_raises(self) -> None:
        """The grid must downsample evenly across all scale levels."""
        with pytest.raises(ValueError, match="grid_resolution"):
            OccupancyFlowConfig(
                grid_resolution=18,
                modes_per_scale=(2, 1, 1),
                num_layers_per_scale=(1, 1, 1),
            )

    def test_non_positive_grid_resolution_raises(self) -> None:
        """grid_resolution must be strictly positive."""
        with pytest.raises(ValueError, match="grid_resolution"):
            OccupancyFlowConfig(grid_resolution=0)

    def test_non_positive_grid_size_raises(self) -> None:
        """grid_size_m must be strictly positive."""
        with pytest.raises(ValueError, match="grid_size_m"):
            OccupancyFlowConfig(grid_size_m=0.0)

    def test_non_positive_temporal_horizon_raises(self) -> None:
        """temporal_horizon must be strictly positive."""
        with pytest.raises(ValueError, match="temporal_horizon"):
            OccupancyFlowConfig(temporal_horizon=0)

    def test_cell_size_derived(self) -> None:
        """cell_size_m is grid_size_m / grid_resolution."""
        config = OccupancyFlowConfig()
        assert config.cell_size_m == pytest.approx(60.0 / 128)

    def test_valid_config_constructs(self) -> None:
        """A coherent configuration passes all coupling checks."""
        config = OccupancyFlowConfig(
            grid_resolution=32,
            modes_per_scale=(4, 2, 2),
            num_layers_per_scale=(1, 1, 1),
        )
        assert config.grid_resolution == 32


class TestModelReadsGridResolution:
    """The model must enforce its configured grid resolution on inputs."""

    def test_mismatched_input_resolution_raises(self) -> None:
        """A grid whose spatial size differs from config.grid_resolution fails fast."""
        config = OccupancyFlowConfig(
            grid_resolution=32,
            modes_per_scale=(2, 1, 1),
            num_layers_per_scale=(1, 1, 1),
            hidden_channels=4,
            attention_heads=4,
            use_gradient_checkpointing=False,
        )
        model = create_occupancy_flow_model(config, nnx.Rngs(params=jax.random.key(0)))
        wrong_grid = OccupancyGrid(
            occupancy=jnp.zeros((16, 16, 3)),
            flow=jnp.zeros((16, 16, 2)),
            map_features=jnp.zeros((16, 16, 3)),
        )
        with pytest.raises(ValueError, match="grid_resolution"):
            model(wrong_grid)
