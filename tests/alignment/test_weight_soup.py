"""Tests for weight-soup parameter interpolation between steering experts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.alignment.weight_soup import make_weight_soup
from simulacrax.models.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)
from tests import support
from tests.alignment.helpers import CONTEXT_DIM, FUTURE_STEPS, NUM_AGENTS


def _make_model(seed: int) -> TrajectoryDiffusionModel:
    cfg = support.make_diffusion_config(
        future_steps=FUTURE_STEPS,
        num_agents_max=NUM_AGENTS + 1,
        context_dim=CONTEXT_DIM,
    )
    return support.make_diffusion_model(cfg, seed=seed)


def _param_leaves(model: TrajectoryDiffusionModel) -> list[jax.Array]:
    return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))


class TestMakeWeightSoup:
    def test_lambda_zero_equals_base(self) -> None:
        base, target = _make_model(0), _make_model(1)
        soup = make_weight_soup(base, target, 0.0)
        for a, b in zip(_param_leaves(soup), _param_leaves(base), strict=True):
            assert jnp.array_equal(a, b)

    def test_lambda_one_equals_target(self) -> None:
        base, target = _make_model(0), _make_model(1)
        soup = make_weight_soup(base, target, 1.0)
        for a, b in zip(_param_leaves(soup), _param_leaves(target), strict=True):
            assert jnp.array_equal(a, b)

    def test_lambda_half_is_exact_mean(self) -> None:
        base, target = _make_model(0), _make_model(1)
        soup = make_weight_soup(base, target, 0.5)
        for s, b, t in zip(
            _param_leaves(soup), _param_leaves(base), _param_leaves(target), strict=True
        ):
            assert jnp.allclose(s, 0.5 * b + 0.5 * t)

    def test_invalid_lambda_raises(self) -> None:
        base, target = _make_model(0), _make_model(1)
        with pytest.raises(ValueError, match="interpolation_weight"):
            make_weight_soup(base, target, -0.1)
        with pytest.raises(ValueError, match="interpolation_weight"):
            make_weight_soup(base, target, 1.5)

    def test_inputs_untouched(self) -> None:
        base, target = _make_model(0), _make_model(1)
        base_before = [p.copy() for p in _param_leaves(base)]
        target_before = [p.copy() for p in _param_leaves(target)]
        make_weight_soup(base, target, 0.3)
        assert all(jnp.array_equal(a, b) for a, b in zip(_param_leaves(base), base_before))
        assert all(jnp.array_equal(a, b) for a, b in zip(_param_leaves(target), target_before))

    def test_soup_can_sample(self) -> None:
        base, target = _make_model(0), _make_model(1)
        soup = make_weight_soup(base, target, 0.5)
        prediction = soup.sample(jnp.ones((NUM_AGENTS, CONTEXT_DIM)), key=jax.random.key(0))
        assert prediction.trajectories.shape == (NUM_AGENTS, FUTURE_STEPS, 4)
        assert bool(jnp.all(jnp.isfinite(prediction.trajectories)))
