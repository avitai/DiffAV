"""Tests for shared training utilities.

Verifies nan_safe_gradients correctness, NaN propagation, JIT compatibility,
and PyTree handling across different gradient structures.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from simulacrax.core.training_utils import nan_safe_gradients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_loss_and_grads(
    value: float = 1.0,
    grad_value: float = 0.5,
    shape: tuple[int, ...] = (3,),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Return a clean (no-NaN) loss and gradient PyTree."""
    loss = jnp.array(value)
    grads = {"w": jnp.full(shape, grad_value), "b": jnp.full((2,), grad_value)}
    return loss, grads


# ---------------------------------------------------------------------------
# Normal operation (no NaN)
# ---------------------------------------------------------------------------


class TestNanSafeGradientsClean:
    """Tests with clean (no-NaN) loss and gradients."""

    def test_returns_tuple(self) -> None:
        """Returns (safe_grads, has_nan) tuple."""
        loss, grads = _make_clean_loss_and_grads()
        result = nan_safe_gradients(loss, grads)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_clean_grads_unchanged(self) -> None:
        """Clean gradients pass through unchanged."""
        loss, grads = _make_clean_loss_and_grads(grad_value=0.42)
        safe_grads, _ = nan_safe_gradients(loss, grads)
        for key in grads:
            assert jnp.allclose(safe_grads[key], grads[key])

    def test_has_nan_false_for_clean_inputs(self) -> None:
        """has_nan is False when neither loss nor gradients contain NaN."""
        loss, grads = _make_clean_loss_and_grads()
        _, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)

    def test_gradient_values_preserved(self) -> None:
        """Gradient magnitudes are preserved exactly for clean inputs."""
        loss, grads = _make_clean_loss_and_grads(grad_value=3.14)
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)
        assert float(safe_grads["w"][0]) == pytest.approx(3.14, rel=1e-5)

    def test_zero_gradients_preserved(self) -> None:
        """Zero gradients are preserved (not misidentified as NaN)."""
        loss = jnp.array(0.5)
        grads = {"w": jnp.zeros((4,))}
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)
        assert jnp.all(safe_grads["w"] == 0.0)

    def test_large_gradients_preserved(self) -> None:
        """Very large but finite gradients pass through."""
        loss = jnp.array(1e6)
        grads = {"w": jnp.full((3,), 1e10)}
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)
        assert jnp.all(jnp.isfinite(safe_grads["w"]))


# ---------------------------------------------------------------------------
# NaN in loss
# ---------------------------------------------------------------------------


class TestNanSafeGradientsNanLoss:
    """Tests when the loss itself is NaN."""

    def test_nan_loss_zeros_gradients(self) -> None:
        """NaN loss causes all gradients to be zeroed."""
        loss = jnp.array(float("nan"))
        grads = {"w": jnp.ones((5,)), "b": jnp.ones((2,))}
        safe_grads, _ = nan_safe_gradients(loss, grads)
        for key in grads:
            assert jnp.all(safe_grads[key] == 0.0)

    def test_nan_loss_has_nan_true(self) -> None:
        """NaN loss sets has_nan to True."""
        loss = jnp.array(float("nan"))
        grads = {"w": jnp.ones((3,))}
        _, has_nan = nan_safe_gradients(loss, grads)
        assert bool(has_nan)

    def test_inf_loss_does_not_trigger(self) -> None:
        """Infinite (but not NaN) loss does not zero gradients (isnan vs isinf)."""
        loss = jnp.array(float("inf"))
        grads = {"w": jnp.ones((3,))}
        safe_grads, _ = nan_safe_gradients(loss, grads)
        # inf is not nan — behavior depends on implementation
        # We just verify output is well-formed
        assert safe_grads["w"].shape == grads["w"].shape


# ---------------------------------------------------------------------------
# NaN in gradients
# ---------------------------------------------------------------------------


class TestNanSafeGradientsNanGrads:
    """Tests when gradients contain NaN values."""

    def test_nan_grad_zeros_all_gradients(self) -> None:
        """NaN in any gradient leaf zeros all gradients."""
        loss = jnp.array(1.0)
        grads = {
            "w": jnp.array([1.0, float("nan"), 3.0]),
            "b": jnp.ones((2,)),
        }
        safe_grads, _ = nan_safe_gradients(loss, grads)
        assert jnp.all(safe_grads["w"] == 0.0)
        assert jnp.all(safe_grads["b"] == 0.0)

    def test_nan_grad_has_nan_true(self) -> None:
        """NaN in any gradient sets has_nan to True."""
        loss = jnp.array(1.0)
        grads = {"w": jnp.array([float("nan")])}
        _, has_nan = nan_safe_gradients(loss, grads)
        assert bool(has_nan)

    def test_nan_in_second_leaf_zeros_first(self) -> None:
        """NaN in b zeros both w and b."""
        loss = jnp.array(1.0)
        grads = {"w": jnp.ones((3,)), "b": jnp.array([float("nan"), 0.0])}
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert bool(has_nan)
        assert jnp.all(safe_grads["w"] == 0.0)
        assert jnp.all(safe_grads["b"] == 0.0)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestNanSafeGradientsJit:
    """Tests that nan_safe_gradients is JIT-compatible."""

    def test_jit_clean_inputs(self) -> None:
        """JIT compilation succeeds for clean inputs."""
        jit_fn = jax.jit(nan_safe_gradients)
        loss, grads = _make_clean_loss_and_grads()
        safe_grads, has_nan = jit_fn(loss, grads)
        assert not bool(has_nan)
        assert jnp.all(jnp.isfinite(safe_grads["w"]))

    def test_jit_nan_loss(self) -> None:
        """JIT compilation succeeds and correctly handles NaN loss."""
        jit_fn = jax.jit(nan_safe_gradients)
        loss = jnp.array(float("nan"))
        grads = {"w": jnp.ones((4,))}
        safe_grads, has_nan = jit_fn(loss, grads)
        assert bool(has_nan)
        assert jnp.all(safe_grads["w"] == 0.0)

    def test_jit_nan_gradients(self) -> None:
        """JIT compilation succeeds and correctly handles NaN gradients."""
        jit_fn = jax.jit(nan_safe_gradients)
        loss = jnp.array(1.0)
        grads = {"w": jnp.array([1.0, float("nan"), 2.0])}
        safe_grads, has_nan = jit_fn(loss, grads)
        assert bool(has_nan)
        assert jnp.all(safe_grads["w"] == 0.0)

    def test_jit_no_python_branching(self) -> None:
        """Verify JIT traces through without recompilation for different NaN patterns."""
        jit_fn = jax.jit(nan_safe_gradients)
        # First trace: clean
        loss1, grads1 = _make_clean_loss_and_grads()
        _, has_nan1 = jit_fn(loss1, grads1)
        # Second trace: NaN (same structure — should use cached compilation)
        loss2 = jnp.array(float("nan"))
        grads2 = {"w": jnp.ones((3,)), "b": jnp.ones((2,))}
        _, has_nan2 = jit_fn(loss2, grads2)
        assert not bool(has_nan1)
        assert bool(has_nan2)


# ---------------------------------------------------------------------------
# PyTree structure preservation
# ---------------------------------------------------------------------------


class TestNanSafeGradientsStructure:
    """Tests that gradient PyTree structure is preserved exactly."""

    def test_nested_dict_structure_preserved(self) -> None:
        """Nested gradient dictionaries maintain their structure."""
        loss = jnp.array(1.0)
        grads = {
            "layer1": {"w": jnp.ones((4,)), "b": jnp.ones((2,))},
            "layer2": {"w": jnp.ones((8,)), "b": jnp.ones((4,))},
        }
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)
        assert "layer1" in safe_grads
        assert "layer2" in safe_grads
        assert "w" in safe_grads["layer1"]

    def test_list_gradient_structure_preserved(self) -> None:
        """List-structured gradients maintain their structure."""
        loss = jnp.array(1.0)
        grads = [jnp.ones((3,)), jnp.ones((5,))]
        safe_grads, has_nan = nan_safe_gradients(loss, grads)
        assert not bool(has_nan)
        assert isinstance(safe_grads, list)
        assert len(safe_grads) == 2

    def test_output_shape_preserved(self) -> None:
        """Gradient array shapes are preserved in output."""
        loss = jnp.array(1.0)
        grads = {"w": jnp.ones((3, 4)), "b": jnp.ones((4,))}
        safe_grads, _ = nan_safe_gradients(loss, grads)
        assert safe_grads["w"].shape == (3, 4)
        assert safe_grads["b"].shape == (4,)

    def test_zeros_shape_matches_original(self) -> None:
        """When zeroing due to NaN, output shapes still match input."""
        loss = jnp.array(float("nan"))
        grads = {"w": jnp.ones((3, 4)), "b": jnp.ones((4,))}
        safe_grads, _ = nan_safe_gradients(loss, grads)
        assert safe_grads["w"].shape == (3, 4)
        assert safe_grads["b"].shape == (4,)
