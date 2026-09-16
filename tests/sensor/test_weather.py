"""Tests for weather augmentation operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffav.sensor.weather import (
    FogAugmentation,
    FogConfig,
    GlareAugmentation,
    GlareConfig,
    RainAugmentation,
    RainConfig,
)


_H, _W = 8, 8
_IMAGE = jnp.full((_H, _W, 3), 0.5, dtype=jnp.float32)


class TestRainAugmentation:
    def test_output_shape_unchanged(self) -> None:
        cfg = RainConfig(field_key="image", intensity=0.3, stochastic=False)
        op = RainAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert out["image"].shape == (_H, _W, 3)

    def test_output_clipped_to_unit(self) -> None:
        cfg = RainConfig(field_key="image", intensity=0.5, stochastic=False)
        op = RainAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert float(jnp.min(out["image"])) >= 0.0
        assert float(jnp.max(out["image"])) <= 1.0

    def test_output_differs_from_input(self) -> None:
        cfg = RainConfig(field_key="image", intensity=0.5, stochastic=False)
        op = RainAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert not jnp.allclose(out["image"], _IMAGE)


class TestFogAugmentation:
    def test_output_shape_unchanged(self) -> None:
        cfg = FogConfig(field_key="image", intensity=0.4, stochastic=False)
        op = FogAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert out["image"].shape == (_H, _W, 3)

    def test_fog_brightens_dark_image(self) -> None:
        dark = jnp.zeros((_H, _W, 3))
        cfg = FogConfig(field_key="image", intensity=0.8, stochastic=False)
        op = FogAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": dark}, {}, {})
        assert float(jnp.mean(out["image"])) > 0.0

    def test_output_clipped_to_unit(self) -> None:
        cfg = FogConfig(field_key="image", intensity=0.4, stochastic=False)
        op = FogAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert float(jnp.min(out["image"])) >= 0.0
        assert float(jnp.max(out["image"])) <= 1.0


class TestGlareAugmentation:
    def test_output_shape_unchanged(self) -> None:
        cfg = GlareConfig(field_key="image", intensity=0.3, stochastic=False)
        op = GlareAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert out["image"].shape == (_H, _W, 3)

    def test_output_clipped_to_unit(self) -> None:
        cfg = GlareConfig(field_key="image", intensity=0.3, stochastic=False)
        op = GlareAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert float(jnp.min(out["image"])) >= 0.0
        assert float(jnp.max(out["image"])) <= 1.0

    def test_glare_increases_brightness(self) -> None:
        cfg = GlareConfig(field_key="image", intensity=0.5, stochastic=False)
        op = GlareAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert float(jnp.mean(out["image"])) >= float(jnp.mean(_IMAGE))


class TestStreakDensitySemantics:
    """Higher streak_density must mean denser streaks, as documented."""

    @staticmethod
    def _streak_transitions(density: float) -> int:
        """Count streak-band transitions across the augmented image."""
        image = jnp.zeros((64, 64, 3))
        cfg = RainConfig(
            field_key="image",
            intensity=1.0,
            streak_density=density,
            streak_length=0.5,
            stochastic=False,
        )
        op = RainAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": image}, {}, {})
        streak_mask = (out["image"][:, :, 0] > 1e-3).astype(jnp.int32)
        return int(jnp.sum(jnp.abs(jnp.diff(streak_mask, axis=1))))

    def test_higher_density_gives_more_streak_bands(self) -> None:
        """density=20 must produce more streak transitions than density=5."""
        sparse = self._streak_transitions(5.0)
        dense = self._streak_transitions(20.0)
        assert dense > sparse
        assert sparse > 0


class TestStochasticIntensity:
    """In stochastic mode each record's intensity is drawn from the key its caller passes."""

    @staticmethod
    def _operators() -> list:
        stochastic = {"field_key": "image", "stochastic": True, "stream_name": "weather"}
        rngs = nnx.Rngs(0, weather=1)
        return [
            RainAugmentation(RainConfig(**stochastic), rngs=rngs),
            FogAugmentation(FogConfig(**stochastic), rngs=rngs),
            GlareAugmentation(GlareConfig(**stochastic), rngs=rngs),
        ]

    def test_a_stochastic_operator_refuses_a_missing_key(self) -> None:
        for op in self._operators():
            with pytest.raises(ValueError, match=type(op).__name__):
                op.apply({"image": _IMAGE}, {}, {})

    def test_the_draw_follows_the_key(self) -> None:
        for op in self._operators():
            first, _, _ = op.apply({"image": _IMAGE}, {}, {}, jax.random.key(1))
            again, _, _ = op.apply({"image": _IMAGE}, {}, {}, jax.random.key(1))
            other, _, _ = op.apply({"image": _IMAGE}, {}, {}, jax.random.key(2))
            assert jnp.array_equal(first["image"], again["image"])
            assert not jnp.array_equal(first["image"], other["image"])

    def test_a_deterministic_operator_ignores_the_key(self) -> None:
        op = FogAugmentation(FogConfig(field_key="image", intensity=0.4), rngs=nnx.Rngs(0))
        keyed, _, _ = op.apply({"image": _IMAGE}, {}, {}, jax.random.key(3))
        plain, _, _ = op.apply({"image": _IMAGE}, {}, {})
        assert jnp.array_equal(keyed["image"], plain["image"])


class TestStatsKwarg:
    """apply() must accept the base class's ``stats`` keyword."""

    def test_all_operators_accept_stats_keyword(self) -> None:
        """Keyword callers of the datarax ModalityOperator API must work."""
        image = jnp.full((8, 8, 3), 0.5)
        operators = [
            RainAugmentation(RainConfig(field_key="image"), rngs=nnx.Rngs(0)),
            FogAugmentation(FogConfig(field_key="image"), rngs=nnx.Rngs(0)),
            GlareAugmentation(GlareConfig(field_key="image"), rngs=nnx.Rngs(0)),
        ]
        for op in operators:
            out, _, _ = op.apply({"image": image}, {}, {}, stats={"unused": 1})
            assert out["image"].shape == image.shape


class TestWeatherTransformCompatibility:
    """Augmentations are JAX paths: jit and image gradients must hold."""

    def test_rain_jit_matches_eager(self) -> None:
        op = RainAugmentation(RainConfig(field_key="image"), rngs=nnx.Rngs(0))
        image = jnp.full((16, 16, 3), 0.5)

        def augment(img: jax.Array) -> jax.Array:
            out, _, _ = op.apply({"image": img}, {}, {})
            return out["image"]

        eager = augment(image)
        jitted = jax.jit(augment)(image)
        # Small absolute tolerance: GPU kernel fusion reorders the mask
        # arithmetic slightly at streak edges.
        assert jnp.allclose(eager, jitted, rtol=1e-5, atol=1e-5)

    def test_fog_gradient_flows_to_image(self) -> None:
        op = FogAugmentation(FogConfig(field_key="image"), rngs=nnx.Rngs(0))
        image = jnp.full((8, 8, 3), 0.5)

        def brightness(img: jax.Array) -> jax.Array:
            out, _, _ = op.apply({"image": img}, {}, {})
            return jnp.mean(out["image"])

        grad = jax.grad(brightness)(image)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0
