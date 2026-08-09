"""Weather augmentation operators for sensor simulation.

Provides JAX-native weather effects that can be composed into realistic
sensor degradation pipelines for autonomous-driving data augmentation:

- Rain streaks (sinusoidal alpha-blend mask)
- Fog/haze (Beer-Lambert atmospheric scattering)
- Glare/bloom (Gaussian radial bloom at a configurable position)

All operators extend :class:`datarax.core.modality.ModalityOperator` and
follow the standard ``apply()`` pattern: extract → transform → clip → remap.

Examples:
    Deterministic fog augmentation:

    ```python
    from flax import nnx
    from diffav.sensor.weather import FogAugmentation, FogConfig

    cfg = FogConfig(field_key="image", intensity=0.6, stochastic=False)
    op = FogAugmentation(cfg, rngs=nnx.Rngs(0))
    out, _, _ = op.apply({"image": image}, {}, {})
    ```

    Composing rain → fog → glare into a single pipeline:

    ```python
    from datarax.operators.composite_operator import (
        CompositeOperatorModule,
        CompositeOperatorConfig,
        CompositionStrategy,
    )
    from diffav.sensor.weather import (
        RainAugmentation, RainConfig,
        FogAugmentation, FogConfig,
        GlareAugmentation, GlareConfig,
    )

    rain = RainAugmentation(RainConfig(field_key="image"), rngs=nnx.Rngs(0))
    fog  = FogAugmentation(FogConfig(field_key="image"),  rngs=nnx.Rngs(1))
    glare = GlareAugmentation(GlareConfig(field_key="image"), rngs=nnx.Rngs(2))
    pipeline = CompositeOperatorModule(
        CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL),
        operators=[rain, fog, glare],
    )
    ```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from flax import nnx

from diffav.core.constants import FOG_EXTINCTION_COEFFICIENT


# ---------------------------------------------------------------------------
# Rain
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RainConfig(ModalityOperatorConfig):
    """Configuration for :class:`RainAugmentation`.

    Extends :class:`~datarax.core.modality.ModalityOperatorConfig` with
    rain-streak parameters.

    Attributes:
        clip_range: Output value range; defaults to ``(0.0, 1.0)`` for
            normalised images.
        intensity: Alpha-blend strength of rain streaks in ``[0, 1]``.
        streak_density: Number of streak bands across the image
            (higher → denser pattern).
        streak_length: Relative streak length as a fraction of image diagonal.
        streak_angle_deg: Streak inclination from vertical in degrees.
    """

    # Override parent default so normalised images are clipped automatically.
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    intensity: float = field(default=0.5, kw_only=True)
    streak_density: float = field(default=20.0, kw_only=True)
    streak_length: float = field(default=0.1, kw_only=True)
    streak_angle_deg: float = field(default=15.0, kw_only=True)

    def __post_init__(self) -> None:
        """Validate rain configuration parameters."""
        # Explicit super(): @dataclass(slots=True) rebuilds the class, breaking
        # the zero-arg form's __class__ cell.
        super(RainConfig, self).__post_init__()
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if self.streak_density <= 0.0:
            raise ValueError(f"streak_density must be > 0, got {self.streak_density}")


class RainAugmentation(ModalityOperator):
    """Rain-streak image augmentation operator.

    Generates a synthetic rain-streak mask via a sinusoidal pattern at the
    configured angle and frequency, then alpha-blends it with the input image:

        output = image * (1 - intensity * mask) + mask * intensity

    This brightens pixels along streak paths (white streaks on bright images,
    visible streaks on dark images).

    Supports deterministic and stochastic modes (inherits from
    :class:`~datarax.core.modality.ModalityOperator`).

    Examples:
        Deterministic:

        ```python
        cfg = RainConfig(field_key="image", intensity=0.4, stochastic=False)
        op = RainAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": image}, {}, {})
        ```
    """

    def __init__(self, config: RainConfig, *, rngs: nnx.Rngs) -> None:
        """Initialise RainAugmentation.

        Args:
            config: Rain augmentation configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: RainConfig = config  # type narrowing for pyright

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply rain-streak augmentation to the configured image field.

        Args:
            data: Element data dict; must contain ``config.field_key``.
            state: Operator state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Optional per-sample random parameters.  When
                ``stochastic=True`` the ``"intensity"`` key overrides the
                config intensity.
            stats: Unused; accepted for datarax ModalityOperator API
                compatibility (keyword callers must not break).

        Returns:
            ``(transformed_data, state, metadata)``
        """
        del stats
        image = self._extract_field(data, self.config.field_key)

        # Resolve effective intensity.
        if self.config.stochastic and random_params is not None:
            intensity = random_params.get("intensity", self.config.intensity)
        else:
            intensity = self.config.intensity

        height, width = image.shape[0], image.shape[1]

        # Build coordinate grids (normalised to [0, 1]).
        y_coords = jnp.linspace(0.0, 1.0, height)
        x_coords = jnp.linspace(0.0, 1.0, width)
        xx, yy = jnp.meshgrid(x_coords, y_coords)

        # Rotate coordinates by streak angle.
        angle_rad = self.config.streak_angle_deg * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotated = cos_a * xx + sin_a * yy  # (H, W)

        # Sinusoidal streak pattern; streak_density is the band count
        # across the image, so higher values mean denser streaks.
        mask = (jnp.sin(rotated * self.config.streak_density * 2.0 * math.pi) + 1.0) * 0.5

        # Apply streak length as a soft threshold (values above threshold → streaks).
        threshold = 1.0 - self.config.streak_length
        mask = jnp.where(mask > threshold, (mask - threshold) / max(1.0 - threshold, 1e-6), 0.0)

        # Broadcast mask to match image channels.
        if image.ndim == 3:
            mask = mask[..., None]  # (H, W, 1)

        augmented = image * (1.0 - intensity * mask) + intensity * mask
        clipped = self._apply_clip_range(augmented)
        return self._remap_field(data, clipped), state, metadata

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        """Generate per-sample intensity values for stochastic mode.

        Args:
            rng: JAX PRNG key.
            data_shapes: Dict mapping field keys to array shapes (batch first).

        Returns:
            ``{"intensity": array of shape (batch_size,)}``
        """
        batch_size = data_shapes[self.config.field_key][0]
        intensity = jax.random.uniform(
            rng, shape=(batch_size,), minval=0.0, maxval=self.config.intensity
        )
        return {"intensity": intensity}


# ---------------------------------------------------------------------------
# Fog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class FogConfig(ModalityOperatorConfig):
    """Configuration for :class:`FogAugmentation`.

    Attributes:
        clip_range: Output value range; defaults to ``(0.0, 1.0)``.
        intensity: Fog density in ``[0, 1]``.  Higher values produce
            more opaque fog.
        fog_color: RGB fog colour as a 3-tuple of floats in ``[0, 1]``.
            Defaults to a slightly bluish white ``(0.9, 0.9, 0.95)``.
    """

    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    intensity: float = field(default=0.4, kw_only=True)
    fog_color: tuple[float, float, float] = field(default=(0.9, 0.9, 0.95), kw_only=True)

    def __post_init__(self) -> None:
        """Validate fog configuration parameters."""
        # Explicit super(): @dataclass(slots=True) rebuilds the class, breaking
        # the zero-arg form's __class__ cell.
        super(FogConfig, self).__post_init__()
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if len(self.fog_color) != 3:
            raise ValueError(f"fog_color must be a 3-tuple, got {self.fog_color}")


class FogAugmentation(ModalityOperator):
    """Fog / haze image augmentation operator.

    Applies Beer-Lambert atmospheric scattering:

        transmission = exp(-intensity * FOG_EXTINCTION_COEFFICIENT)
        output = image * transmission + fog_color * (1 - transmission)

    A dark image with high intensity converges toward ``fog_color``.

    Examples:
        ```python
        cfg = FogConfig(field_key="image", intensity=0.6, stochastic=False)
        op = FogAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": image}, {}, {})
        ```
    """

    def __init__(self, config: FogConfig, *, rngs: nnx.Rngs) -> None:
        """Initialise FogAugmentation.

        Args:
            config: Fog augmentation configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: FogConfig = config

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply fog augmentation to the configured image field.

        Args:
            data: Element data dict; must contain ``config.field_key``.
            state: Operator state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Optional per-sample random parameters.  When
                ``stochastic=True`` the ``"intensity"`` key overrides config.
            stats: Unused; accepted for datarax ModalityOperator API
                compatibility (keyword callers must not break).

        Returns:
            ``(transformed_data, state, metadata)``
        """
        del stats
        image = self._extract_field(data, self.config.field_key)

        if self.config.stochastic and random_params is not None:
            intensity = random_params.get("intensity", self.config.intensity)
        else:
            intensity = self.config.intensity

        # Beer-Lambert transmission factor.
        transmission = jnp.exp(-intensity * FOG_EXTINCTION_COEFFICIENT)

        fog_color = jnp.array(self.config.fog_color, dtype=image.dtype)
        # Broadcast fog_color to (1, 1, 3) when image is (H, W, C).
        if image.ndim == 3:
            fog_color = fog_color[None, None, :]  # (1, 1, 3)

        augmented = image * transmission + fog_color * (1.0 - transmission)
        clipped = self._apply_clip_range(augmented)
        return self._remap_field(data, clipped), state, metadata

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        """Generate per-sample intensity values for stochastic mode.

        Args:
            rng: JAX PRNG key.
            data_shapes: Dict mapping field keys to array shapes (batch first).

        Returns:
            ``{"intensity": array of shape (batch_size,)}``
        """
        batch_size = data_shapes[self.config.field_key][0]
        intensity = jax.random.uniform(
            rng, shape=(batch_size,), minval=0.0, maxval=self.config.intensity
        )
        return {"intensity": intensity}


# ---------------------------------------------------------------------------
# Glare
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GlareConfig(ModalityOperatorConfig):
    """Configuration for :class:`GlareAugmentation`.

    Attributes:
        clip_range: Output value range; defaults to ``(0.0, 1.0)``.
        intensity: Peak bloom brightness added at the glare centre, in ``[0, 1]``.
        glare_position: Normalised (x, y) position of the glare centre where
            ``(0, 0)`` is the top-left corner.  Defaults to ``(0.5, 0.2)``
            (upper-centre, typical sun position).
        glare_radius: Bloom radius as a fraction of the image diagonal.
    """

    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    intensity: float = field(default=0.3, kw_only=True)
    glare_position: tuple[float, float] = field(default=(0.5, 0.2), kw_only=True)
    glare_radius: float = field(default=0.3, kw_only=True)

    def __post_init__(self) -> None:
        """Validate glare configuration parameters."""
        # Explicit super(): @dataclass(slots=True) rebuilds the class, breaking
        # the zero-arg form's __class__ cell.
        super(GlareConfig, self).__post_init__()
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if self.glare_radius <= 0.0:
            raise ValueError(f"glare_radius must be > 0, got {self.glare_radius}")


class GlareAugmentation(ModalityOperator):
    """Glare / bloom image augmentation operator.

    Adds a Gaussian brightness bloom centred at ``glare_position``:

        sigma  = glare_radius * diagonal
        bloom  = intensity * exp(-(d^2) / (2 * sigma^2))
        output = image + bloom

    where ``d`` is the per-pixel distance from the glare centre and
    ``diagonal = sqrt(H^2 + W^2)``.

    Examples:
        ```python
        cfg = GlareConfig(field_key="image", intensity=0.4, stochastic=False)
        op = GlareAugmentation(cfg, rngs=nnx.Rngs(0))
        out, _, _ = op.apply({"image": image}, {}, {})
        ```
    """

    def __init__(self, config: GlareConfig, *, rngs: nnx.Rngs) -> None:
        """Initialise GlareAugmentation.

        Args:
            config: Glare augmentation configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: GlareConfig = config

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Apply glare / bloom augmentation to the configured image field.

        Args:
            data: Element data dict; must contain ``config.field_key``.
            state: Operator state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Optional per-sample random parameters.  When
                ``stochastic=True`` the ``"intensity"`` key overrides config.
            stats: Unused; accepted for datarax ModalityOperator API
                compatibility (keyword callers must not break).

        Returns:
            ``(transformed_data, state, metadata)``
        """
        del stats
        image = self._extract_field(data, self.config.field_key)

        if self.config.stochastic and random_params is not None:
            intensity = random_params.get("intensity", self.config.intensity)
        else:
            intensity = self.config.intensity

        height, width = image.shape[0], image.shape[1]
        diagonal = math.sqrt(height**2 + width**2)
        sigma = self.config.glare_radius * diagonal

        # Pixel coordinate grids (in absolute pixels).
        y_coords = jnp.arange(height, dtype=jnp.float32)
        x_coords = jnp.arange(width, dtype=jnp.float32)
        xx, yy = jnp.meshgrid(x_coords, y_coords)

        # Glare centre in absolute pixel coordinates.
        cx = self.config.glare_position[0] * (width - 1)
        cy = self.config.glare_position[1] * (height - 1)

        # Squared distance from glare centre.
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2  # (H, W)

        # Gaussian bloom (H, W).
        bloom = intensity * jnp.exp(-dist_sq / (2.0 * sigma**2 + 1e-8))

        if image.ndim == 3:
            bloom = bloom[..., None]  # (H, W, 1) → broadcast over channels

        augmented = image + bloom
        clipped = self._apply_clip_range(augmented)
        return self._remap_field(data, clipped), state, metadata

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        """Generate per-sample intensity values for stochastic mode.

        Args:
            rng: JAX PRNG key.
            data_shapes: Dict mapping field keys to array shapes (batch first).

        Returns:
            ``{"intensity": array of shape (batch_size,)}``
        """
        batch_size = data_shapes[self.config.field_key][0]
        intensity = jax.random.uniform(
            rng, shape=(batch_size,), minval=0.0, maxval=self.config.intensity
        )
        return {"intensity": intensity}
