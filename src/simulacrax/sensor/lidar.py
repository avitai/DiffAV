"""Differentiable LiDAR ray caster for sensor simulation.

Casts beams from a configurable spherical scan pattern through a voxel density
grid and returns a point cloud at the first high-density intersection.

The computation is pure JAX — no Python loops — so the module is fully
JIT-compatible and gradients flow through the voxel density field.  The voxel
density grid can be supplied directly or produced by
:class:`artifex.generative_models.models.geometric.voxel.VoxelModel`.

Examples:
    Basic usage with a uniform density grid:

    ```python
    import jax
    import jax.numpy as jnp
    from simulacrax.sensor.lidar import LiDARConfig, LiDARRayCaster

    caster = LiDARRayCaster(LiDARConfig(num_beams=64, max_range=50.0))
    voxel = jnp.full((16, 16, 16), 0.3)  # uniform density
    cloud = caster.cast_rays(
        voxel,
        sensor_position=jnp.zeros(3),
        sensor_rotation=jnp.eye(3),
        key=jax.random.key(0),
    )
    # cloud.points: (64, 3)  — world-space hit positions
    # cloud.intensities: (64,) — per-beam return intensity in [0, 1]
    ```
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from simulacrax.core.config import validate_positive
from simulacrax.core.constants import LIDAR_HIT_WEIGHT_THRESHOLD_FACTOR


@dataclass(frozen=True, slots=True, kw_only=True)
class PointCloud:
    """LiDAR point cloud output.

    Attributes:
        points: Hit positions in world space, shape ``(N, 3)``.
        intensities: Return intensity at each hit, shape ``(N,)``, in ``[0, 1]``.
    """

    points: jax.Array
    intensities: jax.Array


jax.tree_util.register_dataclass(
    PointCloud,
    data_fields=["points", "intensities"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LiDARConfig:
    """Configuration for :class:`LiDARRayCaster`.

    Attributes:
        num_beams: Number of laser beams (spread across a spherical grid).
        horizontal_fov: Full horizontal field of view in degrees.
        vertical_fov_up: Upward vertical FoV in degrees from horizontal.
        vertical_fov_down: Downward vertical FoV in degrees from horizontal.
        max_range: Maximum detection range in world units.
        num_steps: Number of ray-marching steps per beam.
        density_threshold: Cumulated density above which a hit is recorded.
        noise_std: Standard deviation of Gaussian distance noise (world units).
        grid_center: World-space centre of the voxel density grid. The grid
            spans ``[grid_center - max_range, grid_center + max_range]`` on
            each axis; samples outside it contribute zero density.
    """

    num_beams: int = 64
    horizontal_fov: float = 360.0
    vertical_fov_up: float = 15.0
    vertical_fov_down: float = 15.0
    max_range: float = 50.0
    num_steps: int = 32
    density_threshold: float = 0.3
    noise_std: float = 0.01
    grid_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        validate_positive("num_beams", self.num_beams)
        validate_positive("max_range", self.max_range)
        validate_positive("num_steps", self.num_steps)
        if not 0.0 < self.horizontal_fov <= 360.0:
            msg = f"horizontal_fov must be in (0, 360] degrees, got {self.horizontal_fov}"
            raise ValueError(msg)
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")


class LiDARRayCaster:
    """Differentiable spherical LiDAR ray caster.

    Casts ``num_beams`` rays arranged in a spherical scan pattern.  For each
    ray, density is accumulated along the march; the hit point is computed as
    the expectation over a softened transmittance distribution, which is fully
    differentiable with respect to the voxel density field.

    Beams with cumulated density below ``density_threshold`` are clipped to
    ``max_range`` (no hit).

    Args:
        config: LiDAR configuration.
    """

    def __init__(self, config: LiDARConfig) -> None:
        """Initialise LiDARRayCaster.

        Args:
            config: LiDAR configuration.
        """
        self.config = config

    def _beam_directions(self) -> jax.Array:
        """Generate unit-direction vectors for all beams.

        Arranges beams in a spherical grid spanning the configured FoV.

        Returns:
            Direction array of shape ``(num_beams, 3)``.
        """
        cfg = self.config
        n_beams = cfg.num_beams

        # Build an approximate square azimuth × elevation grid.
        n_az = max(1, int(n_beams**0.5))
        n_el = max(1, (n_beams + n_az - 1) // n_az)

        # Full circle uses endpoint=False so -pi and +pi don't duplicate a beam;
        # a partial FoV includes both edges of the configured wedge.
        half_az = math.radians(cfg.horizontal_fov) / 2.0
        full_circle = cfg.horizontal_fov >= 360.0
        az = jnp.linspace(-half_az, half_az, n_az, endpoint=not full_circle)
        el = jnp.linspace(
            -jnp.deg2rad(cfg.vertical_fov_down),
            jnp.deg2rad(cfg.vertical_fov_up),
            n_el,
        )
        az_g, el_g = jnp.meshgrid(az, el)
        az_flat = az_g.ravel()[:n_beams]
        el_flat = el_g.ravel()[:n_beams]

        cos_el = jnp.cos(el_flat)
        dx = cos_el * jnp.cos(az_flat)
        dy = cos_el * jnp.sin(az_flat)
        dz = jnp.sin(el_flat)
        dirs = jnp.stack([dx, dy, dz], axis=-1)  # (B, 3)
        return dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)

    def _sample_voxel(self, voxel: jax.Array, pts: jax.Array) -> jax.Array:
        """Trilinearly sample density from a voxel grid.

        The grid spans ``[grid_center - max_range, grid_center + max_range]``
        per axis; samples outside that volume return zero density instead of
        clamping to edge voxels (which would fabricate phantom returns for
        off-grid sensors).

        Args:
            voxel: Density grid of shape ``(R, R, R)``.
            pts: World-space sample positions, shape ``(N, 3)``.

        Returns:
            Density values, shape ``(N,)``.
        """
        r = voxel.shape[0]
        center = jnp.array(self.config.grid_center)
        max_range = self.config.max_range

        local = pts - center[None, :]
        in_bounds = jnp.all(jnp.abs(local) <= max_range, axis=-1)  # (N,)

        # Map grid-local coords ``[-max_range, max_range]`` → ``[0, R-1]``.
        idx = (local + max_range) / (2.0 * max_range) * (r - 1)
        idx = jnp.clip(idx, 0.0, float(r - 1))

        i0 = jnp.floor(idx).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, r - 1)
        frac = idx - jnp.floor(idx)  # fractional part, (N, 3)

        # Eight corners of the voxel cell.
        d000 = voxel[i0[:, 0], i0[:, 1], i0[:, 2]]
        d001 = voxel[i0[:, 0], i0[:, 1], i1[:, 2]]
        d010 = voxel[i0[:, 0], i1[:, 1], i0[:, 2]]
        d011 = voxel[i0[:, 0], i1[:, 1], i1[:, 2]]
        d100 = voxel[i1[:, 0], i0[:, 1], i0[:, 2]]
        d101 = voxel[i1[:, 0], i0[:, 1], i1[:, 2]]
        d110 = voxel[i1[:, 0], i1[:, 1], i0[:, 2]]
        d111 = voxel[i1[:, 0], i1[:, 1], i1[:, 2]]

        tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]
        density = (
            d000 * (1 - tx) * (1 - ty) * (1 - tz)
            + d001 * (1 - tx) * (1 - ty) * tz
            + d010 * (1 - tx) * ty * (1 - tz)
            + d011 * (1 - tx) * ty * tz
            + d100 * tx * (1 - ty) * (1 - tz)
            + d101 * tx * (1 - ty) * tz
            + d110 * tx * ty * (1 - tz)
            + d111 * tx * ty * tz
        )
        return jnp.where(in_bounds, density, 0.0)

    def cast_rays(
        self,
        voxel_density: jax.Array,
        sensor_position: jax.Array,
        sensor_rotation: jax.Array,
        *,
        key: jax.Array,
    ) -> PointCloud:
        """Cast all beams through a voxel density grid.

        Args:
            voxel_density: Density grid of shape ``(R, R, R)`` with values in
                ``[0, 1]``.  Produced by
                :meth:`artifex.generative_models.models.geometric.voxel.VoxelModel.generate`
                or supplied directly.
            sensor_position: World-space sensor origin, shape ``(3,)``.
            sensor_rotation: Sensor-to-world rotation matrix, shape ``(3, 3)``.
            key: JAX random key for optional distance noise.

        Returns:
            :class:`PointCloud` with hit positions ``(N, 3)`` and intensities
            ``(N,)``.
        """
        cfg = self.config

        # Rotate beam directions into world space.
        dirs_local = self._beam_directions()  # (B, 3)
        dirs_world = dirs_local @ sensor_rotation.T  # (B, 3)

        n_beams = dirs_world.shape[0]
        t_vals = jnp.linspace(0.0, cfg.max_range, cfg.num_steps)  # (S,)

        # Sample positions along each beam: (B, S, 3).
        pts = sensor_position[None, None, :] + dirs_world[:, None, :] * t_vals[None, :, None]

        # Query voxel densities at all sample points.
        pts_flat = pts.reshape(n_beams * cfg.num_steps, 3)
        densities_flat = self._sample_voxel(voxel_density, pts_flat)  # (B*S,)
        densities = densities_flat.reshape(n_beams, cfg.num_steps)  # (B, S)

        # Soft first-hit via transmittance weighting (differentiable).
        # Beer-Lambert opacity scaled by the marching step size, so the
        # absorption profile is invariant to num_steps.
        step_size = cfg.max_range / (cfg.num_steps - 1) if cfg.num_steps > 1 else cfg.max_range
        alpha = 1.0 - jnp.exp(-jnp.clip(densities, 0.0, None) * step_size)
        transmittance = jnp.cumprod(
            jnp.concatenate([jnp.ones((n_beams, 1)), 1.0 - alpha[:, :-1] + 1e-6], axis=1),
            axis=1,
        )  # (B, S)
        weights = alpha * transmittance  # (B, S)

        weights_sum = jnp.sum(weights, axis=1, keepdims=True) + 1e-8
        weights_norm = weights / weights_sum  # (B, S)

        # Expected hit distance and density-weighted intensity.
        hit_t = jnp.sum(weights_norm * t_vals[None, :], axis=1)  # (B,)
        hit_density = jnp.sum(weights_norm * densities, axis=1)  # (B,)

        # Clip beams with insufficient cumulated density to max_range (no hit).
        hit_mask = (
            jnp.sum(weights, axis=1) > cfg.density_threshold * LIDAR_HIT_WEIGHT_THRESHOLD_FACTOR
        )
        hit_t = jnp.where(hit_mask, hit_t, cfg.max_range)

        # Gaussian range noise.
        noise = jax.random.normal(key, (n_beams,)) * cfg.noise_std
        hit_t = jnp.clip(hit_t + noise, 0.0, cfg.max_range)

        # Convert distances to world-space positions.
        hit_points = sensor_position[None, :] + dirs_world * hit_t[:, None]  # (B, 3)
        intensities = jnp.clip(hit_density, 0.0, 1.0)

        return PointCloud(points=hit_points, intensities=intensities)
