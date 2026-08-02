"""Tests for LiDAR ray caster."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from simulacrax.sensor.lidar import LiDARConfig, LiDARRayCaster, PointCloud


_VOXEL_RES = 8
_VOXEL = jnp.full((_VOXEL_RES, _VOXEL_RES, _VOXEL_RES), 0.5, dtype=jnp.float32)
_ORIGIN = jnp.array([0.0, 0.0, 0.0])
_ROTATION = jnp.eye(3)


class TestLiDARConfig:
    def test_defaults(self) -> None:
        cfg = LiDARConfig()
        assert cfg.num_beams > 0
        assert cfg.max_range > 0
        assert cfg.noise_std >= 0

    def test_invalid_num_beams(self) -> None:
        with pytest.raises(ValueError):
            LiDARConfig(num_beams=0)

    def test_invalid_noise_std(self) -> None:
        with pytest.raises(ValueError):
            LiDARConfig(noise_std=-0.1)

    def test_invalid_horizontal_fov(self) -> None:
        with pytest.raises(ValueError, match="horizontal_fov"):
            LiDARConfig(horizontal_fov=0.0)
        with pytest.raises(ValueError, match="horizontal_fov"):
            LiDARConfig(horizontal_fov=361.0)


class TestHorizontalFovWiring:
    """horizontal_fov constrains beam azimuths instead of being decorative."""

    @staticmethod
    def _beam_azimuths(config: LiDARConfig) -> jax.Array:
        directions = LiDARRayCaster(config)._beam_directions()
        return jnp.arctan2(directions[:, 1], directions[:, 0])

    def test_partial_fov_bounds_azimuths(self) -> None:
        azimuths = self._beam_azimuths(LiDARConfig(num_beams=64, horizontal_fov=90.0))
        half = jnp.deg2rad(45.0)
        assert jnp.all(jnp.abs(azimuths) <= half + 1e-6)

    def test_full_fov_spans_circle(self) -> None:
        azimuths = self._beam_azimuths(LiDARConfig(num_beams=64, horizontal_fov=360.0))
        span = jnp.max(azimuths) - jnp.min(azimuths)
        assert span > 1.5 * jnp.pi

    def test_narrower_fov_is_strictly_narrower(self) -> None:
        narrow = self._beam_azimuths(LiDARConfig(num_beams=64, horizontal_fov=60.0))
        wide = self._beam_azimuths(LiDARConfig(num_beams=64, horizontal_fov=180.0))
        assert float(jnp.max(jnp.abs(narrow))) < float(jnp.max(jnp.abs(wide)))

    def test_partial_fov_cast_jit_matches_eager(self) -> None:
        cfg = LiDARConfig(num_beams=16, max_range=10.0, noise_std=0.0, horizontal_fov=120.0)
        caster = LiDARRayCaster(cfg)
        voxel = jnp.ones((8, 8, 8)) * 0.5

        def points(voxel_grid: jax.Array) -> jax.Array:
            return caster.cast_rays(voxel_grid, _ORIGIN, _ROTATION, key=jax.random.key(0)).points

        eager = points(voxel)
        jitted = jax.jit(points)(voxel)
        # f32 tolerance: XLA fuses the ray-march arithmetic differently under jit
        assert jnp.allclose(eager, jitted, rtol=1e-5, atol=1e-5)


class TestLiDARRayCaster:
    @pytest.fixture(scope="class")
    def caster(self) -> LiDARRayCaster:
        return LiDARRayCaster(LiDARConfig(num_beams=16, max_range=10.0))

    def test_cast_rays_returns_point_cloud(self, caster: LiDARRayCaster) -> None:
        result = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(0))
        assert isinstance(result, PointCloud)

    def test_point_cloud_shapes_consistent(self, caster: LiDARRayCaster) -> None:
        result = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(0))
        assert result.points.ndim == 2
        assert result.points.shape[1] == 3
        assert result.intensities.shape == (result.points.shape[0],)

    def test_points_within_max_range(self, caster: LiDARRayCaster) -> None:
        result = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(0))
        dists = jnp.linalg.norm(result.points - _ORIGIN, axis=-1)
        assert float(jnp.max(dists)) <= caster.config.max_range + 1e-3

    def test_intensities_in_range(self, caster: LiDARRayCaster) -> None:
        result = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(0))
        assert float(jnp.min(result.intensities)) >= 0.0
        assert float(jnp.max(result.intensities)) <= 1.0

    def test_returns_finite_values(self, caster: LiDARRayCaster) -> None:
        result = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(0))
        assert jnp.all(jnp.isfinite(result.points))
        assert jnp.all(jnp.isfinite(result.intensities))

    def test_empty_voxel_gives_max_range_hits(self, caster: LiDARRayCaster) -> None:
        """Rays through empty space should reach max range."""
        empty = jnp.zeros((_VOXEL_RES, _VOXEL_RES, _VOXEL_RES))
        result = caster.cast_rays(empty, _ORIGIN, _ROTATION, key=jax.random.key(1))
        dists = jnp.linalg.norm(result.points - _ORIGIN, axis=-1)
        assert float(jnp.mean(dists)) > caster.config.max_range * 0.8


class TestGridFrameAndOpacity:
    """Voxel-grid anchoring and step-size-scaled opacity."""

    def test_off_grid_sensor_gets_no_phantom_returns(self) -> None:
        """Rays entirely outside the grid volume must not hit anything.

        Edge-clamped sampling would read boundary voxel densities along
        the whole ray and fabricate returns.
        """
        cfg = LiDARConfig(num_beams=16, max_range=10.0, noise_std=0.0)
        caster = LiDARRayCaster(cfg)
        dense = jnp.full((_VOXEL_RES, _VOXEL_RES, _VOXEL_RES), 1.0)
        far_origin = jnp.array([100.0, 100.0, 100.0])

        result = caster.cast_rays(dense, far_origin, _ROTATION, key=jax.random.key(0))
        dists = jnp.linalg.norm(result.points - far_origin, axis=-1)
        assert float(jnp.min(dists)) > cfg.max_range * 0.99

    def test_grid_center_translates_the_volume(self) -> None:
        """A grid centred at the sensor reproduces the origin-centred cast."""
        offset = jnp.array([100.0, -50.0, 25.0])
        voxel = jnp.full((_VOXEL_RES, _VOXEL_RES, _VOXEL_RES), 0.5)
        key = jax.random.key(3)

        origin_cfg = LiDARConfig(num_beams=16, max_range=10.0, noise_std=0.0)
        origin_result = LiDARRayCaster(origin_cfg).cast_rays(
            voxel, jnp.zeros(3), _ROTATION, key=key
        )

        shifted_cfg = LiDARConfig(
            num_beams=16,
            max_range=10.0,
            noise_std=0.0,
            grid_center=(100.0, -50.0, 25.0),
        )
        shifted_result = LiDARRayCaster(shifted_cfg).cast_rays(voxel, offset, _ROTATION, key=key)

        assert jnp.allclose(shifted_result.points - offset, origin_result.points, atol=1e-4)
        assert jnp.allclose(shifted_result.intensities, origin_result.intensities, atol=1e-5)

    def test_hit_distance_stable_under_step_refinement(self) -> None:
        """Doubling num_steps must not shift hit distances materially.

        Opacity must scale with the marching step size (Beer-Lambert);
        per-step opacity independent of step size makes absorption — and
        therefore hit distance — a function of the step count.
        """
        voxel = jnp.full((_VOXEL_RES, _VOXEL_RES, _VOXEL_RES), 0.5)

        def mean_hit_distance(num_steps: int) -> float:
            cfg = LiDARConfig(num_beams=16, max_range=10.0, num_steps=num_steps, noise_std=0.0)
            result = LiDARRayCaster(cfg).cast_rays(
                voxel, jnp.zeros(3), _ROTATION, key=jax.random.key(0)
            )
            return float(jnp.mean(jnp.linalg.norm(result.points, axis=-1)))

        coarse = mean_hit_distance(32)
        fine = mean_hit_distance(128)
        assert abs(coarse - fine) / coarse < 0.1


class TestTransformCompatibility:
    """cast_rays is a JAX path: jit, grad, and determinism must hold."""

    def test_jit_matches_eager(self) -> None:
        cfg = LiDARConfig(num_beams=16, max_range=10.0, noise_std=0.0)
        caster = LiDARRayCaster(cfg)

        def points(voxel: jax.Array) -> jax.Array:
            return caster.cast_rays(voxel, _ORIGIN, _ROTATION, key=jax.random.key(0)).points

        eager = points(_VOXEL)
        jitted = jax.jit(points)(_VOXEL)
        # float32 tolerance: XLA fuses the ray-march arithmetic differently
        # under jit (observed max |diff| ~8e-7 on CPU); near-zero coordinate
        # components need the absolute term.
        assert jnp.allclose(eager, jitted, rtol=1e-5, atol=1e-5)

    def test_gradient_flows_to_voxel_density(self) -> None:
        cfg = LiDARConfig(num_beams=16, max_range=10.0, noise_std=0.0)
        caster = LiDARRayCaster(cfg)

        def mean_distance(voxel: jax.Array) -> jax.Array:
            cloud = caster.cast_rays(voxel, _ORIGIN, _ROTATION, key=jax.random.key(0))
            return jnp.mean(jnp.linalg.norm(cloud.points, axis=-1))

        grad = jax.grad(mean_distance)(_VOXEL)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0

    def test_deterministic(self) -> None:
        cfg = LiDARConfig(num_beams=16, max_range=10.0)
        caster = LiDARRayCaster(cfg)
        first = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(5))
        second = caster.cast_rays(_VOXEL, _ORIGIN, _ROTATION, key=jax.random.key(5))
        assert jnp.array_equal(first.points, second.points)
        assert jnp.array_equal(first.intensities, second.intensities)
