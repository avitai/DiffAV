"""Tests for NeRFRenderer sensor module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffav.sensor.nerf_renderer import (
    CameraPose,
    NeRFRenderer,
    NeRFRendererConfig,
    RenderedImage,
)


def test_camera_pose_fields() -> None:
    """CameraPose stores position and rotation with correct shapes."""
    pose = CameraPose(
        position=jnp.zeros(3),
        rotation=jnp.eye(3),
    )
    assert pose.position.shape == (3,)
    assert pose.rotation.shape == (3, 3)


def test_rendered_image_fields() -> None:
    """RenderedImage stores pixels and depth with correct shapes."""
    img = RenderedImage(
        pixels=jnp.zeros((8, 8, 3)),
        depth=jnp.zeros((8, 8)),
    )
    assert img.pixels.shape == (8, 8, 3)
    assert img.depth.shape == (8, 8)


def test_config_defaults() -> None:
    """NeRFRendererConfig default fields are valid."""
    cfg = NeRFRendererConfig(context_dim=16)
    assert cfg.height == 16
    assert cfg.width == 16
    assert cfg.near_plane > 0
    assert cfg.far_plane > cfg.near_plane
    assert cfg.num_samples > 0


def test_config_validates_num_samples() -> None:
    """NeRFRendererConfig raises ValueError for non-positive num_samples."""
    with pytest.raises(ValueError):
        NeRFRendererConfig(context_dim=16, num_samples=0)


def test_config_validates_far_gt_near() -> None:
    """NeRFRendererConfig raises ValueError when far_plane <= near_plane."""
    with pytest.raises(ValueError):
        NeRFRendererConfig(context_dim=16, near_plane=10.0, far_plane=5.0)


class TestNeRFRenderer:
    """Tests for NeRFRenderer forward pass."""

    @pytest.fixture(scope="class")
    def renderer(self) -> NeRFRenderer:
        """Create a small NeRFRenderer for fast testing."""
        cfg = NeRFRendererConfig(
            height=4,
            width=4,
            num_samples=4,
            context_dim=8,
            hidden_dim=16,
            num_frequencies=2,
        )
        return NeRFRenderer(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))

    def test_render_scene_shape(self, renderer: NeRFRenderer) -> None:
        """render_scene returns RenderedImage with correct shapes."""
        scene_ctx = jnp.zeros((4, 8))
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        result = renderer.render_scene(scene_ctx, pose)
        assert isinstance(result, RenderedImage)
        assert result.pixels.shape == (4, 4, 3)
        assert result.depth.shape == (4, 4)

    def test_pixels_in_range(self, renderer: NeRFRenderer) -> None:
        """Rendered pixels are in [0, 1]."""
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        result = renderer.render_scene(jnp.ones((4, 8)), pose)
        assert float(jnp.min(result.pixels)) >= 0.0
        assert float(jnp.max(result.pixels)) <= 1.0

    def test_depth_positive(self, renderer: NeRFRenderer) -> None:
        """Depth values are non-negative."""
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        result = renderer.render_scene(jnp.ones((4, 8)), pose)
        assert float(jnp.min(result.depth)) >= 0.0

    def test_gradient_flows(self, renderer: NeRFRenderer) -> None:
        """Gradients flow through render_scene into model parameters."""
        scene_ctx = jnp.ones((4, 8))
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        graphdef, state = nnx.split(renderer)

        def loss_fn(s: nnx.State) -> jax.Array:
            m = nnx.merge(graphdef, s)
            img = m.render_scene(scene_ctx, pose)
            return jnp.mean(img.pixels)

        grads = jax.grad(loss_fn)(state)
        leaves = jax.tree.leaves(grads)
        assert any(jnp.any(g != 0) for g in leaves), "All gradients are zero"

    def test_jit_compatible(self, renderer: NeRFRenderer) -> None:
        """render_scene runs under nnx.jit without error."""
        scene_ctx = jnp.ones((4, 8))
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        jitted = nnx.jit(NeRFRenderer.render_scene)
        result = jitted(renderer, scene_ctx, pose)
        assert result.pixels.shape == (4, 4, 3)

    def test_different_scene_gives_different_image(self, renderer: NeRFRenderer) -> None:
        """Different scene contexts produce different rendered images."""
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        img1 = renderer.render_scene(jnp.zeros((4, 8)), pose)
        img2 = renderer.render_scene(jnp.ones((4, 8)), pose)
        assert not jnp.allclose(img1.pixels, img2.pixels)


class TestCoordinateNormalization:
    """Ray samples must be normalized before sinusoidal encoding.

    The NeRF bands scale inputs by 2^k * pi; raw world coordinates up to
    the far plane alias severely. Samples are encoded camera-relative,
    scaled by the far plane into [-1, 1].
    """

    def test_positional_encoder_receives_normalized_coordinates(self) -> None:
        """Encoder inputs stay in [-1, 1] even for an off-origin camera."""
        config = NeRFRendererConfig(height=4, width=4, num_samples=4, context_dim=8)
        renderer = NeRFRenderer(config, rngs=nnx.Rngs(0))

        captured: list[jax.Array] = []
        original_encoder = renderer.pos_encoder

        def spying_encoder(coords: jax.Array) -> jax.Array:
            captured.append(coords)
            return original_encoder(coords)

        renderer.pos_encoder = spying_encoder  # type: ignore[assignment]

        pose = CameraPose(
            position=jnp.array([30.0, -20.0, 5.0]),
            rotation=jnp.eye(3),
        )
        renderer.render_scene(jnp.ones((4, config.context_dim)), pose)

        assert captured
        max_abs = max(float(jnp.max(jnp.abs(c))) for c in captured)
        assert max_abs <= 1.0 + 1e-5


class TestRayGeometry:
    """Camera rays must honour the configured field of view."""

    @staticmethod
    def _edge_ray_angle(width: int) -> float:
        """Angle between the optical axis and the outermost-column ray."""
        cfg = NeRFRendererConfig(
            height=3,
            width=width,
            num_samples=4,
            context_dim=8,
            hidden_dim=16,
            num_frequencies=2,
        )
        renderer = NeRFRenderer(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))
        pose = CameraPose(position=jnp.zeros(3), rotation=jnp.eye(3))
        _, dirs = renderer._generate_rays(pose)
        dirs = dirs.reshape(3, width, 3)
        edge = dirs[1, -1]  # middle row, rightmost column
        axis = jnp.array([0.0, 0.0, -1.0])
        cos = jnp.dot(edge, axis) / jnp.linalg.norm(edge)
        return float(jnp.arccos(jnp.clip(cos, -1.0, 1.0)))

    def test_edge_ray_matches_half_fov(self) -> None:
        """The outermost ray leaves at half the configured horizontal FOV."""
        from diffav.core.constants import NERF_CAMERA_HORIZONTAL_FOV_RAD

        angle = self._edge_ray_angle(width=64)
        assert angle == pytest.approx(NERF_CAMERA_HORIZONTAL_FOV_RAD / 2.0, rel=0.05)

    def test_fov_independent_of_resolution(self) -> None:
        """Doubling the image width must not change the field of view."""
        assert self._edge_ray_angle(32) == pytest.approx(self._edge_ray_angle(64), rel=1e-3)
