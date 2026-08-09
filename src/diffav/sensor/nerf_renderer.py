"""NeRF-based differentiable scene rendering for sensor simulation.

Uses sinusoidal positional encoding (opifex) and scene-conditioned
``StandardMLP`` networks (opifex, ``snake`` activation) to render RGB images
and depth maps from scene contexts and camera poses.  The ``snake`` periodic
activation is well-suited to NeRF's sinusoidal positional encodings.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from opifex.neural.base import StandardMLP
from opifex.neural.operators.common.embeddings import SinusoidalEmbedding

from diffav.core.config import validate_positive
from diffav.core.constants import NERF_CAMERA_HORIZONTAL_FOV_RAD


_COORD_DIM = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraPose:
    """Pinhole camera pose in world space.

    Attributes:
        position: World-space sensor origin, shape ``(3,)``.
        rotation: Camera-to-world rotation matrix, shape ``(3, 3)``.
    """

    position: jax.Array
    rotation: jax.Array


jax.tree_util.register_dataclass(
    CameraPose,
    data_fields=["position", "rotation"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedImage:
    """Output of a NeRF render pass.

    Attributes:
        pixels: RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.
        depth: Per-pixel depth, shape ``(H, W,)``, in world units.
    """

    pixels: jax.Array
    depth: jax.Array


jax.tree_util.register_dataclass(
    RenderedImage,
    data_fields=["pixels", "depth"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NeRFRendererConfig:
    """Configuration for NeRFRenderer.

    Attributes:
        height: Output image height in pixels.
        width: Output image width in pixels.
        near_plane: Ray near-clipping distance (world units).
        far_plane: Ray far-clipping distance (world units).
        num_samples: Number of sample points per ray.
        context_dim: Dimensionality of scene context embedding.
        hidden_dim: Width of NeRF MLP hidden layers.
        num_frequencies: NeRF positional encoding frequencies.
    """

    height: int = 16
    width: int = 16
    near_plane: float = 0.1
    far_plane: float = 50.0
    num_samples: int = 16
    context_dim: int = 64
    hidden_dim: int = 32
    num_frequencies: int = 6

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        validate_positive("num_samples", self.num_samples)
        validate_positive("context_dim", self.context_dim)
        validate_positive("hidden_dim", self.hidden_dim)
        validate_positive("num_frequencies", self.num_frequencies)
        if self.near_plane <= 0:
            raise ValueError(f"near_plane must be positive, got {self.near_plane}")
        if self.far_plane <= self.near_plane:
            raise ValueError(
                f"far_plane ({self.far_plane}) must be > near_plane ({self.near_plane})"
            )


class NeRFRenderer(nnx.Module):
    """Scene-conditioned NeRF renderer.

    Encodes camera rays using sinusoidal positional encoding from opifex and
    decodes scene-conditioned density and colour using ``StandardMLP`` networks
    (opifex) with ``snake`` activation, which is well-matched to the periodic
    structure of NeRF positional encodings.

    Architecture:
    - ``scene_mlp``: compresses mean-pooled scene context to ``hidden_dim``.
    - ``field_mlp``: maps ``[encoded_pos | scene_latent]`` to 4 outputs —
      ``[density, r, g, b]``.  Final activations (softplus / sigmoid) are
      applied in ``render_scene``.

    Args:
        config: Renderer configuration.
        rngs: NNX random number generators.
    """

    def __init__(self, config: NeRFRendererConfig, *, rngs: nnx.Rngs) -> None:
        """Initialise NeRFRenderer.

        Args:
            config: Renderer configuration.
            rngs: NNX random number generators.
        """
        self.config = config

        # Positional encoder for 3D ray sample coordinates
        self.pos_encoder = SinusoidalEmbedding(
            in_channels=_COORD_DIM,
            num_frequencies=config.num_frequencies,
            embedding_type="nerf",
        )
        pos_encoded_dim = self.pos_encoder.out_channels  # 2 * freq * 3

        # Scene projector: mean-pool context tokens → hidden_dim latent
        self.scene_mlp = StandardMLP(
            [config.context_dim, config.hidden_dim, config.hidden_dim],
            activation="snake",
            rngs=rngs,
        )

        # Field MLP: [encoded_pos | scene_latent] → [density(1), rgb(3)]
        self.field_mlp = StandardMLP(
            [pos_encoded_dim + config.hidden_dim, config.hidden_dim, config.hidden_dim, 4],
            activation="snake",
            rngs=rngs,
        )

    def _generate_rays(self, camera_pose: CameraPose) -> tuple[jax.Array, jax.Array]:
        """Generate ray origins and directions for all pixels.

        Args:
            camera_pose: Camera pose in world space.

        Returns:
            Tuple of ``(origins, directions)``, each shape ``(H*W, 3)``.
        """
        h, w = self.config.height, self.config.width
        fov_rad = NERF_CAMERA_HORIZONTAL_FOV_RAD

        i = jnp.linspace(-1.0, 1.0, w)
        j = jnp.linspace(1.0, -1.0, h)
        ii, jj = jnp.meshgrid(i, j)

        # Image-plane half-extent is tan(fov/2): ii/jj are already
        # normalized to [-1, 1], so the FOV must not depend on resolution.
        # Vertical extent scales by h/w (square pixels).
        half_extent = jnp.tan(fov_rad / 2.0)
        dirs_cam = jnp.stack(
            [ii * half_extent, jj * half_extent * (h / w), -jnp.ones_like(ii)], axis=-1
        ).reshape(-1, 3)

        dirs_world = dirs_cam @ camera_pose.rotation.T
        dirs_world = dirs_world / (jnp.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8)

        origins = jnp.broadcast_to(camera_pose.position, (h * w, 3))
        return origins, dirs_world

    def render_scene(
        self,
        scene_context: jax.Array,
        camera_pose: CameraPose,
    ) -> RenderedImage:
        """Render a scene from a given camera pose.

        Args:
            scene_context: Scene embedding of shape ``(ctx_len, context_dim)``.
            camera_pose: Camera position and orientation.

        Returns:
            ``RenderedImage`` with ``pixels`` ``(H, W, 3)`` and ``depth`` ``(H, W)``.
        """
        h, w = self.config.height, self.config.width
        cfg = self.config

        # Mean-pool scene context tokens and project to a hidden_dim latent
        scene_latent = self.scene_mlp(jnp.mean(scene_context, axis=0)[None])[0]  # (hidden_dim,)

        origins, directions = self._generate_rays(camera_pose)  # (H*W, 3)

        t_vals = jnp.linspace(cfg.near_plane, cfg.far_plane, cfg.num_samples)

        n_rays = h * w
        pts = origins[:, None, :] + directions[:, None, :] * t_vals[None, :, None]
        pts_flat = pts.reshape(n_rays * cfg.num_samples, 3)

        # Encode camera-relative coordinates normalized by the far plane:
        # the sinusoidal bands scale inputs by 2^k * pi, so raw world
        # coordinates (up to +/-far_plane) alias severely.
        pts_normalized = (pts_flat - camera_pose.position[None, :]) / cfg.far_plane

        scene_lat_broadcast = jnp.broadcast_to(
            scene_latent, (n_rays * cfg.num_samples, scene_latent.shape[-1])
        )

        encoded = self.pos_encoder(pts_normalized)  # (N*S, enc_dim)
        field_out = self.field_mlp(
            jnp.concatenate([encoded, scene_lat_broadcast], axis=-1)
        )  # (N*S, 4)
        density_flat = jax.nn.softplus(field_out[:, 0])  # (N*S,)
        rgb_flat = jax.nn.sigmoid(field_out[:, 1:])  # (N*S, 3)

        density = density_flat.reshape(n_rays, cfg.num_samples)
        rgb = rgb_flat.reshape(n_rays, cfg.num_samples, 3)

        deltas = jnp.concatenate([jnp.diff(t_vals), jnp.array([1e10])], axis=0)
        alpha = 1.0 - jnp.exp(-density * deltas[None, :])
        transmittance = jnp.cumprod(
            jnp.concatenate([jnp.ones((n_rays, 1)), 1.0 - alpha[:, :-1] + 1e-10], axis=1),
            axis=1,
        )
        weights = alpha * transmittance

        pixels_flat = jnp.sum(weights[:, :, None] * rgb, axis=1)
        depth_flat = jnp.sum(weights * t_vals[None, :], axis=1)

        return RenderedImage(
            pixels=pixels_flat.reshape(h, w, 3),
            depth=depth_flat.reshape(h, w),
        )
