"""Sensor realism: NeRF rendering, weather simulation, LiDAR ray casting."""

from simulacrax.sensor.lidar import LiDARConfig, LiDARRayCaster, PointCloud
from simulacrax.sensor.nerf_renderer import (
    CameraPose,
    NeRFRenderer,
    NeRFRendererConfig,
    RenderedImage,
)
from simulacrax.sensor.weather import (
    FogAugmentation,
    FogConfig,
    GlareAugmentation,
    GlareConfig,
    RainAugmentation,
    RainConfig,
)


__all__ = [
    "CameraPose",
    "FogAugmentation",
    "FogConfig",
    "GlareAugmentation",
    "GlareConfig",
    "LiDARConfig",
    "LiDARRayCaster",
    "NeRFRenderer",
    "NeRFRendererConfig",
    "PointCloud",
    "RainAugmentation",
    "RainConfig",
    "RenderedImage",
]
