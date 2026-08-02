"""Modality-specific scene encoders and cross-modal fusion.

Each module provides one encoder operator and its configuration; the
``SceneTokenizer`` in :mod:`simulacrax.data.tokenizer` composes them
into the full pipeline.
"""

from simulacrax.data.encoders.agent import AgentEncoder, AgentEncoderConfig
from simulacrax.data.encoders.camera import CameraEncoder, CameraEncoderConfig
from simulacrax.data.encoders.ego import EgoEncoder, EgoEncoderConfig
from simulacrax.data.encoders.fusion import SceneFusionConfig, SceneFusionOperator
from simulacrax.data.encoders.lidar import LiDAREncoder, LiDAREncoderConfig
from simulacrax.data.encoders.map import MapEncoder, MapEncoderConfig


__all__ = [
    "AgentEncoder",
    "AgentEncoderConfig",
    "CameraEncoder",
    "CameraEncoderConfig",
    "EgoEncoder",
    "EgoEncoderConfig",
    "LiDAREncoder",
    "LiDAREncoderConfig",
    "MapEncoder",
    "MapEncoderConfig",
    "SceneFusionConfig",
    "SceneFusionOperator",
]
