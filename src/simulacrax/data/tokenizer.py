"""Differentiable scene tokenization pipeline.

Composes the modality encoders from :mod:`simulacrax.data.encoders` and
the preprocessing operators from :mod:`simulacrax.data.operators` into a
sequential datarax DAG. Modalities are capability-negotiated: each one is
``auto`` (enabled when the data source provides its required keys),
``required`` (construction fails when they are missing), or ``off``
(always excluded), validated against the source's element spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from datarax.core.operator import OperatorModule
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from flax import nnx

from simulacrax.core.constants import (
    CAMERA_IMAGES,
    LIDAR_POINTS,
    MODALITY_EMBEDDING_KEYS,
    MODALITY_TOKEN_VALID_KEYS,
    ROADGRAPH_KEYS,
    SCENE_EMBEDDING,
    STATE_BACKBONE_KEYS,
    STATE_VALID,
    STATE_X,
    STATE_Y,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.types import FusionStrategy, Modality, ModalityMode
from simulacrax.data.encoders import (
    AgentEncoder,
    AgentEncoderConfig,
    CameraEncoder,
    CameraEncoderConfig,
    EgoEncoder,
    EgoEncoderConfig,
    LiDAREncoder,
    LiDAREncoderConfig,
    MapEncoder,
    MapEncoderConfig,
    SceneFusionConfig,
    SceneFusionOperator,
)
from simulacrax.data.operators import (
    AgentNormalizationConfig,
    AgentNormalizationOperator,
    MapCroppingConfig,
    MapCroppingOperator,
    TemporalStackingConfig,
    TemporalStackingOperator,
)


def required_keys_for(
    modality: Modality,
    feature_fields: tuple[str, ...],
) -> frozenset[str]:
    """Return the raw scenario keys a modality's pipeline leg reads.

    Covers both the preprocessing operators serving the modality and its
    encoder, so negotiation reflects what the composed pipeline actually
    touches.

    Args:
        modality: Modality to look up.
        feature_fields: Temporal fields stacked for agent/ego encoding.

    Returns:
        Frozen set of required data-dict keys.
    """
    match modality:
        case Modality.AGENT:
            return STATE_BACKBONE_KEYS | set(feature_fields) | {STATE_VALID}
        case Modality.EGO:
            return STATE_BACKBONE_KEYS | set(feature_fields)
        case Modality.MAP:
            return STATE_BACKBONE_KEYS | ROADGRAPH_KEYS
        case Modality.LIDAR:
            return frozenset({LIDAR_POINTS})
        case Modality.CAMERA:
            return frozenset({CAMERA_IMAGES})


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizerConfig:
    """Configuration for the complete scene tokenization pipeline.

    Attributes:
        embed_dim: Shared embedding dimension for all encoders.
        num_heads: Attention heads for encoders and fusion.
        num_egnn_layers: EGNN layers in MapEncoder.
        fusion_strategy: Fusion method ("cross_attention", "early", "additive").
        history_steps: History timesteps for temporal stacking.
        crop_radius: Map cropping radius in meters.
        dropout_rate: Global encoder dropout rate (default 0.0). Active only in
            training mode (``model.train()``) and only when the constructing
            ``rngs`` carries a named ``dropout`` stream — the artifex attention
            blocks create their dropout module solely under that condition.
        num_lidar_points: LiDAR subsampling target count.
        max_polylines: Fixed polyline count for MapEncoder.
        feature_fields: Temporal fields to stack for agent encoding.
        agent_modality: Negotiation mode for the agent encoder.
        map_modality: Negotiation mode for the map encoder.
        ego_modality: Negotiation mode for the ego encoder.
        lidar_modality: Negotiation mode for the LiDAR encoder.
        camera_modality: Negotiation mode for the camera encoder.
    """

    embed_dim: int = 256
    num_heads: int = 8
    num_egnn_layers: int = 3
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    history_steps: int = WOD_HISTORY_STEPS
    crop_radius: float = 150.0
    dropout_rate: float = 0.0
    num_lidar_points: int = 1024
    max_polylines: int = 256
    feature_fields: tuple[str, ...] = (STATE_X, STATE_Y)
    agent_modality: ModalityMode = ModalityMode.AUTO
    map_modality: ModalityMode = ModalityMode.AUTO
    ego_modality: ModalityMode = ModalityMode.AUTO
    lidar_modality: ModalityMode = ModalityMode.AUTO
    camera_modality: ModalityMode = ModalityMode.AUTO

    def __post_init__(self) -> None:
        """Validate enum-typed fields."""
        FusionStrategy(self.fusion_strategy)
        for mode in self.modality_modes().values():
            ModalityMode(mode)

    def modality_modes(self) -> dict[Modality, ModalityMode]:
        """Map each modality to its configured negotiation mode.

        Returns:
            Dict in fusion token order (agent, map, ego, lidar, camera).
        """
        return {
            Modality.AGENT: self.agent_modality,
            Modality.MAP: self.map_modality,
            Modality.EGO: self.ego_modality,
            Modality.LIDAR: self.lidar_modality,
            Modality.CAMERA: self.camera_modality,
        }


def resolve_active_modalities(
    config: TokenizerConfig,
    element_spec: Mapping[str, object] | None,
) -> tuple[Modality, ...]:
    """Negotiate the active modality set against a source's element spec.

    ``off`` modalities are always excluded. With a spec, ``auto`` enables a
    modality only when every key it reads is present, while ``required``
    raises when any is missing. Without a spec there is nothing to
    negotiate against, so ``auto`` and ``required`` are both enabled.

    Args:
        config: Tokenizer configuration carrying per-modality modes.
        element_spec: Per-key contract from the data source (only the keys
            are consulted), e.g. ``WODSource.element_spec()``; None when
            the source cannot provide one.

    Returns:
        Active modalities in fusion token order.

    Raises:
        ValueError: When a required modality's keys are missing from the
            spec, or when negotiation leaves no modality active.
    """
    active: list[Modality] = []
    for modality, configured_mode in config.modality_modes().items():
        mode = ModalityMode(configured_mode)
        if mode is ModalityMode.OFF:
            continue
        if element_spec is None:
            active.append(modality)
            continue
        missing = sorted(required_keys_for(modality, config.feature_fields) - element_spec.keys())
        if not missing:
            active.append(modality)
        elif mode is ModalityMode.REQUIRED:
            raise ValueError(
                f"Modality '{modality}' is required but the element spec is missing keys: {missing}"
            )
    if not active:
        raise ValueError(
            "No tokenizer modality is active after negotiation; enable at "
            "least one modality or provide a source that supplies its keys."
        )
    return tuple(active)


def _build_preprocessing_operators(
    config: TokenizerConfig,
    active: tuple[Modality, ...],
    rngs: nnx.Rngs,
) -> list[OperatorModule]:
    """Build the preprocessing operators the active modalities need.

    Ego-frame normalization serves every state-derived modality, map
    cropping only the map leg, and temporal stacking the agent/ego legs.

    Args:
        config: Pipeline configuration.
        active: Negotiated modality set.
        rngs: Flax NNX RNGs.

    Returns:
        Preprocessing operators in pipeline order.
    """
    current_step = config.history_steps - 1
    operators: list[OperatorModule] = []
    if {Modality.AGENT, Modality.MAP, Modality.EGO} & set(active):
        operators.append(
            AgentNormalizationOperator(
                AgentNormalizationConfig(current_step_idx=current_step),
                rngs=rngs,
            )
        )
    if Modality.MAP in active:
        operators.append(
            MapCroppingOperator(
                MapCroppingConfig(
                    crop_radius=config.crop_radius,
                    current_step_idx=current_step,
                ),
                rngs=rngs,
            )
        )
    if {Modality.AGENT, Modality.EGO} & set(active):
        operators.append(
            TemporalStackingOperator(
                TemporalStackingConfig(
                    history_steps=config.history_steps,
                    feature_fields=list(config.feature_fields),
                ),
                rngs=rngs,
            )
        )
    return operators


def _build_encoder(
    modality: Modality,
    config: TokenizerConfig,
    rngs: nnx.Rngs,
) -> OperatorModule:
    """Build the encoder operator for one active modality.

    Args:
        modality: Modality to build.
        config: Pipeline configuration.
        rngs: Flax NNX RNGs.

    Returns:
        The configured encoder operator.
    """
    mlp_hidden = config.history_steps * len(config.feature_fields)
    match modality:
        case Modality.AGENT:
            return AgentEncoder(
                AgentEncoderConfig(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_hidden=mlp_hidden,
                    dropout_rate=config.dropout_rate,
                ),
                rngs=rngs,
            )
        case Modality.MAP:
            return MapEncoder(
                MapEncoderConfig(
                    embed_dim=config.embed_dim,
                    num_egnn_layers=config.num_egnn_layers,
                    max_polylines=config.max_polylines,
                    dropout_rate=config.dropout_rate,
                ),
                rngs=rngs,
            )
        case Modality.EGO:
            return EgoEncoder(
                EgoEncoderConfig(
                    embed_dim=config.embed_dim,
                    mlp_hidden=mlp_hidden,
                ),
                rngs=rngs,
            )
        case Modality.LIDAR:
            return LiDAREncoder(
                LiDAREncoderConfig(
                    embed_dim=config.embed_dim,
                    num_points=config.num_lidar_points,
                    dropout_rate=config.dropout_rate,
                ),
                rngs=rngs,
            )
        case Modality.CAMERA:
            return CameraEncoder(
                CameraEncoderConfig(
                    embed_dim=config.embed_dim,
                ),
                rngs=rngs,
            )


class SceneTokenizer(CompositeOperatorModule):
    """Differentiable scene tokenization pipeline via datarax DAG.

    Composes preprocessing operators and the negotiated modality encoders
    into a sequential pipeline using ``CompositeOperatorModule(SEQUENTIAL)``.
    All operators use JAX operations for vmap/JIT compatibility inside the
    datarax DAG.

    Pipeline (modality legs appear only when active):
        1. AgentNormalizationOperator — ego-centric coordinate normalization
        2. MapCroppingOperator — radius-based roadgraph cropping (map)
        3. TemporalStackingOperator — history stacking (agent/ego)
        4. AgentEncoder — transformer-based agent state encoding
        5. MapEncoder — VectorNet-style hierarchical map encoding
        6. EgoEncoder — MLP-based ego state encoding
        7. LiDAREncoder — transformer-based point cloud encoding
        8. CameraEncoder — CNN-based multi-view image encoding
        9. SceneFusionOperator — cross-modal fusion of the active set

    Attributes:
        active_modalities: Negotiated modality set in fusion token order.
        tokenizer_config: The configuration the pipeline was built from.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        *,
        element_spec: Mapping[str, object] | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize scene tokenizer with the negotiated sub-operators.

        Args:
            config: Pipeline configuration.
            element_spec: Data source contract to negotiate modalities
                against (e.g. ``WODSource.element_spec()``). None skips
                negotiation, enabling every non-``off`` modality.
            rngs: Flax NNX RNGs for parameter initialization.

        Raises:
            ValueError: When negotiation fails (see
                :func:`resolve_active_modalities`).
        """
        if rngs is None:
            # Include a ``dropout`` stream so encoder dropout can activate when
            # ``dropout_rate > 0`` (artifex attention blocks require it); a no-op
            # at the default rate of 0.0.
            rngs = nnx.Rngs(0, dropout=1)

        active = resolve_active_modalities(config, element_spec)

        operators: list[OperatorModule] = _build_preprocessing_operators(config, active, rngs)
        operators.extend(_build_encoder(modality, config, rngs) for modality in active)
        operators.append(
            SceneFusionOperator(
                SceneFusionConfig(
                    embed_dim=config.embed_dim,
                    fusion_strategy=config.fusion_strategy,
                    num_heads=config.num_heads,
                    dropout_rate=config.dropout_rate,
                    input_fields=[MODALITY_EMBEDDING_KEYS[m] for m in active],
                    output_fields=[SCENE_EMBEDDING],
                    validity_fields=[MODALITY_TOKEN_VALID_KEYS[m] for m in active],
                ),
                rngs=rngs,
            )
        )

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=operators,
        )
        super().__init__(composite_config, rngs=rngs)
        self.tokenizer_config = config
        self.active_modalities = active
