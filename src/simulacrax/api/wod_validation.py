"""Stream held-out WOD validation scenes for map-conditioned evaluation.

Shared by the training script's periodic validation and the offline
adversarial-steering benchmark: both need real, map-carrying
:class:`~simulacrax.evaluation.map_conditioned_evaluator.ValidationScene`
objects — logged futures, tokenizer keys, per-scene road edges, and the
``tracks_to_predict`` benchmark flags — extracted from the streamed TFRecords.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from simulacrax.core.constants import (
    ROADGRAPH_KEYS,
    STATE_BACKBONE_KEYS,
    STATE_TRACKS_TO_PREDICT,
    STATE_VALID,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.types import DatasetMode
from simulacrax.data import (
    fixed_shape_road_edges_from_wod_dict,
    prepare_padded_scene,
    resolve_wod_tfrecord_path,
)
from simulacrax.data.operators import AgentNormalizationOperator
from simulacrax.data.wod_source import WODSource, WODSourceConfig
from simulacrax.evaluation.map_conditioned_evaluator import ValidationScene


# Numeric raw keys the SceneTokenizer reads (state + roadgraph); stacked into
# batched tensors and tokenized inside the differentiated step.
TOKENIZER_KEYS = tuple(sorted(STATE_BACKBONE_KEYS | {STATE_VALID} | ROADGRAPH_KEYS))

PreparedScene = tuple[dict[str, np.ndarray], jax.Array, jax.Array, jax.Array, jax.Array]


def prepare_scene(normalized: dict, num_agents: int, future_steps: int) -> PreparedScene | None:
    """Extract ``(scene, trajectories, validity, reference_pose, agent_rows)``.

    The scene dict carries only the numeric keys the tokenizer reads; the
    trajectories/validity/reference_pose/agent_rows come from the padded,
    validity-masked extraction over the top ``num_agents`` agents. ``agent_rows``
    maps each trajectory slot to its source agent so the model can gather the
    matching per-agent embedding as the conditioning anchor, and
    ``reference_pose`` is each agent's current-step pose defining its local frame.

    Args:
        normalized: Ego-normalized WOD scenario dict.
        num_agents: Number of agent slots to extract (padded/masked).
        future_steps: Future horizon to extract.

    Returns:
        The prepared tuple, or ``None`` when no usable extraction exists.
    """
    prepared = prepare_padded_scene(
        normalized,
        num_agents=num_agents,
        history_steps=WOD_HISTORY_STEPS,
        future_steps=future_steps,
    )
    if prepared is None:
        return None
    trajectories, validity, reference_pose, agent_rows = prepared
    scene = {key: np.asarray(normalized[key]) for key in TOKENIZER_KEYS}
    return scene, trajectories, validity, reference_pose, agent_rows


def stream_validation_scenes(
    norm: AgentNormalizationOperator,
    *,
    num_scenes: int,
    max_agents: int,
    future_steps: int,
    split: str = "val",
) -> list[ValidationScene]:
    """Stream held-out validation scenes with fixed-shape road edges.

    Each scene carries the logged futures, the tokenizer keys, and fixed-shape
    per-scene road edges the off-road metric scores against. The uniform road-edge
    shape lets the evaluator stack all scenes and sample every rollout in one
    batched device pass; scenes without a usable extraction are skipped, and an
    edge-less scene keeps an all-invalid road-edge tensor rather than being
    dropped.

    Args:
        norm: Ego-normalization operator applied to each raw scene.
        num_scenes: Number of scenes to collect (non-positive returns empty).
        max_agents: Maximum agents per scene.
        future_steps: Future horizon to extract.
        split: Dataset split to stream (``"val"`` by default).

    Returns:
        The collected validation scenes.
    """
    if num_scenes <= 0:
        return []
    source = WODSource(
        WODSourceConfig(
            wod_path=resolve_wod_tfrecord_path(),
            split=split,
            mode=DatasetMode.STREAMING,
            max_agents=max_agents,
            history_steps=WOD_HISTORY_STEPS,
            future_steps=future_steps,
        )
    )
    scenes: list[ValidationScene] = []
    for element in source:
        if len(scenes) >= num_scenes:
            break
        raw = dict(element.data)
        tracks_flag = np.asarray(raw[STATE_TRACKS_TO_PREDICT])  # (max_objects,)
        normalized, _, _ = norm.apply(raw, {}, {})
        prepared = prepare_scene(normalized, max_agents, future_steps)
        if prepared is None:
            continue
        road_edges = fixed_shape_road_edges_from_wod_dict(normalized)
        scene, traj, valid, reference_pose, agent_rows = prepared
        # The benchmark scores its ``tracks_to_predict`` agents; map that per-object
        # flag onto the selected slots (padded slots repeat index 0 but are masked
        # out of minADE by their all-invalid validity regardless).
        tracks_to_predict = tracks_flag[np.asarray(agent_rows)] > 0  # (num_agents,)
        scenes.append(
            ValidationScene(
                scene={key: jnp.asarray(scene[key]) for key in TOKENIZER_KEYS},
                trajectories=jnp.asarray(traj),
                agent_rows=jnp.asarray(agent_rows),
                valid=jnp.asarray(valid),
                tracks_to_predict=jnp.asarray(tracks_to_predict),
                reference_pose=jnp.asarray(reference_pose),
                road_edges=road_edges,
            )
        )
    return scenes
