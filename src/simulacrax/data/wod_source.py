"""Waymo Open Dataset data source extending datarax DataSourceModule.

Provides WODSourceConfig and WODSource for loading WOD Motion TFRecord
scenarios. Follows datarax's TFDSEagerSource/TFDSStreamingSource patterns
with two loading modes:

**Eager mode** (default): Loads all scenarios to memory at init. Pure Python
iteration after init with no TensorFlow overhead. Ideal for validation splits
and small training subsets.

**Streaming mode**: Lazily iterates TFRecords on-the-fly. Handles datasets
too large for memory. Uses fixed prefetch to avoid TF thread storms.

TFRecord parsing follows Waymax's feature schema and temporal aggregation:
raw TFRecords store state/past/, state/current/, state/future/ separately;
this module concatenates them into state/all/* arrays.

TensorFlow is confined to CPU-only proto parsing; all downstream processing
uses JAX arrays.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.spec import array_to_spec
from datarax.sources._source_base import resolve_wrapped_indices
from datarax.typing import Element, Metadata
from flax import nnx

from simulacrax.core.config import VALID_SPLITS, validate_positive
from simulacrax.core.constants import (
    ROADGRAPH_DIR,
    ROADGRAPH_ID,
    ROADGRAPH_TYPE,
    ROADGRAPH_VALID,
    ROADGRAPH_XYZ,
    SCENARIO_ID,
    STATE_BBOX_YAW,
    STATE_ID,
    STATE_IS_SDC,
    STATE_OBJECTS_OF_INTEREST,
    STATE_TRACKS_TO_PREDICT,
    STATE_TYPE,
    WOD_CURRENT_TIME_INDEX,
    WOD_FUTURE_STEPS,
    WOD_HISTORY_STEPS,
)
from simulacrax.core.types import DatasetMode


logger = logging.getLogger(__name__)

# WOD temporal structure, derived from the canonical scenario constants
_PAST_STEPS = WOD_CURRENT_TIME_INDEX
_CURRENT_STEPS = 1
_FUTURE_STEPS = WOD_FUTURE_STEPS

# WOD defaults
_MAX_NUM_OBJECTS = 128
_MAX_NUM_RG_POINTS = 30000

# Split name mapping (config uses short names, WOD uses full names)
_SPLIT_DIR_MAP = {
    "train": "training",
    "val": "validation",
    "test": "testing",
}


def resolve_wod_tfrecord_path(env_var: str = "WOD_MOTION_TFRECORD_PATH") -> str:
    """Read the WOD Motion TFRecord directory from the environment.

    Args:
        env_var: Environment variable holding the TFRecord directory.

    Returns:
        The configured path.

    Raises:
        RuntimeError: With a setup hint when the variable is unset or empty.
    """
    path = os.environ.get(env_var, "")
    if not path:
        msg = (
            f"{env_var} is not set. Point it at your WOD Motion TFRecord "
            f"directory, e.g. export {env_var}=/data/waymo/motion_v1.2.1/"
            "tf_example, or add it to .env.data (see the installation guide)."
        )
        raise RuntimeError(msg)
    return path


@dataclass(frozen=True, slots=True, kw_only=True)
class WODSourceConfig(StructuralConfig):
    """Configuration for Waymo Open Dataset loading.

    Attributes:
        wod_path: Path to WOD TFRecord data directory.
        split: Dataset split ("train", "val", or "test").
        mode: Loading mode ("eager" or "streaming").
        max_agents: Maximum agents per scenario.
        history_steps: Past context timesteps (default 11 = 1.1s at 10Hz).
        future_steps: Future prediction timesteps (default 80 = 8.0s at 10Hz).
        max_num_rg_points: Max roadgraph sample points (default 30000 for v1.2+).
        prefetch_buffer: Fixed prefetch buffer for streaming mode (default 2).
    """

    wod_path: str = ""
    split: str = "train"
    mode: DatasetMode = DatasetMode.EAGER
    max_agents: int = 32
    history_steps: int = WOD_HISTORY_STEPS
    future_steps: int = WOD_FUTURE_STEPS
    max_num_rg_points: int = _MAX_NUM_RG_POINTS
    prefetch_buffer: int = 2

    def __post_init__(self) -> None:
        """Validate field constraints."""
        # Explicit super(): @dataclass(slots=True) rebuilds the class, breaking
        # the zero-arg form's __class__ cell.
        super(WODSourceConfig, self).__post_init__()
        if not self.wod_path:
            msg = "wod_path is required"
            raise ValueError(msg)
        if self.split not in VALID_SPLITS:
            msg = f"split must be one of {sorted(VALID_SPLITS)}, got {self.split!r}"
            raise ValueError(msg)
        DatasetMode(self.mode)
        validate_positive("max_agents", self.max_agents)
        validate_positive("history_steps", self.history_steps)
        validate_positive("future_steps", self.future_steps)

    @property
    def total_steps(self) -> int:
        """Total timesteps (history + future)."""
        return self.history_steps + self.future_steps


class WODSource(DataSourceModule):
    """Data source for Waymo Open Dataset scenarios.

    Extends datarax's DataSourceModule following TFDSEagerSource/TFDSStreamingSource
    patterns. Loads WOD Motion TFRecords, parses the feature schema, aggregates
    temporal slices (past/current/future to state/all/*), and yields datarax
    Element objects.

    **Eager mode** (default):
        Loads all scenarios at init. Pure Python iteration after loading.
        TensorFlow resources are released after init. Best for validation
        sets and development.

    **Streaming mode**:
        Lazy TFRecord iteration with fixed prefetch. Handles large splits
        without loading everything to memory. Some TF thread overhead.

    For testing, accepts pre-loaded dicts directly via ``raw_scenarios``,
    bypassing TFRecord loading entirely.

    Example::

        # Eager loading (validation)
        config = WODSourceConfig(
            wod_path="/data/waymo/motion_v1.2.1/tf_example",
            split="val",
            mode="eager",
            max_agents=32,
        )
        source = WODSource(config)

        for element in source:
            scene = parse_scenario(element.data)

        # Streaming (large training set)
        config = WODSourceConfig(
            wod_path="/data/waymo",
            split="train",
            mode="streaming",
        )
        source = WODSource(config)
        for element in source:
            process(element.data)

    Attributes:
        config: Source configuration.
    """

    _scenarios: list = nnx.data()
    _scenario_ids: list = nnx.data()
    _indexed_cache: dict | None = nnx.data()

    config: WODSourceConfig

    def __init__(
        self,
        config: WODSourceConfig,
        *,
        raw_scenarios: list[dict] | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize WOD source.

        Args:
            config: WOD source configuration.
            raw_scenarios: Pre-loaded scenario dicts (for testing).
                If None, loads from config.wod_path using TensorFlow.
            rngs: Optional Flax NNX RNGs.
            name: Optional module name.
        """
        if name is None:
            name = f"WODSource({config.split}:{config.mode})"
        super().__init__(config, rngs=rngs, name=name)

        # Streaming state (lazily initialized)
        self._tf_dataset: Any = None
        self._tf_iterator: Iterator[Any] | None = None
        self._features_desc: dict[str, Any] | None = None

        if raw_scenarios is not None:
            # Testing path: pre-loaded data, always eager
            self._scenarios = list(raw_scenarios)
            self._scenario_ids = [s.get(SCENARIO_ID, [b"unknown"])[0] for s in self._scenarios]
        elif config.mode == DatasetMode.EAGER:  # pragma: no cover
            self._scenarios = self._load_eager()
            self._scenario_ids = [s.get(SCENARIO_ID, [b"unknown"])[0] for s in self._scenarios]
            self._cleanup_tf()
        else:  # pragma: no cover
            # Streaming mode: build TF dataset, don't load all at init
            self._scenarios = []
            self._scenario_ids = []
            self._tf_dataset = self._build_streaming_dataset()

        # Key -> stacked-array cache backing stateless get_batch_at.
        # Built at init (never inside a trace — nnx forbids mutation there).
        self._indexed_cache = None
        if self._scenarios:
            self._ensure_indexed_cache()

        # Iteration state
        self._current_idx = 0
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

    # =========================================================================
    # TFRecord Feature Schema
    # =========================================================================

    @staticmethod
    def _get_features_description(  # pragma: no cover
        max_num_objects: int = _MAX_NUM_OBJECTS,
        max_num_rg_points: int = _MAX_NUM_RG_POINTS,
    ) -> dict[str, Any]:
        """Build the TF feature description for WOD Motion TFRecords.

        Follows Waymax's get_features_description() schema. Covers all state
        fields needed by parsers.py plus roadgraph samples.

        Args:
            max_num_objects: Max number of agent objects per scenario.
            max_num_rg_points: Max number of sampled roadgraph points.

        Returns:
            Dict of feature name to tf.io.FixedLenFeature specifications.
        """
        import tensorflow as tf

        roadgraph_features = {
            ROADGRAPH_DIR: tf.io.FixedLenFeature(
                [max_num_rg_points, 3], tf.float32, default_value=None
            ),
            ROADGRAPH_ID: tf.io.FixedLenFeature(
                [max_num_rg_points, 1], tf.int64, default_value=None
            ),
            ROADGRAPH_TYPE: tf.io.FixedLenFeature(
                [max_num_rg_points, 1], tf.int64, default_value=None
            ),
            ROADGRAPH_VALID: tf.io.FixedLenFeature(
                [max_num_rg_points, 1], tf.int64, default_value=None
            ),
            ROADGRAPH_XYZ: tf.io.FixedLenFeature(
                [max_num_rg_points, 3], tf.float32, default_value=None
            ),
        }

        state_features = {
            STATE_ID: tf.io.FixedLenFeature([max_num_objects], tf.float32, default_value=None),
            STATE_TYPE: tf.io.FixedLenFeature([max_num_objects], tf.float32, default_value=None),
            STATE_IS_SDC: tf.io.FixedLenFeature([max_num_objects], tf.int64, default_value=None),
            STATE_TRACKS_TO_PREDICT: tf.io.FixedLenFeature(
                [max_num_objects], tf.int64, default_value=None
            ),
            STATE_OBJECTS_OF_INTEREST: tf.io.FixedLenFeature(
                [max_num_objects], tf.int64, default_value=None
            ),
        }

        num_timesteps = {
            "past": _PAST_STEPS,
            "current": _CURRENT_STEPS,
            "future": _FUTURE_STEPS,
        }
        temporal_fields_float = [
            "bbox_yaw",
            "height",
            "length",
            "speed",
            "vel_yaw",
            "velocity_x",
            "velocity_y",
            "width",
            "x",
            "y",
            "z",
        ]
        temporal_fields_int = ["timestamp_micros", "valid"]

        for time_key, steps in num_timesteps.items():
            for field in temporal_fields_float:
                state_features[f"state/{time_key}/{field}"] = tf.io.FixedLenFeature(
                    [max_num_objects, steps], tf.float32, default_value=None
                )
            for field in temporal_fields_int:
                state_features[f"state/{time_key}/{field}"] = tf.io.FixedLenFeature(
                    [max_num_objects, steps], tf.int64, default_value=None
                )

        features = {}
        features.update(roadgraph_features)
        features.update(state_features)
        return features

    def _features_description(self) -> dict[str, Any]:
        """Return the TF feature description, building it exactly once.

        The ~49-feature schema is identical for every record of a source,
        so it is cached on first use instead of being rebuilt per record.

        Returns:
            Dict of feature name to tf.io.FixedLenFeature specifications.
        """
        if self._features_desc is None:
            self._features_desc = self._get_features_description(
                max_num_rg_points=self.config.max_num_rg_points,
            )
        return self._features_desc

    # =========================================================================
    # Temporal Aggregation
    # =========================================================================

    @staticmethod
    def _aggregate_time_tensors(decoded: dict) -> dict:  # pragma: no cover
        """Concatenate past/current/future state tensors into state/all/*.

        Follows Waymax's aggregate_time_tensors() pattern. Also wraps
        bbox_yaw to [-pi, pi] range.

        Args:
            decoded: Dict of parsed TF tensors with state/past/*, state/current/*,
                state/future/* keys.

        Returns:
            New dict with state/all/* replacing temporal splits, plus all
            non-temporal keys preserved.
        """
        import tensorflow as tf

        state_time_features = set()
        for key in decoded:
            if key.startswith("state/current/"):
                state_time_features.add(key[len("state/current/") :])

        removed_keys: set[str] = set()
        result: dict = {}

        for feature in state_time_features:
            result[f"state/all/{feature}"] = tf.concat(
                [
                    decoded[f"state/past/{feature}"],
                    decoded[f"state/current/{feature}"],
                    decoded[f"state/future/{feature}"],
                ],
                axis=-1,
            )
            removed_keys.add(f"state/past/{feature}")
            removed_keys.add(f"state/current/{feature}")
            removed_keys.add(f"state/future/{feature}")

        # Wrap bbox_yaw to [-pi, pi] (Waymax convention)
        if STATE_BBOX_YAW in result:
            yaw = result[STATE_BBOX_YAW]
            pi_val = tf.constant(np.pi, dtype=yaw.dtype)
            two_pi = tf.constant(2.0 * np.pi, dtype=yaw.dtype)
            result[STATE_BBOX_YAW] = (yaw + pi_val) % two_pi - pi_val

        for key in decoded:
            if key not in removed_keys:
                result[key] = decoded[key]

        return result

    # =========================================================================
    # TFRecord File Discovery
    # =========================================================================

    @staticmethod
    def _find_tfrecord_files(wod_path: str, split: str) -> list[Path]:  # pragma: no cover
        """Find TFRecord files for the given split.

        Args:
            wod_path: Base path to WOD TFRecord data.
            split: Dataset split ("train", "val", or "test").

        Returns:
            Sorted list of TFRecord file paths.

        Raises:
            FileNotFoundError: If no TFRecord files are found.
        """
        base = Path(wod_path)
        split_dir_name = _SPLIT_DIR_MAP.get(split, split)

        candidates = [
            base / split_dir_name,
            base / f"tf_example/{split_dir_name}",
            base,
        ]

        for search_dir in candidates:
            if not search_dir.is_dir():
                continue
            files = sorted(search_dir.glob("*.tfrecord*"))
            if files:
                logger.info("Found %d TFRecord file(s) in %s", len(files), search_dir)
                return files

        msg = (
            f"No TFRecord files found for split={split!r} under {wod_path!r}. "
            f"Searched: {[str(c) for c in candidates]}"
        )
        raise FileNotFoundError(msg)

    # =========================================================================
    # Parse Single Record
    # =========================================================================

    def _parse_single_record(
        self,
        raw_record: Any,
        features_desc: dict,
    ) -> dict:  # pragma: no cover
        """Parse a single TFRecord into an aggregated scenario dict.

        Args:
            raw_record: Serialized TFRecord bytes.
            features_desc: TF feature description dict.

        Returns:
            Scenario dict with state/all/* keys and numpy arrays.
        """
        import tensorflow as tf

        # Parse scenario ID (string feature, separate from numeric features)
        sid_feature = {SCENARIO_ID: tf.io.FixedLenFeature([], tf.string, default_value="")}
        sid_parsed = tf.io.parse_single_example(raw_record, sid_feature)
        scenario_id = sid_parsed[SCENARIO_ID].numpy()

        # Parse all numeric features
        parsed = tf.io.parse_single_example(raw_record, features_desc)

        # Aggregate past/current/future → state/all/*
        aggregated = self._aggregate_time_tensors(parsed)

        # Limit agents if needed
        max_agents = self.config.max_agents
        scenario_dict: dict[str, np.ndarray] = {}
        for key, tensor in aggregated.items():
            arr = tensor.numpy()
            if key.startswith("state/") and max_agents < _MAX_NUM_OBJECTS:
                scenario_dict[key] = arr[:max_agents]
            else:
                scenario_dict[key] = arr

        # Add scenario ID as 1-element bytes array (matches mock data format)
        scenario_dict[SCENARIO_ID] = np.array([scenario_id])

        return scenario_dict

    # =========================================================================
    # Eager Loading (load all at init)
    # =========================================================================

    def _load_eager(self) -> list[dict]:  # pragma: no cover
        """Load all scenarios from TFRecords into memory.

        Follows TFDSEagerSource._load_all_to_jax() pattern. TF resources
        are released after this method via _cleanup_tf().

        Returns:
            List of scenario dicts with numpy arrays.
        """
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        tfrecord_files = self._find_tfrecord_files(self.config.wod_path, self.config.split)
        features_desc = self._features_description()

        dataset = tf.data.TFRecordDataset(
            [str(f) for f in tfrecord_files],
            compression_type="",
        )

        scenarios: list[dict] = []
        for raw_record in dataset:
            scenarios.append(self._parse_single_record(raw_record, features_desc))

        logger.info(
            "Eager-loaded %d scenarios (split=%s, max_agents=%d)",
            len(scenarios),
            self.config.split,
            self.config.max_agents,
        )
        return scenarios

    # =========================================================================
    # Streaming Mode (lazy iteration)
    # =========================================================================

    def _build_streaming_dataset(self) -> Any:  # pragma: no cover
        """Build a TF dataset for streaming iteration.

        Uses fixed prefetch buffer (NOT AUTOTUNE) to avoid TF thread storms,
        following TFDSStreamingSource pattern.

        Returns:
            tf.data.Dataset configured for streaming.
        """
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        tfrecord_files = self._find_tfrecord_files(self.config.wod_path, self.config.split)

        dataset = tf.data.TFRecordDataset(
            [str(f) for f in tfrecord_files],
            compression_type="",
        )

        # CRITICAL: Fixed prefetch, NOT AUTOTUNE (prevents thread storms)
        dataset = dataset.prefetch(self.config.prefetch_buffer)

        logger.info(
            "Built streaming dataset from %d file(s) (split=%s)",
            len(tfrecord_files),
            self.config.split,
        )
        return dataset

    def _streaming_next(self) -> dict:  # pragma: no cover
        """Get next scenario from streaming dataset.

        Returns:
            Parsed and aggregated scenario dict.

        Raises:
            StopIteration: When dataset is exhausted.
        """
        if self._tf_iterator is None:
            dataset = cast(Any, self._tf_dataset)
            self._tf_iterator = iter(dataset)

        raw_record = next(self._tf_iterator)
        return self._parse_single_record(raw_record, self._features_description())

    # =========================================================================
    # TF Cleanup
    # =========================================================================

    @staticmethod
    def _cleanup_tf() -> None:  # pragma: no cover
        """Release TensorFlow resources after eager loading.

        Follows TFDSEagerSource._cleanup_tf() pattern. Ensures no TF
        threads remain active after init.
        """
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            logger.debug("TensorFlow not available, skipping session cleanup")
        gc.collect()

    # =========================================================================
    # Element Conversion
    # =========================================================================

    def _scenario_to_element(self, idx: int, scenario: dict) -> Element:
        """Convert a raw WOD scenario dict to a datarax Element.

        Args:
            idx: Scenario index.
            scenario: Raw WOD scenario dict.

        Returns:
            Element with scenario data and metadata.
        """
        scenario_id = scenario.get(SCENARIO_ID, [b"unknown"])[0]
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode("utf-8")

        metadata = Metadata(
            index=idx,
            key=scenario_id,
            source_info={
                "scenario_id": scenario_id,
                "split": self.config.split,
            },
        )

        return Element(data=scenario, state={}, metadata=metadata)

    # =========================================================================
    # DataSourceModule Interface
    # =========================================================================

    def __len__(self) -> int:
        """Return the number of loaded scenarios.

        For streaming mode, returns 0 if no scenarios have been consumed yet.
        """
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> Element | None:
        """Get scenario Element at the given index.

        Only available in eager mode or after streaming has cached scenarios.

        Args:
            idx: Scenario index.

        Returns:
            Element wrapping the raw scenario dict, or None if out of bounds.
        """
        if idx < 0 or idx >= len(self._scenarios):
            return None
        return self._scenario_to_element(idx, self._scenarios[idx])

    def __iter__(self) -> Iterator[Element]:
        """Iterate over scenarios.

        In eager mode, iterates over loaded scenarios. In streaming mode,
        lazily parses TFRecords from the underlying tf.data.Dataset.
        """
        self._current_idx = 0
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)

        if (
            self.config.mode == DatasetMode.STREAMING and self._tf_dataset is not None
        ):  # pragma: no cover
            return self._stream_iter()
        return self._eager_iter()

    def _eager_iter(self) -> Iterator[Element]:
        """Iterate over eager-loaded scenarios."""
        for i in range(len(self._scenarios)):
            yield self._scenario_to_element(i, self._scenarios[i])

    def _stream_iter(self) -> Iterator[Element]:  # pragma: no cover
        """Iterate over streaming TFRecord dataset."""
        dataset = cast(Any, self._tf_dataset)
        self._tf_iterator = iter(dataset)
        features_desc = self._features_description()

        idx = 0
        for raw_record in self._tf_iterator:
            scenario = self._parse_single_record(raw_record, features_desc)
            yield self._scenario_to_element(idx, scenario)
            idx += 1

    def __next__(self) -> Element:
        """Get next scenario Element.

        Returns:
            Next Element in iteration.

        Raises:
            StopIteration: When all scenarios have been yielded.
        """
        if (
            self.config.mode == DatasetMode.STREAMING and self._tf_dataset is not None
        ):  # pragma: no cover
            scenario = self._streaming_next()
            elem = self._scenario_to_element(self._current_idx, scenario)
            self._current_idx += 1
            return elem

        if self._current_idx >= len(self._scenarios):
            raise StopIteration
        elem = self._scenario_to_element(self._current_idx, self._scenarios[self._current_idx])
        self._current_idx += 1
        return elem

    def reset(self, seed: int | None = None) -> None:
        """Reset iteration state.

        Args:
            seed: Unused, for API compatibility.
        """
        del seed
        self._current_idx = 0
        self.index.set_value(0)
        if self._tf_iterator is not None:
            self._tf_iterator = None

    def get_batch(
        self,
        batch_size: int,
        key: None = None,
    ) -> list[Element]:
        """Get a batch of scenario Elements.

        Only available in eager mode.

        Args:
            batch_size: Number of scenarios to return.
            key: Unused, for API compatibility.

        Returns:
            List of Element objects.
        """
        del key
        batch = []
        for _ in range(batch_size):
            if self._current_idx >= len(self._scenarios):
                break
            elem = self._scenario_to_element(self._current_idx, self._scenarios[self._current_idx])
            batch.append(elem)
            self._current_idx += 1
        return batch

    def supports_indexed_access(self) -> bool:
        """Whether random-access ``get_batch_at`` is available.

        Eager sources (including the ``raw_scenarios`` testing path, which
        is always eager) support indexed access; streaming sources are
        forward-only.

        Returns:
            True when scenario data is materialized in memory.
        """
        return len(self._scenarios) > 0 or self.config.mode == DatasetMode.EAGER

    def _numeric_scenario_keys(self) -> list[str]:
        """Return scenario dict keys holding numeric arrays (JAX-traceable)."""
        return [
            key
            for key, value in self._scenarios[0].items()
            if np.asarray(value).dtype.kind not in "OSU"
        ]

    def _ensure_indexed_cache(self) -> dict[str, np.ndarray]:
        """Build (once) and return the key -> stacked-array cache.

        Stacks every numeric scenario array along a new leading dataset
        dimension so ``get_batch_at`` can gather rows with ``jnp.take``.
        String-valued keys (``scenario/id``) are excluded — they cannot
        cross the JIT boundary.
        """
        if self._indexed_cache is None:
            self._indexed_cache = {
                key: np.stack([np.asarray(s[key]) for s in self._scenarios])
                for key in self._numeric_scenario_keys()
            }
        return self._indexed_cache

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Stateless indexed batch access for Pipeline-driven iteration.

        Returns ``size`` scenarios starting at logical position ``start``
        with wrap-around at the end of the source. Does not advance
        ``self.index`` or any other internal state, and accepts a traced
        ``start`` so the call composes with ``nnx.scan`` / ``nnx.jit``.

        Args:
            start: Starting logical index (concrete int or traced array).
            size: Number of scenarios to return (static Python int).
            key: Unused — WODSource serves scenarios in file order.

        Returns:
            Dict of numeric scenario arrays, each with leading dim ``size``.

        Raises:
            NotImplementedError: In streaming mode (forward-only).
            ValueError: When the source is empty.
        """
        del key
        if not self.supports_indexed_access():
            raise NotImplementedError(
                f"{type(self).__name__} in streaming mode does not support indexed "
                "batch access; use eager mode or the iterator path."
            )
        if not self._scenarios:
            raise ValueError("WODSource is empty; get_batch_at has nothing to serve.")

        stacked = self._ensure_indexed_cache()
        indices = resolve_wrapped_indices(
            start,
            size,
            len(self._scenarios),
            is_random_order=False,
            key=None,
        )
        return {
            key_name: jnp.take(jnp.asarray(value), indices, axis=0)
            for key_name, value in stacked.items()
        }

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Return a per-element shape/dtype contract for downstream consumers.

        Derived from the first materialized scenario; covers the numeric
        keys ``get_batch_at`` emits (string keys are excluded).

        Returns:
            Dict mapping scenario keys to ``jax.ShapeDtypeStruct``.

        Raises:
            NotImplementedError: In streaming mode (no record materialized).
            ValueError: When the eager source is empty.
        """
        if not self.supports_indexed_access():
            raise NotImplementedError(
                f"{type(self).__name__} in streaming mode cannot derive element_spec "
                "without materializing a record; use eager mode."
            )
        if not self._scenarios:
            raise ValueError("WODSource is empty; element_spec cannot be inferred.")
        sample = self._scenarios[0]
        return {key: array_to_spec(sample[key]) for key in self._numeric_scenario_keys()}

    # =========================================================================
    # WOD-Specific Methods
    # =========================================================================

    def get_by_scenario_id(self, scenario_id: str | bytes) -> Element | None:
        """Get scenario by its WOD scenario ID.

        Only available in eager mode.

        Args:
            scenario_id: WOD scenario identifier (str or bytes).

        Returns:
            Element wrapping the scenario, or None if not found.
        """
        if isinstance(scenario_id, str):
            scenario_id = scenario_id.encode("utf-8")
        for idx, sid in enumerate(self._scenario_ids):
            if sid == scenario_id:
                return self._scenario_to_element(idx, self._scenarios[idx])
        return None

    @property
    def scenario_ids(self) -> list:
        """Get list of all scenario IDs (eager mode only)."""
        return list(self._scenario_ids)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WODSource("
            f"split={self.config.split}, "
            f"mode={self.config.mode}, "
            f"scenarios={len(self._scenarios)}, "
            f"epoch={self.epoch.get_value()})"
        )
