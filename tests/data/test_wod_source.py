"""Tests for WOD data source configuration and loading.

Verifies WODSourceConfig validation, WODSource construction,
and datarax DataSourceModule contract. Uses mock data — no real
TFRecords required.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from datarax.typing import Element

from diffav.core.types import DatasetMode
from diffav.data.wod_source import WODSource, WODSourceConfig


# ---------------------------------------------------------------------------
# WODSourceConfig
# ---------------------------------------------------------------------------


class TestWODSourceConfig:
    """Tests for WODSourceConfig extending StructuralConfig."""

    def test_construction_with_defaults(self) -> None:
        """Config can be constructed with required fields and sensible defaults."""
        cfg = WODSourceConfig(
            wod_path="/data/waymo/v2",
            split="train",
            max_agents=32,
        )
        assert cfg.wod_path == "/data/waymo/v2"
        assert cfg.split == "train"
        assert cfg.max_agents == 32
        assert cfg.history_steps == 11
        assert cfg.future_steps == 80

    def test_construction_custom_steps(self) -> None:
        """Config accepts custom history and future steps."""
        cfg = WODSourceConfig(
            wod_path="/data/waymo",
            split="val",
            max_agents=16,
            history_steps=5,
            future_steps=40,
        )
        assert cfg.history_steps == 5
        assert cfg.future_steps == 40

    def test_frozen(self) -> None:
        """Config should be immutable after construction."""
        from datarax.core.config import FrozenInstanceError

        cfg = WODSourceConfig(wod_path="/data", split="train", max_agents=32)
        with pytest.raises(FrozenInstanceError):
            cfg.split = "val"  # type: ignore[misc]

    def test_invalid_split_rejected(self) -> None:
        """Config should reject invalid split values."""
        with pytest.raises(ValueError, match="split"):
            WODSourceConfig(wod_path="/data", split="invalid", max_agents=32)

    def test_empty_wod_path_rejected(self) -> None:
        """Config should reject empty wod_path."""
        with pytest.raises(ValueError, match="wod_path"):
            WODSourceConfig(wod_path="", split="train", max_agents=32)

    def test_max_agents_positive(self) -> None:
        """Config should reject non-positive max_agents."""
        with pytest.raises(ValueError, match="max_agents"):
            WODSourceConfig(wod_path="/data", split="train", max_agents=0)

    def test_history_steps_positive(self) -> None:
        """Config should reject non-positive history_steps."""
        with pytest.raises(ValueError, match="history_steps"):
            WODSourceConfig(wod_path="/data", split="train", max_agents=32, history_steps=0)

    def test_future_steps_positive(self) -> None:
        """Config should reject non-positive future_steps."""
        with pytest.raises(ValueError, match="future_steps"):
            WODSourceConfig(wod_path="/data", split="train", max_agents=32, future_steps=0)

    def test_total_steps_property(self) -> None:
        """total_steps should equal history_steps + future_steps."""
        cfg = WODSourceConfig(wod_path="/data", split="train", max_agents=32)
        assert cfg.total_steps == 91

    def test_mode_default_eager(self) -> None:
        """Default mode is eager."""
        cfg = WODSourceConfig(wod_path="/data", split="train", max_agents=32)
        assert cfg.mode == DatasetMode.EAGER

    def test_mode_streaming(self) -> None:
        """Mode can be set to streaming."""
        cfg = WODSourceConfig(
            wod_path="/data",
            split="train",
            max_agents=32,
            mode=DatasetMode.STREAMING,
        )
        assert cfg.mode == DatasetMode.STREAMING

    def test_invalid_mode_rejected(self) -> None:
        """Config should reject invalid mode values."""
        with pytest.raises(ValueError, match="is not a valid DatasetMode"):
            WODSourceConfig(
                wod_path="/data",
                split="train",
                max_agents=32,
                mode=cast(Any, "invalid"),
            )


# ---------------------------------------------------------------------------
# WODSource — datarax DataSourceModule contract
# ---------------------------------------------------------------------------


class TestWODSource:
    """Tests for WODSource extending datarax DataSourceModule."""

    def test_construction_from_dicts(
        self,
        raw_wod_dict: dict,
    ) -> None:
        """WODSource can be constructed from pre-loaded raw dicts."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert len(source) == 1

    def test_len(self, raw_wod_dict: dict) -> None:
        """__len__ returns the number of loaded scenarios."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict, raw_wod_dict])
        assert len(source) == 2

    def test_getitem_returns_element(self, raw_wod_dict: dict) -> None:
        """__getitem__ returns a datarax Element wrapping the scenario."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        elem = source[0]
        assert isinstance(elem, Element)
        assert "state/all/x" in elem.data

    def test_getitem_metadata(self, raw_wod_dict: dict) -> None:
        """Element metadata contains scenario index and ID."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        elem = source[0]
        assert elem is not None
        assert elem.metadata is not None
        assert elem.metadata.index == 0
        assert elem.metadata.entry_key == "test_scenario_001"
        assert elem.metadata.source_info is not None
        assert elem.metadata.source_info["split"] == "train"

    def test_getitem_out_of_bounds(self, raw_wod_dict: dict) -> None:
        """__getitem__ returns None for out-of-bounds (datarax convention)."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert source[5] is None

    def test_iter_yields_elements(self, raw_wod_dict: dict) -> None:
        """__iter__/__next__ yields Element objects."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict, raw_wod_dict])
        elements = list(source)
        assert len(elements) == 2
        assert all(isinstance(e, Element) for e in elements)
        assert "state/all/x" in elements[0].data

    def test_iter_resets(self, raw_wod_dict: dict) -> None:
        """Calling __iter__ again resets the iteration."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        first_pass = list(source)
        second_pass = list(source)
        assert len(first_pass) == len(second_pass) == 1

    def test_config_accessible(self, raw_wod_dict: dict) -> None:
        """Source exposes its config."""
        cfg = WODSourceConfig(wod_path="/mock", split="val", max_agents=16)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert source.config.split == "val"
        assert source.config.max_agents == 16

    def test_empty_source(self) -> None:
        """WODSource can be constructed with no scenarios."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[])
        assert len(source) == 0
        assert list(source) == []

    def test_reset(self, raw_wod_dict: dict) -> None:
        """reset() allows re-iteration."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        _ = list(source)
        source.reset()
        assert len(list(source)) == 1

    def test_get_batch(self, raw_wod_dict: dict) -> None:
        """get_batch returns a list of Elements."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict, raw_wod_dict, raw_wod_dict])
        batch = source.get_batch(2)
        assert len(batch) == 2
        assert all(isinstance(e, Element) for e in batch)

    def test_get_by_scenario_id(self, raw_wod_dict: dict) -> None:
        """get_by_scenario_id returns the matching Element."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        elem = source.get_by_scenario_id("test_scenario_001")
        assert isinstance(elem, Element)
        assert elem.metadata is not None
        assert elem.metadata.entry_key == "test_scenario_001"

    def test_get_by_scenario_id_not_found(self, raw_wod_dict: dict) -> None:
        """get_by_scenario_id returns None when not found."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert source.get_by_scenario_id("nonexistent") is None

    def test_scenario_ids_property(self, raw_wod_dict: dict) -> None:
        """scenario_ids property lists all loaded IDs."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert len(source.scenario_ids) == 1

    def test_repr(self, raw_wod_dict: dict) -> None:
        """__repr__ includes mode, split, and scenario count."""
        cfg = WODSourceConfig(wod_path="/mock", split="val", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        r = repr(source)
        assert "val" in r
        assert "eager" in r

    def test_epoch_increments_on_iter(self, raw_wod_dict: dict) -> None:
        """Epoch counter increments on each __iter__ call."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])
        assert source.epoch.get_value() == 0
        list(source)
        assert source.epoch.get_value() == 1
        list(source)
        assert source.epoch.get_value() == 2


# ---------------------------------------------------------------------------
# Indexed batch access + element spec (datarax v0.1.4 Pipeline contract)
# ---------------------------------------------------------------------------


def _scenario_variant(raw_wod_dict: dict, index: int) -> dict:
    """Return a copy of the mock scenario with distinct x values and ID."""
    variant = dict(raw_wod_dict)
    variant["state/all/x"] = raw_wod_dict["state/all/x"] + float(index) * 100.0
    variant["scenario/id"] = np.array([f"scenario_{index:03d}".encode()])
    return variant


class TestWODSourceIndexedAccess:
    """Tests for the stateless get_batch_at / element_spec contract."""

    @pytest.fixture()
    def source(self, raw_wod_dict: dict) -> WODSource:
        """Eager source with three distinguishable scenarios."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        scenarios = [_scenario_variant(raw_wod_dict, i) for i in range(3)]
        return WODSource(config=cfg, raw_scenarios=scenarios)

    @pytest.fixture()
    def streaming_source(self, monkeypatch: pytest.MonkeyPatch) -> WODSource:
        """Streaming-mode source with the TF dataset build stubbed out."""
        monkeypatch.setattr(WODSource, "_build_streaming_dataset", lambda self: None)
        cfg = WODSourceConfig(
            wod_path="/mock",
            split="train",
            max_agents=32,
            mode=DatasetMode.STREAMING,
        )
        return WODSource(config=cfg)

    def test_supports_indexed_access_eager(self, source: WODSource) -> None:
        """Eager sources support random access."""
        assert source.supports_indexed_access() is True

    def test_supports_indexed_access_streaming(self, streaming_source: WODSource) -> None:
        """Streaming sources are forward-only."""
        assert streaming_source.supports_indexed_access() is False

    def test_get_batch_at_leading_dim(self, source: WODSource) -> None:
        """Every array in the batch has leading dimension == size."""
        batch = source.get_batch_at(0, 2)
        assert isinstance(batch, dict)
        assert batch
        for key, value in batch.items():
            assert value.shape[0] == 2, key

    def test_get_batch_at_excludes_string_keys(self, source: WODSource) -> None:
        """Non-numeric keys (scenario/id) are excluded from the traced batch."""
        batch = source.get_batch_at(0, 2)
        assert "scenario/id" not in batch

    def test_get_batch_at_values_match_scenarios(
        self, source: WODSource, raw_wod_dict: dict
    ) -> None:
        """Rows come from the scenarios at the requested logical positions."""
        batch = source.get_batch_at(1, 2)
        expected_row0 = raw_wod_dict["state/all/x"] + 100.0
        expected_row1 = raw_wod_dict["state/all/x"] + 200.0
        np.testing.assert_allclose(np.asarray(batch["state/all/x"][0]), expected_row0)
        np.testing.assert_allclose(np.asarray(batch["state/all/x"][1]), expected_row1)

    def test_get_batch_at_wraps_around(self, source: WODSource, raw_wod_dict: dict) -> None:
        """Indices wrap modulo the dataset length."""
        batch = source.get_batch_at(2, 2)
        expected_row0 = raw_wod_dict["state/all/x"] + 200.0
        expected_row1 = raw_wod_dict["state/all/x"]
        np.testing.assert_allclose(np.asarray(batch["state/all/x"][0]), expected_row0)
        np.testing.assert_allclose(np.asarray(batch["state/all/x"][1]), expected_row1)

    def test_get_batch_at_stateless(self, source: WODSource) -> None:
        """Repeated calls return identical data and do not advance iteration state."""
        index_before = source.index.get_value()
        first = source.get_batch_at(0, 2)
        second = source.get_batch_at(0, 2)
        assert source.index.get_value() == index_before
        for key in first:
            np.testing.assert_array_equal(np.asarray(first[key]), np.asarray(second[key]))

    def test_get_batch_at_traceable(self, source: WODSource) -> None:
        """get_batch_at accepts a traced start index under jax.jit."""
        import jax

        jitted = jax.jit(lambda start: source.get_batch_at(start, 2))
        traced = jitted(jnp.asarray(1))
        eager = source.get_batch_at(1, 2)
        for key in eager:
            np.testing.assert_array_equal(np.asarray(traced[key]), np.asarray(eager[key]))

    def test_get_batch_at_streaming_raises(self, streaming_source: WODSource) -> None:
        """Streaming mode has no random access."""
        with pytest.raises(NotImplementedError, match="streaming"):
            streaming_source.get_batch_at(0, 2)

    def test_element_spec_matches_batch(self, source: WODSource) -> None:
        """Spec keys, shapes, and dtypes agree with what get_batch_at emits."""
        import jax

        spec = source.element_spec()
        batch = source.get_batch_at(0, 1)
        assert set(spec) == set(batch)
        for key, struct in spec.items():
            assert isinstance(struct, jax.ShapeDtypeStruct)
            assert batch[key].shape == (1, *struct.shape), key
            assert batch[key].dtype == struct.dtype, key

    def test_element_spec_empty_raises(self) -> None:
        """An empty eager source cannot derive a spec."""
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[])
        with pytest.raises(ValueError, match="empty"):
            source.element_spec()

    def test_element_spec_streaming_raises(self, streaming_source: WODSource) -> None:
        """Streaming mode cannot derive a spec without materializing a record."""
        with pytest.raises(NotImplementedError, match="streaming"):
            streaming_source.element_spec()


# ---------------------------------------------------------------------------
# TFRecord feature schema caching
# ---------------------------------------------------------------------------


class TestFeaturesDescriptionCache:
    """The ~49-feature TF schema must be built once per source, not per record."""

    def test_features_description_built_once(
        self,
        raw_wod_dict: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated schema requests reuse one built description."""
        calls = {"count": 0}
        sentinel = {"schema": "sentinel"}

        def fake_builder(max_num_objects: int = 0, max_num_rg_points: int = 0) -> dict:
            del max_num_objects, max_num_rg_points
            calls["count"] += 1
            return sentinel

        monkeypatch.setattr(WODSource, "_get_features_description", staticmethod(fake_builder))
        cfg = WODSourceConfig(wod_path="/mock", split="train", max_agents=32)
        source = WODSource(config=cfg, raw_scenarios=[raw_wod_dict])

        first = source._features_description()
        second = source._features_description()

        assert first is sentinel
        assert second is sentinel
        assert calls["count"] == 1
