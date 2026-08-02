"""Tests for shared configuration validators in core.config."""

from __future__ import annotations

import pytest

from simulacrax.core.config import (
    VALID_SPLITS,
    validate_positive,
    validate_transformer_fields,
)


class TestValidatePositive:
    """validate_positive rejects non-positive values with a named message."""

    def test_positive_value_passes(self) -> None:
        validate_positive("batch_size", 4)

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            validate_positive("batch_size", 0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            validate_positive("learning_rate", -0.1)


class TestValidateTransformerFields:
    """validate_transformer_fields enforces positivity and head divisibility."""

    def test_valid_fields_pass(self) -> None:
        validate_transformer_fields(64, 8)

    def test_indivisible_heads_rejected(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            validate_transformer_fields(64, 7)

    def test_extra_positive_fields_checked(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            validate_transformer_fields(64, 8, extra_positive={"num_layers": 0})


class TestValidSplits:
    """VALID_SPLITS carries the canonical dataset split names."""

    def test_contains_canonical_splits(self) -> None:
        assert VALID_SPLITS == frozenset({"train", "val", "test"})
