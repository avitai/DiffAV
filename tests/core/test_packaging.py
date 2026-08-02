"""Packaging marker tests."""

from __future__ import annotations

from pathlib import Path

import simulacrax


class TestTypedMarker:
    """The package ships a PEP 561 py.typed marker."""

    def test_py_typed_exists(self) -> None:
        package_dir = Path(simulacrax.__file__).parent
        assert (package_dir / "py.typed").is_file()
