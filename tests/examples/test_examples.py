"""Tests for DiffAV example files.

Following the sibling-repo convention, these tests validate that examples:

1. Follow the documentation structure (cell markers, markdown, metadata)
2. Stay synchronized with their notebook counterparts (content-level)
3. Execute without errors (slow tier; examples that need the WOD dataset
   skip cleanly when it is not configured)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.jupytext_converter import _comparable_cells


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_TIMEOUT_SECONDS = int(os.environ.get("DIFFAV_EXAMPLE_TIMEOUT_SECONDS", "600"))

REQUIRED_SECTIONS = ("Overview",)
LEARNING_GOALS_PATTERN = re.compile(
    r"(?:Learning Goals|Learning Objectives|What You'll Learn)", re.IGNORECASE
)

# Failure signatures identifying a missing external dataset rather than a
# code defect: those runs skip instead of failing.
_DATA_UNAVAILABLE_SIGNATURES = (
    "wod_motion_tfrecord_path is not set",
    "no such file or directory",
    "not found: ",
)


def find_example_files() -> list[Path]:
    """Return every numbered example script under ``examples/``."""
    return sorted(
        py_file for py_file in EXAMPLES_DIR.rglob("*.py") if re.match(r"^\d+_", py_file.name)
    )


def example_id(path: Path) -> str:
    """Short test ID from an example path."""
    return str(path.relative_to(EXAMPLES_DIR))


EXAMPLE_FILES = find_example_files()


class TestExampleStructure:
    """Structure and content conventions for every example."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_cell_markers(self, example_path: Path) -> None:
        content = example_path.read_text()
        markers = re.findall(r"^# %%", content, re.MULTILINE)
        assert len(markers) >= 3, f"Expected at least 3 cell markers, found {len(markers)}"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_markdown_cells(self, example_path: Path) -> None:
        content = example_path.read_text()
        markdown_cells = re.findall(r"^# %% \[markdown\]", content, re.MULTILINE)
        assert len(markdown_cells) >= 2, (
            f"Expected at least 2 markdown cells, found {len(markdown_cells)}"
        )

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_required_sections(self, example_path: Path) -> None:
        content = example_path.read_text()
        for section in REQUIRED_SECTIONS:
            assert re.search(rf"##\s+{section}", content, re.IGNORECASE), (
                f"Missing required section: {section}"
            )

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_metadata_table(self, example_path: Path) -> None:
        content = example_path.read_text()
        assert re.search(r"\*\*Level\*\*", content), "Missing Level in metadata"
        assert re.search(r"\*\*Runtime", content), "Missing Runtime in metadata"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_learning_goals(self, example_path: Path) -> None:
        content = example_path.read_text()
        assert LEARNING_GOALS_PATTERN.search(content), "Missing a learning-goals section"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_no_star_imports(self, example_path: Path) -> None:
        content = example_path.read_text()
        star_imports = re.findall(r"^\s*from\s+\S+\s+import\s+\*", content, re.MULTILINE)
        assert not star_imports, f"Found star imports: {star_imports}"


class TestExampleSync:
    """Content-level notebook synchronization for every example."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_notebook_exists(self, example_path: Path) -> None:
        assert example_path.with_suffix(".ipynb").exists(), (
            f"Missing notebook pair for {example_path.name}"
        )

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_notebook_content_matches(self, example_path: Path) -> None:
        notebook_path = example_path.with_suffix(".ipynb")
        if not notebook_path.exists():
            pytest.skip("Notebook does not exist")
        assert _comparable_cells(example_path) == _comparable_cells(notebook_path), (
            f"{example_path.name} and its notebook differ; run "
            f"scripts/jupytext_converter.py sync {example_path}"
        )


@pytest.mark.slow
class TestExampleExecution:
    """Execute every example end-to-end (slow tier)."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    @pytest.mark.timeout(EXAMPLE_TIMEOUT_SECONDS + 60)
    def test_example_executes(self, example_path: Path, tmp_path: Path) -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                env={
                    **os.environ,
                    "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu"),
                    # Examples honouring this knob shrink training/fit loops
                    # to smoke scale; showcase scale stays for real runs.
                    "DIFFAV_EXAMPLES_SMOKE": "1",
                    # Keep smoke-scale plots away from the showcase
                    # artifacts under docs/assets/images/examples.
                    "DIFFAV_EXAMPLES_OUTPUT_DIR": str(tmp_path),
                },
                text=True,
                timeout=EXAMPLE_TIMEOUT_SECONDS,
                cwd=EXAMPLES_DIR.parent,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Example exceeded {EXAMPLE_TIMEOUT_SECONDS}s")
        if result.returncode != 0:
            lowered = result.stderr.lower()
            if any(sig in lowered for sig in _DATA_UNAVAILABLE_SIGNATURES):
                pytest.skip("Example needs the WOD dataset, which is not configured here.")
            stderr = result.stderr if len(result.stderr) <= 3000 else result.stderr[-3000:]
            pytest.fail(f"Example failed with exit code {result.returncode}:\n{stderr}")
