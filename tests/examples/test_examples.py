"""Tests for DiffAV example files.

Following the sibling-repo convention, these tests validate that examples:

1. Follow the documentation structure (cell markers, markdown, metadata)
2. Stay synchronized with their notebook counterparts (content-level)
3. Execute without errors in their own interpreter, through
   ``substrax.testing.run_example`` (slow tier; an example that needs the WOD
   dataset skips when it is not configured, and a run over budget fails)
4. Write their figures where ``substrax.artifacts`` resolves, never into the tree
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from scripts.jupytext_converter import _comparable_cells
from substrax.testing import discover_examples, run_example, unavailable_reason


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_TIMEOUT_SECONDS = int(os.environ.get("DIFFAV_EXAMPLE_TIMEOUT_SECONDS", "600"))

REQUIRED_SECTIONS = ("Overview",)
LEARNING_GOALS_PATTERN = re.compile(
    r"(?:Learning Goals|Learning Objectives|What You'll Learn)", re.IGNORECASE
)

# The one failure that means the WOD dataset is not configured here, so the run skips
# rather than fails. Broader file-not-found messages are not in this list: they also
# match unrelated defects.
_DATA_UNAVAILABLE_SIGNATURES = ("wod_motion_tfrecord_path is not set",)


def find_example_files() -> list[Path]:
    """Return every numbered example script under ``examples/``."""
    return discover_examples(
        EXAMPLES_DIR, include=lambda path: re.match(r"^\d+_", path.name) is not None
    )


def example_id(path: Path) -> str:
    """Short test ID from an example path."""
    return str(path.relative_to(EXAMPLES_DIR))


EXAMPLE_FILES = find_example_files()


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh output directory, with ``AVITAI_OUTPUT_DIR`` pointing at it."""
    path = tmp_path / "outputs"
    path.mkdir()
    monkeypatch.setenv("AVITAI_OUTPUT_DIR", str(path))
    return path


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

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_outputs_resolve_through_substrax(self, example_path: Path) -> None:
        """An example that saves figures resolves its directory, never a fixed path."""
        content = example_path.read_text()
        assert "DIFFAV_EXAMPLES_OUTPUT_DIR" not in content, (
            f"{example_path.name} reads the retired output variable; AVITAI_OUTPUT_DIR is read by "
            "substrax.artifacts.resolve_output_dir"
        )
        if "savefig(" in content:
            assert "resolve_output_dir(" in content, (
                f"{example_path.name} saves figures without resolve_output_dir"
            )


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
    @pytest.mark.timeout(0)
    def test_example_executes(self, example_path: Path, output_dir: Path) -> None:
        """The example runs as a script in a child on the CPU backend, within its budget.

        The runner enforces ``EXAMPLE_TIMEOUT_SECONDS`` and names it on a timeout, so
        pytest-timeout is off for this test. Examples honouring ``DIFFAV_EXAMPLES_SMOKE``
        shrink training and fit loops to smoke scale; showcase scale stays for real runs.
        """
        run = run_example(
            example_path,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
            call_main=False,
            env={"DIFFAV_EXAMPLES_SMOKE": "1"},
        )
        if run.result.returncode != 0:
            reason = unavailable_reason(run, _DATA_UNAVAILABLE_SIGNATURES)
            if reason is not None:
                pytest.skip("Example needs the WOD dataset, which is not configured here.")
            run.result.check()
