#!/usr/bin/env python3
"""
Documentation Generator for Simulacrax.

============================================

PURPOSE:
    Automatically generates documentation from source code,
    with dynamic discovery of project structure and intelligent organization.

FEATURES:
    - Dynamic project structure discovery
    - Automatic module categorization
    - Docstring extraction and formatting
    - Markdown file aggregation from source
    - Progress tracking and error handling
    - Incremental generation support

USAGE:
    python scripts/generate_docs.py [OPTIONS]

OPTIONS:
    --src-path PATH      Source directory (default: src/simulacrax)
    --docs-path PATH     Documentation output directory (default: docs)
    --clean              Clean existing docs before generation
    --verbose            Enable verbose output
    --incremental        Only regenerate changed files

Examples:
    # Standard generation
    python scripts/generate_docs.py

    # Clean rebuild with verbose output
    python scripts/generate_docs.py --clean --verbose

    # Custom paths
    python scripts/generate_docs.py --src-path src/custom --docs-path docs/custom

OUTPUT:
    Creates/updates documentation in the specified docs directory with:
    - Organized module documentation
    - API reference
    - Copied markdown files from source
    - Updated MkDocs navigation

License: MIT
"""

import argparse
import ast
import hashlib
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    path: Path
    relative_path: Path
    module_name: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    has_docstring: bool = False
    docstring: str | None = None
    imports_count: int = 0
    last_modified: float = 0
    content_hash: str = ""


@dataclass
class DocSection:
    """Documentation section with modules."""

    name: str
    title: str
    path: Path
    modules: list[ModuleInfo] = field(default_factory=list)
    subsections: dict[str, "DocSection"] = field(default_factory=dict)


class ModernDocGenerator:
    """Documentation generator with dynamic discovery and smart organization."""

    def __init__(
        self,
        src_path: str = "src/simulacrax",
        docs_path: str = "docs",
        clean: bool = False,
        verbose: bool = False,
        incremental: bool = False,
    ):
        """Initialize the documentation generator.

        Args:
            src_path: Source code directory.
            docs_path: Documentation output directory.
            clean: Whether to clean existing docs.
            verbose: Enable verbose output.
            incremental: Only regenerate changed files.
        """
        self.src_path = Path(src_path)
        self.docs_path = Path(docs_path)
        self.clean_mode = clean
        self.verbose = verbose
        self.incremental = incremental

        # Cache for incremental builds
        self.cache_file = self.docs_path / ".doc_cache.json"
        self.cache = self._load_cache() if incremental else {}

        # Discovered structure
        self.sections: dict[str, DocSection] = {}
        self.all_modules: list[ModuleInfo] = []

        # Section name mappings for better titles
        self.section_titles = {
            "core": "Core Components",
            "data": "Data Processing",
            "models": "Model Implementations",
            "alignment": "Alignment (DPO/RLHF)",
            "evaluation": "Evaluation Metrics",
            "physics": "Physics Validation",
            "occupancy": "Occupancy Flow",
            "sensor": "Sensor Simulation",
            "api": "Orchestration API",
            "utils": "Utilities",
        }

    def _load_cache(self) -> dict:
        """Load documentation cache for incremental builds."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save documentation cache."""
        if self.incremental:
            try:
                self.docs_path.mkdir(exist_ok=True)
                with open(self.cache_file, "w") as f:
                    json.dump(self.cache, f, indent=2)
            except OSError as e:
                logger.warning(f"Could not save cache: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents for change detection."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except OSError:
            return ""

    def _needs_regeneration(self, module_info: ModuleInfo) -> bool:
        """Check if a module needs documentation regeneration."""
        if not self.incremental:
            return True

        cache_key = str(module_info.relative_path)
        if cache_key not in self.cache:
            return True

        cached_hash = self.cache.get(cache_key, {}).get("hash", "")
        return cached_hash != module_info.content_hash

    def clean_existing_docs(self) -> None:
        """Clean existing documentation, preserving important files."""
        logger.info("Cleaning existing documentation...")

        preserve = {
            "index.md",
            "api.md",
            "_static",
            "_overrides",
            ".doc_cache.json",
        }

        if not self.docs_path.exists():
            self.docs_path.mkdir(parents=True)
            return

        for item in self.docs_path.iterdir():
            if item.name not in preserve:
                if item.is_file():
                    if self.verbose:
                        logger.info(f"  Removing: {item}")
                    item.unlink()
                elif item.is_dir():
                    if self.verbose:
                        logger.info(f"  Removing directory: {item}")
                    shutil.rmtree(item)

    def discover_project_structure(self) -> None:
        """Dynamically discover the project structure."""
        logger.info("Discovering project structure...")

        if not self.src_path.exists():
            logger.error(f"Source path does not exist: {self.src_path}")
            sys.exit(1)

        python_files = list(self.src_path.rglob("*.py"))

        python_files = [
            f
            for f in python_files
            if "__pycache__" not in f.parts
            and not f.name.startswith("test_")
            and "tests" not in f.parts
        ]

        logger.info(f"  Found {len(python_files)} Python files")

        for file_path in python_files:
            module_info = self._extract_module_info(file_path)
            if module_info:
                self.all_modules.append(module_info)
                self._categorize_module(module_info)

        logger.info(f"  Discovered {len(self.sections)} top-level sections")
        if self.verbose:
            for section_name in sorted(self.sections.keys()):
                section = self.sections[section_name]
                logger.info(f"    - {section_name}: {len(section.modules)} modules")

    def _extract_module_info(self, file_path: Path) -> ModuleInfo | None:
        """Extract information from a Python module."""
        try:
            relative_path = file_path.relative_to(self.src_path)

            if file_path.name == "__init__.py":
                if file_path.stat().st_size < 100:
                    return None

            module_parts = [*relative_path.parts[:-1], relative_path.stem]
            module_name = ".".join(module_parts)

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            content_hash = self._get_file_hash(file_path)

            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                if self.verbose:
                    logger.warning(f"  Syntax error in {file_path}: {e}")
                return None

            classes = []
            functions = []
            imports_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_") or node.name in ["__init__", "__call__"]:
                        functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports_count += 1

            docstring = ast.get_docstring(tree)

            return ModuleInfo(
                path=file_path,
                relative_path=relative_path,
                module_name=module_name,
                classes=classes,
                functions=functions,
                has_docstring=docstring is not None,
                docstring=docstring,
                imports_count=imports_count,
                last_modified=file_path.stat().st_mtime,
                content_hash=content_hash,
            )

        except OSError as e:
            if self.verbose:
                logger.warning(f"  Could not process {file_path}: {e}")
            return None

    def _categorize_module(self, module_info: ModuleInfo) -> None:
        """Categorize a module into the appropriate documentation section."""
        parts = module_info.relative_path.parts

        if len(parts) == 1:
            section_name = "root"
        else:
            section_name = parts[0]

        if section_name not in self.sections:
            section_title = self.section_titles.get(
                section_name, section_name.replace("_", " ").title()
            )

            section_path = self.docs_path / section_name
            self.sections[section_name] = DocSection(
                name=section_name, title=section_title, path=section_path
            )

        self.sections[section_name].modules.append(module_info)

    def generate_module_documentation(self, module_info: ModuleInfo) -> str:
        """Generate documentation for a single module."""
        if not self._needs_regeneration(module_info):
            if self.verbose:
                logger.info(f"    Skipping unchanged: {module_info.relative_path}")
            return ""

        lines = []

        module_title = module_info.path.stem.replace("_", " ").title()
        if module_title == "Init":
            module_title = "Package Initialization"

        root_package = self.src_path.name  # "simulacrax"
        full_module_path = f"{root_package}.{module_info.module_name}"

        lines.extend(
            [
                f"# {module_title}",
                "",
                f"::: {full_module_path}",
                "",
            ]
        )

        if self.incremental:
            cache_key = str(module_info.relative_path)
            self.cache[cache_key] = {
                "hash": module_info.content_hash,
                "timestamp": module_info.last_modified,
            }

        return "\n".join(lines)

    def generate_section_documentation(self) -> None:
        """Generate documentation for all sections."""
        logger.info("Generating section documentation...")

        for section_name in sorted(self.sections.keys()):
            section = self.sections[section_name]

            if not section.modules:
                continue

            logger.info(f"  Generating {section.title} ({len(section.modules)} modules)")

            section.path.mkdir(parents=True, exist_ok=True)

            self._generate_section_index(section)

            for module_info in sorted(section.modules, key=lambda m: m.module_name):
                if module_info.path.name == "__init__.py":
                    continue

                module_doc = self.generate_module_documentation(module_info)
                if module_doc:
                    doc_filename = f"{module_info.path.stem}.md"
                    doc_path = section.path / doc_filename

                    with open(doc_path, "w", encoding="utf-8") as f:
                        f.write(module_doc)

    def _generate_section_index(self, section: DocSection) -> None:
        """Generate index file for a documentation section."""
        lines = [
            f"# {section.title}",
            "",
            f"This section contains documentation for {section.title.lower()}.",
            "",
            "## Modules",
            "",
        ]

        module_groups = defaultdict(list)

        for module_info in sorted(section.modules, key=lambda m: m.module_name):
            if module_info.path.name == "__init__.py":
                continue

            rel_parts = module_info.relative_path.parts
            if len(rel_parts) > 2:
                group_name = rel_parts[-2]
            else:
                group_name = "main"

            module_groups[group_name].append(module_info)

        if len(module_groups) == 1 and "main" in module_groups:
            for module_info in module_groups["main"]:
                doc_name = module_info.path.stem
                lines.append(f"- [{doc_name}]({doc_name}.md)")
        else:
            for group_name in sorted(module_groups.keys()):
                if group_name != "main":
                    lines.extend(
                        [
                            "",
                            f"### {group_name.replace('_', ' ').title()}",
                            "",
                        ]
                    )

                for module_info in module_groups[group_name]:
                    doc_name = module_info.path.stem
                    lines.append(f"- [{doc_name}]({doc_name}.md)")

        lines.extend(
            [
                "",
                "## Statistics",
                "",
                f"- Total modules: {len(section.modules)}",
                f"- Total classes: {sum(len(m.classes) for m in section.modules)}",
                f"- Total functions: {sum(len(m.functions) for m in section.modules)}",
            ]
        )

        index_path = section.path / "index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def copy_markdown_files(self) -> None:
        """Copy existing markdown files from source directories."""
        logger.info("Copying markdown files from source...")

        copied_count = 0
        scan_dirs = ["src", "tests", "examples", "notebooks"]

        for dir_name in scan_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                continue

            markdown_files = list(dir_path.rglob("*.md"))

            for md_file in markdown_files:
                if md_file.name == "README.md" and md_file.parent != dir_path:
                    continue

                relative_path = md_file.relative_to(dir_path)
                target_path = self.docs_path / dir_name / relative_path

                target_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(md_file, target_path)
                copied_count += 1

                if self.verbose:
                    logger.info(f"  Copied: {md_file} -> {target_path}")

        logger.info(f"  Copied {copied_count} markdown files")

    def generate_summary(self) -> None:
        """Generate a summary of the documentation generation."""
        logger.info("Documentation Generation Summary")
        logger.info("=" * 50)
        logger.info(f"  Total modules processed: {len(self.all_modules)}")
        logger.info(f"  Total sections created: {len(self.sections)}")
        logger.info(
            f"  Modules with docstrings: {sum(1 for m in self.all_modules if m.has_docstring)}"
        )

        if self.incremental:
            logger.info(f"  Cache entries: {len(self.cache)}")

        logger.info("  Section breakdown:")
        for section_name in sorted(self.sections.keys()):
            section = self.sections[section_name]
            if section.modules:
                logger.info(f"    - {section.title}: {len(section.modules)} modules")

    def run(self) -> None:
        """Run the complete documentation generation process."""
        logger.info("Starting documentation generation...")

        try:
            if self.clean_mode:
                self.clean_existing_docs()

            self.discover_project_structure()
            self.generate_section_documentation()
            self.copy_markdown_files()
            self._save_cache()
            self.generate_summary()

            logger.info("Documentation generation complete!")
            logger.info("Run 'mkdocs build' to build the documentation")
            logger.info("Run 'mkdocs serve' to preview the documentation")

        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"Documentation generation failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)


def main() -> None:
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(
        description="Documentation generator for Simulacrax",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--src-path",
        default="src/simulacrax",
        help="Source code directory (default: src/simulacrax)",
    )

    parser.add_argument(
        "--docs-path", default="docs", help="Documentation output directory (default: docs)"
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean existing documentation before generation"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only regenerate changed files (faster for large projects)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    generator = ModernDocGenerator(
        src_path=args.src_path,
        docs_path=args.docs_path,
        clean=args.clean,
        verbose=args.verbose,
        incremental=args.incremental,
    )

    generator.run()


if __name__ == "__main__":
    main()
