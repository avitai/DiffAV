#!/usr/bin/env python3
"""Documentation Validation and Auto-Fix Script.

PURPOSE:
    Validates MkDocs documentation for broken references and optionally fixes them.
    Checks for missing modules in API documentation and updates paths.

FEATURES:
    - Validates Python module references in mkdocstrings
    - Auto-fixes incorrect module paths
    - Checks for missing directories and files
    - Validates navigation structure in mkdocs.yml
    - Reports issues with detailed suggestions

USAGE:
    python scripts/validate_docs.py [OPTIONS]

OPTIONS:
    --fix           Auto-fix issues where possible
    --check-only    Only check for issues, don't fix
    --verbose       Show detailed output

Examples:
    # Check for issues
    python scripts/validate_docs.py --check-only

    # Auto-fix issues
    python scripts/validate_docs.py --fix

    # Verbose checking with fixes
    python scripts/validate_docs.py --fix --verbose

License: MIT
"""

import argparse
import importlib.util
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MkDocsSafeLoader(yaml.SafeLoader):
    """YAML loader that tolerates MkDocs python/name tags as plain strings."""


def _construct_python_name(
    loader: MkDocsSafeLoader,
    tag_suffix: str,
    node: yaml.Node,
) -> str:
    """Convert ``!!python/name:...`` tags to string values for validation."""
    _ = loader, node
    return tag_suffix


MkDocsSafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    _construct_python_name,
)


class DocValidator:
    """Validates and fixes documentation issues."""

    def __init__(
        self,
        docs_path: str = "docs",
        src_path: str = "src",
        fix: bool = False,
        verbose: bool = False,
    ):
        """Initialize the validator.

        Args:
            docs_path: Path to documentation directory.
            src_path: Path to source code directory.
            fix: Whether to auto-fix issues.
            verbose: Enable verbose output.
        """
        self.docs_path = Path(docs_path)
        self.src_path = Path(src_path)
        self.fix_mode = fix
        self.verbose = verbose

        self.issues: list[dict] = []
        self.fixes_applied: list[str] = []

        self.module_map: dict[str, str] = {}
        self._discover_modules()

    def _discover_modules(self) -> None:
        """Discover all Python modules in the source tree."""
        logger.info("Discovering Python modules...")

        for init_file in self.src_path.rglob("__init__.py"):
            package_dir = init_file.parent

            try:
                rel_path = package_dir.relative_to(self.src_path)
                module_name = str(rel_path).replace("/", ".")

                self.module_map[module_name] = str(package_dir)

                for py_file in package_dir.glob("*.py"):
                    if py_file.name != "__init__.py":
                        file_module = f"{module_name}.{py_file.stem}"
                        self.module_map[file_module] = str(py_file)

            except ValueError:
                continue

        if self.verbose:
            logger.info(f"  Found {len(self.module_map)} modules")

    def _check_module_exists(self, module_name: str) -> bool:
        """Check if a Python module exists.

        Args:
            module_name: Fully qualified module name.

        Returns:
            True if module exists.
        """
        if module_name in self.module_map:
            return True

        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            # Unknown top-level packages raise instead of returning None.
            return False
        return spec is not None

    def _find_correct_module(self, incorrect_module: str) -> str | None:
        """Try to find the correct module path for an incorrect one.

        Args:
            incorrect_module: The incorrect module path.

        Returns:
            Correct module path or None.
        """
        module_end = incorrect_module.split(".")[-1]
        # Only rewrite when the trailing segment identifies exactly one
        # known module — an arbitrary first match would write wrong paths.
        matches = [known for known in self.module_map if known.rsplit(".", 1)[-1] == module_end]
        return matches[0] if len(matches) == 1 else None

    def validate_api_docs(self) -> None:
        """Validate API documentation files for valid module references."""
        logger.info("Validating API documentation...")

        md_files = list(self.docs_path.rglob("*.md"))

        for md_file in md_files:
            with open(md_file) as f:
                content = f.read()

            pattern = r"^::: ([\w\.]+)$"
            lines = content.split("\n")

            issues_found = False
            fixed_lines = []

            for i, line in enumerate(lines):
                match = re.match(pattern, line)
                if match:
                    module_name = match.group(1)

                    if not self._check_module_exists(module_name):
                        issues_found = True
                        issue = {
                            "file": str(md_file),
                            "line": i + 1,
                            "module": module_name,
                            "type": "missing_module",
                        }

                        correct_module = self._find_correct_module(module_name)
                        if correct_module:
                            issue["suggestion"] = correct_module
                            logger.warning(f"  Line {i + 1}: Module '{module_name}' not found")
                            logger.info(f"     -> Suggestion: '{correct_module}'")

                            if self.fix_mode:
                                line = f"::: {correct_module}"
                                self.fixes_applied.append(
                                    f"Fixed: {module_name} -> {correct_module}"
                                )
                        else:
                            logger.error(
                                f"  Line {i + 1}: Module '{module_name}' not found (no suggestion)"
                            )

                        self.issues.append(issue)
                    elif self.verbose:
                        logger.info(f"  Line {i + 1}: Module '{module_name}' exists")

                fixed_lines.append(line)

            if self.fix_mode and issues_found:
                with open(md_file, "w") as f:
                    f.write("\n".join(fixed_lines))
                logger.info(f"  Fixed issues in {md_file}")

    def validate_mkdocs_config(self) -> None:
        """Validate mkdocs.yml configuration."""
        logger.info("Validating MkDocs configuration...")

        mkdocs_file = Path("mkdocs.yml")
        if not mkdocs_file.exists():
            logger.error("  mkdocs.yml not found!")
            return

        with open(mkdocs_file, encoding="utf-8") as f:
            config = yaml.load(f, Loader=MkDocsSafeLoader)

        if "theme" in config and "custom_dir" in config["theme"]:
            custom_dir = Path(config["theme"]["custom_dir"])
            if not custom_dir.exists():
                logger.warning(f"  Theme custom_dir does not exist: {custom_dir}")

                if self.fix_mode:
                    custom_dir.mkdir(parents=True, exist_ok=True)
                    (custom_dir / ".gitkeep").touch()
                    logger.info(f"  Created missing directory: {custom_dir}")
                    self.fixes_applied.append(f"Created theme custom_dir: {custom_dir}")

        if "nav" in config:
            self._validate_nav_structure(config["nav"])

    def _validate_nav_structure(self, nav: list[dict | str], prefix: str = "") -> None:
        """Recursively validate navigation structure.

        Args:
            nav: Navigation structure.
            prefix: Current navigation prefix.
        """
        for item in nav:
            if isinstance(item, dict):
                for title, path_or_subnav in item.items():
                    if isinstance(path_or_subnav, str):
                        doc_path = self.docs_path / path_or_subnav
                        if not doc_path.exists():
                            logger.warning(
                                f"  Navigation references missing file: {path_or_subnav}"
                            )
                            self.issues.append({"type": "missing_nav_file", "file": path_or_subnav})
                    elif isinstance(path_or_subnav, list):
                        self._validate_nav_structure(path_or_subnav, f"{prefix}{title}/")

    def validate_internal_links(self) -> None:
        """Validate internal links in markdown files."""
        logger.info("Validating internal links...")

        md_files = list(self.docs_path.rglob("*.md"))

        for md_file in md_files:
            with open(md_file) as f:
                content = f.read()

            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            matches = re.finditer(link_pattern, content)

            for match in matches:
                link_target = match.group(2)

                if link_target.startswith(("http://", "https://", "#")):
                    continue

                if link_target.endswith(".md"):
                    target_path = md_file.parent / link_target
                    if not target_path.exists():
                        rel_path = md_file.relative_to(self.docs_path)
                        logger.warning(f"  Broken link in {rel_path}: {link_target}")
                        self.issues.append(
                            {
                                "type": "broken_link",
                                "file": str(rel_path),
                                "target": link_target,
                            }
                        )

    def generate_report(self) -> None:
        """Generate a summary report of issues found."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("Validation Summary")
        logger.info("=" * 50)

        if not self.issues:
            logger.info("No issues found! Documentation is valid.")
        else:
            logger.info(f"Found {len(self.issues)} issues:")

            by_type: dict[str, list[dict]] = defaultdict(list)
            for issue in self.issues:
                by_type[issue["type"]].append(issue)

            for issue_type, issues in by_type.items():
                logger.info(f"  {issue_type.replace('_', ' ').title()}: {len(issues)}")
                if self.verbose:
                    for issue in issues[:5]:
                        if "module" in issue:
                            logger.info(f"    - {issue['module']}")
                        elif "file" in issue:
                            logger.info(f"    - {issue['file']}")

        if self.fixes_applied:
            logger.info(f"Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied[:10]:
                logger.info(f"  - {fix}")

    def run(self) -> int:
        """Run all validation checks.

        Returns:
            Exit code (0 if no issues or all fixed, 1 otherwise).
        """
        logger.info("Starting documentation validation...")

        self.validate_mkdocs_config()
        self.validate_api_docs()
        self.validate_internal_links()

        self.generate_report()

        # Fix mode succeeds only when every issue was actually fixed;
        # unfixable issues must fail the run either way.
        unresolved = len(self.issues) - len(self.fixes_applied)
        return 0 if unresolved <= 0 else 1


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate and fix documentation issues",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")

    parser.add_argument(
        "--check-only", action="store_true", help="Only check for issues, don't fix"
    )

    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    parser.add_argument(
        "--docs-path", default="docs", help="Path to documentation directory (default: docs)"
    )

    parser.add_argument(
        "--src-path", default="src", help="Path to source code directory (default: src)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = DocValidator(
        docs_path=args.docs_path,
        src_path=args.src_path,
        fix=args.fix and not args.check_only,
        verbose=args.verbose,
    )

    exit_code = validator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
