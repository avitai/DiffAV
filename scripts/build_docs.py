#!/usr/bin/env python3
"""
Documentation build script for Simulacrax.

This script handles the complete documentation build process:
1. Generate documentation from source code
2. Build the documentation with MkDocs
3. Optionally serve the documentation for preview

License: MIT
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    description: str = "",
) -> bool:
    """Run a command and handle errors.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory for the command.
        description: Human-readable description of the command.

    Returns:
        True if the command succeeded.
    """
    if description:
        print(f"  {description}")

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Command failed: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return False
    except OSError as e:
        print(f"  Unexpected error: {e}")
        return False


def main() -> None:
    """Main build process."""
    parser = argparse.ArgumentParser(description="Build Simulacrax documentation")
    parser.add_argument(
        "--serve", action="store_true", help="Serve documentation after building (for preview)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for serving documentation (default: 8000)"
    )
    parser.add_argument(
        "--skip-generation", action="store_true", help="Skip documentation generation step"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("Starting Simulacrax documentation build process...")

    # Step 1: Generate documentation from source code
    if not args.skip_generation:
        print()
        print("Step 1: Generating documentation from source code...")

        generate_script = project_root / "scripts" / "generate_docs.py"
        if not generate_script.exists():
            print(f"  Documentation generation script not found: {generate_script}")
            sys.exit(1)

        if not run_command(
            [sys.executable, str(generate_script)],
            cwd=project_root,
            description="Running documentation generation...",
        ):
            sys.exit(1)
    else:
        print()
        print("Skipping documentation generation (--skip-generation flag)")

    # Step 2: Build documentation with MkDocs
    print()
    print("Step 2: Building documentation with MkDocs...")

    if not run_command(
        ["uv", "run", "mkdocs", "build"],
        cwd=project_root,
        description="Building documentation...",
    ):
        sys.exit(1)

    print("Documentation build completed successfully!")

    # Step 3: Optionally serve documentation
    if args.serve:
        print()
        print(f"Step 3: Serving documentation on port {args.port}...")
        print(f"Documentation will be available at: http://localhost:{args.port}")
        print("Press Ctrl+C to stop the server")

        try:
            subprocess.run(
                ["uv", "run", "mkdocs", "serve", "--dev-addr", f"0.0.0.0:{args.port}"],
                cwd=project_root,
            )
        except KeyboardInterrupt:
            print()
            print("Documentation server stopped")

    print()
    print("Documentation build process completed!")


if __name__ == "__main__":
    main()
