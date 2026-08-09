#!/usr/bin/env python3
"""Enforce dataclass immutability policy across the DiffAV package.

Policy: every dataclass uses ``frozen=True, slots=True, kw_only=True``.
No exemptions — the datarax v0.1.4+ config bases are frozen, so subclasses
conform directly.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
from dataclasses import is_dataclass


REQUIRED = {
    "frozen": True,
    "slots": True,
    "kw_only": True,
}


def iter_dataclasses(package_name: str) -> list[type]:
    """Discover dataclass types under a package."""
    package = importlib.import_module(package_name)
    classes: list[type] = []

    for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        module = importlib.import_module(module_info.name)
        for _, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == module.__name__ and is_dataclass(obj):
                classes.append(obj)
    return sorted(classes, key=lambda cls: f"{cls.__module__}.{cls.__name__}")


def dataclass_status(cls: type) -> dict[str, bool]:
    """Extract the policy status for a dataclass."""
    params = cls.__dataclass_params__
    return {
        "frozen": bool(params.frozen),
        "slots": bool(getattr(params, "slots", False)),
        "kw_only": bool(getattr(params, "kw_only", False)),
    }


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        default="diffav",
        help="Package to scan (default: diffav)",
    )
    args = parser.parse_args()

    violations: list[str] = []

    classes = iter_dataclasses(args.package)
    print(f"Scanning {len(classes)} dataclasses in package '{args.package}'...")

    for cls in classes:
        fqcn = f"{cls.__module__}.{cls.__name__}"
        status = dataclass_status(cls)
        if all(status[key] == expected for key, expected in REQUIRED.items()):
            continue
        details = ", ".join(f"{key}={status[key]}" for key in ("frozen", "slots", "kw_only"))
        violations.append(f"{fqcn}: {details}")

    if violations:
        print("\nPolicy violations:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("\nDataclass policy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
