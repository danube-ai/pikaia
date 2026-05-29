#!/usr/bin/env python3
"""Verify that the PR version is strictly greater than the base branch version."""

import sys
import tomllib

from packaging.version import Version


def load_version(toml_path: str) -> Version:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    raw = data["project"]["version"]
    return Version(raw)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: check_version.py <current_pyproject.toml> <base_pyproject.toml>")
        sys.exit(2)

    current = load_version(sys.argv[1])
    base = load_version(sys.argv[2])

    if current <= base:
        print(
            f"Version bump required: PR version {current} must be "
            f"strictly greater than base branch version {base}."
        )
        sys.exit(1)

    print(f"Version check passed: {current} > {base}")
