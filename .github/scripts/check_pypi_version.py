#!/usr/bin/env python3
"""Refuse to publish if the package version already exists on the target index.

PyPI uploads are irreversible; this check fails fast with a clear message
when a developer forgets to bump the version before merging.
"""

from __future__ import annotations

import json
import sys
import tomllib
import urllib.error
import urllib.request

INDEX_URLS = {
    "pypi": "https://pypi.org/pypi/{name}/json",
    "testpypi": "https://test.pypi.org/pypi/{name}/json",
}


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: check_pypi_version.py <pypi|testpypi> <pyproject.toml>",
            file=sys.stderr,
        )
        return 2

    index, toml_path = sys.argv[1], sys.argv[2]
    if index not in INDEX_URLS:
        print(f"Unknown index: {index!r} (expected pypi or testpypi)", file=sys.stderr)
        return 2

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    name = data["project"]["name"]
    version = data["project"]["version"]

    url = INDEX_URLS[index].format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"OK: {name} not yet on {index}; {version} is safe to publish.")
            return 0
        print(f"Failed to query {index} ({url}): HTTP {e.code}", file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"Failed to reach {index} ({url}): {e.reason}", file=sys.stderr)
        return 2

    releases = payload.get("releases", {})
    if version in releases:
        print(
            f"REFUSING TO PUBLISH: {name} {version} is already on {index}.\n"
            f"Bump 'project.version' in pyproject.toml before merging.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {name} {version} is not yet on {index}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
