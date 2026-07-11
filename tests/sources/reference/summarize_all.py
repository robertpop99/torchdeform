#!/usr/bin/env python3
"""Run every ``gen_*.py`` reference generator with ``--summary``.

Each generator in this directory can dump a human-readable summary of its
committed golden JSON via ``--summary`` (no MATLAB / heavy deps needed). This
just runs all of them in one go so you don't have to invoke each by hand::

    python tests/sources/reference/summarize_all.py

Any extra arguments are forwarded to each generator, e.g. to pass a different
flag they all understand. Exits non-zero if any generator fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    extra = sys.argv[1:]
    generators = sorted(HERE.glob("gen_*.py"))
    if not generators:
        print("No gen_*.py generators found.", file=sys.stderr)
        return 1

    failures: list[str] = []
    for gen in generators:
        banner = f" {gen.name} "
        print("\n" + banner.center(72, "="))
        result = subprocess.run(
            [sys.executable, str(gen), "--summary", *extra],
            cwd=HERE,
        )
        if result.returncode != 0:
            failures.append(gen.name)
            print(f"[!] {gen.name} exited with code {result.returncode}",
                  file=sys.stderr)

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} generator(s) failed: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    print(f"All {len(generators)} generators summarized successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
