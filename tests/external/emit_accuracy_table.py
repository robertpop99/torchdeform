#!/usr/bin/env python3
"""Regenerate the VMOD cross-validation accuracy table in this dir's README.

The ``test_vmod_comparison.py`` suite compares each torchdeform source against
its VMOD counterpart over ~1000 random samples and already computes the
per-model error distribution. This script runs the *same* drivers (imported from
the test module, so there is no duplicated sampling/adapter logic) and writes a
qualitative ``median / max`` table into README.md, so the committed numbers come
from a real run instead of being typed from memory.

Unlike ``tests/sources/reference/accuracy_report.py`` (which reads frozen JSON
and needs no toolchain), this one genuinely needs **VMOD installed and a manual
run** -- it is the same gated, network-free-but-dependency-heavy suite. The
numbers are seed- and sample-count-dependent, so the table is rendered
**qualitatively** (one significant figure, ``~`` prefix): it says "these came
from an actual seed-0 run", not "reproducible to machine precision".

Usage::

    RUN_VMOD_TESTS=1 python emit_accuracy_table.py            # print the table
    RUN_VMOD_TESTS=1 python emit_accuracy_table.py --write    # splice into README

Honours the same env knobs as the suite (``VMOD_PATH``, ``VMOD_N_SAMPLES``,
``VMOD_SEED``). ``--write`` replaces the block between
``<!-- VMOD-ACCURACY:START -->`` and ``<!-- VMOD-ACCURACY:END -->``.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
MARK_START = "<!-- VMOD-ACCURACY:START -->"
MARK_END = "<!-- VMOD-ACCURACY:END -->"


def _qual(x: float) -> str:
    """One significant figure, ``~`` prefix, normalised exponent: ``~2e-16``."""
    if not math.isfinite(x) or x <= 0.0:
        return "~0"
    exp = math.floor(math.log10(x))
    mant = round(x / 10 ** exp)
    if mant == 10:            # rounding pushed us to the next decade
        mant, exp = 1, exp + 1
    return f"~{mant}e{exp}"


def _import_suite():
    """Import the VMOD test module, arming its manual-run gate first."""
    os.environ.setdefault("RUN_VMOD_TESTS", "1")
    import sys
    sys.path.insert(0, str(HERE))
    try:
        import test_vmod_comparison as suite
    except Exception as exc:  # pytest.skip (no VMOD) surfaces here outside pytest
        raise SystemExit(
            f"Cannot run the VMOD suite: {exc}\n"
            "Install VMOD (`pip install vmod-geodesy`) or set VMOD_PATH, and "
            "run with RUN_VMOD_TESTS=1."
        )
    return suite


def render() -> str:
    import numpy as np

    suite = _import_suite()
    lines: list[str] = []
    lines.append(f"Relative field error over {suite.N_SAMPLES} random samples "
                 f"per model (seed {suite.BASE_SEED}), as `median → max`. "
                 "Qualitative (one significant figure) -- the exact digits shift "
                 "with the seed and sample count.")
    lines.append("")
    lines.append("| torchdeform | VMOD | median | max | notes |")
    lines.append("|---|---|--------|-------|-------|")
    for ours, vmod, run, note in suite.MODELS:
        errs = run()
        median = float(np.quantile(errs, 0.5))
        mx = float(errs.max())
        lines.append(
            f"| `{ours}` | `{vmod}` | {_qual(median)} | {_qual(mx)} | {note} |"
        )
    lines.append("")
    lines.append("*Regenerate with "
                 "`RUN_VMOD_TESTS=1 python emit_accuracy_table.py --write` "
                 "(needs VMOD installed; numbers are from one seed-0 run).*")
    return "\n".join(lines)


def write_readme(block: str) -> None:
    text = README.read_text()
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(
            f"Markers {MARK_START} / {MARK_END} not found in {README}. "
            "Add them where the VMOD accuracy table should live."
        )
    pre, rest = text.split(MARK_START, 1)
    _, post = rest.split(MARK_END, 1)
    README.write_text(f"{pre}{MARK_START}\n\n{block}\n\n{MARK_END}{post}")
    print(f"Wrote VMOD accuracy table into {README}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="splice the table into README.md between the "
                         "VMOD-ACCURACY markers instead of printing it")
    args = ap.parse_args()

    block = render()
    if args.write:
        write_readme(block)
    else:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
