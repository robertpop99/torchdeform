#!/usr/bin/env python3
"""Measure each source model's accuracy against its committed golden fixture.

This runs every ``torchdeform`` source over the frozen reference values in
``../data/`` and reports the achieved error -- the same comparison the
``test_*_source.py`` suites assert, but surfaced as numbers instead of a
pass/fail. It needs **no toolchain** (no MATLAB / Fortran / gfortran): it only
reads the committed JSON, exactly like the tests.

Two families are reported:

* **Forward** -- displacement (metres) against the reference kernels
  (Okada's DC3D, Nikkhoo's pCDM/CDM/pECM, Fialko's penny), across both the
  hand-picked published tables and the random-parameter *volume* fixtures, at
  ``nu = 0.25`` and (where a ``_nu0.32`` sibling exists) ``nu = 0.32``.
* **Gradients** -- OkadaSource only. It is the one source with a hand-written
  backward (``analytic_grad``), so its derivatives have their own external
  ground truth in ``dc3d_golden.json``. Every other source is plain autograd of
  the forward, pinned by ``torch.autograd.gradcheck`` in its test module (no
  external gradient fixture to measure) -- reported here as "gradcheck".

Usage::

    python accuracy_report.py            # print the Markdown tables
    python accuracy_report.py --write    # splice them into README.md markers

The ``--write`` mode replaces the block between
``<!-- ACCURACY:START -->`` and ``<!-- ACCURACY:END -->`` in this directory's
``README.md`` so the committed numbers can never silently drift from the code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from torchdeform.sources import (
    CDMSource,
    OkadaSource,
    OkadaSourceSimple,
    PCDMSource,
    PECMSource,
    PennySource,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
README = HERE / "README.md"
DTYPE = torch.float64

MARK_START = "<!-- ACCURACY:START -->"
MARK_END = "<!-- ACCURACY:END -->"


def t(x):
    return torch.as_tensor(x, dtype=DTYPE)


def _errs(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """(max abs error, error relative to the field's own peak)."""
    diff = (got - want).abs()
    max_abs = float(diff.max())
    scale = float(want.abs().max())
    rel = max_abs / scale if scale > 0 else float("nan")
    return max_abs, rel


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


# --------------------------------------------------------------------------- #
# Forward accuracy
# --------------------------------------------------------------------------- #
def _fwd_okada(fname: str) -> tuple[float, float, int]:
    d = _load(fname)
    out = OkadaSource(poisson_ratio=d["poisson_ratio"])(
        x_obs=t(d["x_obs"]), y_obs=t(d["y_obs"]), z_obs=t(d["z_obs"]),
        source_x=t(d["source_x"]), source_y=t(d["source_y"]),
        dip=t(d["dip"]), strike=t(d["strike"]),
        centroid_depth=t(d["centroid_depth"]),
        length=t(d["length"]), width=t(d["width"]),
        disl1=t(d["disl1"]), disl2=t(d["disl2"]), disl3=t(d["disl3"]),
    )
    got = torch.stack([out.e, out.n, out.u], dim=-1)
    ma, rel = _errs(got, t(d["u_enu"]))
    return ma, rel, d["n_faults"] * d["n_points"]


def _fwd_okada_simple(fname: str) -> tuple[float, float, int]:
    """Surface-only fast path at the fixture's z = 0 points (non-zero strike)."""
    d = _load(fname)
    z = t(d["z_obs"])
    mask = z == 0.0
    b, n = z.shape

    def pp(name):
        return t(d[name])[:, None].expand(b, n)[mask]

    out = OkadaSourceSimple(poisson_ratio=d["poisson_ratio"])(
        x_obs=t(d["x_obs"])[mask][:, None], y_obs=t(d["y_obs"])[mask][:, None],
        source_x=pp("source_x"), source_y=pp("source_y"),
        dip=pp("dip"), strike=pp("strike"),
        centroid_depth=pp("centroid_depth"),
        length=pp("length"), width=pp("width"),
        disl1=pp("disl1"), disl2=pp("disl2"), disl3=pp("disl3"),
    )
    got = torch.stack([out.e, out.n, out.u], dim=-1).reshape(-1, 3)
    ma, rel = _errs(got, t(d["u_enu"])[mask])
    return ma, rel, int(mask.sum())


def _fwd_pcdm_volume(fname: str) -> tuple[float, float, int]:
    d = _load(fname)
    out = PCDMSource(poisson_ratio=d["poisson_ratio"])(
        t(d["x_obs"]), t(d["y_obs"]),
        source_x=t(d["source_x"]), source_y=t(d["source_y"]), depth=t(d["depth"]),
        omega_x=t(d["omega_x"]), omega_y=t(d["omega_y"]), omega_z=t(d["omega_z"]),
        dv_x=t(d["dv_x"]), dv_y=t(d["dv_y"]), dv_z=t(d["dv_z"]),
    )
    got = torch.stack([out.e, out.n, out.u], dim=-1)
    ma, rel = _errs(got, t(d["u_enu"]))
    return ma, rel, d["n_sources"] * d["n_points"]


def _fwd_cdm_volume(fname: str) -> tuple[float, float, int]:
    d = _load(fname)
    out = CDMSource(poisson_ratio=d["poisson_ratio"])(
        t(d["x_obs"]), t(d["y_obs"]),
        source_x=t(d["source_x"]), source_y=t(d["source_y"]), depth=t(d["depth"]),
        omega_x=t(d["omega_x"]), omega_y=t(d["omega_y"]), omega_z=t(d["omega_z"]),
        a_x=t(d["a_x"]), a_y=t(d["a_y"]), a_z=t(d["a_z"]), opening=t(d["opening"]),
    )
    got = torch.stack([out.e, out.n, out.u], dim=-1)
    ma, rel = _errs(got, t(d["u_enu"]))
    return ma, rel, d["n_sources"] * d["n_points"]


def _fwd_pecm_volume(fname: str) -> tuple[float, float, int]:
    d = _load(fname)
    out = PECMSource(
        poisson_ratio=d["poisson_ratio"], shear_modulus=d["shear_modulus"],
    )(
        t(d["x_obs"]), t(d["y_obs"]),
        source_x=t(d["source_x"]), source_y=t(d["source_y"]), depth=t(d["depth"]),
        omega_x=t(d["omega_x"]), omega_y=t(d["omega_y"]), omega_z=t(d["omega_z"]),
        a_x=t(d["a_x"]), a_y=t(d["a_y"]), a_z=t(d["a_z"]), pressure=t(d["pressure"]),
    )
    got = torch.stack([out.e, out.n, out.u], dim=-1)
    ma, rel = _errs(got, t(d["u_enu"]))
    return ma, rel, d["n_sources"] * d["n_points"]


def _fwd_penny_volume(fname: str) -> tuple[float, float, int]:
    d = _load(fname)
    x = t(d["x_obs"])
    y = torch.zeros_like(x)
    zeros = torch.zeros(x.shape[0], dtype=DTYPE)
    out = PennySource(
        poisson_ratio=d["poisson_ratio"], shear_modulus=d["shear_modulus"],
        nis=d["nis"],
    )(x, y, zeros, zeros, depth=t(d["depth"]), radius=t(d["radius"]),
      pressure=t(d["pressure"]))
    got = torch.stack([out.e, out.u], dim=-1)          # Ur (== +East), Uz
    want = torch.stack([t(d["ur"]), t(d["uz"])], dim=-1)
    ma, rel = _errs(got, want)
    return ma, rel, d["n_sources"] * d["n_points"]


# Ordered (source, reference label, callable) forward rows. Volume fixtures with
# a _nu0.32 sibling contribute two rows.
def _forward_rows() -> list[tuple[str, str, float, float, int]]:
    rows: list[tuple[str, str, float, float, int]] = []

    def add(source, ref, fn, *args):
        ma, rel, n = fn(*args)
        rows.append((source, ref, ma, rel, n))

    add("OkadaSource", "Okada (1992) DC3D volume, nu=0.25",
        _fwd_okada, "dc3d_golden.json")
    add("OkadaSource", "Okada (1992) DC3D volume, nu=0.32",
        _fwd_okada, "dc3d_golden_nu0.32.json")
    add("OkadaSourceSimple", "DC3D surface (z=0), nu=0.25",
        _fwd_okada_simple, "dc3d_golden.json")
    add("OkadaSourceSimple", "DC3D surface (z=0), nu=0.32",
        _fwd_okada_simple, "dc3d_golden_nu0.32.json")

    add("PCDMSource", "Nikkhoo (2017) pCDM.m volume, nu=0.25",
        _fwd_pcdm_volume, "pcdm_volume_golden.json")
    add("PCDMSource", "Nikkhoo (2017) pCDM.m volume, nu=0.32",
        _fwd_pcdm_volume, "pcdm_volume_golden_nu0.32.json")

    add("CDMSource", "Nikkhoo (2017) CDM.m volume, nu=0.25",
        _fwd_cdm_volume, "cdm_volume_golden.json")
    add("CDMSource", "Nikkhoo (2017) CDM.m volume, nu=0.32",
        _fwd_cdm_volume, "cdm_volume_golden_nu0.32.json")

    add("PECMSource", "Nikkhoo (2017) pECM.m volume, nu=0.25",
        _fwd_pecm_volume, "pecm_volume_golden.json")
    add("PECMSource", "Nikkhoo (2017) pECM.m volume, nu=0.32",
        _fwd_pecm_volume, "pecm_volume_golden_nu0.32.json")

    add("PennySource", "Fialko (2001) penny volume, nu=0.25",
        _fwd_penny_volume, "penny_volume_golden.json")

    return rows


# --------------------------------------------------------------------------- #
# Okada gradient accuracy (the one source with an external gradient fixture)
# --------------------------------------------------------------------------- #
def _dc3d_map_jacobian(d: dict) -> torch.Tensor:
    """Fault-frame derivatives -> map-frame Jacobian J[...,i,j] = d ENU_i/d coord_j.
    Column-major 9-vector -> J_fault^T; rotate both legs by strike C (self-inverse)."""
    b, n = len(d["strike"]), d["n_points"]
    j_fault = t(d["derivatives_fault_frame"]).reshape(b, n, 3, 3).transpose(-1, -2)
    s, c = torch.sin(t(d["strike"])), torch.cos(t(d["strike"]))
    cmat = torch.zeros(b, n, 3, 3, dtype=DTYPE)
    cmat[..., 0, 0] = s[:, None]; cmat[..., 0, 1] = c[:, None]
    cmat[..., 1, 0] = c[:, None]; cmat[..., 1, 1] = -s[:, None]
    cmat[..., 2, 2] = 1.0
    return cmat @ j_fault @ cmat


def _dc3d_model_run(d: dict, grad_params: set[str]):
    leaves: dict[str, torch.Tensor] = {}

    def leaf(name):
        v = t(d[name])
        if name in grad_params:
            v = v.clone().requires_grad_(True)
            leaves[name] = v
        return v

    out = OkadaSource(poisson_ratio=d["poisson_ratio"], analytic_grad=True)(
        x_obs=leaf("x_obs"), y_obs=leaf("y_obs"), z_obs=leaf("z_obs"),
        source_x=leaf("source_x"), source_y=leaf("source_y"),
        dip=leaf("dip"), strike=t(d["strike"]),
        centroid_depth=leaf("centroid_depth"),
        length=leaf("length"), width=leaf("width"),
        disl1=leaf("disl1"), disl2=leaf("disl2"), disl3=leaf("disl3"),
    )
    return out, leaves


_N_GRAD_POINTS = 8


def _per_point_source_grad(out, param, b, n, k=_N_GRAD_POINTS):
    k = min(k, n)
    g = torch.zeros(b, k, 3, dtype=DTYPE)
    for i, comp in enumerate((out.e, out.n, out.u)):
        for j in range(k):
            sel = torch.zeros(b, n, dtype=DTYPE)
            sel[:, j] = 1.0
            g[:, j, i] = torch.autograd.grad(comp, param, grad_outputs=sel,
                                              retain_graph=True)[0]
    return g


def _grad_strain_err(fname: str) -> tuple[float, float]:
    """Closed-form backward (analytic_grad) vs DC3D's exact obs-coord strain."""
    d = _load(fname)
    x = t(d["x_obs"]).requires_grad_(True)
    y = t(d["y_obs"]).requires_grad_(True)
    z = t(d["z_obs"]).requires_grad_(True)
    out = OkadaSource(poisson_ratio=d["poisson_ratio"], analytic_grad=True)(
        x_obs=x, y_obs=y, z_obs=z,
        source_x=t(d["source_x"]), source_y=t(d["source_y"]),
        dip=t(d["dip"]), strike=t(d["strike"]),
        centroid_depth=t(d["centroid_depth"]),
        length=t(d["length"]), width=t(d["width"]),
        disl1=t(d["disl1"]), disl2=t(d["disl2"]), disl3=t(d["disl3"]),
    )
    j_map = torch.zeros(*x.shape, 3, 3, dtype=DTYPE)
    for i, comp in enumerate((out.e, out.n, out.u)):
        for j, g in enumerate(torch.autograd.grad(comp.sum(), (x, y, z),
                                                  retain_graph=True)):
            j_map[..., i, j] = g
    return _errs(j_map, _dc3d_map_jacobian(d))


def _grad_source_param_err(names: tuple[str, ...]) -> tuple[float, float]:
    """Max error over the named source-parameter gradients (canonical nu=0.25)."""
    d = _load("dc3d_golden.json")
    b, n = d["n_faults"], d["n_points"]
    out, leaves = _dc3d_model_run(d, set(names))
    worst = (0.0, 0.0)
    for name in names:
        got = _per_point_source_grad(out, leaves[name], b, n)
        want = t(d["param_gradients"][name])[:, :got.shape[1]]
        ma, rel = _errs(got, want)
        if ma > worst[0]:
            worst = (ma, rel)
    return worst


def _grad_source_position_err() -> tuple[float, float]:
    """d u/d source_{x,y} = -horizontal strain (exact identity)."""
    d = _load("dc3d_golden.json")
    b, n = d["n_faults"], d["n_points"]
    out, leaves = _dc3d_model_run(d, {"source_x", "source_y"})
    j_map_ref = _dc3d_map_jacobian(d)
    worst = (0.0, 0.0)
    for name, col_idx in (("source_x", 0), ("source_y", 1)):
        got = _per_point_source_grad(out, leaves[name], b, n)
        want = -j_map_ref[..., :got.shape[1], :, col_idx]
        ma, rel = _errs(got, want)
        if ma > worst[0]:
            worst = (ma, rel)
    return worst


def _gradient_rows() -> list[tuple[str, str, float, float]]:
    rows: list[tuple[str, str, float, float]] = []

    ma, rel = _grad_strain_err("dc3d_golden.json")
    rows.append(("d(ENU)/d(x,y,z) strain", "DC3D exact derivs, nu=0.25", ma, rel))
    ma, rel = _grad_strain_err("dc3d_golden_nu0.32.json")
    rows.append(("d(ENU)/d(x,y,z) strain", "DC3D exact derivs, nu=0.32", ma, rel))

    ma, rel = _grad_source_param_err(("disl1", "disl2", "disl3"))
    rows.append(("d(ENU)/d(disl1,2,3)", "DC3D unit-slip (exact)", ma, rel))

    ma, rel = _grad_source_position_err()
    rows.append(("d(ENU)/d(source_x,y)", "-horizontal strain (exact)", ma, rel))

    ma, rel = _grad_source_param_err(("length", "width", "centroid_depth"))
    rows.append(("d(ENU)/d(length,width,depth)", "DC3D Richardson FD", ma, rel))

    ma, rel = _grad_source_param_err(("dip",))
    rows.append(("d(ENU)/d(dip)", "DC3D Richardson FD", ma, rel))

    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{x:.1e}"


def render() -> str:
    fwd = _forward_rows()
    grad = _gradient_rows()

    lines: list[str] = []
    lines.append("### Forward displacement")
    lines.append("")
    lines.append("Each row is a torchdeform port vs the **original authors' "
                 "code** over a random-parameter *volume* (16-24 buried sources "
                 "at 24-32 points each; the DC3D fixture also spans z < 0). Max "
                 "error and error relative to the field's own peak amplitude. "
                 "Both sides are float64.")
    lines.append("")
    lines.append("| Source | Reference | Points | Max abs error (m) | Rel. error |")
    lines.append("|---|---|--:|--:|--:|")
    for source, ref, ma, rel, n in fwd:
        lines.append(f"| `{source}` | {ref} | {n} | {_fmt(ma)} | {_fmt(rel)} |")
    lines.append("")
    lines.append("**Point checks not tabulated** (both are subsumed by the "
                 "random-volume rows above, and stay in the test suites):")
    lines.append("")
    lines.append("- **Okada (1985) Table 2** — the suite's only genuinely "
                 "*published* numeric ground truth (Cases 2-4, three surface "
                 "points, 4 significant figures). `OkadaSource` reproduces it to "
                 "that 4-sig-fig precision; because the reference itself is "
                 "rounded, its error is not comparable to the machine-precision "
                 "rows above, so it is credited here rather than tabled. Checked "
                 "in `test_okada_source.py`.")
    lines.append("- **Nikkhoo (2017) / Fialko (2001) hand-picked orientations** "
                 "(`nikkhoo_golden.json`, `fialko_golden.json`) — a few fixed "
                 "orientations run through the *same* MATLAB kernels as the "
                 "volume fixtures (not published tables — generated by "
                 "`gen_nikkhoo.m` / `gen_fialko.m`), reproduced to machine "
                 "precision. Checked in the respective `test_*_source.py`.")
    lines.append("")
    lines.append("### Gradients")
    lines.append("")
    lines.append("`OkadaSource` is the only source with external gradient "
                 "ground truth (its hand-written `analytic_grad` backward, in "
                 "`dc3d_golden.json`). Every other source is plain autograd of "
                 "the forward, pinned by `torch.autograd.gradcheck` in its test "
                 "module -- no external fixture to measure here.")
    lines.append("")
    lines.append("| Gradient | Reference | Max abs error (/m) | Rel. error |")
    lines.append("|---|---|--:|--:|")
    for quant, ref, ma, rel in grad:
        lines.append(f"| `{quant}` | {ref} | {_fmt(ma)} | {_fmt(rel)} |")
    lines.append("")
    lines.append("Other sources' gradients: `MogiSource`, `PCDMSource`, "
                 "`CDMSource`, `PECMSource`, `PennySource` -- **gradcheck** "
                 "(autograd vs. finite differences) in the respective "
                 "`test_*_source.py`.")
    lines.append("")
    lines.append("*Regenerate this section with "
                 "`python accuracy_report.py --write` (reads the committed "
                 "fixtures; no MATLAB/Fortran needed).*")
    return "\n".join(lines)


def write_readme(block: str) -> None:
    text = README.read_text()
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(
            f"Markers {MARK_START} / {MARK_END} not found in {README}. "
            "Add them where the accuracy table should live."
        )
    pre, rest = text.split(MARK_START, 1)
    _, post = rest.split(MARK_END, 1)
    new = f"{pre}{MARK_START}\n\n{block}\n\n{MARK_END}{post}"
    README.write_text(new)
    print(f"Wrote accuracy table into {README}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="splice the tables into README.md between the "
                         "ACCURACY markers instead of printing them")
    args = ap.parse_args()

    block = render()
    if args.write:
        write_readme(block)
    else:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
