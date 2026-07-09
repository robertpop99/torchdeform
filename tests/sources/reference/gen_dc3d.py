#!/usr/bin/env python3
"""Generate frozen ground-truth for OkadaSource at depth from Okada's own DC3D.

This is the Okada analogue of ``gen_nikkhoo.m`` / ``gen_fialko.m``: it runs the
*original author's* finite-fault code -- Okada's ``DC3D`` (the DC3D.f90 F90
conversion in ``scaffolding/``) -- over many random buried faults observed at
depth, and freezes the result as ``../data/dc3d_golden.json``. The committed
JSON is all the tests need; regenerating requires a Fortran compiler and a local
copy of ``DC3D.f90`` (not redistributed here -- see ``README.md``).

Why this exists on top of Okada (1985) Table 2:
  * Table 2 is three surface points. This covers the *volume* (observations at
    z < 0, at depth) over a wide random parameter space -- the part of
    ``OkadaSource`` that previously had no external ground truth.
  * Cases use a random non-zero strike, so the full map->fault assembly
    (rotation, centroid -> AL/AW placement) is exercised, not just the kernel.
  * DC3D also returns the nine spatial derivatives; those are frozen too so a
    later pass can validate the ``analytic_grad`` backward against the exact
    Okada strain (see the ``derivatives_fault_frame`` field).

Convention bridge (must match ``OkadaSource``)
----------------------------------------------
``OkadaSource`` maps map-frame (East, North) observations to Okada's fault-local
frame with the strike rotation

    x_fault =  dE*sin(strike) + dN*cos(strike)      (along strike)
    y_fault =  dE*cos(strike) - dN*sin(strike)      (across strike, down-dip)

where ``dE = x_obs - source_x``, ``dN = y_obs - source_y``. That 2x2 matrix is
its own inverse, so the fault-frame displacement rotates back to ENU the same
way:

    E = ux*sin(strike) + uy*cos(strike)
    N = ux*cos(strike) - uy*sin(strike)
    U = uz

Centroid placement: ``AL1,AL2 = -/+L/2``; ``AW1,AW2 = -/+W/2``;
``depth = centroid_depth``. Material: ``alpha = 1/(2(1-nu)) = 2/3`` (nu = 0.25).

Regenerate::

    python gen_dc3d.py                 # uses ../../../scaffolding/DC3D.f90
    TORCHDEFORM_DC3D_F90=/path/DC3D.f90 python gen_dc3d.py

Inspect the committed JSON without a Fortran toolchain::

    python gen_dc3d.py --summary       # header, array shapes, per-fault inputs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DEFAULT_DC3D = HERE.parent.parent.parent / "scaffolding" / "DC3D.f90"

ALPHA = 2.0 / 3.0          # nu = 0.25 -> (lambda+mu)/(lambda+2mu) = 2/3
SEED = 20240708
N_FAULTS = 24              # batch dimension B
N_POINTS = 32              # observation points per fault N

# A pure-compute Fortran driver: read fault-frame evaluations one per line from
# stdin, call DC3D, echo iret + displacement + the nine derivatives to stdout.
# Keeping randomness/geometry in Python (below) means Fortran is only ever the
# reference kernel -- no cross-language RNG to reconcile.
DRIVER_F90 = r"""
program dc3d_driver
  implicit none
  real*8 :: alpha, x, y, z, depth, dip
  real*8 :: al1, al2, aw1, aw2, disl1, disl2, disl3
  real*8 :: ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz
  integer :: iret, ios
  do
    read(*, *, iostat=ios) alpha, x, y, z, depth, dip, &
                           al1, al2, aw1, aw2, disl1, disl2, disl3
    if (ios /= 0) exit
    call DC3D(alpha, x, y, z, depth, dip, al1, al2, aw1, aw2, &
              disl1, disl2, disl3, &
              ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz, iret)
    write(*, '(I3, 12(1X, ES25.17))') iret, ux, uy, uz, &
        uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz
  end do
end program dc3d_driver
"""


def find_dc3d() -> Path:
    p = Path(os.environ.get("TORCHDEFORM_DC3D_F90", DEFAULT_DC3D))
    if not p.is_file():
        sys.exit(
            f"DC3D.f90 not found at {p}.\n"
            "This file is not redistributed (see reference/README.md). Point "
            "TORCHDEFORM_DC3D_F90 at a local copy, or place it in scaffolding/."
        )
    return p


def build_driver(dc3d: Path, workdir: Path) -> Path:
    driver_src = workdir / "dc3d_driver.f90"
    driver_src.write_text(DRIVER_F90)
    binary = workdir / "dc3d_driver"
    cmd = [
        "gfortran", "-O2", "-ffree-line-length-none",
        str(dc3d), str(driver_src), "-o", str(binary),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("gfortran not found on PATH; install it to regenerate.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"gfortran failed:\n{e.stderr}")
    return binary


def run_dc3d(binary: Path, rows: np.ndarray) -> np.ndarray:
    """rows: [K,13] fault-frame inputs -> out: [K,13] (iret + 3 disp + 9 deriv)."""
    lines = "\n".join(
        " ".join(f"{v:.17e}" for v in row) for row in rows
    ) + "\n"
    proc = subprocess.run(
        [str(binary)], input=lines, capture_output=True, text=True, check=True
    )
    out = np.array(
        [[float(tok) for tok in ln.split()] for ln in proc.stdout.split("\n") if ln.strip()]
    )
    assert out.shape == (rows.shape[0], 13), (out.shape, rows.shape)
    return out


def sample_fault(rng: np.random.Generator) -> dict:
    """One physically-sane, fully buried fault with a random non-zero strike."""
    length = rng.uniform(2e3, 30e3)
    width = rng.uniform(1e3, 15e3)
    # Spread dip across the whole range including a few near-vertical faults,
    # which stress OkadaSource's 1/cos(dip)**2 general-dip terms.
    dip_deg = float(rng.choice([
        rng.uniform(5.0, 85.0),
        rng.uniform(5.0, 85.0),
        rng.uniform(85.0, 89.0),
    ]))
    strike = rng.uniform(0.2, 2 * math.pi - 0.2)   # keep it clearly non-zero
    min_depth = 0.5 * width * math.sin(math.radians(dip_deg)) + 1e3
    centroid_depth = min_depth + rng.uniform(0.0, 20e3)
    return dict(
        source_x=float(rng.uniform(-5e3, 5e3)),
        source_y=float(rng.uniform(-5e3, 5e3)),
        strike=float(strike),
        dip=math.radians(dip_deg),
        centroid_depth=float(centroid_depth),
        length=float(length),
        width=float(width),
        disl1=float(rng.uniform(-2.0, 2.0)),
        disl2=float(rng.uniform(-2.0, 2.0)),
        disl3=float(rng.uniform(-0.5, 0.5)),
    )


def sample_obs(rng: np.random.Generator, f: dict) -> tuple[float, float, float]:
    """One observation point in map frame; ~1/4 at the surface, rest at depth."""
    reach = 3.0 * max(f["length"], f["width"], f["centroid_depth"])
    x = f["source_x"] + rng.uniform(-reach, reach)
    y = f["source_y"] + rng.uniform(-reach, reach)
    if rng.random() < 0.25:
        z = 0.0
    else:
        # z in (-1.3*centroid_depth, 0): straddles the fault in depth.
        z = -rng.uniform(0.0, 1.3 * f["centroid_depth"])
    return float(x), float(y), float(z)


def to_fault_frame(
    f: dict, x: float, y: float, z: float, *,
    length: float | None = None, width: float | None = None,
    depth: float | None = None, dip: float | None = None,
    disl: tuple[float, float, float] | None = None,
) -> list[float]:
    """Map-frame observation + (optionally perturbed) fault -> DC3D inputs.

    The observation's fault-frame position depends only on strike and source
    location, so the L/W/depth/dip/disl overrides (used to build finite-diff and
    unit-slip references) leave xf, yf, z untouched -- they only move the fault
    edges / dip / slip fed to DC3D.
    """
    length = f["length"] if length is None else length
    width = f["width"] if width is None else width
    depth = f["centroid_depth"] if depth is None else depth
    dip = f["dip"] if dip is None else dip
    d1, d2, d3 = (f["disl1"], f["disl2"], f["disl3"]) if disl is None else disl

    s, c = math.sin(f["strike"]), math.cos(f["strike"])
    de, dn = x - f["source_x"], y - f["source_y"]
    xf = de * s + dn * c
    yf = de * c - dn * s
    return [
        ALPHA, xf, yf, z, depth, math.degrees(dip),
        -0.5 * length, 0.5 * length, -0.5 * width, 0.5 * width, d1, d2, d3,
    ]


def rotate_to_enu(strike: float, ux: float, uy: float, uz: float):
    s, c = math.sin(strike), math.cos(strike)
    e = ux * s + uy * c
    n = ux * c - uy * s
    return e, n, uz


def rotate_enu_array(strike: float, uxyz: np.ndarray) -> np.ndarray:
    """Rotate fault-frame displacement columns [..., 3] to ENU."""
    s, c = math.sin(strike), math.cos(strike)
    ux, uy, uz = uxyz[..., 0], uxyz[..., 1], uxyz[..., 2]
    return np.stack([ux * s + uy * c, ux * c - uy * s, uz], axis=-1)


# Central-difference steps for the source-parameter gradient references. Absolute
# for dip (radians); relative-to-value for the length-scale parameters. Tuned so
# the O(h^2) truncation and O(eps/h) round-off both stay well below the ~1e-6
# tolerance the finite-difference tests use.
H_DIP = 1e-5                # radians
H_REL = 1e-4               # fraction of length / width / centroid_depth

# Gradient-reference evaluation variants, in the fixed row order the driver sees.
# "base" is the nominal fault; slip{1,2,3} are unit-slip responses (exact
# d u / d disl_k); the +/- pairs feed central differences for dip/L/W/depth.
_GRAD_VARIANTS = (
    "base", "slip1", "slip2", "slip3",
    "dip_p", "dip_m", "len_p", "len_m",
    "wid_p", "wid_m", "dep_p", "dep_m",
)


def _variant_rows(f: dict, pts: list) -> np.ndarray:
    """Build [n_variants * N, 13] DC3D inputs for one fault (see _GRAD_VARIANTS)."""
    hL, hW, hD = H_REL * f["length"], H_REL * f["width"], H_REL * f["centroid_depth"]
    overrides = {
        "base": {},
        "slip1": dict(disl=(1.0, 0.0, 0.0)),
        "slip2": dict(disl=(0.0, 1.0, 0.0)),
        "slip3": dict(disl=(0.0, 0.0, 1.0)),
        "dip_p": dict(dip=f["dip"] + H_DIP), "dip_m": dict(dip=f["dip"] - H_DIP),
        "len_p": dict(length=f["length"] + hL), "len_m": dict(length=f["length"] - hL),
        "wid_p": dict(width=f["width"] + hW), "wid_m": dict(width=f["width"] - hW),
        "dep_p": dict(depth=f["centroid_depth"] + hD),
        "dep_m": dict(depth=f["centroid_depth"] - hD),
    }
    blocks = [
        [to_fault_frame(f, *p, **overrides[name]) for p in pts]
        for name in _GRAD_VARIANTS
    ]
    return np.array(blocks).reshape(len(_GRAD_VARIANTS) * len(pts), 13)


def main() -> None:
    dc3d = find_dc3d()
    rng = np.random.default_rng(SEED)

    faults = [sample_fault(rng) for _ in range(N_FAULTS)]

    # Per fault, sample N_POINTS observations, resampling any that land on a
    # DC3D singularity (iret != 0) so the golden set is all well-defined points.
    with tempfile.TemporaryDirectory() as tmp:
        binary = build_driver(dc3d, Path(tmp))

        x_obs = np.zeros((N_FAULTS, N_POINTS))
        y_obs = np.zeros((N_FAULTS, N_POINTS))
        z_obs = np.zeros((N_FAULTS, N_POINTS))
        enu = np.zeros((N_FAULTS, N_POINTS, 3))
        deriv = np.zeros((N_FAULTS, N_POINTS, 9))
        # ENU source-parameter gradient references [B, N, 3] per parameter.
        grad = {k: np.zeros((N_FAULTS, N_POINTS, 3))
                for k in ("disl1", "disl2", "disl3",
                          "dip", "length", "width", "centroid_depth")}
        nv = len(_GRAD_VARIANTS)

        for bi, f in enumerate(faults):
            # Sample all N points; evaluate the nominal fault *and* all gradient
            # variants (unit slips + dip/L/W/depth +/- steps) in one DC3D call.
            # Resample any point whose *nominal or any variant* hits a
            # singularity, so every finite-difference reference is well-defined.
            pts = [sample_obs(rng, f) for _ in range(N_POINTS)]
            for _ in range(1000):
                rows = _variant_rows(f, pts)
                out = run_dc3d(binary, rows).reshape(nv, N_POINTS, 13)
                bad = np.nonzero(np.any(out[..., 0].round().astype(int) != 0, axis=0))[0]
                if bad.size == 0:
                    break
                for j in bad:
                    pts[j] = sample_obs(rng, f)
            else:
                raise RuntimeError("could not clear all singular obs points")

            v = {name: out[i] for i, name in enumerate(_GRAD_VARIANTS)}  # [N,13] each
            enu_of = lambda name: rotate_enu_array(f["strike"], v[name][:, 1:4])

            for pj, (x, y, z) in enumerate(pts):
                x_obs[bi, pj], y_obs[bi, pj], z_obs[bi, pj] = x, y, z
            enu[bi] = enu_of("base")
            deriv[bi] = v["base"][:, 4:13]

            # Exact: displacement is linear in slip, so d u / d disl_k is the
            # unit-slip response G_k directly.
            grad["disl1"][bi] = enu_of("slip1")
            grad["disl2"][bi] = enu_of("slip2")
            grad["disl3"][bi] = enu_of("slip3")

            # Central differences for the parameters with no closed form here.
            hL, hW = H_REL * f["length"], H_REL * f["width"]
            hD = H_REL * f["centroid_depth"]
            grad["dip"][bi] = (enu_of("dip_p") - enu_of("dip_m")) / (2 * H_DIP)
            grad["length"][bi] = (enu_of("len_p") - enu_of("len_m")) / (2 * hL)
            grad["width"][bi] = (enu_of("wid_p") - enu_of("wid_m")) / (2 * hW)
            grad["centroid_depth"][bi] = (enu_of("dep_p") - enu_of("dep_m")) / (2 * hD)

    payload = {
        "_comment": (
            "Ground truth for OkadaSource at depth, from Okada's DC3D (DC3D.f90). "
            "Generated by tests/sources/reference/gen_dc3d.py. Do not edit by hand."
        ),
        "alpha": ALPHA,
        "poisson_ratio": 0.25,
        "seed": SEED,
        "n_faults": N_FAULTS,
        "n_points": N_POINTS,
        # Per-fault (map-frame) source parameters, shape [B].
        "source_x": [f["source_x"] for f in faults],
        "source_y": [f["source_y"] for f in faults],
        "strike": [f["strike"] for f in faults],
        "dip": [f["dip"] for f in faults],
        "centroid_depth": [f["centroid_depth"] for f in faults],
        "length": [f["length"] for f in faults],
        "width": [f["width"] for f in faults],
        "disl1": [f["disl1"] for f in faults],
        "disl2": [f["disl2"] for f in faults],
        "disl3": [f["disl3"] for f in faults],
        # Observations [B, N] and ENU ground truth [B, N, 3].
        "x_obs": x_obs.tolist(),
        "y_obs": y_obs.tolist(),
        "z_obs": z_obs.tolist(),
        "u_enu": enu.tolist(),
        # Fault-frame spatial derivatives [B, N, 9] in DC3D column order
        # (uxx,uyx,uzx, uxy,uyy,uzy, uxz,uyz,uzz) for the analytic_grad strain pass.
        "derivatives_fault_frame": deriv.tolist(),
        # ENU source-parameter gradients d(E,N,U)/d(param), each [B, N, 3].
        # disl{1,2,3} are exact (unit-slip responses); dip/length/width/
        # centroid_depth are central differences (steps below). source_x/source_y
        # gradients are omitted: half-space horizontal homogeneity makes them
        # exactly the negated horizontal spatial strain (derived in the test).
        "param_gradients": {k: g.tolist() for k, g in grad.items()},
        "grad_fd_steps": {"dip_rad": H_DIP, "length_width_depth_rel": H_REL},
    }

    DATA.mkdir(exist_ok=True)
    outfile = DATA / "dc3d_golden.json"
    outfile.write_text(json.dumps(payload))
    n = N_FAULTS * N_POINTS
    print(f"wrote {outfile} ({n} points, {outfile.stat().st_size / 1024:.0f} KiB)")


def summarize(path: Path = DATA / "dc3d_golden.json") -> None:
    """Print the committed golden JSON's header, array shapes, and the per-fault
    *input* parameters as a table -- a human-readable view of the compact blob
    without needing a Fortran toolchain to regenerate it. Outputs (u_enu,
    derivatives, gradients) are only reported by shape: they are a wall of floats
    the tests, not people, read.
    """
    if not path.is_file():
        sys.exit(f"golden file not found at {path}")
    data = json.loads(path.read_text())

    print(f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    print(data["_comment"])
    print()
    print("header")
    for k in ("alpha", "poisson_ratio", "seed", "n_faults", "n_points"):
        print(f"  {k:14s} {data[k]}")

    print()
    print("array shapes")
    array_fields = ["x_obs", "y_obs", "z_obs", "u_enu", "derivatives_fault_frame"]
    for k in array_fields:
        print(f"  {k:24s} {np.asarray(data[k]).shape}")
    for k, g in data["param_gradients"].items():
        print(f"  param_gradients[{k!r}]".ljust(26) + f" {np.asarray(g).shape}")

    print()
    print(f"per-fault inputs (strike/dip in degrees, depth/L/W in km)")
    header = ["idx", "strike", "dip", "depth", "L", "W",
              "src_x", "src_y", "disl1", "disl2", "disl3"]
    widths = [3, 7, 6, 7, 6, 6, 8, 8, 7, 7, 7]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for i in range(data["n_faults"]):
        row = [
            f"{i:d}",
            f"{math.degrees(data['strike'][i]):.1f}",
            f"{math.degrees(data['dip'][i]):.1f}",
            f"{data['centroid_depth'][i] / 1e3:.2f}",
            f"{data['length'][i] / 1e3:.2f}",
            f"{data['width'][i] / 1e3:.2f}",
            f"{data['source_x'][i]:.0f}",
            f"{data['source_y'][i]:.0f}",
            f"{data['disl1'][i]:.2f}",
            f"{data['disl2'][i]:.2f}",
            f"{data['disl3'][i]:.2f}",
        ]
        print("  " + " ".join(v.rjust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary", action="store_true",
        help="print header, array shapes, and per-fault inputs from the "
             "committed JSON, then exit (no Fortran toolchain needed).",
    )
    args = parser.parse_args()
    if args.summary:
        summarize()
    else:
        main()
