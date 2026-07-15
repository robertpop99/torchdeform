#!/usr/bin/env python3
"""Generate a random-volume forward ground truth for ``PCDMSource`` from the
original Nikkhoo et al. (2017) MATLAB ``pCDM.m``.

This is the pCDM analogue of ``gen_dc3d.py``: where ``gen_nikkhoo.m`` freezes a
handful of *fixed, hand-picked* points, this sweeps a wide **random parameter
volume** (source depth, orientation, potencies, and observation location) and
freezes the forward ENU displacement as ``../data/pcdm_volume_golden.json``.
That is the coverage ``PCDMSource``'s forward previously lacked -- the fixed
Nikkhoo points check three orientations, not the parameter space.

Why forward only (no gradient references, unlike ``gen_dc3d.py``):
  ``PCDMSource`` is a plain differentiable forward -- no hand-written backward
  (contrast ``OkadaSource.analytic_grad``, whose closed-form strain needs an
  external check). Its gradients are autograd of the forward, already covered by
  ``torch.autograd.gradcheck`` in ``test_pcdm_source.py`` (self-contained: it
  compares autograd to finite differences of the forward). So a correct forward
  plus a passing gradcheck pins the gradients; golden gradients would be
  redundant. This fixture therefore validates only the forward.

Same spirit as ``gen_dc3d.py``: randomness/geometry stays in Python (numpy) and
MATLAB is only ever the pure reference kernel -- no cross-language RNG to
reconcile. One ``matlab -batch`` call evaluates every sampled point.

Convention bridge (must match ``PCDMSource``)
---------------------------------------------
``pCDM.m`` takes rotation angles ``omegaX/Y/Z`` in **degrees** and returns
surface ENU displacement; ``PCDMSource`` takes them in **radians**. Sampling
works in degrees (for MATLAB) but the JSON stores radians, so the test feeds the
values to ``PCDMSource`` directly. Material: ``nu = 0.25`` (the source default).
Potencies ``DVx/DVy/DVz`` share a sign (pCDM rejects mixed signs).

Regenerate::

    python gen_pcdm_volume.py                 # needs MATLAB + vendored nikkhoo/
    python gen_pcdm_volume.py --summary        # inspect committed JSON, no MATLAB
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
NIKKHOO = HERE / "nikkhoo"

NU = 0.25                  # PCDMSource default Poisson ratio
SEED = 20240709
N_SOURCES = 16             # batch dimension B
N_POINTS = 24              # observation points per source N

# A pure-compute MATLAB driver: read one evaluation per row from INPUT (each row
# is X Y X0 Y0 depth omegaX omegaY omegaZ DVx DVy DVz nu), call pCDM per row, and
# write ue un uv to OUTPUT at full double precision. Absolute paths are baked in
# so there is no CLI quoting or cwd dependence. pCDM is per-point independent, so
# one-row-per-point keeps the driver trivial and fully general over source params.
DRIVER_M = r"""
addpath('{nikkhoo}');
inp = load('{input}');
K = size(inp, 1);
out = zeros(K, 3);
for i = 1:K
    r = inp(i, :);
    [ue, un, uv] = pCDM(r(1), r(2), r(3), r(4), r(5), ...
                        r(6), r(7), r(8), r(9), r(10), r(11), r(12));
    out(i, :) = [ue un uv];
end
fid = fopen('{output}', 'w');
fprintf(fid, '%.17e %.17e %.17e\n', out');
fclose(fid);
"""


def run_pcdm(rows: np.ndarray) -> np.ndarray:
    """rows: [K,12] pCDM inputs -> out: [K,3] ENU surface displacement.

    Marshals through temp files and one ``matlab -batch`` call. MATLAB is only
    the reference kernel; every sampled point is Python-generated.
    """
    if not (NIKKHOO / "pCDM.m").is_file():
        sys.exit(f"vendored pCDM.m not found in {NIKKHOO}.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp, outp = tmp / "in.txt", tmp / "out.txt"
        np.savetxt(inp, rows, fmt="%.17e")
        driver = tmp / "pcdm_driver.m"
        driver.write_text(DRIVER_M.format(
            nikkhoo=NIKKHOO, input=inp, output=outp,
        ))
        try:
            subprocess.run(
                ["matlab", "-batch", f"run('{driver}')"],
                check=True, capture_output=True, text=True,
            )
        except FileNotFoundError:
            sys.exit("matlab not found on PATH; install it to regenerate.")
        except subprocess.CalledProcessError as e:
            sys.exit(f"matlab failed:\n{e.stderr or e.stdout}")
        out = np.loadtxt(outp).reshape(rows.shape[0], 3)
    assert out.shape == (rows.shape[0], 3), (out.shape, rows.shape)
    return out


def sample_source(rng: np.random.Generator) -> dict:
    """One physically-sane buried pCDM with a random orientation and potencies.

    Angles are sampled in degrees (pCDM's unit); potencies share a random sign
    (pCDM rejects mixed signs) with independent magnitudes so the three PTDs are
    genuinely anisotropic.
    """
    sign = float(rng.choice([-1.0, 1.0]))
    return dict(
        source_x=float(rng.uniform(-2e3, 2e3)),
        source_y=float(rng.uniform(-2e3, 2e3)),
        depth=float(rng.uniform(1.5e3, 8e3)),
        omega_x_deg=float(rng.uniform(-180.0, 180.0)),
        omega_y_deg=float(rng.uniform(-180.0, 180.0)),
        omega_z_deg=float(rng.uniform(-180.0, 180.0)),
        dv_x=sign * float(rng.uniform(1e5, 1e7)),
        dv_y=sign * float(rng.uniform(1e5, 1e7)),
        dv_z=sign * float(rng.uniform(1e5, 1e7)),
    )


def sample_obs(rng: np.random.Generator, s: dict) -> tuple[float, float]:
    """One surface observation point in map frame; footprint scales with depth.

    Surface only (pCDM is a surface solution) and never singular:
    ``r = sqrt(dx^2 + dy^2 + depth^2) >= depth > 0`` everywhere.
    """
    reach = 4.0 * s["depth"]
    x = s["source_x"] + rng.uniform(-reach, reach)
    y = s["source_y"] + rng.uniform(-reach, reach)
    return float(x), float(y)


def main(nu: float = NU) -> None:
    rng = np.random.default_rng(SEED)
    sources = [sample_source(rng) for _ in range(N_SOURCES)]
    pts = [[sample_obs(rng, s) for _ in range(N_POINTS)] for s in sources]

    # One row per (source, point): X Y X0 Y0 depth omegaX omegaY omegaZ DV* nu.
    rows = np.array([
        [x, y, s["source_x"], s["source_y"], s["depth"],
         s["omega_x_deg"], s["omega_y_deg"], s["omega_z_deg"],
         s["dv_x"], s["dv_y"], s["dv_z"], nu]
        for s, sp in zip(sources, pts) for (x, y) in sp
    ])
    enu = run_pcdm(rows).reshape(N_SOURCES, N_POINTS, 3)

    x_obs = np.array([[p[0] for p in sp] for sp in pts])
    y_obs = np.array([[p[1] for p in sp] for sp in pts])

    payload = {
        "_comment": (
            "Random-volume forward ground truth for PCDMSource, from the "
            "original Nikkhoo (2017) MATLAB pCDM.m. Generated by "
            "tests/sources/reference/gen_pcdm_volume.py. Do not edit by hand."
        ),
        "poisson_ratio": nu,
        "seed": SEED,
        "n_sources": N_SOURCES,
        "n_points": N_POINTS,
        # Per-source parameters, shape [B]. Angles in RADIANS (PCDMSource units).
        "source_x": [s["source_x"] for s in sources],
        "source_y": [s["source_y"] for s in sources],
        "depth": [s["depth"] for s in sources],
        "omega_x": [math.radians(s["omega_x_deg"]) for s in sources],
        "omega_y": [math.radians(s["omega_y_deg"]) for s in sources],
        "omega_z": [math.radians(s["omega_z_deg"]) for s in sources],
        "dv_x": [s["dv_x"] for s in sources],
        "dv_y": [s["dv_y"] for s in sources],
        "dv_z": [s["dv_z"] for s in sources],
        # Observations [B, N] and ENU ground truth [B, N, 3].
        "x_obs": x_obs.tolist(),
        "y_obs": y_obs.tolist(),
        "u_enu": enu.tolist(),
    }

    DATA.mkdir(exist_ok=True)
    # nu = 0.25 keeps the canonical filename; other ratios get a suffixed
    # sibling (e.g. pcdm_volume_golden_nu0.32.json) so the default fixture and
    # the off-default material check coexist.
    suffix = "" if nu == NU else f"_nu{nu:g}"
    outfile = DATA / f"pcdm_volume_golden{suffix}.json"
    outfile.write_text(json.dumps(payload))
    n = N_SOURCES * N_POINTS
    print(f"wrote {outfile} ({n} points, {outfile.stat().st_size / 1024:.0f} KiB)")


def summarize(path: Path = DATA / "pcdm_volume_golden.json") -> None:
    """Print the committed golden JSON's header, array shapes, and the per-source
    *input* parameters as a table -- a human-readable view of the compact blob
    without needing MATLAB. Outputs (u_enu) are only reported by shape.
    """
    if not path.is_file():
        sys.exit(f"golden file not found at {path}")
    data = json.loads(path.read_text())

    print(f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    print(data["_comment"])
    print()
    print("header")
    for k in ("poisson_ratio", "seed", "n_sources", "n_points"):
        print(f"  {k:14s} {data[k]}")

    print()
    print("array shapes")
    for k in ("x_obs", "y_obs", "u_enu"):
        print(f"  {k:8s} {np.asarray(data[k]).shape}")

    print()
    print("per-source inputs (omega in degrees, depth in km, potencies in m^3)")
    header = ["idx", "omega_x", "omega_y", "omega_z", "depth",
              "src_x", "src_y", "dv_x", "dv_y", "dv_z"]
    widths = [3, 8, 8, 8, 6, 7, 7, 10, 10, 10]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for i in range(data["n_sources"]):
        row = [
            f"{i:d}",
            f"{math.degrees(data['omega_x'][i]):.1f}",
            f"{math.degrees(data['omega_y'][i]):.1f}",
            f"{math.degrees(data['omega_z'][i]):.1f}",
            f"{data['depth'][i] / 1e3:.2f}",
            f"{data['source_x'][i]:.0f}",
            f"{data['source_y'][i]:.0f}",
            f"{data['dv_x'][i]:.2e}",
            f"{data['dv_y'][i]:.2e}",
            f"{data['dv_z'][i]:.2e}",
        ]
        print("  " + " ".join(v.rjust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary", action="store_true",
        help="print header, array shapes, and per-source inputs from the "
             "committed JSON, then exit (no MATLAB needed).",
    )
    parser.add_argument(
        "--nu", type=float, default=NU,
        help="Poisson ratio for the reference run (default %(default)s). "
             "Non-default values write a _nu<value>-suffixed fixture.",
    )
    args = parser.parse_args()
    if args.summary:
        summarize()
    else:
        main(args.nu)
