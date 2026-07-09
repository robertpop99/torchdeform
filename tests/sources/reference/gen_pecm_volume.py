#!/usr/bin/env python3
"""Generate a random-volume forward ground truth for ``PECMSource`` from the
original Nikkhoo et al. (2017) MATLAB ``pECM.m``.

The point-ellipsoidal-cavity analogue of ``gen_pcdm_volume.py``: where
``gen_nikkhoo.m`` freezes two hand-picked orientations, this sweeps a wide random
parameter space (depth, orientation, semi-axes, pressure, observation location)
and freezes the forward ENU displacement as ``../data/pecm_volume_golden.json``.

Forward only, for the same reason as the pCDM fixture: ``PECMSource`` is a plain
differentiable forward with no hand-written backward, so its gradients are
autograd of the forward, already covered by ``torch.autograd.gradcheck`` in
``test_pecm_source.py``. See ``gen_pcdm_volume.py`` / README for the rationale.

Convention bridge (must match ``PECMSource``)
---------------------------------------------
``pECM.m`` takes ``omegaX/Y/Z`` in **degrees**, semi-axes ``ax/ay/az`` in metres,
pressure ``p`` in Pa, and the Lame constants ``mu, lambda`` explicitly.
``PECMSource(poisson_ratio=nu, shear_modulus=mu)`` takes angles in **radians** and
derives ``lambda = 2*mu*nu/(1-2nu)`` internally, so this script passes exactly
that lambda to the kernel. Material: ``nu = 0.25``, ``mu = 3e10`` -> ``lambda =
3e10`` (source defaults). Sampling works in degrees; the JSON stores radians.

Regenerate::

    python gen_pecm_volume.py                 # needs MATLAB + vendored nikkhoo/
    python gen_pecm_volume.py --summary        # inspect committed JSON, no MATLAB
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

NU = 0.25                  # PECMSource default Poisson ratio
MU = 3.0e10                # PECMSource default shear modulus (Pa)
LAMBDA = 2.0 * MU * NU / (1.0 - 2.0 * NU)   # = 3e10 for nu = 0.25
SEED = 20240709
N_SOURCES = 16             # batch dimension B
N_POINTS = 24              # observation points per source N

# Pure-compute MATLAB driver: one evaluation per input row
# (X Y X0 Y0 depth omegaX omegaY omegaZ ax ay az p mu lambda), call pECM per row,
# write ue un uv at full double precision. Absolute paths baked in.
DRIVER_M = r"""
addpath('{nikkhoo}');
inp = load('{input}');
K = size(inp, 1);
out = zeros(K, 3);
for i = 1:K
    r = inp(i, :);
    [ue, un, uv] = pECM(r(1), r(2), r(3), r(4), r(5), r(6), r(7), r(8), ...
                        r(9), r(10), r(11), r(12), r(13), r(14));
    out(i, :) = [ue un uv];
end
fid = fopen('{output}', 'w');
fprintf(fid, '%.17e %.17e %.17e\n', out');
fclose(fid);
"""


def run_pecm(rows: np.ndarray) -> np.ndarray:
    """rows: [K,14] pECM inputs -> out: [K,3] ENU surface displacement."""
    if not (NIKKHOO / "pECM.m").is_file():
        sys.exit(f"vendored pECM.m not found in {NIKKHOO}.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp, outp = tmp / "in.txt", tmp / "out.txt"
        np.savetxt(inp, rows, fmt="%.17e")
        driver = tmp / "pecm_driver.m"
        driver.write_text(DRIVER_M.format(nikkhoo=NIKKHOO, input=inp, output=outp))
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
    """One physically-sane buried pECM: random orientation, semi-axes, pressure.

    Semi-axes stay small relative to depth (point-source regime); pressure
    carries a random sign (inflation or deflation)."""
    return dict(
        source_x=float(rng.uniform(-2e3, 2e3)),
        source_y=float(rng.uniform(-2e3, 2e3)),
        depth=float(rng.uniform(2e3, 8e3)),
        omega_x_deg=float(rng.uniform(-180.0, 180.0)),
        omega_y_deg=float(rng.uniform(-180.0, 180.0)),
        omega_z_deg=float(rng.uniform(-180.0, 180.0)),
        a_x=float(rng.uniform(100.0, 400.0)),
        a_y=float(rng.uniform(100.0, 400.0)),
        a_z=float(rng.uniform(100.0, 400.0)),
        pressure=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5e6, 5e6)),
    )


def sample_obs(rng: np.random.Generator, s: dict) -> tuple[float, float]:
    """One surface observation point; footprint scales with depth (never singular:
    the source is buried, obs at the surface)."""
    reach = 4.0 * s["depth"]
    return (float(s["source_x"] + rng.uniform(-reach, reach)),
            float(s["source_y"] + rng.uniform(-reach, reach)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    sources = [sample_source(rng) for _ in range(N_SOURCES)]
    pts = [[sample_obs(rng, s) for _ in range(N_POINTS)] for s in sources]

    rows = np.array([
        [x, y, s["source_x"], s["source_y"], s["depth"],
         s["omega_x_deg"], s["omega_y_deg"], s["omega_z_deg"],
         s["a_x"], s["a_y"], s["a_z"], s["pressure"], MU, LAMBDA]
        for s, sp in zip(sources, pts) for (x, y) in sp
    ])
    enu = run_pecm(rows).reshape(N_SOURCES, N_POINTS, 3)

    x_obs = np.array([[p[0] for p in sp] for sp in pts])
    y_obs = np.array([[p[1] for p in sp] for sp in pts])

    payload = {
        "_comment": (
            "Random-volume forward ground truth for PECMSource, from the original "
            "Nikkhoo (2017) MATLAB pECM.m. Generated by "
            "tests/sources/reference/gen_pecm_volume.py. Do not edit by hand."
        ),
        "poisson_ratio": NU,
        "shear_modulus": MU,
        "lambda": LAMBDA,
        "seed": SEED,
        "n_sources": N_SOURCES,
        "n_points": N_POINTS,
        # Per-source parameters [B]. Angles in RADIANS (PECMSource units).
        "source_x": [s["source_x"] for s in sources],
        "source_y": [s["source_y"] for s in sources],
        "depth": [s["depth"] for s in sources],
        "omega_x": [math.radians(s["omega_x_deg"]) for s in sources],
        "omega_y": [math.radians(s["omega_y_deg"]) for s in sources],
        "omega_z": [math.radians(s["omega_z_deg"]) for s in sources],
        "a_x": [s["a_x"] for s in sources],
        "a_y": [s["a_y"] for s in sources],
        "a_z": [s["a_z"] for s in sources],
        "pressure": [s["pressure"] for s in sources],
        "x_obs": x_obs.tolist(),
        "y_obs": y_obs.tolist(),
        "u_enu": enu.tolist(),
    }

    DATA.mkdir(exist_ok=True)
    outfile = DATA / "pecm_volume_golden.json"
    outfile.write_text(json.dumps(payload))
    n = N_SOURCES * N_POINTS
    print(f"wrote {outfile} ({n} points, {outfile.stat().st_size / 1024:.0f} KiB)")


def summarize(path: Path = DATA / "pecm_volume_golden.json") -> None:
    """Header, array shapes, and the per-source *input* parameters as a table --
    a human-readable view of the compact blob without needing MATLAB."""
    if not path.is_file():
        sys.exit(f"golden file not found at {path}")
    data = json.loads(path.read_text())

    print(f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    print(data["_comment"])
    print()
    print("header")
    for k in ("poisson_ratio", "shear_modulus", "lambda", "seed",
              "n_sources", "n_points"):
        print(f"  {k:14s} {data[k]}")

    print()
    print("array shapes")
    for k in ("x_obs", "y_obs", "u_enu"):
        print(f"  {k:8s} {np.asarray(data[k]).shape}")

    print()
    print("per-source inputs (omega in degrees, depth in km, semi-axes in m, p in MPa)")
    header = ["idx", "omega_x", "omega_y", "omega_z", "depth",
              "a_x", "a_y", "a_z", "p_MPa"]
    widths = [3, 8, 8, 8, 6, 6, 6, 6, 8]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for i in range(data["n_sources"]):
        row = [
            f"{i:d}",
            f"{math.degrees(data['omega_x'][i]):.1f}",
            f"{math.degrees(data['omega_y'][i]):.1f}",
            f"{math.degrees(data['omega_z'][i]):.1f}",
            f"{data['depth'][i] / 1e3:.2f}",
            f"{data['a_x'][i]:.0f}",
            f"{data['a_y'][i]:.0f}",
            f"{data['a_z'][i]:.0f}",
            f"{data['pressure'][i] / 1e6:.2f}",
        ]
        print("  " + " ".join(v.rjust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary", action="store_true",
        help="print header, array shapes, and per-source inputs from the "
             "committed JSON, then exit (no MATLAB needed).",
    )
    args = parser.parse_args()
    if args.summary:
        summarize()
    else:
        main()
