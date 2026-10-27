#!/usr/bin/env python3
"""Generate a random-volume forward ground truth for ``SpheroidSource`` from
Cervelli's USGS MATLAB ``spheroid.m`` (Yang et al. 1988, corrected).

The spheroid analogue of ``gen_pecm_volume.py``: it sweeps a wide random
parameter space (depth, orientation, prolate *and* oblate semi-axes, strength,
observation location) and freezes the forward ENU surface displacement, in both
strength modes, as ``../data/spheroid_volume_golden.json``.

Each source also gets two deliberately awkward observation points: where its
polar axis pierces the surface (``spheroid.m``'s thresholded ``R1 + r3bar -> 0``
limit, which ``SpheroidSource`` handles by an exact rewrite instead; only when
that point is within the sampling reach, see ``sample_obs``), and one on the
strike line (``y2 = 0``, where ``spheroid.m`` divides by ``y2`` inside an
``atan``). A few sources are pinned to exactly horizontal / vertical axes, to
reach ``spheroid.m``'s ``st == 0`` and ``ct == 0`` branches.

Random dips within 2 degrees of vertical and shapes within 2% of a sphere are
redrawn: there ``spheroid.m``'s own general-branch round-off grows (~1e-9 of the
peak and worse), so the fixture would measure MATLAB's conditioning rather than
the port. ``SpheroidSource``'s interpolation bands for those limits are checked
against a 50-digit evaluation instead (``gen_spheroid_mp.py``).

Forward only: ``SpheroidSource`` is a plain differentiable forward with no
hand-written backward, so its gradients are autograd of the forward, covered by
``torch.autograd.gradcheck`` in ``test_spheroid_source.py``.

Convention bridge (must match ``SpheroidSource``)
-------------------------------------------------
``spheroid.m`` takes ``m = [a b dip strike X1 X2 X3 strength]`` with angles in
**degrees**, ``X3 = -depth``, and a strength whose **negative** inflates.
``SpheroidSource`` takes radians and positive-inflating strength, so this script
passes ``-strength`` to the kernel. The JSON stores radians and the
``SpheroidSource`` sign. Material: ``nu = 0.25``, ``mu = 3e10`` (source defaults).

Regenerate::

    python gen_spheroid_volume.py              # needs MATLAB + vendored cervelli/
    python gen_spheroid_volume.py --nu 0.32    # off-default Poisson ratio sibling
    python gen_spheroid_volume.py --summary    # inspect committed JSON, no MATLAB
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
CERVELLI = HERE / "cervelli"

NU = 0.25                  # SpheroidSource default Poisson ratio
MU = 3.0e10                # SpheroidSource default shear modulus (Pa)
SEED = 20261026
N_SOURCES = 20             # batch dimension B
N_POINTS = 24              # observation points per source N (incl. 2 special)
PINNED_DIPS = {0: 90.0, 1: 0.0, 2: 90.0, 3: 0.0}   # source index -> exact dip (deg)

# Pure-compute MATLAB driver: one spheroid.m call per source (all its points at
# once, z = 0), for each strength mode. Absolute paths baked in.
DRIVER_M = r"""
addpath('{cervelli}');
M = load('{models}');
X = load('{xs}');
Y = load('{ys}');
modes = {{'pressure', 'volume'}};
for j = 1:2
    out = zeros(size(M, 1) * size(X, 2), 3);
    for i = 1:size(M, 1)
        xyz = [X(i, :); Y(i, :); zeros(1, size(X, 2))];
        m = [M(i, 1:7) M(i, 7 + j)]';     % strength column 8 (p) or 9 (dV)
        u = spheroid(m, xyz, {nu}, {mu}, modes{{j}});
        out((i - 1) * size(X, 2) + (1:size(X, 2)), :) = u';
    end
    fid = fopen(sprintf('{outbase}_%d.txt', j), 'w');
    fprintf(fid, '%.17e %.17e %.17e\n', out');
    fclose(fid);
end
"""


def run_spheroid(models: np.ndarray, xs: np.ndarray, ys: np.ndarray, nu: float):
    """models: [B,9] = (a b dip strike X1 X2 X3 P_matlab V_matlab); xs/ys [B,N]
    -> (u_pressure, u_volume), each [B,N,3] ENU in spheroid.m's sign."""
    if not (CERVELLI / "spheroid.m").is_file():
        sys.exit(f"vendored spheroid.m not found in {CERVELLI}.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, arr in (("models", models), ("xs", xs), ("ys", ys)):
            np.savetxt(tmp / f"{name}.txt", arr, fmt="%.17e")
        driver = tmp / "spheroid_driver.m"
        driver.write_text(DRIVER_M.format(
            cervelli=CERVELLI, models=tmp / "models.txt", xs=tmp / "xs.txt",
            ys=tmp / "ys.txt", outbase=tmp / "out", nu=repr(nu), mu=repr(MU)))
        # Log to a file, not a pipe: MATLAB's detached service host can inherit
        # and hold a pipe open, hanging capture_output after MATLAB exits.
        log = tmp / "matlab.log"
        try:
            with log.open("w") as fh:
                subprocess.run(["matlab", "-batch", f"run('{driver}')"], check=True,
                               stdin=subprocess.DEVNULL, stdout=fh,
                               stderr=subprocess.STDOUT)
        except FileNotFoundError:
            sys.exit("matlab not found on PATH; install it to regenerate.")
        except subprocess.CalledProcessError:
            sys.exit(f"matlab failed:\n{log.read_text()}")
        shape = (xs.shape[0], xs.shape[1], 3)
        return tuple(np.loadtxt(tmp / f"out_{j}.txt").reshape(shape) for j in (1, 2))


def sample_source(rng: np.random.Generator, i: int) -> dict:
    """One physically-sane buried spheroid: prolate or oblate, any orientation,
    random-signed pressure and volume change."""
    while True:
        depth = float(rng.uniform(2e3, 8e3))
        a = float(rng.uniform(200.0, 1500.0))
        if rng.random() < 0.6:
            b = a * float(rng.uniform(0.15, 0.98))     # prolate
        else:
            b = a / float(rng.uniform(0.3, 0.98))      # oblate
        dip = PINNED_DIPS.get(i, float(rng.uniform(0.0, 180.0)))
        if i not in PINNED_DIPS and abs(dip - 90.0) < 2.0:
            continue
        th = math.radians(dip)
        if math.hypot(a * math.sin(th), b * math.cos(th)) < 0.8 * depth:
            break
    return dict(
        source_x=float(rng.uniform(-2e3, 2e3)),
        source_y=float(rng.uniform(-2e3, 2e3)),
        depth=depth, polar_radius=a, equatorial_radius=b,
        dip_deg=dip, strike_deg=float(rng.uniform(0.0, 360.0)),
        pressure=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5e6, 5e6)),
        delta_v=float(rng.choice([-1.0, 1.0]) * rng.uniform(1e5, 1e7)),
    )


def sample_obs(rng: np.random.Generator, s: dict) -> tuple[list[float], list[float]]:
    """N surface points: random within 4 depths, plus the axis piercing point
    (when it lies within that reach) and a point on the strike line.

    A piercing point farther out (shallow dips) is replaced by another
    strike-line point: there ``spheroid.m`` itself is wrong -- ``R1 + r3bar``
    is formed directly, so its round-off (~eps * distance) rivals the fixed
    ``ZERO_THRESHOLD = 1e-11`` and the limit branch fires unreliably (e.g. 91 m
    450 km out, where the true field is ~1e-8 m). ``SpheroidSource`` is exact there; the
    test suite checks it by continuity instead.
    """
    reach = 4.0 * s["depth"]
    xs = list(s["source_x"] + rng.uniform(-reach, reach, N_POINTS - 2))
    ys = list(s["source_y"] + rng.uniform(-reach, reach, N_POINTS - 2))
    th, az = math.radians(s["dip_deg"]), math.radians(s["strike_deg"])
    # distance along the azimuth of the piercing point (inf if horizontal)
    t = -s["depth"] / math.tan(th) if math.sin(th) > 1e-12 else math.inf
    if abs(t) > reach:
        t = float(rng.uniform(-reach, reach))
    for along in (t, float(rng.uniform(-reach, reach))):
        xs.append(s["source_x"] + along * math.sin(az))
        ys.append(s["source_y"] + along * math.cos(az))
    return xs, ys


def main(nu: float = NU) -> None:
    rng = np.random.default_rng(SEED)
    sources = [sample_source(rng, i) for i in range(N_SOURCES)]
    obs = [sample_obs(rng, s) for s in sources]
    x_obs = np.array([o[0] for o in obs])
    y_obs = np.array([o[1] for o in obs])

    models = np.array([
        [s["polar_radius"], s["equatorial_radius"], s["dip_deg"], s["strike_deg"],
         s["source_x"], s["source_y"], -s["depth"], -s["pressure"], -s["delta_v"]]
        for s in sources
    ])
    u_p, u_v = run_spheroid(models, x_obs, y_obs, nu)

    payload = {
        "_comment": (
            "Random-volume forward ground truth for SpheroidSource, from Cervelli's "
            "USGS MATLAB spheroid.m. Generated by "
            "tests/sources/reference/gen_spheroid_volume.py. Do not edit by hand."
        ),
        "poisson_ratio": nu,
        "shear_modulus": MU,
        "seed": SEED,
        "n_sources": N_SOURCES,
        "n_points": N_POINTS,
        # Per-source parameters [B]. Angles in RADIANS, positive strength
        # inflating (SpheroidSource conventions).
        "source_x": [s["source_x"] for s in sources],
        "source_y": [s["source_y"] for s in sources],
        "depth": [s["depth"] for s in sources],
        "polar_radius": [s["polar_radius"] for s in sources],
        "equatorial_radius": [s["equatorial_radius"] for s in sources],
        "dip": [math.radians(s["dip_deg"]) for s in sources],
        "strike": [math.radians(s["strike_deg"]) for s in sources],
        "pressure": [s["pressure"] for s in sources],
        "delta_v": [s["delta_v"] for s in sources],
        "x_obs": x_obs.tolist(),
        "y_obs": y_obs.tolist(),
        "u_enu_pressure": u_p.tolist(),
        "u_enu_volume": u_v.tolist(),
    }

    DATA.mkdir(exist_ok=True)
    suffix = "" if nu == NU else f"_nu{nu:g}"
    outfile = DATA / f"spheroid_volume_golden{suffix}.json"
    outfile.write_text(json.dumps(payload))
    n = N_SOURCES * N_POINTS
    print(f"wrote {outfile} ({n} points x 2 modes, "
          f"{outfile.stat().st_size / 1024:.0f} KiB)")


def summarize(path: Path = DATA / "spheroid_volume_golden.json") -> None:
    """Header, array shapes, and the per-source *input* parameters as a table --
    a human-readable view of the compact blob without needing MATLAB."""
    if not path.is_file():
        sys.exit(f"golden file not found at {path}")
    data = json.loads(path.read_text())

    print(f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    print(data["_comment"])
    print()
    print("header")
    for k in ("poisson_ratio", "shear_modulus", "seed", "n_sources", "n_points"):
        print(f"  {k:14s} {data[k]}")

    print()
    print("array shapes")
    for k in ("x_obs", "y_obs", "u_enu_pressure", "u_enu_volume"):
        print(f"  {k:15s} {np.asarray(data[k]).shape}")

    print()
    print("per-source inputs (angles in degrees, depth in km, semi-axes in m, "
          "p in MPa, dV in 1e6 m^3)")
    header = ["idx", "dip", "strike", "depth", "a", "b", "a/b", "p_MPa", "dV"]
    widths = [3, 7, 7, 6, 6, 6, 5, 7, 7]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for i in range(data["n_sources"]):
        a, b = data["polar_radius"][i], data["equatorial_radius"][i]
        row = [
            f"{i:d}",
            f"{math.degrees(data['dip'][i]):.1f}",
            f"{math.degrees(data['strike'][i]):.1f}",
            f"{data['depth'][i] / 1e3:.2f}",
            f"{a:.0f}", f"{b:.0f}", f"{a / b:.2f}",
            f"{data['pressure'][i] / 1e6:.2f}",
            f"{data['delta_v'][i] / 1e6:.2f}",
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
