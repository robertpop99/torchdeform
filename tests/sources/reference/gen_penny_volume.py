#!/usr/bin/env python3
"""Generate a random-volume forward ground truth for ``PennySource`` from the
original Fialko, Khazan & Simons (2001) MATLAB penny-crack code.

The penny-crack analogue of ``gen_pcdm_volume.py`` and ``gen_dc3d.py``: where
``gen_fialko.m`` freezes three fixed depth ratios on a fixed radial grid, this
sweeps a random volume and freezes the forward displacement as
``../data/penny_volume_golden.json``. The Fialko solution is dimensionless in
``h = depth/radius``, so the "volume" is 1-D in ``h`` (times the radial sampling):
each source draws a random ``h`` and a random set of dimensionless radii ``r/a``.

Forward only, like the other volume fixtures: ``PennySource`` is a plain
differentiable forward with no hand-written backward, so its gradients are
autograd of the forward, covered by ``torch.autograd.gradcheck`` in
``test_penny_source.py``. See ``gen_pcdm_volume.py`` / README for the rationale.

Reference code: the **original** Fialko ``penny.tar.gz`` (scalar-radius ``Q.m``,
whose ``intgr.m`` already carries the correct per-radius vertical formula) --
*not* the GeodMod mirror, whose vectorised ``Uz`` line is buggy. The driver below
recomputes ``Uz`` with that same per-radius loop, so the result is correct
regardless of which ``intgr.m`` variant is installed. Observation points lie on
the +East radius (``y = 0``), so the horizontal field is purely radial:
``ue == Ur``, ``un == 0``.

NOT redistributed: Fialko's penny code carries no redistribution license.
Download the original into a local ``./penny/`` (git-ignored) before running --
e.g. ``wget http://igppweb.ucsd.edu/~fialko/Assets/Software/penny.tar.gz`` (see
reference/README.md). Physical scaling matches ``PennySource``: dimensionless
``(Uz, Ur)`` depend only on ``h``; ``Pf = 2(1-nu) a P / mu`` sets the scale, with
``uz_phys = -Uz*Pf`` and ``ur_phys = Ur*Pf``. Material: ``nu = 0.25``,
``mu = 3e10`` (source defaults); ``nis = 2`` (PennySource default).

Regenerate::

    python gen_penny_volume.py                 # needs MATLAB + downloaded penny/
    python gen_penny_volume.py --summary        # inspect committed JSON, no MATLAB
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PENNY = HERE / "penny"

NU = 0.25                  # PennySource default Poisson ratio
MU = 3.0e10                # PennySource default shear modulus (Pa)
NIS = 2                    # sub-intervals (PennySource default)
FRED_EPS = 1e-10           # tight Fredholm convergence
SEED = 20240709
N_SOURCES = 16             # number of random depth ratios h (batch dimension B)
N_POINTS = 24              # dimensionless radii per source N

# MATLAB driver over a batch of random depth ratios. For each h it solves the
# Fredholm equation once, then evaluates Ur (unaffected by the mirror bug) via
# intgr, and Uz per-radius with the ORIGINAL Fialko loop formula -- identical to
# intgr.m's own (correct) Uz line, recomputed here so the result does not depend
# on whether the installed intgr.m carries the buggy vectorised Uz. Q(h,t,r,n)
# takes a *scalar* radius r and vector node t, returning a vector over t. Reads
# H [D] and R [D x NR]; writes [Uz | Ur] as [D x 2*NR].
DRIVER_M = r"""
addpath('{penny}');
global NumLegendreTerms %#ok<GVMIS>
H = load('{hfile}');
R = load('{rfile}');
nis = {nis};
ep = {eps};
D = numel(H);
NR = size(R, 2);
Uz = zeros(D, NR);
Ur = zeros(D, NR);
for k = 1:D
    h = H(k);
    [fi, psi, t, Wt] = fredholm(h, nis, ep);
    rk = R(k, :);
    [~, ur] = intgr(rk, fi, psi, h, Wt, t);     % Ur is correct in intgr.m
    for j = 1:NR                                 % Uz: original per-radius loop
        rj = rk(j);
        Q1 = Q(h, t, rj, 1);
        Q2 = Q(h, t, rj, 2);
        Q3 = Q(h, t, rj, 3);
        Uz(k, j) = sum(Wt .* (fi .* (Q1 + h*Q2) + psi .* (Q1./t - Q3)));
    end
    Ur(k, :) = ur(:)';
end
M = [Uz Ur];
fid = fopen('{output}', 'w');
for k = 1:D
    fprintf(fid, '%.17e ', M(k, :));
    fprintf(fid, '\n');
end
fclose(fid);
"""


def run_penny(h: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h: [D] depth ratios, r: [D,NR] dimensionless radii ->
    (Uz, Ur) each [D,NR], dimensionless."""
    if not (PENNY / "fredholm.m").is_file():
        sys.exit(
            f"Fialko penny code not found in {PENNY}.\n"
            "Download it there first (see gen_fialko.m / reference/README.md)."
        )
    D, NR = r.shape
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        hfile, rfile, outp = tmp / "h.txt", tmp / "r.txt", tmp / "out.txt"
        np.savetxt(hfile, h, fmt="%.17e")
        np.savetxt(rfile, r, fmt="%.17e")
        driver = tmp / "penny_driver.m"
        driver.write_text(DRIVER_M.format(
            penny=PENNY, hfile=hfile, rfile=rfile, output=outp, nis=NIS, eps=FRED_EPS,
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
        m = np.loadtxt(outp).reshape(D, 2 * NR)
    return m[:, :NR], m[:, NR:]           # Uz, Ur


def main() -> None:
    rng = np.random.default_rng(SEED)
    # Per source: crack radius a, depth ratio h -> depth = h*a, pressure P.
    # h drawn over the well-behaved range Fialko's own examples use (>= 0.8).
    a = rng.uniform(500.0, 3000.0, N_SOURCES)
    h = rng.uniform(0.8, 3.5, N_SOURCES)
    depth = h * a
    pressure = rng.choice([-1.0, 1.0], N_SOURCES) * rng.uniform(0.5e6, 5e6, N_SOURCES)
    # Dimensionless radii per source: inside (r<1) and outside (r>1) the crack.
    r_dimless = rng.uniform(0.01, 3.0, (N_SOURCES, N_POINTS))

    uz, ur = run_penny(h, r_dimless)                 # dimensionless [B, N]

    pf = 2.0 * (1.0 - NU) * a * pressure / MU        # [B] physical scale
    ur_phys = ur * pf[:, None]                       # radial (== +East here)
    uz_phys = -uz * pf[:, None]                      # vertical (Fialko sign)
    x_obs = r_dimless * a[:, None]                   # physical radius on +East

    payload = {
        "_comment": (
            "Random-volume forward ground truth for PennySource, from the original "
            "Fialko et al. (2001) MATLAB penny-crack code (corrected Uz). Generated "
            "by tests/sources/reference/gen_penny_volume.py. Do not edit by hand."
        ),
        "poisson_ratio": NU,
        "shear_modulus": MU,
        "nis": NIS,
        "seed": SEED,
        "n_sources": N_SOURCES,
        "n_points": N_POINTS,
        # Per-source parameters [B].
        "radius": a.tolist(),
        "depth": depth.tolist(),
        "pressure": pressure.tolist(),
        # Observations on the +East radius [B, N] and physical displacement [B, N].
        # y_obs is implicitly 0, so the horizontal field is purely radial (== ue).
        "x_obs": x_obs.tolist(),
        "ur": ur_phys.tolist(),
        "uz": uz_phys.tolist(),
    }

    DATA.mkdir(exist_ok=True)
    outfile = DATA / "penny_volume_golden.json"
    outfile.write_text(json.dumps(payload))
    n = N_SOURCES * N_POINTS
    print(f"wrote {outfile} ({n} points, {outfile.stat().st_size / 1024:.0f} KiB)")


def summarize(path: Path = DATA / "penny_volume_golden.json") -> None:
    """Header, array shapes, and the per-source *input* parameters as a table --
    a human-readable view of the compact blob without needing MATLAB."""
    if not path.is_file():
        sys.exit(f"golden file not found at {path}")
    data = json.loads(path.read_text())

    print(f"{path.name}  ({path.stat().st_size / 1024:.0f} KiB)")
    print(data["_comment"])
    print()
    print("header")
    for k in ("poisson_ratio", "shear_modulus", "nis", "seed",
              "n_sources", "n_points"):
        print(f"  {k:14s} {data[k]}")

    print()
    print("array shapes")
    for k in ("x_obs", "ur", "uz"):
        print(f"  {k:6s} {np.asarray(data[k]).shape}")

    print()
    print("per-source inputs (h = depth/radius; depth/radius in km, p in MPa)")
    header = ["idx", "h", "depth", "radius", "p_MPa"]
    widths = [3, 6, 7, 7, 8]
    print("  " + " ".join(h.rjust(w) for h, w in zip(header, widths)))
    for i in range(data["n_sources"]):
        depth_i, radius_i = data["depth"][i], data["radius"][i]
        row = [
            f"{i:d}",
            f"{depth_i / radius_i:.2f}",
            f"{depth_i / 1e3:.2f}",
            f"{radius_i / 1e3:.2f}",
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
