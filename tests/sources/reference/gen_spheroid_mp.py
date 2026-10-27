#!/usr/bin/env python3
"""Generate a 50-digit ground truth for ``SpheroidSource`` near its singular
limits (nearly vertical axis, nearly spherical shape).

``spheroid.m``'s general expressions lose precision there in ``float64`` -- as
``1/cos(dip)^2`` toward vertical, and as ``1/c^2`` toward a sphere where the two
foci merge -- so MATLAB cannot referee ``SpheroidSource``'s interpolation bands.
This script instead evaluates the *unmodified* surface expressions of
``spheroid.m`` (the general ``ct ~= 0`` branch of its ``evaluate``, at ``x3 = 0``)
in ``mpmath`` at 50 significant digits, where the cancellations are harmless,
and freezes the result as ``../data/spheroid_mp_golden.json``.

It is a transcription of ``cervelli/spheroid.m``, independent of the
reformulated port in ``torchdeform.sources.spheroid``: the original
``log(R1 + r3bar)`` / ``atan(.../(ct y2))`` forms are kept. Points exactly on the
strike line (where that ``atan`` divides by zero), and the point where the polar
axis pierces the surface (where ``R1 + r3bar`` underflows the 50 digits once the
coordinates' own round-off sets the off-axis distance), are therefore offset by
1 mm.
Exactly vertical and exactly spherical cases are excluded for the same reason;
those use closed forms checked elsewhere (``spheroid.m``'s vertical branch in the
MATLAB fixture, Mogi for the sphere).

Conventions as in ``gen_spheroid_volume.py``: the JSON stores radians and the
``SpheroidSource`` sign (positive strength inflating).

Regenerate::

    python gen_spheroid_mp.py            # needs mpmath (no MATLAB)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

NU = 0.25
MU = 3.0e10
SEED = 20261027
N_POINTS = 10
mp.mp.dps = 50

# (dip offset from vertical in degrees, a/b - 1): near-vertical, near-sphere, both
CASES = (
    [(dd, 1.0) for dd in (1e-3, -5e-2, 3e-1, -5.5e-1, 5.9e-1, -6.5e-1, 1.0)]
    + [(30.0, e) for e in (1e-6, -1e-4, 3e-3, -9e-3, 1.1e-2)]
    + [(-62.0, e) for e in (-1e-6, 5e-3)]
    + [(dd, e) for dd, e in ((2e-2, 4e-3), (-0.4, -6e-3), (0.3, 2e-5))]
    # shallow axes: piercing point tens to hundreds of km out
    + [(dd, 1.0) for dd in (-85.0, 88.5, -89.3)]
)


def spheroid_mp(x, y, X1, X2, depth, a, b, dip_deg, strike_deg, strength, nu, mu, mode):
    """``spheroid.m`` surface displacement, general branch, in mpmath (its sign)."""
    x, y, X1, X2, depth, a, b, strength, nu, mu = map(
        mp.mpf, (x, y, X1, X2, depth, a, b, strength, nu, mu))
    dip = mp.radians(mp.mpf(dip_deg) % 180)
    ct, st = mp.cos(dip), mp.sin(dip)
    strike = mp.radians(90 - mp.mpf(strike_deg))
    cs, ss = mp.cos(strike), mp.sin(strike)
    C0 = 4 * (1 - nu); C1 = C0 * (1 - 2 * nu); C2 = C0 - 1; C3 = C0 + 1
    c = mp.sqrt(mp.mpc(a * a - b * b)); b2 = b * b
    L0 = mp.log((a - c) / (a + c))
    d = 1 / (b2 * c * (2 * b2 - C0 * a ** 2) * L0 - a * c ** 2 * (4 * b2 + C0 * a ** 2)
             + a * b ** 4 * (1 + nu) * L0 ** 2)
    a1 = d * b2 * (2 * c * (a ** 2 + 2 * b2) + 3 * a * b2 * L0)
    b1 = d * (a ** 2 * c * C0 + b2 * (c + C3 * (c + a * L0)))
    C5 = strength / (4 * mu)
    if mode == "volume":
        C5 = strength * 3 / (8 * b2 * mp.pi * (-2 * b1 * c ** 3 - a1 * (a * L0 * (1 - 2 * nu) + c * C3)))
    dx, dy = x - X1, y - X2
    x1 = cs * dx + ss * dy; x2 = -ss * dx + cs * dy
    x3 = mp.mpf(0); X3 = -depth
    x3hat = x3 - X3; x3bar = x3 + X3
    r2 = x1 * st + x3hat * ct; r3 = x1 * ct - x3hat * st
    q2 = x1 * st - x3bar * ct; q3 = -x1 * ct - x3bar * st

    def evaluate(Z):
        y1 = x1 - Z * ct; y2 = x2; y3 = -x3hat - Z * st; y3bar = -x3bar + Z * st
        r3bar = r3 - Z; q3bar = q3 + Z
        rho2 = y1 ** 2 + y2 ** 2
        R1 = mp.sqrt(rho2 + y3 ** 2); R2 = mp.sqrt(rho2 + y3bar ** 2)
        R1r3 = R1 + r3bar; R2q3 = R2 + q3bar; R2y3 = R2 + y3bar
        L1 = mp.log(R1r3)
        A1star = a1 / (R1 * R1r3) + b1 * (L1 + (r3 + Z) / R1r3)
        L2 = mp.log(R2q3); L3 = mp.log(R2y3)
        A1 = Z / R1 + L1; A1bar = Z / R2 - L2
        A2 = R1 - r3 * L1; A2bar = R2 - q3 * L2
        A3 = Z * r3bar / R1 + R1; A3bar = Z * q3bar / R2 - R2
        A1starbar = -a1 / (R2 * R2q3) - b1 * (L2 + (q3 - Z) / R2q3)
        phi = mp.atan(-(ct * q2 + (1 + st) * R2q3) / (ct * y2))
        f1 = C1 * (Z * y1 / R2y3 - 6 * st * y2 * phi / ct ** 2
                   + 3 / ct ** 2 * (q2 * (st * L2 - L3) + ct * (R2 - y3bar))
                   - 2 * ct * A2bar - 2 / ct * (x3bar * L3 + q3 * L2))
        f2 = C1 * (Z * y2 / R2y3 - 6 * q2 * phi / ct ** 2 - 3 * y2 / ct ** 2 * (L2 - st * L3)
                   + 2 * y2 * L2 - 4 * x3bar / ct * phi)
        f3 = C1 * (-2 * y2 * phi / ct + 2 * st * A2bar + q3 * L3 - Z + q2 / ct * (L2 - st * L3))
        # x3 = 0: the F1star / F2star / B4 terms and the x3 parts of F1, F2 vanish
        F1 = -C2 * A1bar
        F2 = -C2 * A3bar
        Bstar = 2 * A2 * b1 + a1 / R1 + C2 * (2 * A2bar * b1 + a1 / R2)
        B1 = -2 * C2 * (X3 * (Z / R2 - L2) + st * (A2bar - Z ** 2 / R2))
        B2 = A3 + C0 * (A2 + A2bar)
        B3 = A1star * r2 + C2 * q2 * A1starbar
        U1 = b2 * (st * B3 + ct * Bstar) + a1 * (st * (q2 * F1 - A1 * r2) - ct * (F2 + B2) + f1)
        U2 = b2 * y2 * (A1star + C2 * A1starbar) - a1 * (y2 * (A1 - F1) - f2)
        U3 = b2 * (ct * B3 - st * Bstar) + a1 * (-ct * (q2 * F1 + A1 * r2) - st * (F2 - B2) - B1 + f3)
        return U1, U2, U3

    p, m = evaluate(c), evaluate(-c)
    u = [mp.re(C5 * (pp - mm)) for pp, mm in zip(p, m)]
    return [float(cs * u[0] - ss * u[1]), float(ss * u[0] + cs * u[1]), float(u[2])]


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = []
    for dip_off, rel in CASES:
        b = float(rng.uniform(400.0, 1200.0))
        a = b * (1.0 + rel)
        dip_deg = 90.0 + dip_off
        depth = float(rng.uniform(3e3, 7e3))
        strike_deg = float(rng.uniform(0.0, 360.0))
        X1, X2 = (float(v) for v in rng.uniform(-1e3, 1e3, 2))
        reach = 3.0 * depth
        xs = list(X1 + rng.uniform(-reach, reach, N_POINTS - 2))
        ys = list(X2 + rng.uniform(-reach, reach, N_POINTS - 2))
        th, az = math.radians(dip_deg), math.radians(strike_deg)
        # axis piercing point and a point on the strike line, both offset by 1 mm
        for along in (-depth / math.tan(th), rng.uniform(-reach, reach)):
            off = 1e-3
            xs.append(X1 + along * math.sin(az) - off * math.cos(az))
            ys.append(X2 + along * math.cos(az) + off * math.sin(az))
        case = dict(source_x=X1, source_y=X2, depth=depth, polar_radius=a,
                    equatorial_radius=b, dip=math.radians(dip_deg),
                    strike=math.radians(strike_deg), pressure=2e6, delta_v=3e6,
                    x_obs=xs, y_obs=ys)
        for mode, key in (("pressure", "pressure"), ("volume", "delta_v")):
            # spheroid.m's sign: negative strength inflates
            case[f"u_enu_{mode}"] = [
                spheroid_mp(x, y, X1, X2, depth, a, b, dip_deg, strike_deg,
                            -case[key], NU, MU, mode)
                for x, y in zip(xs, ys)
            ]
        cases.append(case)

    payload = {
        "_comment": (
            "50-digit ground truth for SpheroidSource near vertical / spherical, "
            "from an mpmath transcription of spheroid.m's surface expressions. "
            "Generated by tests/sources/reference/gen_spheroid_mp.py. Do not edit "
            "by hand."
        ),
        "poisson_ratio": NU,
        "shear_modulus": MU,
        "seed": SEED,
        "mp_dps": mp.mp.dps,
        "cases": cases,
    }
    DATA.mkdir(exist_ok=True)
    outfile = DATA / "spheroid_mp_golden.json"
    outfile.write_text(json.dumps(payload))
    print(f"wrote {outfile} ({len(cases)} cases x {N_POINTS} points x 2 modes)")


if __name__ == "__main__":
    main()
