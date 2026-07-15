"""Cross-validation of torchdeform source models against VMOD (uafgeotools).

This is a **manual** test suite -- it is *not* part of CI. It compares the
surface displacement fields produced by torchdeform's analytic source models
against the independent `VMOD <https://github.com/uafgeotools/vmod>`_ geodesy
package, over ~1000 random parameter combinations per model. The goal is a
high-confidence, third-party check that our forward models are numerically
correct (not just internally self-consistent).

Running it
----------
The suite is skipped unless ``RUN_VMOD_TESTS=1`` is set, so it never runs in a
plain ``pytest`` invocation / CI::

    RUN_VMOD_TESTS=1 pytest tests/external/test_vmod_comparison.py -v -s

VMOD must be importable. Either ``pip install vmod-geodesy`` (pulls in utm,
rasterio, hankel, ...), or point ``VMOD_PATH`` at a checkout of the repo::

    RUN_VMOD_TESTS=1 VMOD_PATH="scaffolding/Link to vmod" pytest tests/external/... -s

Knobs (all optional env vars):

* ``VMOD_N_SAMPLES``  -- random parameter sets per model (default 1000).
* ``VMOD_SEED``       -- base RNG seed (default 0).

Convention bridging
-------------------
torchdeform and VMOD implement the same physics but with different internal
parameter conventions. Each adapter below maps a torchdeform parameter set to
the equivalent VMOD call; the mappings were pinned down empirically to machine
precision on non-singular grids (see the module for details). The notable ones:

* **Okada** -- VMOD's fault frame is the mirror image of ours across the strike
  line (its down-dip axis has the opposite sign), so we reflect the observation
  points across the strike line, evaluate VMOD, then reflect the horizontal
  output back. VMOD's ``model`` also cannot carry slip *and* opening in one
  call, so a mixed dislocation is the sum of a ``type='slip'`` and a
  ``type='open'`` evaluation (Okada is linear in slip).
* **CDM** -- we build our rotation matrix (``Rz @ Ry @ Rx``) and re-extract the
  Euler angles in VMOD's order (``Rx @ Ry @ Rz``); the per-axis matrices are
  otherwise identical.
* **Penny** -- VMOD's ``dP`` is our ``pressure`` and its ``mu`` is our
  ``shear_modulus``. This is the only model that doesn't agree to ~machine
  precision (~1e-3 rel instead of ~1e-13): the Fialko penny crack has no
  elementary closed form, and the two libraries evaluate its semi-infinite
  Hankel transform differently -- see ``test_penny_vs_vmod`` for the full
  explanation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# --------------------------------------------------------------------------- #
# Gating: manual-only + VMOD availability
# --------------------------------------------------------------------------- #
if os.environ.get("RUN_VMOD_TESTS") != "1":
    pytest.skip(
        "VMOD comparison is a manual suite; set RUN_VMOD_TESTS=1 to run it.",
        allow_module_level=True,
    )


def _load_vmod():
    """Import VMOD, falling back to a local checkout via ``VMOD_PATH``."""
    try:
        import vmod  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    if os.environ.get("VMOD_PATH"):
        candidates.append(Path(os.environ["VMOD_PATH"]))
    # Default: the vendored checkout under scaffolding (repo root is 2 up).
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "scaffolding" / "Link to vmod")
    for c in candidates:
        if (c / "vmod" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            return
    pytest.skip(
        "VMOD not importable. `pip install vmod-geodesy` or set VMOD_PATH to a "
        "VMOD checkout.",
        allow_module_level=True,
    )


_load_vmod()

try:
    from vmod.source.mogi import Mogi
    from vmod.source.okada import Okada
    from vmod.source.penny import Penny
    from vmod.source.cdm import Cdm
except ImportError as exc:  # pragma: no cover - dependency issue
    pytest.skip(f"VMOD import failed ({exc}).", allow_module_level=True)

from torchdeform.sources.mogi import MogiSource
from torchdeform.sources.okada import OkadaSource
from torchdeform.sources.penny import PennySource
from torchdeform.sources.cdm import CDMSource

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
N_SAMPLES = int(os.environ.get("VMOD_N_SAMPLES", "1000"))
BASE_SEED = int(os.environ.get("VMOD_SEED", "0"))
POISSON = 0.25          # both libraries default to nu = 0.25
SHEAR_MODULUS = 30e9    # penny: our shear modulus == VMOD's mu

# Observation grid: 11x11 spanning +/-15 km, nudged by an irrational-ish offset
# so points never land exactly on a fault edge / source (avoids singularities).
_G = np.linspace(-15000.0, 15000.0, 11)
_GX, _GY = np.meshgrid(_G, _G)
XOBS = _GX.ravel() + 137.13     # [Npts]
YOBS = _GY.ravel() - 89.71
NPTS = XOBS.size


def _vinst(cls):
    """A VMOD source instance without constructing a Data object.

    VMOD's ``model()`` methods are effectively standalone (they don't touch
    ``self.data``), so we bypass ``__init__`` to avoid its heavy dependencies.
    """
    return cls.__new__(cls)


def _rel_field_error(ours: np.ndarray, vmod: np.ndarray) -> float:
    """Max abs component difference normalised by the field's peak amplitude."""
    denom = np.max(np.abs(vmod)) + 1e-12
    return float(np.max(np.abs(ours - vmod)) / denom)


def _report(name: str, errs: np.ndarray, tol: float) -> None:
    q = np.quantile(errs, [0.5, 0.9, 0.99, 1.0])
    print(
        f"\n[{name}] N={errs.size}  median={q[0]:.2e}  p90={q[1]:.2e}  "
        f"p99={q[2]:.2e}  max={q[3]:.2e}  (tol={tol:.1e})"
    )
    n_bad = int(np.sum(errs > tol))
    if n_bad:
        print(f"    {n_bad} sample(s) exceed tol; worst={errs.max():.3e}")


# --------------------------------------------------------------------------- #
# Mogi
# --------------------------------------------------------------------------- #
def _run_mogi() -> np.ndarray:
    rng = np.random.default_rng(BASE_SEED + 1)
    xc = rng.uniform(-500, 500, N_SAMPLES)
    yc = rng.uniform(-500, 500, N_SAMPLES)
    depth = rng.uniform(1000, 10000, N_SAMPLES)
    dV = rng.uniform(-5e6, 5e6, N_SAMPLES)

    xt = torch.tensor(np.tile(XOBS, (N_SAMPLES, 1)))
    yt = torch.tensor(np.tile(YOBS, (N_SAMPLES, 1)))
    disp = MogiSource(poisson_ratio=POISSON)(
        xt, yt, torch.tensor(xc), torch.tensor(yc),
        torch.tensor(depth), torch.tensor(dV),
    )
    OE, ON, OU = disp.e.numpy(), disp.n.numpy(), disp.u.numpy()

    m = _vinst(Mogi)
    errs = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        ve, vn, vu = m.model(XOBS, YOBS, xc[i], yc[i], depth[i], dV[i], nu=POISSON)
        ours = np.stack([OE[i], ON[i], OU[i]])
        vmod = np.stack([ve, vn, vu])
        errs[i] = _rel_field_error(ours, vmod)
    return errs


def test_mogi_vs_vmod():
    errs = _run_mogi()
    tol = 1e-10
    _report("Mogi", errs, tol)
    assert errs.max() < tol


# --------------------------------------------------------------------------- #
# Okada
# --------------------------------------------------------------------------- #
def _strike_line_reflection(strike_rad: float) -> np.ndarray:
    """2x2 reflection across the strike line (direction (sin, cos) in E/N)."""
    c2, s2 = np.cos(2 * strike_rad), np.sin(2 * strike_rad)
    return np.array([[-c2, s2], [s2, c2]])


def _okada_vmod(strike_deg, dip_deg, d1, d2, d3, xc, yc, depth, L, W):
    """VMOD Okada evaluated in torchdeform's convention.

    Reflect obs across the strike line, sum the slip + opening contributions
    (Okada is linear; VMOD can't do both in one call), reflect output back.
    """
    R = _strike_line_reflection(np.radians(strike_deg))
    P = np.stack([XOBS - xc, YOBS - yc])
    Pp = R @ P
    xm, ym = Pp[0] + xc, Pp[1] + yc

    tot = np.zeros((3, NPTS))
    if d1 != 0.0 or d2 != 0.0:
        o = _vinst(Okada)
        o.type = "slip"
        slip = float(np.hypot(d1, d2))
        rake = float(np.degrees(np.arctan2(d2, d1)))
        tot = tot + np.asarray(
            o.model(xm, ym, xc, yc, depth, L, W, slip, strike_deg % 360, dip_deg, rake)
        )
    if d3 != 0.0:
        o = _vinst(Okada)
        o.type = "open"
        tot = tot + np.asarray(
            o.model(xm, ym, xc, yc, depth, L, W, d3, strike_deg % 360, dip_deg)
        )
    eh = R @ np.stack([tot[0], tot[1]])
    return np.stack([eh[0], eh[1], tot[2]])


def _run_okada() -> np.ndarray:
    rng = np.random.default_rng(BASE_SEED + 2)
    strike = rng.uniform(0, 360, N_SAMPLES)
    dip = rng.uniform(5, 89, N_SAMPLES)
    L = rng.uniform(2000, 8000, N_SAMPLES)
    W = rng.uniform(1500, 5000, N_SAMPLES)
    # keep the fault buried: depth > W/2 * sin(dip) (both libs require this)
    min_depth = 0.5 * W * np.sin(np.radians(dip))
    depth = min_depth + rng.uniform(500, 6000, N_SAMPLES)
    xc = rng.uniform(-500, 500, N_SAMPLES)
    yc = rng.uniform(-500, 500, N_SAMPLES)
    d1 = rng.uniform(-2, 2, N_SAMPLES)     # strike-slip
    d2 = rng.uniform(-2, 2, N_SAMPLES)     # dip-slip
    d3 = rng.uniform(-1, 1, N_SAMPLES)     # tensile / opening

    xt = torch.tensor(np.tile(XOBS, (N_SAMPLES, 1)))
    yt = torch.tensor(np.tile(YOBS, (N_SAMPLES, 1)))
    zt = torch.zeros_like(xt)
    disp = OkadaSource(poisson_ratio=POISSON)(
        xt, yt, zt,
        torch.tensor(xc), torch.tensor(yc),
        torch.tensor(np.radians(dip)), torch.tensor(np.radians(strike)),
        torch.tensor(depth), torch.tensor(L), torch.tensor(W),
        torch.tensor(d1), torch.tensor(d2), torch.tensor(d3),
    )
    OE, ON, OU = disp.e.numpy(), disp.n.numpy(), disp.u.numpy()

    errs = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        vmod = _okada_vmod(
            strike[i], dip[i], d1[i], d2[i], d3[i],
            xc[i], yc[i], depth[i], L[i], W[i],
        )
        ours = np.stack([OE[i], ON[i], OU[i]])
        errs[i] = _rel_field_error(ours, vmod)
    return errs


def test_okada_vs_vmod():
    errs = _run_okada()
    # analytic model; residual is dominated by obs points nearest the fault
    tol = 1e-4
    _report("Okada", errs, tol)
    assert np.quantile(errs, 0.99) < tol
    assert errs.max() < 5 * tol


# --------------------------------------------------------------------------- #
# Penny-shaped crack
# --------------------------------------------------------------------------- #
def _run_penny() -> np.ndarray:
    rng = np.random.default_rng(BASE_SEED + 3)
    xc = rng.uniform(-500, 500, N_SAMPLES)
    yc = rng.uniform(-500, 500, N_SAMPLES)
    # Restrict to the deeper, moderate-radius regime typical of magmatic sources.
    #
    # Penny is the one model where the two libraries genuinely differ (~1e-3 rel,
    # vs machine precision for the closed-form models). Fialko's penny crack has
    # no elementary closed form: it needs a Fredholm solve for the auxiliary
    # functions plus a *semi-infinite* Hankel transform,
    # int_0^inf ... e^{-xi h} J_n(xi r) dxi  (h = depth/radius).
    #   - torchdeform evaluates that infinite integral in closed form (Fialko's
    #     S/C improper integrals, appendix B) and is validated against the
    #     original Fialko MATLAB reference to rtol 1e-6 (test_penny_source.py).
    #   - VMOD evaluates it *numerically* -- Gauss-Legendre, 41+41 nodes,
    #     truncated at xi=60. So the gap is essentially VMOD's truncation +
    #     quadrature error, largest for shallow/wide cracks (small h -> the
    #     e^{-xi h} tail decays slowly past xi=60) and far-field points (J_n(xi r)
    #     oscillates faster than the fixed node set resolves). We sample the
    #     deeper regime to keep that error in the sub-percent range.
    depth = rng.uniform(4000, 9000, N_SAMPLES)
    radius = rng.uniform(0.2, 0.75, N_SAMPLES) * depth
    pressure = rng.uniform(-20e6, 20e6, N_SAMPLES)

    xt = torch.tensor(np.tile(XOBS, (N_SAMPLES, 1)))
    yt = torch.tensor(np.tile(YOBS, (N_SAMPLES, 1)))
    disp = PennySource(poisson_ratio=POISSON, shear_modulus=SHEAR_MODULUS)(
        xt, yt, torch.tensor(xc), torch.tensor(yc),
        torch.tensor(depth), torch.tensor(radius), torch.tensor(pressure),
    )
    OE, ON, OU = disp.e.numpy(), disp.n.numpy(), disp.u.numpy()

    p = _vinst(Penny)
    errs = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        ve, vn, vu = p.model(
            XOBS, YOBS, xc[i], yc[i], depth[i], pressure[i], radius[i],
            mu=SHEAR_MODULUS, nu=POISSON,
        )
        ours = np.stack([OE[i], ON[i], OU[i]])
        vmod = np.stack([ve, vn, vu])
        errs[i] = _rel_field_error(ours, vmod)
    return errs


def test_penny_vs_vmod():
    errs = _run_penny()
    # ~1e-3 rel = VMOD's numerical Hankel transform vs our closed-form one; in
    # absolute terms ~0.1% of the peak (e.g. 0.17 mm on a 168 mm signal).
    tol = 5e-3
    _report("Penny", errs, tol)
    assert np.quantile(errs, 0.99) < tol
    assert errs.max() < 2e-2


# --------------------------------------------------------------------------- #
# Compound Dislocation Model (finite CDM)
# --------------------------------------------------------------------------- #
def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _cdm_euler_to_vmod(ox, oy, oz):
    """torchdeform angles (Rz@Ry@Rx) -> VMOD angles (Rx@Ry@Rz), in degrees.

    The per-axis matrices are identical between the libraries; only the
    composition order differs, so we rebuild our matrix and re-decompose it in
    VMOD's order.
    """
    M = _Rz(oz) @ _Ry(oy) @ _Rx(ox)
    b = np.arcsin(-np.clip(M[0, 2], -1.0, 1.0))
    a = np.arctan2(M[1, 2], M[2, 2])
    c = np.arctan2(M[0, 1], M[0, 0])
    return np.degrees(a), np.degrees(b), np.degrees(c)


def _run_cdm() -> np.ndarray:
    rng = np.random.default_rng(BASE_SEED + 4)
    xc = rng.uniform(-500, 500, N_SAMPLES)
    yc = rng.uniform(-500, 500, N_SAMPLES)
    depth = rng.uniform(3000, 9000, N_SAMPLES)
    ox = rng.uniform(-np.pi, np.pi, N_SAMPLES)   # radians (our convention)
    oy = rng.uniform(-np.pi, np.pi, N_SAMPLES)
    oz = rng.uniform(-np.pi, np.pi, N_SAMPLES)
    ax = rng.uniform(100, 600, N_SAMPLES)        # semi-axes
    ay = rng.uniform(100, 600, N_SAMPLES)
    az = rng.uniform(100, 600, N_SAMPLES)
    opening = rng.uniform(-1, 1, N_SAMPLES)

    xt = torch.tensor(np.tile(XOBS, (N_SAMPLES, 1)))
    yt = torch.tensor(np.tile(YOBS, (N_SAMPLES, 1)))
    disp = CDMSource(poisson_ratio=POISSON)(
        xt, yt, torch.tensor(xc), torch.tensor(yc), torch.tensor(depth),
        torch.tensor(ox), torch.tensor(oy), torch.tensor(oz),
        torch.tensor(ax), torch.tensor(ay), torch.tensor(az),
        torch.tensor(opening),
    )
    OE, ON, OU = disp.e.numpy(), disp.n.numpy(), disp.u.numpy()

    errs = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        wa, wb, wc = _cdm_euler_to_vmod(ox[i], oy[i], oz[i])
        o = _vinst(Cdm)
        vmod = np.asarray(
            o.model(XOBS, YOBS, xc[i], yc[i], depth[i], wa, wb, wc,
                    ax[i], ay[i], az[i], opening[i], nu=POISSON)
        )
        ours = np.stack([OE[i], ON[i], OU[i]])
        errs[i] = _rel_field_error(ours, vmod)
    return errs


def test_cdm_vs_vmod():
    errs = _run_cdm()
    tol = 1e-4
    _report("CDM", errs, tol)
    assert np.quantile(errs, 0.99) < tol
    assert errs.max() < 5 * tol


# --------------------------------------------------------------------------- #
# Registry consumed by the accuracy-table emitter (emit_accuracy_table.py).
# Order matches the README table. Each entry: (torchdeform class, VMOD class,
# error-array producer, qualitative note). Keeping this next to the tests means
# the emitted table measures exactly what the tests measure -- one driver, no
# duplicated sampling/adapter logic to drift.
# --------------------------------------------------------------------------- #
MODELS = [
    ("MogiSource", "Mogi", _run_mogi, "closed-form, machine precision"),
    ("OkadaSource", "Okada", _run_okada, "closed-form; max is 1 near-fault point"),
    ("CDMSource", "Cdm", _run_cdm, "closed-form; max is 1 near-source point"),
    ("PennySource", "Penny", _run_penny, "numerical Hankel transform (see below)"),
]
