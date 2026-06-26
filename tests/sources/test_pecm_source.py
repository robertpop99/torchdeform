"""
Tests for ``PECMSource`` (point Ellipsoidal Cavity Model) and ``ecm_potencies``.

Focus areas:

* **Elliptic integrals** - the batched Carlson ``RF``/``RD`` match SciPy's
  independent implementation to machine precision (optional, skipped without
  SciPy).
* **Correctness** - a *spherical* cavity is a centre of dilatation: radial,
  rotation-invariant and Mogi-shaped. The Eshelby step is continuous across the
  oblate / prolate / spherical branch boundaries and invariant to axis
  relabelling. Displacement is linear in pressure; positive pressure uplifts.
  A full pECM equals a pCDM driven by the potencies it computes.
* **Batchability** - a batched call equals a per-item loop.
* **Differentiability** - ``gradcheck`` passes (triaxial, away from branch
  boundaries) and gradients stay finite.
* **Validation / dtype / device.**

Run with::

    pytest test_pecm_source.py -v
"""
import pytest
import torch

from torchdeform import PECMSource, PCDMSource, MogiSource, Displacement
from torchdeform.sources.pecm import ecm_potencies, _carlson_rf, _carlson_rd


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

NU = 0.25
MU = 3.0e10
K = 2.0 * MU * (1.0 + NU) / (3.0 * (1.0 - 2.0 * NU))


def _grid(B, n=21, extent=20_000.0, dtype=DTYPE):
    ax = torch.linspace(-extent / 2, extent / 2, n, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x = xx.reshape(1, -1).expand(B, -1).contiguous()
    y = yy.reshape(1, -1).expand(B, -1).contiguous()
    return x, y


def _f(B, v, dtype=DTYPE):
    return torch.full((B,), float(v), dtype=dtype)


def _src(**kw):
    return PECMSource(poisson_ratio=NU, shear_modulus=MU, **kw)


# --------------------------------------------------------------------------- #
# Carlson elliptic integrals vs SciPy
# --------------------------------------------------------------------------- #
def test_carlson_matches_scipy():
    sp = pytest.importorskip("scipy.special")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(64, generator=g, dtype=DTYPE) * 5 + 1e-2
    y = torch.rand(64, generator=g, dtype=DTYPE) * 5 + 1e-2
    z = torch.rand(64, generator=g, dtype=DTYPE) * 5 + 1e-2
    rf = _carlson_rf(x, y, z)
    rd = _carlson_rd(x, y, z)
    rf_ref = torch.as_tensor(sp.elliprf(x.numpy(), y.numpy(), z.numpy()))
    rd_ref = torch.as_tensor(sp.elliprd(x.numpy(), y.numpy(), z.numpy()))
    assert torch.allclose(rf, rf_ref, rtol=1e-12, atol=0)
    assert torch.allclose(rd, rd_ref, rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- #
# Basic
# --------------------------------------------------------------------------- #
def test_output_shape_and_finite():
    B, n = 3, 11
    x, y = _grid(B, n)
    out = _src()(
        x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
        omega_x=_f(B, 0.3), omega_y=_f(B, -0.4), omega_z=_f(B, 1.0),
        a_x=_f(B, 900), a_y=_f(B, 600), a_z=_f(B, 300), pressure=_f(B, 8e6),
    )
    assert isinstance(out, Displacement)
    assert out.e.shape == (B, n * n)
    for t in (out.e, out.n, out.u):
        assert torch.isfinite(t).all()


def test_gradients_finite_for_sphere_and_spheroids():
    """At an exact sphere (a_x = a_y = a_z) the oblate branch's ``acos(a3/a1)`` and
    the prolate branch's ``acosh(a1/a3)`` hit their infinite-slope point at +/-1
    while being discarded by the spherical override -- a naive backward returns
    NaN for the semi-axis gradients. Guard finiteness for the sphere and both
    spheroid degeneracies."""
    B, n = 2, 9
    x, y = _grid(B, n)
    for axes in ((500, 500, 500), (500, 500, 300), (500, 300, 300)):
        params = {
            "source_x": _f(B, 0), "source_y": _f(B, 0), "depth": _f(B, 4000),
            "omega_x": _f(B, 0.3), "omega_y": _f(B, 0.2), "omega_z": _f(B, 0.6),
            "a_x": _f(B, axes[0]), "a_y": _f(B, axes[1]), "a_z": _f(B, axes[2]),
            "pressure": _f(B, 8e6),
        }
        for p in params.values():
            p.requires_grad_(True)
        out = _src()(x, y, **params)
        (out.e.sum() + out.n.sum() + out.u.sum()).backward()
        for name, p in params.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (axes, name)


# --------------------------------------------------------------------------- #
# Spherical cavity == centre of dilatation
# --------------------------------------------------------------------------- #
def _sphere(B, x, y, a=500.0, depth=4000.0, p=10e6, omega=(0.0, 0.0, 0.0)):
    z0 = torch.zeros(B, dtype=DTYPE)
    return _src()(x, y, z0, z0, _f(B, depth),
                  _f(B, omega[0]), _f(B, omega[1]), _f(B, omega[2]),
                  _f(B, a), _f(B, a), _f(B, a), _f(B, p))


def test_sphere_is_radial_and_rotation_invariant():
    B = 1
    x, y = _grid(B)
    d0 = _sphere(B, x, y, omega=(0.0, 0.0, 0.0))
    d1 = _sphere(B, x, y, omega=(0.7, -1.1, 2.0))
    # rotation invariant
    assert torch.allclose(d0.u, d1.u, atol=1e-9)
    assert torch.allclose(d0.e, d1.e, atol=1e-9)
    # purely radial horizontal field
    assert (d0.e * y - d0.n * x).abs().max() < 1e-6


def test_sphere_matches_mogi_shape():
    B = 1
    x, y = _grid(B)
    d = _sphere(B, x, y)
    z0 = torch.zeros(B, dtype=DTYPE)
    mg = MogiSource()(x, y, z0, z0, _f(B, 4000), delta_v=_f(B, 1.0))

    def corr(a, b):
        a = a.flatten() - a.flatten().mean()
        b = b.flatten() - b.flatten().mean()
        return (a @ b / (a.norm() * b.norm())).item()

    assert corr(d.u, mg.u) > 0.9999
    assert corr(d.e, mg.e) > 0.9999


def test_positive_pressure_uplifts():
    B, n = 1, 21
    x, y = _grid(B, n)
    d = _sphere(B, x, y)
    assert d.u[0, n * n // 2] > 0


def test_pressure_linearity_and_sign():
    B = 1
    x, y = _grid(B)
    common = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                  omega_x=_f(B, 0.3), omega_y=_f(B, -0.5), omega_z=_f(B, 1.1),
                  a_x=_f(B, 900), a_y=_f(B, 600), a_z=_f(B, 300))
    d = _src()(x, y, pressure=_f(B, 8e6), **common)
    d2 = _src()(x, y, pressure=_f(B, 16e6), **common)
    dn = _src()(x, y, pressure=_f(B, -8e6), **common)
    assert torch.allclose(d2.u, 2 * d.u, atol=1e-9)
    assert torch.allclose(dn.u, -d.u, atol=1e-9)


# --------------------------------------------------------------------------- #
# Eshelby step: branch continuity, permutation invariance, integration
# --------------------------------------------------------------------------- #
def _pot(ax, ay, az, p=10e6):
    return ecm_potencies(_f(1, ax), _f(1, ay), _f(1, az), _f(1, p), NU, K)[0]


def test_branch_continuity():
    """No jump at the branch boundary: a point just inside the degenerate branch
    matches a point just inside the triaxial branch (same near-degenerate
    ellipsoid, different formula). BRANCH_TOL on the relative axis gap is 1e-6."""
    # across the oblate boundary (rel12 ~ 1e-6)
    assert torch.allclose(_pot(1000, 999.9995, 400), _pot(1000, 999.998, 400), rtol=1e-4)
    # across the prolate boundary (rel23 ~ 1e-6)
    assert torch.allclose(_pot(1000, 400.0005, 400), _pot(1000, 400.002, 400), rtol=1e-4)
    # across the spherical boundary
    assert torch.allclose(_pot(500, 500.0002, 499.9998), _pot(500, 500.002, 499.998), rtol=1e-4)
    # exact degenerate geometries stay finite
    for v in (_pot(1000, 1000, 400), _pot(1000, 400, 400), _pot(500, 500, 500)):
        assert torch.isfinite(v).all()


def test_axis_permutation_invariance():
    base = _pot(1000, 700, 300)            # (DVx, DVy, DVz)
    perm = _pot(300, 1000, 700)            # axes -> (z, x, y)
    # potency follows its axis: perm = (base_z, base_x, base_y)
    assert torch.allclose(perm, base[[2, 0, 1]], rtol=1e-10)


def test_sphere_potencies_isotropic():
    v = _pot(500, 500, 500)
    assert torch.allclose(v, v[0].expand(3), rtol=1e-10)


def test_matches_pcdm_with_computed_potencies():
    """A full pECM equals a pCDM driven by the potencies pECM computes."""
    B = 1
    x, y = _grid(B, 11)
    z0 = torch.zeros(B, dtype=DTYPE)
    args = dict(omega_x=_f(B, 0.3), omega_y=_f(B, -0.5), omega_z=_f(B, 1.1))
    ax, ay, az, p = _f(B, 900), _f(B, 600), _f(B, 300), _f(B, 8e6)
    d = _src()(x, y, z0, z0, _f(B, 4000), a_x=ax, a_y=ay, a_z=az, pressure=p, **args)
    DV = ecm_potencies(ax, ay, az, p, NU, K)
    dp = PCDMSource(poisson_ratio=NU)(
        x, y, z0, z0, _f(B, 4000), dv_x=DV[:, 0], dv_y=DV[:, 1], dv_z=DV[:, 2], **args)
    assert torch.allclose(d.u, dp.u, atol=1e-12)
    assert torch.allclose(d.e, dp.e, atol=1e-12)


# --------------------------------------------------------------------------- #
# Batchability
# --------------------------------------------------------------------------- #
def test_batched_matches_loop():
    B, n = 4, 11
    x, y = _grid(B, n)
    g = torch.Generator().manual_seed(0)
    depth = 3000 + 5000 * torch.rand(B, generator=g, dtype=DTYPE)
    ox = torch.randn(B, generator=g, dtype=DTYPE)
    oy = torch.randn(B, generator=g, dtype=DTYPE)
    oz = torch.randn(B, generator=g, dtype=DTYPE)
    a_x = 400 + 700 * torch.rand(B, generator=g, dtype=DTYPE)
    a_y = 400 + 700 * torch.rand(B, generator=g, dtype=DTYPE)
    a_z = 200 + 400 * torch.rand(B, generator=g, dtype=DTYPE)
    p = (torch.rand(B, generator=g, dtype=DTYPE) - 0.5) * 2e7

    src = _src()
    full = src(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
               omega_x=ox, omega_y=oy, omega_z=oz,
               a_x=a_x, a_y=a_y, a_z=a_z, pressure=p)
    for b in range(B):
        one = src(x[b:b + 1], y[b:b + 1], source_x=_f(1, 0), source_y=_f(1, 0),
                  depth=depth[b:b + 1], omega_x=ox[b:b + 1], omega_y=oy[b:b + 1],
                  omega_z=oz[b:b + 1], a_x=a_x[b:b + 1], a_y=a_y[b:b + 1],
                  a_z=a_z[b:b + 1], pressure=p[b:b + 1])
        assert torch.allclose(full.e[b:b + 1], one.e, atol=1e-9)
        assert torch.allclose(full.u[b:b + 1], one.u, atol=1e-9)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_bad_shape_raises():
    x, y = _grid(2, 5)
    with pytest.raises(ValueError):
        _src()(x, y, source_x=_f(3, 0), source_y=_f(3, 0), depth=_f(3, 4000),
               omega_x=_f(3, 0), omega_y=_f(3, 0), omega_z=_f(3, 0),
               a_x=_f(3, 900), a_y=_f(3, 600), a_z=_f(3, 300), pressure=_f(3, 8e6))


# --------------------------------------------------------------------------- #
# Differentiability  (unit-scale, triaxial, away from branch boundaries)
# --------------------------------------------------------------------------- #
def test_gradcheck():
    B = 2
    ax = torch.linspace(-3.0, 3.0, 4, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x = xx.reshape(1, -1).expand(B, -1).contiguous()
    y = yy.reshape(1, -1).expand(B, -1).contiguous()

    sx = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    sy = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    depth = _f(B, 2.0).requires_grad_(True)
    ox = _f(B, 0.2).requires_grad_(True)
    oy = _f(B, -0.3).requires_grad_(True)
    oz = _f(B, 0.5).requires_grad_(True)
    ax_ = _f(B, 0.6).requires_grad_(True)
    ay_ = _f(B, 0.45).requires_grad_(True)
    az_ = _f(B, 0.25).requires_grad_(True)
    p = _f(B, 1e-3).requires_grad_(True)

    # unit shear modulus keeps the problem at unit scale for gradcheck's eps
    model = PECMSource(poisson_ratio=NU, shear_modulus=1.0)

    def fn(sx, sy, depth, ox, oy, oz, ax_, ay_, az_, p):
        d = model(x, y, source_x=sx, source_y=sy, depth=depth,
                  omega_x=ox, omega_y=oy, omega_z=oz,
                  a_x=ax_, a_y=ay_, a_z=az_, pressure=p)
        return torch.stack([d.e, d.n, d.u], dim=-1)

    assert torch.autograd.gradcheck(fn, (sx, sy, depth, ox, oy, oz, ax_, ay_, az_, p))


def test_gradients_finite_and_nonzero():
    B = 2
    x, y = _grid(B, 9)
    depth = _f(B, 4000).requires_grad_(True)
    p = _f(B, 8e6).requires_grad_(True)
    d = _src()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
               omega_x=_f(B, 0.3), omega_y=_f(B, 0.2), omega_z=_f(B, 1.0),
               a_x=_f(B, 900), a_y=_f(B, 600), a_z=_f(B, 300), pressure=p)
    d.u.pow(2).mean().backward()
    for t in (depth, p):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
def test_dtype_float32():
    B = 2
    x, y = _grid(B, 7, dtype=torch.float32)
    out = PECMSource(poisson_ratio=NU, shear_modulus=MU, internal_dtype=torch.float32)(
        x, y, source_x=_f(B, 0, torch.float32), source_y=_f(B, 0, torch.float32),
        depth=_f(B, 4000, torch.float32),
        omega_x=_f(B, 0.2, torch.float32), omega_y=_f(B, 0.1, torch.float32),
        omega_z=_f(B, 0.5, torch.float32),
        a_x=_f(B, 900, torch.float32), a_y=_f(B, 600, torch.float32),
        a_z=_f(B, 300, torch.float32), pressure=_f(B, 8e6, torch.float32),
    )
    assert out.u.dtype == torch.float32
    assert torch.isfinite(out.u).all()


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
def test_runs_on_cuda():
    B = 2
    x, y = _grid(B, 9)
    x, y = x.cuda(), y.cuda()
    out = _src()(x, y, source_x=_f(B, 0).cuda(), source_y=_f(B, 0).cuda(),
                 depth=_f(B, 4000).cuda(), omega_x=_f(B, 0.3).cuda(),
                 omega_y=_f(B, 0.2).cuda(), omega_z=_f(B, 1.0).cuda(),
                 a_x=_f(B, 900).cuda(), a_y=_f(B, 600).cuda(),
                 a_z=_f(B, 300).cuda(), pressure=_f(B, 8e6).cuda())
    assert out.u.device.type == "cuda" and torch.isfinite(out.u).all()


# --------------------------------------------------------------------------- #
# External reference: original Nikkhoo (2017) MATLAB pECM.m
# --------------------------------------------------------------------------- #
import json
import math
from pathlib import Path

_GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "data" / "nikkhoo_golden.json").read_text()
)


def _col(v):
    return torch.tensor([float(v)], dtype=DTYPE)


class TestNikkhooReference:
    """Golden values from the original Nikkhoo et al. (2017) MATLAB ``pECM.m``.

    Ground truth produced by ``tests/sources/reference/gen_nikkhoo.m`` (run via
    MATLAB) with ``nu = 0.25`` and ``mu = 3e10`` (lambda = 3e10) -- matching the
    ``PECMSource`` defaults. The Python port reproduces the reference code to
    machine precision.
    """

    meta = _GOLDEN["meta"]

    @pytest.mark.parametrize(
        "row", _GOLDEN["pecm"], ids=[r["name"] for r in _GOLDEN["pecm"]]
    )
    def test_pecm_matches_matlab(self, row):
        m = self.meta
        x = torch.tensor(m["X"], dtype=DTYPE).reshape(1, -1)
        y = torch.tensor(m["Y"], dtype=DTYPE).reshape(1, -1)
        om = [math.radians(a) for a in row["omega"]]
        out = PECMSource(poisson_ratio=m["nu"], shear_modulus=m["mu"])(
            x, y, _col(m["X0"]), _col(m["Y0"]), _col(m["depth"]),
            _col(om[0]), _col(om[1]), _col(om[2]),
            _col(row["a"][0]), _col(row["a"][1]), _col(row["a"][2]),
            _col(row["p"]),
        )
        torch.testing.assert_close(out.e[0], torch.tensor(row["ue"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(out.n[0], torch.tensor(row["un"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(out.u[0], torch.tensor(row["uv"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
