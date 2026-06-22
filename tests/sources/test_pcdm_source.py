"""
Tests for ``PCDMSource`` (point Compound Dislocation Model).

Focus areas:

* **Correctness** - an *isotropic* pCDM (equal potencies) is rotation-invariant,
  radially symmetric, and matches the Mogi center-of-dilatation field in shape;
  positive potency produces uplift.
* **Batchability** - a batched call equals a per-item loop.
* **Differentiability** - ``torch.autograd.gradcheck`` passes through every
  continuous input, and gradients stay finite.
* **Validation** - mixed-sign potencies and bad shapes are rejected.

Run with::

    pytest test_pcdm_source.py -v
"""
import math

import pytest
import torch

from torchdeform import PCDMSource, MogiSource, Displacement


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _grid(B, n=21, extent=20_000.0, dtype=DTYPE):
    ax = torch.linspace(-extent / 2, extent / 2, n, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x = xx.reshape(1, -1).expand(B, -1).contiguous()
    y = yy.reshape(1, -1).expand(B, -1).contiguous()
    return x, y


def _f(B, v, dtype=DTYPE):
    return torch.full((B,), float(v), dtype=dtype)


# --------------------------------------------------------------------------- #
# Basic
# --------------------------------------------------------------------------- #
def test_output_shape_and_finite():
    B, n = 3, 11
    x, y = _grid(B, n)
    dv = _f(B, 5e6)
    out = PCDMSource()(
        x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 3000),
        omega_x=_f(B, 0.3), omega_y=_f(B, -0.4), omega_z=_f(B, 1.0),
        dv_x=dv, dv_y=_f(B, 4e6), dv_z=_f(B, 6e6),
    )
    assert isinstance(out, Displacement)
    assert out.e.shape == (B, n * n)
    for t in (out.e, out.n, out.u):
        assert torch.isfinite(t).all()


# --------------------------------------------------------------------------- #
# Isotropic case: rotation-invariant, radial, Mogi-shaped, uplift
# --------------------------------------------------------------------------- #
def test_isotropic_is_rotation_invariant():
    B = 2
    x, y = _grid(B)
    dv = _f(B, 1e7)
    common = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                  dv_x=dv, dv_y=dv, dv_z=dv)
    d0 = PCDMSource()(x, y, omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0), **common)
    d1 = PCDMSource()(x, y, omega_x=_f(B, 0.7), omega_y=_f(B, -1.1), omega_z=_f(B, 2.0), **common)
    assert torch.allclose(d0.e, d1.e, atol=1e-8)
    assert torch.allclose(d0.n, d1.n, atol=1e-8)
    assert torch.allclose(d0.u, d1.u, atol=1e-8)


def test_isotropic_is_radial():
    B = 2
    x, y = _grid(B)
    dv = _f(B, 1e7)
    d = PCDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                     omega_x=_f(B, 0.5), omega_y=_f(B, 0.2), omega_z=_f(B, 1.3),
                     dv_x=dv, dv_y=dv, dv_z=dv)
    # horizontal displacement is purely radial: ue*y == un*x
    assert (d.e * y - d.n * x).abs().max() < 1e-6


def test_isotropic_matches_mogi_shape():
    """Isotropic pCDM == Mogi center of dilatation in shape (corr ~ 1)."""
    B = 2
    x, y = _grid(B)
    dv = _f(B, 1e7)
    pc = PCDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                      omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
                      dv_x=dv, dv_y=dv, dv_z=dv)
    mg = MogiSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0),
                      depth=_f(B, 4000), delta_v=_f(B, 1e7))

    def corr(a, b):
        a = a.flatten() - a.flatten().mean()
        b = b.flatten() - b.flatten().mean()
        return (a @ b / (a.norm() * b.norm())).item()

    assert corr(pc.u, mg.u) > 0.9999
    assert corr(pc.e, mg.e) > 0.9999


def test_positive_potency_gives_uplift():
    B = 1
    n = 21
    x, y = _grid(B, n)
    dv = _f(B, 1e7)
    d = PCDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                     omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
                     dv_x=dv, dv_y=dv, dv_z=dv)
    center = n * n // 2     # the (0, 0) pixel of the centred grid
    assert d.u[0, center] > 0


def test_deflation_flips_sign():
    B = 1
    x, y = _grid(B)
    common = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                  omega_x=_f(B, 0.4), omega_y=_f(B, 0.1), omega_z=_f(B, 0.9))
    up = PCDMSource()(x, y, dv_x=_f(B, 1e7), dv_y=_f(B, 8e6), dv_z=_f(B, 5e6), **common)
    down = PCDMSource()(x, y, dv_x=_f(B, -1e7), dv_y=_f(B, -8e6), dv_z=_f(B, -5e6), **common)
    assert torch.allclose(up.u, -down.u, atol=1e-9)


# --------------------------------------------------------------------------- #
# Batchability
# --------------------------------------------------------------------------- #
def test_batched_matches_loop():
    B, n = 4, 11
    x, y = _grid(B, n)
    g = torch.Generator().manual_seed(0)
    depth = 2000 + 6000 * torch.rand(B, generator=g, dtype=DTYPE)
    ox = torch.randn(B, generator=g, dtype=DTYPE)
    oy = torch.randn(B, generator=g, dtype=DTYPE)
    oz = torch.randn(B, generator=g, dtype=DTYPE)
    sign = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0).to(DTYPE)
    dvx = sign * (1e6 + 1e7 * torch.rand(B, generator=g, dtype=DTYPE))
    dvy = sign * (1e6 + 1e7 * torch.rand(B, generator=g, dtype=DTYPE))
    dvz = sign * (1e6 + 1e7 * torch.rand(B, generator=g, dtype=DTYPE))

    src = PCDMSource()
    full = src(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
               omega_x=ox, omega_y=oy, omega_z=oz, dv_x=dvx, dv_y=dvy, dv_z=dvz)

    for b in range(B):
        one = src(x[b:b+1], y[b:b+1], source_x=_f(1, 0), source_y=_f(1, 0),
                  depth=depth[b:b+1], omega_x=ox[b:b+1], omega_y=oy[b:b+1],
                  omega_z=oz[b:b+1], dv_x=dvx[b:b+1], dv_y=dvy[b:b+1], dv_z=dvz[b:b+1])
        assert torch.allclose(full.e[b:b+1], one.e, atol=1e-9)
        assert torch.allclose(full.u[b:b+1], one.u, atol=1e-9)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_mixed_sign_potency_raises():
    B = 2
    x, y = _grid(B, 5)
    with pytest.raises(ValueError, match="share a sign"):
        PCDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 3000),
                     omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
                     dv_x=_f(B, 1e7), dv_y=_f(B, -1e7), dv_z=_f(B, 1e7))


def test_bad_shape_raises():
    x, y = _grid(2, 5)
    with pytest.raises(ValueError):
        PCDMSource()(x, y, source_x=_f(3, 0), source_y=_f(3, 0), depth=_f(3, 3000),
                     omega_x=_f(3, 0), omega_y=_f(3, 0), omega_z=_f(3, 0),
                     dv_x=_f(3, 1e7), dv_y=_f(3, 1e7), dv_z=_f(3, 1e7))


# --------------------------------------------------------------------------- #
# Differentiability  (unit-scale problem so gradcheck's single eps is valid)
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
    ox = _f(B, 0.1).requires_grad_(True)
    oy = _f(B, -0.2).requires_grad_(True)
    oz = _f(B, 0.5).requires_grad_(True)
    vx = _f(B, 1.2e-3).requires_grad_(True)
    vy = _f(B, 1.0e-3).requires_grad_(True)
    vz = _f(B, 0.7e-3).requires_grad_(True)

    def fn(sx, sy, depth, ox, oy, oz, vx, vy, vz):
        d = PCDMSource()(x, y, source_x=sx, source_y=sy, depth=depth,
                         omega_x=ox, omega_y=oy, omega_z=oz,
                         dv_x=vx, dv_y=vy, dv_z=vz)
        return torch.stack([d.e, d.n, d.u], dim=-1)

    assert torch.autograd.gradcheck(fn, (sx, sy, depth, ox, oy, oz, vx, vy, vz))


def test_gradients_finite_and_nonzero():
    B = 2
    x, y = _grid(B, 9)
    depth = _f(B, 4000).requires_grad_(True)
    dv = _f(B, 1e7).requires_grad_(True)
    d = PCDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
                     omega_x=_f(B, 0.3), omega_y=_f(B, 0.2), omega_z=_f(B, 1.0),
                     dv_x=dv, dv_y=_f(B, 8e6), dv_z=_f(B, 6e6))
    d.u.pow(2).mean().backward()
    for t in (depth, dv):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
def test_dtype_float32():
    B = 2
    x, y = _grid(B, 7, dtype=torch.float32)
    out = PCDMSource(internal_dtype=torch.float32)(
        x, y, source_x=_f(B, 0, torch.float32), source_y=_f(B, 0, torch.float32),
        depth=_f(B, 3000, torch.float32),
        omega_x=_f(B, 0.2, torch.float32), omega_y=_f(B, 0.1, torch.float32),
        omega_z=_f(B, 0.5, torch.float32),
        dv_x=_f(B, 1e7, torch.float32), dv_y=_f(B, 1e7, torch.float32),
        dv_z=_f(B, 1e7, torch.float32),
    )
    assert out.u.dtype == torch.float32


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
def test_runs_on_cuda():
    B = 2
    x, y = _grid(B, 9)
    x, y = x.cuda(), y.cuda()
    out = PCDMSource()(x, y, source_x=_f(B, 0).cuda(), source_y=_f(B, 0).cuda(),
                       depth=_f(B, 4000).cuda(), omega_x=_f(B, 0.3).cuda(),
                       omega_y=_f(B, 0.2).cuda(), omega_z=_f(B, 1.0).cuda(),
                       dv_x=_f(B, 1e7).cuda(), dv_y=_f(B, 8e6).cuda(), dv_z=_f(B, 6e6).cuda())
    assert out.u.device.type == "cuda" and torch.isfinite(out.u).all()
