"""
Tests for ``CDMSource`` (finite Compound Dislocation Model).

Focus areas:

* **Correctness** - the CDM reduces to a point CDM (:class:`PCDMSource`) in the
  far field with potencies ``4*a_y*a_z*opening`` etc. (Example-3 of Nikkhoo's
  ``pCDM.m``): high field correlation, and convergence to the point limit for a
  small/deep source. At ``omega = 0`` the axis-aligned geometry has exact
  reflection symmetry. Positive opening produces uplift; sign flips with sign of
  opening.
* **Batchability** - a batched call equals a per-item loop.
* **Differentiability** - ``torch.autograd.gradcheck`` passes through every
  continuous input (at a generic, non-degenerate orientation), and gradients stay
  finite.
* **Validation / dtype / device.**

The far-field cross-check against the independently validated ``PCDMSource``
stands in for a direct comparison with the (MATLAB) reference output.

Run with::

    pytest test_cdm_source.py -v
"""
import pytest
import torch

from torchdeform import CDMSource, PCDMSource, Displacement


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
    out = CDMSource()(
        x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
        omega_x=_f(B, 0.3), omega_y=_f(B, -0.4), omega_z=_f(B, 1.0),
        a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=_f(B, 3.0),
    )
    assert isinstance(out, Displacement)
    assert out.e.shape == (B, n * n)
    for t in (out.e, out.n, out.u):
        assert torch.isfinite(t).all()


def test_runs_at_zero_orientation():
    """omega = 0 makes the box axis-aligned: two sides of two rectangles are
    exactly vertical (the masked, degenerate path). It must stay finite."""
    B = 2
    x, y = _grid(B, 9)
    out = CDMSource()(
        x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
        omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
        a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=_f(B, 3.0),
    )
    for t in (out.e, out.n, out.u):
        assert torch.isfinite(t).all()


# --------------------------------------------------------------------------- #
# Correctness: far-field reduction to the point CDM
# --------------------------------------------------------------------------- #
def _cdm_pcdm_pair(X, Y, depth, ox, oy, oz, ax, ay, az, opening):
    B = X.shape[0]
    z = torch.zeros(B, dtype=DTYPE)
    cdm = CDMSource()(X, Y, z, z, depth, ox, oy, oz, ax, ay, az, opening)
    # far-field equivalence: DVx = 4*ay*az*opening, etc. (full axes = 2*semi-axis)
    pc = PCDMSource()(X, Y, z, z, depth, ox, oy, oz,
                      4 * ay * az * opening, 4 * ax * az * opening,
                      4 * ax * ay * opening)
    return cdm, pc


def test_farfield_matches_pcdm_shape():
    """CDM and pCDM fields are highly correlated (Example-3 of pCDM.m)."""
    B = 1
    X = torch.linspace(-20000, 20000, 2001, dtype=DTYPE).reshape(1, -1)
    Y = torch.zeros_like(X)
    cdm, pc = _cdm_pcdm_pair(
        X, Y, _f(B, 4000), _f(B, 0), _f(B, 0), _f(B, 0),
        _f(B, 1250), _f(B, 1000), _f(B, 350), _f(B, 5.0),
    )

    def corr(a, b):
        a = a.flatten() - a.flatten().mean()
        b = b.flatten() - b.flatten().mean()
        return (a @ b / (a.norm() * b.norm())).item()

    assert corr(cdm.u, pc.u) > 0.99
    assert corr(cdm.e, pc.e) > 0.99


def test_converges_to_pcdm_for_small_deep_source():
    """A small source seen from far converges to its point-CDM limit."""
    B = 1
    X = torch.linspace(-40000, 40000, 2001, dtype=DTYPE).reshape(1, -1)
    Y = torch.zeros_like(X)
    cdm, pc = _cdm_pcdm_pair(
        X, Y, _f(B, 8000), _f(B, 0), _f(B, 0), _f(B, 0),
        _f(B, 300), _f(B, 250), _f(B, 100), _f(B, 5.0),
    )
    rel = (cdm.u - pc.u).abs().max() / pc.u.abs().max()
    assert rel < 0.01


def test_farfield_matches_pcdm_when_rotated():
    """The far-field equivalence holds at a generic orientation too."""
    B = 1
    X = torch.linspace(-40000, 40000, 1501, dtype=DTYPE).reshape(1, -1)
    Y = (0.3 * X)
    cdm, pc = _cdm_pcdm_pair(
        X, Y, _f(B, 8000), _f(B, 0.4), _f(B, -0.6), _f(B, 1.2),
        _f(B, 300), _f(B, 250), _f(B, 100), _f(B, 5.0),
    )
    rel = (cdm.u - pc.u).abs().max() / pc.u.abs().max()
    assert rel < 0.02


# --------------------------------------------------------------------------- #
# Symmetry / sign
# --------------------------------------------------------------------------- #
def test_axis_aligned_reflection_symmetry():
    """At omega = 0 the geometry is symmetric under x -> -x: ue is odd, un/uv
    are even in x."""
    B = 1
    x, y = _grid(B, 21)
    common = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                  omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
                  a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=_f(B, 3.0))
    src = CDMSource()
    d = src(x, y, **common)
    dr = src(-x, y, **common)
    assert (d.e + dr.e).abs().max() < 1e-9
    assert (d.n - dr.n).abs().max() < 1e-9
    assert (d.u - dr.u).abs().max() < 1e-9


def test_positive_opening_gives_uplift():
    B, n = 1, 21
    x, y = _grid(B, n)
    d = CDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                    omega_x=_f(B, 0), omega_y=_f(B, 0), omega_z=_f(B, 0),
                    a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=_f(B, 3.0))
    center = n * n // 2
    assert d.u[0, center] > 0


def test_opening_sign_flips_field():
    B = 1
    x, y = _grid(B)
    common = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
                  omega_x=_f(B, 0.4), omega_y=_f(B, 0.1), omega_z=_f(B, 0.9),
                  a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300))
    up = CDMSource()(x, y, opening=_f(B, 3.0), **common)
    down = CDMSource()(x, y, opening=_f(B, -3.0), **common)
    assert torch.allclose(up.u, -down.u, atol=1e-9)
    assert torch.allclose(up.e, -down.e, atol=1e-9)


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
    a_x = 300 + 700 * torch.rand(B, generator=g, dtype=DTYPE)
    a_y = 300 + 700 * torch.rand(B, generator=g, dtype=DTYPE)
    a_z = 100 + 400 * torch.rand(B, generator=g, dtype=DTYPE)
    op = -2.0 + 4.0 * torch.rand(B, generator=g, dtype=DTYPE)

    src = CDMSource()
    full = src(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
               omega_x=ox, omega_y=oy, omega_z=oz,
               a_x=a_x, a_y=a_y, a_z=a_z, opening=op)

    for b in range(B):
        one = src(x[b:b + 1], y[b:b + 1], source_x=_f(1, 0), source_y=_f(1, 0),
                  depth=depth[b:b + 1], omega_x=ox[b:b + 1], omega_y=oy[b:b + 1],
                  omega_z=oz[b:b + 1], a_x=a_x[b:b + 1], a_y=a_y[b:b + 1],
                  a_z=a_z[b:b + 1], opening=op[b:b + 1])
        assert torch.allclose(full.e[b:b + 1], one.e, atol=1e-9)
        assert torch.allclose(full.n[b:b + 1], one.n, atol=1e-9)
        assert torch.allclose(full.u[b:b + 1], one.u, atol=1e-9)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_bad_shape_raises():
    x, y = _grid(2, 5)
    with pytest.raises(ValueError):
        CDMSource()(x, y, source_x=_f(3, 0), source_y=_f(3, 0), depth=_f(3, 4000),
                    omega_x=_f(3, 0), omega_y=_f(3, 0), omega_z=_f(3, 0),
                    a_x=_f(3, 900), a_y=_f(3, 700), a_z=_f(3, 300), opening=_f(3, 3.0))


# --------------------------------------------------------------------------- #
# Differentiability  (unit-scale, generic non-degenerate orientation)
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
    ax_ = _f(B, 0.5).requires_grad_(True)
    ay_ = _f(B, 0.4).requires_grad_(True)
    az_ = _f(B, 0.25).requires_grad_(True)
    op = _f(B, 1e-3).requires_grad_(True)

    def fn(sx, sy, depth, ox, oy, oz, ax_, ay_, az_, op):
        d = CDMSource()(x, y, source_x=sx, source_y=sy, depth=depth,
                        omega_x=ox, omega_y=oy, omega_z=oz,
                        a_x=ax_, a_y=ay_, a_z=az_, opening=op)
        return torch.stack([d.e, d.n, d.u], dim=-1)

    assert torch.autograd.gradcheck(fn, (sx, sy, depth, ox, oy, oz, ax_, ay_, az_, op))


def test_gradients_finite_and_nonzero():
    B = 2
    x, y = _grid(B, 9)
    depth = _f(B, 4000).requires_grad_(True)
    op = _f(B, 3.0).requires_grad_(True)
    d = CDMSource()(x, y, source_x=_f(B, 0), source_y=_f(B, 0), depth=depth,
                    omega_x=_f(B, 0.3), omega_y=_f(B, 0.2), omega_z=_f(B, 1.0),
                    a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=op)
    d.u.pow(2).mean().backward()
    for t in (depth, op):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
def test_dtype_float32():
    B = 2
    x, y = _grid(B, 7, dtype=torch.float32)
    out = CDMSource(internal_dtype=torch.float32)(
        x, y, source_x=_f(B, 0, torch.float32), source_y=_f(B, 0, torch.float32),
        depth=_f(B, 4000, torch.float32),
        omega_x=_f(B, 0.2, torch.float32), omega_y=_f(B, 0.1, torch.float32),
        omega_z=_f(B, 0.5, torch.float32),
        a_x=_f(B, 900, torch.float32), a_y=_f(B, 700, torch.float32),
        a_z=_f(B, 300, torch.float32), opening=_f(B, 3.0, torch.float32),
    )
    assert out.u.dtype == torch.float32


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
def test_runs_on_cuda():
    B = 2
    x, y = _grid(B, 9)
    x, y = x.cuda(), y.cuda()
    out = CDMSource()(x, y, source_x=_f(B, 0).cuda(), source_y=_f(B, 0).cuda(),
                      depth=_f(B, 4000).cuda(), omega_x=_f(B, 0.3).cuda(),
                      omega_y=_f(B, 0.2).cuda(), omega_z=_f(B, 1.0).cuda(),
                      a_x=_f(B, 900).cuda(), a_y=_f(B, 700).cuda(),
                      a_z=_f(B, 300).cuda(), opening=_f(B, 3.0).cuda())
    assert out.u.device.type == "cuda" and torch.isfinite(out.u).all()
