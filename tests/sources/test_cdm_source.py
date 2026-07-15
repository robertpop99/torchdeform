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


def test_gradients_finite_at_vertical_sides():
    """At omega = 0 (and any omega_z-only rotation) some rectangle sides are
    exactly vertical. The forward masks them, but ``acos(+/-1)`` / ``sqrt(0)`` have
    infinite slope, so a naive backward returns NaN for every geometry parameter.
    Guard that all gradients stay finite for these degenerate orientations."""
    B = 2
    x, y = _grid(B, 9)
    for omega in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.7)):
        params = {
            "source_x": _f(B, 0), "source_y": _f(B, 0), "depth": _f(B, 4000),
            "omega_x": _f(B, omega[0]), "omega_y": _f(B, omega[1]),
            "omega_z": _f(B, omega[2]),
            "a_x": _f(B, 900), "a_y": _f(B, 700), "a_z": _f(B, 300),
            "opening": _f(B, 3.0),
        }
        for p in params.values():
            p.requires_grad_(True)
        out = CDMSource()(x, y, **params)
        (out.e.sum() + out.n.sum() + out.u.sum()).backward()
        for name, p in params.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


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


def _valid_kwargs(B, **over):
    kw = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 4000),
              omega_x=_f(B, 0.3), omega_y=_f(B, 0.2), omega_z=_f(B, 0.5),
              a_x=_f(B, 900), a_y=_f(B, 700), a_z=_f(B, 300), opening=_f(B, 3.0))
    kw.update(over)
    return kw


@pytest.mark.parametrize("bad", [0.0, -4000.0])
def test_nonpositive_depth_raises(bad):
    # a source at/above the free surface is outside the buried half-space.
    x, y = _grid(2, 5)
    with pytest.raises(ValueError, match="depth"):
        CDMSource()(x, y, **_valid_kwargs(2, depth=_f(2, bad)))


@pytest.mark.parametrize("axis", ["a_x", "a_y", "a_z"])
@pytest.mark.parametrize("bad", [0.0, -300.0])
def test_nonpositive_semi_axis_raises(axis, bad):
    x, y = _grid(2, 5)
    with pytest.raises(ValueError, match=axis):
        CDMSource()(x, y, **_valid_kwargs(2, **{axis: _f(2, bad)}))


def test_vertex_above_surface_raises():
    """A shallow, tilted box whose rectangle vertices poke above z = 0 is outside
    the half-space solution (the reference CDM.m errors here); it must raise, not
    silently return an unphysical field."""
    x, y = _grid(1, 5)
    with pytest.raises(ValueError, match="free surface"):
        CDMSource()(x, y, **_valid_kwargs(
            1, depth=_f(1, 200), a_z=_f(1, 600), omega_x=_f(1, 0.9),
            omega_y=_f(1, 0.0), omega_z=_f(1, 0.0)))
    # same box safely buried: no error
    out = CDMSource()(x, y, **_valid_kwargs(
        1, depth=_f(1, 4000), a_z=_f(1, 600), omega_x=_f(1, 0.9),
        omega_y=_f(1, 0.0), omega_z=_f(1, 0.0)))
    assert torch.isfinite(out.u).all()


def test_surface_touching_vertex_allowed():
    """Vertices exactly at z = 0 are the boundary of validity (analogous to a
    surface-rupturing Okada fault): the reference CDM.m only rejects vertices
    strictly *above* the surface, so touching must not raise. Off the touching
    vertex the field is finite; directly above it the RD kernel is genuinely
    singular (no DC3D-style zeroing convention here), which is why observation
    points below skip x = 0."""
    # axis-aligned: the z-spanning rectangles reach z = -depth + a_z, so
    # depth == a_z puts the top edge exactly at the surface.
    x = torch.tensor([[-10e3, -5e3, 5e3, 10e3]], dtype=DTYPE)
    y = torch.full_like(x, 2e3)
    out = CDMSource()(x, y, **_valid_kwargs(
        1, depth=_f(1, 300), a_x=_f(1, 200), a_y=_f(1, 200), a_z=_f(1, 300),
        omega_x=_f(1, 0.0), omega_y=_f(1, 0.0), omega_z=_f(1, 0.0)))
    for tns in (out.e, out.n, out.u):
        assert torch.isfinite(tns).all()


def test_raises_if_any_batch_element_degenerate():
    # a single bad element in an otherwise valid batch must still raise.
    x, y = _grid(3, 5)
    depth = _f(3, 4000); depth[1] = -1.0
    with pytest.raises(ValueError, match="depth"):
        CDMSource()(x, y, **_valid_kwargs(3, depth=depth))
    a_y = _f(3, 700); a_y[2] = 0.0
    with pytest.raises(ValueError, match="a_y"):
        CDMSource()(x, y, **_valid_kwargs(3, a_y=a_y))


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


# --------------------------------------------------------------------------- #
# External reference: original Nikkhoo (2017) MATLAB CDM.m
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
    """Golden values from the original Nikkhoo et al. (2017) MATLAB ``CDM.m``.

    Ground truth produced by ``tests/sources/reference/gen_nikkhoo.m`` (run via
    MATLAB) with ``nu = 0.25`` -- matching ``CDMSource`` defaults. Semi-axes are
    passed as-is (the reference doubles them internally). The Python port
    reproduces the reference code to machine precision.
    """

    meta = _GOLDEN["meta"]

    @pytest.mark.parametrize(
        "row", _GOLDEN["cdm"], ids=[r["name"] for r in _GOLDEN["cdm"]]
    )
    def test_cdm_matches_matlab(self, row):
        m = self.meta
        x = torch.tensor(m["X"], dtype=DTYPE).reshape(1, -1)
        y = torch.tensor(m["Y"], dtype=DTYPE).reshape(1, -1)
        om = [math.radians(a) for a in row["omega"]]
        out = CDMSource()(
            x, y, _col(m["X0"]), _col(m["Y0"]), _col(m["depth"]),
            _col(om[0]), _col(om[1]), _col(om[2]),
            _col(row["a"][0]), _col(row["a"][1]), _col(row["a"][2]),
            _col(row["opening"]),
        )
        torch.testing.assert_close(out.e[0], torch.tensor(row["ue"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(out.n[0], torch.tensor(row["un"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(out.u[0], torch.tensor(row["uv"], dtype=DTYPE), rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------- #
# External reference: CDM over a random parameter volume
# --------------------------------------------------------------------------- #
_CDM_VOLUME_DIR = Path(__file__).resolve().parent / "data"


def _tt(x):
    return torch.as_tensor(x, dtype=DTYPE)


class TestCDMVolume:
    """CDMSource against the original Nikkhoo (2017) MATLAB ``CDM.m`` over a
    random *parameter* volume.

    Complements ``TestNikkhooReference``'s two hand-picked orientations: the
    fixture ``data/cdm_volume_golden.json`` freezes ``CDM.m``'s forward ENU for
    16 random buried CDMs -- random depth, full 3-axis orientation, semi-axes and
    (signed) opening -- observed at 24 surface points each. The ``_nu0.32``
    sibling repeats the identical geometry at ``nu = 0.32``, verifying the
    non-trivial Poisson-ratio dependence that the ``nu = 0.25``-only fixtures
    cannot see. Forward only: CDMSource has no hand-written backward, so its
    gradients are covered by ``test_gradcheck`` (autograd vs. finite
    differences), not here. Regenerate with ``reference/gen_cdm_volume.py
    [--nu 0.32]`` (needs MATLAB + vendored nikkhoo/); the committed JSON is all
    this test needs. See reference/README.md.
    """

    @pytest.mark.parametrize(
        "fname", ["cdm_volume_golden.json", "cdm_volume_golden_nu0.32.json"]
    )
    def test_cdm_volume_displacement(self, fname):
        golden = _CDM_VOLUME_DIR / fname
        assert golden.is_file(), (
            f"{golden} missing; regenerate with reference/gen_cdm_volume.py"
        )
        d = json.loads(golden.read_text())

        out = CDMSource(poisson_ratio=d["poisson_ratio"])(
            _tt(d["x_obs"]), _tt(d["y_obs"]),
            source_x=_tt(d["source_x"]), source_y=_tt(d["source_y"]),
            depth=_tt(d["depth"]),
            omega_x=_tt(d["omega_x"]), omega_y=_tt(d["omega_y"]),
            omega_z=_tt(d["omega_z"]),
            a_x=_tt(d["a_x"]), a_y=_tt(d["a_y"]), a_z=_tt(d["a_z"]),
            opening=_tt(d["opening"]),
        )
        got = torch.stack([out.e, out.n, out.u], dim=-1)   # [B, N, 3] ENU
        want = _tt(d["u_enu"])
        # Both sides float64; the port reproduces CDM.m to ~1e-13 relative, so
        # rtol has ~4 orders of headroom while still catching small regressions.
        assert torch.allclose(got, want, rtol=1e-9, atol=1e-12), (
            "CDMSource disagrees with CDM.m: max abs diff "
            f"{(got - want).abs().max().item():.3e} m"
        )
