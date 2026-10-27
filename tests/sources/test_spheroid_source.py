"""
Tests for ``SpheroidSource`` (finite pressurised spheroid, Yang et al. 1988, as
corrected by Cervelli's ``spheroid.m``).

Focus areas:

* **External references** - Cervelli's own surface test cases (from
  ``spheroid.m``'s built-in ``test``); a random parameter volume run through the
  original MATLAB ``spheroid.m`` (prolate and oblate, both strength modes, two
  Poisson ratios); and a 50-digit evaluation of ``spheroid.m``'s expressions in
  the near-vertical / near-spherical limits, where MATLAB's own ``float64``
  round-off is too large to referee the interpolation bands.
* **Limits** - an exact sphere is Mogi; the field is continuous through the
  sphere (prolate <-> oblate) and vertical-axis bands; a deep spheroid
  approaches the independently validated point ellipsoid (``PECMSource``).
* **Regularity** - finite and continuous at the axis piercing point and on the
  strike line, the two places ``spheroid.m`` special-cases.
* **Symmetry / sign** - ``(dip, strike) == (pi - dip, strike + pi)``; a vertical
  spheroid is radial and strike-independent; mirror symmetry about the axis
  plane; displacement linear in strength, positive inflating.
* **Batchability** - a batched call (mixing every code path) equals a per-item
  loop.
* **Differentiability** - ``gradcheck`` through every input at generic,
  singular-point, band and oblate configurations.
* **Validation / dtype / device.**

Run with::

    pytest test_spheroid_source.py -v
"""
import json
import math
from pathlib import Path

import pytest
import torch

from torchdeform import SpheroidSource, MogiSource, PECMSource, Displacement
from torchdeform.sources.spheroid import VERTICAL_BAND, SPHERE_BAND


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DATA = Path(__file__).resolve().parent / "data"

NU = 0.25
MU = 3.0e10
PARAM_KEYS = ("source_x", "source_y", "depth", "polar_radius", "equatorial_radius",
              "dip", "strike")


def _grid(B, n=21, extent=30_000.0, dtype=DTYPE):
    ax = torch.linspace(-extent / 2, extent / 2, n, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x = xx.reshape(1, -1).expand(B, -1).contiguous()
    y = yy.reshape(1, -1).expand(B, -1).contiguous()
    return x, y


def _f(B, v, dtype=DTYPE):
    return torch.full((B,), float(v), dtype=dtype)


def _tt(x):
    return torch.as_tensor(x, dtype=DTYPE)


def _kw(B=1, **over):
    kw = dict(source_x=_f(B, 0), source_y=_f(B, 0), depth=_f(B, 5000),
              polar_radius=_f(B, 2000), equatorial_radius=_f(B, 700),
              dip=_f(B, 0.9), strike=_f(B, 0.6), strength=_f(B, 5e6))
    kw.update({k: (_f(B, v) if isinstance(v, (int, float)) else v)
               for k, v in over.items()})
    return kw


def _enu(d):
    return torch.stack([d.e, d.n, d.u], dim=-1)


def _src(**kw):
    return SpheroidSource(poisson_ratio=NU, shear_modulus=MU, **kw)


def _axis_point(along, across=0.0, strike=0.6, sx=0.0, sy=0.0):
    """Surface point ``along`` the axis azimuth and ``across`` it (x1, x2)."""
    ss, cs = math.sin(strike), math.cos(strike)
    return sx + along * ss - across * cs, sy + along * cs + across * ss


# --------------------------------------------------------------------------- #
# Basic
# --------------------------------------------------------------------------- #
def test_output_shape_and_finite():
    B, n = 3, 11
    x, y = _grid(B, n)
    out = _src()(x, y, **_kw(B))
    assert isinstance(out, Displacement)
    assert out.e.shape == (B, n * n)
    for t in (out.e, out.n, out.u):
        assert torch.isfinite(t).all()


def test_positive_strength_uplifts_and_is_linear():
    x, y = _grid(1)
    for mode, s in (("pressure", 5e6), ("volume", 2e6)):
        src = _src(strength_type=mode)
        d = _enu(src(x, y, **_kw(strength=s)))
        d2 = _enu(src(x, y, **_kw(strength=2 * s)))
        dn = _enu(src(x, y, **_kw(strength=-s)))
        assert d[..., 2].max() > 0 and d[..., 2].max() > -d[..., 2].min()
        torch.testing.assert_close(d2, 2 * d, rtol=1e-12, atol=0)
        torch.testing.assert_close(dn, -d, rtol=1e-12, atol=0)


def test_volume_mode_ignores_shear_modulus():
    x, y = _grid(1, 9)
    a = SpheroidSource(shear_modulus=1e9, strength_type="volume")(x, y, **_kw(strength=1e6))
    b = SpheroidSource(shear_modulus=7e10, strength_type="volume")(x, y, **_kw(strength=1e6))
    torch.testing.assert_close(_enu(a), _enu(b), rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Cervelli's own surface test cases (spheroid.m built-in ``test``, cases 12-15)
# --------------------------------------------------------------------------- #
# m = [a b dip strike X1 X2 X3 P], nu = 1/3, mu = 1, observed at the origin.
# spheroid.m's sign has positive P deflating, hence strength = -434 here.
CERVELLI_SURFACE_CASES = [
    # (a, b, dip_deg, uz)
    (3000.0, 2000.0, 90.0, -22599.412825045478),
    (2000.0, 3000.0, 90.0, -80174.178160942203),
    (2000.0, 3000.0, 0.0, -40267.722640112508),
    (3000.0, 2000.0, 0.0, -41665.789244705156),
]


@pytest.mark.parametrize("a,b,dip_deg,uz", CERVELLI_SURFACE_CASES)
def test_cervelli_builtin_surface_cases(a, b, dip_deg, uz):
    """Directly above the centre: the axis piercing point of a vertical spheroid,
    and the strike line of a horizontal one -- both special cases in spheroid.m."""
    z = torch.zeros(1, 1, dtype=DTYPE)
    out = SpheroidSource(poisson_ratio=1.0 / 3.0, shear_modulus=1.0)(
        z, z, _f(1, 0), _f(1, 0), _f(1, 10_000), _f(1, a), _f(1, b),
        _f(1, math.radians(dip_deg)), _f(1, math.radians(90.0)), _f(1, -434.0))
    assert abs(out.e.item()) < 1e-9 * abs(uz)
    assert abs(out.n.item()) < 1e-9 * abs(uz)
    assert out.u.item() == pytest.approx(uz, rel=1e-10)


# --------------------------------------------------------------------------- #
# External reference: spheroid.m over a random parameter volume
# --------------------------------------------------------------------------- #
class TestSpheroidVolume:
    """SpheroidSource against Cervelli's MATLAB ``spheroid.m`` over a random
    *parameter* volume.

    ``data/spheroid_volume_golden.json`` freezes ``spheroid.m``'s surface ENU for
    20 random buried spheroids (prolate and oblate, any orientation, including
    exactly vertical and horizontal axes) at 24 points each -- among them the
    axis piercing point and a point on the strike line -- in both strength modes.
    The ``_nu0.32`` sibling repeats the geometry at ``nu = 0.32``. Regenerate with
    ``reference/gen_spheroid_volume.py [--nu 0.32]`` (needs MATLAB + vendored
    cervelli/); the committed JSON is all this test needs.
    """

    @pytest.mark.parametrize(
        "fname", ["spheroid_volume_golden.json", "spheroid_volume_golden_nu0.32.json"]
    )
    @pytest.mark.parametrize("mode,key", [("pressure", "pressure"), ("volume", "delta_v")])
    def test_spheroid_volume_displacement(self, fname, mode, key):
        golden = DATA / fname
        assert golden.is_file(), (
            f"{golden} missing; regenerate with reference/gen_spheroid_volume.py"
        )
        d = json.loads(golden.read_text())
        assert d["shear_modulus"] == MU
        out = SpheroidSource(poisson_ratio=d["poisson_ratio"],
                             shear_modulus=d["shear_modulus"], strength_type=mode)(
            _tt(d["x_obs"]), _tt(d["y_obs"]), *(_tt(d[k]) for k in PARAM_KEYS),
            _tt(d[key]))
        got, want = _enu(out), _tt(d[f"u_enu_{mode}"])
        # Both float64. The port matches spheroid.m to ~3e-10 of each source's
        # peak (and is the more accurate of the two against a 50-digit reference),
        # so 1e-8 leaves headroom while catching any real regression.
        peak = want.abs().amax(dim=(1, 2), keepdim=True)
        err = ((got - want).abs() / peak).max().item()
        assert err < 1e-8, f"SpheroidSource vs spheroid.m: {err:.2e} of peak"


# --------------------------------------------------------------------------- #
# External reference: 50-digit spheroid.m expressions near the singular limits
# --------------------------------------------------------------------------- #
_MP = json.loads((DATA / "spheroid_mp_golden.json").read_text())


class TestNearSingularLimits:
    """The near-vertical and near-spherical interpolation bands (and their edges)
    against ``spheroid.m``'s own expressions evaluated at 50 digits
    (``reference/gen_spheroid_mp.py``), where ``float64`` MATLAB is too noisy."""

    @pytest.mark.parametrize(
        "case", _MP["cases"],
        ids=[f"dip{math.degrees(c['dip']):.3f}_ab{c['polar_radius'] / c['equatorial_radius']:.6f}"
             for c in _MP["cases"]])
    @pytest.mark.parametrize("mode,key", [("pressure", "pressure"), ("volume", "delta_v")])
    def test_matches_50_digit_reference(self, case, mode, key):
        src = SpheroidSource(poisson_ratio=_MP["poisson_ratio"],
                             shear_modulus=_MP["shear_modulus"], strength_type=mode)
        out = src(_tt(case["x_obs"])[None], _tt(case["y_obs"])[None],
                  *(_tt([case[k]]) for k in PARAM_KEYS), _tt([case[key]]))
        got, want = _enu(out)[0], _tt(case[f"u_enu_{mode}"])
        err = ((got - want).abs().max() / want.abs().max()).item()
        assert err < 1e-8, f"{err:.2e} of peak"


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
def test_exact_sphere_is_mogi():
    x, y = _grid(2)
    kw = _kw(2, polar_radius=800.0, equatorial_radius=800.0, strength=3e6)
    vol = SpheroidSource(poisson_ratio=NU, strength_type="volume")(x, y, **kw)
    mogi = MogiSource(poisson_ratio=NU)(x, y, kw["source_x"], kw["source_y"],
                                        kw["depth"], delta_v=kw["strength"])
    torch.testing.assert_close(_enu(vol), _enu(mogi), rtol=1e-12, atol=1e-18)
    # pressure mode: Mogi with delta_v = pi a^3 p / mu
    pres = _src()(x, y, **kw)
    mogi_p = MogiSource(poisson_ratio=NU)(
        x, y, kw["source_x"], kw["source_y"], kw["depth"],
        delta_v=math.pi * 800.0 ** 3 * kw["strength"] / MU)
    torch.testing.assert_close(_enu(pres), _enu(mogi_p), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("dip", [0.4, math.pi / 2, 2.3])
def test_continuous_through_sphere_band(dip):
    """Prolate -> sphere -> oblate is smooth: just inside and just outside the
    band agree to first order, and the sphere sits between its neighbours."""
    x, y = _grid(1, 11)
    b = 900.0

    def u(rel):
        return _enu(_src()(x, y, **_kw(polar_radius=b * (1 + rel), equatorial_radius=b,
                                       dip=dip)))

    peak = u(0.0).abs().max()
    for edge in (SPHERE_BAND, -SPHERE_BAND):
        eps = 1e-10 * math.copysign(1.0, edge)
        jump = (u(edge + eps) - u(edge - eps)).abs().max() / peak
        assert jump < 1e-8, f"jump {jump:.1e} at band edge {edge}"
    # second difference across the sphere is tiny (smooth, no kink)
    h = 1e-3
    curv = (u(h) - 2 * u(0.0) + u(-h)).abs().max() / peak
    assert curv < 1e-5


def test_continuous_through_vertical_band():
    x, y = _grid(1, 11)

    def u(dip, **over):
        return _enu(_src()(x, y, **_kw(dip=dip, **over)))

    for over in ({}, {"polar_radius": 700.0, "equatorial_radius": 1500.0}):
        peak = u(math.pi / 2, **over).abs().max()
        for edge in (math.pi / 2 - VERTICAL_BAND, math.pi / 2 + VERTICAL_BAND):
            jump = (u(edge + 1e-9, **over) - u(edge - 1e-9, **over)).abs().max() / peak
            assert jump < 1e-8, f"jump {jump:.1e} at dip {edge}"


@pytest.mark.parametrize("axes", [(800.0, 400.0), (400.0, 800.0)])
@pytest.mark.parametrize("dip,strike,pecm_axes", [
    (math.pi / 2, 0.0, "z"),          # vertical polar axis
    (0.0, math.pi / 2, "x"),          # horizontal, pointing east
])
def test_far_field_approaches_point_ellipsoid(axes, dip, strike, pecm_axes):
    """A small, deep spheroid tends to the pressurised point ellipsoid (pECM,
    Eshelby potencies) -- an independent check of shape, orientation, sign and
    amplitude -- with the finite-size correction shrinking as (a / depth)^2."""
    a, b = axes
    n = 31
    ax = torch.linspace(-60e3, 60e3, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x, y = xx.reshape(1, -1), yy.reshape(1, -1)
    semi = {"x": (a, b, b), "z": (b, b, a)}[pecm_axes]
    errs = []
    for depth in (10e3, 20e3):
        s = _src()(x, y, **_kw(depth=depth, polar_radius=a, equatorial_radius=b,
                               dip=dip, strike=strike, strength=1e7))
        z0 = _f(1, 0)
        p = PECMSource(poisson_ratio=NU, shear_modulus=MU)(
            x, y, z0, z0, _f(1, depth), z0, z0, z0,
            _f(1, semi[0]), _f(1, semi[1]), _f(1, semi[2]), _f(1, 1e7))
        errs.append(((_enu(s) - _enu(p)).abs().max() / p.u.abs().max()).item())
    assert errs[0] < 1e-2
    assert errs[1] < errs[0] / 3.0          # ~ (a/d)^2 convergence


# --------------------------------------------------------------------------- #
# Regularity at spheroid.m's special points
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dip", [0.9, 2.2, math.radians(10.0), math.pi / 2])
def test_regular_at_axis_piercing_point(dip):
    """Where the polar axis pierces the surface the field is finite and
    continuous (spheroid.m's ``R1 + r3bar -> 0`` limit), even far out."""
    d = 5000.0
    along = -d / math.tan(dip)
    pts = [_axis_point(along), _axis_point(along + 1e-3), _axis_point(along, 1e-3)]
    x = _tt([[p[0] for p in pts]])
    y = _tt([[p[1] for p in pts]])
    out = _enu(_src()(x, y, **_kw(dip=dip, depth=d)))[0]
    assert torch.isfinite(out).all()
    scale = _enu(_src()(*_grid(1, 21), **_kw(dip=dip, depth=d))).abs().max()
    assert (out[1:] - out[0]).abs().max() / scale < 1e-6


def test_continuous_across_strike_line():
    """No branch jump of the ``atan`` across ``y2 = 0``; the cross-strike
    component is odd and the others even about the axis plane."""
    along = torch.linspace(-30e3, 30e3, 61, dtype=DTYPE)
    for across in (0.0, 1e-3, -1e-3):
        pts = [_axis_point(float(t), across) for t in along]
        out = _src()(_tt([[p[0] for p in pts]]), _tt([[p[1] for p in pts]]), **_kw())
        if across == 0.0:
            on = _enu(out)
        elif across > 0:
            plus = _enu(out)
        else:
            minus = _enu(out)
    scale = on.abs().max()
    assert ((plus - on).abs().max() / scale) < 1e-6
    assert ((minus - on).abs().max() / scale) < 1e-6


def test_mirror_symmetry_about_axis_plane():
    strike = 0.6
    pts = [_axis_point(a, c, strike) for a in (-9e3, -2e3, 1e3, 7e3) for c in (1.5e3, 6e3)]
    mir = [_axis_point(a, -c, strike) for a in (-9e3, -2e3, 1e3, 7e3) for c in (1.5e3, 6e3)]
    src = _src()
    u = src(_tt([[p[0] for p in pts]]), _tt([[p[1] for p in pts]]), **_kw(strike=strike))
    m = src(_tt([[p[0] for p in mir]]), _tt([[p[1] for p in mir]]), **_kw(strike=strike))
    ss, cs = math.sin(strike), math.cos(strike)
    along = lambda d: ss * d.e + cs * d.n
    across = lambda d: -cs * d.e + ss * d.n
    scale = u.u.abs().max()
    assert (along(u) - along(m)).abs().max() / scale < 1e-10
    assert (across(u) + across(m)).abs().max() / scale < 1e-10
    assert (u.u - m.u).abs().max() / scale < 1e-10


# --------------------------------------------------------------------------- #
# Orientation conventions
# --------------------------------------------------------------------------- #
def test_dip_reflection_equivalence():
    """(dip, strike) and (pi - dip, strike + pi) describe the same cavity, and
    dip is periodic in pi."""
    x, y = _grid(1, 11)
    src = _src()
    base = _enu(src(x, y, **_kw(dip=0.5, strike=0.6)))
    refl = _enu(src(x, y, **_kw(dip=math.pi - 0.5, strike=0.6 + math.pi)))
    per = _enu(src(x, y, **_kw(dip=0.5 + math.pi, strike=0.6)))
    scale = base.abs().max()
    assert (base - refl).abs().max() / scale < 1e-10
    assert (base - per).abs().max() / scale < 1e-10


def test_vertical_spheroid_is_radial_and_strike_independent():
    x, y = _grid(1)
    src = _src()
    d0 = src(x, y, **_kw(dip=math.pi / 2, strike=0.0))
    d1 = src(x, y, **_kw(dip=math.pi / 2, strike=2.1))
    scale = d0.u.abs().max()
    assert (_enu(d0) - _enu(d1)).abs().max() / scale < 1e-12
    assert (d0.e * y - d0.n * x).abs().max() / (scale * 15e3) < 1e-12


def test_axis_plunges_toward_strike():
    """``dip``/``strike`` orient the axis as documented, checked independently of
    spheroid.m: a deep spheroid plunging 30 deg toward north (strike 0) matches
    the point ellipsoid whose north semi-axis is tilted down by the same angle
    (pECM ``omega_x = +30 deg`` takes the y axis to ``(0, cos, -sin)``), and not
    its mirror image."""
    n = 31
    ax = torch.linspace(-60e3, 60e3, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    x, y = xx.reshape(1, -1), yy.reshape(1, -1)
    tilt = math.radians(30.0)
    s = _enu(_src()(x, y, **_kw(depth=20e3, polar_radius=1500.0, equatorial_radius=400.0,
                                dip=tilt, strike=0.0, strength=1e7)))
    z0 = _f(1, 0)

    def pecm(omega_x):
        return _enu(PECMSource(poisson_ratio=NU, shear_modulus=MU)(
            x, y, z0, z0, _f(1, 20e3), _f(1, omega_x), z0, z0,
            _f(1, 400.0), _f(1, 1500.0), _f(1, 400.0), _f(1, 1e7)))

    same, mirror = pecm(tilt), pecm(-tilt)
    scale = same.abs().max()
    assert (s - same).abs().max() / scale < 1e-2
    assert (s - mirror).abs().max() / scale > 1e-1


# --------------------------------------------------------------------------- #
# Batchability
# --------------------------------------------------------------------------- #
def test_batched_matches_loop():
    """One batch mixing every path: generic prolate, oblate, exact / near
    vertical, exact / near sphere, horizontal, and a per-source location."""
    rows = [
        dict(polar_radius=2000.0, equatorial_radius=700.0, dip=0.9),
        dict(polar_radius=700.0, equatorial_radius=1500.0, dip=2.0),
        dict(polar_radius=1800.0, equatorial_radius=600.0, dip=math.pi / 2),
        dict(polar_radius=1800.0, equatorial_radius=600.0, dip=math.pi / 2 + 0.004),
        dict(polar_radius=900.0, equatorial_radius=900.0, dip=0.3),
        dict(polar_radius=903.0, equatorial_radius=900.0, dip=math.pi / 2 - 0.002),
        dict(polar_radius=1500.0, equatorial_radius=500.0, dip=0.0),
    ]
    B = len(rows)
    x, y = _grid(B, 9)
    kw = {k: torch.stack([_f(1, r.get(k, v))[0] for r in rows])
          for k, v in dict(source_x=0.0, source_y=0.0, depth=5000.0,
                           polar_radius=0.0, equatorial_radius=0.0, dip=0.0,
                           strike=0.6, strength=5e6).items()}
    kw["source_x"] = torch.linspace(-2e3, 2e3, B, dtype=DTYPE)
    src = _src()
    full = _enu(src(x, y, **kw))
    for i in range(B):
        one = _enu(src(x[i:i + 1], y[i:i + 1], **{k: v[i:i + 1] for k, v in kw.items()}))
        # an oblate row puts its whole sub-batch in complex arithmetic, so a
        # prolate row may differ from its solo (real) evaluation by round-off
        torch.testing.assert_close(full[i:i + 1], one, rtol=1e-10, atol=1e-14)


def test_broadcasts_single_source_over_batch():
    B = 3
    x, y = _grid(B, 7)
    out = _src()(x, y, **_kw(1))
    one = _src()(x[:1], y[:1], **_kw(1))
    for i in range(B):
        torch.testing.assert_close(_enu(out)[i], _enu(one)[0], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_bad_shape_raises():
    x, y = _grid(2, 5)
    with pytest.raises(ValueError):
        _src()(x, y, **_kw(3))


@pytest.mark.parametrize("name", ["depth", "polar_radius", "equatorial_radius"])
@pytest.mark.parametrize("bad", [0.0, -600.0])
def test_nonpositive_raises(name, bad):
    x, y = _grid(2, 5)
    with pytest.raises(ValueError, match=name):
        _src()(x, y, **_kw(2, **{name: bad}))


def test_above_surface_raises():
    x, y = _grid(2, 5)
    dip = _f(2, math.pi / 2)
    dip[1] = 0.0          # horizontal: half-height is b, buried
    _src()(x, y, **_kw(2, depth=1500.0, polar_radius=2000.0, equatorial_radius=500.0,
                       dip=_f(2, 0.0)))
    with pytest.raises(ValueError, match="free surface"):
        _src()(x, y, **_kw(2, depth=1500.0, polar_radius=2000.0,
                           equatorial_radius=500.0, dip=dip))


def test_bad_strength_type_raises():
    with pytest.raises(ValueError, match="strength_type"):
        SpheroidSource(strength_type="potency")


# --------------------------------------------------------------------------- #
# Differentiability  (unit scale: mu = 1, lengths ~ 1)
# --------------------------------------------------------------------------- #
_GRAD_PTS = [(3.0, 1.2), (-2.5, -0.4), (0.15, -0.09), (-1.0, 3.5)]


@pytest.mark.parametrize("a,b,dip,mode", [
    (2.0, 0.7, 0.96, "pressure"),                      # generic prolate
    (0.7, 1.5, 0.4, "pressure"),                       # oblate
    (2.0, 0.7, 0.0, "pressure"),                       # horizontal axis
    (2.0, 0.7, math.pi / 2, "pressure"),               # exactly vertical
    (2.0, 0.7, math.pi / 2 - 0.006, "volume"),         # inside the vertical band
    (1.0, 1.0, 0.9, "pressure"),                       # exact sphere
    (1.004, 1.0, 0.9, "volume"),                       # sphere band, prolate side
    (0.996, 1.0, math.pi / 2 + 0.004, "pressure"),     # both bands, oblate side
], ids=["generic", "oblate", "horizontal", "vertical", "vertical-band", "sphere",
        "sphere-band", "both-bands"])
def test_gradcheck(a, b, dip, mode):
    """Through every input, with the axis piercing point and a strike-line
    point among the observations."""
    strike, sx, sy, d = 0.7, 0.15, -0.09, 5.0
    pts = list(_GRAD_PTS)
    if abs(math.cos(dip)) < 1 - 1e-12:   # piercing point exists
        pts.append(_axis_point(-d / math.tan(dip), 0.0, strike, sx, sy))
    pts.append(_axis_point(4.0, 0.0, strike, sx, sy))
    x = _tt([[p[0] for p in pts]]).requires_grad_(True)
    y = _tt([[p[1] for p in pts]]).requires_grad_(True)
    prm = [_f(1, v).requires_grad_(True) for v in (sx, sy, d, a, b, dip, strike, 1.0)]
    model = SpheroidSource(poisson_ratio=NU, shear_modulus=1.0, strength_type=mode)

    def fn(*args):
        return _enu(model(*args))

    assert torch.autograd.gradcheck(fn, (x, y, *prm))


def test_gradients_finite_on_grid_through_source():
    """A grid through the centre hits the piercing point of a vertical axis and
    the strike line exactly; every gradient stays finite."""
    x, y = _grid(2, 11)
    prm = _kw(2, dip=math.pi / 2, strike=0.0)
    for v in prm.values():
        v.requires_grad_(True)
    _enu(_src()(x, y, **prm)).pow(2).sum().backward()
    for name, v in prm.items():
        assert torch.isfinite(v.grad).all(), name


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
def test_dtype_float32():
    B = 2
    x, y = _grid(B, 7, dtype=torch.float32)
    kw = {k: v.float() for k, v in _kw(B).items()}
    out = SpheroidSource(internal_dtype=torch.float32)(x, y, **kw)
    assert out.u.dtype == torch.float32
    assert torch.isfinite(out.u).all()


@pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
def test_runs_on_cuda():
    B = 3
    x, y = _grid(B, 9)
    kw = {k: v.cuda() for k, v in _kw(B, dip=torch.tensor(
        [0.9, math.pi / 2, 0.3], dtype=DTYPE)).items()}
    out = _src()(x.cuda(), y.cuda(), **kw)
    assert out.u.device.type == "cuda" and torch.isfinite(out.u).all()
