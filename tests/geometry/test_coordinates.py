"""
Tests for the WGS84 coordinate transforms (``geometry/coordinates.py``).

The transforms are part of the differentiable forward path (e.g. building
per-pixel ECEF positions for satellite-geometry LOS), so both numerical
correctness and gradient flow are checked:

* **Round trips** - geodetic -> ECEF -> geodetic recovers the input to
  sub-millimetre accuracy; local ENU -> ECEF -> local ENU recovers the
  horizontal coordinates.
* **Differentiability** - ``gradcheck`` passes through ``geodetic_to_ecef``,
  ``ecef_to_geodetic`` and ``ecef_to_local_enu``.

Run with::

    pytest test_coordinates.py -v
"""
import pytest
import torch

from torchdeform import (
    Geodetic,
    ECEF,
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_local_enu,
    local_enu_to_ecef,
    geodetic_to_local_enu,
    local_enu_to_geodetic,
)


DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Round-trip correctness
# --------------------------------------------------------------------------- #
def test_geodetic_ecef_round_trip():
    lat = torch.tensor([0.0, 12.5, -33.9, 51.5], dtype=DTYPE)
    lon = torch.tensor([0.0, -77.0, 151.2, -0.1], dtype=DTYPE)
    h = torch.tensor([0.0, 100.0, 2500.0, -50.0], dtype=DTYPE)

    geo = Geodetic(lat_deg=lat, lon_deg=lon, height_m=h)
    back = ecef_to_geodetic(geodetic_to_ecef(geo))

    assert torch.allclose(back.lat_deg, lat, atol=1e-9)
    assert torch.allclose(back.lon_deg, lon, atol=1e-9)
    assert torch.allclose(back.height_m, h, atol=1e-6)   # < 1 micron


def test_ecef_to_geodetic_stable_at_high_latitude():
    """Height stays accurate near and at the pole (pole-stable formula)."""
    lat = torch.tensor([85.0, 89.9, 90.0], dtype=DTYPE)
    lon = torch.tensor([10.0, -120.0, 0.0], dtype=DTYPE)
    h = torch.tensor([0.0, 1000.0, 3000.0], dtype=DTYPE)

    geo = Geodetic(lat_deg=lat, lon_deg=lon, height_m=h)
    back = ecef_to_geodetic(geodetic_to_ecef(geo))

    assert torch.isfinite(back.height_m).all()
    assert torch.allclose(back.lat_deg, lat, atol=1e-7)
    assert torch.allclose(back.height_m, h, atol=1e-6)


def test_known_ecef_value_at_equator():
    """At lat=lon=h=0 the point sits on the +X axis at the semi-major axis."""
    geo = Geodetic.from_degrees(0.0, 0.0, 0.0)
    ecef = geodetic_to_ecef(geo)
    assert ecef.x == pytest.approx(6378137.0, abs=1e-3)
    assert float(ecef.y) == pytest.approx(0.0, abs=1e-6)
    assert float(ecef.z) == pytest.approx(0.0, abs=1e-6)


def test_local_enu_round_trip():
    """ENU -> ECEF -> ENU recovers all three coordinates to machine precision.

    ``local_enu_to_ecef`` is the exact tangent-plane inverse of
    ``ecef_to_local_enu``, so the round trip is exact even for large offsets and
    a non-zero up component.
    """
    ref = Geodetic.from_degrees(
        torch.tensor([40.0, -10.0]),
        torch.tensor([20.0, 5.0]),
        torch.tensor([0.0, 1500.0]),
    )
    x_obs = torch.tensor([[0.0, 30_000.0, -50_000.0]], dtype=DTYPE).expand(2, -1).contiguous()
    y_obs = torch.tensor([[0.0, -20_000.0, 40_000.0]], dtype=DTYPE).expand(2, -1).contiguous()
    z_obs = torch.tensor([[0.0, 1_000.0, -300.0]], dtype=DTYPE).expand(2, -1).contiguous()

    ecef = local_enu_to_ecef(x_obs, y_obs, ref, z_obs)
    east, north, up = ecef_to_local_enu(ecef, ref)

    assert torch.allclose(east, x_obs, atol=1e-6)
    assert torch.allclose(north, y_obs, atol=1e-6)
    assert torch.allclose(up, z_obs, atol=1e-6)


def test_local_enu_to_ecef_defaults_up_to_zero():
    """Omitting z_obs places points on the tangent plane (up == 0)."""
    ref = Geodetic.from_degrees(torch.tensor([35.0]), torch.tensor([12.0]),
                                torch.tensor([0.0]))
    x_obs = torch.tensor([[0.0, 1000.0]], dtype=DTYPE)
    y_obs = torch.tensor([[0.0, -800.0]], dtype=DTYPE)

    ecef = local_enu_to_ecef(x_obs, y_obs, ref)
    _, _, up = ecef_to_local_enu(ecef, ref)
    assert up.abs().max() < 1e-6


def test_geodetic_local_enu_round_trip():
    """geodetic -> ENU -> geodetic recovers the input points."""
    ref = Geodetic.from_degrees(torch.tensor([40.0]), torch.tensor([20.0]),
                                torch.tensor([0.0]))
    pts = Geodetic.from_degrees(
        torch.tensor([[40.0, 40.1, 39.95]]),
        torch.tensor([[20.0, 20.2, 19.9]]),
        torch.tensor([[0.0, 500.0, 1200.0]]),
    )
    e, n, u = geodetic_to_local_enu(pts, ref)
    back = local_enu_to_geodetic(e, n, ref, u)

    assert torch.allclose(back.lat_deg, pts.lat_deg, atol=1e-9)
    assert torch.allclose(back.lon_deg, pts.lon_deg, atol=1e-9)
    assert torch.allclose(back.height_m, pts.height_m, atol=1e-5)


def test_geodetic_to_local_enu_reference_maps_to_origin():
    """The reference point itself maps to ENU (0, 0, 0)."""
    ref = Geodetic.from_degrees(torch.tensor([12.3]), torch.tensor([-4.5]),
                                torch.tensor([100.0]))
    pts = Geodetic.from_degrees(torch.tensor([[12.3]]), torch.tensor([[-4.5]]),
                                torch.tensor([[100.0]]))
    e, n, u = geodetic_to_local_enu(pts, ref)
    assert torch.allclose(torch.stack([e, n, u]), torch.zeros(3, 1, 1, dtype=DTYPE),
                          atol=1e-6)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_gradcheck_geodetic_to_ecef():
    lat = torch.tensor([40.0, -12.0], dtype=DTYPE, requires_grad=True)
    lon = torch.tensor([20.0, 5.0], dtype=DTYPE, requires_grad=True)
    h = torch.tensor([100.0, 2000.0], dtype=DTYPE, requires_grad=True)

    def f(la, lo, hh):
        return geodetic_to_ecef(Geodetic(la, lo, hh)).xyz

    assert torch.autograd.gradcheck(f, (lat, lon, h), atol=1e-4, rtol=1e-3)


def test_gradcheck_ecef_to_geodetic():
    # a plausible near-surface ECEF point
    geo = Geodetic.from_degrees(
        torch.tensor([35.0, -5.0]),
        torch.tensor([10.0, 140.0]),
        torch.tensor([500.0, 1200.0]),
    )
    ecef0 = geodetic_to_ecef(geo)
    x = ecef0.x.clone().detach().requires_grad_(True)
    y = ecef0.y.clone().detach().requires_grad_(True)
    z = ecef0.z.clone().detach().requires_grad_(True)

    def f(xx, yy, zz):
        g = ecef_to_geodetic(ECEF(xx, yy, zz))
        return torch.stack((g.lat_deg, g.lon_deg, g.height_m), dim=-1)

    # ECEF inputs are ~6.4e6 m, so the default eps=1e-6 is far below float64
    # resolution at that magnitude; use a 1 cm step for the finite differences.
    assert torch.autograd.gradcheck(f, (x, y, z), eps=1e-2, atol=1e-5, rtol=1e-3)


def test_gradcheck_ecef_to_local_enu():
    ref = Geodetic.from_degrees(
        torch.tensor([40.0]),
        torch.tensor([20.0]),
        torch.tensor([0.0]),
    )
    ecef0 = local_enu_to_ecef(
        torch.tensor([[0.0, 1500.0]], dtype=DTYPE),
        torch.tensor([[0.0, -800.0]], dtype=DTYPE),
        ref,
    )
    x = ecef0.x.clone().detach().requires_grad_(True)
    y = ecef0.y.clone().detach().requires_grad_(True)
    z = ecef0.z.clone().detach().requires_grad_(True)

    def f(xx, yy, zz):
        east, north, up = ecef_to_local_enu(ECEF(xx, yy, zz), ref)
        return torch.stack((east, north, up), dim=-1)

    # Large ECEF input magnitudes -> use a 1 cm finite-difference step.
    assert torch.autograd.gradcheck(f, (x, y, z), eps=1e-2, atol=1e-5, rtol=1e-3)
