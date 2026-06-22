"""
Tests for ``los_vector_from_satellite`` (exact satellite/ground LOS geometry).

The key property is that the ENU frame uses the geodetic (ellipsoidal) normal,
consistent with ``ecef_to_local_enu`` -- a satellite placed along the local
geodetic vertical must give an LOS of exactly ``(0, 0, 1)``, which would not
hold for a geocentric-radial "up".

Run with::

    pytest test_los_from_satellite.py -v
"""
import math

import torch

from torchdeform import Geodetic, ECEF, geodetic_to_ecef, los_vector_from_satellite


DTYPE = torch.float64


def _geodetic_up(lat_deg, lon_deg):
    φ = math.radians(lat_deg)
    λ = math.radians(lon_deg)
    return torch.tensor([math.cos(φ) * math.cos(λ),
                         math.cos(φ) * math.sin(λ),
                         math.sin(φ)], dtype=DTYPE)


def test_nadir_satellite_gives_pure_up():
    """Satellite along the geodetic normal -> LOS = (0, 0, 1)."""
    lat, lon = 45.0, 30.0
    ref = Geodetic.from_degrees(torch.tensor([lat]), torch.tensor([lon]),
                                torch.tensor([0.0]))
    g = geodetic_to_ecef(ref)                      # [1] per component
    ground = ECEF(g.x[:, None], g.y[:, None], g.z[:, None])   # [1, 1]

    up = _geodetic_up(lat, lon)
    sat = ECEF(g.x + 700_000.0 * up[0],
               g.y + 700_000.0 * up[1],
               g.z + 700_000.0 * up[2])            # [1]

    los = los_vector_from_satellite(sat, ground)
    assert los.e.abs().max() < 1e-9
    assert los.n.abs().max() < 1e-9
    assert torch.allclose(los.u, torch.ones_like(los.u), atol=1e-9)


def test_unit_norm():
    torch.manual_seed(0)
    ref = Geodetic.from_degrees(torch.tensor([12.0, -40.0]),
                                torch.tensor([100.0, 5.0]),
                                torch.tensor([0.0, 0.0]))
    g = geodetic_to_ecef(ref)
    # a small grid of ground points near each reference
    offs = torch.randn(2, 6, dtype=DTYPE) * 5_000.0
    ground = ECEF(g.x[:, None] + offs, g.y[:, None] + offs, g.z[:, None] + offs)
    # satellites ~700 km up and off to the side
    sat = ECEF(g.x + 2e5, g.y - 1e5, g.z + 7e5)

    los = los_vector_from_satellite(sat, ground)
    assert torch.allclose(los.norm, torch.ones_like(los.norm), atol=1e-9)


def test_points_toward_satellite():
    """Projection of (satellite - ground) onto the LOS is positive (range up)."""
    ref = Geodetic.from_degrees(torch.tensor([20.0]), torch.tensor([0.0]),
                                torch.tensor([0.0]))
    g = geodetic_to_ecef(ref)
    ground = ECEF(g.x[:, None], g.y[:, None], g.z[:, None])
    up = _geodetic_up(20.0, 0.0)
    sat = ECEF(g.x + 1e5 + 700_000.0 * up[0],
               g.y + 700_000.0 * up[1],
               g.z + 700_000.0 * up[2])
    los = los_vector_from_satellite(sat, ground)
    # LOS points ground->satellite, so up component dominates and is positive
    assert (los.u > 0).all()
