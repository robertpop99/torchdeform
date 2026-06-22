"""
Differentiable geodetic coordinate transforms on the WGS84 ellipsoid.

Conversions between the three coordinate representations used in torchdeform:

* :class:`~torchdeform.core.Geodetic` -- latitude/longitude/height,
* :class:`~torchdeform.core.ECEF` -- Earth-Centered Earth-Fixed Cartesian, and
* local East/North/Up (ENU) about a reference point.

All operations are pure tensor maths (no SciPy/pyproj), so they are batched and
differentiable end-to-end. The private ``_*`` helpers operate on raw tensors
(radians internally); the public functions take/return the
:mod:`torchdeform.core` dataclasses and handle the degree<->radian conversion.

ECEF<->geodetic uses Bowring's closed-form (non-iterative) approximation, which
is accurate to well under a millimetre for near-surface heights. The local-ENU
transforms (:func:`ecef_to_local_enu` / :func:`local_enu_to_ecef`) are an exact
tangent-plane (local Cartesian) pair and are exact inverses of each other.
"""
from typing import Optional, Any

import torch
from torch import Tensor

from ..core import Geodetic, ECEF, DeviceLikeType

# WGS84 reference ellipsoid parameters.
WGS84_A = 6378137.0                     # semi-major axis (metres)
WGS84_F = 1.0 / 298.257223563           # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)    # first eccentricity squared


def _geodetic_to_ecef(
    lat_rad: Tensor,
    lon_rad: Tensor,
    h: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    lat, lon in radians, h in metres.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ECEF ``(x, y, z)`` in metres, each shaped like the inputs.
    """

    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)

    sin_lon = torch.sin(lon_rad)
    cos_lon = torch.cos(lon_rad)

    N = WGS84_A / torch.sqrt(
        1.0 - WGS84_E2 * sin_lat**2
    )

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (
        N * (1.0 - WGS84_E2) + h
    ) * sin_lat

    return x, y, z


def geodetic_to_ecef(geo: Geodetic) -> ECEF:
    """Convert :class:`Geodetic` (lat/lon/height) to :class:`ECEF` on WGS84.

    Parameters
    ----------
    geo : Geodetic
        Latitude/longitude in degrees, height in metres.

    Returns
    -------
    ECEF
        Cartesian coordinates in metres, shape matching the inputs.
    """
    x, y, z = _geodetic_to_ecef(
        lat_rad=torch.deg2rad(geo.lat_deg),
        lon_rad=torch.deg2rad(geo.lon_deg),
        h=geo.height_m,
    )
    return ECEF(
        x=x,
        y=y,
        z=z,
    )


def local_enu_to_ecef(
    x_obs: Any,                 # [B,N] east
    y_obs: Any,                 # [B,N] north
    reference: Geodetic,
    z_obs: Optional[Any] = None,   # [B,N] up (defaults to 0)
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
) -> ECEF:
    """
    Local East/North/Up coordinates -> ECEF (exact tangent-plane transform).

    The exact inverse of :func:`ecef_to_local_enu`: it applies the transpose of
    the ECEF->ENU rotation about the ``reference`` point and adds the reference's
    ECEF position. Given ``e, n, u = ecef_to_local_enu(p, ref)``, the call
    ``local_enu_to_ecef(e, n, ref, u)`` recovers ``p`` to machine precision.

    Parameters
    ----------
    x_obs, y_obs : array-like
        Local East/North offsets [B, N] (metres) from the per-image reference.
    reference : Geodetic
        Per-image ENU origin, ``[B]`` per component.
    z_obs : array-like, optional
        Local Up offsets [B, N] (metres). Defaults to zero, i.e. points on the
        tangent plane through the reference.

    Returns
    -------
    ECEF
        Cartesian coordinates [B, N] in metres.

    Notes
    -----
    This is a tangent-plane (local Cartesian) frame, so ``z_obs`` is height above
    the plane through the reference, not ellipsoidal height; a point with
    ``z_obs = 0`` far from the reference therefore sits slightly below the
    ellipsoid (by ``~d^2 / 2R``). This is the standard ENU convention and matches
    :func:`ecef_to_local_enu`.
    """
    device = device if device is not None else reference.device
    dtype = dtype if dtype is not None else reference.dtype

    e = torch.as_tensor(x_obs, device=device, dtype=dtype)
    n = torch.as_tensor(y_obs, device=device, dtype=dtype)
    u = (torch.zeros_like(e) if z_obs is None
         else torch.as_tensor(z_obs, device=device, dtype=dtype))

    # Reference position and orientation. [:, None] so the per-image [B] values
    # broadcast against the [B, N] observation grid.
    ref = geodetic_to_ecef(reference)
    lat0 = torch.deg2rad(
        torch.as_tensor(reference.lat_deg, device=device, dtype=dtype)
    )[:, None]
    lon0 = torch.deg2rad(
        torch.as_tensor(reference.lon_deg, device=device, dtype=dtype)
    )[:, None]

    sin_lat = torch.sin(lat0)
    cos_lat = torch.cos(lat0)
    sin_lon = torch.sin(lon0)
    cos_lon = torch.cos(lon0)

    # ENU -> ECEF offset: transpose of the rotation used in _ecef_to_local_enu.
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    return ECEF(
        x=ref.x[:, None] + dx,
        y=ref.y[:, None] + dy,
        z=ref.z[:, None] + dz,
    )


def _ecef_to_geodetic(
    xyz: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    xyz [...,3]

    Returns
    -------
    lat_rad
    lon_rad
    h
    """

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (
        WGS84_A**2 - b**2
    ) / b**2

    p = torch.sqrt(x**2 + y**2)

    theta = torch.atan2(
        z * WGS84_A,
        p * b,
    )

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    lat = torch.atan2(
        z + ep2 * b * sin_t**3,
        p - WGS84_E2 * WGS84_A * cos_t**3,
    )

    lon = torch.atan2(y, x)

    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)

    N = WGS84_A / torch.sqrt(
        1.0 - WGS84_E2 * sin_lat**2
    )

    # Pole-stable height. Avoids the h = p / cos(lat) - N form (which blows up as
    # cos(lat) -> 0). From p = (N + h) cos(lat) and z = (N(1-e^2) + h) sin(lat):
    #   p * cos(lat) + z * sin(lat) = N (1 - e^2 sin^2(lat)) + h
    h = p * cos_lat + z * sin_lat - N * (1.0 - WGS84_E2 * sin_lat**2)

    return lat, lon, h


def ecef_to_geodetic(ecef: ECEF) -> Geodetic:
    """Convert :class:`ECEF` Cartesian coordinates to :class:`Geodetic` on WGS84.

    Uses Bowring's closed-form inverse with a pole-stable height formula (no
    division by ``cos(lat)``), so it is well behaved at all latitudes. Returns
    latitude/longitude in degrees and ellipsoidal height in metres, with shape
    matching the input.
    """
    xyz = ecef.xyz
    lat, long, h = _ecef_to_geodetic(xyz)
    return Geodetic(
        lat_deg=torch.rad2deg(lat),
        lon_deg=torch.rad2deg(long),
        height_m=h,
    )


def _ecef_to_local_enu(
    xyz: Tensor,         # [B,N,3]
    lat0_deg: Tensor,    # [B]
    lon0_deg: Tensor,    # [B]
    height_m: Tensor | float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns

    east [B,N]
    north [B,N]
    up [B,N]
    """

    lat0 = torch.deg2rad(lat0_deg[:, None])   # [B, 1]
    lon0 = torch.deg2rad(lon0_deg[:, None])   # [B, 1]
    h = torch.as_tensor(height_m, dtype=lat0.dtype, device=lat0.device)
    if h.ndim == 1:
        h = h[:, None]                        # [B, 1], matches lat0/lon0

    center_x, center_y, center_z = _geodetic_to_ecef(
        lat0,
        lon0,
        h,
    )
    center = torch.stack([center_x, center_y, center_z], dim=-1)   # [B, 1, 3]

    d = xyz - center                          # [B, N, 3] - [B, 1, 3]

    sin_lat = torch.sin(lat0)
    cos_lat = torch.cos(lat0)

    sin_lon = torch.sin(lon0)
    cos_lon = torch.cos(lon0)

    east = (
        -sin_lon * d[..., 0]
        + cos_lon * d[..., 1]
    )

    north = (
        -sin_lat * cos_lon * d[..., 0]
        - sin_lat * sin_lon * d[..., 1]
        + cos_lat * d[..., 2]
    )

    up = (
        cos_lat * cos_lon * d[..., 0]
        + cos_lat * sin_lon * d[..., 1]
        + sin_lat * d[..., 2]
    )

    return east, north, up


def ecef_to_local_enu(ecef: ECEF, reference: Geodetic) -> tuple[Tensor, Tensor, Tensor]:
    """Express ECEF coordinates as local East/North/Up about a reference point.

    Parameters
    ----------
    ecef : ECEF
        Points to convert, shape ``[B, N]`` per component.
    reference : Geodetic
        Per-image ENU origin, shape ``[B]`` per component.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(east, north, up)`` in metres, each shaped ``[B, N]``.
    """
    return _ecef_to_local_enu(
        xyz=ecef.xyz,
        lat0_deg=reference.lat_deg,
        lon0_deg=reference.lon_deg,
        height_m=reference.height_m,
    )


def geodetic_to_local_enu(
    geo: Geodetic,
    reference: Geodetic,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert geodetic points to local East/North/Up about a reference point.

    Convenience composition of :func:`geodetic_to_ecef` and
    :func:`ecef_to_local_enu`.

    Parameters
    ----------
    geo : Geodetic
        Points to convert, ``[B, N]`` per component.
    reference : Geodetic
        Per-image ENU origin, ``[B]`` per component.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(east, north, up)`` in metres, each shaped ``[B, N]``.
    """
    return ecef_to_local_enu(geodetic_to_ecef(geo), reference)


def local_enu_to_geodetic(
    x_obs: Any,
    y_obs: Any,
    reference: Geodetic,
    z_obs: Optional[Any] = None,
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
) -> Geodetic:
    """Convert local East/North/Up offsets to geodetic lat/lon/height.

    Inverse of :func:`geodetic_to_local_enu`; convenience composition of
    :func:`local_enu_to_ecef` and :func:`ecef_to_geodetic`. Arguments match
    :func:`local_enu_to_ecef`.

    Returns
    -------
    Geodetic
        Latitude/longitude in degrees and ellipsoidal height in metres,
        ``[B, N]`` per component.
    """
    ecef = local_enu_to_ecef(x_obs, y_obs, reference, z_obs, device, dtype)
    return ecef_to_geodetic(ecef)
