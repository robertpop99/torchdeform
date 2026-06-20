from typing import Optional, Any

import torch
from torch import Tensor

from ..core import Geodetic, ECEF, DeviceLikeType

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _geodetic_to_ecef(
    lat_rad: Tensor,
    lon_rad: Tensor,
    h: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    lat, lon in radians
    h in meters

    Returns [...,3]
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
    x_obs: Any,       # [B,N]
    y_obs: Any,       # [B,N]
    reference: Geodetic,
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
) -> ECEF:
    """
    Local EN coordinates -> ECEF.

    Uses WGS84 ellipsoid.
    """
    device = device if device is not None else reference.device
    dtype = dtype if dtype is not None else reference.dtype

    x_obs = torch.as_tensor(x_obs, device=device, dtype=dtype)
    y_obs = torch.as_tensor(y_obs, device=device, dtype=dtype)

    lat0_deg = reference.lat_deg
    lon0_deg = reference.lon_deg
    height_m = reference.height_m

    lat0_deg = torch.as_tensor(
        lat0_deg,
        dtype=dtype,
        device=device,
    )[:, None]
    lat0 = torch.deg2rad(lat0_deg)

    lon0_deg = torch.as_tensor(
        lon0_deg,
        dtype=dtype,
        device=device,
    )[:, None]
    lon0 = torch.deg2rad(lon0_deg)

    h = torch.as_tensor(
        height_m,
        dtype=dtype,
        device=device,
    )

    # Small-angle geodesic displacement.

    sin_lat = torch.sin(lat0)

    N = WGS84_A / torch.sqrt(
        1.0 - WGS84_E2 * sin_lat**2
    )

    M = (
        WGS84_A * (1.0 - WGS84_E2)
        / (1.0 - WGS84_E2 * sin_lat**2) ** 1.5
    )

    lat = lat0 + y_obs / (M + h)

    lon = lon0 + x_obs / (
        (N + h) * torch.cos(lat0)
    )

    x, y, z =  _geodetic_to_ecef(
        lat,
        lon,
        h,
    )

    return ECEF(
        x=x,
        y=y,
        z=z,
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

    N = WGS84_A / torch.sqrt(
        1.0 - WGS84_E2 * sin_lat**2
    )

    h = p / torch.cos(lat) - N

    return lat, lon, h


def ecef_to_geodetic(ecef: ECEF) -> Geodetic:
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

    lat0 = torch.deg2rad(lat0_deg[:, None])
    lon0 = torch.deg2rad(lon0_deg[:, None])
    h = torch.as_tensor(height_m)

    center_x, center_y, center_z = _geodetic_to_ecef(
        lat0,
        lon0,
        h,
    )
    center = torch.stack([center_x, center_y, center_z], dim=-1)

    d = xyz - center[:, None, :]

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
    return _ecef_to_local_enu(
        xyz=ecef.xyz,
        lat0_deg=reference.lat_deg,
        lon0_deg=reference.lon_deg,
        height_m=reference.height_m,
    )
