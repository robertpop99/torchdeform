"""
Line-of-sight projection for synthetic Sentinel-1 InSAR training data.

Convention (verified against known Sentinel-1 ascending/descending vectors)
---------------------------------------------------------------------------
Unit LOS vector points GROUND -> SATELLITE, in an East/North/Up frame, so a
positive projected value means ground motion TOWARD the satellite (range
*decrease*). This is the Hanssen / standard-InSAR convention.

    look_az = heading + look_side * 90 deg      (right-looking => look_side=+1)
    los_e = sin(incidence) * sin(look_az)
    los_n = sin(incidence) * cos(look_az)
    los_u = cos(incidence)

where `heading` is the satellite flight azimuth in degrees clockwise from
North, and `incidence` is the radar incidence angle (angle from vertical).

Sanity values (right-looking, inc=39 deg):
    Ascending  heading ~ -12  deg  ->  (+0.62, +0.13, +0.78)   looks East
    Descending heading ~ -168 deg  ->  (-0.62, +0.13, +0.78)   looks West

"""

import math
from typing import Optional, Any

import torch
from torch import Tensor

from ..core import LOSVector, DeviceLikeType, ECEF
from ..geometry.coordinates import _ecef_to_geodetic

# ---------------------------------------------------------------------------
# Sentinel-1 IW realistic parameter ranges (for random sampling in training)
# ---------------------------------------------------------------------------
# Incidence angle across the IW swath: ~29 deg (near range) to ~46 deg (far
# range).
#
# Heading (flight azimuth, deg CW from North; here in the signed (-180, 180]
# convention, so headings sit just *west* of north -> negative). Sentinel-1 is
# near-polar (inclination 98.18 deg, retrograde), and the heading depends on the
# scene latitude phi via sin(A) = cos(i) / cos(phi): it is ~-8 deg ascending /
# ~-172 deg descending at the equator and steepens toward the poles
# (~-19 / -161 deg near 65 deg latitude). The ranges below span that global
# spread; narrow them to your area of interest's latitude band for tighter
# samples. A convenience sampler (sample_s1_geometry) picks a pass then draws
# uniformly within the band.
S1_INCIDENCE_RANGE_DEG = (29.0, 46.0)
S1_HEADING_ASCENDING_DEG = (-19.0, -8.0)       # equator (-8) -> ~65 deg lat (-19)
S1_HEADING_DESCENDING_DEG = (-172.0, -161.0)   # equator (-172) -> ~65 deg lat (-161)
S1_LOOK_SIDE = +1  # right-looking


def sample_s1_geometry(batch, generator: Optional[torch.Generator] = None, device: Optional[DeviceLikeType] = "cpu", dtype: torch.dtype = torch.float64,
                       p_ascending: float = 0.5):
    """Randomly sample (heading_deg, incidence_deg) per image for Sentinel-1.

    Picks a pass direction (ascending/descending) per image, jitters the
    heading around its nominal value, and draws incidence uniformly across
    the IW swath. Returns two [batch] tensors ready for los_vector().

    See also :data:`~torchdeform.simulation.DEFAULT_S1_GEOMETRY_PRIOR` for the
    composable typed-prior equivalent: it returns a dict that plugs straight into
    ``los_vector(**...)``, also yields ``look_side``, lets you reweight the
    ascending/descending split, and can be mixed with
    :class:`~torchdeform.simulation.PriorMixture`. This function is the quick
    one-call form.
    """
    def u(lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, generator=generator,
                                            device=device, dtype=dtype)

    asc = torch.rand(batch, generator=generator, device=device) < p_ascending
    h_asc = u(*S1_HEADING_ASCENDING_DEG, (batch,))
    h_desc = u(*S1_HEADING_DESCENDING_DEG, (batch,))
    heading = torch.where(asc.to(device), h_asc, h_desc)
    incidence = u(*S1_INCIDENCE_RANGE_DEG, (batch,))
    return heading, incidence


def _los_from_angles(heading_rad: Tensor, incidence_rad: Tensor,
                     look_side: int | Tensor) -> LOSVector:
    """Core formula. All args broadcastable tensors in radians.

    ``look_side`` may be a scalar (``+1`` right-looking, ``-1`` left-looking) or a
    per-image ``[B]`` tensor; in the latter case it is unsqueezed to ``[B, 1]`` so
    it broadcasts against the ``[B, 1]`` / ``[B, N]`` angle grids.
    """
    if isinstance(look_side, Tensor) and look_side.ndim == 1:
        look_side = look_side[:, None]
    look_az = heading_rad + look_side * (math.pi / 2.0)
    sin_i = torch.sin(incidence_rad)
    los_e = sin_i * torch.sin(look_az)
    los_n = sin_i * torch.cos(look_az)
    los_u = torch.cos(incidence_rad)
    # return los_e, los_n, los_u
    return LOSVector(
        e=los_e,
        n=los_n,
        u=los_u,
    )


# ---------------------------------------------------------------------------
# 1. Per-image: one heading + one incidence per image  ->  [B, 1] vectors
# ---------------------------------------------------------------------------
def los_vector(
    heading_deg: Any,        # [B]  flight azimuth, deg CW from North
    incidence_deg: Any,      # [B]  radar incidence angle, deg from vertical
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
    look_side: int | Tensor = S1_LOOK_SIDE,
) -> LOSVector:
    """One LOS vector per image, shaped [B, 1] to broadcast against [B, N]
    displacement fields. Use when a single incidence per scene is acceptable
    (small scenes; cross-swath incidence variation neglected)."""
    heading_deg = torch.as_tensor(heading_deg, device=device, dtype=dtype)
    incidence_deg = torch.as_tensor(incidence_deg, device=device, dtype=dtype)

    heading_deg = heading_deg[:, None]
    incidence_deg = incidence_deg[:, None]

    return _los_from_angles(
        torch.deg2rad(heading_deg), torch.deg2rad(incidence_deg), look_side
    )


# ---------------------------------------------------------------------------
# 2. Per-pixel: heading + incidence given at every observation point
# ---------------------------------------------------------------------------
def los_vector_per_pixel(
    heading_deg: Any,        # [B, N]  (or broadcastable to it)
    incidence_deg: Any,      # [B, N]
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
    look_side: int | Tensor = S1_LOOK_SIDE,
) -> LOSVector:
    """LOS vector evaluated independently at every pixel. Use when you have a
    per-pixel incidence (and possibly heading) raster -- the physically
    correct choice, since incidence varies a degree or two across the swath.
    Returns three [B, N] tensors aligned with the displacement field."""
    heading_deg = torch.as_tensor(heading_deg, device=device, dtype=dtype)
    incidence_deg = torch.as_tensor(incidence_deg, device=device, dtype=dtype)
    return _los_from_angles(
        torch.deg2rad(heading_deg), torch.deg2rad(incidence_deg), look_side
    )


# ---------------------------------------------------------------------------
# 3. Per-image center value, incidence varied across the scene by geometry
# ---------------------------------------------------------------------------
def los_vector_from_center(
    heading_deg: Any,         # [B]   flight azimuth, deg CW from North
    incidence_center_deg: Any,  # [B] incidence at the scene CENTER, deg
    x_obs: Any,               # [B, N]  same coords you feed Okada (meters)
    y_obs: Any,               # [B, N]
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
    look_side: int | Tensor = S1_LOOK_SIDE,
    sat_alt_m: float = 693_000.0,  # Sentinel-1 nominal orbit altitude ~693 km
) -> LOSVector:
    """Per-pixel LOS from a single center incidence, reconstructing how
    incidence changes across the scene from the imaging geometry.

    Incidence grows with ground-range distance from nadir. Approximating a
    flat Earth and a satellite at height H, the ground-range coordinate of a
    pixel at incidence theta is rho = H * tan(theta). The cross-range offset
    of a pixel relative to the scene center (projected onto the look/range
    direction) shifts rho, and we invert tan to get the local incidence:

        rho_center = H * tan(theta_center)
        rho_pixel  = rho_center + (range-direction offset from center)
        theta_pixel = atan(rho_pixel / H)

    The range (cross-track) direction on the ground is the look azimuth.
    Pixels are projected onto it to get their signed range offset. This
    captures the dominant near<->far incidence gradient (~a degree or two
    across an IW sub-scene) without needing a per-pixel incidence raster.

    Notes / assumptions:
      - Flat-Earth, straight-ray approximation: fine for the ~tens-of-km
        scenes typical of a single fault, not for full 250 km swaths.
      - x_obs/y_obs are East/North meters about the scene origin. If your
        scene center is not at (0,0), the mean of x_obs/y_obs is removed so
        "center incidence" is referenced to the centroid of the grid.
    """
    heading_deg = torch.as_tensor(heading_deg, device=device, dtype=dtype)
    incidence_center_deg = torch.as_tensor(incidence_center_deg, device=device, dtype=dtype)
    x_obs = torch.as_tensor(x_obs, device=device, dtype=dtype)
    y_obs = torch.as_tensor(y_obs, device=device, dtype=dtype)

    heading_rad = torch.deg2rad(heading_deg)
    look_az = heading_rad + look_side * (math.pi / 2.0)  # [B]

    # Unit vector of the ground-range (look) direction in ENU-horizontal.
    look_e = torch.sin(look_az)[:, None]  # [B,1]
    look_n = torch.cos(look_az)[:, None]  # [B,1]

    # Signed range offset of each pixel from the scene center, in meters.
    xc = x_obs - x_obs.mean(dim=1, keepdim=True)
    yc = y_obs - y_obs.mean(dim=1, keepdim=True)
    range_offset = xc * look_e + yc * look_n  # [B, N]

    # Reconstruct per-pixel incidence from center incidence + offset.
    H = sat_alt_m
    theta_c = torch.deg2rad(incidence_center_deg)            # [B]
    rho_c = H * torch.tan(theta_c)[:, None]               # [B,1] ground range at center
    rho_pix = rho_c + range_offset                        # [B,N]
    incidence_rad = torch.atan(rho_pix / H)               # [B,N]

    # Heading is constant across the (small) scene.
    return _los_from_angles(heading_rad[:, None], incidence_rad, look_side)


def los_vector_from_center_curved(
    heading_deg: Any,
    incidence_center_deg: Any,
    x_obs: Any,
    y_obs: Any,
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
    look_side: int | Tensor = S1_LOOK_SIDE,
    earth_radius_m: float = 6378137.0,
) -> LOSVector:
    """
    Curved-Earth version of los_vector_from_center.

    x_obs, y_obs are EN coordinates [B, N] in meters relative to the
    scene centre.

    Assumes:
        - spherical Earth
        - constant satellite heading across scene
        - incidence specified at scene centre

    Appropriate for scenes hundreds of km across.
    """
    heading_deg = torch.as_tensor(heading_deg,device=device,dtype=dtype)
    incidence_center_deg = torch.as_tensor( incidence_center_deg,device=device,dtype=dtype)

    x_obs = torch.as_tensor(x_obs,device=device,dtype=dtype)
    y_obs = torch.as_tensor(y_obs,device=device,dtype=dtype)

    heading_rad = torch.deg2rad(heading_deg)
    incidence_center_rad = torch.deg2rad(incidence_center_deg)

    # Look azimuth.
    look_az = heading_rad + look_side * (math.pi / 2.0)

    look_e = torch.sin(look_az)[:, None]
    look_n = torch.cos(look_az)[:, None]

    # Remove scene-centre offset.
    xc = x_obs - x_obs.mean(dim=1, keepdim=True)
    yc = y_obs - y_obs.mean(dim=1, keepdim=True)

    # Range-direction offset.
    range_offset = xc * look_e + yc * look_n

    R = earth_radius_m

    # Angular offset along Earth's surface.
    delta = range_offset / R

    # Centre incidence.
    theta0 = incidence_center_rad[:, None]

    # Curvature correction:
    #
    # local vertical rotates by delta
    #
    # near range -> incidence decreases
    # far range  -> incidence increases
    #
    incidence = theta0 + delta

    return _los_from_angles(
        heading_rad[:, None],
        incidence,
        look_side,
    )


def _los_vector_from_satellite(
    sat_xyz: Any,      # [B,3] satellite ECEF position (m)
    ground_xyz: Any,   # [B,N,3] ground ECEF coordinates (m)
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
) -> LOSVector:
    """
    Compute LOS vectors from satellite and ground positions.

    Parameters
    ----------
    sat_xyz
        Satellite ECEF coordinates [B,3].

    ground_xyz
        Ground ECEF coordinates [B,N,3].

    device

    dtype

    Returns
    -------
    LOSVector
        Unit LOS vectors in the local ENU frame.

    Notes
    -----
    The ENU frame uses the geodetic (ellipsoidal) normal at each ground point,
    consistent with :func:`~torchdeform.geometry.coordinates.ecef_to_local_enu`,
    so the incidence angle is referenced to the local vertical of the WGS84
    ellipsoid rather than the geocentric radial direction.
    """
    sat_xyz = torch.as_tensor(sat_xyz,device=device,dtype=dtype)
    ground_xyz = torch.as_tensor(ground_xyz,device=device,dtype=dtype)

    # Expand satellite position to pixels.
    sat_xyz = sat_xyz[:, None, :]  # [B,1,3]

    # Unit LOS in ECEF, pointing ground -> satellite.
    los_ecef = sat_xyz - ground_xyz  # [B,N,3]
    los_ecef = los_ecef / torch.linalg.norm(
        los_ecef,
        dim=-1,
        keepdim=True,
    )

    lx = los_ecef[..., 0]
    ly = los_ecef[..., 1]
    lz = los_ecef[..., 2]

    # Geodetic ENU basis at each ground point (ellipsoidal normal). This is the
    # transpose of the ECEF->ENU rotation in _ecef_to_local_enu, applied to a
    # direction (no translation), so the result is consistent with that frame.
    lat, lon, _ = _ecef_to_geodetic(ground_xyz)   # [B,N], radians
    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)
    sin_lon = torch.sin(lon)
    cos_lon = torch.cos(lon)

    los_e = -sin_lon * lx + cos_lon * ly
    los_n = -sin_lat * cos_lon * lx - sin_lat * sin_lon * ly + cos_lat * lz
    los_u = cos_lat * cos_lon * lx + cos_lat * sin_lon * ly + sin_lat * lz

    return LOSVector(
        e=los_e,
        n=los_n,
        u=los_u,
    )


def los_vector_from_satellite(satellite: ECEF, ground: ECEF) -> LOSVector:
    """Per-pixel LOS vectors from explicit satellite and ground ECEF positions.

    The most physically faithful entry point: no flat/curved-Earth incidence
    model, just the exact geometry between each ground point and the satellite,
    expressed in the local geodetic ENU frame (ground -> satellite, positive
    toward the satellite).

    Parameters
    ----------
    satellite : ECEF
        Satellite position, ``[B]`` per component (one per image).
    ground : ECEF
        Ground positions, ``[B, N]`` per component.

    Returns
    -------
    LOSVector
        Unit LOS vectors ``[B, N]`` in the local ENU frame.
    """
    dtype = satellite.dtype
    if dtype is None:
        dtype = torch.float64

    return _los_vector_from_satellite(
        sat_xyz=satellite.xyz,
        ground_xyz=ground.xyz,
        device=satellite.device,
        dtype=dtype,
    )
