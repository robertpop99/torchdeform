"""
Synthetic digital elevation models (DEMs) for InSAR testing.

Two terrain generators share one public entry point, ``synthetic_dem``, selected
by ``method``:

* ``method="ridged"`` (default) -- *realistic-looking* terrain: a ridged
  multifractal (summed octaves of ``(1 - |noise|)^2`` value noise, each weighted
  by the one above) that produces sharp ridgelines and asymmetric peaks/valleys,
  refined by domain warping (``warp``) for meandering ridges and, optionally, a
  mass-conserving thermal-erosion pass (``erosion_iters``) that rounds young
  ridges into mature, eroded massifs. Slower (~3x fBm; erosion much more), but
  looks like real topography.

* ``method="fbm"`` -- fractional-Brownian (fractal) surface via spectral
  synthesis: filter white noise by a power-law amplitude ``k^(-beta/2)`` and
  inverse-FFT, the same trick as the turbulent atmosphere but with a terrain-like
  spectral slope. For a 2-D fractional-Brownian surface with Hurst exponent H the
  power spectrum goes as ``k^(-(2H+2))``, so ``beta = 2H + 2`` and the fractal
  dimension is ``D = 3 - H``. Natural terrain sits around H ~ 0.5-0.9
  (D ~ 2.1-2.5); higher H is smoother. Cheap and statistically faithful, but its
  isotropic Gaussian structure reads as "cloudy blobs" rather than mountains.

Referencing to non-negative, sea-level-like elevations is controlled by
``positive`` (shift so the minimum sits at ``base_elevation``) or ``fold``
(reflect negative excursions upward, ``abs``, so valleys form sharp floors
instead of the broad basins a plain shift leaves -- more realistic for
high-relief terrain).

Both paths are batched, fully differentiable, seed-reproducible, and normalised
to a zero-mean / unit-std fractal field before scaling, so ``relief`` (std, m),
``ramp`` (peak-to-peak regional tilt, m), ``base_elevation`` and ``positive``
behave identically regardless of ``method``. Output is ``[B, rows, cols]`` in
metres, ready to feed ``stratified_aps(...)``.

Notes
-----
* ``relief`` is the standard deviation of the *fractal* component (metres); a
  non-zero ``ramp`` adds to the total range on top of it.
* Self-contained (own spectral / value-noise synthesis) so topography does not
  depend on the atmosphere module; the fBm path deliberately mirrors ``atm.py``.
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F

from ..core import DeviceLikeType


def _fractal_surface(batch, rows, cols, beta, psizex, psizey,
                     generator, device, dtype, eps=1e-12):
    """Zero-mean, unit-std fractal field ``[batch, rows, cols]``, PSD ~ k^-beta."""
    ky = 2.0 * math.pi * torch.fft.fftfreq(rows, d=psizey, device=device, dtype=dtype)
    kx = 2.0 * math.pi * torch.fft.rfftfreq(cols, d=psizex, device=device, dtype=dtype)
    k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    filt = torch.zeros_like(k)
    nz = k > 0
    filt[nz] = k[nz] ** (-beta / 2.0)
    filt[0, 0] = 0.0                                   # zero-mean surface

    w = torch.randn(batch, rows, cols, generator=generator, device=device, dtype=dtype)
    f = torch.fft.irfft2(torch.fft.rfft2(w) * filt, s=(rows, cols))
    f = f - f.mean(dim=(-2, -1), keepdim=True)
    return f / f.std(dim=(-2, -1), keepdim=True).clamp_min(eps)


def _normalize(f, eps=1e-12):
    """Zero-mean, unit-std per image ``[batch, rows, cols]``."""
    f = f - f.mean(dim=(-2, -1), keepdim=True)
    return f / f.std(dim=(-2, -1), keepdim=True).clamp_min(eps)


def _lattice_noise(batch, rows, cols, res_y, res_x, generator, device, dtype):
    """Smooth value noise: a random coarse lattice bicubically upsampled to grid."""
    res_y = max(2, int(res_y))
    res_x = max(2, int(res_x))
    lattice = torch.randn(batch, 1, res_y + 1, res_x + 1,
                          generator=generator, device=device, dtype=dtype)
    up = F.interpolate(lattice, size=(rows, cols), mode="bicubic", align_corners=True)
    return up[:, 0]


def _fractal_noise(batch, rows, cols, *, octaves, lacunarity, gain, ridged,
                   base_res, generator, device, dtype):
    """Multi-octave value noise ``[batch, rows, cols]`` (ridged multifractal if set)."""
    total = torch.zeros(batch, rows, cols, device=device, dtype=dtype)
    weight = torch.ones(batch, rows, cols, device=device, dtype=dtype)
    amp = 1.0
    res = float(base_res)
    maxres = float(min(rows, cols))
    for _ in range(octaves):
        r = min(res, maxres)
        n = _lattice_noise(batch, rows, cols, r, r, generator, device, dtype)
        if ridged:
            n = 1.0 - n.abs()          # fold to make ridges
            n = n * n                  # sharpen the crests
            n = n * weight             # multifractal: detail rides on the ridges
            weight = (n * 2.0).clamp(0.0, 1.0)
        total = total + amp * n
        amp *= gain
        res *= lacunarity
    return total


def _domain_warp(field, strength, base_res, generator, device, dtype):
    """Displace terrain coordinates by low-frequency noise -> meandering ridges."""
    batch, rows, cols = field.shape
    dx = _lattice_noise(batch, rows, cols, base_res, base_res, generator, device, dtype)
    dy = _lattice_noise(batch, rows, cols, base_res, base_res, generator, device, dtype)
    ys = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx[None] + strength * dx, gy[None] + strength * dy], dim=-1)
    warped = F.grid_sample(field[:, None], grid, mode="bilinear",
                           padding_mode="reflection", align_corners=True)
    return warped[:, 0]


def _shift(t, dy, dx):
    """Shift content by ``(dy, dx)`` cells with edge (replicate) padding."""
    p = F.pad(t, (1, 1, 1, 1), mode="replicate")
    rows, cols = t.shape[-2], t.shape[-1]
    return p[..., 1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]


# 8-neighbour offsets with their grid distances (orthogonal 1, diagonal sqrt(2)).
_EROSION_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                       (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
                       (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]


def _thermal_erosion(field, iters, talus, rate, eps=1e-12):
    """Mass-conserving thermal erosion: material whose downhill slope exceeds the
    talus angle slides to lower neighbours, rounding sharp ridges into mature,
    eroded-looking massifs. Uses 8 neighbours (distance-weighted) so it stays
    isotropic -- a 4-neighbour scheme stamps grid-aligned striations."""
    x = field[:, None]                                     # [B, 1, rows, cols]
    for _ in range(iters):
        # positive slope (drop per unit distance) toward each neighbour
        d = torch.stack([(x - _shift(x, dy, dx)).clamp_min(0.0) / dist
                         for dy, dx, dist in _EROSION_NEIGHBOURS], dim=0)
        dmax = d.amax(dim=0)
        give = (rate * (dmax - talus).clamp_min(0.0)) * (d / d.sum(0).clamp_min(eps))
        inflow = sum(_shift(give[k], -dy, -dx)
                     for k, (dy, dx, _) in enumerate(_EROSION_NEIGHBOURS))
        x = x - give.sum(0) + inflow
    return x[:, 0]


def synthetic_dem(
    batch: int,
    rows: int,
    cols: int,
    *,
    relief: float = 1000.0,           # std of the fractal component (metres)
    hurst: float = 0.8,               # roughness; beta = 2*hurst + 2  (fbm only)
    beta: Optional[float] = None,     # override the spectral slope directly (fbm)
    method: str = "ridged",           # "ridged" | "fbm"
    octaves: int = 8,                 # ridged: number of noise octaves
    lacunarity: float = 2.0,          # ridged: frequency step between octaves
    gain: float = 0.5,                # ridged: amplitude step between octaves
    base_res: int = 3,                # ridged: lattice cells of the coarsest octave
    warp: float = 0.2,                # ridged: domain-warp strength (0 disables)
    erosion_iters: int = 0,           # ridged: thermal-erosion passes (0 disables; ~40-80 for a mature look)
    erosion_talus: float = 0.02,      # ridged: talus slope threshold (std units)
    erosion_rate: float = 0.1,        # ridged: fraction moved per pass
    ramp: float = 0.0,                # peak-to-peak regional tilt added (metres)
    base_elevation: float = 0.0,      # metres
    positive: bool = False,           # shift so min elevation == base_elevation
    fold: bool = False,               # reflect (abs) for sharp valley floors; implies non-negative
    psizex: float = 1.0,
    psizey: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate ``[batch, rows, cols]`` synthetic DEMs (metres).

    ``method="ridged"`` (default) builds realistic-looking terrain from a ridged
    multifractal, domain-warped (``warp``) and optionally thermally eroded
    (``erosion_iters``); ``method="fbm"`` is the fast fractional-Brownian surface.
    The two methods share all output shaping: with ``ramp == 0``, ``positive ==
    False`` and ``fold == False`` the field has per-image std exactly ``relief``
    and mean ``base_elevation``. ``positive == True`` shifts each DEM so its
    minimum equals ``base_elevation``; ``fold == True`` reflects the surface
    (``abs``) so valleys become sharp floors rather than broad basins, also
    referenced to ``base_elevation`` (both make the output non-negative, and
    under either ``relief`` becomes an amplitude scale rather than the exact std).
    """
    if method == "fbm":
        if beta is None:
            beta = 2.0 * float(hurst) + 2.0
        f = _fractal_surface(batch, rows, cols, beta, psizex, psizey,
                             generator, device, dtype)
    elif method == "ridged":
        f = _fractal_noise(batch, rows, cols, octaves=octaves, lacunarity=lacunarity,
                           gain=gain, ridged=True, base_res=base_res,
                           generator=generator, device=device, dtype=dtype)
        if warp > 0.0:
            f = _domain_warp(f, warp, base_res, generator, device, dtype)
        if erosion_iters > 0:
            f = _thermal_erosion(_normalize(f), erosion_iters,
                                 erosion_talus, erosion_rate)
        f = _normalize(f)
    else:
        raise ValueError(f"method must be 'fbm' or 'ridged', got {method!r}")

    dem = f * relief
    if fold:
        dem = dem.abs()                                # sharp valley floors, non-negative

    if ramp:
        yy, xx = torch.meshgrid(
            torch.arange(rows, device=dem.device, dtype=dtype) * psizey,
            torch.arange(cols, device=dem.device, dtype=dtype) * psizex,
            indexing="ij",
        )
        xx = xx - xx.mean()
        yy = yy - yy.mean()
        theta = 2.0 * math.pi * torch.rand(batch, generator=generator,
                                           device=dem.device, dtype=dtype)
        plane = (torch.cos(theta)[:, None, None] * xx[None]
                 + torch.sin(theta)[:, None, None] * yy[None])
        ptp = (plane.amax(dim=(-2, -1), keepdim=True)
               - plane.amin(dim=(-2, -1), keepdim=True)).clamp_min(1e-12)
        dem = dem + plane / ptp * ramp

    if positive or fold:
        dem = dem - dem.amin(dim=(-2, -1), keepdim=True) + base_elevation
    else:
        dem = dem + base_elevation
    return dem
