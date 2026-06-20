"""
Synthetic fractal digital elevation models (DEMs) for InSAR testing.

Terrain is generated as a fractional-Brownian (fractal) surface via spectral
synthesis -- filter white noise by a power-law amplitude ``k^(-beta/2)`` and
inverse-FFT -- the same trick as the turbulent atmosphere, but with a
terrain-like spectral slope. For a 2-D fractional-Brownian surface with Hurst
exponent H, the power spectrum goes as ``k^(-(2H+2))``, so ``beta = 2H + 2``
and the fractal dimension is ``D = 3 - H``. Natural terrain sits around
H ~ 0.5-0.9 (D ~ 2.1-2.5); higher H is smoother.

Optional extras: a regional planar tilt (``ramp``, peak-to-peak metres) and a
shift to positive, sea-level-referenced elevations (``positive``). Output is
``[B, rows, cols]`` in metres, ready to feed ``stratified_aps(...)`` so the
topography-correlated atmosphere can be exercised without real DEM data.

Notes
-----
* ``relief`` is the standard deviation of the *fractal* component (metres); a
  non-zero ``ramp`` adds to the total range on top of it.
* Self-contained (own spectral synthesis) so topography does not depend on the
  atmosphere module; it deliberately mirrors ``atm.py``'s approach.
"""
import math
from typing import Optional

import torch

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


def synthetic_dem(
    batch: int,
    rows: int,
    cols: int,
    *,
    relief: float = 1000.0,           # std of the fractal component (metres)
    hurst: float = 0.8,               # roughness; beta = 2*hurst + 2
    beta: Optional[float] = None,     # override the spectral slope directly
    ramp: float = 0.0,                # peak-to-peak regional tilt added (metres)
    base_elevation: float = 0.0,      # metres
    positive: bool = False,           # shift so min elevation == base_elevation
    psizex: float = 1.0,
    psizey: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate ``[batch, rows, cols]`` synthetic fractal DEMs (metres).

    With ``ramp == 0`` and ``positive == False`` the field has per-image std
    exactly ``relief`` and mean ``base_elevation``. With ``positive == True``
    each DEM is shifted so its minimum equals ``base_elevation`` (sea-level-like).
    """
    if beta is None:
        beta = 2.0 * float(hurst) + 2.0

    f = _fractal_surface(batch, rows, cols, beta, psizex, psizey,
                         generator, device, dtype)
    dem = f * relief

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

    if positive:
        dem = dem - dem.amin(dim=(-2, -1), keepdim=True) + base_elevation
    else:
        dem = dem + base_elevation
    return dem

