"""
Synthetic turbulent atmospheric phase screens (APS) for InSAR training data.

This module generates the components of the no-deformation phase background:

* the *turbulent* component (:class:`TurbulentAPS`) -- the spatially-correlated,
  stochastic signal from 3-D turbulent mixing of water vapour;
* the *stratified* / topography-correlated component (:func:`stratified_aps`,
  :class:`StratifiedAPS`) -- the (per-interferogram) deterministic delay that
  tracks elevation, because the atmosphere is vertically stratified. Needs a DEM;
  and
* a long-wavelength *orbital ramp* (:func:`orbital_ramp`) -- the smooth low-order
  polynomial trend (residual orbit error etc.) that usually dominates the look of
  a real "noise-only" interferogram (a couple of widely-spaced fringes / slow
  gradient).

Add them for a full screen (see :func:`atmospheric_phase_screen`). A realistic
background is typically ``orbital_ramp + stratified + turbulent``, with the
turbulent part the smallest in a calm scene.

Why spectral synthesis instead of covariance + Cholesky
-------------------------------------------------------
The classic approach (build the n x n pixel covariance matrix, Cholesky-factor
it, multiply white noise) is exact but costs O(n^2) memory and O(n^3) compute,
where n = rows*cols. A 256x256 scene has n = 65 536 and an n x n double matrix
is ~34 GB -- unusable. It is provided here only as ``correlated_noise_cholesky``
for validation on tiny grids.

Instead we synthesise the field in the Fourier domain: filter white noise by
H(k) = sqrt(PSD(k)) and inverse-FFT. This is O(B * n * log n), a few MB of
memory, fully batched, and differentiable. Two power spectra are offered:

* ``model="powerlaw"`` (default, recommended for turbulent APS):
  PSD(k) ~ k^(-beta). Kolmogorov turbulence gives a 2-D slope beta = 8/3,
  i.e. a fractal screen -- the physically motivated choice (Hanssen, 2001).
* ``model="exponential"``: reproduces the covariance maxvar * exp(-alpha r)
  used by the original MATLAB ``pcmc_atm`` (covmodel_type=0), via the analytic
  2-D spectrum of an exponential covariance, H(k) = (alpha^2 + k^2)^(-3/4),
  alpha = 1 / correlation_length. Matches that covariance to a few percent.

Convention / units
------------------
Output is a real field ``[B, rows, cols]`` with per-image spatial RMS set by
``rms`` and zero mean (the DC component is removed). Interpret it in whatever
units you pass ``rms`` in -- typically **radians of interferometric phase**
(add it to your wrapped/unwrapped phase) or metres of slant-range delay.

Parameter guidance (Sentinel-1-ish)
-----------------------------------
* ``rms`` (per-image strength): turbulent APS is commonly a few mm to several
  cm of delay. In phase, delay d maps to 4*pi*d/lambda; for lambda = 5.55 cm,
  1 cm of delay ~ 2.3 rad. So calm scenes ~ 0.3-1 rad, active/humid ~ 2-10 rad.
  Pass a float for a fixed strength, or a ``[B]`` tensor to randomise per image.
* ``beta`` (power-law): 8/3 is the Kolmogorov default; empirically 2.0-3.5 all
  look like plausible APS (larger beta = smoother, more long-wavelength power).
* ``correlation_length`` (exponential model): the e-folding length of the
  covariance, i.e. 1/alpha. APS decorrelates over ~ a few km; pass it in the
  same length units as ``psizex/psizey``.
* ``psizex/psizey``: ground pixel spacing (metres). Sets the physical scale of
  the wavenumber grid, so correlation lengths come out in real units and
  anisotropic pixels are handled.

Stratified component
--------------------
``stratified_aps(dem, coeff, model=...)`` maps a DEM to a topo-correlated screen.
The coefficient is signed and varies per interferogram (the differential delay
between two dates can correlate either way with height), so randomise it.

* ``model="linear"``: ``coeff * (h - h_ref)``; ``coeff`` is a phase-elevation
  gradient in **rad / metre**. From the delay-elevation gradient g (commonly
  ~1-10 cm of delay per km of relief): coeff = (4*pi/lambda) * g. For Sentinel-1
  (lambda=5.55 cm) that is ~2e-3 to 2e-2 rad/m; sample signed, e.g. +/-5e-3.
* ``model="exponential"`` (Doin et al., 2009): ``coeff * (exp(-h/H_s) - ref)``;
  ``coeff`` is an amplitude in **radians** (a few rad), ``scale_height`` H_s is
  the tropospheric scale height ~ 6000 m. Higher topography -> less delay, so a
  positive amplitude gives delay decreasing with elevation.
* ``reference``: what to subtract so the screen is referenced sensibly --
  ``"mean"`` (default, zero-mean screen), ``"min"``, ``None`` (0), or a float
  reference elevation in metres.

Hole-effect covariance models
-----------------------------
The oscillating "hole-effect" covariances from the original MATLAB --
``expcos`` (``maxvar exp(-alpha r) cos(beta r)``) and ``ebessel``
(``maxvar exp(-alpha r) J0(2 pi r / w)``) -- are available, but only on the
covariance + Cholesky path (:func:`correlated_noise_cholesky`, via its ``model``
argument), since the spectral generator would need their 2-D PSDs. They are
niche; the power-law and exponential models cover the mainstream use case.

Calibrating from data
---------------------
:func:`covariance_vs_distance` estimates a field's radial autocovariance and
:func:`fit_exponential_covariance` fits ``maxvar exp(-alpha r)`` to it, so the
generator parameters (variance, ``correlation_length = 1/alpha``) can be measured
from a real interferogram and fed straight back into the exponential
:func:`turbulent_aps` / :func:`correlated_noise_cholesky`.
"""
import math
from typing import Optional, Union

import torch
import torch.nn as nn

from ..core import DeviceLikeType


# --------------------------------------------------------------------------- #
# Spectral filter
# --------------------------------------------------------------------------- #
def _radial_wavenumber(rows, cols, psizey, psizex, device, dtype):
    """Angular radial wavenumber |k| (rad / length) on the rfft2 grid."""
    ky = 2.0 * math.pi * torch.fft.fftfreq(rows, d=psizey, device=device, dtype=dtype)
    kx = 2.0 * math.pi * torch.fft.rfftfreq(cols, d=psizex, device=device, dtype=dtype)
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)   # [rows, cols//2+1]


def spectral_filter(
    rows: int,
    cols: int,
    psizex: float = 1.0,
    psizey: float = 1.0,
    model: str = "powerlaw",
    beta: float = 8.0 / 3.0,
    correlation_length: Optional[float] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """sqrt(PSD) filter on the rfft2 grid, shape ``[rows, cols//2+1]``.

    The absolute scale is irrelevant (the field is renormalised to ``rms``),
    only the shape in k matters. DC is zeroed so the field has zero mean.

    Parameters
    ----------
    rows, cols : int
        Grid shape of the field the filter will be applied to.
    psizex, psizey : float
        Pixel spacing along x/y (same length units as ``correlation_length``);
        sets the physical scale of the wavenumber grid.
    model : {'powerlaw', 'exponential'}
        Power-spectrum shape (see module docstring).
    beta : float
        Power-law slope (``model='powerlaw'``); 8/3 is Kolmogorov.
    correlation_length : float, optional
        e-folding length of the covariance (``model='exponential'``); required
        for that model, must be > 0.
    device, dtype
        Device/dtype of the returned filter.

    Returns
    -------
    torch.Tensor
        Real ``sqrt(PSD)`` filter on the rfft2 half-spectrum grid.
    """
    k = _radial_wavenumber(rows, cols, psizey, psizex, device, dtype)
    if model == "powerlaw":
        h = torch.zeros_like(k)
        nz = k > 0
        h[nz] = k[nz] ** (-beta / 2.0)
    elif model == "exponential":
        if correlation_length is None or correlation_length <= 0:
            raise ValueError("exponential model needs correlation_length > 0")
        alpha = 1.0 / float(correlation_length)
        h = (alpha * alpha + k * k) ** (-0.75)      # sqrt of 2-D PSD of exp(-a r)
    else:
        raise ValueError(f"unknown model {model!r} (use 'powerlaw' or 'exponential')")
    h[0, 0] = 0.0                                   # zero-mean field
    return h


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class TurbulentAPS(nn.Module):
    """Batched, differentiable turbulent atmospheric phase-screen generator.

    Precomputes the spectral filter once (like the deformation generators
    precompute their observation grid), then each call filters fresh white
    noise. Memory is O(B * rows * cols); compute is O(B * rows * cols * log).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        psizex: float = 1.0,
        psizey: float = 1.0,
        model: str = "powerlaw",
        beta: float = 8.0 / 3.0,
        correlation_length: Optional[float] = None,
        device: Optional[DeviceLikeType] = "cpu",
        internal_dtype: torch.dtype = torch.float64,
        num_eps: float = 1e-12,
    ):
        """Precompute the spectral filter for a fixed grid and spectrum.

        Parameters
        ----------
        rows, cols : int
            Output screen shape.
        psizex, psizey : float
            Ground pixel spacing (metres); see :func:`spectral_filter`.
        model : {'powerlaw', 'exponential'}
            Power-spectrum shape (see module docstring).
        beta : float
            Power-law slope (``model='powerlaw'``); 8/3 is Kolmogorov.
        correlation_length : float, optional
            e-folding length (``model='exponential'``), must be > 0.
        device : DeviceLikeType, optional
            Device the cached ``filt`` buffer lives on.
        internal_dtype : torch.dtype
            Dtype used for the filter and the FFTs in :meth:`forward`.
        num_eps : float
            Floor on the per-image std when renormalising to ``rms``.
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.internal_dtype = internal_dtype
        self.num_eps = num_eps
        filt = spectral_filter(rows, cols, psizex, psizey, model, beta,
                               correlation_length, device=device, dtype=internal_dtype)
        self.register_buffer("filt", filt)          # [rows, cols//2+1]

    def forward(
        self,
        batch: int,
        rms: Union[float, torch.Tensor] = 1.0,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[batch, rows, cols]`` screens with per-image spatial RMS ``rms``.

        ``rms`` may be a scalar or a ``[batch]`` tensor (randomise strength per
        image). Pass your own white ``noise`` ``[batch, rows, cols]`` to control
        reproducibility or to differentiate through the noise.
        """
        dtype = self.internal_dtype
        device = self.filt.device
        if noise is None:
            noise = torch.randn(batch, self.rows, self.cols, generator=generator,
                                device=device, dtype=dtype)
        else:
            noise = noise.to(device=device, dtype=dtype)

        spec = torch.fft.rfft2(noise) * self.filt          # [B, rows, cols//2+1]
        field = torch.fft.irfft2(spec, s=(self.rows, self.cols))   # [B, rows, cols]

        field = field - field.mean(dim=(-2, -1), keepdim=True)
        std = field.std(dim=(-2, -1), keepdim=True).clamp_min(self.num_eps)
        rms_t = torch.as_tensor(rms, device=device, dtype=dtype)
        if rms_t.ndim == 1:
            rms_t = rms_t.reshape(-1, 1, 1)
        return field / std * rms_t


def turbulent_aps(
    batch, rows, cols, *,
    rms: Union[float, torch.Tensor] = 1.0,
    psizex: float = 1.0,
    psizey: float = 1.0,
    model: str = "powerlaw",
    beta: float = 8.0 / 3.0,
    correlation_length: Optional[float] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional one-shot wrapper around :class:`TurbulentAPS`.

    Builds a :class:`TurbulentAPS` and calls it once; convenient when you don't
    want to keep the generator around. Constructor arguments (``rows``, ``cols``,
    ``psize*``, ``model``, ``beta``, ``correlation_length``) match
    :class:`TurbulentAPS`; ``batch``, ``rms``, ``generator`` and ``noise`` match
    :meth:`TurbulentAPS.forward`. Returns ``[batch, rows, cols]``.
    """
    gen = TurbulentAPS(rows, cols, psizex, psizey, model, beta,
                       correlation_length, device=device, internal_dtype=dtype)
    return gen(batch, rms=rms, generator=generator, noise=noise)


# --------------------------------------------------------------------------- #
# Faithful covariance + Cholesky reference  (SMALL GRIDS ONLY)
# --------------------------------------------------------------------------- #
def correlated_noise_cholesky(
    rows: int,
    cols: int,
    maxvar: float,
    alpha: float,
    N: int,
    model: str = "exponential",
    beta: Optional[float] = None,
    bessel_w: Optional[float] = None,
    psizex: float = 1.0,
    psizey: float = 1.0,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
    jitter: float = 1e-9,
) -> torch.Tensor:
    """Direct PyTorch port of ``pcmc_atm``: covariance + Cholesky synthesis.

    Builds the full ``n x n`` covariance for the chosen model and draws
    correlated noise via Cholesky. Returns ``[N, rows, cols]``.

    Models (``cov(r)``, with ``maxvar`` the zero-lag variance):

    * ``"exponential"`` -- ``maxvar * estatisticsxp(-alpha r)`` (covmodel_type 0).
    * ``"expcos"`` -- ``maxvar * exp(-alpha r) * cos(beta r)`` (covmodel_type 1):
      an oscillating "hole-effect" covariance; needs ``beta`` (the oscillation
      wavenumber, rad/length). NOTE: this is only positive-definite for a gentle
      oscillation (roughly ``beta <= alpha``); a pronounced hole makes the
      covariance invalid and the Cholesky will fail (this is why the original
      MATLAB effectively only ran the exponential model).
    * ``"ebessel"`` -- ``maxvar * exp(-alpha r) * J0(2 pi r / bessel_w)``
      (covmodel_type 2): exponential-Bessel hole-effect covariance; needs
      ``bessel_w`` (the oscillation wavelength). A valid covariance, though it may
      still need a larger ``jitter`` on some grids.

    Parameters
    ----------
    rows, cols : int
        Grid shape (``n = rows * cols``).
    maxvar : float
        Variance at zero lag (the covariance amplitude).
    alpha : float
        Inverse correlation length (1/length); larger = decorrelates faster.
    N : int
        Number of independent screens to draw.
    model : {'exponential', 'expcos', 'ebessel'}
        Covariance model (see above).
    beta : float, optional
        Oscillation wavenumber for ``model='expcos'`` (rad/length).
    bessel_w : float, optional
        Oscillation wavelength for ``model='ebessel'`` (same units as distance).
    psizex, psizey : float
        Pixel spacing used to compute inter-pixel distances ``r``.
    device, dtype, generator
        Standard tensor placement / RNG controls.
    jitter : float
        Diagonal loading (``jitter * maxvar``) added for numerical PD-ness. The
        hole-effect models are not guaranteed positive-definite, so the Cholesky
        can still fail on some grids -- increase ``jitter`` if so.

    WARNING: O(n^2) memory, O(n^3) compute with n = rows*cols. Usable only for
    tiny grids (<= ~64x64). Use :class:`TurbulentAPS` for anything real; this
    exists to validate the spectral generator and to exercise covariance models.
    """
    n = rows * cols
    yy, xx = torch.meshgrid(
        torch.arange(rows, device=device, dtype=dtype),
        torch.arange(cols, device=device, dtype=dtype),
        indexing="ij",
    )
    xv = (xx.reshape(n) * psizex)
    yv = (yy.reshape(n) * psizey)
    dx = xv[:, None] - xv[None, :]
    dy = yv[:, None] - yv[None, :]
    r = torch.sqrt(dx * dx + dy * dy)

    decay = maxvar * torch.exp(-alpha * r)
    if model == "exponential":
        vcm = decay
    elif model == "expcos":
        if beta is None:
            raise ValueError("model='expcos' needs beta (oscillation wavenumber)")
        vcm = decay * torch.cos(beta * r)
    elif model == "ebessel":
        if bessel_w is None or bessel_w <= 0:
            raise ValueError("model='ebessel' needs bessel_w > 0 (oscillation wavelength)")
        vcm = decay * torch.special.bessel_j0(2.0 * math.pi * r / bessel_w)
    else:
        raise ValueError(
            f"unknown model {model!r} (use 'exponential', 'expcos' or 'ebessel')"
        )

    # diagonal loading for numerical PD-ness (the matrix is only marginally PD,
    # and the hole-effect models may be indefinite)
    vcm = vcm + jitter * maxvar * torch.eye(n, device=device, dtype=dtype)
    L = torch.linalg.cholesky(vcm)                 # vcm = L L^T
    z = torch.randn(N, n, generator=generator, device=device, dtype=dtype)
    x = z @ L.T                                    # [N, n], cov(row) = vcm
    return x.reshape(N, rows, cols)


# --------------------------------------------------------------------------- #
# Stratified / topography-correlated component
# --------------------------------------------------------------------------- #
def stratified_aps(
    dem: torch.Tensor,                       # [H, W] or [B, H, W] elevation (metres)
    coeff: Union[float, torch.Tensor],       # [B] or scalar (rad/m linear; rad exp)
    model: str = "linear",
    scale_height: float = 6000.0,            # metres, exponential model only
    reference: Union[str, float, None] = "mean",
    device: Optional[DeviceLikeType] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Topography-correlated phase screen from a DEM. Returns ``[B, H, W]``.

    ``model="linear"`` -> ``coeff * (h - h_ref)`` (coeff in rad/m).
    ``model="exponential"`` -> ``coeff * (exp(-h/H_s) - t_ref)`` (coeff in rad).
    ``reference`` selects ``h_ref`` / ``t_ref``: per-image ``"mean"`` (default,
    zero-mean screen), ``"min"``, ``None`` (=0), or a reference elevation float.
    Differentiable in both ``dem`` and ``coeff``.

    Parameters
    ----------
    dem : torch.Tensor
        Elevation in metres, ``[H, W]`` or ``[B, H, W]``.
    coeff : float or torch.Tensor
        Signed per-image stratification coefficient (scalar or ``[B]``); rad/m
        for ``model='linear'``, rad for ``model='exponential'``. ``dem`` and
        ``coeff`` are broadcast to a common batch (mismatch raises ``ValueError``).
    model : {'linear', 'exponential'}
        Phase-elevation relation (see module docstring for guidance).
    scale_height : float
        Tropospheric scale height ``H_s`` in metres (exponential model only).
    reference : {'mean', 'min', None} or float
        What to subtract so the screen is referenced sensibly; a float is a
        reference elevation in metres.
    device, dtype
        Tensor placement for the computation/output.
    """
    dem = torch.as_tensor(dem, device=device, dtype=dtype)
    if dem.ndim == 2:
        dem = dem[None]                                   # [1, H, W]
    coeff = torch.as_tensor(coeff, device=dem.device, dtype=dtype).reshape(-1)

    bsz = max(dem.shape[0], coeff.shape[0])
    if dem.shape[0] == 1 and bsz > 1:
        dem = dem.expand(bsz, -1, -1)
    if coeff.shape[0] == 1 and bsz > 1:
        coeff = coeff.expand(bsz)
    if dem.shape[0] != coeff.shape[0]:
        raise ValueError(f"batch mismatch: dem {dem.shape[0]} vs coeff {coeff.shape[0]}")

    if model == "linear":
        field = dem
    elif model == "exponential":
        field = torch.exp(-dem / scale_height)
    else:
        raise ValueError(f"unknown model {model!r} (use 'linear' or 'exponential')")

    if reference == "mean":
        ref = field.mean(dim=(-2, -1), keepdim=True)
    elif reference == "min":
        ref = field.amin(dim=(-2, -1), keepdim=True)
    elif reference is None or reference == "none":
        ref = torch.zeros((), device=dem.device, dtype=dtype)
    else:  # a reference elevation in metres -> map through the same transform
        h_ref = torch.as_tensor(float(reference), device=dem.device, dtype=dtype)
        ref = h_ref if model == "linear" else torch.exp(-h_ref / scale_height)

    return coeff[:, None, None] * (field - ref)


class StratifiedAPS(nn.Module):
    """Module wrapper around :func:`stratified_aps` (symmetry with TurbulentAPS).

    Stores the model configuration so the screen can be produced from just
    ``(dem, coeff)`` at call time.
    """

    def __init__(
        self,
        model: str = "linear",
        scale_height: float = 6000.0,
        reference: Union[str, float, None] = "mean",
        internal_dtype: torch.dtype = torch.float64,
    ):
        """Configure the stratified model.

        Parameters
        ----------
        model : {'linear', 'exponential'}
            Phase-elevation relation (see :func:`stratified_aps`).
        scale_height : float
            Tropospheric scale height in metres (exponential model only).
        reference : {'mean', 'min', None} or float
            Reference subtracted from the transformed elevation.
        internal_dtype : torch.dtype
            Dtype used for the computation.
        """
        super().__init__()
        self.model = model
        self.scale_height = scale_height
        self.reference = reference
        self.internal_dtype = internal_dtype

    def forward(self, dem: torch.Tensor, coeff: Union[float, torch.Tensor]) -> torch.Tensor:
        """Return the stratified screen ``[B, H, W]`` for ``dem`` and ``coeff``.

        See :func:`stratified_aps` for the meaning of ``dem`` and ``coeff``.
        """
        return stratified_aps(dem, coeff, self.model, self.scale_height,
                              self.reference, device=None, dtype=self.internal_dtype)


def sample_stratified_coeff(
    batch: int,
    model: str = "linear",
    k_range=(-5e-3, 5e-3),                    # rad/m, linear model
    a_range=(-3.0, 3.0),                      # rad, exponential model
    generator: Optional[torch.Generator] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample a signed per-image stratification coefficient ``[batch]``.

    Parameters
    ----------
    batch : int
        Number of coefficients to draw.
    model : {'linear', 'exponential'}
        Selects which range is used: ``k_range`` for linear, ``a_range`` for
        exponential.
    k_range : tuple[float, float]
        Coefficient range for the linear model (rad/m); sampled uniformly.
    a_range : tuple[float, float]
        Amplitude range for the exponential model (rad); sampled uniformly.
    generator, device, dtype
        Standard RNG / tensor placement controls.

    Returns
    -------
    torch.Tensor
        ``[batch]`` signed coefficients, ready to pass as ``coeff`` to
        :func:`stratified_aps`.
    """
    lo, hi = k_range if model == "linear" else a_range
    return lo + (hi - lo) * torch.rand(batch, generator=generator,
                                       device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Orbital / long-wavelength ramp
# --------------------------------------------------------------------------- #
def orbital_ramp(
    batch: int,
    rows: int,
    cols: int,
    *,
    rms: Union[float, torch.Tensor] = 1.0,
    order: int = 1,
    coeff: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[DeviceLikeType] = "cpu",
    dtype: torch.dtype = torch.float64,
    num_eps: float = 1e-12,
) -> torch.Tensor:
    """Smooth long-wavelength phase ramp ``[batch, rows, cols]``.

    Models the residual orbital-error trend (and other very-long-wavelength
    background) that dominates the look of real no-deformation interferograms: a
    low-order 2-D polynomial across the scene. With ``order=1`` it is a random
    plane ``a*x + b*y``; with ``order=2`` it adds the quadratic terms
    ``x^2, y^2, x*y``. The DC term is dropped (zero mean) and the field is
    renormalised to per-image spatial RMS ``rms`` -- so it composes additively
    with :func:`turbulent_aps` / :func:`stratified_aps` in the same units.

    Coordinates are normalised to ``[-1, 1]`` across the scene, so the result is
    independent of pixel spacing (only the grid aspect ratio matters).

    Parameters
    ----------
    batch, rows, cols : int
        Output shape.
    rms : float or torch.Tensor
        Per-image spatial RMS of the ramp (scalar, or ``[batch]`` to randomise
        strength per image); same units as the other screens (typically radians).
    order : {1, 2}
        Polynomial order: 1 = planar (the dominant orbital term), 2 adds
        quadratic curvature.
    coeff : torch.Tensor, optional
        Polynomial coefficients ``[batch, K]`` (``K = 2`` for ``order=1``, ``5``
        for ``order=2``). Pass your own for reproducibility or to differentiate
        through them; otherwise standard-normal coefficients are drawn.
    generator, device, dtype, num_eps
        Standard RNG / tensor-placement controls; ``num_eps`` floors the per-image
        std during RMS renormalisation.

    Returns
    -------
    torch.Tensor
        Zero-mean ramp ``[batch, rows, cols]`` with per-image RMS ``rms``.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 (planar) or 2 (planar+quadratic), got {order!r}")

    ys = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")          # [rows, cols]

    terms = [xx, yy]
    if order == 2:
        terms += [xx * xx, yy * yy, xx * yy]
    basis = torch.stack(terms, dim=0)                       # [K, rows, cols]
    k = basis.shape[0]

    if coeff is None:
        coeff = torch.randn(batch, k, generator=generator, device=device, dtype=dtype)
    else:
        coeff = coeff.to(device=device, dtype=dtype)
        if coeff.shape != (batch, k):
            raise ValueError(f"coeff must have shape {(batch, k)} for order={order}, "
                             f"got {tuple(coeff.shape)}")

    field = torch.einsum("bk,khw->bhw", coeff, basis)       # [batch, rows, cols]

    field = field - field.mean(dim=(-2, -1), keepdim=True)
    std = field.std(dim=(-2, -1), keepdim=True).clamp_min(num_eps)
    rms_t = torch.as_tensor(rms, device=device, dtype=dtype)
    if rms_t.ndim == 1:
        rms_t = rms_t.reshape(-1, 1, 1)
    return field / std * rms_t


def atmospheric_phase_screen(
    turbulent: torch.Tensor,                  # [B, H, W] from TurbulentAPS
    dem: torch.Tensor,                        # [H, W] or [B, H, W]
    coeff: Union[float, torch.Tensor],
    model: str = "linear",
    scale_height: float = 6000.0,
    reference: Union[str, float, None] = "mean",
    orbital: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full screen = turbulent + stratified (+ optional orbital ramp), ``[B, H, W]``.

    Convenience adder for a precomputed turbulent screen, a fresh stratified one
    on the same grid, and an optional precomputed long-wavelength ramp. Combining
    all three is what reproduces the smooth, few-fringe look of real
    no-deformation interferograms.

    Parameters
    ----------
    turbulent : torch.Tensor
        Turbulent screen ``[B, H, W]`` (e.g. from :class:`TurbulentAPS`); its
        device/dtype are reused for the stratified part.
    dem : torch.Tensor
        Elevation ``[H, W]`` or ``[B, H, W]`` (metres).
    coeff : float or torch.Tensor
        Stratification coefficient; see :func:`stratified_aps`.
    model, scale_height, reference
        Forwarded to :func:`stratified_aps`.
    orbital : torch.Tensor, optional
        Long-wavelength ramp ``[B, H, W]`` (e.g. from :func:`orbital_ramp`) to add
        on top. Omitted by default.
    """
    strat = stratified_aps(dem, coeff, model, scale_height, reference,
                           device=turbulent.device, dtype=turbulent.dtype)
    total = turbulent + strat
    if orbital is not None:
        total = total + orbital
    return total


# --------------------------------------------------------------------------- #
# Covariance estimation (calibrate the generators from data)
# --------------------------------------------------------------------------- #
def covariance_vs_distance(
    field: torch.Tensor,
    psizex: float = 1.0,
    psizey: float = 1.0,
    n_bins: int = 50,
    max_distance: Optional[float] = None,
    demean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial autocovariance vs distance of a field (Wiener-Khinchine estimate).

    The inverse of the generators: given a screen (e.g. a real, deramped
    interferogram, or a synthetic one), estimate how its covariance decays with
    distance. The 2-D autocovariance is obtained spectrally on a zero-padded grid
    (so it is the *linear*, non-wrapping autocorrelation) and normalised by the
    per-lag sample overlap -- the unbiased estimate, free of the triangular taper
    of the naive ``|FFT|^2`` estimator -- then averaged into radial distance bins.

    Parameters
    ----------
    field : torch.Tensor
        ``[H, W]`` or ``[B, H, W]``.
    psizex, psizey : float
        Pixel spacing (sets the physical distance axis).
    n_bins : int
        Number of distance bins.
    max_distance : float, optional
        Largest distance binned; defaults to half the smaller scene dimension
        (the unbiased estimate gets noisy at large lags, where few pairs overlap).
    demean : bool, default True
        Remove each image's spatial mean first. This is the right choice for real
        data (the true mean is unknown), but note it removes long-wavelength power
        and so makes the *apparent* covariance decay faster than the underlying
        field's -- the recovered correlation length is biased short on a finite
        scene (more so the longer the true correlation). Set ``False`` to validate
        against a generator on an already-zero-mean synthetic field.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``distance`` ``[n_bins]`` (bin centres) and ``covariance`` shaped
        ``[n_bins]`` for a 2-D input or ``[B, n_bins]`` for a batch.
    """
    squeeze = field.ndim == 2
    if squeeze:
        field = field[None]
    if field.ndim != 3:
        raise ValueError(f"field must be [H, W] or [B, H, W], got {tuple(field.shape)}")

    B, H, W = field.shape
    dtype, device = field.dtype, field.device

    if demean:
        field = field - field.mean(dim=(-2, -1), keepdim=True)
    # zero-padded (linear) autocorrelation via FFT, zero lag at [0, 0]
    spec = torch.fft.rfft2(field, s=(2 * H, 2 * W))
    ac = torch.fft.irfft2(spec.real ** 2 + spec.imag ** 2, s=(2 * H, 2 * W))
    ac = ac[:, :H, :W]                                   # non-negative lags

    # unbiased normalisation: number of overlapping pairs at each lag
    oy = (H - torch.arange(H, device=device, dtype=dtype)).clamp_min(1.0)
    ox = (W - torch.arange(W, device=device, dtype=dtype)).clamp_min(1.0)
    ac = ac / (oy[:, None] * ox[None, :])               # [B, H, W], ac[:,0,0] = variance

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype) * psizey,
        torch.arange(W, device=device, dtype=dtype) * psizex,
        indexing="ij",
    )
    r = torch.sqrt(yy * yy + xx * xx).reshape(-1)       # [H*W]

    if max_distance is None:
        max_distance = 0.5 * min(H * psizey, W * psizex)
    bin_w = max_distance / n_bins
    # don't bin finer than the pixel spacing, else near-distance bins are empty
    # (the discrete lag grid has no samples between 0 and one pixel)
    min_bw = min(psizex, psizey)
    if bin_w < min_bw:
        bin_w = min_bw
        n_bins = max(1, int(math.ceil(max_distance / bin_w)))

    keep = r < max_distance
    r = r[keep]
    bin_idx = (r / bin_w).to(torch.long).clamp_(max=n_bins - 1)
    ac_flat = ac.reshape(B, -1)[:, keep]               # [B, M]

    counts = torch.zeros(n_bins, device=device, dtype=dtype).index_add_(
        0, bin_idx, torch.ones_like(r))
    sums = torch.zeros(B, n_bins, device=device, dtype=dtype).index_add_(
        1, bin_idx, ac_flat)
    cov = sums / counts.clamp_min(1.0)

    distance = (torch.arange(n_bins, device=device, dtype=dtype) + 0.5) * bin_w
    return distance, (cov[0] if squeeze else cov)


def fit_exponential_covariance(
    field: torch.Tensor,
    psizex: float = 1.0,
    psizey: float = 1.0,
    n_bins: int = 50,
    max_distance: Optional[float] = None,
    demean: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``maxvar * exp(-alpha r)`` to a field's radial autocovariance.

    Recovers the parameters of the exponential covariance model, ready to feed
    back into the generators: ``correlation_length = 1 / alpha`` for the
    exponential :func:`turbulent_aps`, and ``(maxvar, alpha)`` for
    :func:`correlated_noise_cholesky`.

    ``maxvar`` is the zero-lag variance (per image); ``alpha`` is recovered from
    the binned covariance curve (:func:`covariance_vs_distance`) by a stable,
    closed-form weighted log-linear fit -- ``log cov ~ log maxvar - alpha r``,
    weighted by the (positive) covariance so the reliable near field dominates
    and the noisy/negative tail is ignored.

    Parameters
    ----------
    field : torch.Tensor
        ``[H, W]`` or ``[B, H, W]``.
    psizex, psizey, n_bins, max_distance, demean
        Passed to :func:`covariance_vs_distance` (see its note on the finite-scene
        bias from ``demean``).
    eps : float
        Numerical floor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``variance`` and ``correlation_length`` (= ``1 / alpha``), each a scalar
        for a 2-D input or ``[B]`` for a batch.
    """
    squeeze = field.ndim == 2
    if squeeze:
        field = field[None]

    distance, cov = covariance_vs_distance(
        field, psizex, psizey, n_bins, max_distance, demean)
    if cov.ndim == 1:
        cov = cov[None]

    centred = field - field.mean(dim=(-2, -1), keepdim=True) if demean else field
    var = centred.pow(2).mean(dim=(-2, -1)).clamp_min(eps)   # zero-lag covariance

    r = distance[None]                                  # [1, n_bins]
    maxvar = var[:, None]                               # [B, 1]

    # weighted log-linear fit: log(cov) ~ log(maxvar) - alpha r, weighted by the
    # positive covariance (down-weights the noisy / negative far-field tail)
    w = cov.clamp_min(0.0)
    log_ratio = torch.log(cov.clamp_min(eps)) - torch.log(maxvar)
    num = (w * r * log_ratio).sum(dim=1)
    den = (w * r * r).sum(dim=1).clamp_min(eps)
    alpha = (-num / den).clamp_min(eps)

    corr_len = 1.0 / alpha
    if squeeze:
        return var[0], corr_len[0]
    return var, corr_len

