"""
Synthetic digital elevation models (DEMs) for InSAR testing.

Three terrain generators share one public entry point, ``synthetic_dem``, selected
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

* ``method="spim"`` -- *stream-power incision*: an actual landscape-evolution
  model, ``dz/dt = U - K A^m S^n + D grad^2 z``, iterated from noise. What the
  noise-based methods cannot produce is **drainage connectivity**: real terrain is
  organised by branching valley networks that all drain somewhere, and no amount
  of spectral shaping gives you that. Here it emerges on its own, because
  incision is driven by upstream contributing area ``A``, so valleys compete,
  capture each other and merge. The pay-off is a landscape with the slope-area
  scaling ``S ~ A^(-m/n)`` measured on real rivers; the cost is a few hundred
  iterations (~9 s for a batch of 8 at 256x256 on CPU, vs milliseconds for fBm).
  Runs coarse then refines at higher resolution (see ``spim_res`` /
  ``spim_fine_res``), and crops off the base-level frame it evolves against.
  **Not a gradient path** -- unlike ``"fbm"`` and ``"ridged"`` it runs under
  ``no_grad`` and returns a leaf tensor. That is deliberate: a DEM is input
  *data* here (same status as ``DEMPatchSampler`` patches), and gradients through
  a landscape model would be meaningless anyway -- drainage networks reorganise
  discontinuously when a divide migrates or a basin is captured.

Referencing to non-negative, sea-level-like elevations is controlled by
``positive`` (shift so the minimum sits at ``base_elevation``) or ``fold``
(reflect negative excursions upward, ``abs``, so valleys form sharp floors
instead of the broad basins a plain shift leaves -- more realistic for
high-relief terrain).

All three paths are batched, seed-reproducible, and normalised to a zero-mean /
unit-std field before scaling, so ``relief`` (std, m), ``ramp`` (peak-to-peak
regional tilt, m), ``base_elevation`` and ``positive`` behave identically
regardless of ``method``. Output is ``[B, rows, cols]`` in metres, ready to feed
``stratified_aps(...)``.

Notes
-----
* ``relief`` is the standard deviation of the *fractal* component (metres); a
  non-zero ``ramp`` adds to the total range on top of it.
* Self-contained (own spectral / value-noise synthesis) so topography does not
  depend on the atmosphere module; the fBm path deliberately mirrors ``atm.py``.
* ``"fbm"`` and ``"ridged"`` are differentiable; ``"spim"`` is not (above).
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


def _shift(t, dy, dx, mode="replicate"):
    """Shift content by ``(dy, dx)`` cells. ``mode="replicate"`` repeats the border
    (and needs a 4-D ``[B, C, rows, cols]`` input); ``mode="constant"`` pads with
    zeros and works for any rank -- the SPIM helpers use it on ``[B, rows, cols]``
    fields, which is exact there because their boundary ring is pinned to zero."""
    p = F.pad(t, (1, 1, 1, 1), mode=mode)
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


# --------------------------------------------------------------------------- #
# Stream-power landscape evolution ("spim")
# --------------------------------------------------------------------------- #
# Grid-native, batched stand-ins for the three sequential algorithms a serious
# landscape-evolution model uses: D8 receiver graphs -> multiple-flow-direction
# weights, priority-flood depression filling -> min-pool relaxation, and the
# Braun-Willett stack traversal for drainage area -> a fixed-point iteration.
# None of them need gradients (see ``synthetic_dem``), so they favour the
# batched-tensor formulation over the asymptotically better serial one.

def _pin_ring(z, value=0.0):
    """Clamp the outermost ring of cells to ``value`` (base level / outlet).
    ``value`` may be a scalar or a same-shaped field to copy the ring from."""
    src = value if torch.is_tensor(value) else None
    z[:, 0, :] = src[:, 0, :] if src is not None else value
    z[:, -1, :] = src[:, -1, :] if src is not None else value
    z[:, :, 0] = src[:, :, 0] if src is not None else value
    z[:, :, -1] = src[:, :, -1] if src is not None else value
    return z


def _flow_weights(z, p, eps=1e-12):
    """Multiple-flow-direction (Freeman 1991) routing: each cell splits its water
    among downhill neighbours in proportion to ``(drop / distance) ** p``, with
    ``p -> inf`` recovering hard D8 and smaller ``p`` looking more braided.
    Returns ``(weights, drops)``, both ``[8, B, rows, cols]``. The drops are
    rescaled by their own per-cell maximum before the power: without that,
    ``p = 4`` on the ~1e-6 gradients that pit filling leaves across former lakes
    underflows to zero and flats stop routing entirely."""
    d = torch.stack([(z - _shift(z, dy, dx, mode="constant")).clamp_min(0.0) / dist
                     for dy, dx, dist in _EROSION_NEIGHBOURS], dim=0)
    wp = (d / d.amax(dim=0, keepdim=True).clamp_min(eps)).pow(p)
    # pits (no downhill neighbour) fall out as all-zero weights, which is correct:
    # they hold their water rather than routing it somewhere uphill.
    return wp / wp.sum(dim=0).clamp_min(eps), d


def _fill_pits(z, zf, iters, delta):
    """Depression filling by parallel relaxation of ``zf = max(z, minpool(zf) + delta)``
    -- the batched equivalent of priority flood. Raises every closed basin to its
    spill elevation so flow routes across it, leaving a ``delta``-per-cell gradient
    so filled lakes still have a downhill direction. ``zf=None`` cold-starts from
    ``+inf`` and converges downward in O(longest path); passing the previous
    surface warm-starts it, which is what the evolution loop does -- the fill
    barely moves between timesteps, so a handful of iterations tracks it."""
    zf = torch.full_like(z, float("inf")) if zf is None else torch.maximum(zf, z)
    _pin_ring(zf, z)                          # the ring is the outlet: no filling there
    for _ in range(iters):
        # min-pool: max_pool2d pads with -inf, so the negated pool ignores
        # out-of-grid neighbours instead of dragging edge cells down.
        lo = -F.max_pool2d(-zf[:, None], 3, stride=1, padding=1)[:, 0]
        zf = torch.maximum(z, lo + delta)
        _pin_ring(zf, z)
    return zf


def _drainage_area(w, area, iters):
    """Upstream contributing area (in cells) as the fixed point of ``A = 1 + W^T A``:
    every cell keeps its own unit area and gathers the fractions its neighbours
    send it. Warm-started -- ``area`` carries across evolution steps, so a few
    iterations track a slowly-moving fixed point instead of re-converging from
    scratch (~40x cheaper over a full run). Mind the shift sign: the gather is
    ``-dy, -dx``, mirroring ``inflow`` in ``_thermal_erosion``."""
    for _ in range(iters):
        give = w * area
        area = 1.0 + sum(_shift(give[k], -dy, -dx, mode="constant")
                         for k, (dy, dx, _) in enumerate(_EROSION_NEIGHBOURS))
    return area


def _laplacian(z):
    """4-neighbour Laplacian for the hillslope-diffusion term. Zero-padded, i.e. a
    Dirichlet-0 edge -- consistent with the ring being pinned to base level, which
    is re-imposed straight after the update anyway."""
    return (_shift(z, -1, 0, mode="constant") + _shift(z, 1, 0, mode="constant")
            + _shift(z, 0, -1, mode="constant") + _shift(z, 0, 1, mode="constant")
            - 4.0 * z)


def _spim_initial_state(batch, rows, cols, *, uplift_res, roughness, generator,
                        device, dtype, eps=1e-12):
    """Uplift field, starting surface and the routing state the loop warm-starts from."""
    # Uplift: low-frequency and non-negative, so mountain belts get large-scale
    # structure (uniform uplift gives a monotonous egg-crate).
    u = _lattice_noise(batch, rows, cols, uplift_res, uplift_res,
                       generator, device, dtype)
    u = u - u.amin(dim=(-2, -1), keepdim=True)
    u = u / u.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)

    # Initial surface: plain (non-ridged) broadband noise to nucleate valleys.
    # Ridged noise here would pre-stamp the structure the model exists to
    # generate. The *amplitude* matters more than it looks: it has to be a decent
    # fraction of the relief uplift will build (~uplift_rate x total time, O(1) at
    # the defaults), because it is the only thing breaking the symmetry of a
    # smooth uplift field on a square base-level frame. Too small (1e-3) and
    # valleys nucleate as a regular linear instability -- grid-aligned parallel
    # combs and 45-degree corner fans instead of dendritic networks.
    z = roughness * _fractal_noise(batch, rows, cols, octaves=8, lacunarity=2.0,
                                   gain=0.5, ridged=False, base_res=16,
                                   generator=generator, device=device, dtype=dtype)
    z = z - z.amin(dim=(-2, -1), keepdim=True)
    _pin_ring(z, 0.0)                                  # outlets on every edge

    zf = _fill_pits(z, None, max(rows, cols), z.new_tensor(1e-9))
    return z, u, torch.ones_like(z), zf


def _stream_power_evolve(z, u, area, zf, ring, *, steps, m, n, mfd_p, erodibility,
                         diffusion, uplift_rate, accum_iters, fill_iters, refresh,
                         cfl, eps=1e-12):
    """Iterate ``dz/dt = U - K A^m S^n + D grad^2 z`` until dendritic drainage
    emerges. Explicit stepping with ``dt`` capped by *both* the advective
    (``K A^m``) and diffusive (``4 D``) stability limits -- the diffusive one bites
    only for large ``diffusion`` but blows up identically, so both are checked.
    ``ring`` is the boundary condition re-imposed every step: ``0.0`` for the coarse
    pass (base level on all four edges), or a frozen field for the refinement pass,
    which inherits its frame from the coarse result. Pixel spacing is 1: the output
    is normalised, so only shape survives. Returns the evolved ``(z, area, zf)`` so
    the next resolution can warm-start from them."""
    w, d = _flow_weights(zf, mfd_p)
    dt_diffusive = cfl / (4.0 * diffusion) if diffusion > 0 else float("inf")
    dt = None

    for step in range(steps):
        if step % refresh == 0 or dt is None:
            # delta scales with the current relief so the tie-break gradient stays
            # negligible as the landscape grows (z starts ~1e-3, ends O(1)).
            delta = 1e-6 * (z.amax(dim=(-2, -1), keepdim=True)
                            - z.amin(dim=(-2, -1), keepdim=True)).clamp_min(eps)
            zf = _fill_pits(z, zf, fill_iters, delta)
            w, d = _flow_weights(zf, mfd_p)
            area = _drainage_area(w, area, accum_iters)
            # Recompute dt whenever A moves: the incision wave travels at K*A^m, and
            # explicit diffusion adds its own independent limit -- take the min of
            # both or raising `diffusion` blows up with the same checkerboard
            # signature that high-A cells produce. Per *image* (not a batch-wide
            # scalar): a shared dt would make every image step at the slowest one's
            # pace, so identical settings would evolve less far at larger batch.
            am = erodibility * area.pow(m).amax(dim=(-2, -1), keepdim=True)
            dt = (cfl / am.clamp_min(eps)).clamp_max(dt_diffusive)

        slope = (w * d).sum(dim=0).clamp_min(0.0)      # slope along the flow paths
        incision = erodibility * area.pow(m) * (slope if n == 1.0 else slope.pow(n))
        z = z + dt * (uplift_rate * u - incision + diffusion * _laplacian(z))
        z = z.clamp_min(0.0)                           # nothing erodes below base level
        _pin_ring(z, ring)                             # must be *after* the update

    return z, area, zf


def _spim_terrain(batch, rows, cols, *, res, fine_res, detail, margin, steps,
                  fine_steps, uplift_res, roughness, generator, device, dtype, **kw):
    """Two-resolution (coarse-to-fine) landscape evolution.

    The coarse pass costs O(steps x res^2) and is what actually organises the
    drainage network, so it runs on a small grid. Upsampling that straight to the
    output leaves planar, low-poly hillslopes: the surface simply has no structure
    between the coarse Nyquist and the output grid. Restoring it with noise does
    not work -- it reads as film grain over visible facets -- so the refinement
    pass re-runs the *same physics* at full resolution for a short while instead,
    warm-started from the upsampled state. That is Cordonnier's coarse-then-amplify
    pipeline with a physical amplifier."""
    # One scale factor for both axes -- capping each independently would squash a
    # 256x64 grid to 64x64 and the drainage network with it. The grids are sized so
    # that *after* the crop the fine pass still has at least the requested
    # resolution, and floored at 48 so a small request still gets a grid big enough
    # to grow a network on.
    long_axis = max(rows, cols)
    want = int(math.ceil(long_axis / max(1e-6, 1.0 - 2.0 * margin)))
    fine_long = min(int(fine_res), max(48, want))
    coarse_long = min(int(res), fine_long)

    def _grid(long_target):
        s = long_target / long_axis
        return max(16, int(round(rows * s))), max(16, int(round(cols * s)))

    lem_rows, lem_cols = _grid(coarse_long)
    z, u, area, zf = _spim_initial_state(batch, lem_rows, lem_cols,
                                         uplift_res=uplift_res, roughness=roughness,
                                         generator=generator, device=device,
                                         dtype=dtype)
    z, area, zf = _stream_power_evolve(z, u, area, zf, 0.0, steps=steps, **kw)

    if fine_steps > 0 and fine_long > coarse_long:
        fine_rows, fine_cols = _grid(fine_long)
        scale = (fine_rows * fine_cols) / (lem_rows * lem_cols)
        z, u, area, zf = (_upsample(t, fine_rows, fine_cols)
                          for t in (z, u, area, zf))
        z = z.clamp_min(0.0)
        _pin_ring(z, 0.0)
        area = area.clamp_min(1.0) * scale       # A is in cells; cells got smaller
        zf = torch.maximum(zf, z)
        _pin_ring(zf, z)
        z, area, zf = _stream_power_evolve(z, u, area, zf, 0.0,
                                           steps=fine_steps, **kw)

    # Every edge is pinned to base level, so the raw field is a bowl draining
    # outward on all four sides -- a frame that would be identical in every sample
    # and trivially learnable. Cropping it off leaves interior terrain with rivers
    # running out of frame, the way a real DEM patch looks.
    er, ec = z.shape[-2], z.shape[-1]
    my, mx = int(round(er * margin)), int(round(ec * margin))
    if my > 0 and mx > 0 and er - 2 * my >= 16 and ec - 2 * mx >= 16:
        z = z[:, my:er - my, mx:ec - mx]

    lem_res = max(z.shape[-2], z.shape[-1])
    if z.shape[-2:] != (rows, cols):
        z = _upsample(z, rows, cols)

    z = _normalize(z)
    if detail <= 0.0:
        return z
    return _spim_detail(z, detail, lem_res, generator, device, dtype)


def _upsample(t, rows, cols):
    return F.interpolate(t[:, None].contiguous(), size=(rows, cols),
                         mode="bicubic", align_corners=True)[:, 0]


def _spim_detail(z, detail, lem_res, generator, device, dtype, eps=1e-12):
    """Add roughness in the band the evolved grid cannot represent: above its
    Nyquist, below the output's. Only whole octaves that *fit* in that band are
    used -- ``base_res`` already sits near the output Nyquist, so the extra octaves
    a fractal would normally stack on top alias into blocky garbage. Amplitude
    tapers with elevation, keeping basin floors smooth (deposition fills them in
    real terrain) while high ground gets bare-rock texture."""
    batch, rows, cols = z.shape
    octaves = max(1, int(math.floor(math.log2(max(rows, cols) / max(1, lem_res)))) + 1)
    hf = _normalize(_fractal_noise(batch, rows, cols, octaves=octaves,
                                   lacunarity=2.0, gain=0.5, ridged=True,
                                   base_res=lem_res, generator=generator,
                                   device=device, dtype=dtype))
    w = z - z.amin(dim=(-2, -1), keepdim=True)
    w = (w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)).clamp_min(0.25)
    return z + detail * w * hf


def synthetic_dem(
    batch: int,
    rows: int,
    cols: int,
    *,
    relief: float = 1000.0,           # std of the fractal component (metres)
    hurst: float = 0.8,               # roughness; beta = 2*hurst + 2  (fbm only)
    beta: Optional[float] = None,     # override the spectral slope directly (fbm)
    method: str = "ridged",           # "ridged" | "fbm" | "spim"
    octaves: int = 8,                 # ridged: number of noise octaves
    lacunarity: float = 2.0,          # ridged: frequency step between octaves
    gain: float = 0.5,                # ridged: amplitude step between octaves
    base_res: int = 3,                # ridged: lattice cells of the coarsest octave
    warp: float = 0.2,                # ridged: domain-warp strength (0 disables)
    erosion_iters: int = 0,           # ridged: thermal-erosion passes (0 disables; ~40-80 for a mature look)
    erosion_talus: float = 0.02,      # ridged: talus slope threshold (std units)
    erosion_rate: float = 0.1,        # ridged: fraction moved per pass
    spim_steps: int = 400,            # spim: coarse-pass landscape-evolution iterations
    spim_fine_steps: int = 200,       # spim: refinement iterations at fine resolution (0 disables)
    spim_res: int = 112,              # spim: coarse-pass LEM resolution (long axis)
    spim_fine_res: int = 176,         # spim: refinement-pass resolution (long axis)
    spim_margin: float = 0.12,        # spim: fraction cropped per side (drops the base-level frame)
    spim_m: float = 0.5,              # spim: drainage-area exponent
    spim_n: float = 1.0,              # spim: slope exponent (m/n ~ 0.5 is the constrained ratio)
    spim_mfd_p: float = 4.0,          # spim: flow-routing exponent (higher -> closer to D8)
    spim_erodibility: float = 1.0,    # spim: K; only its ratio to diffusion/uplift matters
    spim_diffusion: float = 0.4,      # spim: hillslope diffusion D (higher -> broader, smoother valleys)
    spim_uplift_rate: float = 1.0,    # spim: U scale (sets relief before normalisation)
    spim_uplift_res: int = 3,         # spim: lattice cells of the uplift field
    spim_roughness: float = 1.0,      # spim: initial-surface amplitude; too small -> parallel, grid-aligned valleys
    spim_detail: float = 0.06,        # spim: amplitude of restored high-freq detail (0 disables)
    spim_accum_iters: int = 4,        # spim: drainage iterations per step (A is warm-started)
    spim_fill_iters: int = 2,         # spim: pit-fill relaxations per step (also warm-started)
    spim_refresh: int = 2,            # spim: recompute routing/drainage every k steps
    spim_cfl: float = 0.8,            # spim: timestep safety factor
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
    (``erosion_iters``); ``method="fbm"`` is the fast fractional-Brownian surface;
    ``method="spim"`` runs a stream-power landscape-evolution model (``spim_*``
    parameters) whose dendritic drainage networks the other two cannot produce --
    slower, and **not differentiable** (it runs under ``no_grad`` and returns a
    leaf tensor; see the module docstring for why that is the right contract).
    All three methods share the output shaping: with ``ramp == 0``, ``positive ==
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
    elif method == "spim":
        # not a gradient path (see the module docstring); no autograd graph either
        with torch.no_grad():
            f = _spim_terrain(batch, rows, cols, res=spim_res, fine_res=spim_fine_res, detail=spim_detail,
                              margin=spim_margin, steps=spim_steps,
                              fine_steps=spim_fine_steps, m=spim_m, n=spim_n,
                              mfd_p=spim_mfd_p, erodibility=spim_erodibility,
                              diffusion=spim_diffusion, uplift_rate=spim_uplift_rate,
                              uplift_res=spim_uplift_res, roughness=spim_roughness,
                              accum_iters=spim_accum_iters,
                              fill_iters=spim_fill_iters,
                              refresh=max(1, spim_refresh), cfl=spim_cfl,
                              generator=generator, device=device, dtype=dtype)
        f = _normalize(f)
    else:
        raise ValueError(f"method must be 'fbm', 'ridged' or 'spim', got {method!r}")

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
