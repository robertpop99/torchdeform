# torchdeform

![Tests](https://github.com/robertpop99/torchdeform/actions/workflows/tests.yml/badge.svg)

**Differentiable synthetic geophysical deformation and atmosphere in PyTorch.**

`torchdeform` generates synthetic InSAR-style data end-to-end — ground
deformation from analytic source models, projected to satellite line-of-sight,
converted to interferometric phase, plus turbulent and topography-correlated
atmosphere. Every step is batched and fully differentiable, so it works both as
a synthetic **data generator** (for training networks) and as differentiable
**layers/forward models** inside an optimisation or inversion loop.

## Features

- **Source models** (`torchdeform.sources`): Mogi point source, the full Okada
  rectangular finite-fault dislocation (`OkadaSource`) and its surface-only
  fast path (`OkadaSourceSimple`), and a penny-shaped crack. All differentiable
  in their parameters, with a `training_safe=True` mode that smooths the Okada
  singularities for finite gradients.
- **Observation operators** (`torchdeform.observation`): Sentinel-1 line-of-sight
  geometry and displacement ⇄ phase conversions, phase wrapping and a
  wrap-invariant loss.
- **Atmosphere** (`torchdeform.atmosphere`): spectral turbulent atmospheric phase
  screens (Kolmogorov / exponential) and topography-correlated stratified delay.
- **Simulation helpers** (`torchdeform.simulation`): synthetic fractal DEMs and
  parameter priors for randomised scene generation.
- **Geometry & core** (`torchdeform.geometry`, `torchdeform.core`): WGS84
  coordinate transforms and tensor-backed data structures (`Displacement`,
  `LOSVector`, `ECEF`, `Geodetic`).

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.11 and PyTorch.

## Quick start

End-to-end: a finite-fault rupture → LOS → wrapped interferometric phase with
atmosphere.

```python
import torch
from torchdeform import OkadaSourceSimple, los_vector
from torchdeform.observation.insar import to_phase, wrap_phase
from torchdeform.atmosphere import turbulent_aps, stratified_aps, orbital_ramp
from torchdeform.simulation import synthetic_dem, UniformPrior

B, rows, cols = 4, 64, 64

# Observation grid: East/North metres, flattened to [B, N]
ax = torch.linspace(-15_000, 15_000, cols)
ay = torch.linspace(-15_000, 15_000, rows)
yy, xx = torch.meshgrid(ay, ax, indexing="ij")
x_obs = xx.reshape(1, -1).expand(B, -1)
y_obs = yy.reshape(1, -1).expand(B, -1)

# 1. Surface deformation from an Okada fault (parameters batched over B)
src = OkadaSourceSimple(training_safe=True)
disp = src(
    x_obs, y_obs,
    fault_x=torch.zeros(B), fault_y=torch.zeros(B),
    dip=torch.deg2rad(torch.full((B,), 45.0)),
    strike=torch.deg2rad(torch.full((B,), 30.0)),
    centroid_depth=UniformPrior(2000.0, 6000.0).sample((B,)),
    length=torch.full((B,), 8000.0),
    width=torch.full((B,), 5000.0),
    disl1=torch.full((B,), 0.5),   # strike-slip (m)
    disl2=torch.full((B,), 0.3),   # dip-slip (m)
    disl3=torch.zeros(B),          # opening (m)
)

# 2. Project to Sentinel-1 line of sight and convert to phase
los = los_vector(heading_deg=torch.full((B,), -13.0),
                 incidence_deg=torch.full((B,), 39.0))
phase = to_phase(disp.to_los(los)).reshape(B, rows, cols)

# 3. Add a realistic background: orbital ramp + topo-correlated + turbulent APS
#    (in a calm scene the long-wavelength ramp dominates; turbulence is texture)
dem = synthetic_dem(B, rows, cols, relief=600.0)
aps = orbital_ramp(B, rows, cols, rms=4.0)                              # long-wavelength trend
aps = aps + stratified_aps(dem, coeff=torch.full((B,), 3e-3))          # tracks topography
aps = aps + turbulent_aps(B, rows, cols, rms=0.8)                       # powerlaw: scale-free
# (the default powerlaw turbulence is fractal, so psizex/psizey have no effect; set them
#  to your ground sampling only for model="exponential" with a physical correlation_length)

# 4. Wrapped interferogram, [B, rows, cols]
ifg = wrap_phase(phase + aps)
```

Because the whole chain is differentiable, you can also backpropagate a loss on
`ifg` (or `phase`) all the way to the source/atmosphere parameters — see
`tests/test_pipeline_gradients.py`.

## Conventions

- **Units**: distances in metres, angles in radians inside the models
  (`los_vector` takes degrees), phase in radians.
- **Shapes**: observation points are batched as `[B, N]`; per-image source
  parameters are `[B]`; rasterised fields (DEM, atmosphere) are `[B, rows, cols]`.
- **Phase sign**: `phase = -4π·d_los/λ`, with LOS displacement positive toward
  the satellite (Hanssen / standard-InSAR convention).
- **Precision**: models compute internally in `float64` by default for accuracy
  (the Okada and penny solvers in particular). Pass `internal_dtype=torch.float32`
  to the source constructors, or a `dtype=` to the functional generators, if you
  need lighter/GPU-friendly precision.

## Testing

```bash
pytest
```

The suite checks correctness against independent references, batching, and —
central to this library — differentiability via `torch.autograd.gradcheck`,
including health checks near the regularised singularities.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
