# torchdeform

![Tests](https://github.com/robertpop99/torchdeform/actions/workflows/tests.yml/badge.svg)

**Differentiable synthetic geophysical deformation and atmosphere in PyTorch.**

`torchdeform` generates synthetic InSAR-style data end-to-end — ground
deformation from analytic source models, projected to satellite line-of-sight,
converted to interferometric phase, plus turbulent and topography-correlated
atmosphere. Every step is batched and fully differentiable, so it works both as
a synthetic **data generator** (for training networks) and as differentiable
**layers/forward models** inside an optimisation or inversion loop.

📓 **Visual tour:** [`examples/tutorial.ipynb`](examples/tutorial.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertpop99/torchdeform/blob/main/examples/tutorial.ipynb)
walks through the whole library step by step with plots — sources,
LOS/phase/wrapping, atmosphere, the full pipeline, datasets, and covariance
fitting.

📗 **Hands-on guide:** [`examples/guide.ipynb`](examples/guide.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robertpop99/torchdeform/blob/main/examples/guide.ipynb)
a slower tutorial that explains in greater detail each
step — building a deformation signal, writing your own parameter priors, and
inverting an interferogram by gradient descent.

## Features

- **Source models** (`torchdeform.sources`): Mogi point source, the full Okada
  rectangular finite-fault dislocation (`OkadaSource`) and its surface-only
  fast path (`OkadaSourceSimple`), and a penny-shaped crack. For magmatic
  sources there is a compound-dislocation family — `PCDMSource` (point compound
  dislocation, a triaxial volcanic point source), `CDMSource` (its finite
  counterpart, with a `cdm_params_from_shape` helper and sphere/prolate/oblate/
  dyke/sill style presets), and `PECMSource` (point ellipsoidal cavity, a
  uniformly pressurised point ellipsoid). All differentiable
  in their parameters. The Okada sources take two optional gradient modes
  (default: plain autograd of the exact forward): `analytic_grad=True` returns
  closed-form Okada strains for gradients that stay accurate even at the singular
  fault geometries (vertical dip, on-fault points); `smooth_grad=True` instead
  smooths the singularities. See [Gradients](#gradients).
- **Observation operators** (`torchdeform.observation`): Sentinel-1 line-of-sight
  geometry and displacement ⇄ phase conversions, phase wrapping and a
  wrap-invariant loss.
- **Atmosphere** (`torchdeform.atmosphere`): spectral turbulent atmospheric phase
  screens (Kolmogorov / exponential) and topography-correlated stratified delay,
  plus covariance diagnostics (`covariance_vs_distance`,
  `fit_exponential_covariance`) for measuring and calibrating screen statistics.
- **Simulation helpers** (`torchdeform.simulation`): synthetic fractal DEMs *and
  real Copernicus GLO-30 topography* (`DEMPatchSampler`), plus parameter priors
  for randomised scene generation.
- **Geometry & core** (`torchdeform.geometry`, `torchdeform.core`): WGS84
  coordinate transforms and tensor-backed data structures (`Displacement`,
  `LOSVector`, `ECEF`, `Geodetic`).

## Installation

```bash
pip install git+https://github.com/robertpop99/torchdeform.git
```

Requires Python ≥ 3.11 and PyTorch.

CPU-only users should install PyTorch CPU before torchdeform:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you want to use real DEMs downloaded from Copernicus GLO-30, you need to also install rasterio:

```bash
pip install "torchdeform[dem] @ git+https://github.com/robertpop99/torchdeform.git"
```


## Quick start

End-to-end: a finite-fault rupture → LOS → wrapped interferometric phase with
atmosphere.

```python
import torch
from torchdeform import OkadaSourceSimple, los_vector
from torchdeform.observation.insar import to_phase, wrap_phase
from torchdeform.atmosphere import turbulent_aps, stratified_aps, orbital_ramp
from torchdeform.simulation import DEMPatchSampler, synthetic_dem, UniformPrior

B, rows, cols = 4, 500, 500
psizex, psizey = 100, 100 # metres per pixel

# Observation grid: East/North metres, flattened to [B, N]
ax = torch.arange(-cols / 2 * psizex, cols / 2 * psizex, psizex)
ay = torch.arange(-rows / 2 * psizey, rows / 2 * psizey, psizey)
yy, xx = torch.meshgrid(ay, ax, indexing="ij")
x_obs = xx.reshape(1, -1).expand(B, -1)
y_obs = yy.reshape(1, -1).expand(B, -1)

# 1. Surface deformation from an Okada fault (parameters batched over B)
src = OkadaSourceSimple()
disp = src(
    x_obs, y_obs,
    source_x=torch.zeros(B), source_y=torch.zeros(B),
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
#    (los_vector also takes look_side: +1 right-looking [default], -1 left-looking)
los = los_vector(heading_deg=torch.full((B,), -13.0),
                 incidence_deg=torch.full((B,), 39.0))
phase = to_phase(disp.to_los(los)).reshape(B, rows, cols)

# 3. Add a realistic background: orbital ramp + topo-correlated + turbulent APS
#    (in a calm scene the long-wavelength ramp dominates; turbulence is texture)
# Real topography from Copernicus GLO-30 (public, no auth; needs network +
# `torchdeform[dem]`). For a zero-setup offline DEM instead, swap this line for
# `dem = synthetic_dem(B, rows, cols, relief=600.0)`. See "Real topography" below.
dem = DEMPatchSampler.from_copernicus(4, patch_rows=rows, patch_cols=cols)(B)
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

## Gradients

Every model outputs a `Displacement` (the field values). The *derivatives* —
e.g. how the displacement changes with fault dip or depth — come from PyTorch's
autograd: mark the inputs you care about with `requires_grad_()`, build any
scalar from the output, and call `.backward()`. The derivatives then appear in
each input's `.grad`.

```python
import torch
from torchdeform import OkadaSource

# Fault parameters whose gradients we want. requires_grad_(True) tells PyTorch
# to track them; analytic_grad=True gives gradients that stay accurate even for
# near-vertical faults (where the default autograd is ill-conditioned).
dip   = torch.tensor([1.20], requires_grad=True)   # radians (~69 deg)
depth = torch.tensor([6000.0], requires_grad=True) # centroid depth, metres

g = torch.linspace(-2e4, 2e4, 32)                  # a 32x32 observation grid
xx, yy = torch.meshgrid(g, g, indexing="ij")
x_obs, y_obs = xx.reshape(1, -1), yy.reshape(1, -1)
z_obs = torch.zeros_like(x_obs)                    # surface

model = OkadaSource(analytic_grad=True)
disp = model(
    x_obs, y_obs, z_obs,
    source_x=torch.zeros(1), source_y=torch.zeros(1),
    dip=dip, strike=torch.full((1,), 0.5),
    centroid_depth=depth, length=torch.full((1,), 1e4),
    width=torch.full((1,), 5e3),
    disl1=torch.zeros(1), disl2=torch.ones(1), disl3=torch.zeros(1),  # dip-slip
)

# A scalar to differentiate. In an inversion this is your data misfit; here we
# just use the summed squared vertical displacement as an example.
objective = disp.u.pow(2).sum()
objective.backward()           # populates .grad on dip and depth

print(dip.grad, depth.grad)    # d(objective)/d(dip), d(objective)/d(depth)
```

The same pattern differentiates w.r.t. the observation coordinates
(`x_obs.requires_grad_(True)`), which gives the spatial strain — the analytic
backend returns these in closed form, matching Okada's published derivative
tables. To fit a model to data, wrap this in a `torch.optim` loop:

```python
opt = torch.optim.Adam([dip, depth], lr=1e-2)
for _ in range(200):
    opt.zero_grad()
    disp = model(x_obs, y_obs, z_obs, source_x=torch.zeros(1), ...)
    loss = ((disp.u - observed) ** 2).mean()
    loss.backward()
    opt.step()
```

For forward-only use (synthetic data generation), leave `analytic_grad=False`
(the default) — gradients aren't computed, and the analytic path costs nothing.

Need **second-order** information (Hessians — e.g. Laplace-approximation
uncertainties or Newton steps)? Take it with ordinary autograd double-backward
(`torch.autograd.grad(..., create_graph=True)`, then differentiate again) — but in
the **default** mode, not `analytic_grad`: the analytic backend is a custom
autograd function and is first-order only, so a Hessian through it raises rather
than silently returning a wrong value.

## Datasets

For training you usually don't want to wire the pipeline by hand each time.
`torchdeform.simulation` provides **generators** that compose priors, source
models and atmosphere, and thin **`Dataset`** wrappers over them. A dataset is
reproducible per index (seed `base_seed + i`) and yields *physical* samples
(unwrapped fields + labels); your ML-target encoding goes in a `transform`.

```python
import torch
from torchdeform import MogiSource, OkadaSourceSimple, okada_params_from_fault
from torchdeform.observation.insar import phase_to_unit_circle
from torchdeform.simulation import (
    ObservationGrid, SourceGenerator, DeformationGenerator, GeometryGenerator,
    AtmosphereGenerator, InterferogramGenerator, InsarDataset,
    UniformPrior, DEMPatchSampler, synthetic_dem,
    DEFAULT_MOGI_PRIOR, DEFAULT_EARTHQUAKE_PRIOR, DEFAULT_S1_GEOMETRY_PRIOR,
)

grid = ObservationGrid(64, 64, psizex=500.0, psizey=500.0)

# which source types to generate, and how often (location sampled per item)
deformation = DeformationGenerator(
    grid,
    sources={
        "mogi": SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR),
        "quake": SourceGenerator(
            OkadaSourceSimple(), DEFAULT_EARTHQUAKE_PRIOR,
            to_forward=okada_params_from_fault,
        ),
    },
    weights={"mogi": 1.0, "quake": 2.0},
)

geometry = GeometryGenerator(DEFAULT_S1_GEOMETRY_PRIOR)        # Sentinel-1 asc/desc

# Real topography as the DEM source: a DEMPatchSampler is a drop-in for the `dem=`
# callable. A stack of Copernicus tiles (fetched once) yields unlimited patches via
# random crops/flips. For a zero-setup offline DEM, use instead:
#   dem=lambda b, g: synthetic_dem(b, grid.rows, grid.cols, relief=600.0, generator=g)
dem_source = DEMPatchSampler.from_copernicus(8, patch_rows=grid.rows, patch_cols=grid.cols)

atmosphere = AtmosphereGenerator(
    grid,
    orbital_rms=UniformPrior(2.0, 5.0),
    turbulent_rms=UniformPrior(0.5, 1.5),
    strat_coeff=UniformPrior(-3e-3, 3e-3),
    dem=dem_source,
)

pipeline = InterferogramGenerator(deformation, geometry, atmosphere)

# the dataset yields physical samples; encode YOUR training targets in `transform`
def to_training(sample):
    image = phase_to_unit_circle(sample.wrapped().squeeze(0), channel_dim=0)  # (cos, sin), [2, H, W]
    label = sample.deformation.source_type[0]              # + normalised params, etc.
    return image, label

dataset = InsarDataset(pipeline, length=10_000, transform=to_training)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
images, labels = next(iter(loader))                       # [32, 2, 64, 64], 32 source-type labels
```

Each sample stores the **unwrapped** `deformation_phase` (plus `atmosphere`,
`los`, and source `params`/`source_type`/location); call `sample.wrapped()` for
the observable interferogram. Without a `transform`, `dataset[i]` returns the raw
sample for inspection. The same applies to `DeformationDataset` over a
`DeformationGenerator` if you only need the displacement field.

Mapping samples to normalised regression targets (sin/cos angles, log-scaled
depths, network head layout, ...) is intentionally left to your code — the
library produces physical quantities, not training tensors.

## Real topography

The stratified (topography-correlated) atmosphere is only as realistic as its
DEM. `synthetic_dem` fabricates plausible fractal terrain with zero setup, but you
can drop in **real elevation** anywhere a DEM is taken. `DEMPatchSampler` samples
random `[B, rows, cols]` patches from real rasters and is callable with the exact
`(batch, generator) -> dem` signature the generators expect — so it goes straight
into `AtmosphereGenerator(dem=...)`, or you can call it directly like
`synthetic_dem`:

```python
from torchdeform.simulation import (
    DEMPatchSampler, download_copernicus_glo30_tiles, AtmosphereGenerator, UniformPrior,
)

# A. Fetch random global land tiles straight into memory (needs network + rasterio):
dem = DEMPatchSampler.from_copernicus(8, patch_rows=grid.rows, patch_cols=grid.cols)

# B. Or download the tiles to disk once, then sample from them offline afterwards:
paths = download_copernicus_glo30_tiles(n=8)              # ~30 m global, public, no auth
dem = DEMPatchSampler.from_files(paths, patch_rows=grid.rows, patch_cols=grid.cols)

# C. Or use your own GeoTIFF / .npy / .npz rasters:
dem = DEMPatchSampler.from_files("my_dems/", patch_rows=grid.rows, patch_cols=grid.cols)

atmosphere = AtmosphereGenerator(grid, strat_coeff=UniformPrior(-3e-3, 3e-3), dem=dem)
dem(4)                                                    # -> [4, rows, cols] real terrain (m)
```

Copernicus GLO-30 is public and needs no login. Reading GeoTIFFs needs the
optional `rasterio` extra (`pip install 'torchdeform[dem]'`), while `.npy`/`.npz`
rasters load with numpy alone. A finite stack of tiles yields effectively
unlimited variety through random tile choice, crop location, flips and rotations,
and stays seed-reproducible (all randomness flows through the passed
`torch.Generator`, exactly like `synthetic_dem`).

## Priors

The `DEFAULT_*` priors (and the `DEFAULT_CDM_*` magmatic-style presets) are
**general-purpose, plug-and-use defaults** — sensible ranges to get a pipeline
running, *not* calibrated or authoritative values for any particular volcano,
fault, or region. Treat them as a starting point, and define your own whenever a
task needs specific ranges.

A prior bundle is a plain dataclass, so inspect any preset by printing it:

```python
from torchdeform.simulation import DEFAULT_MOGI_PRIOR, DEFAULT_CDM_PRIORS
print(DEFAULT_MOGI_PRIOR)            # fields and their ranges
print(DEFAULT_CDM_PRIORS["dyke"])    # styles: sphere / prolate / oblate / dyke / sill
```

Rolling your own is a one-liner — a bundle is just per-parameter distributions
named after the model's `forward` arguments:

```python
from torchdeform.simulation import MogiPrior, LogUniformPrior, SignedLogUniformPrior

my_prior = MogiPrior(
    depth=LogUniformPrior(500.0, 4_000.0),       # e.g. shallow sources only
    delta_v=SignedLogUniformPrior(1e5, 1e7),
)
my_prior.sample((8,))                            # -> {"depth": [8], "delta_v": [8]}
```

The same applies to `OkadaPrior`, `PCDMPrior`, `CDMPrior`, etc. Each field takes
any `Prior`, and there's a broad family to choose from: uniform on linear, log or
sign-symmetric-log scales (`UniformPrior`, `LogUniformPrior`,
`ReverseLogUniformPrior`, `SignedLogUniformPrior`), Gaussians and their truncated /
log / signed-log variants (`NormalPrior`, `TruncatedNormalPrior`, `LogNormalPrior`,
`SignedLogNormalPrior`), a `PowerLawPrior`, a `VonMisesPrior` for angles, discrete
and mixture priors (`ConstantPrior`, `ChoicePrior`, `MultimodalPrior`), or your own
`Prior` subclass — plus `make_prior` to build one from a short spec. The CDM
magmatic-style presets in particular use
torchdeform's own sampling ranges, not the inversion bounds of any study; see
[`docs/cdm_style_provenance.md`](docs/cdm_style_provenance.md) for what each
style means and where it comes from.

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
