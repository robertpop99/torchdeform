# torchdeform — project overview

**Differentiable synthetic geophysical deformation and atmosphere in PyTorch.**

Generates synthetic InSAR-style data end-to-end: analytic source models →
ground deformation → satellite line-of-sight → interferometric phase, plus
turbulent and topography-correlated atmosphere. Every step is **batched and
fully differentiable**, so it works both as a synthetic data generator (for
training networks) and as differentiable forward models inside an
optimisation/inversion loop.

- Python ≥ 3.11, PyTorch. Packaged under `src/` layout (`src/torchdeform`).
- Models compute internally in `float64` by default (accuracy of Okada/penny
  solvers); pass `internal_dtype`/`dtype` for float32.
- Visual tour / docs: `examples/tutorial.ipynb`. README has quick-start + dataset examples.

## Conventions
- Units: metres, radians inside models (`los_vector` takes degrees), phase in radians.
- Shapes: observation points `[B, N]`; per-image source params `[B]`;
  rasterised fields (DEM, atmosphere) `[B, rows, cols]`.
- Phase sign: `phase = -4π·d_los/λ`, LOS displacement positive toward satellite
  (Hanssen / standard-InSAR convention).

## Package layout (`src/torchdeform/`)

- **`__init__.py`** — re-exports the common API at top level. `__version__ = "0.1.0"`.

- **`core.py`** — tensor-backed dataclasses (`TensorDataclassMixin` gives
  `.to/.detach/.device/.dtype/.cpu/.cuda/.clone`):
  - `Displacement` (E/N/U fields, `.to_los(los)`)
  - `LOSVector` (`.project(disp)`, `.norm()`)
  - `ECEF`, `Geodetic` (coordinate containers, conversions between them).

- **`sources/`** — analytic deformation source models (all `nn.Module`,
  subclass `SourceModel` in `base.py`, return a `Displacement`):
  - `mogi.py` → `MogiSource` (point source)
  - `okada.py` → `OkadaSource` (full rectangular finite-fault dislocation),
    `OkadaSourceSimple` (surface-only fast path), `okada_params_from_fault`.
    Biggest file (~2600 lines). Reference Fortran: `scaffolding/DC3D.f90`
    (used to port the strain kernels).
    Two mutually-exclusive gradient modes on both classes (default: plain
    autograd of the exact forward): `smooth_grad=True` smooths singularities for
    finite (approximate) gradients; `analytic_grad=True` keeps exact forward
    values and returns the closed-form DC3D strain as the backward (kernels
    `ua_/ub_/uc_displacement_and_derivatives`) — exact obs-coordinate + source
    gradients even on singular fault planes, geometry/slips via autograd of the
    exact forward, dip via a wide central difference (accurate through vertical).
    `analytic_grad` routes through `_evaluate` + an autograd.Function; the
    forward-only path skips the strain so data-gen pays nothing.
    Only the finite rectangular `DC3D` is ported, **not** the point source
    `DC3D0` (deliberate): DC3D0's volumetric/tensile potencies are already
    covered by `MogiSource`/`PCDMSource`, and its only unique part — the point
    *shear* double-couple — is a far-field approximation, redundant for the
    near-field static regime InSAR observes. If parity is ever wanted, check it
    as a *test* (finite `DC3D` → `DC3D0` as `L,W→0`), not as a user-facing source.
  - `penny.py` → `PennySource` (penny-shaped crack)
  - `pcdm.py` → `PCDMSource` (point compound dislocation model)

- **`observation/`** — geometry + observation operators:
  - `los.py` → `los_vector`, `los_vector_per_pixel`, `los_vector_from_center`,
    `los_vector_from_center_curved`, `los_vector_from_satellite`; S1 geometry
    constants + `sample_s1_geometry`.
  - `insar.py` → `to_los`, `to_phase`, `phase_to_los`, `wrap_phase`,
    `add_wrap`, `subtract_wrap`, `phase_to_complex`, `phase_to_unit_circle`,
    `unit_circle_to_phase`, `wrapped_phase_loss` / `WrappedPhaseLoss`,
    `S1_C_BAND_WAVELENGTH`.

- **`atmosphere/atm.py`** — spectral turbulent APS + stratified delay:
  `turbulent_aps`/`TurbulentAPS` (Kolmogorov/exponential), `spectral_filter`,
  `correlated_noise_cholesky`, `stratified_aps`/`StratifiedAPS` (topo-correlated),
  `sample_stratified_coeff`, `orbital_ramp`, `atmospheric_phase_screen`,
  `covariance_vs_distance`, `fit_exponential_covariance`.

- **`simulation/`** — randomised scene generation for datasets:
  - `dem.py` → `synthetic_dem` (fractal DEMs)
  - `priors.py` → `Prior` family (`UniformPrior`, `LogUniformPrior`,
    `SignedLogUniformPrior`, `ConstantPrior`, `MultimodalPrior`, `make_prior`),
    per-source prior bundles (`MogiPrior`, `OkadaPrior`, `PennyPrior`,
    `PCDMPrior`, `GeometryPrior`, `LocationPrior`), `DEFAULT_*_PRIOR` presets
    (earthquake/dyke/sill/mogi/penny/pcdm/S1 geometry), mixtures.
  - `generators.py` → `ObservationGrid`, `SourceGenerator`,
    `DeformationGenerator` (+`DeformationSample`), `GeometryGenerator`,
    `AtmosphereGenerator`, `InterferogramGenerator` (+`InterferogramSample`),
    `centered_location`.
  - `datasets.py` → `DeformationDataset`, `InsarDataset` (torch Datasets;
    reproducible per index via `base_seed + i`; yield *physical* samples,
    ML-target encoding goes in a user `transform`).

- **`geometry/coordinates.py`** — WGS84 transforms: `geodetic_to_ecef`,
  `ecef_to_geodetic`, `ecef_to_local_enu`, `local_enu_to_ecef`,
  `geodetic_to_local_enu`, `local_enu_to_geodetic`.

## Tests (`tests/`)
Mirror the package: `sources/`, `observation/`, `atmosphere/`, `geometry/`,
`simulation/`, plus `test_pipeline_gradients.py` (end-to-end differentiability).
Correctness vs independent references, batching, and `torch.autograd.gradcheck`
(incl. near regularised singularities). Run with `pytest`.

## Other dirs / notes
- **`cdm/` and `turbulent/`** — original **MATLAB** source (`.m` files) being
  ported to Python. **Temporary scaffolding — likely to be deleted once porting
  is finished.** Not part of the package.
- `delete/` — scratch, ignore.
- `examples/tutorial.ipynb` — step-by-step visual walkthrough.
- CI: `.github/workflows/tests.yml` (tests) + `publish-testpypi.yml`.
- License: Apache 2.0.
