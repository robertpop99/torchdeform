# Source-model reference values

External ground truth for the source models, in the same spirit as the
Okada (1985) Table 2 checklist used by `test_okada_source.py`. Most of these
papers published no fixed numeric table, so the "tables" here are produced by
running the **original authors' code** — MATLAB for the volcanic sources, and
Okada's own Fortran `DC3D` for the finite fault at depth — and freezing the
output as JSON under [`../data/`](../data). The Python ports reproduce these
values to machine precision (relative error ~1e-13 for the Nikkhoo models,
~1e-11 for Fialko, ~1e-11 for DC3D).

Regenerating requires MATLAB (volcanic sources) or a Fortran compiler (DC3D).
The JSON files in `../data/` are committed, so the tests do **not** need either
to run.

## Models and provenance

| Data file | Generator | Reference code | Paper |
|---|---|---|---|
| `data/nikkhoo_golden.json` | `gen_nikkhoo.m` | `nikkhoo/{pCDM,CDM,pECM}.m` (vendored) | Nikkhoo, Walter, Lundgren & Prats-Iraola (2017), *Compound dislocation models (CDMs) for volcano deformation analyses*, GJI 208(2), 877–894 |
| `data/fialko_golden.json` | `gen_fialko.m` | `penny/*.m` (**download separately**) | Fialko, Khazan & Simons (2001), *Deformation due to a pressurized horizontal circular crack in an elastic half-space*, GJI 146(1), 181–190 |
| `data/dc3d_golden.json` | `gen_dc3d.py` | `DC3D.f90` (**not redistributed**) | Okada (1992), *Internal deformation due to shear and tensile faults in a half-space*, BSSA 82(2), 1018–1040 (`DC3D` routine) |
| `data/pcdm_volume_golden.json` | `gen_pcdm_volume.py` | `nikkhoo/pCDM.m` (vendored) | Nikkhoo et al. (2017), as above (random-volume forward check for `PCDMSource`) |
| `data/cdm_volume_golden.json` | `gen_cdm_volume.py` | `nikkhoo/CDM.m` (vendored) | Nikkhoo et al. (2017), as above (random-volume forward check for `CDMSource`) |
| `data/pecm_volume_golden.json` | `gen_pecm_volume.py` | `nikkhoo/pECM.m` (vendored) | Nikkhoo et al. (2017), as above (random-volume forward check for `PECMSource`) |
| `data/penny_volume_golden.json` | `gen_penny_volume.py` | `penny/*.m` (**download separately**) | Fialko et al. (2001), as above (random-volume forward check for `PennySource`) |

`nikkhoo/` holds the **pristine official** `pCDM.m`, `CDM.m` and `pECM.m` from
Nikkhoo's `CDM_ECM_RD.zip` (release `2022_Oct_24`, from
<https://volcanodeformation.com/software>), byte-identical to the official
files except for line endings normalised to LF.

Fialko's penny-crack code is **not redistributed here** (see Licensing). To
regenerate `fialko_golden.json` / `penny_volume_golden.json`, download the
**original** into a local `penny/` subdirectory (git-ignored):

```bash
wget http://igppweb.ucsd.edu/~fialko/Assets/Software/penny.tar.gz
```

Use the original, **not** the
[GeodMod mirror](https://github.com/falkamelung/GeodMod/tree/master/deformation_sources/penny):
the two use different array-shape conventions. The original `Q.m` takes a
**scalar** radius `r` (with `t` the node vector) and its `intgr.m` already has the
correct per-radius `Uz`; the GeodMod mirror vectorises `Q` over the radii and
carries a bug in that vectorised `Uz` line (see below). `gen_penny_volume.py`
targets the original (per-radius) form; `gen_fialko.m` was written against the
mirror's vectorised `Q` and would need updating to run against the original
(follow-up). The committed JSON fixtures (plain numbers) are all the tests need.

`dc3d_golden.json` is different in kind from the volcanic fixtures: it is not a
handful of published table points but a **random volume** — 24 buried faults
(each with a non-zero strike) observed at 32 points apiece, ~3/4 of them *below*
the surface (`z < 0`). That is the part of `OkadaSource` that Table 2 (three
surface points) does not reach. It runs the same `DC3D.f90` used to port the
strain kernels (in `scaffolding/`, **not** redistributed here — see Licensing);
`gen_dc3d.py` compiles it with `gfortran` behind a tiny in-memory driver.

Beyond displacements, the JSON carries two families of gradient ground truth so
the tests can check `OkadaSource`'s backward (not just its forward). **Okada is
the only source with external gradient references, and deliberately so**: it is
the only one with a hand-written backward (`analytic_grad`, a closed-form DC3D
strain) that can be wrong *independently of the forward*, so its derivatives need
their own external check. Every other source is plain autograd of the forward, so
`torch.autograd.gradcheck` (autograd vs. finite differences) already pins its
gradients once the forward is validated — nothing external to freeze. The two
families below are themselves a mix: exact where a closed form exists (unit-slip
responses; the homogeneity identity for source position) and **DC3D
finite-differences** where it does not (`dip/length/width/centroid_depth`):

- `derivatives_fault_frame` — DC3D's nine spatial derivatives `d u_i / d x_j`
  (fault-local, column-major). Rotated to the map frame, these check the
  `analytic_grad` closed-form strain (all nine components, at depth).
- `param_gradients` — ENU gradients `d(E,N,U)/d(param)` for `disl{1,2,3}`
  (exact: unit-slip responses, `d u / d disl_k = G_k` by slip-linearity) and for
  `dip`, `length`, `width`, `centroid_depth` (DC3D **central differences**;
  steps in `grad_fd_steps`). Source-position gradients are *not* stored — a
  half-space is horizontally homogeneous, so `d u / d source_x = -d u / d x_obs`
  exactly, and the test derives them from `derivatives_fault_frame`.

All of it is frozen at generation time, so no Fortran is needed to run the tests.

The four `*_volume_golden.json` fixtures apply the same *random-volume* idea as
DC3D to the volcanic sources, but with a MATLAB kernel instead of Fortran. Where
`gen_nikkhoo.m` / `gen_fialko.m` freeze a handful of hand-picked points,
`gen_{pcdm,cdm,pecm,penny}_volume.py` each sweep a wide random parameter space —
16 buried sources at 24 points apiece — and freeze the forward displacement:

- **pCDM / CDM / pECM** — random depth, full 3-axis orientation, and the model's
  own shape/strength parameters (pCDM anisotropic same-sign potencies; CDM
  semi-axes + signed opening; pECM semi-axes + signed pressure), at 24 surface
  points each.
- **penny** — the Fialko solution is dimensionless in `h = depth/radius`, so its
  "volume" is 1-D in `h`: each crack draws a random radius, `h` and signed
  pressure, sampled at 24 radii on the +East line (`y = 0`, purely radial). It
  uses the same corrected `Uz` as `fialko_golden.json` (recomputed with the
  original loop formula, independent of the buggy mirror — see below).

Structurally these are the `gen_dc3d.py` pattern ported to MATLAB: randomness and
geometry stay in Python (numpy), and one `matlab -batch` call runs the reference
kernel purely as a function, so there is no cross-language RNG to reconcile. They
store **forward displacement only** — no gradient references, unlike DC3D. The
volcanic sources have no hand-written backward (contrast
`OkadaSource.analytic_grad`), so their gradients are autograd of the forward and
are already pinned by `torch.autograd.gradcheck` in the respective
`test_*_source.py` (autograd vs. finite differences, self-contained). A correct
forward plus a passing gradcheck leaves nothing external for golden gradients to
catch.

## Licensing

- `nikkhoo/*.m` — MIT, © Mehdi Nikkhoo (GFZ); license text is in each file
  header. MIT permits redistribution with that notice, so these are vendored.
- Fialko's penny code is **deliberately not committed**. It carries no explicit
  redistribution license — only author attribution in `calc_crack.m`; it is
  academic code "available from the authors" (Fialko et al. 2001), and GeodMod's
  BSD-2-Clause covers Amelung's distribution, not Fialko's underlying code. We
  ship only the derived numeric fixture (`data/fialko_golden.json`, not
  copyrightable) plus the generator that runs against a locally-downloaded copy.

- `DC3D.f90` is **not committed** anywhere in the repo (it is a local,
  git-untracked file in `scaffolding/`). `gen_dc3d.py` looks for it there, or at
  `$TORCHDEFORM_DC3D_F90`, and errors with a clear message if absent. We ship
  only the derived numeric fixture (`data/dc3d_golden.json`).

The vendored `nikkhoo/*.m` are third-party reference code used only to generate
test fixtures; they are not part of the `torchdeform` package (Apache-2.0).

## Conventions (must match the torchdeform API)

- Material: `nu = 0.25`, `mu = 3e10 Pa` (so `lambda = 3e10`) — the source
  defaults. pECM/Fialko take `mu` explicitly; the dimensionless penny solution
  depends only on `h = depth/radius`, with `nu`/`mu` entering through
  `Pf = 2(1-nu)·a·P/mu`.
- Rotation angles `omega`: the fixed `nikkhoo_golden.json` stores **degrees**
  (MATLAB), so its tests apply `math.radians`. The `*_volume_golden.json`
  fixtures instead store **radians** (the Python generators convert), so their
  tests feed the values straight to the source. Either way the kernel is called
  in degrees; only the JSON convention differs.
- CDM/pECM semi-axes are passed as-is; the reference doubles them internally.
- pCDM potencies are in m³; CDM `opening` and Fialko `P` are tensile opening (m)
  and pressure (Pa).
- DC3D: `alpha = 1/(2(1-nu)) = 2/3` for `nu = 0.25`. `gen_dc3d.py` works in
  map-frame (East/North) inputs and rotates DC3D's fault-local displacement back
  to ENU with the same strike rotation `OkadaSource` uses (its 2×2 matrix is its
  own inverse), so the fixture exercises the full map→fault assembly, not just
  the kernel. Centroid placement: `AL1,AL2 = ∓L/2`, `AW1,AW2 = ∓W/2`,
  `depth = centroid_depth`; `disl1,2,3` are strike/dip/tensile slip in metres.

## Regenerating

```bash
cd tests/sources/reference
matlab -batch "run('gen_nikkhoo.m')"               # uses vendored nikkhoo/
# For Fialko: first download the penny code into ./penny/ (see above), then:
matlab -batch "run('gen_fialko.m')"
# For DC3D: needs gfortran + a local DC3D.f90 (default: ../../../scaffolding/):
python gen_dc3d.py                                 # or TORCHDEFORM_DC3D_F90=/path/DC3D.f90 python gen_dc3d.py
# Volume fixtures: Python drivers over a MATLAB kernel (need MATLAB + nikkhoo/,
# and, for penny, the downloaded penny/ code):
python gen_pcdm_volume.py
python gen_cdm_volume.py
python gen_pecm_volume.py
python gen_penny_volume.py
```

`gen_fialko.m` / `gen_penny_volume.py` error with a clear message if `./penny/`
is missing; `gen_dc3d.py` and the volume generators do the same if their kernel
or compiler (`DC3D.f90` + `gfortran`, or `matlab`) is missing.

The **Python** generators take a `--summary` flag that reads the committed JSON
and prints the header, output array shapes, and a table of the per-source
**input** parameters — no regeneration, no toolchain (not even MATLAB, since it
only reads the frozen file):

```bash
python gen_dc3d.py --summary
python gen_pcdm_volume.py --summary        # and gen_{cdm,pecm,penny}_volume.py
```

The plain MATLAB fixtures (`nikkhoo_golden.json`, `fialko_golden.json`) have no
`--summary`: they are generated by `.m` scripts, so a toolchain-free view would
need a Python reader — but they are tiny and already pretty-printed, so just open
them. The volume generators show the path taken the other way: a Python driver
over a MATLAB kernel, which gets `--summary` for free.

## Bug in Fialko's `intgr.m` (vertical displacement)

The GeodMod mirror of `penny/intgr.m` has a bug in the **vertical** displacement
introduced by a later "speedup" vectorization. Its active line factors `fi`
over the `psi` terms:

```matlab
Uz = sum(Wt2.*fi2.*(Qf1 + h*Qf2 + psi2.*Qf1./tt - Qf3));   % WRONG
```

whereas the **original** Fialko loop (still present as a comment in the same
file) is:

```matlab
Uz = sum(Wt.*( fi.*(Q1 + h*Q2) + psi.*(Q1./t - Q3) ));     % CORRECT
```

This causes a ~50 % error in `Uz`; the radial component `Ur` is unaffected.
`torchdeform`'s `penny.py` implements the **correct** original formula. The
**original** `penny.tar.gz` (the source documented above) already uses this loop
in its `intgr.m`, so it is correct as-is; the bug is specific to the GeodMod
mirror. Both generators stay robust regardless: `gen_penny_volume.py` recomputes
`Uz` with the per-radius loop, and `gen_fialko.m` likewise recomputes it (though
via the mirror's vectorised `Q`, which is why it needs updating for the original
code — see the download note above).
