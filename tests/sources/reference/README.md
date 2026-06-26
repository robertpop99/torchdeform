# Source-model reference values

External ground truth for the volcanic source models, in the same spirit as the
Okada (1985) Table 2 checklist used by `test_okada_source.py`. None of these
papers published a fixed numeric table, so the "tables" here are produced by
running the **original authors' MATLAB** and freezing the output as JSON under
[`../data/`](../data). The Python ports reproduce these values to machine
precision (relative error ~1e-13 for the Nikkhoo models, ~1e-11 for Fialko).

Regenerating requires MATLAB. The JSON files in `../data/` are committed, so the
tests do **not** need MATLAB to run.

## Models and provenance

| Data file | Generator | Reference code | Paper |
|---|---|---|---|
| `data/nikkhoo_golden.json` | `gen_nikkhoo.m` | `nikkhoo/{pCDM,CDM,pECM}.m` (vendored) | Nikkhoo, Walter, Lundgren & Prats-Iraola (2017), *Compound dislocation models (CDMs) for volcano deformation analyses*, GJI 208(2), 877–894 |
| `data/fialko_golden.json` | `gen_fialko.m` | `penny/*.m` (**download separately**) | Fialko, Khazan & Simons (2001), *Deformation due to a pressurized horizontal circular crack in an elastic half-space*, GJI 146(1), 181–190 |

`nikkhoo/` holds the **pristine official** `pCDM.m`, `CDM.m` and `pECM.m` from
Nikkhoo's `CDM_ECM_RD.zip` (release `2022_Oct_24`, from
<https://volcanodeformation.com/software>), byte-identical to the official
files except for line endings normalised to LF.

Fialko's penny-crack code is **not redistributed here** (see Licensing). To
regenerate `fialko_golden.json`, download it into a local `penny/` subdirectory
(git-ignored) from the
[GeodMod mirror](https://github.com/falkamelung/GeodMod/tree/master/deformation_sources/penny)
(originally `topex.ucsd.edu/pub/fialko/penny`); the needed files are
`Q.m Qr.m RtWt.m fpkernel.m fred.m fredholm.m intgr.m`. The committed
`data/fialko_golden.json` (plain numbers) is all the tests need.

## Licensing

- `nikkhoo/*.m` — MIT, © Mehdi Nikkhoo (GFZ); license text is in each file
  header. MIT permits redistribution with that notice, so these are vendored.
- Fialko's penny code is **deliberately not committed**. It carries no explicit
  redistribution license — only author attribution in `calc_crack.m`; it is
  academic code "available from the authors" (Fialko et al. 2001), and GeodMod's
  BSD-2-Clause covers Amelung's distribution, not Fialko's underlying code. We
  ship only the derived numeric fixture (`data/fialko_golden.json`, not
  copyrightable) plus the generator that runs against a locally-downloaded copy.

The vendored `nikkhoo/*.m` are third-party reference code used only to generate
test fixtures; they are not part of the `torchdeform` package (Apache-2.0).

## Conventions (must match the torchdeform API)

- Material: `nu = 0.25`, `mu = 3e10 Pa` (so `lambda = 3e10`) — the source
  defaults. pECM/Fialko take `mu` explicitly; the dimensionless penny solution
  depends only on `h = depth/radius`, with `nu`/`mu` entering through
  `Pf = 2(1-nu)·a·P/mu`.
- Rotation angles `omega` are stored in **degrees** in the JSON (MATLAB); the
  Python API takes **radians**, so the tests apply `math.radians`.
- CDM/pECM semi-axes are passed as-is; the reference doubles them internally.
- pCDM potencies are in m³; CDM `opening` and Fialko `P` are tensile opening (m)
  and pressure (Pa).

## Regenerating

```bash
cd tests/sources/reference
matlab -batch "run('gen_nikkhoo.m')"               # uses vendored nikkhoo/
# For Fialko: first download the penny code into ./penny/ (see above), then:
matlab -batch "run('gen_fialko.m')"
```

`gen_fialko.m` errors with a clear message if `./penny/` is missing.

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
`torchdeform`'s `penny.py` implements the **correct** original formula, so
`gen_fialko.m` recomputes `Uz` with that formula rather than trusting
`intgr.m`'s vertical output.
