# External cross-validation tests

Manual-only tests that check torchdeform against **independent third-party
libraries**. They are *not* part of CI — they need extra dependencies and are
meant to be run by hand for confidence, not on every push.

## `test_vmod_comparison.py`

Compares every torchdeform source model that has a counterpart in
[VMOD](https://github.com/uafgeotools/vmod) (uafgeotools) over ~1000 random
parameter combinations each, on a fixed observation grid.

Run it:

```bash
RUN_VMOD_TESTS=1 pytest tests/external/test_vmod_comparison.py -v -s
```

Without `RUN_VMOD_TESTS=1` the whole module skips (so a plain `pytest` / CI run
never touches it). `-s` lets the per-model error summaries print.

VMOD must be importable. Either install it:

```bash
pip install vmod-geodesy      # pulls in utm, rasterio, hankel, ...
```

or point `VMOD_PATH` at a checkout:

```bash
RUN_VMOD_TESTS=1 VMOD_PATH="scaffolding/Link to vmod" \
    pytest tests/external/test_vmod_comparison.py -s
```

(If `VMOD_PATH` is unset the test also falls back to the vendored checkout under
`scaffolding/Link to vmod` automatically.)

Optional knobs:

| Env var          | Default | Meaning                          |
|------------------|---------|----------------------------------|
| `VMOD_N_SAMPLES` | `1000`  | random parameter sets per model  |
| `VMOD_SEED`      | `0`     | base RNG seed                    |

### What's compared

Relative field error over 1000 random samples, as `median → max`:

| torchdeform      | VMOD    | median | max   | notes                     |
|------------------|---------|--------|-------|---------------------------|
| `MogiSource`     | `Mogi`  | ~2e-16 | ~5e-16| closed-form, machine prec |
| `OkadaSource`    | `Okada` | ~2e-13 | ~1e-9 | closed-form; max is 1 near-fault point |
| `CDMSource`      | `Cdm`   | ~7e-13 | ~1e-6 | closed-form; max is 1 near-source point |
| `PennySource`    | `Penny` | ~1e-3  | <5e-3 | numerical Hankel transform (see below) |

The Okada/CDM `max` columns are a normalization artifact, not real disagreement
— see "Why Penny agrees only to ~1e-3" below for the full breakdown.

`PCDMSource` and `PECMSource` have no VMOD counterpart (VMOD's `cdm` is the
*finite* CDM, matched by our `CDMSource`) and are not compared here.

### Why Penny agrees "only" to ~1e-3 (and the others to machine precision)

The numbers in the table are **relative** field errors (`max|ours − vmod| /
peak displacement`), not absolute distances. 1e-3 means the two codes differ by
~0.1% of the signal — e.g. **0.17 mm on a 168 mm crack**, scaling with the
signal, never "1e-3 mm".

Mogi, Okada and CDM are **closed-form** elementary functions: both libraries
evaluate essentially the same algebra, so they agree to float64 machine
precision. Their *median* error is ~1e-13; the larger *max* figures in the table
(Okada ~1e-9, CDM ~1e-6) are a **normalization artifact**, not real
disagreement. They come from the single observation point nearest a dislocation
edge / the buried source, where the kernels (logs, arctangents) go near-singular
and the two codes' rounding diverges by tens of nanometres — divided by a small
peak amplitude, that reads as ~1e-6 relative. CDM's tail is a touch larger than
Okada's because it sums three near-singular rectangular dislocations rather than
one (the Euler-angle re-decomposition is exact to machine epsilon and is *not*
the cause).

The **Penny** (Fialko 2001) crack has **no elementary closed form**. It requires
a Fredholm solve for auxiliary functions plus a *semi-infinite* Hankel
transform, ∫₀^∞ … e^(−ξh) Jₙ(ξr) dξ (h = depth/radius). The two libraries do
that integral differently:

- **torchdeform** evaluates it **in closed form** (Fialko's S/C improper
  integrals, appendix B) and is validated against the original Fialko MATLAB
  reference to `rtol 1e-6` (`tests/sources/test_penny_source.py`).
- **VMOD** evaluates it **numerically** — Gauss–Legendre, 41+41 nodes, truncated
  at ξ = 60.

So the ~1e-3 gap is essentially **VMOD's truncation + quadrature error**, largest
for shallow/wide cracks (small h, where the e^(−ξh) tail decays slowly past
ξ = 60) and for far-field points (Jₙ(ξr) oscillates faster than the fixed node
set resolves). It is not torchdeform uncertainty — refining our quadrature
doesn't move the gap. The Penny test samples the deeper magmatic regime
(depth ≥ 4 km, radius/depth ∈ [0.2, 0.75]) to keep that error sub-percent.

The two libraries use different internal parameter conventions; each adapter in
the test maps a torchdeform parameter set to the equivalent VMOD call. The
non-obvious mappings (Okada's strike-line reflection and slip/opening split,
CDM's Euler-angle re-ordering) are documented in the test module's docstring.
