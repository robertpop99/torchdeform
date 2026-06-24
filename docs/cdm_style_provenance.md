# CDM magmatic-style parameterisation — provenance

This document records where the magmatic-style CDM parameterisation in torchdeform
comes from: `cdm_params_from_shape` (`src/torchdeform/sources/cdm.py`) and the
`DEFAULT_CDM_*` presets (`src/torchdeform/simulation/priors.py`).

**Why this exists.** The style recipes (which semi-axes are equal/zero per style)
and the validity thresholds could superficially resemble a colleague's
GPL-3.0-licensed MATLAB setups. To keep torchdeform's implementation cleanly
Apache-2.0 and independent of that code, the parameterisation was extracted
*clean-room* from the **published paper and its supplement only** — by a worker
with no exposure to the MATLAB source — and our implementation was then verified
to agree with that independent extraction (sphere/prolate/oblate match exactly;
dyke/sill differ only in the numerical proxy used for a nominally-zero semi-axis).
Every quantitative choice below is cited to a page/line/table in the publication.

This is engineering/academic-integrity documentation, not legal advice.

## Sources (the only scientific sources used)

- **PAPER** — Ireland, B., Biggs, J., Albino, F., et al. (2026), "Along-rift
  variations in magma system geometry observed using Sentinel-1 InSAR data from
  the East African Rift System," ESS Open Archive,
  doi:10.22541/essoar.15001947/v1.
- **SUPP** — Supporting Information for the same paper.

If you use these CDM-style priors, please cite the paper above.

## Framework

The forward source is a Compound Dislocation Model (CDM; Nikkhoo et al. 2017):
three orthogonal rectangular dislocations with a common centre, sized by the
semi-axes `(a_x, a_y, a_z)` (the paper's half-lengths `X_l, Y_l, Z_l`) and
oriented by rotations `(omega_x, omega_y, omega_z)` = the paper's `(Ω_x, Ω_y, Ω_z)`
(PAPER §3.2.1 p.10 l.372–378; Fig. 2 p.11). Potency→opening uses the Nikkhoo CDM
relation `dV = 4(a_x a_y + a_x a_z + a_y a_z)·opening`, consistent with the paper's
definition of `dV` as the cumulative volume change across the three dislocations
with equivalent openings (PAPER §3.2.1 p.10 l.376–378). `A` is the long/short axis
aspect ratio (PAPER p.10 l.378; Fig. 2 caption).

The paper defines six CDM classes; the requested "sill" is the paper's *elongated
sill*, with the *symmetric sill* recovered as the special case `aspect = 1`,
`omega_z = 0` (`DEFAULT_CDM_SILL_SYMMETRIC_PRIOR`).

## Per-style semi-axis construction

Let `R` = radius/half-size, `A` = aspect. (PAPER §3.2.1 p.10 l.379–386; SUPP
Table S1 p.13.)

| style              | (a_x, a_y, a_z)        | source |
|--------------------|------------------------|--------|
| sphere             | `R, R, R`              | p.10 l.380–381; Table S1 "Spherical" |
| oblate spheroid    | `R, R, R·A`            | p.10 l.384–385 ("X_l = Y_l, Z_l = X_l·A"); Table S1 |
| prolate spheroid   | `R·A, R·A, R`          | p.10 l.385–386 ("X_l = Y_l = Z_l·A", long axis Z); Table S1 |
| dyke (vertical)    | `0, R, R·A`            | p.10 l.383–384 ("X_l = 0"); Table S1 "Dyke" |
| sill (elongated)   | `R, R·A, 0`            | p.10 l.382–383 ("Z_l = 0"); Table S1 "Elongated sill" |

`omega_y = 0` for all styles (PAPER p.10 l.386; Table S1). A nominally-zero axis
(dyke `a_x`, sill `a_z`) is the only **non-paper, numerical** detail: `CDMSource`
does not special-case a zero axis, so it is set to `flat_axis_ratio · radius`
(default `1e-4`), a thin sheet converging to the zero-thickness limit. The paper
uses literal `0`.

## Orientation degrees of freedom

(PAPER §3.2.1 p.10 l.380–393; SUPP Table S1.)

| style            | omega_x (dip)     | omega_y | omega_z (strike)   |
|------------------|-------------------|---------|--------------------|
| sphere           | fixed 0           | 0       | fixed 0            |
| oblate / prolate | free, 0–45°       | 0       | free, 0–180°       |
| dyke             | fixed 0           | 0       | free, 0–180°       |
| symmetric sill   | fixed 0           | 0       | fixed 0            |
| elongated sill   | fixed 0           | 0       | free, 0–180°       |

## Physical validity constraints (SUPP Table S2, p.19)

The paper sets the displacement field to zero when a candidate violates these.
torchdeform instead honours them by *range design* in the presets (a synthetic
data generator should not emit blank samples).

| style            | constraint(s)                                   | paper's attribution |
|------------------|-------------------------------------------------|---------------------|
| sphere, prolate  | `radius/depth < 0.35` and `dV/V < 0.01`         | this study (SUPP Figs S5–S9) |
| oblate           | `radius/depth < 0.35` and `dV/V < 0.5`          | this study |
| symmetric sill   | `radius/depth < 1`                              | this study |
| elongated sill   | (none)                                          | — |
| dyke             | `max(2a_y, 2a_z)/opening > 1000` and `a_y/a_z > 0.25` | Kavanagh & Sparks (2011); Krumbholz et al. (2014); this study |

Aspect `A` is bounded `0.2–0.5` for spheroids "to ensure distinctness between the
spheroidal, spherical and planar classes" (PAPER p.10 l.393–394; Table S1).
Inversion initial bounds (PAPER p.10 l.387–391; Table S1): depth `0.5–10 km`,
half-lengths `0.25–10 km`, `dV ±1.5×10⁸ m³`. These are *inversion* bounds, not a
stated sampling prior; torchdeform's `DEFAULT_CDM_*` ranges are its own choice and
intentionally differ (e.g. tighter, submersion-safe) — see those presets.

## What the paper does NOT specify (gaps)

These are flagged so they are not mistaken for paper-sourced facts:

1. The numerical fractional thickness for a "zero" semi-axis (`flat_axis_ratio`) —
   a torchdeform numerical detail; the paper uses `0`.
2. How a single `radius` + `aspect` maps onto the dyke's / elongated sill's **two
   independent** in-plane half-lengths — the paper treats them as two free
   parameters; reducing them to `radius` and `radius·aspect` is a torchdeform
   modelling choice. (For sphere/oblate/prolate, `radius·aspect` *is* the paper's.)
3. Sampling distributions (uniform/log) per style — not given; the paper provides
   only inversion bounds/seeds.
4. The explicit chamber-volume `V` formula behind the `dV/V` constraint.

## Verification

`cdm_params_from_shape` was checked numerically against the independent clean-room
implementation: identical `a_x/a_y/a_z/opening` for sphere/prolate/oblate, and
identical up to the `flat_axis_ratio` proxy for the zero axis of dyke/sill.
