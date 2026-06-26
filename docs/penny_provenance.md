# Penny-shaped crack (Fialko et al. 2001) — provenance

This document records where the penny-shaped (horizontal circular) crack source
model in torchdeform comes from: `PennySource` and the quadrature helpers in
`src/torchdeform/sources/penny.py`.

**Why this exists.** A pressurized horizontal circular crack is a standard
sill/laccolith forward model, and reference MATLAB implementations exist (the
authors' own code, and mirrors in dMODELS / GeodMod). To keep torchdeform's
implementation cleanly Apache-2.0 and independent of any of that code, the
mathematics was extracted **clean-room** from the published paper *only*, by a
worker with **no** exposure to any MATLAB `.m` source, any prior `penny.py`, or
any third-party penny-crack code. Every routine below is cited to a specific
equation in the publication. The implementation was then verified purely as a
black box against numerical invariants and golden output values (see
"Verification").

This is engineering/academic-integrity documentation, not legal advice.

## Source (the only scientific source used)

- **PAPER** — Fialko, Y., Khazan, Y. & Simons, M. (2001), "Deformation due to a
  pressurized horizontal circular crack in an elastic half-space, with
  applications to volcano geodesy," *Geophys. J. Int.* **146**, 181–190.

Everything in `penny.py` is derived from the equations of this paper (the
self-contained Appendices A and B in particular). No other source was consulted
for the mathematics. If you use this model, please cite the paper above.

## Permission

Beyond the clean-room derivation above, the lead author granted explicit
permission to build on this work:

> "You can use and distribute codes based on our 2001 paper. I'm glad people
> find them useful."
> — Yuri Fialko, personal communication, 26 June 2026

This permission covers code based on Fialko et al. (2001) generally; together
with the clean-room implementation it removes any ambiguity about the model's
provenance in this repository (including earlier development history).

## Framework

A horizontal circular ("penny-shaped") crack of radius `R` is buried at depth
`H` in an elastic half-space and loaded by a uniform excess pressure `dP`
(PAPER §3, Fig. 1). All lengths are normalised by `R` and all stresses by the
shear modulus `mu`; the dimensionless depth is `h = H/R` (PAPER §3). The surface
displacement has no elementary closed form: it follows from a pair of coupled
Fredholm integral equations of the second kind for two auxiliary "image"
functions `phi(t)`, `psi(t)` (PAPER eq 27 / Appendix A eq A1), after which the
surface displacements are recovered from a second integral with closed-form
kernels (PAPER eq 30 / Appendix B). The dimensional displacement is obtained by
multiplying the dimensionless result by `Pf = 2(1-nu) R dP / mu` (PAPER §3;
eqs B1–B2 carry the leading `2(1-nu) p0` factor, with `p0 = dP/mu`).

## Routine → equation mapping

| `penny.py` routine | derives from | notes |
|--------------------|--------------|-------|
| `ROOT16`, `WEIGHT16` | standard 16-pt Gauss–Legendre constants (Abramowitz & Stegun 1972) | Appendix A states the integrals are evaluated with 16-point Gaussian quadrature on each subinterval. |
| `build_quadrature` | Appendix A (numerical integration: subdivide [0,1] into equal panels, 16-pt GL on each) | composite GL grid on [0,1]; weights scaled so they integrate to 1. |
| `_fredholm_kernels` → `T1`,`T2`,`T3`,`T4` | eqs **A2, A3, A4, A5** | built from `R1,R2,R3` (eq 25) and `P1,P2,P3,P4` (Appendix A); `T4(t,r)=T3(r,t)` (eq A5). |
| `_solve_fredholm` | eq **A1** (= eq 27 for hydrostatic pressure) | discretised on the GL grid and assembled as a **single** dense `2M×2M` linear system `(I-K)[phibar;psibar]=b` with forcing `-2t/pi` (eq A1); solved once with `torch.linalg.solve` (no successive-approximation iteration), so it is batched and autograd-differentiable. |
| `_surface_kernels` → `S0^0`,`S0^1`,`C0^1`,`S1^{-1}`,`S1^0`,`C1^0`,`C1^1`,`S1^1` | eqs **B5–B12**, with `X1`,`X2` from eq **B13** | the closed forms of the `S/C` improper integrals (eqs B3–B4). |
| `_surface_displacement` (vertical `Uz`) | eq **B1** (= eq 28 with eqs 24, 26) | integrand `(S0^0 + h·S0^1)·phibar + (S0^0/t − C0^1)·psibar`, assembled into eq 30. |
| `_surface_displacement` (radial `Ur`) | eq **B2** (= eq 29 with eqs 24, 26) | integrand `(S1^{-1}/t − C1^0 − (h/t)·S1^0 + h·C1^1)·psibar − h·S1^1·phibar`, from eq 29's `[(1−ξh)Ψ − ξhΦ]` kernel expanded through the `S/C` integrals (B3–B12); assembled into eq 30. |
| `forward` (dimensional scaling, sign) | PAPER §3 normalisation; eqs B1–B2 leading factor `2(1-nu) p0` | multiply dimensionless displacements by `Pf = 2(1-nu) R dP / mu`; the paper's `Uz` is positive downward, so the ENU `u` component is negated to be positive upward. Horizontal field is purely radial (axisymmetric, eqs 12/13) and is decomposed into E/N along the unit vector from the crack centre. |

## What the paper specifies vs. torchdeform numerical details

These are flagged so they are not mistaken for paper-sourced facts:

1. **Linear-system solve.** The paper solves eq A1 by *successive approximations*
   (iterated kernel series; Delves & Mohamed 1985), noting convergence degrades
   for `h ≤ 0.18`. torchdeform instead assembles the *same* discretised system as
   one dense linear system and solves it directly (`torch.linalg.solve`). This is
   mathematically the fixed point of the iteration but is unconditionally solvable,
   batched, and differentiable through the solve. The discretisation (16-pt GL,
   `nis` panels) is the paper's; the direct solve is a torchdeform choice.
2. **`num_eps` guards.** Small additive `num_eps` terms guard denominators and
   `sqrt` arguments that can vanish on-axis (`r→0`) or for `depth→0`. These are a
   numerical detail, inert at realistic values; the paper's closed forms are exact.
3. **ENU decomposition and upward-positive sign convention** are torchdeform
   interface conventions (the paper works in `(r, z)` with `z` downward).

## Verification

`PennySource` was validated **only** as a black box, with no reference to any
implementation code:

- **Quadrature units** — `build_quadrature` reproduces 16-pt Gauss–Legendre and
  integrates polynomials on [0,1] to machine precision.
- **Physical invariants** — axisymmetry; horizontal displacement purely radial;
  linearity in pressure; proportionality to `(1−nu)` and `1/mu`; length
  self-similarity.
- **Independent physics** — directly above a *deep* crack the vertical
  displacement converges (from below) to the Mogi point source with the
  penny-crack volume change `dV = 16(1−nu) a³ P / (3 mu)`.
- **Golden values** — inline golden numbers and `tests/sources/data/fialko_golden.json`
  (numerical outputs only) guard against regressions.
- **Differentiability** — `torch.autograd.gradcheck` / `gradgradcheck` through the
  Fredholm linear solve, in both source parameters and observation coordinates.

All of `tests/sources/test_penny_source.py` (38 tests) passes.
