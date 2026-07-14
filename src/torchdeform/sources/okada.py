"""
Okada (1992) rectangular finite-fault dislocation model, differentiable.

A PyTorch port of Okada's classic ``DC3D`` Fortran routine for the surface (and
sub-surface) displacement of a uniform-slip rectangular dislocation in an
elastic half-space -- the workhorse model for earthquake/fault deformation. It
handles strike-slip, dip-slip and tensile (opening) components.

Two public source classes are provided:

* :class:`OkadaSource` -- the general model for arbitrary observation depth
  ``z <= 0``; assembles the real-source, image-source and depth (UA/UB/UC)
  contributions exactly as ``DC3D`` does.
* :class:`OkadaSourceSimple` -- the surface-only (``z = 0``) specialisation, in
  which the UA and UC terms cancel/drop and only UB remains; cheaper when you
  only need surface displacement.

Structure
---------
The file mirrors the Fortran sub-routine decomposition. ``dccon0`` precomputes
medium/dip constants (:class:`DCCon0`); ``dccon1``/``dccon2`` precompute the
per-observation geometry (:class:`DCCon1`/:class:`DCCon2`); ``ua_displacement``,
``ub_displacement`` and ``uc_displacement_*`` evaluate the three displacement
kernels. The leading ``_soft_*`` / ``_safe_*`` helpers are smooth
regularisations of the hard ``abs``/``max``/branch operations.

Differentiability / ``training_safe``
-------------------------------------
The exact Fortran logic contains hard branches and zero-snapping at geometric
singularities (vertical dip, fault edges, ``r + d -> 0``) that produce
non-finite or discontinuous gradients. Passing ``training_safe=True`` replaces
those with smooth blends/attenuations, trading a tiny accuracy loss away from
the singular manifolds for finite, well-behaved gradients during optimisation.
The handful of ``*_EPS`` module constants set the regularisation scales.

There is also an *analytic-backward* mode, ``analytic_grad=True`` (on both
:class:`OkadaSource` and :class:`OkadaSourceSimple`): it keeps the exact
(un-smoothed) forward values and supplies the closed-form Okada strain (the
``ua_``/``ub_``/``uc_displacement_and_derivatives`` kernels) as the
observation-coordinate gradient. Unlike the smoothed path this is exact to the
Okada-85 table even on the singular manifolds (vertical dip, on-trace points),
and unlike plain autograd of the exact forward it has no ``cd = 0`` NaN trap.
``dip`` (whose explicit ``1/cos(dip)`` terms have no closed-form strain) is taken
from autograd of the exact forward -- exact to ~machine precision -- except in a
thin near-vertical band (``|cos(dip)| < DIP_FD_BAND``, dip within ~0.17 deg of
vertical) where those terms make autograd ill-conditioned; there it falls back to
a Richardson-extrapolated finite difference whose wide steps clear the vertical
singularity (``DIP_FD_RICH_STEP``), good to ~1e-6 right through dip = 90 deg.
``smooth_grad`` and ``analytic_grad`` are mutually exclusive gradient modes; with
neither set the forward is exact and gradients are plain autograd (which can be
non-finite right on the singular manifolds).

Second derivatives (Hessians) are available by ordinary autograd double-backward
(``create_graph=True``) in the default and ``smooth_grad`` modes, whose forwards
are plain-autograd graphs. ``analytic_grad`` is a custom ``autograd.Function`` and
is **first-order only**: its hand-written backward is not itself differentiable,
so a Hessian through it raises rather than silently returning a wrong value. Use
the default mode if you need second-order information.

Attribution
-----------
Derived from the ``DC3D`` subroutine of the Fortran90 file ``DC3D.f90`` by
Takuya Miyashita (https://github.com/hydrocoast/DC3D.f90, MIT licensed), a
free-form conversion of Y. Okada's original FORTRAN code (copyright NIED). The
full MIT copyright and permission notice is reproduced in the project ``NOTICE``
file.

Reference: Okada, Y. (1992), "Internal deformation due to shear and tensile
faults in a half-space", Bulletin of the Seismological Society of America,
82(2), 1018-1040.
"""
import copy
import torch
from dataclasses import dataclass
import math
import warnings
from torch import Tensor

from .base import SourceModel, default_num_eps, NUM_EPS_F64, DEFAULT_POISSON_RATIO
from ..core import Displacement

# Analytic-Jacobian bundle returned by ``_evaluate(..., return_strain=True)``:
# maps a parameter name to its (d ue, d un, d uu) triple. See ``_make_contract``.
_StrainDict = dict[str, tuple[Tensor, Tensor, Tensor]]

GEOM_EPS = 1e-6   # Okada-style branch / “treat as zero” (physical tolerance, metres)
RD_EPS   = 1e-8   # float64 UB singularity guard for r + d
SMOOTH_EPS = 1e-8  # float64 smooth_grad reciprocal/division scale
# RD_EPS and SMOOTH_EPS are both 1e-8-class reciprocal-stability floors; float32
# (~1e-7 eps) and float16 (~1e-3 eps) would swallow them, so _okada_grad_floors
# lifts both together onto this shared coarse ladder (cf. base.default_num_eps).
GRAD_FLOOR_F32 = 1e-4
GRAD_FLOOR_F16 = 1e-2
BLEND_EPS  = 1e-4  # smooth_grad cd~0 blend width (physical, dtype-independent)
# The num guard (denominator/log/sqrt floor) is owned by base.default_num_eps,
# which dtype-scales it; the kernels below default to base.NUM_EPS_F64 only as a
# fallback for standalone calls -- on the model path _evaluate always passes the
# dtype-resolved value explicitly.
# Near-vertical band (|cos(dip)|) in which float32 cannot resolve Okada's exact
# formula. Note the subtlety: Okada's closed-form cos(dip)=0 expression (our "_b"
# branch) is exact only *at* vertical -- off vertical (89.5 deg, say) it is wrong
# by percent (2.5% at 89.5, 5% at 89), and the physically correct value there is
# the general-dip "_a" formula. But "_a" carries 1/cos(dip)**2 factors whose
# numerators vanish like cos(dip)**2; in float32 that (big - big)/cos**2 cancels
# to noise, so the *forward* is off by orders of magnitude for 0 < |cos| < ~1e-2.
# There is no separate closed form to reach for here (Okada's dip=90 form is a
# single point, already used exactly), only a precision requirement the general
# formula imposes. Inside this band we therefore evaluate that same exact formula
# in float64 and cast the result back (see _OkadaBase._compute_dtype); float64 is
# accurate down to |cos| ~ 1e-6, which is why it is the default compute dtype.
#
# Choice of 0.1 -- safe against gross failure, but slightly optimistic against the
# 1e-3 float32 contract (see test_forward_float32_accurate_near_vertical). Measured
# pure-float32 error (promotion disabled) grows smoothly as |cos(dip)| shrinks:
# ~1e-5 at |cos|=0.5, ~1.5e-4 by |cos|=0.26 (dip 75 deg), then steeply near the
# edge. It never leaks an orders-of-magnitude failure -- those start around
# |cos|<0.05, well inside the band. But right at the band edge |cos|~0.1 (dip
# ~84 deg) it reaches ~5e-4 to ~1.3e-3 depending on scene, i.e. a normal geometry
# already *exceeds* 1e-3 there (the promoted band is what keeps the contract; the
# un-promoted edge is the one regime that misses it -- see
# test_forward_float32_band_edge_margin). Widening to 0.2 (dip ~78.5 deg) would
# pull the edge back to ~2e-4 and restore 1e-3, at the cost of promoting steep
# (>78 deg) faults to float64; near-vertical dykes already promote regardless and
# common 45-70 deg faults stay float32 either way, so the extra cost is modest.
# Default is 0.1 (favours float32 speed); a float32 user near ~80-84 deg dip should
# expect ~1e-3, not float32 baseline. This is the *default* of the per-model
# ``f32_vertical_band`` constructor knob -- callers who want tighter near-vertical
# float32 accuracy can raise it (e.g. to 0.2) at the cost of promoting more scenes.
F32_VERTICAL_BAND = 0.1

# analytic_grad dip derivative: |cos(dip)| below which we fall back from autograd
# of the exact forward to a finite-difference estimate. Away from vertical,
# autograd gives the dip gradient to ~machine precision (4-6 orders tighter than
# any FD); but the kernel's 1/cos(dip)**2 terms make it ill-conditioned in this
# thin band (and it is exactly zero at dip = 90 deg, where dccon0 snaps cos(dip)).
# Empirically (float64) autograd's relative error stays <~1e-5 down to
# |cos(dip)| ~ 1e-2 (dip ~89.4 deg), climbs through ~1e-4 near |cos| ~ 2e-3 (dip
# ~89.9 deg), and the crossover where the FD becomes the better estimate is
# |cos(dip)| ~ 1.5e-3 (dip ~0.09 deg off vertical). 3e-3 (dip within ~0.17 deg of
# vertical) is a conservative margin above that crossover.
DIP_FD_BAND = 3e-3

# Base step for the dip FD fallback, which uses Richardson extrapolation
# (4*D(H/2) - D(H))/3, D(s) the central difference at step s. A *single* central
# difference is unusable across the band: for any fixed step some in-band dip puts
# one sample essentially on the removable singularity at 90 deg, spiking the
# relative error to ~1e-2 (worse than the autograd it replaces). Richardson fixes
# this two ways: (i) its smallest sample step, H/2, is set ABOVE the band
# half-width (~asin(DIP_FD_BAND) ~ DIP_FD_BAND rad) so all four samples clear
# vertical and stay well-conditioned for every dip in the band; (ii) it cancels
# the O(H^2) truncation term, so the wide steps do not cost accuracy -- the result
# is ~1e-6 relative right through vertical, independent of scene geometry.
# H/2 = 2 * DIP_FD_BAND keeps (i) tied to the band if either is retuned.
DIP_FD_RICH_STEP = 4.0 * DIP_FD_BAND  # H = 1.2e-2 rad; samples at +/-6e-3, +/-1.2e-2


def _okada_grad_floors(dtype: torch.dtype) -> tuple[float, float]:
    """``(smooth_eps, rd_eps)`` scaled to the compute dtype.

    Both guard near-singular reciprocals (the ``smooth_grad`` divisions and the
    exact-mode ``r + d -> 0`` shift). The float64 values (``1e-8``) are the
    validated defaults; ``float32``'s ~1e-7 epsilon would swallow them, so lift
    the floors for coarser dtypes. See :func:`~torchdeform.sources.base.default_num_eps`.
    """
    if dtype == torch.float64:
        return SMOOTH_EPS, RD_EPS
    if dtype == torch.float32:
        return GRAD_FLOOR_F32, GRAD_FLOOR_F32
    return GRAD_FLOOR_F16, GRAD_FLOOR_F16  # float16 / bfloat16


# ---------------------------------------------------------------------
# Training-safe helper ops
# ---------------------------------------------------------------------

def _soft_abs(x: Tensor, eps: float) -> Tensor:
    # Smooth approximation of |x|
    return torch.sqrt(x * x + eps * eps)


def _soft_pos(x: Tensor, eps: float) -> Tensor:
    # Smooth approximation of max(x, 0), shifted strictly positive for logs
    return 0.5 * (x + _soft_abs(x, eps)) + eps


def _safe_inv(x: Tensor, eps: float) -> Tensor:
    # Smooth reciprocal: x / (x^2 + eps^2)
    return x / (x * x + eps * eps)


def _safe_div(num: Tensor, den: Tensor, eps: float) -> Tensor:
    # Smooth division: num * den / (den^2 + eps^2)
    return num * den / (den * den + eps * eps)


def _soft_blend_from_abs(x: Tensor, eps: float) -> Tensor:
    # Returns ~1 when |x| >> eps, ~0 when |x| << eps
    x2 = x * x
    return x2 / (x2 + eps * eps)


@dataclass(slots=True)
class DCCon0:
    """
    Constants depending only on elastic parameter alpha and dip.

    All tensors are broadcast-compatible.
    """

    alpha: torch.Tensor

    sd: torch.Tensor
    cd: torch.Tensor

    sdsd: torch.Tensor
    cdcd: torch.Tensor
    sdcd: torch.Tensor

    alp1: torch.Tensor
    alp2: torch.Tensor
    alp3: torch.Tensor
    alp4: torch.Tensor
    alp5: torch.Tensor


def dccon0(
    alpha: torch.Tensor,
    dip_rad: torch.Tensor,
    internal_dtype=torch.float64,
    *,
    training_safe: bool = False,
    geom_eps: float = GEOM_EPS,
) -> DCCon0:
    """
    Exact mode:
        behaves like the current implementation.

    training_safe=True:
        removes the hard vertical-dip snapping to keep gradients finite/smooth.
    """
    dip_rad = dip_rad.to(internal_dtype)
    alpha = torch.as_tensor(alpha, dtype=internal_dtype, device=dip_rad.device)
    alpha = alpha + torch.zeros_like(dip_rad)

    sd = torch.sin(dip_rad)
    cd = torch.cos(dip_rad)

    if not training_safe:
        # Original exact behavior
        small_cd = torch.abs(cd) < geom_eps
        cd = torch.where(small_cd, torch.zeros_like(cd), cd)
        sd = torch.where(small_cd, torch.sign(sd), sd)

    sdsd = sd * sd
    cdcd = cd * cd
    sdcd = sd * cd

    alp1 = (1.0 - alpha) / 2.0
    alp2 = alpha / 2.0
    alp3 = (1.0 - alpha) / alpha
    alp4 = 1.0 - alpha
    alp5 = alpha

    return DCCon0(
        alpha=alpha,
        sd=sd,
        cd=cd,
        sdsd=sdsd,
        cdcd=cdcd,
        sdcd=sdcd,
        alp1=alp1,
        alp2=alp2,
        alp3=alp3,
        alp4=alp4,
        alp5=alp5,
    )


@dataclass(slots=True)
class DCCon1:
    """
    Geometry constants for Okada point-source terms.

    Corresponds to DCCON1 in DC3D.f.
    """

    p: torch.Tensor
    q: torch.Tensor

    s: torch.Tensor
    t: torch.Tensor

    xy: torch.Tensor

    x2: torch.Tensor
    y2: torch.Tensor
    d2: torch.Tensor

    r: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r5: torch.Tensor
    r7: torch.Tensor

    a3: torch.Tensor
    a5: torch.Tensor
    b3: torch.Tensor
    c3: torch.Tensor

    qr: torch.Tensor
    qrx: torch.Tensor

    uy: torch.Tensor
    uz: torch.Tensor

    vy: torch.Tensor
    vz: torch.Tensor

    wy: torch.Tensor
    wz: torch.Tensor


def dccon1(
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    sd: torch.Tensor,
    cd: torch.Tensor,
    *,
    internal_dtype: torch.dtype = torch.float64,
    eps: float = NUM_EPS_F64,
) -> DCCon1:
    """
    Differentiable vectorized DCCON1.

    Parameters
    ----------
    x, y, d
        Fault-coordinate system coordinates.

    sd, cd
        sin(dip), cos(dip) from dccon0.

    Returns
    -------
    DCCon1
    """

    x = x.to(internal_dtype)
    y = y.to(internal_dtype)
    d = d.to(internal_dtype)

    sd = sd.to(internal_dtype)
    cd = cd.to(internal_dtype)

    # --------------------------------------------------
    # Geometry transform
    # --------------------------------------------------

    p = y * cd + d * sd
    q = y * sd - d * cd

    s = p * sd + q * cd
    t = p * cd - q * sd

    # --------------------------------------------------
    # Coordinate powers
    # --------------------------------------------------

    xy = x * y

    x2 = x * x
    y2 = y * y
    d2 = d * d

    # --------------------------------------------------
    # Regularized distance
    # --------------------------------------------------

    r = torch.sqrt(x2 + y2 + d2 + eps * eps)  # the ONLY regularization point
    r2 = r * r  # strictly ≥ eps_len², never zero
    r3 = r2 * r
    r5 = r3 * r2
    r7 = r5 * r2

    inv_r2 = 1.0 / r2  # safe, no eps needed
    inv_r5 = 1.0 / r5

    # --------------------------------------------------
    # Okada coefficients
    # --------------------------------------------------

    a3 = 1.0 - 3.0 * x2 * inv_r2

    a5 = 1.0 - 5.0 * x2 * inv_r2

    b3 = 1.0 - 3.0 * y2 * inv_r2

    c3 = 1.0 - 3.0 * d2 * inv_r2

    # --------------------------------------------------
    # QR terms
    # --------------------------------------------------

    qr = 3.0 * q * inv_r5

    qrx = 5.0 * qr * x * inv_r2

    # --------------------------------------------------
    # U/V/W terms
    # --------------------------------------------------

    uy = sd - 5.0 * y * q * inv_r2

    uz = cd + 5.0 * d * q * inv_r2

    vy = s - 5.0 * y * p * q * inv_r2

    vz = t + 5.0 * d * p * q * inv_r2

    wy = uy + sd

    wz = uz + cd

    return DCCon1(
        p=p,
        q=q,

        s=s,
        t=t,

        xy=xy,

        x2=x2,
        y2=y2,
        d2=d2,

        r=r,
        r2=r2,
        r3=r3,
        r5=r5,
        r7=r7,

        a3=a3,
        a5=a5,
        b3=b3,
        c3=c3,

        qr=qr,
        qrx=qrx,

        uy=uy,
        uz=uz,

        vy=vy,
        vz=vz,

        wy=wy,
        wz=wz,
    )


@dataclass(slots=True)
class DCCon2:
    """
    Per-observation geometry constants for Okada finite-source terms.

    Corresponds to ``DCCON2`` in ``DC3D.f``; produced by :func:`dccon2` and
    consumed by the UA/UB/UC displacement kernels. All fields are
    broadcast-compatible tensors over the corner/observation grid.
    """

    xi: torch.Tensor
    et: torch.Tensor
    q: torch.Tensor

    r: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r5: torch.Tensor

    y: torch.Tensor
    d: torch.Tensor

    tt: torch.Tensor

    alx: torch.Tensor
    ale: torch.Tensor

    x11: torch.Tensor
    y11: torch.Tensor

    x32: torch.Tensor
    y32: torch.Tensor

    ey: torch.Tensor
    ez: torch.Tensor

    fy: torch.Tensor
    fz: torch.Tensor

    gy: torch.Tensor
    gz: torch.Tensor

    hy: torch.Tensor
    hz: torch.Tensor


def dccon2(
    xi: torch.Tensor,
    et: torch.Tensor,
    q: torch.Tensor,
    sd: torch.Tensor,
    cd: torch.Tensor,
    kxi: torch.Tensor,
    ket: torch.Tensor,
    *,
    internal_dtype: torch.dtype = torch.float64,
    eps: float = NUM_EPS_F64,
    training_safe: bool = False,
    smooth_eps: float = SMOOTH_EPS,
    geom_eps: float = GEOM_EPS,
):
    """
    Exact mode:
        current branchy Okada behavior.

    training_safe=True:
        no hard zero-snapping; no hard branch-zeroing for x11/y11/x32/y32;
        uses smooth log arguments and smooth reciprocals.
    """
    xi = xi.to(internal_dtype)
    et = et.to(internal_dtype)
    q = q.to(internal_dtype)
    sd = sd.to(internal_dtype)
    cd = cd.to(internal_dtype)

    if not training_safe:
        xi = torch.where(torch.abs(xi) < geom_eps, torch.zeros_like(xi), xi)
        et = torch.where(torch.abs(et) < geom_eps, torch.zeros_like(et), et)
        q  = torch.where(torch.abs(q)  < geom_eps, torch.zeros_like(q),  q)

    # --------------------------------------------------
    # powers / radius
    # --------------------------------------------------
    r2_raw = xi * xi + et * et + q * q
    # the ONLY regularization point
    if training_safe:
        r = torch.sqrt(r2_raw + smooth_eps * smooth_eps)
    else:
        r = torch.sqrt(r2_raw + eps * eps)
    r2 = r * r  # NOT r2_raw — derive from regularized r
    r3 = r2 * r
    r5 = r3 * r2

    # transformed coordinates+ eps
    y = et * cd + q * sd
    d = et * sd - q * cd
    y = y.expand_as(r)
    d = d.expand_as(r)

    if not training_safe:
        # ---------------- exact branch logic ----------------
        kxi_eff = kxi | ((r + xi) < eps)
        ket_eff = ket | ((r + et) < eps)

        qr = q * r
        tt_mask = torch.abs(qr) < eps
        qr_safe = torch.where(tt_mask, torch.ones_like(qr), qr)  # dummy=1, never used in output
        tt = torch.where(tt_mask, torch.zeros_like(qr), torch.atan((xi * et) / qr_safe))

        r_minus_xi = torch.clamp(r - xi, min=0.0) + eps
        r_plus_xi = torch.clamp(r + xi, min=0.0) + eps
        alx = torch.where(
            kxi_eff,
            -torch.log(r_minus_xi),
            torch.log(r_plus_xi),
        )

        x11_else = 1.0 / (r * r_plus_xi + eps)
        x32_else = ((r + r_plus_xi) * x11_else * x11_else / r)
        x11 = torch.where(kxi_eff, torch.zeros_like(r), x11_else)
        x32 = torch.where(kxi_eff, torch.zeros_like(r), x32_else)

        r_minus_et = torch.clamp(r - et, min=0.0) + eps
        r_plus_et = torch.clamp(r + et, min=0.0) + eps
        ale = torch.where(
            ket_eff,
            -torch.log(r_minus_et),
            torch.log(r_plus_et),
        )

        y11_else = 1.0 / (r * r_plus_et + eps)
        y32_else = ((r + r_plus_et) * y11_else * y11_else / r)
        y11 = torch.where(ket_eff, torch.zeros_like(r), y11_else)
        y32 = torch.where(ket_eff, torch.zeros_like(r), y32_else)

        inv_r3 = 1.0 / r3

    else:
        # ---------------- training-safe smooth branch ----------------
        den = q * r
        num = xi * et

        ratio = _safe_div(num, den, smooth_eps)

        # Soft gate to attenuate only extremely near the singular manifold
        gate = (den * den) / (den * den + (10.0 * smooth_eps) ** 2)

        tt = gate * torch.atan(ratio)

        # Since r >= |xi| and r >= |et|, r + xi and r + et are nonnegative in theory.
        # Use soft positive guards for logs and reciprocals near zero.
        r_plus_xi = _soft_pos(r + xi, smooth_eps)
        r_plus_et = _soft_pos(r + et, smooth_eps)

        alx = torch.log(r_plus_xi)
        ale = torch.log(r_plus_et)

        rrpx = r * r_plus_xi
        rrpe = r * r_plus_et

        x11 = _safe_inv(rrpx, smooth_eps)
        y11 = _safe_inv(rrpe, smooth_eps)

        # Same structural form as the exact branch, but no hard zeroing
        x32 = (r + r_plus_xi) * x11 * x11 * _safe_inv(r, smooth_eps)
        y32 = (r + r_plus_et) * y11 * y11 * _safe_inv(r, smooth_eps)

        inv_r3 = _safe_inv(r3, smooth_eps)

    # --------------------------------------------------
    # EY / EZ / FY / FZ / GY / GZ / HY / HZ
    # --------------------------------------------------
    ey = sd / r - y * q * inv_r3
    ez = cd / r + d * q * inv_r3

    xi2 = xi * xi
    fy = d * inv_r3 + xi2 * y32 * sd
    fz = y * inv_r3 + xi2 * y32 * cd

    gy = 2.0 * x11 * sd - y * q * x32
    gz = 2.0 * x11 * cd + d * q * x32

    hy = d * q * x32 + xi * q * y32 * sd
    hz = y * q * x32 + xi * q * y32 * cd

    return DCCon2(
        xi=xi,
        et=et,
        q=q,
        r=r,
        r2=r2,
        r3=r3,
        r5=r5,
        y=y,
        d=d,
        tt=tt,
        alx=alx,
        ale=ale,
        x11=x11,
        y11=y11,
        x32=x32,
        y32=y32,
        ey=ey,
        ez=ez,
        fy=fy,
        fz=fz,
        gy=gy,
        gz=gz,
        hy=hy,
        hz=hz,
    )


@dataclass(slots=True)
class UADisplacement:
    """UA (infinite-medium) displacement kernel output; fault-frame ``ux/uy/uz``."""
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor


def ua_displacement(
    disl1: Tensor,
    disl2: Tensor,
    disl3: Tensor,
    c0: DCCon0,
    c2: DCCon2,
):
    """
    Differentiable vectorized Okada UA displacement kernel.

    Parameters
    ----------
    disl1
        strike-slip

    disl2
        dip-slip

    disl3
        tensile opening

    c0
        output of dccon0()

    c2
        output of dccon2()

    Returns
    -------
    UADisplacement
    """

    pi2 = 2.0 * math.pi

    disl1 = disl1[:, None, None, None]
    disl2 = disl2[:, None, None, None]
    disl3 = disl3[:, None, None, None]

    alp1 = c0.alp1[:, None, None, None]
    alp2 = c0.alp2[:, None, None, None]

    xi = c2.xi
    q = c2.q

    # --------------------------------------------------
    # recover quantities from DCCON2
    # --------------------------------------------------

    tt = c2.tt

    ale = c2.ale
    alx = c2.alx

    qx = c2.q * c2.x11
    qy = c2.q * c2.y11

    # ==================================================
    # STRIKE-SLIP
    # ==================================================

    ux_ss = (
        tt / 2.0
        + alp2 * xi * qy
    )

    uy_ss = (
        alp2 * q / c2.r
    )

    uz_ss = (
        alp1 * ale
        - alp2 * q * qy
    )

    # ==================================================
    # DIP-SLIP
    # ==================================================

    ux_ds = (
        alp2 * q / c2.r
    )

    et = c2.et

    # uy_ds = (
    #     tt / 2.0
    #     + alp2 * c2.y * qx
    # )
    uy_ds = (
        tt / 2.0
        + alp2 * et * qx
    )


    uz_ds = (
        alp1 * alx
        - alp2 * q * qx
    )

    # ==================================================
    # TENSILE
    # ==================================================

    ux_tf = (
        -alp1 * ale
        -alp2 * q * qy
    )

    uy_tf = (
        -alp1 * alx
        -alp2 * q * qx
    )

    uz_tf = (
        tt / 2.0
        - alp2 * (
            et * qx
            + xi * qy
        )
    )

    ux = (
        disl1 * ux_ss
        + disl2 * ux_ds
        + disl3 * ux_tf
    ) / pi2

    uy = (
        disl1 * uy_ss
        + disl2 * uy_ds
        + disl3 * uy_tf
    ) / pi2

    uz = (
        disl1 * uz_ss
        + disl2 * uz_ds
        + disl3 * uz_tf
    ) / pi2

    return UADisplacement(
        ux=ux,
        uy=uy,
        uz=uz,
    )


@dataclass(slots=True)
class UAResult:
    """UA kernel output: 3 displacements + 9 spatial derivatives.

    Field order matches the Fortran ``uA`` output ``du(1..12)``: ``ux, uy, uz``
    then the fault-frame derivatives ``u{x,y,z}{x,y,z}``. ``[B, 2, 2, N]``.
    """
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor
    uxx: torch.Tensor
    uyx: torch.Tensor
    uzx: torch.Tensor
    uxy: torch.Tensor
    uyy: torch.Tensor
    uzy: torch.Tensor
    uxz: torch.Tensor
    uyz: torch.Tensor
    uzz: torch.Tensor


def ua_displacement_and_derivatives(
    disl1: torch.Tensor,
    disl2: torch.Tensor,
    disl3: torch.Tensor,
    c0: DCCon0,
    c2: DCCon2,
):
    """Exact UA kernel: displacement + 9 spatial derivatives (Fortran ``uA``).

    Adds the strain block (``du4..12``) to the displacement-only
    :func:`ua_displacement`, for the analytic Jacobian of the full
    :class:`OkadaSource`. Exact-mode only.
    """
    pi2 = 2.0 * math.pi

    d1 = disl1[:, None, None, None]
    d2 = disl2[:, None, None, None]
    d3 = disl3[:, None, None, None]

    alp1 = c0.alp1[:, None, None, None]
    alp2 = c0.alp2[:, None, None, None]
    sd = c0.sd[:, None, None, None]
    cd = c0.cd[:, None, None, None]

    xi = c2.xi
    et = c2.et
    q = c2.q
    r = c2.r
    r3 = c2.r3
    y = c2.y      # ytilde
    d = c2.d      # dtilde
    x11 = c2.x11
    y11 = c2.y11
    y32 = c2.y32
    ey, ez = c2.ey, c2.ez
    fy, fz = c2.fy, c2.fz
    gy, gz = c2.gy, c2.gz
    hy, hz = c2.hy, c2.hz
    theta = c2.tt
    alx = c2.alx
    ale = c2.ale

    xi2 = xi * xi
    q2 = q * q
    xy = xi * y11
    qx = q * x11
    qy = q * y11

    du = [torch.zeros_like(xi) for _ in range(12)]

    # strike-slip
    du[0] = du[0] + d1 / pi2 * (theta / 2.0 + alp2 * xi * qy)
    du[1] = du[1] + d1 / pi2 * (alp2 * q / r)
    du[2] = du[2] + d1 / pi2 * (alp1 * ale - alp2 * q * qy)
    du[3] = du[3] + d1 / pi2 * (-alp1 * qy - alp2 * xi2 * q * y32)
    du[4] = du[4] + d1 / pi2 * (-alp2 * xi * q / r3)
    du[5] = du[5] + d1 / pi2 * (alp1 * xy + alp2 * xi * q2 * y32)
    du[6] = du[6] + d1 / pi2 * (alp1 * xy * sd + alp2 * xi * fy + d / 2.0 * x11)
    du[7] = du[7] + d1 / pi2 * (alp2 * ey)
    du[8] = du[8] + d1 / pi2 * (alp1 * (cd / r + qy * sd) - alp2 * q * fy)
    du[9] = du[9] + d1 / pi2 * (alp1 * xy * cd + alp2 * xi * fz + y / 2.0 * x11)
    du[10] = du[10] + d1 / pi2 * (alp2 * ez)
    du[11] = du[11] + d1 / pi2 * (-alp1 * (sd / r - qy * cd) - alp2 * q * fz)

    # dip-slip
    du[0] = du[0] + d2 / pi2 * (alp2 * q / r)
    du[1] = du[1] + d2 / pi2 * (theta / 2.0 + alp2 * et * qx)
    du[2] = du[2] + d2 / pi2 * (alp1 * alx - alp2 * q * qx)
    du[3] = du[3] + d2 / pi2 * (-alp2 * xi * q / r3)
    du[4] = du[4] + d2 / pi2 * (-qy / 2.0 - alp2 * et * q / r3)
    du[5] = du[5] + d2 / pi2 * (alp1 / r + alp2 * q2 / r3)
    du[6] = du[6] + d2 / pi2 * (alp2 * ey)
    du[7] = du[7] + d2 / pi2 * (alp1 * d * x11 + xy / 2.0 * sd + alp2 * et * gy)
    du[8] = du[8] + d2 / pi2 * (alp1 * y * x11 - alp2 * q * gy)
    du[9] = du[9] + d2 / pi2 * (alp2 * ez)
    du[10] = du[10] + d2 / pi2 * (alp1 * y * x11 + xy / 2.0 * cd + alp2 * et * gz)
    du[11] = du[11] + d2 / pi2 * (-alp1 * d * x11 - alp2 * q * gz)

    # tensile
    du[0] = du[0] + d3 / pi2 * (-alp1 * ale - alp2 * q * qy)
    du[1] = du[1] + d3 / pi2 * (-alp1 * alx - alp2 * q * qx)
    du[2] = du[2] + d3 / pi2 * (theta / 2.0 - alp2 * (et * qx + xi * qy))
    du[3] = du[3] + d3 / pi2 * (-alp1 * xy + alp2 * xi * q2 * y32)
    du[4] = du[4] + d3 / pi2 * (-alp1 / r + alp2 * q2 / r3)
    du[5] = du[5] + d3 / pi2 * (-alp1 * qy - alp2 * q * q2 * y32)
    du[6] = du[6] + d3 / pi2 * (-alp1 * (cd / r + qy * sd) - alp2 * q * fy)
    du[7] = du[7] + d3 / pi2 * (-alp1 * y * x11 - alp2 * q * gy)
    du[8] = du[8] + d3 / pi2 * (alp1 * (d * x11 + xy * sd) + alp2 * q * hy)
    du[9] = du[9] + d3 / pi2 * (alp1 * (sd / r - qy * cd) - alp2 * q * fz)
    du[10] = du[10] + d3 / pi2 * (alp1 * d * x11 - alp2 * q * gz)
    du[11] = du[11] + d3 / pi2 * (alp1 * (y * x11 + xy * cd) + alp2 * q * hz)

    return UAResult(*du)


@dataclass(slots=True)
class UBDisplacement:
    """UB (half-space surface) displacement kernel output; fault-frame ``ux/uy/uz``."""
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor


def ub_displacement(
    disl1: torch.Tensor,
    disl2: torch.Tensor,
    disl3: torch.Tensor,
    c0: DCCon0,
    c2: DCCon2,
    *,
    eps: float = NUM_EPS_F64,
    training_safe: bool = False,
    smooth_eps: float = SMOOTH_EPS,
    blend_eps: float = BLEND_EPS,
    rd_eps: float = RD_EPS,
):
    """
    Exact mode:
        close to your current UB kernel.

    training_safe=True:
        - smooths reciprocals/divisions
        - blends cd!=0 and cd≈0 formulas smoothly
        - replaces hard near_rd zeroing with attenuation
    """
    pi2 = 2.0 * math.pi

    disl1 = disl1[:, None, None, None]
    disl2 = disl2[:, None, None, None]
    disl3 = disl3[:, None, None, None]

    xi = c2.xi
    et = c2.et
    q = c2.q
    r = c2.r
    y = c2.y
    d = c2.d
    x11 = c2.x11
    y11 = c2.y11
    fy = c2.fy
    fz = c2.fz
    gy = c2.gy
    gz = c2.gz
    tt = c2.tt
    ale = c2.ale

    alp3 = c0.alp3[:, None, None, None]
    sd = c0.sd[:, None, None, None]
    cd = c0.cd[:, None, None, None]
    sdsd = c0.sdsd[:, None, None, None]
    sdcd = c0.sdcd[:, None, None, None]

    rd = r + d

    if not training_safe:
        # ---------------- exact behavior ----------------
        near_rd = torch.abs(rd) < rd_eps
        rd_safe = torch.where(
            rd >= 0.0,
            rd + rd_eps,
            rd - rd_eps,
        )

        d11 = 1.0 / (r * rd_safe + eps)
        qx = q * x11
        qy = q * y11
        xy = xi * y11

        cd_mask = torch.abs(cd) > GEOM_EPS

        x = torch.sqrt(xi * xi + q * q + eps * eps)

        atan_num = et * (x + q * cd) + x * (r + x) * sd
        atan_den = xi * (r + x) * cd

        xi_mask = torch.abs(xi) > eps
        den_safe = torch.where(xi_mask, atan_den, torch.ones_like(atan_den))
        ai4_a = torch.where(
            xi_mask,
            (xi / rd_safe * sd * cd + 2.0 * torch.atan(atan_num / den_safe)) / (cd * cd + eps),
            torch.zeros_like(xi),
        )

        ai3_a = (
            y * cd / rd_safe
            - ale
            + sd * torch.log(rd_safe)
        ) / (cd * cd + eps)

        ak1_a = (xi * (d11 - y11 * sd)) / (cd + eps)
        ak3_a = (q * y11 - y * d11) / (cd + eps)
        aj2_a = (xi * y / rd_safe) * d11
        aj5_a = -(d + y * y / rd_safe) * d11
        aj3_a = (ak1_a - aj2_a * sd) / (cd + eps)
        aj6_a = (ak3_a - aj5_a * sd) / (cd + eps)

        rd2 = rd_safe * rd_safe
        ai3_b = (et / rd_safe + y * q / rd2 - ale) / 2.0
        ai4_b = (xi * y / rd2) / 2.0
        ak1_b = (xi * q / rd_safe) * d11
        ak3_b = (sd / rd_safe) * (xi * xi * d11 - 1.0)
        aj2_b = (xi * y / rd_safe)  * d11
        aj5_b = -(d + y * y / rd_safe) * d11
        aj3_b = -(xi / rd2) * (q * q * d11 - 0.5)
        aj6_b = -(y / rd2) * (xi * xi * d11 - 0.5)

        ai3 = torch.where(cd_mask, ai3_a, ai3_b)
        ai4 = torch.where(cd_mask, ai4_a, ai4_b)
        ak1 = torch.where(cd_mask, ak1_a, ak1_b)
        ak3 = torch.where(cd_mask, ak3_a, ak3_b)
        aj2 = torch.where(cd_mask, aj2_a, aj2_b)
        aj3 = torch.where(cd_mask, aj3_a, aj3_b)
        aj5 = torch.where(cd_mask, aj5_a, aj5_b)
        aj6 = torch.where(cd_mask, aj6_a, aj6_b)

        ai1 = -xi / rd_safe * cd - ai4 * sd
        ai2 = torch.log(rd_safe) + ai3 * sd
        ak2 = 1.0 / r + ak3 * sd
        ak4 = xy * cd - ak1 * sd
        aj1 = aj5 * cd - aj6 * sd
        aj4 = -xy - aj2 * cd + aj3 * sd

        ux_ss = -xi * qy - tt - alp3 * ai1 * sd
        uy_ss = -q / r + alp3 * y / rd_safe * sd
        uz_ss = q * qy - alp3 * ai2 * sd

        ux_ds = -q / r + alp3 * ai3 * sdcd
        uy_ds = -et * qx - tt - alp3 * xi / rd_safe * sdcd
        uz_ds = q * qx + alp3 * ai4 * sdcd

        ux_tf = q * qy - alp3 * ai3 * sdsd
        uy_tf = q * qx + alp3 * xi / rd_safe * sdsd
        uz_tf = et * qx + xi * qy - tt - alp3 * ai4 * sdsd

        ux = (disl1 * ux_ss + disl2 * ux_ds + disl3 * ux_tf) / pi2
        uy = (disl1 * uy_ss + disl2 * uy_ds + disl3 * uy_tf) / pi2
        uz = (disl1 * uz_ss + disl2 * uz_ds + disl3 * uz_tf) / pi2

        ux = torch.where(near_rd, torch.zeros_like(ux), ux)
        uy = torch.where(near_rd, torch.zeros_like(uy), uy)
        uz = torch.where(near_rd, torch.zeros_like(uz), uz)

        return UBDisplacement(ux=ux, uy=uy, uz=uz)

    # ---------------- training-safe behavior ----------------

    # Smooth reciprocals / logs / branches
    rd_pos = _soft_pos(rd, rd_eps)
    rd2 = rd_pos * rd_pos

    d11 = _safe_inv(r * rd_pos, smooth_eps)
    qx = q * x11
    qy = q * y11
    xy = xi * y11

    # Smooth blend between cd!=0 formulas and cd≈0 formulas
    cd_blend = _soft_blend_from_abs(cd, blend_eps)

    x = torch.sqrt(xi * xi + q * q + smooth_eps * smooth_eps)

    atan_num = et * (x + q * cd) + x * (r + x) * sd
    atan_den = xi * (r + x) * cd

    xi_mask = torch.abs(xi) > eps
    den_safe = torch.where(atan_den >= 0.0, atan_den + eps, atan_den - eps)
    atan_term = 2.0 * torch.atan(atan_num / den_safe)

    ai4_a_num = _safe_div(xi, rd_pos, smooth_eps) * sd * cd + atan_term
    ai4_a = torch.where(xi_mask, _safe_div(ai4_a_num, cd * cd, smooth_eps), torch.zeros_like(xi))

    ai3_a_num = _safe_div(y * cd, rd_pos, smooth_eps) - ale + sd * torch.log(rd_pos)
    ai3_a = _safe_div(ai3_a_num, cd * cd, smooth_eps)

    ak1_a = _safe_div(xi * (d11 - y11 * sd), cd, smooth_eps)
    ak3_a = _safe_div(q * y11 - y * d11, cd, smooth_eps)

    aj2_a = (xi * y * _safe_inv(rd_pos, smooth_eps)) * d11
    aj5_a = -(d + y * y * _safe_inv(rd_pos, smooth_eps)) * d11
    aj3_a = _safe_div(ak1_a - aj2_a * sd, cd, smooth_eps)
    aj6_a = _safe_div(ak3_a - aj5_a * sd, cd, smooth_eps)

    # cd≈0 branch
    ai3_b = 0.5 * (_safe_div(et, rd_pos, smooth_eps) + y * q * _safe_inv(rd2, smooth_eps) - ale)
    ai4_b = 0.5 * xi * y * _safe_inv(rd2, smooth_eps)

    ak1_b = (xi * q * _safe_inv(rd_pos, smooth_eps)) * d11
    ak3_b = (sd * _safe_inv(rd_pos, smooth_eps)) * (xi * xi * d11 - 1.0)

    aj2_b = (xi * y * _safe_inv(rd_pos, smooth_eps)) * d11
    aj5_b = -(d + y * y * _safe_inv(rd_pos, smooth_eps)) * d11
    aj3_b = -(xi * _safe_inv(rd2, smooth_eps)) * (q * q * d11 - 0.5)
    aj6_b = -(y * _safe_inv(rd2, smooth_eps)) * (xi * xi * d11 - 0.5)

    # Smooth blend instead of hard torch.where(cd_mask, ...)
    ai3 = cd_blend * ai3_a + (1.0 - cd_blend) * ai3_b
    ai4 = cd_blend * ai4_a + (1.0 - cd_blend) * ai4_b
    ak1 = cd_blend * ak1_a + (1.0 - cd_blend) * ak1_b
    ak3 = cd_blend * ak3_a + (1.0 - cd_blend) * ak3_b
    aj2 = cd_blend * aj2_a + (1.0 - cd_blend) * aj2_b
    aj3 = cd_blend * aj3_a + (1.0 - cd_blend) * aj3_b
    aj5 = cd_blend * aj5_a + (1.0 - cd_blend) * aj5_b
    aj6 = cd_blend * aj6_a + (1.0 - cd_blend) * aj6_b

    ai1 = -_safe_div(xi * cd, rd_pos, smooth_eps) - ai4 * sd
    ai2 = torch.log(rd_pos) + ai3 * sd

    ak2 = _safe_inv(r, smooth_eps) + ak3 * sd
    ak4 = xy * cd - ak1 * sd
    aj1 = aj5 * cd - aj6 * sd
    aj4 = -xy - aj2 * cd + aj3 * sd

    ux_ss = -xi * qy - tt - alp3 * ai1 * sd
    uy_ss = -q * _safe_inv(r, smooth_eps) + alp3 * y * _safe_inv(rd_pos, smooth_eps) * sd
    uz_ss = q * qy - alp3 * ai2 * sd

    ux_ds = -q * _safe_inv(r, smooth_eps) + alp3 * ai3 * sdcd
    uy_ds = -et * qx - tt - alp3 * xi * _safe_inv(rd_pos, smooth_eps) * sdcd
    uz_ds = q * qx + alp3 * ai4 * sdcd

    ux_tf = q * qy - alp3 * ai3 * sdsd
    uy_tf = q * qx + alp3 * xi * _safe_inv(rd_pos, smooth_eps) * sdsd
    uz_tf = et * qx + xi * qy - tt - alp3 * ai4 * sdsd

    ux = (disl1 * ux_ss + disl2 * ux_ds + disl3 * ux_tf) / pi2
    uy = (disl1 * uy_ss + disl2 * uy_ds + disl3 * uy_tf) / pi2
    uz = (disl1 * uz_ss + disl2 * uz_ds + disl3 * uz_tf) / pi2

    # Smooth attenuation near rd≈0 instead of hard zeroing
    rd_att = _soft_blend_from_abs(rd, rd_eps)
    ux = rd_att * ux
    uy = rd_att * uy
    uz = rd_att * uz

    return UBDisplacement(
        ux=ux,
        uy=uy,
        uz=uz,
    )


@dataclass(slots=True)
class UBResult:
    """UB kernel output: 3 displacements + 9 spatial derivatives.

    Field order matches the Fortran ``uB`` output ``du(1..12)``: ``ux, uy, uz``
    followed by the fault-frame derivatives ``u{x,y,z}{x,y,z}`` (e.g. ``uyx`` is
    ``d(uy)/dx``). All fields are ``[B, 2, 2, N]`` over the corner grid.
    """
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor
    uxx: torch.Tensor
    uyx: torch.Tensor
    uzx: torch.Tensor
    uxy: torch.Tensor
    uyy: torch.Tensor
    uzy: torch.Tensor
    uxz: torch.Tensor
    uyz: torch.Tensor
    uzz: torch.Tensor


def ub_displacement_and_derivatives(
    disl1: torch.Tensor,
    disl2: torch.Tensor,
    disl3: torch.Tensor,
    c0: DCCon0,
    c2: DCCon2,
):
    """Exact UB kernel returning displacement *and* its 9 spatial derivatives.

    Faithful port of the Fortran ``uB`` subroutine (``DC3D.f90``), which returns
    all 12 ``du`` components.  The existing :func:`ub_displacement` keeps only the
    displacement (``du1..3``); this adds the strain block (``du4..12``) needed for
    an analytic Jacobian of the Okada forward.

    Exact-mode only (no ``training_safe`` smoothing): the derivatives are the
    closed-form analytic strains, so they stay finite and Okada-table-accurate
    even on the singular manifolds where autograd of a smoothed forward drifts.

    Returns
    -------
    UBResult
        12 fault-frame components over the ``[B, 2, 2, N]`` corner grid.
    """
    pi2 = 2.0 * math.pi

    d1 = disl1[:, None, None, None]
    d2 = disl2[:, None, None, None]
    d3 = disl3[:, None, None, None]

    xi = c2.xi
    et = c2.et
    q = c2.q
    r = c2.r
    r3 = c2.r3
    y = c2.y
    d = c2.d
    x11 = c2.x11
    x32 = c2.x32
    y11 = c2.y11
    y32 = c2.y32
    ey, ez = c2.ey, c2.ez
    fy, fz = c2.fy, c2.fz
    gy, gz = c2.gy, c2.gz
    hy, hz = c2.hy, c2.hz
    theta = c2.tt
    ale = c2.ale

    alp3 = c0.alp3[:, None, None, None]
    sd = c0.sd[:, None, None, None]
    cd = c0.cd[:, None, None, None]

    xi2 = xi * xi
    q2 = q * q
    rd = r + d
    d11 = 1.0 / (r * rd)
    aj2 = xi * y / rd * d11
    aj5 = -(d + y * y / rd) * d11

    # cd != 0 and cd == 0 branches (Okada p.1034), mirroring ub_displacement.
    cd_mask = torch.abs(cd) > GEOM_EPS
    x_ = torch.sqrt(xi2 + q2)
    xi_mask = xi != 0.0
    den = xi * (r + x_) * cd
    den_safe = torch.where(xi_mask, den, torch.ones_like(den))
    ai4_a = torch.where(
        xi_mask,
        (xi / rd * sd * cd
         + 2.0 * torch.atan((et * (x_ + q * cd) + x_ * (r + x_) * sd) / den_safe))
        / (cd * cd),
        torch.zeros_like(xi),
    )
    ai3_a = (y * cd / rd - ale + sd * torch.log(rd)) / (cd * cd)
    ak1_a = xi * (d11 - y11 * sd) / cd
    ak3_a = (q * y11 - y * d11) / cd
    aj3_a = (ak1_a - aj2 * sd) / cd
    aj6_a = (ak3_a - aj5 * sd) / cd

    rd2 = rd * rd
    ai3_b = (et / rd + y * q / rd2 - ale) / 2.0
    ai4_b = xi * y / rd2 / 2.0
    ak1_b = xi * q / rd * d11
    ak3_b = sd / rd * (xi2 * d11 - 1.0)
    aj3_b = -xi / rd2 * (q2 * d11 - 0.5)
    aj6_b = -y / rd2 * (xi2 * d11 - 0.5)

    ai3 = torch.where(cd_mask, ai3_a, ai3_b)
    ai4 = torch.where(cd_mask, ai4_a, ai4_b)
    ak1 = torch.where(cd_mask, ak1_a, ak1_b)
    ak3 = torch.where(cd_mask, ak3_a, ak3_b)
    aj3 = torch.where(cd_mask, aj3_a, aj3_b)
    aj6 = torch.where(cd_mask, aj6_a, aj6_b)

    xy = xi * y11
    ai1 = -xi / rd * cd - ai4 * sd
    ai2 = torch.log(rd) + ai3 * sd
    ak2 = 1.0 / r + ak3 * sd
    ak4 = xy * cd - ak1 * sd
    aj1 = aj5 * cd - aj6 * sd
    aj4 = -xy - aj2 * cd + aj3 * sd

    qx = q * x11
    qy = q * y11

    # du[i] accumulates disl1/2/3 contributions, exactly as the Fortran loops.
    du = [torch.zeros_like(xi) for _ in range(12)]

    # strike-slip
    du[0] = du[0] + d1 / pi2 * (-xi * qy - theta - alp3 * ai1 * sd)
    du[1] = du[1] + d1 / pi2 * (-q / r + alp3 * y / rd * sd)
    du[2] = du[2] + d1 / pi2 * (q * qy - alp3 * ai2 * sd)
    du[3] = du[3] + d1 / pi2 * (xi2 * q * y32 - alp3 * aj1 * sd)
    du[4] = du[4] + d1 / pi2 * (xi * q / r3 - alp3 * aj2 * sd)
    du[5] = du[5] + d1 / pi2 * (-xi * q2 * y32 - alp3 * aj3 * sd)
    du[6] = du[6] + d1 / pi2 * (-xi * fy - d * x11 + alp3 * (xy + aj4) * sd)
    du[7] = du[7] + d1 / pi2 * (-ey + alp3 * (1.0 / r + aj5) * sd)
    du[8] = du[8] + d1 / pi2 * (q * fy - alp3 * (qy - aj6) * sd)
    du[9] = du[9] + d1 / pi2 * (-xi * fz - y * x11 + alp3 * ak1 * sd)
    du[10] = du[10] + d1 / pi2 * (-ez + alp3 * y * d11 * sd)
    du[11] = du[11] + d1 / pi2 * (q * fz + alp3 * ak2 * sd)

    # dip-slip
    du[0] = du[0] + d2 / pi2 * (-q / r + alp3 * ai3 * sd * cd)
    du[1] = du[1] + d2 / pi2 * (-et * qx - theta - alp3 * xi / rd * sd * cd)
    du[2] = du[2] + d2 / pi2 * (q * qx + alp3 * ai4 * sd * cd)
    du[3] = du[3] + d2 / pi2 * (xi * q / r3 + alp3 * aj4 * sd * cd)
    du[4] = du[4] + d2 / pi2 * (et * q / r3 + qy + alp3 * aj5 * sd * cd)
    du[5] = du[5] + d2 / pi2 * (-q2 / r3 + alp3 * aj6 * sd * cd)
    du[6] = du[6] + d2 / pi2 * (-ey + alp3 * aj1 * sd * cd)
    du[7] = du[7] + d2 / pi2 * (-et * gy - xy * sd + alp3 * aj2 * sd * cd)
    du[8] = du[8] + d2 / pi2 * (q * gy + alp3 * aj3 * sd * cd)
    du[9] = du[9] + d2 / pi2 * (-ez - alp3 * ak3 * sd * cd)
    du[10] = du[10] + d2 / pi2 * (-et * gz - xy * cd - alp3 * xi * d11 * sd * cd)
    du[11] = du[11] + d2 / pi2 * (q * gz - alp3 * ak4 * sd * cd)

    # tensile
    du[0] = du[0] + d3 / pi2 * (q * qy - alp3 * ai3 * sd * sd)
    du[1] = du[1] + d3 / pi2 * (q * qx + alp3 * xi / rd * sd * sd)
    du[2] = du[2] + d3 / pi2 * (et * qx + xi * qy - theta - alp3 * ai4 * sd * sd)
    du[3] = du[3] + d3 / pi2 * (-xi * q2 * y32 - alp3 * aj4 * sd * sd)
    du[4] = du[4] + d3 / pi2 * (-q2 / r3 - alp3 * aj5 * sd * sd)
    du[5] = du[5] + d3 / pi2 * (q * q2 * y32 - alp3 * aj6 * sd * sd)
    du[6] = du[6] + d3 / pi2 * (q * fy - alp3 * aj1 * sd * sd)
    du[7] = du[7] + d3 / pi2 * (q * gy - alp3 * aj2 * sd * sd)
    du[8] = du[8] + d3 / pi2 * (-q * hy - alp3 * aj3 * sd * sd)
    du[9] = du[9] + d3 / pi2 * (q * fz + alp3 * ak3 * sd * sd)
    du[10] = du[10] + d3 / pi2 * (q * gz + alp3 * xi * d11 * sd * sd)
    du[11] = du[11] + d3 / pi2 * (-q * hz + alp3 * ak4 * sd * sd)

    return UBResult(*du)


@dataclass(slots=True)
class UCResult:
    """UC (depth-dependent) kernel output: 3 displacements + 9 spatial derivatives.

    Field order matches the Fortran ``DC3D`` UC output: ``ux, uy, uz`` followed
    by the derivatives ``u{x,y,z}{x,y,z}``.
    """
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor
    uxx: torch.Tensor
    uyx: torch.Tensor
    uzx: torch.Tensor
    uxy: torch.Tensor
    uyy: torch.Tensor
    uzy: torch.Tensor
    uxz: torch.Tensor
    uyz: torch.Tensor
    uzz: torch.Tensor


def uc_displacement_and_derivatives(
    z: torch.Tensor,
    disl1: torch.Tensor,
    disl2: torch.Tensor,
    disl3: torch.Tensor,
    c0: DCCon0,
    c2: DCCon2,
):
    """
    PyTorch translation of Okada finite-fault UC subroutine.

    Parameters
    ----------
    z : tensor [B] or broadcastable
        Observation depth in Okada convention (same z you pass into DC3D).
    disl1, disl2, disl3 : tensor [B]
        Strike-slip, dip-slip, tensile components.
    c0 : DCCon0
        Output of dccon0(...)
    c2 : DCCon2
        Output of dccon2(...)

    Returns
    -------
    UCResult
        12 displacement / derivative components, matching the Fortran order:
        1 ux, 2 uy, 3 uz, 4 uxx, 5 uyx, 6 uzx,
        7 uxy, 8 uyy, 9 uzy, 10 uxz, 11 uyz, 12 uzz
    """
    pi2 = 2.0 * math.pi

    # Broadcast slips / z to [B,1,1,1]
    disl1 = disl1[:, None, None, None]
    disl2 = disl2[:, None, None, None]
    disl3 = disl3[:, None, None, None]

    # z = z[:, None, None, None]
    if z.ndim == 1:
        z = z[:, None, None, None]  # [B,1,1,1]
    elif z.ndim == 2:
        z = z[:, None, None, :]  # [B,1,1,N]
    else:
        raise ValueError("z must have shape [B] or [B,N]")

    # Broadcast medium / dip constants
    alp4 = c0.alp4[:, None, None, None]
    alp5 = c0.alp5[:, None, None, None]
    sd   = c0.sd[:,   None, None, None]
    cd   = c0.cd[:,   None, None, None]
    sdsd = c0.sdsd[:, None, None, None]
    cdcd = c0.cdcd[:, None, None, None]
    sdcd = c0.sdcd[:, None, None, None]

    # Pull geometry from c2
    xi  = c2.xi
    et  = c2.et
    q   = c2.q
    r   = c2.r
    r2  = c2.r2
    r3  = c2.r3
    r5  = c2.r5
    y   = c2.y
    d   = c2.d
    x11 = c2.x11
    y11 = c2.y11
    x32 = c2.x32
    y32 = c2.y32

    xi2 = xi * xi
    et2 = et * et
    q2  = q * q

    c = d + z

    # Fortran:
    # X53=(8*R2+9*R*XI+3*XI2)*X11^3/R2
    # Y53=(8*R2+9*R*ET+3*ET2)*Y11^3/R2
    x53 = (8.0 * r2 + 9.0 * r * xi + 3.0 * xi2) * (x11 ** 3) / r2
    y53 = (8.0 * r2 + 9.0 * r * et + 3.0 * et2) * (y11 ** 3) / r2

    h   = q * cd - z
    z32 = sd / r3 - h * y32
    z53 = 3.0 * sd / r5 - h * y53

    y0 = y11 - xi2 * y32
    z0 = z32 - xi2 * z53

    ppy = cd / r3 + q * y32 * sd
    ppz = sd / r3 - q * y32 * cd

    qq  = z * y32 + z32 + z0
    qqy = 3.0 * c * d / r5 - qq * sd
    qqz = 3.0 * c * y / r5 - qq * cd + q * y32

    xy = xi * y11
    qx = q * x11
    qy = q * y11
    qr = 3.0 * q / r5

    cdr = (c + d) / r3
    yy0 = y / r3 - y0 * cd

    # initialize all 12 components
    u1  = torch.zeros_like(xi)
    u2  = torch.zeros_like(xi)
    u3  = torch.zeros_like(xi)
    u4  = torch.zeros_like(xi)
    u5  = torch.zeros_like(xi)
    u6  = torch.zeros_like(xi)
    u7  = torch.zeros_like(xi)
    u8  = torch.zeros_like(xi)
    u9  = torch.zeros_like(xi)
    u10 = torch.zeros_like(xi)
    u11 = torch.zeros_like(xi)
    u12 = torch.zeros_like(xi)

    # ==================================================
    # STRIKE-SLIP CONTRIBUTION
    # ==================================================
    du1  = alp4 * xy * cd - alp5 * xi * q * z32
    du2  = alp4 * (cd / r + 2.0 * qy * sd) - alp5 * c * q / r3
    du3  = alp4 * qy * cd - alp5 * (c * et / r3 - z * y11 + xi2 * z32)
    du4  = alp4 * y0 * cd - alp5 * q * z0
    du5  = -alp4 * xi * (cd / r3 + 2.0 * q * y32 * sd) + alp5 * c * xi * qr
    du6  = -alp4 * xi * q * y32 * cd + alp5 * xi * (3.0 * c * et / r5 - qq)
    du7  = -alp4 * xi * ppy * cd - alp5 * xi * qqy
    du8  = (
        alp4 * 2.0 * (d / r3 - y0 * sd) * sd
        - y / r3 * cd
        - alp5 * (cdr * sd - et / r3 - c * y * qr)
    )
    du9  = (
        -alp4 * q / r3
        + yy0 * sd
        + alp5 * (cdr * cd + c * d * qr - (y0 * cd + q * z0) * sd)
    )
    du10 = alp4 * xi * ppz * cd - alp5 * xi * qqz
    du11 = (
        alp4 * 2.0 * (y / r3 - y0 * cd) * sd
        + d / r3 * cd
        - alp5 * (cdr * cd + c * d * qr)
    )
    du12 = (
        yy0 * cd
        - alp5 * (cdr * sd - c * y * qr - y0 * sdsd + q * z0 * cd)
    )

    u1  = u1  + disl1 / pi2 * du1
    u2  = u2  + disl1 / pi2 * du2
    u3  = u3  + disl1 / pi2 * du3
    u4  = u4  + disl1 / pi2 * du4
    u5  = u5  + disl1 / pi2 * du5
    u6  = u6  + disl1 / pi2 * du6
    u7  = u7  + disl1 / pi2 * du7
    u8  = u8  + disl1 / pi2 * du8
    u9  = u9  + disl1 / pi2 * du9
    u10 = u10 + disl1 / pi2 * du10
    u11 = u11 + disl1 / pi2 * du11
    u12 = u12 + disl1 / pi2 * du12

    # ==================================================
    # DIP-SLIP CONTRIBUTION
    du1  = alp4 * cd / r - qy * sd - alp5 * c * q / r3
    du2  = alp4 * y * x11 - alp5 * c * et * q * x32
    du3  = -d * x11 - xy * sd - alp5 * c * (x11 - q2 * x32)
    du4  = -alp4 * xi / r3 * cd + alp5 * c * xi * qr + xi * q * y32 * sd
    du5  = -alp4 * y / r3 + alp5 * c * et * qr
    du6  = d / r3 - y0 * sd + alp5 * c / r3 * (1.0 - 3.0 * q2 / r2)
    du7  = -alp4 * et / r3 + y0 * sdsd - alp5 * (cdr * sd - c * y * qr)
    du8  = (
        alp4 * (x11 - y * y * x32)
        - alp5 * c * ((d + 2.0 * q * cd) * x32 - y * et * q * x53)
    )
    du9  = (
        xi * ppy * sd + y * d * x32
        + alp5 * c * ((y + 2.0 * q * sd) * x32 - y * q2 * x53)
    )
    du10 = -q / r3 + y0 * sdcd - alp5 * (cdr * cd + c * d * qr)
    du11 = (
        alp4 * y * d * x32
        - alp5 * c * ((y - 2.0 * q * sd) * x32 + d * et * q * x53)
    )
    du12 = (
        -xi * ppz * sd + x11 - d * d * x32
        - alp5 * c * ((d - 2.0 * q * cd) * x32 - d * q2 * x53)
    )

    u1  = u1  + disl2 / pi2 * du1
    u2  = u2  + disl2 / pi2 * du2
    u3  = u3  + disl2 / pi2 * du3
    u4  = u4  + disl2 / pi2 * du4
    u5  = u5  + disl2 / pi2 * du5
    u6  = u6  + disl2 / pi2 * du6
    u7  = u7  + disl2 / pi2 * du7
    u8  = u8  + disl2 / pi2 * du8
    u9  = u9  + disl2 / pi2 * du9
    u10 = u10 + disl2 / pi2 * du10
    u11 = u11 + disl2 / pi2 * du11
    u12 = u12 + disl2 / pi2 * du12

    # ==================================================
    # TENSILE CONTRIBUTION
    # ==================================================
    du1  = -alp4 * (sd / r + qy * cd) - alp5 * (z * y11 - q2 * z32)
    du2  = alp4 * 2.0 * xy * sd + d * x11 - alp5 * c * (x11 - q2 * x32)
    du3  = alp4 * (y * x11 + xy * cd) + alp5 * q * (c * et * x32 + xi * z32)
    du4  = alp4 * xi / r3 * sd + xi * q * y32 * cd + alp5 * xi * (3.0 * c * et / r5 - 2.0 * z32 - z0)
    du5  = alp4 * 2.0 * y0 * sd - d / r3 + alp5 * c / r3 * (1.0 - 3.0 * q2 / r2)
    du6  = -alp4 * yy0 - alp5 * (c * et * qr - q * z0)
    du7  = alp4 * (q / r3 + y0 * sdcd) + alp5 * (z / r3 * cd + c * d * qr - q * z0 * sd)
    du8  = (
        -alp4 * 2.0 * xi * ppy * sd - y * d * x32
        + alp5 * c * ((y + 2.0 * q * sd) * x32 - y * q2 * x53)
    )
    du9  = (
        -alp4 * (xi * ppy * cd - x11 + y * y * x32)
        + alp5 * (c * ((d + 2.0 * q * cd) * x32 - y * et * q * x53) + xi * qqy)
    )
    du10 = -et / r3 + y0 * cdcd - alp5 * (z / r3 * sd - c * y * qr - y0 * sdsd + q * z0 * cd)
    du11 = (
        alp4 * 2.0 * xi * ppz * sd - x11 + d * d * x32
        - alp5 * c * ((d - 2.0 * q * cd) * x32 - d * q2 * x53)
    )
    du12 = (
        alp4 * (xi * ppz * cd + y * d * x32)
        + alp5 * (c * ((y - 2.0 * q * sd) * x32 + d * et * q * x53) + xi * qqz)
    )

    u1  = u1  + disl3 / pi2 * du1
    u2  = u2  + disl3 / pi2 * du2
    u3  = u3  + disl3 / pi2 * du3
    u4  = u4  + disl3 / pi2 * du4
    u5  = u5  + disl3 / pi2 * du5
    u6  = u6  + disl3 / pi2 * du6
    u7  = u7  + disl3 / pi2 * du7
    u8  = u8  + disl3 / pi2 * du8
    u9  = u9  + disl3 / pi2 * du9
    u10 = u10 + disl3 / pi2 * du10
    u11 = u11 + disl3 / pi2 * du11
    u12 = u12 + disl3 / pi2 * du12

    return UCResult(
        ux=u1,  uy=u2,  uz=u3,
        uxx=u4, uyx=u5, uzx=u6,
        uxy=u7, uyy=u8, uzy=u9,
        uxz=u10, uyz=u11, uzz=u12,
    )


@dataclass(slots=True)
class UCDisplacement:
    """UC kernel displacement only (the three components, no derivatives)."""
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor


def uc_displacement_only(
        z: torch.Tensor,
        disl1: torch.Tensor,
        disl2: torch.Tensor,
        disl3: torch.Tensor,
        c0: DCCon0,
        c2: DCCon2,
):
    """Convenience wrapper returning only the UC displacement (no derivatives).

    Calls :func:`uc_displacement_and_derivatives` and keeps just the three
    displacement components as a :class:`UCDisplacement`.
    """
    full = uc_displacement_and_derivatives(
        z=z,
        disl1=disl1,
        disl2=disl2,
        disl3=disl3,
        c0=c0,
        c2=c2,
    )
    return UCDisplacement(
        ux=full.ux,
        uy=full.uy,
        uz=full.uz,
    )


def _dccon2_on_corners(x, p, q, al1_b, al2_b, aw1_b, aw2_b, c0, *,
                       dtype, num_eps, geom_eps, smooth_eps, training_safe):
    """Build the [B,2,2,N] (xi, et, q) corner grid for one source and evaluate
    ``dccon2`` on it.

    Encodes Okada's Fortran corner semantics
    ``DCCON2(XI(J), ET(K), Q, SD, CD, KXI(K), KET(J))`` -- kxi varies with the ET
    corner, ket with the XI corner. Shared verbatim by the finite fault's real and
    image sources and by the surface-only model, so it lives here once.

    Returns ``(xi4, et4, q4, c2)``.
    """
    xi = torch.stack([x - al1_b, x - al2_b], dim=1)     # [B,2,N]
    et = torch.stack([p - aw1_b, p - aw2_b], dim=1)     # [B,2,N]
    xi4 = xi[:, :, None, :].expand(-1, 2, 2, -1)        # [B,2,2,N]
    et4 = et[:, None, :, :].expand(-1, 2, 2, -1)
    q4 = q[:, None, None, :].expand(-1, 2, 2, -1)

    xi1 = xi4[:, 0, 0, :]
    xi2 = xi4[:, 1, 0, :]
    et1 = et4[:, 0, 0, :]
    et2 = et4[:, 0, 1, :]
    q0 = q4[:, 0, 0, :]

    r12 = torch.sqrt(xi1 * xi1 + et2 * et2 + q0 * q0)
    r21 = torch.sqrt(xi2 * xi2 + et1 * et1 + q0 * q0)
    r22 = torch.sqrt(xi2 * xi2 + et2 * et2 + q0 * q0)

    kxi1 = (xi1 < 0.0) & ((r21 + xi2) < geom_eps)
    kxi2 = (xi1 < 0.0) & ((r22 + xi2) < geom_eps)
    ket1 = (et1 < 0.0) & ((r12 + et2) < geom_eps)
    ket2 = (et1 < 0.0) & ((r22 + et2) < geom_eps)

    kxi = torch.stack([kxi1, kxi2], dim=1)[:, None, :, :].expand(-1, 2, 2, -1)
    ket = torch.stack([ket1, ket2], dim=1)[:, :, None, :].expand(-1, 2, 2, -1)

    c2 = dccon2(
        xi=xi4, et=et4, q=q4,
        sd=c0.sd[:, None, None, None],
        cd=c0.cd[:, None, None, None],
        kxi=kxi, ket=ket,
        internal_dtype=dtype,
        eps=num_eps,
        training_safe=training_safe,
        smooth_eps=smooth_eps,
        geom_eps=geom_eps,
    )
    return xi4, et4, q4, c2


def _singular_mask(xi4, et4, q4, geom_eps):
    """Okada's on-dislocation zeroing mask [B, N] from a corner grid: points where
    ``q == 0`` and an edge is straddled. Corner values below ``geom_eps`` are
    snapped to zero first (Okada's convention)."""
    xi1 = xi4[:, 0, 0, :]
    xi2 = xi4[:, 1, 0, :]
    et1 = et4[:, 0, 0, :]
    et2 = et4[:, 0, 1, :]
    q0 = q4[:, 0, 0, :]

    def snap(t):
        return torch.where(torch.abs(t) < geom_eps, torch.zeros_like(t), t)

    qz, xi1z, xi2z, et1z, et2z = snap(q0), snap(xi1), snap(xi2), snap(et1), snap(et2)
    return (
        (qz == 0.0) &
        (
            (((xi1z * xi2z) <= 0.0) & ((et1z * et2z) == 0.0)) |
            (((et1z * et2z) <= 0.0) & ((xi1z * xi2z) == 0.0))
        )
    )  # [B, N]


class _OkadaBase(SourceModel):
    """
    Shared configuration for the Okada source models.

    Holds the common constructor (elastic/dip constants and the regularisation
    epsilons) and the centroid-based fault-geometry helper used by both
    :class:`OkadaSource` and :class:`OkadaSourceSimple`. Not meant to be
    instantiated directly -- it leaves ``forward`` abstract.
    """

    def __init__(
        self,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        internal_dtype: torch.dtype = torch.float64,
        smooth_grad: bool = False,
        analytic_grad: bool = False,
        num_eps: float | None = None,
        f32_vertical_band: float = F32_VERTICAL_BAND,
    ):
        """
        Parameters
        ----------
        poisson_ratio : float, default 0.25
            Poisson's ratio of the elastic half-space (sets ``alpha``).
        internal_dtype : torch.dtype, default torch.float64
            Dtype used for the internal computation; inputs are cast to it.
        smooth_grad : bool, default False
            Gradient mode. If True, replace the hard singularity branches with
            smooth blends/attenuations so gradients stay finite everywhere (see
            module docstring). This also perturbs the *forward values* slightly
            near the singular manifolds. Mutually exclusive with ``analytic_grad``.
        analytic_grad : bool, default False
            Gradient mode. If True, keep the exact (un-smoothed) forward values
            and return closed-form Okada strains as the backward, giving gradients
            that are accurate even on the singular manifolds (vertical dip,
            on-fault/on-trace points) where ``smooth_grad`` drifts and plain
            autograd hits a ``cos(dip) = 0`` NaN. Costs ~2x the backward (the
            forward is unchanged); see the README example. Mutually exclusive with
            ``smooth_grad``. First-order only: this is a custom
            ``autograd.Function``, so Hessians via double-backward are not
            available in this mode -- use the default mode (a plain-autograd graph)
            when you need second-order information.
        num_eps : float or None, default None
            Numerical guard for denominators/logs/sqrt. ``None`` picks a floor
            matched to ``internal_dtype`` (``1e-12`` for float64 underflows
            float32, re-exposing the singularities); pass a float to override.
            The remaining internal guards (``geom_eps``, ``rd_eps``,
            ``smooth_eps``, ``blend_eps``) are no longer constructor knobs -- they
            are derived from ``internal_dtype`` per call (see ``_okada_grad_floors``
            and the module ``*_EPS`` constants).
        f32_vertical_band : float, default ``F32_VERTICAL_BAND`` (0.1)
            Only relevant for a reduced-precision ``internal_dtype`` (float32/16).
            ``|cos(dip)|`` below which a batch is promoted to float64 for the near-
            vertical band the reduced precision cannot resolve (see
            ``_compute_dtype``). Trades float32 speed against near-vertical
            accuracy: the 0.1 default keeps float32 for dips up to ~84 deg but lets
            the error there reach ~1e-3; raise toward ~0.2 to hold the error under
            ~1e-3 through the edge (promoting steep, >78 deg, faults to float64), or
            lower it to keep more scenes in float32 if you can tolerate the error.
            See the ``F32_VERTICAL_BAND`` note for the measured tradeoff.
        """
        super().__init__()
        if smooth_grad and analytic_grad:
            raise ValueError(
                "smooth_grad and analytic_grad are mutually exclusive gradient "
                "modes; set at most one to True."
            )
        self.alpha = 1.0 / (2.0 * (1.0 - poisson_ratio))
        self.internal_dtype = internal_dtype
        self.smooth_grad = smooth_grad
        self.analytic_grad = analytic_grad
        # None -> dtype-appropriate floor resolved per call (see _resolve_num_eps).
        self.num_eps = num_eps
        self.f32_vertical_band = f32_vertical_band

    def _compute_dtype(self, dip: Tensor) -> torch.dtype:
        """Internal compute dtype for this evaluation.

        Normally ``self.internal_dtype``. A reduced-precision request is bumped
        to ``float64`` when the batch contains a near-vertical dip
        (``|cos(dip)| < self.f32_vertical_band``): there the correct value is
        Okada's general-dip formula, whose ``1/cos(dip)**2`` terms float32 cannot
        resolve (see ``F32_VERTICAL_BAND``). This is not a numerical guard or an
        alternative formula -- it is the *same* exact formula, evaluated in the
        precision it requires, with the result cast back to ``internal_dtype`` so
        the choice is invisible. Well-conditioned scenes keep the fast
        reduced-precision path.
        """
        requested = self.internal_dtype
        if requested == torch.float64:
            return requested
        cd = torch.cos(dip.detach().to(torch.float64)).abs()
        if bool((cd < self.f32_vertical_band).any()):
            return torch.float64
        return requested

    def _f64_twin(self) -> "_OkadaBase":
        """A cheap float64 copy of this model (shares nothing mutable of note).

        Lets the analytic backward evaluate the wide-step dip finite difference
        in float64. Two things need the precision near vertical: Okada's exact
        general-dip forward (as in :meth:`_compute_dtype`) and the FD subtraction
        of two nearly-equal losses. Both run in double; only the resulting
        gradient is cast back to the input dtype.
        """
        twin = copy.copy(self)
        twin.internal_dtype = torch.float64
        return twin

    @staticmethod
    def _validate_geometry(centroid_depth, length, width, dip):
        """Reject unphysical fault dimensions before they silently misbehave.

        ``length``/``width`` scale ``al1/al2`` and ``aw1/aw2``; a *negative*
        dimension flips the sign of the Chinnery subtraction and returns a
        plausible-looking but sign-flipped displacement (a zero dimension
        collapses the fault to zero output), neither of which raises on its own.
        ``centroid_depth <= 0`` puts the centroid at/above the free surface,
        outside the buried-fault half-space the Okada solution is derived for.

        Protrusion (soft) : the shallowest edge of the dip-tilted rectangle sits
        at ``centroid_depth - (width/2)|sin(dip)|``. At ``0`` the fault just
        reaches the surface -- a valid surface-rupturing fault -- but a *negative*
        top-edge depth means part of the dislocation lies above the free surface,
        outside the elastic medium. That still evaluates (Chinnery's corner
        superposition is oblivious to the surface) but is not a valid half-space
        problem, so it is a ``warning`` rather than a hard error (it does not fail
        numerically, the ``= 0`` boundary is legitimate, and callers may probe the
        regime deliberately).
        """
        if bool((length <= 0).any()):
            raise ValueError("length must be strictly positive")
        if bool((width <= 0).any()):
            raise ValueError("width must be strictly positive")
        if bool((centroid_depth <= 0).any()):
            raise ValueError("centroid_depth must be strictly positive")
        # top (shallowest) edge depth; guard the == 0 surface-rupturing case with
        # GEOM_EPS (the physical "treat as zero" tolerance, metres) so floating-point
        # noise at the surface does not warn spuriously.
        top_edge = centroid_depth - 0.5 * width * torch.sin(dip).abs()
        if bool((top_edge < -GEOM_EPS).any()):
            warnings.warn(
                "Okada fault top edge is above the free surface "
                "(centroid_depth < (width/2)*|sin(dip)|); part of the fault lies "
                "outside the elastic half-space, so the solution is unphysical "
                "there. Bury the fault or reduce its width.",
                stacklevel=3,
            )

    @staticmethod
    def _fault_geometry(centroid_depth, length, width):
        """Centroid-based fault corners ``(al1, al2, aw1, aw2, depth)``.

        Along-strike half-lengths ``al1/al2 = -/+ length/2`` and down-dip
        half-widths ``aw1/aw2 = -/+ width/2``; depth is the centroid depth.
        """
        al1 = -0.5 * length
        al2 = +0.5 * length
        aw1 = -0.5 * width
        aw2 = +0.5 * width
        depth = centroid_depth
        return al1, al2, aw1, aw2, depth


class OkadaSource(_OkadaBase):
    """
    General finite-fault Okada displacement model for arbitrary observation depth z.

    Conventions
    -----------
    - Internal geometry is centroid-based:
        depth = centroid_depth
        aw1 = -width/2
        aw2 = +width/2
    - Observation z follows Okada convention:
        z = 0   at the surface
        z < 0   below the surface
        z > 0   invalid

    Input fault location (source_x, source_y) is the map location of the fault centroid.
    Construction parameters are documented on :class:`_OkadaBase`.
    """

    def forward(
        self,
        x_obs: Tensor, y_obs: Tensor, z_obs: Tensor,
        source_x: Tensor, source_y: Tensor, dip: Tensor, strike: Tensor,
        centroid_depth: Tensor, length: Tensor, width: Tensor,
        disl1: Tensor, disl2: Tensor, disl3: Tensor,
    ) -> Displacement:
        """Displacement of a finite rectangular fault at observation depth ``z``.

        With ``analytic_grad=True`` and gradients required, routes through the
        closed-form analytic backward; otherwise evaluates directly. The forward
        values are identical either way. See :meth:`_evaluate` for the full
        parameter docs.
        """
        args = (x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3)
        if (self.analytic_grad and torch.is_grad_enabled()
                and any(a.requires_grad for a in args)):
            e, n, u = _OkadaAnalyticFn.apply(self, *args)
            return Displacement(e=e, n=n, u=u)
        return self._evaluate(*args)

    def _evaluate(
        self,
        x_obs: Tensor,          # [B, N]
        y_obs: Tensor,          # [B, N]
        z_obs: Tensor,          # [B] or [B, N], Okada convention: z <= 0
        source_x: Tensor,        # [B]
        source_y: Tensor,        # [B]
        dip: Tensor,            # [B] radians
        strike: Tensor,         # [B] radians
        centroid_depth: Tensor, # [B], meters
        length: Tensor,         # [B], meters
        width: Tensor,          # [B], meters
        disl1: Tensor,               # [B], meters
        disl2: Tensor,               # [B], meters
        disl3: Tensor,               # [B], meters
        return_strain: bool = False,
    ) -> "Displacement | tuple[Displacement, _StrainDict]":
        """Displacement of a finite rectangular fault at observation depth ``z``.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        z_obs : Tensor
            Observation depth [B] (or per-pixel [B, N]), Okada convention
            ``z <= 0`` (0 at the surface, negative below). A positive value
            raises ``ValueError``.
        source_x, source_y : Tensor
            Map position of the fault centroid [B] in metres.
        dip, strike : Tensor
            Fault dip and strike [B] in radians.
        centroid_depth : Tensor
            Depth of the fault centroid [B] in metres (positive down; must be
            > 0, i.e. the fault is buried below the free surface).
        length, width : Tensor
            Fault length (along strike) and width (down dip) [B] in metres
            (both must be > 0).
        disl1, disl2, disl3 : Tensor
            Strike-slip, dip-slip and tensile (opening) dislocations [B] in metres.
        return_strain : bool
            When ``True`` also return the analytic obs-coordinate Jacobian used by
            the ``analytic_grad`` backward (see Returns).

        Returns
        -------
        Displacement, or (Displacement, strain)
            ENU displacement [B, N] in metres (singular points zeroed). With
            ``return_strain=True`` a ``(Displacement, strain)`` pair, where
            ``strain`` maps ``"x_obs"``/``"y_obs"``/``"z_obs"`` to the
            ``(d ue, d un, d uu)`` triple w.r.t. that observation coordinate.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "dip": dip,
             "strike": strike, "centroid_depth": centroid_depth,
             "length": length, "width": width,
             "disl1": disl1, "disl2": disl2, "disl3": disl3},
        )

        requested_dtype = self.internal_dtype
        # Near-vertical scenes need Okada's exact general-dip formula evaluated in
        # float64 (float32 cannot resolve its 1/cos(dip)**2 terms); _compute_dtype
        # bumps the compute dtype there and we cast back to requested_dtype on
        # return. num_eps/floors follow the *compute* dtype.
        dtype = self._compute_dtype(dip)
        # Numerical guards, scaled to the compute dtype (the float64 values are
        # the validated defaults; float32's ~1e-7 epsilon would swallow them).
        num_eps = self.num_eps if self.num_eps is not None else default_num_eps(dtype)
        geom_eps = GEOM_EPS            # physical "treat as zero" tolerance (metres)
        smooth_eps, rd_eps = _okada_grad_floors(dtype)
        blend_eps = BLEND_EPS

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        z_obs = z_obs.to(dtype)

        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)

        dip = dip.to(dtype)
        strike = strike.to(dtype)
        centroid_depth = centroid_depth.to(dtype)
        length = length.to(dtype)
        width = width.to(dtype)

        disl1 = disl1.to(dtype)
        disl2 = disl2.to(dtype)
        disl3 = disl3.to(dtype)


        if z_obs.ndim == 1:
            z_b = z_obs[:, None]   # [B,1]
        else:
            z_b = z_obs            # [B,N]

        if torch.any(z_b > 0):
            raise ValueError("Okada convention requires z <= 0. Use z=0 at surface, z<0 below surface.")

        self._validate_geometry(centroid_depth, length, width, dip)

        al1, al2, aw1, aw2, depth = self._fault_geometry(
            centroid_depth.to(dtype),
            length.to(dtype),
            width.to(dtype),
        )

        # Observation coordinates relative to centroid reference point
        dx = x_obs - source_x.to(dtype)[:, None]
        dy = y_obs - source_y.to(dtype)[:, None]

        ss = torch.sin(strike.to(dtype))
        cs = torch.cos(strike.to(dtype))

        # Same local coordinate convention as your working simplified/native class
        x = dx * ss[:, None] + dy * cs[:, None]
        y = dx * cs[:, None] - dy * ss[:, None]

        c0 = dccon0(
            alpha=torch.as_tensor(self.alpha, device=x.device, dtype=dtype),
            dip_rad=dip.to(dtype),
            internal_dtype=dtype,
            training_safe=self.smooth_grad,
            geom_eps=geom_eps,
        )

        # -----------------------------------------
        # REAL-SOURCE contribution: D = depth + z
        # -----------------------------------------
        d_real = depth[:, None] + z_b
        sd_b = c0.sd[:, None]
        cd_b = c0.cd[:, None]

        p_real = y * cd_b + d_real * sd_b
        q_real = y * sd_b - d_real * cd_b

        # [B,2,N]
        al1_b = al1[:, None]
        al2_b = al2[:, None]
        aw1_b = aw1[:, None]
        aw2_b = aw2[:, None]

        # Real source corner grid + dccon2 (Okada Fortran KXI/KET semantics).
        _, _, _, c2_real = _dccon2_on_corners(
            x, p_real, q_real, al1_b, al2_b, aw1_b, aw2_b, c0,
            dtype=dtype, num_eps=num_eps, geom_eps=geom_eps,
            smooth_eps=smooth_eps, training_safe=self.smooth_grad,
        )

        ua_real = ua_displacement(
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2_real,
        )

        # Real-source displacement rotated into map / ENU frame
        cd4 = c0.cd[:, None, None, None]
        sd4 = c0.sd[:, None, None, None]

        real_x = -ua_real.ux
        real_y = -ua_real.uy * cd4 + ua_real.uz * sd4
        real_z = -ua_real.uy * sd4 - ua_real.uz * cd4

        # -----------------------------------------
        # IMAGE-SOURCE contribution: D = depth - z
        # -----------------------------------------
        d_img = depth[:, None] - z_b

        p_img = y * cd_b + d_img * sd_b
        q_img = y * sd_b - d_img * cd_b

        # Image source (D = depth - z): corner grid, dccon2, on-dislocation mask.
        xi_i4, et_i4, q_i4, c2_img = _dccon2_on_corners(
            x, p_img, q_img, al1_b, al2_b, aw1_b, aw2_b, c0,
            dtype=dtype, num_eps=num_eps, geom_eps=geom_eps,
            smooth_eps=smooth_eps, training_safe=self.smooth_grad,
        )
        singular = _singular_mask(xi_i4, et_i4, q_i4, geom_eps)

        ua_img = ua_displacement(
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2_img,
        )

        ub_img = ub_displacement(
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2_img,
            eps=num_eps,
            training_safe=self.smooth_grad,
            smooth_eps=smooth_eps,
            blend_eps=blend_eps,
            rd_eps=rd_eps,
        )

        uc_img = uc_displacement_only(
            z=z_b,
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2_img,
        )

        # -------------------------------------------------
        # IMAGE contribution:
        #   UA + UB + z * UC
        #
        # IMPORTANT:
        # The vertical rotation/sign pattern follows Okada DC3D.
        # -------------------------------------------------

        z4 = z_b[:, None, None, :]   # [B,1,1,N]

        img_x = (
                ua_img.ux
                + ub_img.ux
                + z4 * uc_img.ux
        )

        img_y = (
            (
                ua_img.uy
                + ub_img.uy
                + z4 * uc_img.uy
            ) * cd4
            -
            (
                ua_img.uz
                + ub_img.uz
                + z4 * uc_img.uz
            ) * sd4
        )

        # NOTE:
        # The Z*UC term enters with opposite sign here.
        img_z = (
            (
                ua_img.uy
                + ub_img.uy
                - z4 * uc_img.uy
            ) * sd4
            +
            (
                ua_img.uz
                + ub_img.uz
                - z4 * uc_img.uz
            ) * cd4
        )

        # -------------------------------------------------
        # Signed corner summation
        #
        # (+ - - +) pattern
        # -------------------------------------------------

        def corner_sum(arr):
            return (
                arr[:, 0, 0, :]
                - arr[:, 0, 1, :]
                - arr[:, 1, 0, :]
                + arr[:, 1, 1, :]
            )

        ux_fault = corner_sum(real_x + img_x)
        uy_fault = corner_sum(real_y + img_y)
        uz_fault = corner_sum(real_z + img_z)

        # -------------------------------------------------
        # Rotate fault coordinates back to ENU
        # -------------------------------------------------

        ss2 = ss[:, None]
        cs2 = cs[:, None]

        ue = ux_fault * ss2 + uy_fault * cs2
        un = ux_fault * cs2 - uy_fault * ss2
        uu = uz_fault

        ue = torch.where(singular, torch.zeros_like(ue), ue)
        un = torch.where(singular, torch.zeros_like(un), un)
        uu = torch.where(singular, torch.zeros_like(uu), uu)

        disp = Displacement(e=ue, n=un, u=uu)
        if not return_strain:
            return disp if dtype == requested_dtype else disp.to(requested_dtype)

        # ------------------------------------------------------------------
        # Analytic Jacobian d(ue, un, uu) / d(x_obs, y_obs, z_obs) via the full
        # DC3D strain assembly: real source (UA, z-derivative group sign-flipped)
        # + image source (UA + UB + z*UC, z-derivative group gets an extra +UC
        # displacement term).  Each of the 12 components is assembled in the fault
        # frame, then rotated to ENU and chained to the observation coordinates.
        # ------------------------------------------------------------------
        uaR = ua_displacement_and_derivatives(disl1, disl2, disl3, c0, c2_real)
        uaI = ua_displacement_and_derivatives(disl1, disl2, disl3, c0, c2_img)
        ubI = ub_displacement_and_derivatives(disl1, disl2, disl3, c0, c2_img)
        ucI = uc_displacement_and_derivatives(z_b, disl1, disl2, disl3, c0, c2_img)
        z4 = z_b[:, None, None, :]

        def _grp(R, g):
            return (
                (R.ux, R.uy, R.uz),
                (R.uxx, R.uyx, R.uzx),
                (R.uxy, R.uyy, R.uzy),
                (R.uxz, R.uyz, R.uzz),
            )[g]

        uc_disp = (ucI.ux, ucI.uy, ucI.uz)   # UC displacement (extra z-deriv term)

        def _fault_group(g):
            """Assemble fault-frame (Fx, Fy, Fz) for derivative group g (0=disp,
            1=d/dx, 2=d/dy, 3=d/dz), summed over corners."""
            ar, br, cr = _grp(uaR, g)
            if g != 3:
                rx = -ar
                ry = -br * cd4 + cr * sd4
                rz = -br * sd4 - cr * cd4
            else:                            # z-derivative group: sign flipped
                rx = ar
                ry = br * cd4 - cr * sd4
                rz = br * sd4 + cr * cd4

            aa, ba, ca = _grp(uaI, g)
            ab, bb, cb = _grp(ubI, g)
            ac, bc, cc = _grp(ucI, g)
            sb = ba + bb
            sc = ca + cb
            ix = aa + ab + z4 * ac
            iy = (sb + z4 * bc) * cd4 - (sc + z4 * cc) * sd4
            iz = (sb - z4 * bc) * sd4 + (sc - z4 * cc) * cd4
            if g == 3:                       # + UC displacement term
                ix = ix + uc_disp[0]
                iy = iy + uc_disp[1] * cd4 - uc_disp[2] * sd4
                iz = iz - uc_disp[1] * sd4 - uc_disp[2] * cd4

            return (corner_sum(rx + ix), corner_sum(ry + iy), corner_sum(rz + iz))

        dFdx = _fault_group(1)   # d(fault disp)/dx_fault
        dFdy = _fault_group(2)
        dFdz = _fault_group(3)

        def _enu(fx, fy, fz):
            return fx * ss2 + fy * cs2, fx * cs2 - fy * ss2, fz

        # chain fault directions -> observation coords (z unrotated by strike)
        def _obs(gx, gy, gz):
            # gx/gy/gz are fault-frame triples d(F)/d(x_f, y_f, z_f)
            jx = _enu(*(ss2 * a + cs2 * b for a, b in zip(gx, gy)))
            jy = _enu(*(cs2 * a - ss2 * b for a, b in zip(gx, gy)))
            jz = _enu(*gz)
            zero = torch.zeros_like(jx[0])
            sg = lambda tup: tuple(torch.where(singular, zero, v) for v in tup)
            return sg(jx), sg(jy), sg(jz)

        jx, jy, jz = _obs(dFdx, dFdy, dFdz)
        strain = {
            "x_obs": jx,   # each: (d ue, d un, d uu) / d(coord)
            "y_obs": jy,
            "z_obs": jz,
        }
        if dtype != requested_dtype:
            disp = disp.to(requested_dtype)
            strain = {k: tuple(t.to(requested_dtype) for t in v)
                      for k, v in strain.items()}
        return disp, strain


def _dip_fd_fallback(model, ev, dip, g_dip_ag, ge, gn, gu, disp_at_dip):
    """Overwrite the autograd dip gradient with a Richardson FD near vertical.

    ``g_dip_ag`` is autograd of the exact forward, which is exact to ~machine
    precision away from the vertical manifold but ill-conditioned in a thin band
    (``|cos(dip)| < DIP_FD_BAND``) where the kernel's ``1/cos(dip)**2`` terms blow
    up (and dip is zeroed exactly at 90 deg, where ``dccon0`` snaps ``cos(dip)``).
    For the -- rare -- batch elements in that band, fall back to Richardson
    extrapolation of the exact forward (see ``DIP_FD_RICH_STEP``): its wide sample
    steps clear the vertical singularity for every in-band dip and it cancels the
    O(H^2) truncation, giving ~1e-6 through vertical. The FD runs in float64: near
    vertical Okada's general-dip forward needs the precision (as in
    ``_compute_dtype``) and so does the subtraction of nearly-equal losses; only
    the result is cast back to dip's dtype. Skipped entirely (no extra forward
    evals) when no element is near vertical -- the common case.

    ``disp_at_dip(fd_ev, cast, dip_value)`` re-evaluates the exact forward at
    ``dip_value`` (every other parameter captured by the caller, cast via ``cast``)
    and returns the displacement; the only piece that differs between the full and
    surface-only Okada functions.
    """
    near_vert = torch.cos(dip.detach().to(torch.float64)).abs() < DIP_FD_BAND
    if not bool(near_vert.any()):
        return g_dip_ag
    fd_ev, cast = (model._f64_twin()._evaluate, lambda u: u.to(torch.float64)) \
        if model.internal_dtype != torch.float64 else (ev, lambda u: u)
    gef, gnf, guf = cast(ge), cast(gn), cast(gu)

    def _dip_loss(dip_value):
        o = disp_at_dip(fd_ev, cast, dip_value)
        return (o.e * gef + o.n * gnf + o.u * guf).sum(dim=1)   # [B]

    dip_c = cast(dip)
    H = DIP_FD_RICH_STEP
    _D = lambda st: (_dip_loss(dip_c + st) - _dip_loss(dip_c - st)) / (2.0 * st)
    with torch.no_grad():
        g_fd = ((4.0 * _D(H / 2) - _D(H)) / 3.0).to(dip.dtype)
    return torch.where(near_vert.to(g_fd.device), g_fd, g_dip_ag)


def _make_contract(strain, ge, gn, gu):
    """Bind the per-parameter VJP: sum a parameter's saved per-output strain
    against the grad_outputs. Returns ``contract(key) -> ge*se + gn*sn + gu*su``."""
    def contract(key):
        se, sn, su = strain[key]
        return ge * se + gn * sn + gu * su
    return contract


class _OkadaAnalyticFn(torch.autograd.Function):
    """Analytic-backward implementation of :class:`OkadaSource` (``analytic_grad``).

    The forward runs the exact ``model._evaluate`` and the DC3D strain in one
    pass.  The backward returns the closed-form observation-coordinate strain (and
    translation-invariant source gradients), and autograd of the exact forward for
    dip/geometry/slips -- with a Richardson-extrapolated finite difference
    overwriting ``dip`` only in the thin near-vertical band where its autograd is
    ill-conditioned (see ``DIP_FD_BAND``/``DIP_FD_RICH_STEP``)."""

    @staticmethod
    def forward(ctx, model, x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3):
        with torch.no_grad():
            disp, strain = model._evaluate(
                x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3,
                return_strain=True,
            )
        ctx.model = model
        ctx.z_ndim = z_obs.ndim
        flat = [t for k in ("x_obs", "y_obs", "z_obs") for t in strain[k]]
        ctx.save_for_backward(
            x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
            centroid_depth, length, width, disl1, disl2, disl3, *flat,
        )
        return disp.e, disp.n, disp.u

    @staticmethod
    def backward(ctx, ge, gn, gu):
        s = ctx.saved_tensors
        (x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
         centroid_depth, length, width, disl1, disl2, disl3) = s[:13]
        strain = {k: s[13 + 3 * i: 16 + 3 * i]
                  for i, k in enumerate(("x_obs", "y_obs", "z_obs"))}
        ev = ctx.model._evaluate
        # needs_input_grad aligned to forward inputs:
        # (model, x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
        #  centroid_depth, length, width, disl1, disl2, disl3)
        ng = ctx.needs_input_grad
        contract = _make_contract(strain, ge, gn, gu)

        # Obs-coordinate + source-location grads: cheap closed-form contractions of
        # the saved strain. gx/gy feed both the obs grad and the translation-
        # invariant source grad, so compute them if either end is requested, then
        # drop the obs grad if only the source was asked for.
        gx = contract("x_obs") if (ng[1] or ng[4]) else None   # [B, N]
        gy = contract("y_obs") if (ng[2] or ng[5]) else None
        gz = contract("z_obs") if ng[3] else None
        gsx = -gx.sum(dim=1) if ng[4] else None                # translation invariance
        gsy = -gy.sum(dim=1) if ng[5] else None
        if not ng[1]:
            gx = None
        if not ng[2]:
            gy = None
        if ng[3] and ctx.z_ndim == 1:      # scalar z per image -> reduce over N
            gz = gz.sum(dim=1)

        # dip + geometry + slips: autograd of the EXACT forward. dip rides the
        # same re-evaluation for free; away from the vertical manifold its autograd
        # is exact to ~machine precision (the wide-FD fallback below overwrites only
        # the thin near-vertical band where 1/cos(dip)**2 makes it ill-conditioned).
        # The shared re-evaluation is the dominant backward cost, so skip it whole
        # when none of dip/geometry/slips is requested, and ask autograd only for
        # the leaves that are.
        g_dip = g_strike = g_depth = g_length = g_width = g_d1 = g_d2 = g_d3 = None
        mask = ng[6:14]   # dip, strike, depth, length, width, disl1, disl2, disl3
        if any(mask):
            dip_l = dip.detach().clone().requires_grad_(True)
            geo = [v.detach().clone().requires_grad_(True)
                   for v in (strike, centroid_depth, length, width,
                             disl1, disl2, disl3)]
            leaves = [dip_l, *geo]         # aligned to mask
            with torch.enable_grad():
                out = ev(x_obs, y_obs, z_obs, source_x, source_y, dip_l,
                         geo[0], geo[1], geo[2], geo[3], geo[4], geo[5], geo[6])
                loss = (out.e * ge + out.n * gn + out.u * gu).sum()
            gi = iter(torch.autograd.grad(loss, [lf for lf, m in zip(leaves, mask) if m]))
            g_dip_ag, g_strike, g_depth, g_length, g_width, g_d1, g_d2, g_d3 = \
                [next(gi) if m else None for m in mask]
            # dip: autograd above, with a Richardson FD overwriting only the thin
            # near-vertical band where it is ill-conditioned (see _dip_fd_fallback).
            if ng[6]:
                g_dip = _dip_fd_fallback(
                    ctx.model, ev, dip, g_dip_ag, ge, gn, gu,
                    lambda fd_ev, cast, dip_value: fd_ev(
                        cast(x_obs), cast(y_obs), cast(z_obs), cast(source_x),
                        cast(source_y), dip_value, cast(strike), cast(centroid_depth),
                        cast(length), cast(width), cast(disl1), cast(disl2), cast(disl3)))

        # order: model, x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
        #        centroid_depth, length, width, disl1, disl2, disl3
        return (None, gx, gy, gz, gsx, gsy, g_dip, g_strike, g_depth,
                g_length, g_width, g_d1, g_d2, g_d3)


class OkadaSourceSimple(_OkadaBase):
    """
    Surface-only (``z = 0``) specialisation of :class:`OkadaSource`.

    When the observation depth is exactly zero the real-source and image-source
    UA terms cancel and the depth-dependent UC term drops out, leaving only the
    UB contribution. This class evaluates that reduced form -- numerically
    identical to :class:`OkadaSource` at the surface but cheaper, since it skips
    the real-source and UA/UC kernels.

    Conventions are the same as :class:`OkadaSource`: centroid-based geometry,
    ``(source_x, source_y)`` the map location of the fault centroid, distances in
    metres and dip/strike in radians. Construction parameters are documented on
    :class:`_OkadaBase`.
    """

    def forward(
            self,
            x_obs: Tensor, y_obs: Tensor,
            source_x: Tensor, source_y: Tensor, dip: Tensor, strike: Tensor,
            centroid_depth: Tensor, length: Tensor, width: Tensor,
            disl1: Tensor, disl2: Tensor, disl3: Tensor,
    ) -> Displacement:
        """Surface displacement of a finite rectangular fault (``z = 0``).

        With ``analytic_grad=True`` and gradients required, routes through the
        closed-form analytic backward; otherwise evaluates directly. The forward
        values are identical either way. See :meth:`_evaluate` for full docs.
        """
        args = (x_obs, y_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3)
        if (self.analytic_grad and torch.is_grad_enabled()
                and any(a.requires_grad for a in args)):
            e, n, u = _OkadaSimpleAnalyticFn.apply(self, *args)
            return Displacement(e=e, n=n, u=u)
        return self._evaluate(*args)

    def _evaluate(
            self,
            x_obs: Tensor,  # [B, N]
            y_obs: Tensor,  # [B, N]
            source_x: Tensor,  # [B]
            source_y: Tensor,  # [B]
            dip: Tensor,  # [B] radians
            strike: Tensor,  # [B] radians
            centroid_depth: Tensor,  # [B], meters
            length: Tensor,  # [B], meters
            width: Tensor,  # [B], meters
            disl1: Tensor,  # [B], meters
            disl2: Tensor,  # [B], meters
            disl3: Tensor,  # [B], meters
            return_strain: bool = False,
    ) -> "Displacement | tuple[Displacement, _StrainDict]":
        """Surface displacement of a finite rectangular fault (``z = 0``).

        Same parameters as :meth:`OkadaSource.forward` but without ``z_obs``
        (the observation depth is fixed at the surface).

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            Map position of the fault centroid [B] in metres.
        dip, strike : Tensor
            Fault dip and strike [B] in radians.
        centroid_depth : Tensor
            Depth of the fault centroid [B] in metres (positive down; must be
            > 0, i.e. the fault is buried below the free surface).
        length, width : Tensor
            Fault length (along strike) and width (down dip) [B] in metres
            (both must be > 0).
        disl1, disl2, disl3 : Tensor
            Strike-slip, dip-slip and tensile (opening) dislocations [B] in metres.
        return_strain : bool
            When ``True`` also return the analytic Jacobian used by the
            ``analytic_grad`` backward (see Returns).

        Returns
        -------
        Displacement, or (Displacement, strain)
            ENU surface displacement [B, N] in metres (singular points zeroed).
            With ``return_strain=True`` a ``(Displacement, strain)`` pair, where
            ``strain`` maps each analytically differentiated parameter
            (``"x_obs"``/``"y_obs"``/``"length"``/``"width"``/``"depth"``/
            ``"strike"``) to its ``(d ue, d un, d uu)`` triple.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "dip": dip,
             "strike": strike, "centroid_depth": centroid_depth,
             "length": length, "width": width,
             "disl1": disl1, "disl2": disl2, "disl3": disl3},
        )

        requested_dtype = self.internal_dtype
        # Near-vertical scenes need Okada's exact general-dip formula evaluated in
        # float64 (float32 cannot resolve its 1/cos(dip)**2 terms); _compute_dtype
        # bumps the compute dtype there and we cast back to requested_dtype on
        # return. num_eps/floors follow the *compute* dtype.
        dtype = self._compute_dtype(dip)
        # Numerical guards, scaled to the compute dtype (the float64 values are
        # the validated defaults; float32's ~1e-7 epsilon would swallow them).
        num_eps = self.num_eps if self.num_eps is not None else default_num_eps(dtype)
        geom_eps = GEOM_EPS            # physical "treat as zero" tolerance (metres)
        smooth_eps, rd_eps = _okada_grad_floors(dtype)
        blend_eps = BLEND_EPS

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)

        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)

        dip = dip.to(dtype)
        strike = strike.to(dtype)
        centroid_depth = centroid_depth.to(dtype)
        length = length.to(dtype)
        width = width.to(dtype)

        disl1 = disl1.to(dtype)
        disl2 = disl2.to(dtype)
        disl3 = disl3.to(dtype)

        self._validate_geometry(centroid_depth, length, width, dip)

        al1, al2, aw1, aw2, depth = self._fault_geometry(
            centroid_depth, length, width
        )

        dx = x_obs - source_x[:, None]
        dy = y_obs - source_y[:, None]

        cs = torch.cos(strike)[:, None]
        ss = torch.sin(strike)[:, None]

        x = dx * ss + dy * cs
        y = dx * cs - dy * ss

        c0 = dccon0(
            alpha=torch.as_tensor(
                self.alpha,
                device=x.device,
                dtype=x.dtype,
            ),
            dip_rad=dip,
            internal_dtype=dtype,
            training_safe=self.smooth_grad,
            geom_eps=geom_eps,
        )

        d = depth

        sd_b = c0.sd[:, None]  # [B,1]
        cd_b = c0.cd[:, None]  # [B,1]
        d_b = d[:, None]  # [B,1]

        p = y * cd_b + d_b * sd_b
        q = y * sd_b - d_b * cd_b

        # x, y, p, q are [B, N]

        al1_b = al1[:, None]
        al2_b = al2[:, None]
        aw1_b = aw1[:, None]
        aw2_b = aw2[:, None]

        # Corner grid + dccon2 (Okada Fortran KXI/KET semantics) and the
        # on-dislocation mask.
        xi4, et4, q4, c2 = _dccon2_on_corners(
            x, p, q, al1_b, al2_b, aw1_b, aw2_b, c0,
            dtype=dtype, num_eps=num_eps, geom_eps=geom_eps,
            smooth_eps=smooth_eps, training_safe=self.smooth_grad,
        )
        singular = _singular_mask(xi4, et4, q4, geom_eps)

        ub = ub_displacement(
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2,
            eps=num_eps,
            training_safe=self.smooth_grad,
            smooth_eps=smooth_eps,
            blend_eps=blend_eps,
            rd_eps=rd_eps,
        )

        sign = torch.tensor(
            [[1.0, -1.0],
             [-1.0, 1.0]],
            device=x.device,
            dtype=x.dtype,
        )

        sign = sign[None, :, :, None]

        sd = c0.sd[:, None, None, None]
        cd = c0.cd[:, None, None, None]

        # z = 0 surface displacement: UA cancels, UC drops out
        corner_x = ub.ux
        corner_y = ub.uy * cd - ub.uz * sd
        corner_z = ub.uy * sd + ub.uz * cd

        ux_fault = (sign * corner_x).sum(dim=(1, 2))
        uy_fault = (sign * corner_y).sum(dim=(1, 2))
        uz_fault = (sign * corner_z).sum(dim=(1, 2))

        ue = ux_fault * ss + uy_fault * cs
        un = ux_fault * cs - uy_fault * ss
        uu = uz_fault

        ue = torch.where(singular, torch.zeros_like(ue), ue)
        un = torch.where(singular, torch.zeros_like(un), un)
        uu = torch.where(singular, torch.zeros_like(uu), uu)

        disp = Displacement(e=ue, n=un, u=uu)
        if not return_strain:
            return disp if dtype == requested_dtype else disp.to(requested_dtype)

        # ------------------------------------------------------------------
        # Analytic Jacobian of (ue, un, uu) w.r.t. the inputs that enter the
        # surface field only through (xi, et, q) or the strike rotation:
        # observation coords, source location, length, width, depth, strike.
        # (dip also enters sd/cd explicitly in the kernel, so it is not covered
        # here -- the analytic_grad backward gets it by finite difference.)
        #
        # At z = 0 the UA (real+image) and z*UC contributions cancel for the
        # derivative groups exactly as they do for displacement, so the surface
        # strain comes purely from UB (see DC3D.f90 assembly).
        # ------------------------------------------------------------------
        dub = ub_displacement_and_derivatives(disl1, disl2, disl3, c0, c2)

        def _assemble(triple, w):
            # Fault-frame triple (ux/uy/uz parts) over the corner grid, with the
            # dip rotation + signed corner sum + a per-corner weight w applied.
            a, b, cc = triple
            sw = sign * w
            fx = (sw * a).sum(dim=(1, 2))
            fy = (sw * (b * cd - cc * sd)).sum(dim=(1, 2))
            fz = (sw * (b * sd + cc * cd)).sum(dim=(1, 2))
            return fx, fy, fz

        def _enu(fx, fy, fz):
            return fx * ss + fy * cs, fx * cs - fy * ss, fz

        one = torch.ones((), device=x.device, dtype=x.dtype)
        # per-corner weights d(xi)/dL on the j-axis and d(et)/dW on the k-axis
        wL = x.new_tensor([0.5, -0.5]).reshape(1, 2, 1, 1)
        wW = x.new_tensor([0.5, -0.5]).reshape(1, 1, 2, 1)

        disp_t = (dub.ux, dub.uy, dub.uz)
        xstr_t = (dub.uxx, dub.uyx, dub.uzx)      # d/d(xi)        (= d/dx_station)
        ystr_t = (dub.uxy, dub.uyy, dub.uzy)      # d/dy_station
        # The kernel z-derivative is a rotation of the (et, q) partials:
        #   du_y = cd u_et + sd u_q,   du_z = cd u_q - sd u_et
        # so u_et = cd du_y - sd du_z  and  d/d(depth) = sd u_et - cd u_q = -du_z.
        uet_t = (cd * dub.uxy - sd * dub.uxz,
                 cd * dub.uyy - sd * dub.uyz,
                 cd * dub.uzy - sd * dub.uzz)
        # d/d(depth) triple = -(z-strain)
        zstr_t = (-dub.uxz, -dub.uyz, -dub.uzz)

        # fault-frame displacement and x/y strains (no corner weighting)
        Fx, Fy, Fz = _assemble(disp_t, one)
        SXx, SXy, SXz = _assemble(xstr_t, one)    # dF/dx_station
        SYx, SYy, SYz = _assemble(ystr_t, one)    # dF/dy_station

        # observation coords: x_station = dx ss + dy cs, y_station = dx cs - dy ss
        xobs = _enu(SXx * ss + SYx * cs, SXy * ss + SYy * cs, SXz * ss + SYz * cs)
        yobs = _enu(SXx * cs - SYx * ss, SXy * cs - SYy * ss, SXz * cs - SYz * ss)

        # length (via xi) and width (via et), with per-corner weights
        length_j = _enu(*_assemble(xstr_t, wL))
        width_j = _enu(*_assemble(uet_t, wW))
        # depth: d/d(depth) = d/dz_station  (both enter via d = depth + z)
        depth_j = _enu(*_assemble(zstr_t, one))

        # strike: enters x_station/y_station AND the outer ENU rotation.
        xs = dx * cs - dy * ss          # d(x_station)/d(strike)
        ys = -dx * ss - dy * cs         # d(y_station)/d(strike)
        dFx_s = SXx * xs + SYx * ys
        dFy_s = SXy * xs + SYy * ys
        dFz_s = SXz * xs + SYz * ys
        strike_j = (
            dFx_s * ss + Fx * cs + dFy_s * cs - Fy * ss,
            dFx_s * cs - Fx * ss - dFy_s * ss - Fy * cs,
            dFz_s,
        )

        def _zero_sing(triple):
            z = torch.zeros_like(triple[0])
            return tuple(torch.where(singular, z, g) for g in triple)

        strain = {
            "x_obs": _zero_sing(xobs),     # each: (d ue, d un, d uu) / d(param)
            "y_obs": _zero_sing(yobs),
            "length": _zero_sing(length_j),
            "width": _zero_sing(width_j),
            "depth": _zero_sing(depth_j),
            "strike": _zero_sing(strike_j),
        }
        if dtype != requested_dtype:
            disp = disp.to(requested_dtype)
            strain = {k: tuple(t.to(requested_dtype) for t in v)
                      for k, v in strain.items()}
        return disp, strain


class _OkadaSimpleAnalyticFn(torch.autograd.Function):
    """Analytic-backward implementation of :class:`OkadaSourceSimple`.

    Closed-form Okada surface strain for ``x_obs``/``y_obs``/``source``/``length``/
    ``width``/``centroid_depth``/``strike``; slips and ``dip`` via autograd of the
    exact forward, with a Richardson-extrapolated finite difference overwriting
    ``dip`` only in the thin near-vertical band where its ``1/cos(dip)**2`` terms
    make autograd ill-conditioned (see ``DIP_FD_BAND``/``DIP_FD_RICH_STEP``)."""

    # which inputs get analytic gradients, in the strain-dict keys
    _ANALYTIC = ("x_obs", "y_obs", "length", "width", "depth", "strike")

    @staticmethod
    def forward(ctx, model, x_obs, y_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3):
        with torch.no_grad():
            disp, strain = model._evaluate(
                x_obs, y_obs, source_x, source_y, dip, strike,
                centroid_depth, length, width, disl1, disl2, disl3,
                return_strain=True,
            )
        ctx.model = model
        # flatten the strain dict (6 params x 3 outputs) for save_for_backward
        flat = [t for k in _OkadaSimpleAnalyticFn._ANALYTIC for t in strain[k]]
        ctx.save_for_backward(
            x_obs, y_obs, source_x, source_y, dip, strike, centroid_depth,
            length, width, disl1, disl2, disl3, *flat,
        )
        return disp.e, disp.n, disp.u

    @staticmethod
    def backward(ctx, ge, gn, gu):
        saved = ctx.saved_tensors
        (x_obs, y_obs, source_x, source_y, dip, strike, centroid_depth,
         length, width, disl1, disl2, disl3) = saved[:12]
        keys = _OkadaSimpleAnalyticFn._ANALYTIC
        strain = {k: saved[12 + 3 * i: 15 + 3 * i] for i, k in enumerate(keys)}
        ev = ctx.model._evaluate
        # needs_input_grad aligned to forward inputs:
        # (model, x_obs, y_obs, source_x, source_y, dip, strike, centroid_depth,
        #  length, width, disl1, disl2, disl3)
        ng = ctx.needs_input_grad
        contract = _make_contract(strain, ge, gn, gu)

        # Analytic gradients (cheap closed-form strain contractions). gx/gy feed
        # both the obs grad and the translation-invariant source grad, so compute
        # them if either end is requested, then drop the obs grad if only the source
        # was asked for.
        gx = contract("x_obs") if (ng[1] or ng[3]) else None  # [B, N]
        gy = contract("y_obs") if (ng[2] or ng[4]) else None
        # Translation invariance: field depends on (x_obs - source_x).
        gsx = -gx.sum(dim=1) if ng[3] else None
        gsy = -gy.sum(dim=1) if ng[4] else None
        if not ng[1]:
            gx = None
        if not ng[2]:
            gy = None
        # per-image scalar params: contract then sum over observation points
        g_strike = contract("strike").sum(dim=1) if ng[6] else None
        g_depth = contract("depth").sum(dim=1) if ng[7] else None
        g_length = contract("length").sum(dim=1) if ng[8] else None
        g_width = contract("width").sum(dim=1) if ng[9] else None

        # dip + slips: autograd of the EXACT forward. Slips are linear (exact and
        # stable); dip rides the same re-evaluation for free and is exact to
        # ~machine precision away from the vertical manifold -- 4-6 orders tighter
        # than the wide FD -- so autograd is used there and the FD fallback below
        # overwrites only the thin near-vertical band. This shared re-evaluation is
        # the dominant backward cost, so skip it whole when neither dip nor a slip
        # is requested, and ask autograd only for the leaves that are.
        g_dip = g_d1 = g_d2 = g_d3 = None
        mask = (ng[5], ng[10], ng[11], ng[12])   # dip, disl1, disl2, disl3
        if any(mask):
            dip_l = dip.detach().clone().requires_grad_(True)
            slips = [disl1.detach().clone().requires_grad_(True),
                     disl2.detach().clone().requires_grad_(True),
                     disl3.detach().clone().requires_grad_(True)]
            leaves = [dip_l, *slips]       # aligned to mask
            with torch.enable_grad():
                out = ev(x_obs, y_obs, source_x, source_y, dip_l, strike,
                         centroid_depth, length, width, slips[0], slips[1], slips[2])
                loss = (out.e * ge + out.n * gn + out.u * gu).sum()
            gi = iter(torch.autograd.grad(loss, [lf for lf, m in zip(leaves, mask) if m]))
            g_dip_ag, g_d1, g_d2, g_d3 = [next(gi) if m else None for m in mask]

            # dip: autograd above, with a Richardson FD overwriting only the thin
            # near-vertical band where it is ill-conditioned (see _dip_fd_fallback).
            if ng[5]:
                g_dip = _dip_fd_fallback(
                    ctx.model, ev, dip, g_dip_ag, ge, gn, gu,
                    lambda fd_ev, cast, dip_value: fd_ev(
                        cast(x_obs), cast(y_obs), cast(source_x), cast(source_y),
                        dip_value, cast(strike), cast(centroid_depth), cast(length),
                        cast(width), cast(disl1), cast(disl2), cast(disl3)))

        # order: model, x_obs, y_obs, source_x, source_y, dip, strike,
        #        centroid_depth, length, width, disl1, disl2, disl3
        return (None, gx, gy, gsx, gsy, g_dip, g_strike, g_depth,
                g_length, g_width, g_d1, g_d2, g_d3)


# Geophysical fault parameter names consumed by okada_params_from_fault.
_FAULT_PARAM_KEYS = (
    "strike", "dip", "rake", "slip", "opening", "top_depth", "length", "width",
)


def okada_params_from_fault(params: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert geophysical fault parameters to :class:`OkadaSource` inputs.

    Bridges the constraint-friendly, intuitive parametrisation sampled by an
    :class:`~torchdeform.simulation.OkadaPrior` to the kwargs the Okada source
    models actually consume. This is pure geometry (no ML / normalisation), so it
    is reusable independently of the priors.

    The conversion is::

        strike, dip   <- degrees -> radians
        disl1 = slip * cos(rake)      # strike-slip
        disl2 = slip * sin(rake)      # dip-slip
        disl3 = opening               # tensile
        centroid_depth = top_depth + 0.5 * width * sin(dip)

    Parameters
    ----------
    params : dict[str, Tensor]
        Must contain ``strike``, ``dip``, ``rake`` (degrees), ``slip``,
        ``opening`` (m), ``top_depth``, ``length``, ``width`` (m) -- e.g. the
        output of ``OkadaPrior.sample(...)``. Any other keys (e.g. ``source_x``,
        ``source_y``) are passed through unchanged.

    Returns
    -------
    dict[str, Tensor]
        ``strike`` and ``dip`` in radians, plus ``centroid_depth``, ``length``,
        ``width``, ``disl1``, ``disl2``, ``disl3`` -- ready to splat into
        ``OkadaSource.forward`` / ``OkadaSourceSimple.forward`` alongside the
        observation coordinates and ``source_x`` / ``source_y``.
    """
    strike = torch.deg2rad(params["strike"])
    dip = torch.deg2rad(params["dip"])
    rake = torch.deg2rad(params["rake"])
    slip = params["slip"]
    width = params["width"]

    out: dict[str, Tensor] = {
        "strike": strike,
        "dip": dip,
        "length": params["length"],
        "width": width,
        "centroid_depth": params["top_depth"] + 0.5 * width * torch.sin(dip),
        "disl1": slip * torch.cos(rake),
        "disl2": slip * torch.sin(rake),
        "disl3": params["opening"],
    }
    # Pass through anything that isn't a recognised fault parameter (e.g. location).
    for k, v in params.items():
        if k not in _FAULT_PARAM_KEYS:
            out[k] = v
    return out
