import torch
from dataclasses import dataclass
import math
from torch import Tensor

from .base import SourceModel
from ..core import Displacement

GEOM_EPS = 1e-6   # Okada-style branch / “treat as zero”
NUM_EPS  = 1e-12  # float64 denominator/log/sqrt safety
RD_EPS   = 1e-8   # UB singularity guard for r + d


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
    eps: float = 1e-12,
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
    eps: float = 1e-12,
    training_safe: bool = False,
    smooth_eps: float = 1e-8,
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
        xi = torch.where(torch.abs(xi) < GEOM_EPS, torch.zeros_like(xi), xi)
        et = torch.where(torch.abs(et) < GEOM_EPS, torch.zeros_like(et), et)
        q  = torch.where(torch.abs(q)  < GEOM_EPS, torch.zeros_like(q),  q)

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
class UBDisplacement:
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
    eps: float = 1e-12,
    training_safe: bool = False,
    smooth_eps: float = 1e-8,
    blend_eps: float = 1e-4,
    rd_eps: float = 1e-8,
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
        near_rd = torch.abs(rd) < RD_EPS
        rd_safe = torch.where(
            rd >= 0.0,
            rd + RD_EPS,
            rd - RD_EPS,
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
class UCResult:
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
    *,
    num_eps: float = 1e-12,
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
    num_eps : float
        Numerical guard for denominator safety only.

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
        *,
        num_eps: float = 1e-12
):
    full = uc_displacement_and_derivatives(
        z=z,
        disl1=disl1,
        disl2=disl2,
        disl3=disl3,
        c0=c0,
        c2=c2,
        num_eps=num_eps,
    )
    return UCDisplacement(
        ux=full.ux,
        uy=full.uy,
        uz=full.uz,
    )


class OkadaSource(SourceModel):
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

    Input fault location (fault_x, fault_y) is the map location of the fault centroid.
    """


    def __init__(
        self,
        poisson_ratio: float = 0.25,
        internal_dtype: torch.dtype = torch.float64,
        training_safe: bool = False,
        num_eps: float = NUM_EPS,
        geom_eps: float = GEOM_EPS,
        rd_eps: float = RD_EPS,
        smooth_eps: float = 1e-8,
        blend_eps: float = 1e-4,
    ):
        """
        Okada Fault
        :param poisson_ratio:
        :param internal_dtype:
        :param training_safe:
        :param num_eps:
        :param geom_eps:
        :param rd_eps:
        :param smooth_eps:
        :param blend_eps:
        """
        super().__init__()
        self.alpha = 1.0 / (2.0 * (1.0 - poisson_ratio))
        self.internal_dtype = internal_dtype
        self.training_safe = training_safe
        self.num_eps = num_eps
        self.geom_eps = geom_eps
        self.rd_eps = rd_eps
        self.smooth_eps = smooth_eps
        self.blend_eps = blend_eps

    @staticmethod
    def _fault_geometry(centroid_depth, length, width):
        al1 = -0.5 * length
        al2 = +0.5 * length
        aw1 = -0.5 * width
        aw2 = +0.5 * width
        depth = centroid_depth
        return al1, al2, aw1, aw2, depth

    def forward(
        self,
        x_obs: Tensor,          # [B, N]
        y_obs: Tensor,          # [B, N]
        z_obs: Tensor,          # [B] or [B, N], Okada convention: z <= 0
        fault_x: Tensor,        # [B]
        fault_y: Tensor,        # [B]
        dip: Tensor,            # [B] radians
        strike: Tensor,         # [B] radians
        centroid_depth: Tensor, # [B], meters
        length: Tensor,         # [B], meters
        width: Tensor,          # [B], meters
        disl1: Tensor,               # [B], meters
        disl2: Tensor,               # [B], meters
        disl3: Tensor,               # [B], meters
    ) -> Displacement:
        dtype = self.internal_dtype

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        z_obs = z_obs.to(dtype)

        fault_x = fault_x.to(dtype)
        fault_y = fault_y.to(dtype)

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

        al1, al2, aw1, aw2, depth = self._fault_geometry(
            centroid_depth.to(dtype),
            length.to(dtype),
            width.to(dtype),
        )

        # Observation coordinates relative to centroid reference point
        dx = x_obs - fault_x.to(dtype)[:, None]
        dy = y_obs - fault_y.to(dtype)[:, None]

        ss = torch.sin(strike.to(dtype))
        cs = torch.cos(strike.to(dtype))

        # Same local coordinate convention as your working simplified/native class
        x = dx * ss[:, None] + dy * cs[:, None]
        y = dx * cs[:, None] - dy * ss[:, None]

        c0 = dccon0(
            alpha=torch.as_tensor(self.alpha, device=x.device, dtype=dtype),
            dip_rad=dip.to(dtype),
            internal_dtype=dtype,
            training_safe=self.training_safe,
            geom_eps=self.geom_eps,
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

        xi_r = torch.stack([x - al1_b, x - al2_b], dim=1)   # [B,2,N]
        et_r = torch.stack([p_real - aw1_b, p_real - aw2_b], dim=1)  # [B,2,N]

        # Expand to full 2x2 corner grid
        xi_r4 = xi_r[:, :, None, :].expand(-1, 2, 2, -1)    # [B,2,2,N]
        et_r4 = et_r[:, None, :, :].expand(-1, 2, 2, -1)    # [B,2,2,N]
        q_r4  = q_real[:, None, None, :].expand(-1, 2, 2, -1)

        # Corner masks (KXI / KET) for real source
        xi1 = xi_r4[:, 0, 0, :]
        xi2 = xi_r4[:, 1, 0, :]
        et1 = et_r4[:, 0, 0, :]
        et2 = et_r4[:, 0, 1, :]
        q0  = q_r4[:, 0, 0, :]

        epsg = self.geom_eps

        r12 = torch.sqrt(xi1 * xi1 + et2 * et2 + q0 * q0)
        r21 = torch.sqrt(xi2 * xi2 + et1 * et1 + q0 * q0)
        r22 = torch.sqrt(xi2 * xi2 + et2 * et2 + q0 * q0)

        kxi1 = (xi1 < 0.0) & ((r21 + xi2) < epsg)
        kxi2 = (xi1 < 0.0) & ((r22 + xi2) < epsg)
        ket1 = (et1 < 0.0) & ((r12 + et2) < epsg)
        ket2 = (et1 < 0.0) & ((r22 + et2) < epsg)

        # IMPORTANT: Fortran semantics:
        #   DCCON2(XI(J), ET(K), Q, SD, CD, KXI(K), KET(J))
        kxi_r = torch.stack([kxi1, kxi2], dim=1)[:, None, :, :].expand(-1, 2, 2, -1)
        ket_r = torch.stack([ket1, ket2], dim=1)[:, :, None, :].expand(-1, 2, 2, -1)

        c2_real = dccon2(
            xi=xi_r4,
            et=et_r4,
            q=q_r4,
            sd=c0.sd[:, None, None, None],
            cd=c0.cd[:, None, None, None],
            kxi=kxi_r,
            ket=ket_r,
            internal_dtype=dtype,
            eps=self.num_eps,
            training_safe=self.training_safe,
            smooth_eps=self.smooth_eps,
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

        xi_i = torch.stack([x - al1_b, x - al2_b], dim=1)
        et_i = torch.stack([p_img - aw1_b, p_img - aw2_b], dim=1)

        xi_i4 = xi_i[:, :, None, :].expand(-1, 2, 2, -1)
        et_i4 = et_i[:, None, :, :].expand(-1, 2, 2, -1)
        q_i4  = q_img[:, None, None, :].expand(-1, 2, 2, -1)

        xi1 = xi_i4[:, 0, 0, :]
        xi2 = xi_i4[:, 1, 0, :]
        et1 = et_i4[:, 0, 0, :]
        et2 = et_i4[:, 0, 1, :]
        q0  = q_i4[:, 0, 0, :]

        eps0 = self.geom_eps

        # mimic Okada's zeroing convention
        qz = torch.where(torch.abs(q0) < eps0, torch.zeros_like(q0), q0)
        xi1z = torch.where(torch.abs(xi1) < eps0, torch.zeros_like(xi1), xi1)
        xi2z = torch.where(torch.abs(xi2) < eps0, torch.zeros_like(xi2), xi2)
        et1z = torch.where(torch.abs(et1) < eps0, torch.zeros_like(et1), et1)
        et2z = torch.where(torch.abs(et2) < eps0, torch.zeros_like(et2), et2)

        singular = (
                (qz == 0.0) &
                (
                        (((xi1z * xi2z) <= 0.0) & ((et1z * et2z) == 0.0)) |
                        (((et1z * et2z) <= 0.0) & ((xi1z * xi2z) == 0.0))
                )
        )  # [B, N]

        r12 = torch.sqrt(xi1 * xi1 + et2 * et2 + q0 * q0)
        r21 = torch.sqrt(xi2 * xi2 + et1 * et1 + q0 * q0)
        r22 = torch.sqrt(xi2 * xi2 + et2 * et2 + q0 * q0)

        kxi1 = (xi1 < 0.0) & ((r21 + xi2) < epsg)
        kxi2 = (xi1 < 0.0) & ((r22 + xi2) < epsg)
        ket1 = (et1 < 0.0) & ((r12 + et2) < epsg)
        ket2 = (et1 < 0.0) & ((r22 + et2) < epsg)

        kxi_i = torch.stack([kxi1, kxi2], dim=1)[:, None, :, :].expand(-1, 2, 2, -1)
        ket_i = torch.stack([ket1, ket2], dim=1)[:, :, None, :].expand(-1, 2, 2, -1)

        c2_img = dccon2(
            xi=xi_i4,
            et=et_i4,
            q=q_i4,
            sd=c0.sd[:, None, None, None],
            cd=c0.cd[:, None, None, None],
            kxi=kxi_i,
            ket=ket_i,
            internal_dtype=dtype,
            eps=self.num_eps,
            training_safe=self.training_safe,
            smooth_eps=self.smooth_eps,
        )

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
            eps=self.num_eps,
            training_safe=self.training_safe,
            smooth_eps=self.smooth_eps,
            blend_eps=self.blend_eps,
            rd_eps=self.rd_eps,
        )

        uc_img = uc_displacement_only(
            z=z_b,
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2_img,
            num_eps=self.num_eps,
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

        return Displacement(
            e=ue,
            n=un,
            u=uu,
        )


class OkadaSourceSimple(SourceModel):
    def __init__(
        self,
        poisson_ratio: float = 0.25,
        internal_dtype: torch.dtype = torch.float64,
        training_safe: bool = False,
        num_eps: float = NUM_EPS,
        geom_eps: float = GEOM_EPS,
        rd_eps: float = RD_EPS,
        smooth_eps: float = 1e-8,
        blend_eps: float = 1e-4,
    ):
        """
        Simplification of the Okada Fault for the case Z = 0
        :param poisson_ratio:
        :param internal_dtype:
        :param training_safe:
        :param num_eps:
        :param geom_eps:
        :param rd_eps:
        :param smooth_eps:
        :param blend_eps:
        """
        super().__init__()
        self.alpha = 1.0 / (2.0 * (1.0 - poisson_ratio))
        self.internal_dtype = internal_dtype
        self.training_safe = training_safe
        self.num_eps = num_eps
        self.geom_eps = geom_eps
        self.rd_eps = rd_eps
        self.smooth_eps = smooth_eps
        self.blend_eps = blend_eps

    @staticmethod
    def _fault_geometry(centroid_depth, length, width):
        al1 = -0.5 * length
        al2 = +0.5 * length
        aw1 = -0.5 * width
        aw2 = +0.5 * width
        depth = centroid_depth
        return al1, al2, aw1, aw2, depth

    def forward(
            self,
            x_obs: Tensor,  # [B, N]
            y_obs: Tensor,  # [B, N]
            fault_x: Tensor,  # [B]
            fault_y: Tensor,  # [B]
            dip: Tensor,  # [B] radians
            strike: Tensor,  # [B] radians
            centroid_depth: Tensor,  # [B], meters
            length: Tensor,  # [B], meters
            width: Tensor,  # [B], meters
            disl1: Tensor,  # [B], meters
            disl2: Tensor,  # [B], meters
            disl3: Tensor,  # [B], meters
    ) -> Displacement:
        dtype = self.internal_dtype

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)

        fault_x = fault_x.to(dtype)
        fault_y = fault_y.to(dtype)

        dip = dip.to(dtype)
        strike = strike.to(dtype)
        centroid_depth = centroid_depth.to(dtype)
        length = length.to(dtype)
        width = width.to(dtype)

        disl1 = disl1.to(dtype)
        disl2 = disl2.to(dtype)
        disl3 = disl3.to(dtype)

        al1, al2, aw1, aw2, depth = self._fault_geometry(
            centroid_depth, length, width
        )

        dx = x_obs - fault_x[:, None]
        dy = y_obs - fault_y[:, None]

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
            internal_dtype=self.internal_dtype,
            training_safe=self.training_safe,
            geom_eps=self.geom_eps,
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

        xi = torch.stack([x - al1_b, x - al2_b], dim=1)  # [B,2,N]
        et = torch.stack([p - aw1_b, p - aw2_b], dim=1)  # [B,

        # Build full 2x2 corner grid explicitly
        xi = xi[:, :, None, :].expand(-1, 2, 2, -1)  # [B,2,2,N]
        et = et[:, None, :, :].expand(-1, 2, 2, -1)  # [B,2,2,N]
        q = q[:, None, None, :].expand(-1, 2, 2, -1)  # [B,2,2,N]

        xi1 = xi[:, 0, 0, :]  # [B,N]
        xi2 = xi[:, 1, 0, :]  # [B,N]
        et1 = et[:, 0, 0, :]  # [B,N]
        et2 = et[:, 0, 1, :]  # [B,N]
        q0 = q[:, 0, 0, :]  # [B,N]

        eps0 = self.geom_eps

        # mimic Okada's zeroing convention
        qz = torch.where(torch.abs(q0) < eps0, torch.zeros_like(q0), q0)
        xi1z = torch.where(torch.abs(xi1) < eps0, torch.zeros_like(xi1), xi1)
        xi2z = torch.where(torch.abs(xi2) < eps0, torch.zeros_like(xi2), xi2)
        et1z = torch.where(torch.abs(et1) < eps0, torch.zeros_like(et1), et1)
        et2z = torch.where(torch.abs(et2) < eps0, torch.zeros_like(et2), et2)

        singular = (
                (qz == 0.0) &
                (
                        (((xi1z * xi2z) <= 0.0) & ((et1z * et2z) == 0.0)) |
                        (((et1z * et2z) <= 0.0) & ((xi1z * xi2z) == 0.0))
                )
        )  # [B, N]

        eps = self.geom_eps

        r12 = torch.sqrt(xi1 * xi1 + et2 * et2 + q0 * q0)
        r21 = torch.sqrt(xi2 * xi2 + et1 * et1 + q0 * q0)
        r22 = torch.sqrt(xi2 * xi2 + et2 * et2 + q0 * q0)

        kxi1 = (xi1 < 0.0) & ((r21 + xi2) < eps)
        kxi2 = (xi1 < 0.0) & ((r22 + xi2) < eps)
        ket1 = (et1 < 0.0) & ((r12 + et2) < eps)
        ket2 = (et1 < 0.0) & ((r22 + et2) < eps)

        # Fortran semantics:
        #   DCCON2(XI(J), ET(K), Q, SD, CD, KXI(K), KET(J))

        # kxi varies with K (ET corner), not J
        kxi_base = torch.stack([kxi1, kxi2], dim=1)[:, None, :, :]  # [B,1,2,N]
        kxi = kxi_base.expand(-1, 2, 2, -1)  # [B,2,2,N]

        # ket varies with J (XI corner), not K
        ket_base = torch.stack([ket1, ket2], dim=1)[:, :, None, :]  # [B,2,1,N]
        ket = ket_base.expand(-1, 2, 2, -1)  # [B,2,2,N]

        c2 = dccon2(
            xi=xi,
            et=et,
            q=q,
            sd=c0.sd[:, None, None, None],
            cd=c0.cd[:, None, None, None],
            kxi=kxi,
            ket=ket,
            internal_dtype=self.internal_dtype,
            eps=self.num_eps,
            training_safe=self.training_safe,
            smooth_eps=self.smooth_eps,
        )

        ub = ub_displacement(
            disl1=disl1,
            disl2=disl2,
            disl3=disl3,
            c0=c0,
            c2=c2,
            eps=self.num_eps,
            training_safe=self.training_safe,
            smooth_eps=self.smooth_eps,
            blend_eps=self.blend_eps,
            rd_eps=self.rd_eps,
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

        return Displacement(
            e=ue,
            n=un,
            u=uu,
        )
