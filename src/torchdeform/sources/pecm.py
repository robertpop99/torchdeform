"""
Point Ellipsoidal Cavity Model (pECM) source.

A uniformly **pressurized** point ellipsoidal cavity in an elastic half-space,
after Nikkhoo et al. (2017). Unlike the point CDM
(:class:`~torchdeform.sources.PCDMSource`), which is parameterised directly by
three potencies, the pECM is parameterised *physically* -- by the cavity pressure
``p``, the ellipsoid semi-axes ``a_x/a_y/a_z`` and the elastic moduli -- and
internally converts these into the three pCDM potencies via the Eshelby (1957)
shape tensor. It is therefore the natural forward model for a pressurised magma
chamber of arbitrary (triaxial) ellipsoidal shape.

The pipeline is:

1. Sort the semi-axes and evaluate the Eshelby shape tensor ``S`` (the
   Eshelby/Esh interior tensor), which needs Legendre elliptic integrals,
   computed here with Carlson's (1995) duplication algorithm (``RF``/``RD``).
2. Solve ``(S - I) eT = -p/(3K) 1`` for the transformation strain ``eT`` and form
   the three potencies ``DV = V eT`` (``V`` the cavity volume), restored to the
   original axis order.
3. Feed the potencies to three orthogonal point tensile dislocations -- exactly
   the point-CDM displacement, reusing :func:`pcdm._ptd_disp_surf`.

This is a differentiable, batched PyTorch port of the surface-displacement part
of Nikkhoo's ``pECM.m``. The triaxial / oblate / prolate / spherical branches of
the shape tensor are evaluated and selected with :func:`torch.where`, with every
denominator guarded so the unused branches never produce non-finite values that
could poison gradients.

Attribution
-----------
Ported from the MATLAB ``pECM`` by Mehdi Nikkhoo (MIT licensed). The full MIT
copyright and permission notice is reproduced in the project ``NOTICE`` file.

References:
Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017),
"Compound dislocation models (CDMs) for volcano deformation analyses",
Geophysical Journal International, 208(2), 877-894.
Eshelby, J. D. (1957), Proc. R. Soc. Lond. A, 241, 376-396.
Carlson, B. C. (1995), Numer. Algor., 10, 13-26.
"""
import math

import torch
from torch import Tensor

from .base import SourceModel
from .pcdm import _rotation_matrix, _ptd_disp_surf
from ..core import Displacement

NUM_EPS = 1e-12        # float64 denominator/sqrt safety
AXIS_FLOOR = 1e-12     # minimum semi-axis (avoid zero-volume geometries)
BRANCH_TOL = 1e-6      # relative axis closeness for oblate/prolate/sphere
CARLSON_TOL = 1e-16    # Carlson series truncation tolerance
CARLSON_MAX_ITER = 60


def _carlson_rf(x: Tensor, y: Tensor, z: Tensor, tol: float = CARLSON_TOL) -> Tensor:
    """Carlson's symmetric elliptic integral ``R_F(x, y, z)`` (Carlson 1995).

    Batched and differentiable over flat tensors; ``x, y, z`` must be
    non-negative with at most one zero. Iterates the duplication algorithm,
    freezing each element once its convergence criterion is met.
    """
    xm, ym, zm = x.clone(), y.clone(), z.clone()
    A0 = (x + y + z) / 3.0
    Q = torch.maximum(torch.maximum((A0 - x).abs(), (A0 - y).abs()),
                      (A0 - z).abs()) / (3.0 * tol) ** (1.0 / 6.0)
    Am = A0.clone()
    pow4 = torch.ones_like(A0)
    for _ in range(CARLSON_MAX_ITER):
        active = Am.abs() <= Q / pow4
        if not bool(active.any()):
            break
        lam = (xm * ym).sqrt() + (xm * zm).sqrt() + (ym * zm).sqrt()
        Am = torch.where(active, (Am + lam) / 4.0, Am)
        xm = torch.where(active, (xm + lam) / 4.0, xm)
        ym = torch.where(active, (ym + lam) / 4.0, ym)
        zm = torch.where(active, (zm + lam) / 4.0, zm)
        pow4 = torch.where(active, pow4 * 4.0, pow4)
    X = (A0 - x) / pow4 / Am
    Y = (A0 - y) / pow4 / Am
    Z = -X - Y
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    return (1.0 - E2 / 10.0 + E3 / 14.0 + E2 * E2 / 24.0
            - 3.0 * E2 * E3 / 44.0) / Am.sqrt()


def _carlson_rd(x: Tensor, y: Tensor, z: Tensor, tol: float = CARLSON_TOL) -> Tensor:
    """Carlson's symmetric elliptic integral ``R_D(x, y, z)`` (Carlson 1995).

    Batched and differentiable; ``z`` must be non-zero and at most one of
    ``x, y`` may be zero.
    """
    xm, ym, zm = x.clone(), y.clone(), z.clone()
    A0 = (x + y + 3.0 * z) / 5.0
    Q = torch.maximum(torch.maximum((A0 - x).abs(), (A0 - y).abs()),
                      (A0 - z).abs()) / (tol / 4.0) ** (1.0 / 6.0)
    Am = A0.clone()
    pow4 = torch.ones_like(A0)
    S = torch.zeros_like(A0)
    for _ in range(CARLSON_MAX_ITER):
        active = Am.abs() <= Q / pow4
        if not bool(active.any()):
            break
        lam = (xm * ym).sqrt() + (xm * zm).sqrt() + (ym * zm).sqrt()
        S = torch.where(active, S + (1.0 / pow4) / zm.sqrt() / (zm + lam), S)
        Am = torch.where(active, (Am + lam) / 4.0, Am)
        xm = torch.where(active, (xm + lam) / 4.0, xm)
        ym = torch.where(active, (ym + lam) / 4.0, ym)
        zm = torch.where(active, (zm + lam) / 4.0, zm)
        pow4 = torch.where(active, pow4 * 4.0, pow4)
    X = (A0 - x) / pow4 / Am
    Y = (A0 - y) / pow4 / Am
    Z = -(X + Y) / 3.0
    E2 = X * Y - 6.0 * Z * Z
    E3 = (3.0 * X * Y - 8.0 * Z * Z) * Z
    E4 = 3.0 * (X * Y - Z * Z) * Z * Z
    E5 = X * Y * Z * Z * Z
    return (1.0 - 3.0 * E2 / 14.0 + E3 / 6.0 + 9.0 * E2 * E2 / 88.0
            - 3.0 * E4 / 22.0 - 9.0 * E2 * E3 / 52.0
            + 3.0 * E5 / 26.0) / pow4 / Am ** 1.5 + 3.0 * S


def _shape_tensor_ecm(a1: Tensor, a2: Tensor, a3: Tensor, nu: float, eps: float):
    """Eshelby (1957) interior shape-tensor block ``S_iijj`` for a sorted ellipsoid.

    ``a1 >= a2 >= a3 > 0`` (semi-axes). Returns the nine ``S_iijj`` components
    (``S1111, S1122, S1133, S2211, S2222, S2233, S3311, S3322, S3333``) as a
    ``[B, 3, 3]`` matrix. Triaxial / oblate / prolate / spherical cases are
    evaluated with guarded denominators and selected with :func:`torch.where`.
    """
    a1s, a2s, a3s = a1 * a1, a2 * a2, a3 * a3
    d12 = (a1s - a2s).clamp_min(eps)   # guarded squared-axis differences
    d23 = (a2s - a3s).clamp_min(eps)
    d13 = (a1s - a3s).clamp_min(eps)

    four_pi = 4.0 * math.pi

    # --- triaxial ( a1 > a2 > a3 ) --------------------------------------- #
    sin_theta = torch.sqrt((1.0 - a3s / a1s).clamp_min(eps))
    c = 1.0 / (sin_theta * sin_theta)
    k2 = (d12 / d13).clamp(0.0, 1.0)
    F = _carlson_rf((c - 1.0).clamp_min(eps), (c - k2).clamp_min(eps), c)
    E = F - k2 / 3.0 * _carlson_rd((c - 1.0).clamp_min(eps), (c - k2).clamp_min(eps), c)
    sqrt13 = torch.sqrt(d13)
    abc = a1 * a2 * a3
    I1_t = four_pi * abc / d12 / sqrt13 * (F - E)
    I3_t = four_pi * abc / d23 / sqrt13 * (a2 * sqrt13 / (a1 * a3) - E)
    I2_t = four_pi - I1_t - I3_t
    I12_t = (I2_t - I1_t) / d12
    I13_t = (I3_t - I1_t) / d13
    I11_t = (four_pi / a1s - I12_t - I13_t) / 3.0
    I23_t = (I3_t - I2_t) / d23
    I22_t = (four_pi / a2s - I23_t - I12_t) / 3.0
    I33_t = (four_pi / a3s - I13_t - I23_t) / 3.0

    # relative closeness -> branch membership. Computed here (rather than just
    # before ``pick`` below) because the oblate/prolate branches need ``sphere``
    # to keep the discarded spherical result's gradient finite.
    rel12 = (a1 - a2) / a1
    rel23 = (a2 - a3) / a1
    oblate = rel12 < BRANCH_TOL
    prolate = rel23 < BRANCH_TOL
    sphere = oblate & prolate

    # --- oblate ( a1 = a2 > a3 ) ----------------------------------------- #
    # For an exact sphere rat -> 1, where acos'(1) and sqrt'(0) are infinite and
    # the oblate result is discarded (overridden by the sphere closed form). Feed
    # a safe rat there so the dead branch carries no NaN into the backward pass.
    rat = (a3 / a1).clamp(-1.0, 1.0)
    rat = torch.where(sphere, torch.full_like(rat, 0.5), rat)
    I1_o = 2.0 * math.pi * abc / d13 ** 1.5 * (
        torch.acos(rat) - rat * torch.sqrt((1.0 - rat * rat).clamp_min(0.0)))
    I3_o = four_pi - 2.0 * I1_o
    I13_o = (I3_o - I1_o) / d13
    I11_o = math.pi / a1s - I13_o / 4.0
    I23_o = I13_o
    I22_o = math.pi / a2s - I23_o / 4.0
    I33_o = (four_pi / a3s - 2.0 * I13_o) / 3.0
    I12_o = I11_o

    # --- prolate ( a1 > a2 = a3 ) ---------------------------------------- #
    # Same guard as the oblate branch: for a sphere ar -> 1 (acosh'(1), sqrt'(0)
    # infinite) and this result is discarded.
    ar = (a1 / a3).clamp_min(1.0)
    ar = torch.where(sphere, torch.full_like(ar, 2.0), ar)
    I2_p = 2.0 * math.pi * abc / d13 ** 1.5 * (
        ar * torch.sqrt((ar * ar - 1.0).clamp_min(0.0)) - torch.acosh(ar))
    I1_p = four_pi - 2.0 * I2_p
    I12_p = (I2_p - I1_p) / d12
    I11_p = (four_pi / a1s - 2.0 * I12_p) / 3.0
    I22_p = math.pi / a2s - I12_p / 4.0
    I23_p = I22_p
    I13_p = I12_p
    I33_p = (four_pi / a3s - I13_p - I23_p) / 3.0

    def pick(t, o, p):
        return torch.where(oblate, o, torch.where(prolate, p, t))

    I1 = pick(I1_t, I1_o, I1_p)
    I2 = pick(I2_t, I1_o, I2_p)   # oblate: I2 = I1
    I3 = pick(I3_t, I3_o, I2_p)   # prolate: I3 = I2
    I11 = pick(I11_t, I11_o, I11_p)
    I12 = pick(I12_t, I12_o, I12_p)
    I13 = pick(I13_t, I13_o, I13_p)
    I22 = pick(I22_t, I22_o, I22_p)
    I23 = pick(I23_t, I23_o, I23_p)
    I33 = pick(I33_t, I33_o, I33_p)
    # symmetric partners
    I21, I31, I32 = I12, I13, I23

    f1 = 3.0 / (8.0 * math.pi * (1.0 - nu))
    f2 = (1.0 - 2.0 * nu) / (8.0 * math.pi * (1.0 - nu))
    g = 1.0 / (8.0 * math.pi * (1.0 - nu))
    S1111 = f1 * a1s * I11 + f2 * I1
    S1122 = g * a2s * I12 - f2 * I1
    S1133 = g * a3s * I13 - f2 * I1
    S2211 = g * a1s * I21 - f2 * I2
    S2222 = f1 * a2s * I22 + f2 * I2
    S2233 = g * a3s * I23 - f2 * I2
    S3311 = g * a1s * I31 - f2 * I3
    S3322 = g * a2s * I32 - f2 * I3
    S3333 = f1 * a3s * I33 + f2 * I3

    # sphere: closed form, overrides the (guarded, unused) general result
    sph_diag = (7.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
    sph_off = (5.0 * nu - 1.0) / (15.0 * (1.0 - nu))
    S1111 = torch.where(sphere, torch.full_like(S1111, sph_diag), S1111)
    S2222 = torch.where(sphere, torch.full_like(S2222, sph_diag), S2222)
    S3333 = torch.where(sphere, torch.full_like(S3333, sph_diag), S3333)
    S1122 = torch.where(sphere, torch.full_like(S1122, sph_off), S1122)
    S1133 = torch.where(sphere, torch.full_like(S1133, sph_off), S1133)
    S2211 = torch.where(sphere, torch.full_like(S2211, sph_off), S2211)
    S2233 = torch.where(sphere, torch.full_like(S2233, sph_off), S2233)
    S3311 = torch.where(sphere, torch.full_like(S3311, sph_off), S3311)
    S3322 = torch.where(sphere, torch.full_like(S3322, sph_off), S3322)

    row0 = torch.stack([S1111, S1122, S1133], dim=-1)
    row1 = torch.stack([S2211, S2222, S2233], dim=-1)
    row2 = torch.stack([S3311, S3322, S3333], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)   # [B, 3, 3]


def ecm_potencies(
    a_x: Tensor, a_y: Tensor, a_z: Tensor, pressure: Tensor,
    nu: float, bulk_K: float, eps: float = NUM_EPS,
):
    """Three pCDM potencies of a uniformly pressurised point ellipsoidal cavity.

    Implements the Eshelby step of ``pECM``: it sorts the semi-axes, builds the
    shape tensor, solves for the transformation strain ``eT`` and returns the
    potencies ``(DVx, DVy, DVz)`` (m^3) in the *original* X/Y/Z axis order,
    ready to drive three orthogonal point tensile dislocations.

    Parameters
    ----------
    a_x, a_y, a_z : Tensor
        Semi-axes ``[B]`` (m) along X/Y/Z before rotation.
    pressure : Tensor
        Cavity-wall pressure ``[B]`` (same units as the moduli, Pa).
    nu : float
        Poisson's ratio.
    bulk_K : float
        Bulk modulus ``K`` (Pa).
    eps : float
        Numerical guard.

    Returns
    -------
    Tensor
        Potencies ``[B, 3]`` = ``(DVx, DVy, DVz)``.
    """
    a = torch.stack([a_x, a_y, a_z], dim=-1).clamp_min(AXIS_FLOOR)   # [B, 3]
    ai, idx = torch.sort(a, dim=-1, descending=True)
    a1, a2, a3 = ai[:, 0], ai[:, 1], ai[:, 2]

    Sm = _shape_tensor_ecm(a1, a2, a3, nu, eps)        # [B, 3, 3]
    eye = torch.eye(3, dtype=Sm.dtype, device=Sm.device)
    Sm = Sm - eye                                       # (S - I)

    rhs = (pressure / (3.0 * bulk_K)).unsqueeze(-1).expand(-1, 3)   # [B, 3]
    eT = -torch.linalg.solve(Sm, rhs)                   # [B, 3], sorted-axis frame
    # uniformly-pressurised cavity: every eT shares the sign of p (else clamp)
    same_sign = torch.sign(eT) == torch.sign(pressure).unsqueeze(-1)
    eT = torch.where(same_sign, eT, torch.zeros_like(eT))

    V = 4.0 / 3.0 * math.pi * a_x * a_y * a_z           # [B] volume
    pot_sorted = V.unsqueeze(-1) * eT                   # [B, 3] sorted order
    DV = torch.zeros_like(pot_sorted).scatter(1, idx, pot_sorted)   # original order
    return DV


class PECMSource(SourceModel):
    """
    Point Ellipsoidal Cavity Model: a uniformly pressurised point ellipsoid.

    A physically parameterised volcanic point source: given the cavity pressure
    and the (triaxial) ellipsoid semi-axes ``a_x/a_y/a_z``, it computes the
    equivalent point-CDM potencies through the Eshelby (1957) shape tensor and
    sums three orthogonal point tensile dislocations. A spherical cavity reduces
    to a Mogi-like centre of dilatation; oblate/prolate cavities give sill/dyke
    styles.

    Conventions
    -----------
    - All distances in metres; depth positive downward. The source sits at
      ``(source_x, source_y, -depth)``.
    - Rotation angles ``omega_x/y/z`` in radians (clockwise about X/Y/Z),
      consistent with :class:`~torchdeform.sources.PCDMSource`.
    - ``pressure`` in Pa (same units as the elastic moduli); positive pressure
      inflates and produces uplift.
    - Material is set on the model: ``poisson_ratio`` and ``shear_modulus`` (mu,
      Pa); the bulk modulus is ``K = 2 mu (1 + nu) / (3 (1 - 2 nu))``.
    - Returns ENU surface displacement in metres. Displacement scales linearly
      with ``pressure`` and with ``1 / shear_modulus``.
    """

    def __init__(
        self,
        poisson_ratio: float = 0.25,
        shear_modulus: float = 3.0e10,
        internal_dtype: torch.dtype = torch.float64,
        num_eps: float | None = None,
    ):
        """
        Parameters
        ----------
        poisson_ratio : float, default 0.25
            Poisson's ratio of the elastic half-space.
        shear_modulus : float, default 3e10
            Shear modulus ``mu`` (Pa). Sets the absolute displacement scale.
        internal_dtype : torch.dtype, default torch.float64
            Dtype used for the internal computation; inputs are cast to it.
        num_eps : float or None, default None
            Numerical guard for denominators / sqrt. ``None`` picks a floor
            matched to ``internal_dtype`` (``1e-12`` for float64 underflows
            float32); pass a float to override.
        """
        super().__init__()
        self.v = poisson_ratio
        self.mu = shear_modulus
        # K = lambda + 2 mu / 3, with lambda = 2 mu nu / (1 - 2 nu)
        self.K = 2.0 * shear_modulus * (1.0 + poisson_ratio) / (
            3.0 * (1.0 - 2.0 * poisson_ratio))
        self.internal_dtype = internal_dtype
        self.num_eps = num_eps

    def forward(
        self,
        x_obs: Tensor,      # [B, N] east coordinates (m)
        y_obs: Tensor,      # [B, N] north coordinates (m)
        source_x: Tensor,   # [B] source east (m)
        source_y: Tensor,   # [B] source north (m)
        depth: Tensor,      # [B] depth, positive down (m)
        omega_x: Tensor,    # [B] rotation about X (rad)
        omega_y: Tensor,    # [B] rotation about Y (rad)
        omega_z: Tensor,    # [B] rotation about Z (rad)
        a_x: Tensor,        # [B] semi-axis along X before rotation (m)
        a_y: Tensor,        # [B] semi-axis along Y before rotation (m)
        a_z: Tensor,        # [B] semi-axis along Z before rotation (m)
        pressure: Tensor,   # [B] cavity-wall pressure (Pa)
    ) -> Displacement:
        """Surface displacement from a pressurised point ellipsoidal cavity.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the source [B] in metres.
        depth : Tensor
            Source depth [B] in metres (positive down).
        omega_x, omega_y, omega_z : Tensor
            Clockwise rotation angles about the X/Y/Z axes [B] in radians.
        a_x, a_y, a_z : Tensor
            Ellipsoid semi-axes [B] (m) along X/Y/Z before rotation.
        pressure : Tensor
            Cavity-wall pressure [B] (Pa); positive inflates.

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "depth": depth,
             "omega_x": omega_x, "omega_y": omega_y, "omega_z": omega_z,
             "a_x": a_x, "a_y": a_y, "a_z": a_z, "pressure": pressure},
        )

        dtype = self.internal_dtype
        num_eps = self._resolve_num_eps()
        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)
        depth = depth.to(dtype)
        omega_x = omega_x.to(dtype)
        omega_y = omega_y.to(dtype)
        omega_z = omega_z.to(dtype)
        a_x = a_x.to(dtype)
        a_y = a_y.to(dtype)
        a_z = a_z.to(dtype)
        pressure = pressure.to(dtype)

        DV = ecm_potencies(a_x, a_y, a_z, pressure, self.v, self.K, num_eps)

        rot = _rotation_matrix(omega_x, omega_y, omega_z)   # [B, 3, 3]
        dx = x_obs - source_x[:, None]
        dy = y_obs - source_y[:, None]
        depth_b = depth[:, None]

        ue = torch.zeros_like(dx)
        un = torch.zeros_like(dx)
        uv = torch.zeros_like(dx)
        for k in range(3):
            nx = rot[:, 0, k][:, None]
            ny = rot[:, 1, k][:, None]
            nz = rot[:, 2, k][:, None]
            e, n, u = _ptd_disp_surf(dx, dy, depth_b, nx, ny, nz,
                                     DV[:, k][:, None], self.v, self._resolve_num_eps())
            ue = ue + e
            un = un + n
            uv = uv + u

        return Displacement(e=ue, n=un, u=uv)
