"""
Compound Dislocation Model (CDM) source.

A *finite* volcanic source built from three mutually orthogonal **rectangular**
tensile dislocations in an elastic half-space, after Nikkhoo et al. (2017). It is
the near-field-accurate counterpart of the point CDM
(:class:`~torchdeform.sources.PCDMSource`): each rectangle has finite semi-axes
``a_x/a_y/a_z`` and a common ``opening``, so in the far field the CDM converges to
a pCDM with potencies ``4*a_y*a_z*opening`` etc., while close to the source it
captures the finite extent that a point source cannot.

Each rectangle's surface field is assembled from the half-space *angular
dislocation* solution: every side of the rectangle contributes an angular
dislocation pair (``AngSetupFSC``), evaluated in an artefact-free configuration
chosen per observation point.

This is a differentiable, batched PyTorch port of the surface-displacement part
of Nikkhoo's ``CDM.m``. The per-point configuration switch is expressed as a
single :func:`torch.where` on the angular-dislocation angle (both endpoints of a
side share the same switch), and exactly-vertical sides -- which occur for an
axis-aligned box, i.e. ``omega_* = 0`` -- are masked to zero exactly as in the
reference, with a safe angle substituted first so gradients stay finite.

Attribution
-----------
Ported from the MATLAB ``CDM`` by Mehdi Nikkhoo (MIT licensed). The full MIT
copyright and permission notice is reproduced in the project ``NOTICE`` file.

Reference: Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017),
"Compound dislocation models (CDMs) for volcano deformation analyses",
Geophysical Journal International, 208(2), 877-894.
"""
import math

import torch
from torch import Tensor

from .base import SourceModel, DEFAULT_POISSON_RATIO
from .pcdm import _rotation_matrix
from ..core import Displacement

# |sin(beta)| below this => side is treated as vertical (its contribution is
# masked to zero, the exact limit). Set well above the machine-eps used by the
# reference: the half-space angular-dislocation surface formula suffers
# catastrophic cancellation for *near*-vertical sides (spurious O(1)..O(1e13)
# spikes for sin(beta) ~ 1e-9..1e-4 in float64), and a side within this angle of
# vertical contributes negligibly, so folding it onto the vertical limit is both
# accurate and numerically safe.
DEGENERATE_SIN = 1e-4


def _ang_dis_disp_surf(
    y1: Tensor, y2: Tensor, beta: Tensor,
    b1: Tensor, b2: Tensor, b3: Tensor, nu: float, a: Tensor, eps: float,
):
    """Surface displacement of one angular dislocation in a half-space.

    Direct translation of Nikkhoo's ``AngDisDispSurf``. All tensors broadcast
    over ``[B, N]``; ``beta`` is the angular-dislocation angle (may be negative),
    ``b1/b2/b3`` the slip components in the angular-dislocation frame and ``a``
    the (positive) depth of the dislocation vertex.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(v1, v2, v3)`` displacement components in the angular-dislocation frame.
    """
    sinB = torch.sin(beta)
    cosB = torch.cos(beta)
    sinB_safe = torch.where(sinB.abs() < eps, torch.full_like(sinB, eps), sinB)
    cotB = cosB / sinB_safe

    z1 = y1 * cosB + a * sinB
    z3 = y1 * sinB - a * cosB
    r2 = y1 * y1 + y2 * y2 + a * a + eps
    r = torch.sqrt(r2)
    r3 = r2 * r
    r5 = r3 * r2

    # The Burgers function Fi = 2*atan2(y2, (r+a)*cot(beta/2) - y1)
    half = beta / 2.0
    sin_half = torch.sin(half)
    sin_half = torch.where(sin_half.abs() < eps, torch.full_like(sin_half, eps), sin_half)
    cot_half = torch.cos(half) / sin_half
    Fi = 2.0 * torch.atan2(y2, (r + a) * cot_half - y1)

    inv = 1.0 - 2.0 * nu
    rpa = r + a
    rmz = r - z3
    cotB2 = cotB * cotB

    v1b1 = b1 / 2.0 / math.pi * (
        (1.0 - inv * cotB2) * Fi
        + y2 / rpa * (inv * (cotB + y1 / 2.0 / rpa) - y1 / r)
        - y2 * (r * sinB - y1) * cosB / r / rmz
    )
    v2b1 = b1 / 2.0 / math.pi * (
        inv * ((0.5 + cotB2) * torch.log(rpa) - cotB / sinB_safe * torch.log(rmz))
        - 1.0 / rpa * (inv * (y1 * cotB - a / 2.0 - y2 * y2 / 2.0 / rpa) + y2 * y2 / r)
        + y2 * y2 * cosB / r / rmz
    )
    v3b1 = b1 / 2.0 / math.pi * (
        inv * Fi * cotB
        + y2 / rpa * (2.0 * nu + a / r)
        - y2 * cosB / rmz * (cosB + a / r)
    )

    v1b2 = b2 / 2.0 / math.pi * (
        -inv * ((0.5 - cotB2) * torch.log(rpa) + cotB2 * cosB * torch.log(rmz))
        - 1.0 / rpa * (inv * (y1 * cotB + 0.5 * a + y1 * y1 / 2.0 / rpa) - y1 * y1 / r)
        + z1 * (r * sinB - y1) / r / rmz
    )
    v2b2 = b2 / 2.0 / math.pi * (
        (1.0 + inv * cotB2) * Fi
        - y2 / rpa * (inv * (cotB + y1 / 2.0 / rpa) - y1 / r)
        - y2 * z1 / r / rmz
    )
    v3b2 = b2 / 2.0 / math.pi * (
        -inv * cotB * (torch.log(rpa) - cosB * torch.log(rmz))
        - y1 / rpa * (2.0 * nu + a / r)
        + z1 / rmz * (cosB + a / r)
    )

    v1b3 = b3 / 2.0 / math.pi * (y2 * (r * sinB - y1) * sinB / r / rmz)
    v2b3 = b3 / 2.0 / math.pi * (-y2 * y2 * sinB / r / rmz)
    v3b3 = b3 / 2.0 / math.pi * (Fi + y2 * (r * cosB + a) * sinB / r / rmz)

    v1 = v1b1 + v1b2 + v1b3
    v2 = v2b1 + v2b2 + v2b3
    v3 = v3b1 + v3b2 + v3b3
    return v1, v2, v3


def _coord_trans(vec: Tensor, A: Tensor) -> Tensor:
    """Apply ``A @ vec`` per batch item. ``A`` is ``[B, 3, 3]``, ``vec`` ``[B, ..., 3]``."""
    return torch.einsum("bij,b...j->b...i", A, vec)


def _ang_setup_fsc(
    X: Tensor, Y: Tensor,
    bX: Tensor, bY: Tensor, bZ: Tensor,
    PA: Tensor, PB: Tensor, nu: float, eps: float,
):
    """Displacement of the angular-dislocation pair on one side ``PA->PB`` of an RD.

    Translation of Nikkhoo's ``AngSetupFSC``. ``X, Y`` are EFCS observation
    coordinates ``[B, N]``; ``bX/bY/bZ`` the slip vector ``[B, 1]``; ``PA, PB``
    the side endpoints ``[B, 3]``. Vertical sides (``|sin(beta)| ~ 0``) contribute
    zero, matching the reference; a safe angle is substituted there so gradients
    stay finite.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(ue, un, uv)`` EFCS displacement ``[B, N]``.
    """
    side = PB - PA                                   # [B, 3]
    length = torch.linalg.norm(side, dim=-1, keepdim=True).clamp_min(eps)  # [B, 1]
    hlen = torch.sqrt(side[:, 0] ** 2 + side[:, 1] ** 2 + eps * eps)  # [B] horiz length
    sin_beta = (hlen / length[:, 0])                 # [B]
    degenerate = sin_beta <= DEGENERATE_SIN          # [B] (vertical side)

    # Both ``acos`` (slope -inf at +/-1) and ``hlen = sqrt(.)`` (slope +inf at 0)
    # are singular for an exactly vertical side. The output of such a side is
    # masked to zero, but ``torch.where`` still routes a 0 cotangent into the
    # unused branch and ``0 * inf = NaN`` would poison every geometry gradient.
    # Substitute a safe argument *before* the singular op (acos(0) = pi/2, and the
    # +eps*eps inside the sqrt) so the masked branch carries no NaN backward.
    cos_arg = (-side[:, 2] / length[:, 0]).clamp(-1.0, 1.0)
    cos_arg = torch.where(degenerate, torch.zeros_like(cos_arg), cos_arg)
    beta = torch.acos(cos_arg)                       # [B] in (0, pi); pi/2 if degenerate

    # ADCS basis (columns of A), expressed in EFCS
    hlen_safe = hlen.clamp_min(eps)
    ey1 = torch.stack([side[:, 0] / hlen_safe, side[:, 1] / hlen_safe,
                       torch.zeros_like(hlen_safe)], dim=-1)   # [B, 3]
    # ey3 = (0, 0, -1); ey2 = cross(ey3, ey1) = (ey1_y, -ey1_x, 0)
    ey2 = torch.stack([ey1[:, 1], -ey1[:, 0], torch.zeros_like(hlen_safe)], dim=-1)
    ey3 = torch.zeros_like(ey1)
    ey3[:, 2] = -1.0
    A = torch.stack([ey1, ey2, ey3], dim=-1)         # [B, 3, 3], columns ey1,ey2,ey3

    # observation offsets from PA, in EFCS, transformed to ADCS
    off = torch.stack([X - PA[:, 0:1], Y - PA[:, 1:2],
                       (-PA[:, 2:3]).expand_as(X)], dim=-1)    # [B, N, 3]
    yA = _coord_trans(off, A)                        # [B, N, 3]
    y1A, y2A = yA[..., 0], yA[..., 1]

    yAB = _coord_trans(side.unsqueeze(1), A)[:, 0, :]  # [B, 3]
    y1B = y1A - yAB[:, 0:1]
    y2B = y2A - yAB[:, 1:2]

    slip = torch.stack([bX[:, 0], bY[:, 0], bZ[:, 0]], dim=-1)  # [B, 3]
    b = _coord_trans(slip.unsqueeze(1), A)[:, 0, :]   # [B, 3]
    b1 = b[:, 0:1]
    b2 = b[:, 1:2]
    b3 = b[:, 2:3]

    beta_col = beta[:, None]                          # [B, 1]
    # artefact-free configuration: I where (beta*y1A) >= 0 uses -pi+beta, else beta
    I = (beta_col * y1A) >= 0
    beta_eff = torch.where(I, -math.pi + beta_col, beta_col)  # [B, N]

    aA = -PA[:, 2:3]                                  # [B, 1] vertex depth (>0)
    aB = -PB[:, 2:3]
    v1A, v2A, v3A = _ang_dis_disp_surf(y1A, y2A, beta_eff, b1, b2, b3, nu, aA, eps)
    v1B, v2B, v3B = _ang_dis_disp_surf(y1B, y2B, beta_eff, b1, b2, b3, nu, aB, eps)

    v = torch.stack([v1B - v1A, v2B - v2A, v3B - v3A], dim=-1)  # [B, N, 3]
    # transform back to EFCS via A' (transpose)
    u = torch.einsum("bji,bnj->bni", A, v)            # [B, N, 3]
    ue, un, uv = u[..., 0], u[..., 1], u[..., 2]

    mask = (~degenerate)[:, None]
    z = torch.zeros_like(ue)
    return (torch.where(mask, ue, z),
            torch.where(mask, un, z),
            torch.where(mask, uv, z))


def _rd_disp_surf(X, Y, V1, V2, V3, V4, opening, nu, eps):
    """Surface displacement of one rectangular dislocation (RD).

    Sums the angular-dislocation contributions of the four sides ``V1V2``,
    ``V2V3``, ``V3V4``, ``V4V1``. ``V*`` are the rectangle vertices ``[B, 3]`` and
    ``opening`` ``[B, 1]`` the tensile component.
    """
    normal = torch.linalg.cross(V2 - V1, V4 - V1, dim=-1)            # [B, 3]
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True).clamp_min(eps)
    bX = (opening * normal[:, 0:1])
    bY = (opening * normal[:, 1:2])
    bZ = (opening * normal[:, 2:3])

    ue = torch.zeros_like(X)
    un = torch.zeros_like(X)
    uv = torch.zeros_like(X)
    for PA, PB in ((V1, V2), (V2, V3), (V3, V4), (V4, V1)):
        e, n, u = _ang_setup_fsc(X, Y, bX, bY, bZ, PA, PB, nu, eps)
        ue = ue + e
        un = un + n
        uv = uv + u
    return ue, un, uv


class CDMSource(SourceModel):
    """
    Compound Dislocation Model: a finite triaxial volcanic source.

    Sums three mutually orthogonal **rectangular** tensile dislocations sharing a
    common centroid, orientation (``omega_x/y/z``) and ``opening``. The
    rectangles have semi-axes ``a_x/a_y/a_z`` along the X/Y/Z axes before
    rotation, and lie in the Y-Z, X-Z and X-Y planes respectively. Equal
    semi-axes give a near-isotropic (Mogi-like) inflation; unequal semi-axes give
    dyke/sill/ellipsoidal styles. In the far field the CDM matches a
    :class:`~torchdeform.sources.PCDMSource` with potencies
    ``dv_x = 4*a_y*a_z*opening`` etc.

    Conventions
    -----------
    - All distances in metres; depth positive downward. The centroid is at
      ``(source_x, source_y, -depth)``.
    - Rotation angles ``omega_x/y/z`` in radians (clockwise about X/Y/Z),
      consistent with :class:`~torchdeform.sources.PCDMSource`.
    - Semi-axes ``a_x/a_y/a_z`` and ``opening`` in metres; semi-axes should be
      positive. The model assumes the source stays below the free surface (the
      shallowest rectangle corner above the centroid must not breach the
      surface); very shallow/large geometries are unphysical for a half-space.
    - Returns ENU surface displacement in metres.
    """

    def __init__(
        self,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        internal_dtype: torch.dtype = torch.float64,
        num_eps: float | None = None,
    ):
        """
        Parameters
        ----------
        poisson_ratio : float, default 0.25
            Poisson's ratio of the elastic half-space.
        internal_dtype : torch.dtype, default torch.float64
            Dtype used for the internal computation; inputs are cast to it.
        num_eps : float or None, default None
            Numerical guard for denominators / sqrt. ``None`` picks a floor
            matched to ``internal_dtype`` (``1e-12`` for float64 underflows
            float32); pass a float to override.
        """
        super().__init__()
        self.v = poisson_ratio
        self.internal_dtype = internal_dtype
        self.num_eps = num_eps

    def forward(
        self,
        x_obs: Tensor,      # [B, N] east coordinates (m)
        y_obs: Tensor,      # [B, N] north coordinates (m)
        source_x: Tensor,   # [B] source east (m)
        source_y: Tensor,   # [B] source north (m)
        depth: Tensor,      # [B] centroid depth, positive down (m)
        omega_x: Tensor,    # [B] rotation about X (rad)
        omega_y: Tensor,    # [B] rotation about Y (rad)
        omega_z: Tensor,    # [B] rotation about Z (rad)
        a_x: Tensor,        # [B] semi-axis along X before rotation (m)
        a_y: Tensor,        # [B] semi-axis along Y before rotation (m)
        a_z: Tensor,        # [B] semi-axis along Z before rotation (m)
        opening: Tensor,    # [B] tensile opening of the rectangles (m)
    ) -> Displacement:
        """Surface displacement from a (finite) compound dislocation model.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the centroid [B] in metres.
        depth : Tensor
            Centroid depth [B] in metres (positive down).
        omega_x, omega_y, omega_z : Tensor
            Clockwise rotation angles about the X/Y/Z axes [B] in radians.
        a_x, a_y, a_z : Tensor
            Semi-axes [B] (m) of the model along X/Y/Z before rotation.
        opening : Tensor
            Tensile opening [B] (m) shared by the three rectangles.

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "depth": depth,
             "omega_x": omega_x, "omega_y": omega_y, "omega_z": omega_z,
             "a_x": a_x, "a_y": a_y, "a_z": a_z, "opening": opening},
        )
        if torch.any(depth <= 0):
            raise ValueError("depth must be strictly positive")
        for name, val in (("a_x", a_x), ("a_y", a_y), ("a_z", a_z)):
            if torch.any(val <= 0):
                raise ValueError(f"{name} must be strictly positive")

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
        opening = opening.to(dtype)

        rot = _rotation_matrix(omega_x, omega_y, omega_z)   # [B, 3, 3]
        c1 = rot[:, :, 0]    # [B, 3]
        c2 = rot[:, :, 1]
        c3 = rot[:, :, 2]

        # full axes (the reference doubles the semi-axes internally)
        ax = (2.0 * a_x)[:, None]
        ay = (2.0 * a_y)[:, None]
        az = (2.0 * a_z)[:, None]

        P0 = torch.stack([source_x, source_y, -depth], dim=-1)   # [B, 3] centroid

        # RD_X: in the Y-Z plane (spanned by c2, c3)
        P1 = P0 + ay * c2 / 2.0 + az * c3 / 2.0
        P2 = P1 - ay * c2
        P3 = P2 - az * c3
        P4 = P1 - az * c3

        # RD_Y: in the X-Z plane (spanned by c1, c3)
        Q1 = P0 - ax * c1 / 2.0 + az * c3 / 2.0
        Q2 = Q1 + ax * c1
        Q3 = Q2 - az * c3
        Q4 = Q1 - az * c3

        # RD_Z: in the X-Y plane (spanned by c1, c2)
        R1 = P0 + ax * c1 / 2.0 + ay * c2 / 2.0
        R2 = R1 - ax * c1
        R3 = R2 - ay * c2
        R4 = R1 - ay * c2

        # Half-space solution: every dislocation vertex must stay at or below
        # the free surface (the reference CDM.m errors with "The CDM must be
        # under the free surface!"). Above-surface vertices would otherwise
        # produce silently unphysical fields.
        vert_z = torch.stack(
            [V[:, 2] for V in (P1, P2, P3, P4, Q1, Q2, Q3, Q4, R1, R2, R3, R4)],
            dim=-1,
        )
        if torch.any(vert_z > 0):
            raise ValueError(
                "CDM extends above the free surface: all dislocation vertices "
                "must satisfy z <= 0. Increase depth or shrink the semi-axes."
            )

        op = opening[:, None]
        ue = torch.zeros_like(x_obs)
        un = torch.zeros_like(x_obs)
        uv = torch.zeros_like(x_obs)
        for V1, V2, V3, V4 in ((P1, P2, P3, P4), (Q1, Q2, Q3, Q4), (R1, R2, R3, R4)):
            e, n, u = _rd_disp_surf(x_obs, y_obs, V1, V2, V3, V4, op, self.v, num_eps)
            ue = ue + e
            un = un + n
            uv = uv + u

        return Displacement(e=ue, n=un, u=uv)


#: The magmatic styles understood by :func:`cdm_params_from_shape` (and the
#: :class:`~torchdeform.simulation.CDMPrior` built on it).
CDM_STYLES = ("sphere", "prolate", "oblate", "dyke", "sill")

#: Thin-axis fraction of ``radius`` for the dyke/sill degenerate (out-of-plane)
#: semi-axis: a nonzero sheet thickness that converges to the zero-thickness limit
#: as it -> 0. The single default shared by :func:`cdm_params_from_shape` and
#: :class:`~torchdeform.simulation.CDMPrior`. (Distinct from the near-vertical
#: kernel guard DEGENERATE_SIN, which happens to share the value.)
FLAT_AXIS_RATIO = 1e-4


def cdm_params_from_shape(
    style: str,
    depth: Tensor,
    radius: Tensor,
    aspect: Tensor,
    dv: Tensor,
    omega_x: Tensor,
    omega_z: Tensor,
    flat_axis_ratio: float = FLAT_AXIS_RATIO,
) -> dict[str, Tensor]:
    """Convert a magmatic-style shape parametrisation to :class:`CDMSource` inputs.

    Restricts the full CDM to the standard volcano-geodesy source geometries
    (sphere / prolate- & oblate-spheroid / dyke / sill) by fixing the three
    semi-axes from an in-plane ``radius`` and an ``aspect`` ratio. This is the CDM
    analogue of :func:`~torchdeform.sources.okada.okada_params_from_fault`: pure
    geometry, so it lives with the source model and is reusable independently of
    the priors (the :class:`~torchdeform.simulation.CDMPrior` calls it under the
    hood).

    Each style is the elementary definition of that shape; the potency ``dv``
    (m^3) is converted to ``opening`` by inverting the CDM potency relation
    ``DV = (2a_x.2a_y + 2a_x.2a_z + 2a_y.2a_z) * opening`` (Nikkhoo et al. 2017)::

        sphere : a_x = a_y = a_z = radius                      (isotropic)
        prolate: a_x = a_y = radius*aspect, a_z = radius
        oblate : a_x = a_y = radius,        a_z = radius*aspect
        dyke   : a_y = radius, a_z = radius*aspect, a_x ~ 0    (vertical sheet)
        sill   : a_x = radius, a_y = radius*aspect, a_z ~ 0    (horizontal sheet)

    For the dyke/sill the out-of-plane semi-axis is nominally zero (a sheet);
    :class:`CDMSource` does not special-case a zero axis, so it is set to
    ``flat_axis_ratio * radius`` -- a thin, but nonzero, sheet that converges to
    the zero-thickness limit as ``flat_axis_ratio -> 0``. ``omega_y`` is always 0
    (these styles tilt only about X and rotate about Z).

    The same family of styles is used for InSAR source inversion by Ireland et al.
    (2026) [doi:10.22541/essoar.15001947/v1]; this is an independent implementation
    from the source geometry and the cited literature. The per-style recipes and
    their provenance (clean-room from that publication) are documented, with
    citations, in ``docs/cdm_style_provenance.md``.

    Parameters
    ----------
    style : str
        One of :data:`CDM_STYLES`.
    depth : Tensor
        Centroid depth ``[B]`` (m, positive down).
    radius : Tensor
        In-plane semi-axis ``[B]`` (m). Its exact role depends on ``style``.
    aspect : Tensor
        Dimensionless axis ratio ``[B]`` (unused for ``sphere``).
    dv : Tensor
        Potency / volume ``[B]`` (m^3); converted to ``opening``.
    omega_x, omega_z : Tensor
        Tilt about X and rotation about Z ``[B]`` (radians).
    flat_axis_ratio : float, default 1e-4
        Thin-axis fraction of ``radius`` used for the dyke/sill degenerate axis.

    Returns
    -------
    dict[str, Tensor]
        ``depth``, ``omega_x/y/z``, ``a_x/a_y/a_z``, ``opening`` -- ready to splat
        into :meth:`CDMSource.forward` alongside the observation coordinates and
        ``source_x`` / ``source_y``.
    """
    if style not in CDM_STYLES:
        raise ValueError(f"unknown CDM style {style!r}; expected one of {CDM_STYLES}")
    r, c = radius, aspect
    if style == "sphere":
        a_x, a_y, a_z = r, r, r
    elif style == "prolate":
        a_x, a_y, a_z = r * c, r * c, r
    elif style == "oblate":
        a_x, a_y, a_z = r, r, r * c
    elif style == "dyke":
        a_x, a_y, a_z = flat_axis_ratio * r, r, r * c
    else:  # sill
        a_x, a_y, a_z = r, r * c, flat_axis_ratio * r

    opening = dv / (4.0 * (a_x * a_y + a_x * a_z + a_y * a_z))
    return {
        "depth": depth,
        "omega_x": omega_x,
        "omega_y": torch.zeros_like(omega_x),
        "omega_z": omega_z,
        "a_x": a_x,
        "a_y": a_y,
        "a_z": a_z,
        "opening": opening,
    }
