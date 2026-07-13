"""
Point Compound Dislocation Model (pCDM) source.

A volcanic point source built from three mutually orthogonal point tensile
dislocations (Okada, 1985) in an elastic half-space, after Nikkhoo et al.
(2017). It generalises the isotropic point source -- an isotropic pCDM is a
center of dilatation, i.e. Mogi-like -- to anisotropic/triaxial volume changes,
so dykes, sills and general ellipsoidal cavities all appear as special cases.

This is a differentiable, batched PyTorch port of the surface-displacement part
of Nikkhoo's ``pCDM.m``. Instead of extracting a strike/dip per dislocation
(which has gradient kinks at vertical normals and needs ``isnan`` patching), the
three dislocation normals are taken directly as the columns of the orientation
matrix and each per-dislocation horizontal rotation is built from the normal's
components -- a "direct-rotation" formulation that is smooth everywhere for
``depth > 0``.

Attribution
-----------
Ported from the MATLAB ``pCDM`` by Mehdi Nikkhoo (MIT licensed). The full MIT
copyright and permission notice is reproduced in the project ``NOTICE`` file.

Reference: Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2017),
"Compound dislocation models (CDMs) for volcano deformation analyses",
Geophysical Journal International, 208(2), 877-894.
"""
import math

import torch
from torch import Tensor

from .base import SourceModel, DEFAULT_POISSON_RATIO
from ..core import Displacement

# |sin(dip)| (= horizontal magnitude of the unit normal) below this => the PTD is
# treated as vertical and the in-plane frame is fixed to (1, 0). The frame
# ``(-ny, -nx)/h`` suffers catastrophic cancellation in its gradient for a
# *near*-vertical normal (the orientation gradient blows up to ~1e9 as h -> 0 in
# float64), so this is set well above machine eps; a normal within ~1e-6 rad of
# vertical is axisymmetric to O(1e-6), making the fixed-frame choice negligible in
# the forward while keeping the backward bounded.
DEGENERATE_SIN = 1e-6


def _rotation_matrix(ox: Tensor, oy: Tensor, oz: Tensor) -> Tensor:
    """Batched ``Rz @ Ry @ Rx`` rotation matrices ``[B, 3, 3]`` from angles ``[B]``.

    Angles are in radians and follow the clockwise-about-axis convention of
    Nikkhoo's pCDM. The columns of the returned matrix are the unit normals of
    the three point tensile dislocations.
    """
    cx, sx = torch.cos(ox), torch.sin(ox)
    cy, sy = torch.cos(oy), torch.sin(oy)
    cz, sz = torch.cos(oz), torch.sin(oz)
    z = torch.zeros_like(ox)
    o = torch.ones_like(ox)

    def mat(rows):
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)

    rx = mat([[o, z, z], [z, cx, sx], [z, -sx, cx]])
    ry = mat([[cy, z, -sy], [z, o, z], [sy, z, cy]])
    rz = mat([[cz, sz, z], [-sz, cz, z], [z, z, o]])
    return rz @ ry @ rx


def _ptd_disp_surf(dx, dy, depth, nx, ny, nz, dv, nu, eps):
    """Surface displacement of one point tensile dislocation (PTD).

    Parameters
    ----------
    dx, dy : Tensor
        Observation offsets from the source, ``[B, N]``.
    depth, nx, ny, nz, dv : Tensor
        Depth, the three components of the (unit) dislocation normal, and the
        potency, each ``[B, 1]``.
    nu : float
        Poisson's ratio.
    eps : float
        Numerical guard.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(ue, un, uv)`` each ``[B, N]``.
    """
    # sin(dip) = horizontal magnitude of the normal; cos(dip) = vertical comp.
    # The +eps*eps inside the sqrt keeps its slope finite at a vertical normal
    # (nx = ny = 0): ``h`` feeds the output directly as ``sd`` and also the
    # ``-n/h_safe`` branch of the frame below, so a bare sqrt(0) (slope +inf)
    # would make ``0 * inf = NaN`` poison every orientation gradient there.
    h = torch.sqrt(nx * nx + ny * ny + eps * eps)
    h_safe = h.clamp_min(eps)
    vertical = h <= DEGENERATE_SIN
    # horizontal rotation by beta = strike - 90, expressed via the normal:
    #   cos(beta) = -ny / h,  sin(beta) = -nx / h
    # (at a vertical normal the field is axisymmetric, so any frame works -> (1,0))
    cos_b = torch.where(vertical, torch.ones_like(h), -ny / h_safe)
    sin_b = torch.where(vertical, torch.zeros_like(h), -nx / h_safe)
    sd = h          # sin(dip)
    cd = nz         # cos(dip)

    # rotate observation coordinates into the dislocation's strike frame
    xr = cos_b * dx - sin_b * dy
    yr = sin_b * dx + cos_b * dy
    d = depth

    r2 = xr * xr + yr * yr + d * d + eps
    r = torch.sqrt(r2)
    r3 = r2 * r
    r5 = r3 * r2
    rpd = r + d
    rpd2 = rpd * rpd
    rpd3 = rpd2 * rpd

    inv = 1.0 - 2.0 * nu
    a3 = 3.0 * r + d
    I1 = inv * yr * (1.0 / (r * rpd2) - xr * xr * a3 / (r3 * rpd3))
    I2 = inv * xr * (1.0 / (r * rpd2) - yr * yr * a3 / (r3 * rpd3))
    I3 = inv * xr / r3 - I2
    I5 = inv * (1.0 / (r * rpd) - xr * xr * (2.0 * r + d) / (r3 * rpd2))

    q = yr * sd - d * cd
    q2 = q * q
    fac = dv / (2.0 * math.pi)
    sd2 = sd * sd

    ue_l = fac * (3.0 * xr * q2 / r5 - I3 * sd2)
    un_l = fac * (3.0 * yr * q2 / r5 - I1 * sd2)
    uv = fac * (3.0 * d * q2 / r5 - I5 * sd2)

    # rotate the horizontal displacement back to the Earth-fixed frame
    ue = cos_b * ue_l + sin_b * un_l
    un = -sin_b * ue_l + cos_b * un_l
    return ue, un, uv


class PCDMSource(SourceModel):
    """
    Point Compound Dislocation Model: a triaxial volcanic point source.

    Sums three mutually orthogonal point tensile dislocations whose common
    orientation is set by the rotation angles ``omega_x/y/z`` and whose strengths
    are the potencies ``dv_x/y/z`` (volume, m^3). Equal potencies give an
    isotropic source (center of dilatation, Mogi-like); unequal potencies give
    dyke/sill/ellipsoidal-cavity styles.

    Conventions
    -----------
    - All distances in metres; depth positive downward.
    - Rotation angles ``omega_x/y/z`` in radians (consistent with the other
      sources; see :class:`~torchdeform.simulation.PCDMPrior`, which samples them).
    - Potencies ``dv_x/y/z`` in m^3 and must share a sign per item (a coherent
      inflation or deflation); mixed signs raise ``ValueError``.
    - Returns ENU surface displacement in metres.

    Known limitation (orientation gradient at an exactly vertical PTD normal)
    ------------------------------------------------------------------------
    When the orientation makes one of the three PTD normals *exactly* vertical
    (an axis-aligned box, ``omega_x = omega_y = 0`` for any ``omega_z``), the
    in-plane frame ``(-ny, -nx)/h`` -- with ``h`` the horizontal magnitude of the
    unit normal -- is built around an azimuth that is undefined at the pole. The
    forward stays correct (the field is axisymmetric there, so the fixed ``(1, 0)``
    frame is exact) and the gradient is finite and bounded, but the *orientation*
    gradient (``d/d omega``) at that exact configuration is only approximate
    (correctly signed, right order of magnitude, but not exact). For any tilt
    beyond ``DEGENERATE_SIN`` (~1e-6 rad) the gradient is exact, so this affects
    only a measure-zero set that :class:`~torchdeform.simulation.PCDMPrior` (which
    samples ``omega`` continuously) does not hit; the axis-aligned presets that
    *do* fix ``omega = 0`` route through :class:`~torchdeform.sources.CDMSource` /
    :class:`~torchdeform.sources.PECMSource`, whose degenerate-case gradients are
    exact. In practice the worst case is a single bounded, correctly-signed step
    when an optimisation is initialised exactly at ``omega = 0``.

    The cause is purely the ``1/h`` strike-frame parametrisation, not a missing
    closed-form derivative: the field is genuinely smooth in the unit normal (and
    hence in ``omega``) through the pole, so a hand-written analytic backward would
    hit the same ``0/0`` and need the same special-casing. The proper fix, if
    pole-exact orientation gradients are ever needed, is to make the *forward*
    singularity-free -- express the strike rotation directly via ``nx, ny, nz``
    (so the explicit ``1/h`` azimuth never appears and the ``1/h`` factors cancel
    against the ``sd = h`` terms and the outer inverse rotation) -- after which
    plain autograd is exact everywhere with no degenerate special case. That is
    cheaper and lower-risk than authoring a closed-form Jacobian.
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
        dv_x: Tensor,       # [B] potency normal to X before rotation (m^3)
        dv_y: Tensor,       # [B] potency normal to Y before rotation (m^3)
        dv_z: Tensor,       # [B] potency normal to Z before rotation (m^3)
    ) -> Displacement:
        """Surface displacement from a point compound dislocation model.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the source [B] in metres.
        depth : Tensor
            Source depth [B] in metres (positive down; must be > 0, the source
            is buried in the half-space).
        omega_x, omega_y, omega_z : Tensor
            Clockwise rotation angles about the X/Y/Z axes [B] in radians,
            setting the source orientation.
        dv_x, dv_y, dv_z : Tensor
            Potencies [B] (m^3) of the three orthogonal point tensile
            dislocations (normal to X/Y/Z before rotation). Must share a sign
            per item.

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "depth": depth,
             "omega_x": omega_x, "omega_y": omega_y, "omega_z": omega_z,
             "dv_x": dv_x, "dv_y": dv_y, "dv_z": dv_z},
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
        dv_x = dv_x.to(dtype)
        dv_y = dv_y.to(dtype)
        dv_z = dv_z.to(dtype)

        if bool((depth <= 0).any()):
            raise ValueError("depth must be strictly positive")

        # physical constraint: a coherent volume change -> the potencies share a sign
        pos = (dv_x > 0) | (dv_y > 0) | (dv_z > 0)
        neg = (dv_x < 0) | (dv_y < 0) | (dv_z < 0)
        if bool((pos & neg).any()):
            raise ValueError(
                "PCDMSource requires dv_x, dv_y, dv_z to share a sign per item "
                "(a coherent inflation or deflation)"
            )

        rot = _rotation_matrix(omega_x, omega_y, omega_z)   # [B, 3, 3]

        dx = x_obs - source_x[:, None]
        dy = y_obs - source_y[:, None]
        depth_b = depth[:, None]

        ue = torch.zeros_like(dx)
        un = torch.zeros_like(dx)
        uv = torch.zeros_like(dx)
        for k, dv in enumerate((dv_x, dv_y, dv_z)):
            nx = rot[:, 0, k][:, None]
            ny = rot[:, 1, k][:, None]
            nz = rot[:, 2, k][:, None]
            e, n, u = _ptd_disp_surf(dx, dy, depth_b, nx, ny, nz,
                                     dv[:, None], self.v, num_eps)
            ue = ue + e
            un = un + n
            uv = uv + u

        return Displacement(e=ue, n=un, u=uv)
