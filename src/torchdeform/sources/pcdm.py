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

from .base import SourceModel
from ..core import Displacement

NUM_EPS = 1e-12   # float64 denominator/sqrt safety


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
    h = torch.sqrt(nx * nx + ny * ny)
    h_safe = h.clamp_min(eps)
    vertical = h <= eps
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
    """

    def __init__(
        self,
        poisson_ratio: float = 0.25,
        internal_dtype: torch.dtype = torch.float64,
        num_eps: float = NUM_EPS,
    ):
        """
        Parameters
        ----------
        poisson_ratio : float, default 0.25
            Poisson's ratio of the elastic half-space.
        internal_dtype : torch.dtype, default torch.float64
            Dtype used for the internal computation; inputs are cast to it.
        num_eps : float, default NUM_EPS
            Numerical guard for denominators / sqrt.
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
            Source depth [B] in metres (positive down).
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
                                     dv[:, None], self.v, self.num_eps)
            ue = ue + e
            un = un + n
            uv = uv + u

        return Displacement(e=ue, n=un, u=uv)
