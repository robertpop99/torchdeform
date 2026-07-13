"""
Mogi point-source deformation model.

Implements the classic Mogi (1958) solution for the surface displacement caused
by a point pressure/volume change in an elastic half-space -- the standard
first-order model for an inflating/deflating magma chamber. The implementation
is batched and differentiable in both the observation coordinates and the source
parameters. See :class:`MogiSource`.
"""
import math
import torch
from torch import Tensor

from .base import SourceModel, DEFAULT_POISSON_RATIO
from ..core import Displacement


class MogiSource(SourceModel):
    """
    Mogi point source displacement model.

    Conventions
    -----------
    - All distances in meters
    - Volume change in m^3
    - Depth is positive downward
    - Returns ENU displacement in meters
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
        x_obs: Tensor,      # [B, N] east coordinates of observation points (m)
        y_obs: Tensor,      # [B, N] north coordinates of observation points (m)
        source_x: Tensor,   # [B] source east (m)
        source_y: Tensor,   # [B] source north (m)
        depth: Tensor,      # [B] or [B, N], positive downward (m)
        delta_v: Tensor,    # [B] volume change (m^3)
    ) -> Displacement:
        """Surface displacement from a Mogi point source.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the source [B] in metres.
        depth : Tensor
            Source depth [B] (or per-pixel [B, N]), positive downward, metres
            (must be > 0; the source is buried in the half-space).
        delta_v : Tensor
            Volume change [B] in m^3 (positive = inflation).

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y,
             "depth": depth, "delta_v": delta_v},
        )

        dtype = self.internal_dtype
        num_eps = self._resolve_num_eps()

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)
        delta_v = delta_v.to(dtype)

        if bool((depth <= 0).any()):
            raise ValueError("depth must be strictly positive")

        if depth.ndim == 1:
            depth_b = depth.to(dtype)[:, None]   # [B,1]
        else:
            depth_b = depth.to(dtype)            # [B,N]

        dx = x_obs - source_x[:, None]   # east offset
        dy = y_obs - source_y[:, None]   # north offset

        r2 = dx * dx + dy * dy + depth_b * depth_b
        r32 = torch.pow(r2 + num_eps, 1.5)

        coef = ((1.0 - self.v) / math.pi) * delta_v   # [B]
        coef = coef[:, None]                          # [B,1]

        ue = coef * dx / r32
        un = coef * dy / r32
        uu = coef * depth_b / r32

        return Displacement(
            e=ue,
            n=un,
            u=uu,
        )
