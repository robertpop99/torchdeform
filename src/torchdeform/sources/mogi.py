import math
import torch
from torch import Tensor

from .base import SourceModel
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
        poisson_ratio: float =0.25,
        internal_dtype: torch.dtype = torch.float64,
        num_eps: float = 1e-12,
    ):
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
        dtype = self.internal_dtype

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)
        delta_v = delta_v.to(dtype)

        if depth.ndim == 1:
            depth_b = depth.to(dtype)[:, None]   # [B,1]
        else:
            depth_b = depth.to(dtype)            # [B,N]

        dx = x_obs - source_x[:, None]   # east offset
        dy = y_obs - source_y[:, None]   # north offset

        r2 = dx * dx + dy * dy + depth_b * depth_b
        r32 = torch.pow(r2 + self.num_eps, 1.5)

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
