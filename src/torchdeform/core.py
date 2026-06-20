from __future__ import annotations
from dataclasses import dataclass, fields, replace
from typing import Any, Optional, TypeAlias, Union, Self

import torch
from torch import Tensor

DeviceLikeType: TypeAlias = Union[str, torch.device, int]


# noinspection PyTypeChecker,PyDataclass
class TensorDataclassMixin:

    def to(self, *args, **kwargs) -> Self:
        values = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                value = value.to(*args, **kwargs)

            elif hasattr(value, "to"):
                value = value.to(*args, **kwargs)

            values[f.name] = value

        return replace(self, **values)

    def detach(self) -> Self:
        values = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, torch.Tensor):
                value = value.detach()

            elif hasattr(value, "detach"):
                value = value.detach()

            values[f.name] = value

        return replace(self, **values)

    @property
    def device(self) -> Optional[DeviceLikeType]:
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                return value.device

            if hasattr(value, "device"):
                return value.device

        return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                return value.dtype

            if hasattr(value, "dtype"):
                return value.dtype

        return None

    def cpu(self) -> Self:
        return self.to("cpu")

    def cuda(self) -> Self:
        return self.to("cuda")

    def clone(self) -> Self:
        values = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, torch.Tensor):
                value = value.clone()

            elif hasattr(value, "clone"):
                value = value.clone()

            values[f.name] = value

        return replace(self, **values)


@dataclass(slots=True)
class Displacement(TensorDataclassMixin):
    """
    Ground displacement field.

    Attributes
    ----------
    e : Tensor
        East displacement [B, N]
    n : Tensor
        North displacement [B, N]
    u : Tensor
        Up displacement [B, N]
    """

    e: Tensor
    n: Tensor
    u: Tensor

    def to_los(self, los: LOSVector) -> Tensor:
        return (
            self.e * los.e +
            self.n * los.n +
            self.u * los.u
        )


@dataclass(slots=True)
class LOSVector(TensorDataclassMixin):
    """
    Line-of-sight unit vector.

    Attributes
    ----------
    e : Tensor
        East component [B, N]
    n : Tensor
        North component [B, N]
    u : Tensor
        Up component [B, N]
    """

    e: Tensor
    n: Tensor
    u: Tensor

    def project(self, disp: Displacement) -> Tensor:
        return (
            self.e * disp.e +
            self.n * disp.n +
            self.u * disp.u
        )

    @property
    def norm(self) -> Tensor:
        return torch.sqrt(
            self.e**2 +
            self.n**2 +
            self.u**2
        )


@dataclass(slots=True)
class ECEF(TensorDataclassMixin):
    x: Tensor
    y: Tensor
    z: Tensor

    @classmethod
    def from_xyz(
        cls,
        x: Any,
        y: Any,
        z: Any,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLikeType] = None,
    ) -> ECEF:
        return cls(
            x=torch.as_tensor(x, dtype=dtype, device=device),
            y=torch.as_tensor(y, dtype=dtype, device=device),
            z=torch.as_tensor(z, dtype=dtype, device=device),
        )

    @property
    def xyz(self) -> Tensor:
        return torch.stack(
            (self.x, self.y, self.z),
            dim=-1,
        )

    def to_geodetic(self) -> Geodetic:
        from .geometry.coordinates import ecef_to_geodetic

        return ecef_to_geodetic(self)

    def to_local_enu(self, reference: Geodetic) -> tuple[Tensor, Tensor, Tensor]:
        from .geometry.coordinates import ecef_to_local_enu

        return ecef_to_local_enu(self, reference)


@dataclass(slots=True)
class Geodetic(TensorDataclassMixin):
    lat_deg: Tensor
    lon_deg: Tensor
    height_m: Tensor

    @classmethod
    def from_degrees(
        cls,
        lat_deg: Any,
        lon_deg: Any,
        height_m: Optional[Any] = None,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLikeType] = None,
    ) -> Geodetic:
        lat_deg = torch.as_tensor(lat_deg, dtype=dtype, device=device)
        lon_deg = torch.as_tensor(lon_deg, dtype=dtype, device=device)
        if height_m is None:
            height_m = torch.zeros_like(lat_deg, dtype=torch.float64, device=device)
        else:
            height_m = torch.as_tensor(height_m, dtype=dtype, device=device)

        return cls(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            height_m=height_m,
        )

    def to_ecef(self) -> ECEF:
        from .geometry.coordinates import geodetic_to_ecef

        return geodetic_to_ecef(self)


# @dataclass(slots=True)
# class InSARPhase(TensorDataclassMixin):
#     phase: Tensor