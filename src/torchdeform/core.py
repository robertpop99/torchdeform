"""
Core tensor-backed data structures shared across torchdeform.

This module defines the small dataclasses that the deformation, observation and
coordinate code pass around -- ground displacement fields (:class:`Displacement`),
line-of-sight unit vectors (:class:`LOSVector`) and the two geodetic coordinate
representations (:class:`ECEF`, :class:`Geodetic`).

Every dataclass mixes in :class:`TensorDataclassMixin`, which gives them the
tensor-like ergonomics (``.to(...)``, ``.detach()``, ``.cpu()``, ``.device`` ...)
without anyone having to reimplement them per field. All fields are plain
``torch.Tensor`` objects, so the structures are fully differentiable and can be
moved between devices/dtypes like ordinary tensors.

Shape convention
----------------
Spatial quantities are batched as ``[B, N]`` (B images, N observation points).
"""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
from typing import Any, Optional, TypeAlias, Union, Self

import torch
from torch import Tensor

DeviceLikeType: TypeAlias = Union[str, torch.device, int]


# noinspection PyTypeChecker,PyDataclass
class TensorDataclassMixin:
    """
    Mixin giving a tensor-holding dataclass tensor-like behaviour.

    All transforms are implemented by :meth:`_map`, which walks the dataclass
    fields and calls the named method on every field that supports it (a
    :class:`~torch.Tensor`, or a nested object exposing the same method, e.g.
    another ``TensorDataclassMixin``), returning a new instance via
    :func:`dataclasses.replace`. The original instance is left unchanged.
    """

    def _map(self, method: str, *args, **kwargs) -> Self:
        """Return a copy with ``value.<method>(*args, **kwargs)`` applied to every
        field that has that method; other fields are copied unchanged."""
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, method):
                value = getattr(value, method)(*args, **kwargs)
            values[f.name] = value
        return replace(self, **values)

    def to(self, *args, **kwargs) -> Self:
        """Return a copy with every tensor field moved/cast via ``Tensor.to``.

        Accepts the same arguments as :meth:`torch.Tensor.to` (device, dtype,
        ...). Non-tensor fields exposing a ``.to`` method are forwarded too;
        anything else is copied as-is.
        """
        return self._map("to", *args, **kwargs)

    def detach(self) -> Self:
        """Return a copy with every tensor field detached from the autograd graph."""
        return self._map("detach")

    @property
    def device(self) -> Optional[DeviceLikeType]:
        """Device of the first tensor-like field, or ``None`` if there is none."""
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                return value.device

            if hasattr(value, "device"):
                return value.device

        return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Dtype of the first tensor-like field, or ``None`` if there is none."""
        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, Tensor):
                return value.dtype

            if hasattr(value, "dtype"):
                return value.dtype

        return None

    def cpu(self) -> Self:
        """Return a copy with every tensor field moved to the CPU."""
        return self.to("cpu")

    def cuda(self) -> Self:
        """Return a copy with every tensor field moved to the current CUDA device."""
        return self.to("cuda")

    def clone(self) -> Self:
        """Return a deep copy with every tensor field cloned (autograd preserved)."""
        return self._map("clone")


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
        """Project this displacement onto a line-of-sight vector.

        Parameters
        ----------
        los : LOSVector
            Line-of-sight unit vector (broadcastable to this field's shape).

        Returns
        -------
        Tensor
            Scalar LOS displacement ``e*los.e + n*los.n + u*los.u`` [B, N].
            Positive values follow ``los``'s sign convention (ground -> satellite).
        """
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
        """Project a displacement field onto this line-of-sight vector.

        Convenience alias for :meth:`Displacement.to_los` (the canonical
        implementation of the projection); returns the scalar LOS displacement
        ``e*disp.e + n*disp.n + u*disp.u`` [B, N].
        """
        return disp.to_los(self)

    @property
    def norm(self) -> Tensor:
        """Euclidean norm of the vector [B, N] (1 for a true unit LOS vector)."""
        return torch.sqrt(
            self.e**2 +
            self.n**2 +
            self.u**2
        )


@dataclass(slots=True)
class ECEF(TensorDataclassMixin):
    """
    Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.

    Attributes
    ----------
    x, y, z : Tensor
        ECEF coordinates in metres. Any broadcast-compatible shape (e.g. a
        scalar per item, ``[B]``, or ``[B, N]`` for per-pixel positions).
    """

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
        """Build an :class:`ECEF` from array-likes, coercing each to a tensor.

        ``x``/``y``/``z`` may be Python numbers, lists or tensors; each is passed
        through :func:`torch.as_tensor` with the given ``dtype`` and ``device``.
        """
        return cls(
            x=torch.as_tensor(x, dtype=dtype, device=device),
            y=torch.as_tensor(y, dtype=dtype, device=device),
            z=torch.as_tensor(z, dtype=dtype, device=device),
        )

    @property
    def xyz(self) -> Tensor:
        """The three components stacked along a trailing axis, shape ``[..., 3]``."""
        return torch.stack(
            (self.x, self.y, self.z),
            dim=-1,
        )

    def to_geodetic(self) -> Geodetic:
        """Convert to :class:`Geodetic` (lat/lon/height) on the WGS84 ellipsoid."""
        from .geometry.coordinates import ecef_to_geodetic

        return ecef_to_geodetic(self)

    def to_local_enu(self, reference: Geodetic) -> tuple[Tensor, Tensor, Tensor]:
        """Express these coordinates as local East/North/Up about ``reference``.

        Returns a tuple ``(east, north, up)`` of tensors in metres relative to
        the ``reference`` geodetic origin.
        """
        from .geometry.coordinates import ecef_to_local_enu

        return ecef_to_local_enu(self, reference)


@dataclass(slots=True)
class Geodetic(TensorDataclassMixin):
    """
    Geodetic coordinates on the WGS84 ellipsoid.

    Attributes
    ----------
    lat_deg : Tensor
        Geodetic latitude in degrees.
    lon_deg : Tensor
        Longitude in degrees.
    height_m : Tensor
        Ellipsoidal height in metres.
    """

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
        """Build a :class:`Geodetic` from array-likes (degrees, degrees, metres).

        ``height_m`` is optional; when omitted it defaults to zeros shaped like
        ``lat_deg``. All inputs are coerced via :func:`torch.as_tensor`.
        """
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
        """Convert to :class:`ECEF` Cartesian coordinates on the WGS84 ellipsoid."""
        from .geometry.coordinates import geodetic_to_ecef

        return geodetic_to_ecef(self)


# @dataclass(slots=True)
# class InSARPhase(TensorDataclassMixin):
#     phase: Tensor