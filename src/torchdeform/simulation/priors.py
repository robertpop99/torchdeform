import math
from dataclasses import dataclass, field
from typing import Sequence, Optional

import torch
from torch import Tensor

from ..core import DeviceLikeType


def _rand(
        size: Sequence[int],
        generator: torch.Generator | None = None,
        device: Optional[DeviceLikeType] = None,
        dtype: torch.dtype = torch.float64
) -> Tensor:
    return torch.rand(size, generator=generator, device=device, dtype=dtype)


@dataclass(slots=True)
class UniformPrior:
    low: float
    high: float

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError("high must be > low")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64
    ) -> Tensor:
        return self.low + (self.high - self.low) * _rand(size, generator, device, dtype)

    __call__ = sample


@dataclass(slots=True)
class LogUniformPrior:
    low: float
    high: float

    _lo: float = field(init=False)
    _hi: float = field(init=False)

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError("low must be > 0")

        if self.high <= self.low:
            raise ValueError("high must be > low")

        self._lo = math.log10(self.low)
        self._hi = math.log10(self.high)

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64
    ) -> Tensor:
        u = _rand(
            size,
            generator,
            device,
            dtype,
        )

        return torch.pow(
            10.0,
            self._lo + (self._hi - self._lo) * u,
        )

    __call__ = sample


@dataclass(slots=True)
class SignedLogUniformPrior:
    low: float
    high: float

    _logu: LogUniformPrior = field(init=False)

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError("low must be > 0")

        if self.high <= self.low:
            raise ValueError("high must be > low")

        self._logu = LogUniformPrior(
            low=self.low,
            high=self.high
        )

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64
    ) -> Tensor:
        sign_u = _rand(
            size,
            generator,
            device,
            dtype,
        )

        sign = 1.0 - 2.0 * (sign_u < 0.5).to(dtype)  # ±1, correct dtype

        magnitude = self._logu.sample(
            size,
            generator,
            device,
            dtype,
        )

        return sign * magnitude

    __call__ = sample


