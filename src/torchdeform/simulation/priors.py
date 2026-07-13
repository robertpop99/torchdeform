"""
Parameter priors for sampling synthetic source/scene parameters.

Lightweight, callable distributions used to randomise source parameters when
generating training data (depths, volume changes, slip, pressures, ...).

Scalar priors
-------------
Each scalar prior implements the :class:`Prior` interface -- ``sample(size, ...)``
(also callable directly) returning a tensor of independent draws:

* :class:`UniformPrior` -- uniform on ``[low, high]``.
* :class:`LogUniformPrior` -- log-uniform on ``[low, high]`` (both > 0); use for
  scale parameters spanning orders of magnitude. Concentrates mass near ``low``.
* :class:`ReverseLogUniformPrior` -- the mirror image of ``LogUniformPrior``,
  concentrating mass near ``high`` instead (e.g. depths biased deep).
* :class:`SignedPrior` -- wrap any positive-magnitude prior and attach a random
  +/-1 sign; the composable base for signed scale parameters.
* :class:`SignedLogUniformPrior` -- log-uniform magnitude with a random sign
  (a ``SignedPrior`` preset); use for signed scale parameters (inflation/deflation).
* :class:`NormalPrior` -- Gaussian ``N(mean, std)``; a *peaked* alternative to
  ``UniformPrior`` for "typical value with spread".
* :class:`TruncatedNormalPrior` -- Gaussian truncated to ``[low, high]``: peaked
  but bounded, so it never emits an out-of-range value (e.g. negative depth).
* :class:`LogNormalPrior` -- log-normal ``median * exp(sigma * N(0, 1))``; the
  peaked analogue of ``LogUniformPrior`` for positive scale parameters.
* :class:`SignedLogNormalPrior` -- log-normal magnitude with a random sign
  (a ``SignedPrior`` preset).
* :class:`PowerLawPrior` -- bounded power law ``pdf proportional to x**-alpha`` on
  ``[low, high]`` (Gutenberg--Richter / fractal size scaling; ``alpha=1`` recovers
  log-uniform, larger ``alpha`` piles more mass near ``low``).
* :class:`VonMisesPrior` -- circular Gaussian for angles; wraps correctly across
  the 0/360 seam, and ``concentration -> 0`` is uniform on the circle.
* :class:`ConstantPrior` -- always returns a fixed value (to pin a parameter).
* :class:`ChoicePrior` -- draw from a fixed set of values with optional weights
  (e.g. ``look_side`` in ``{+1, -1}``).
* :class:`MultimodalPrior` -- a finite mixture of scalar priors (per-draw weighted
  choice); for parameters with separated modes, e.g. a Sentinel-1 heading.

Subclass :class:`Prior` to add a distribution and it works anywhere a prior is
accepted. :func:`make_prior` builds one from a ``(low, high, mode)`` spec, handy
for config-driven setups.

Prior bundles
-------------
:class:`PriorBundle` subclasses bundle one named scalar prior per parameter,
named exactly as a consuming function's arguments, so ``f(**bundle.sample(n))``
works: source models (:class:`MogiPrior`, :class:`PennyPrior`,
:class:`OkadaPrior`, :class:`PCDMPrior`) and acquisition geometry
(:class:`GeometryPrior`, whose fields feed :func:`~torchdeform.observation.los_vector`).
``OkadaPrior`` samples a geophysical fault parametrisation; convert it with
:func:`~torchdeform.sources.okada.okada_params_from_fault`.

Default instances are provided (:data:`DEFAULT_MOGI_PRIOR`,
:data:`DEFAULT_S1_GEOMETRY_PRIOR`, ...). The Okada defaults
(:data:`DEFAULT_EARTHQUAKE_PRIOR`, :data:`DEFAULT_DYKE_PRIOR`,
:data:`DEFAULT_SILL_PRIOR`) are opinionated presets -- ordinary ``OkadaPrior``
instances -- not distinct types. The source defaults are collected in
:data:`DEFAULT_PRIORS`.

(``SourcePrior`` is a backwards-compatible alias for ``PriorBundle``.)

Mixtures
--------
:class:`PriorMixture` holds several bundles with relative selection ``weights``
and samples a key per batch item (then that bundle's parameters). Weights live on
the mixture, not the bundles, so a bundle can be reused across datasets with
different mixes. (``SourceMixture`` is a backwards-compatible alias.)

Mapping parameters to an ML target space (normalisation, sin/cos angle
encodings, network head layout, ...) is intentionally left to the application:
this module produces physical parameters, not training targets.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Sequence, Optional

import torch
from torch import Tensor

from ..core import DeviceLikeType
from ..observation.los import (
    S1_INCIDENCE_RANGE_DEG,
    S1_HEADING_ASCENDING_DEG,
    S1_HEADING_DESCENDING_DEG,
    S1_LOOK_SIDE,
)
# CDM shape->forward adapter lives with the source model (cf. okada_params_from_fault);
# re-exported here so CDMPrior and the simulation namespace can use it.
from ..sources.cdm import cdm_params_from_shape, CDM_STYLES, FLAT_AXIS_RATIO


def _rand(
        size: Sequence[int],
        generator: torch.Generator | None = None,
        device: Optional[DeviceLikeType] = None,
        dtype: torch.dtype = torch.float64
) -> Tensor:
    return torch.rand(size, generator=generator, device=device, dtype=dtype)


class Prior(ABC):
    """Abstract base for a scalar parameter prior.

    A prior is any object that can draw independent samples of one scalar
    parameter into a tensor. Concrete subclasses implement :meth:`sample`;
    instances are also directly callable (``prior(size, ...)``). Subclass this
    to add a new distribution and it will work anywhere a ``Prior`` is accepted
    (e.g. as a field of a :class:`SourcePrior`).
    """

    __slots__ = ()

    @abstractmethod
    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` from the prior."""
        ...

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.sample(*args, **kwargs)


@dataclass(slots=True)
class UniformPrior(Prior):
    """Uniform prior on the closed interval ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Interval bounds; ``high`` must be strictly greater than ``low``.
    """

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
        """Draw a tensor of shape ``size`` uniform on ``[low, high]``.

        Also callable directly (``prior(size, ...)``). ``generator``/``device``/
        ``dtype`` are forwarded to the underlying :func:`torch.rand`.
        """
        return self.low + (self.high - self.low) * _rand(size, generator, device, dtype)


@dataclass(slots=True)
class LogUniformPrior(Prior):
    """Log-uniform prior on ``[low, high]`` (draws uniform in ``log10``).

    Use for positive scale parameters that span several orders of magnitude, so
    each decade is equally likely.

    Parameters
    ----------
    low, high : float
        Interval bounds; both must be ``> 0`` and ``high > low``.
    """

    low: float
    high: float

    _lo: float = field(init=False, repr=False)
    _hi: float = field(init=False, repr=False)

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
        """Draw a tensor of shape ``size`` log-uniform on ``[low, high]``.

        Also callable directly. Returns strictly positive values.
        """
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


@dataclass(slots=True)
class SignedPrior(Prior):
    """Wrap a positive-magnitude prior and apply a random +/-1 sign per draw.

    The composable way to make any positive scale prior *signed*: draw a
    magnitude from ``magnitude`` and multiply it by +1 or -1 with equal
    probability. Use for signed scale parameters (volume change -- inflation vs
    deflation; slip direction). ``SignedPrior(LogUniformPrior(...))``,
    ``SignedPrior(LogNormalPrior(...))`` and
    ``SignedPrior(ReverseLogUniformPrior(...))`` all work, so there is no need for
    a dedicated ``Signed<X>`` class per magnitude shape. The named
    :class:`SignedLogUniformPrior` / :class:`SignedLogNormalPrior` are thin
    presets over this (kept for the ``make_prior`` bridge and readability).

    Parameters
    ----------
    magnitude : Prior
        Prior for the (positive) magnitude. Whatever it returns is multiplied by
        the random sign, so give it a strictly positive distribution.
    """

    magnitude: Prior

    def __post_init__(self):
        if not isinstance(self.magnitude, Prior):
            raise TypeError("magnitude must be a Prior")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw ``size`` signed values: a random +/-1 times a ``magnitude`` draw.

        Also callable directly. Sign and magnitude are drawn independently, sign
        first, so a given ``generator`` reproduces the same stream as the
        dedicated signed presets.
        """
        sign_u = _rand(size, generator, device, dtype)
        sign = 1.0 - 2.0 * (sign_u < 0.5).to(dtype)         # ±1, correct dtype
        magnitude = self.magnitude.sample(size, generator, device, dtype)
        return sign * magnitude


class SignedLogUniformPrior(SignedPrior):
    """Symmetric signed log-uniform prior (a :class:`SignedPrior` preset).

    Draws a log-uniform magnitude on ``[low, high]`` and multiplies it by a
    random sign (+1/-1 with equal probability). Use for signed scale parameters
    such as volume change (inflation vs deflation) or slip direction. Equivalent
    to ``SignedPrior(LogUniformPrior(low, high))``; kept as a named class for the
    ``make_prior('signed_log')`` bridge and readability.

    Parameters
    ----------
    low, high : float
        Magnitude bounds; both must be ``> 0`` and ``high > low``. The realised
        values fall in ``[-high, -low] u [low, high]``.
    """

    __slots__ = ()

    def __init__(self, low: float, high: float):
        super().__init__(LogUniformPrior(low, high))


@dataclass(slots=True)
class ConstantPrior(Prior):
    """Degenerate prior that always returns a fixed value.

    Useful for pinning a parameter -- e.g. ``opening=ConstantPrior(0.0)`` for a
    pure-shear fault, or ``rake=ConstantPrior(0.0)`` for a pure-opening source.

    Parameters
    ----------
    value : float
        The constant value every draw returns.
    """

    value: float

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Return a tensor of shape ``size`` filled with ``value``."""
        return torch.full(tuple(size), float(self.value), device=device, dtype=dtype)


@dataclass(slots=True)
class MultimodalPrior(Prior):
    """Finite mixture of scalar priors: each draw picks one component by weight.

    Per element, a component prior is chosen ~ Categorical(weights) and sampled
    from. Use for a parameter whose distribution has separated modes -- e.g. a
    Sentinel-1 heading, which clusters around the ascending and descending
    azimuths. (It is only genuinely "multimodal" if the components are separated;
    overlapping components simply blend.)

    Parameters
    ----------
    priors : Sequence[Prior]
        Component priors to mix.
    weights : Sequence[float], optional
        Relative (unnormalised) selection weight per component; must be strictly
        positive. Defaults to uniform.
    """

    priors: Sequence[Prior]
    weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        if len(self.priors) == 0:
            raise ValueError("priors must be non-empty")
        if self.weights is not None:
            if len(self.weights) != len(self.priors):
                raise ValueError("weights must have one entry per prior")
            if any(float(w) <= 0.0 for w in self.weights):
                raise ValueError("weights must be strictly positive")

    def _probabilities(self, device: Optional[DeviceLikeType]) -> Tensor:
        w = [1.0] * len(self.priors) if self.weights is None else [float(x) for x in self.weights]
        t = torch.tensor(w, dtype=torch.float64, device=device)
        return t / t.sum()

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw ``size``, each element from a weighted-random component prior."""
        comps = torch.stack(
            [p.sample(size, generator, device, dtype) for p in self.priors], dim=0
        )                                                   # [K, *size]
        flat = comps.reshape(comps.shape[0], -1)            # [K, M]
        probs = self._probabilities(device)
        choice = torch.multinomial(probs, flat.shape[1], replacement=True,
                                   generator=generator)     # [M]
        return flat.gather(0, choice[None, :]).reshape(tuple(size))


@dataclass(slots=True)
class ReverseLogUniformPrior(Prior):
    """Mirror image of :class:`LogUniformPrior`, concentrating mass near ``high``.

    ``LogUniformPrior`` piles its density up against ``low``; reflecting a
    log-uniform draw about the interval (``high + low - x``) gives the same
    log-shaped concentration piled against ``high`` instead. Use it for a
    positive parameter you want biased toward the *top* of its range -- e.g. a
    depth that should usually be deep.

    Parameters
    ----------
    low, high : float
        Interval bounds; both must be ``> 0`` and ``high > low`` (the log
        spacing is defined on ``[low, high]`` before reflection).
    """

    low: float
    high: float

    _logu: LogUniformPrior = field(init=False, repr=False)

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError("low must be > 0")

        if self.high <= self.low:
            raise ValueError("high must be > low")

        self._logu = LogUniformPrior(low=self.low, high=self.high)

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` log-concentrated near ``high``.

        Also callable directly. Values fall in ``[low, high]``.
        """
        x = self._logu.sample(size, generator, device, dtype)
        return (self.low + self.high) - x


@dataclass(slots=True)
class NormalPrior(Prior):
    """Gaussian prior ``N(mean, std)``.

    A peaked alternative to :class:`UniformPrior` when a parameter has a typical
    value with some spread rather than a flat plausible range. Unbounded -- use
    :class:`TruncatedNormalPrior` when draws must stay within physical limits.

    Parameters
    ----------
    mean : float
        Distribution mean.
    std : float
        Standard deviation; must be ``> 0``.
    """

    mean: float
    std: float

    def __post_init__(self):
        if self.std <= 0:
            raise ValueError("std must be > 0")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` from ``N(mean, std)``."""
        z = torch.randn(size, generator=generator, device=device, dtype=dtype)
        return self.mean + self.std * z


@dataclass(slots=True)
class TruncatedNormalPrior(Prior):
    """Gaussian ``N(mean, std)`` truncated to ``[low, high]``.

    Peaked like :class:`NormalPrior` but bounded, so every draw is a physically
    valid value (e.g. a depth that clusters around a nominal value yet is
    guaranteed positive). Sampled exactly by inverse-CDF, not rejection.

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of the *untruncated* Gaussian (``std > 0``).
    low, high : float
        Truncation bounds; ``high`` must be ``> low``.
    """

    mean: float
    std: float
    low: float
    high: float

    def __post_init__(self):
        if self.std <= 0:
            raise ValueError("std must be > 0")

        if self.high <= self.low:
            raise ValueError("high must be > low")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` from the truncated Gaussian.

        Uses inverse-CDF sampling (``u`` uniform in ``[Phi(a), Phi(b)]`` mapped
        back through the normal quantile), so all draws lie in ``[low, high]``.
        """
        a = (self.low - self.mean) / self.std
        b = (self.high - self.mean) / self.std
        lo = torch.special.ndtr(torch.tensor(a, dtype=dtype, device=device))
        hi = torch.special.ndtr(torch.tensor(b, dtype=dtype, device=device))
        u = _rand(size, generator, device, dtype)
        z = torch.special.ndtri(lo + (hi - lo) * u)
        return (self.mean + self.std * z).clamp(self.low, self.high)


@dataclass(slots=True)
class LogNormalPrior(Prior):
    """Log-normal prior: ``median * exp(sigma * N(0, 1))``.

    The peaked analogue of :class:`LogUniformPrior` for positive scale
    parameters (volume change, slip, pressure, ...): unimodal in log-space with
    heavy tails, rather than flat across decades.

    Parameters
    ----------
    median : float
        Median of the distribution (``exp`` of the underlying normal's mean);
        must be ``> 0``.
    sigma : float
        Standard deviation of ``ln(X)`` (the log-space spread); must be ``> 0``.
    """

    median: float
    sigma: float

    def __post_init__(self):
        if self.median <= 0:
            raise ValueError("median must be > 0")

        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` of strictly positive log-normal values."""
        z = torch.randn(size, generator=generator, device=device, dtype=dtype)
        return self.median * torch.exp(self.sigma * z)


class SignedLogNormalPrior(SignedPrior):
    """Symmetric signed log-normal prior (a :class:`SignedPrior` preset).

    Draws a log-normal magnitude and multiplies it by a random sign (+1/-1 with
    equal probability). The peaked counterpart to :class:`SignedLogUniformPrior`
    for signed scale parameters (inflation vs deflation, slip direction).
    Equivalent to ``SignedPrior(LogNormalPrior(median, sigma))``.

    Parameters
    ----------
    median, sigma : float
        Median and log-space standard deviation of the magnitude (both ``> 0``);
        see :class:`LogNormalPrior`.
    """

    __slots__ = ()

    def __init__(self, median: float, sigma: float):
        super().__init__(LogNormalPrior(median, sigma))


@dataclass(slots=True)
class PowerLawPrior(Prior):
    """Bounded power-law prior: ``pdf(x) proportional to x**-alpha`` on ``[low, high]``.

    Motivated by fractal / Gutenberg--Richter size scaling (fault dimensions,
    seismic moment), where large events are rarer than small ones by a power
    law. ``alpha=1`` recovers :class:`LogUniformPrior`; larger ``alpha`` piles
    more mass near ``low`` (steeper size--frequency falloff), ``alpha < 1`` more
    mass near ``high``. Sampled exactly by inverse-CDF.

    Parameters
    ----------
    low, high : float
        Support bounds; both must be ``> 0`` and ``high > low``.
    alpha : float
        Power-law exponent (density ``proportional to x**-alpha``).
    """

    low: float
    high: float
    alpha: float

    def __post_init__(self):
        if self.low <= 0:
            raise ValueError("low must be > 0")

        if self.high <= self.low:
            raise ValueError("high must be > low")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` from the bounded power law."""
        u = _rand(size, generator, device, dtype)
        a, b = self.low, self.high
        if abs(self.alpha - 1.0) < 1e-12:               # pdf ∝ 1/x  ->  log-uniform
            return a * (b / a) ** u
        m = 1.0 - self.alpha
        return (a ** m + u * (b ** m - a ** m)) ** (1.0 / m)


def _von_mises_centered(
        size: tuple[int, ...],
        kappa: float,
        generator: torch.Generator | None,
        device: Optional[DeviceLikeType],
        dtype: torch.dtype,
) -> Tensor:
    """Von Mises deviates about mean direction 0 (radians), Best & Fisher (1979).

    Vectorised rejection sampler that respects ``generator`` (unlike
    ``torch.distributions.VonMises``, which draws from the global RNG), so
    per-index dataset reproducibility is preserved.
    """
    n = 1
    for s in size:
        n *= s
    tau = 1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)
    rho = (tau - math.sqrt(2.0 * tau)) / (2.0 * kappa)
    r = (1.0 + rho * rho) / (2.0 * rho)

    out = torch.empty(n, device=device, dtype=dtype)
    todo = torch.arange(n, device=device)
    while todo.numel() > 0:                             # resample only the rejects
        m = int(todo.numel())
        u1 = _rand((m,), generator, device, dtype)
        u2 = _rand((m,), generator, device, dtype)
        u3 = _rand((m,), generator, device, dtype)
        z = torch.cos(math.pi * u1)
        f = (1.0 + r * z) / (r + z)
        c = kappa * (r - f)
        accept = (c * (2.0 - c) - u2 > 0) | (torch.log(c / u2) + 1.0 - c >= 0)
        theta = torch.sign(u3 - 0.5) * torch.acos(f.clamp(-1.0, 1.0))
        acc = accept.nonzero(as_tuple=False).squeeze(1)
        out[todo[acc]] = theta[acc]
        todo = todo[~accept]
    return out.reshape(size)


@dataclass(slots=True)
class VonMisesPrior(Prior):
    """Von Mises (circular Gaussian) prior for an angle.

    The natural peaked prior for a *periodic* parameter (fault strike, rake,
    pCDM azimuth): unlike a truncated Gaussian it wraps correctly, so density
    piled near 0/360 is not split by the seam. ``concentration -> 0`` is uniform
    on the circle; larger ``concentration`` clusters tightly about ``loc``.

    Parameters
    ----------
    loc : float
        Mean direction (in degrees if ``degrees`` else radians).
    concentration : float
        Concentration ``kappa >= 0`` (inverse dispersion; the circular analogue
        of ``1/std**2``).
    degrees : bool, default False
        If True, ``loc`` and the returned samples are in degrees; otherwise
        radians. Output is wrapped to ``(-180, 180]`` (``(-pi, pi]``).
    """

    loc: float
    concentration: float
    degrees: bool = False

    def __post_init__(self):
        if self.concentration < 0:
            raise ValueError("concentration must be >= 0")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw a tensor of shape ``size`` from the von Mises distribution."""
        size = tuple(size)
        loc = math.radians(self.loc) if self.degrees else float(self.loc)
        kappa = float(self.concentration)
        if kappa < 1e-8:                                # degenerate -> uniform circle
            theta = loc + math.pi * (2.0 * _rand(size, generator, device, dtype) - 1.0)
        else:
            theta = loc + _von_mises_centered(size, kappa, generator, device, dtype)
        theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi   # wrap
        if self.degrees:
            theta = theta * (180.0 / math.pi)
        return theta


@dataclass(slots=True)
class ChoicePrior(Prior):
    """Draw from a fixed finite set of scalar values with optional weights.

    Unlike :class:`MultimodalPrior` (which mixes sub-*priors*), this samples from
    an explicit list of numbers -- handy for discrete parameters, e.g.
    ``look_side`` in ``{+1, -1}`` or a small set of nominal incidence angles.

    Parameters
    ----------
    values : Sequence[float]
        The candidate values; must be non-empty.
    weights : Sequence[float], optional
        Relative (unnormalised) selection weight per value; must be strictly
        positive. Defaults to uniform.
    """

    values: Sequence[float]
    weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        if len(self.values) == 0:
            raise ValueError("values must be non-empty")
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError("weights must have one entry per value")
            if any(float(w) <= 0.0 for w in self.weights):
                raise ValueError("weights must be strictly positive")

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Draw ``size``, each element a weighted-random pick from ``values``."""
        vals = torch.tensor([float(v) for v in self.values], dtype=dtype, device=device)
        w = ([1.0] * len(self.values) if self.weights is None
             else [float(x) for x in self.weights])
        probs = torch.tensor(w, dtype=torch.float64, device=device)
        probs = probs / probs.sum()
        n = 1
        for s in size:
            n *= s
        choice = torch.multinomial(probs, n, replacement=True, generator=generator)
        return vals[choice].reshape(tuple(size))


# --------------------------------------------------------------------------- #
# Bridge: build a scalar prior from a (low, high, mode) spec
# --------------------------------------------------------------------------- #
_MODES: dict[str, type[Prior]] = {
    "uniform": UniformPrior,
    "log": LogUniformPrior,            # log-uniform (mass near low)
    "reverse_log": ReverseLogUniformPrior,  # log-uniform mirrored (mass near high)
    "signed_log": SignedLogUniformPrior,   # log-uniform magnitude, random sign
}


def make_prior(low: float, high: float, mode: str = "uniform") -> Prior:
    """Build a scalar :class:`Prior` from a ``(low, high, mode)`` spec.

    Bridges plain configuration (e.g. parsed from a YAML/JSON file) onto the
    typed prior classes, so external configs need not import the classes
    directly.

    Parameters
    ----------
    low, high : float
        Bounds passed to the chosen prior.
    mode : {'uniform', 'log', 'reverse_log', 'signed_log'}, default 'uniform'
        Which prior to construct. Only the ``(low, high)`` families are reachable
        here; peaked/shape priors (normal, log-normal, power-law, von Mises,
        choice) take extra parameters and are constructed directly.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised.
    """
    try:
        cls = _MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown prior mode {mode!r} (use {', '.join(map(repr, _MODES))})"
        ) from None
    return cls(low, high)


# --------------------------------------------------------------------------- #
# Prior bundles: typed bundles of named per-parameter priors
# --------------------------------------------------------------------------- #
class PriorBundle:
    """Base class for a bundle of named per-parameter :class:`Prior` objects.

    Subclasses are dataclasses whose fields are individual :class:`Prior`
    instances, one per parameter, named exactly as the consuming function's
    arguments (so ``f(**bundle.sample(size))`` works -- a source model's
    ``forward`` for a :class:`MogiPrior`, ``los_vector`` for a
    :class:`GeometryPrior`, ...). :meth:`sample` draws every ``Prior`` field at
    once and returns them keyed by field name, so adding or renaming a parameter
    is a single edit.

    Sampling *weights* (how often to pick this bundle relative to others) are
    deliberately not stored here -- they are a property of a particular dataset's
    mixture, not of the bundle itself, and live on :class:`PriorMixture`.

    ``SourcePrior`` is a backwards-compatible alias for this class.
    """

    __slots__ = ()

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> dict[str, Tensor]:
        """Sample every :class:`Prior`-typed field into a ``{name: tensor}`` dict.

        Non-prior fields are skipped. Also callable directly.
        """
        out: dict[str, Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Prior):
                out[f.name] = value.sample(size, generator, device, dtype)
        return out

    def __call__(self, *args, **kwargs) -> dict[str, Tensor]:
        return self.sample(*args, **kwargs)


#: Backwards-compatible alias (the bundle base used to be source-specific).
SourcePrior = PriorBundle


@dataclass(slots=True)
class MogiPrior(PriorBundle):
    """Prior over Mogi point-source parameters (fields match ``MogiSource``)."""

    depth: Prior
    delta_v: Prior


@dataclass(slots=True)
class PennyPrior(PriorBundle):
    """Prior over penny-shaped crack parameters (fields match ``PennySource``)."""

    depth: Prior
    radius: Prior
    pressure: Prior


@dataclass(slots=True)
class OkadaPrior(PriorBundle):
    """Prior over rectangular-dislocation (Okada) fault parameters.

    Sampled in a geophysical, constraint-friendly parametrisation rather than the
    raw ``OkadaSource`` inputs: angles in degrees, ``slip``/``opening`` in metres,
    and ``top_depth`` (depth of the shallow fault edge) so that ``top_depth >= 0``
    trivially keeps the fault below the surface. Map to ``OkadaSource.forward``
    kwargs with :func:`~torchdeform.sources.okada.okada_params_from_fault`.

    The ``earthquake`` / ``dyke`` / ``sill`` defaults are just ``OkadaPrior``
    presets with appropriate ranges (and the irrelevant slip/opening pinned to
    zero via :class:`ConstantPrior`), not separate types.

    Fields
    ------
    strike, dip, rake : Prior
        Orientation and slip-rake angles in degrees.
    slip : Prior
        Shear slip magnitude (m); split into strike/dip slip via ``rake``.
    opening : Prior
        Tensile opening (m).
    top_depth : Prior
        Depth of the up-dip fault edge (m, positive down).
    length, width : Prior
        Along-strike and down-dip fault dimensions (m).
    """

    strike: Prior
    dip: Prior
    rake: Prior
    slip: Prior
    opening: Prior
    top_depth: Prior
    length: Prior
    width: Prior


@dataclass(slots=True)
class PCDMPrior(PriorBundle):
    """Prior over point compound dislocation model (pCDM) parameters.

    Fields match :class:`~torchdeform.sources.PCDMSource` (minus the location):
    ``depth``, the orientation angles ``omega_x/y/z`` (radians), and the three
    potencies ``dv_x/y/z`` (m^3).

    The potency priors should produce **positive magnitudes**; ``PCDMSource``
    requires the three potencies to share a sign, which this prior enforces by
    drawing one random sign per item and applying it to all three (so you get
    both inflation and deflation). Set ``signed=False`` to keep them positive
    (inflation only).
    """

    depth: Prior
    omega_x: Prior
    omega_y: Prior
    omega_z: Prior
    dv_x: Prior
    dv_y: Prior
    dv_z: Prior
    signed: bool = True

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> dict[str, Tensor]:
        """Sample the pCDM parameters, applying a shared sign to the potencies.

        See :class:`SourcePrior.sample`; additionally, when ``signed`` is True a
        single ``+/-1`` is drawn per item and multiplied into ``dv_x/y/z`` so the
        three potencies always share a sign.
        """
        # NB: explicit base call -- zero-arg super() breaks under @dataclass(slots=True)
        out = PriorBundle.sample(self, size, generator, device, dtype)
        if self.signed:
            u = torch.rand(size, generator=generator, device=device, dtype=dtype)
            sign = 1.0 - 2.0 * (u < 0.5).to(dtype)      # +/-1, shared by the 3 potencies
            for key in ("dv_x", "dv_y", "dv_z"):
                out[key] = out[key] * sign
        return out


@dataclass(slots=True)
class CDMPrior(PriorBundle):
    """Prior over finite Compound Dislocation Model (CDM) parameters by style.

    Samples a magmatic-style shape -- ``depth``, an in-plane ``radius``, an
    ``aspect`` ratio, the potency ``dv`` (m^3), and the orientation angles
    ``omega_x`` (tilt) and ``omega_z`` (azimuth) -- and maps it to the raw
    :class:`~torchdeform.sources.CDMSource` inputs with
    :func:`cdm_params_from_shape`, so ``CDMSource(**prior.sample(size),
    source_x=..., source_y=..., x_obs=..., y_obs=...)`` works (``to_forward`` is
    not needed; the keys already match ``forward``).

    The ``style`` selects how the three semi-axes are built from ``radius`` /
    ``aspect`` (see :func:`cdm_params_from_shape`); the standard
    dyke / sill / sphere / prolate / oblate presets are
    :data:`DEFAULT_CDM_PRIORS`. ``dv`` priors should produce **positive
    magnitudes**; with ``signed=True`` (default) a single ``+/-1`` is drawn per
    item and applied to ``dv`` so you get both inflation and deflation.

    Notes
    -----
    The presets also carry physical validity constraints from the volcano-geodesy
    literature -- a compactness bound ``radius/depth < 0.35``, a dyke aspect bound
    ``a_y/a_z > 0.25`` (Kavanagh & Sparks 2011; Krumbholz et al. 2014), and a
    ``dv/V`` chamber-compressibility guideline (Anderson & Segall 2011; Heap et al.
    2020). These couple parameters,
    so rather than "zero the displacement when violated" they are honoured here by
    *choosing default ranges that satisfy them* (see :data:`DEFAULT_CDM_PRIORS`);
    tighten the ranges, or post-filter samples, if you need hard enforcement.

    The same set of magmatic styles, with these constraints, is used by Ireland
    et al. (2026) [doi:10.22541/essoar.15001947/v1]; this is an independent
    implementation from the underlying source geometry and that literature.

    Fields
    ------
    depth, radius, aspect, dv, omega_x, omega_z : Prior
        The shape parameters (see :func:`cdm_params_from_shape`).
    style : str
        Magmatic style, one of :data:`CDM_STYLES`.
    signed : bool
        Apply a shared random sign to ``dv`` (inflation/deflation).
    flat_axis_ratio : float
        Thin-axis fraction for the dyke/sill degenerate semi-axis.
    """

    depth: Prior
    radius: Prior
    aspect: Prior
    dv: Prior
    omega_x: Prior
    omega_z: Prior
    style: str = "sphere"
    signed: bool = True
    flat_axis_ratio: float = FLAT_AXIS_RATIO

    def sample(
            self,
            size: Sequence[int],
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> dict[str, Tensor]:
        """Sample the shape parameters and map them to ``CDMSource`` kwargs.

        Draws ``depth/radius/aspect/dv/omega_x/omega_z``, applies a shared
        ``+/-1`` to ``dv`` when ``signed``, then returns
        :func:`cdm_params_from_shape` for ``style``.
        """
        if self.style not in CDM_STYLES:
            raise ValueError(
                f"unknown CDM style {self.style!r}; expected one of {CDM_STYLES}")
        # NB: explicit base call -- zero-arg super() breaks under @dataclass(slots=True)
        out = PriorBundle.sample(self, size, generator, device, dtype)
        dv = out["dv"]
        if self.signed:
            u = torch.rand(size, generator=generator, device=device, dtype=dtype)
            sign = 1.0 - 2.0 * (u < 0.5).to(dtype)
            dv = dv * sign
        return cdm_params_from_shape(
            self.style, depth=out["depth"], radius=out["radius"],
            aspect=out["aspect"], dv=dv, omega_x=out["omega_x"],
            omega_z=out["omega_z"], flat_axis_ratio=self.flat_axis_ratio)


@dataclass(slots=True)
class GeometryPrior(PriorBundle):
    """Prior over acquisition geometry (fields match :func:`los_vector`).

    Samples ``heading_deg`` and ``incidence_deg`` (degrees) plus ``look_side``
    (a ``[B]`` tensor: ``+1`` right-looking, ``-1`` left-looking), so
    ``los_vector(**prior.sample(size))`` works directly.

    ``look_side`` defaults to a constant ``+1`` (the usual single-sensor case);
    pass ``ConstantPrior(-1.0)`` for a left-looking sensor. A bimodal heading
    (e.g. ascending vs descending) is expressed with :class:`MultimodalPrior`;
    when several genuinely different platforms are mixed, use one ``GeometryPrior``
    per platform and combine them with :class:`PriorMixture`.

    Fields
    ------
    heading_deg : Prior
        Flight azimuth (degrees CW from North).
    incidence_deg : Prior
        Radar incidence angle (degrees from vertical).
    look_side : Prior
        Look side, ``+1``/``-1`` (default ``ConstantPrior(1.0)``).
    """

    heading_deg: Prior
    incidence_deg: Prior
    look_side: Prior = field(default_factory=lambda: ConstantPrior(1.0))


@dataclass(slots=True)
class LocationPrior(PriorBundle):
    """Prior over a source's map location.

    ``sample`` returns ``{"source_x": [B], "source_y": [B]}`` (metres), matching
    the ``source_x``/``source_y`` arguments of the source models. Typically a
    uniform jitter about the scene centre.
    """

    source_x: Prior
    source_y: Prior


# --------------------------------------------------------------------------- #
# Default priors (angles in degrees, lengths/depths in metres, slip/opening in
# metres). One instance per source type, collected in DEFAULT_PRIORS. The Okada
# defaults are opinionated OkadaPrior presets, not distinct types.
# --------------------------------------------------------------------------- #
DEFAULT_EARTHQUAKE_PRIOR = OkadaPrior(
    strike=UniformPrior(0.0, 360.0),
    dip=UniformPrior(30.0, 90.0),
    rake=UniformPrior(-180.0, 180.0),
    slip=UniformPrior(0.1, 5.0),
    opening=ConstantPrior(0.0),          # pure shear
    top_depth=UniformPrior(1_000.0, 6_000.0),
    length=LogUniformPrior(1_000.0, 30_000.0),
    width=UniformPrior(1_000.0, 9_000.0),
)

DEFAULT_DYKE_PRIOR = OkadaPrior(
    strike=UniformPrior(0.0, 360.0),
    dip=UniformPrior(75.0, 90.0),
    rake=ConstantPrior(0.0),
    slip=ConstantPrior(0.0),             # pure opening
    opening=UniformPrior(0.1, 5.0),
    top_depth=UniformPrior(500.0, 3_500.0),
    length=LogUniformPrior(1_000.0, 30_000.0),
    width=UniformPrior(1_000.0, 9_000.0),   # down-dip ("height")
)

DEFAULT_SILL_PRIOR = OkadaPrior(
    strike=UniformPrior(0.0, 180.0),     # 180 deg symmetry: no need to sample 0-360
    dip=UniformPrior(0.0, 10.0),
    rake=ConstantPrior(0.0),
    slip=ConstantPrior(0.0),             # pure opening
    opening=UniformPrior(0.1, 5.0),
    top_depth=UniformPrior(1_000.0, 9_000.0),
    length=UniformPrior(500.0, 8_000.0),
    width=UniformPrior(500.0, 5_000.0),
)

DEFAULT_MOGI_PRIOR = MogiPrior(
    depth=LogUniformPrior(1_000.0, 20_000.0),
    delta_v=SignedLogUniformPrior(1e5, 1e8),
)

DEFAULT_PENNY_PRIOR = PennyPrior(
    depth=LogUniformPrior(1_000.0, 20_000.0),
    radius=UniformPrior(1_000.0, 7_000.0),
    pressure=SignedLogUniformPrior(1e5, 1e7),
)

DEFAULT_PCDM_PRIOR = PCDMPrior(
    depth=LogUniformPrior(1_000.0, 20_000.0),
    omega_x=UniformPrior(-math.pi / 2, math.pi / 2),   # tilt about X (rad)
    omega_y=UniformPrior(-math.pi / 2, math.pi / 2),   # tilt about Y (rad)
    omega_z=UniformPrior(0.0, 2.0 * math.pi),          # azimuth about Z (rad)
    dv_x=LogUniformPrior(1e5, 1e8),                     # positive magnitudes;
    dv_y=LogUniformPrior(1e5, 1e8),                     # a shared random sign is
    dv_z=LogUniformPrior(1e5, 1e8),                     # applied (signed=True)
)

#: Default prior per source type, keyed by name.
DEFAULT_PRIORS: dict[str, PriorBundle] = {
    "earthquake": DEFAULT_EARTHQUAKE_PRIOR,
    "dyke": DEFAULT_DYKE_PRIOR,
    "sill": DEFAULT_SILL_PRIOR,
    "mogi": DEFAULT_MOGI_PRIOR,
    "penny": DEFAULT_PENNY_PRIOR,
    "pcdm": DEFAULT_PCDM_PRIOR,
}

# --------------------------------------------------------------------------- #
# Finite-CDM presets, one per standard volcano-geodesy magmatic style. Ranges are
# chosen so the validity constraints from the literature hold by construction:
# compact sources keep radius/depth < 0.35; sheets stay submerged (vertical extent
# < min depth) and within the radius/depth artefact guideline; the dyke aspect
# respects a_y/a_z > 0.25 (Kavanagh & Sparks 2011; Krumbholz et al. 2014). dv
# ranges follow the dv/V chamber-compressibility guideline (Anderson & Segall
# 2011; Heap et al. 2020) for
# typical sizes (not hard-enforced for the smallest cavities). The same style
# family is used for InSAR inversion by Ireland et al. (2026); this is an
# independent implementation. Drive them with CDMSource; angles in radians,
# lengths in m, dv (potency) in m^3.
# --------------------------------------------------------------------------- #
DEFAULT_CDM_SPHERE_PRIOR = CDMPrior(
    depth=LogUniformPrior(3_000.0, 15_000.0),
    radius=UniformPrior(200.0, 1_000.0),     # radius/depth <= 1000/3000 < 0.35
    aspect=ConstantPrior(1.0),               # unused (isotropic)
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=ConstantPrior(0.0),              # orientation irrelevant for a sphere
    omega_z=ConstantPrior(0.0),
    style="sphere",
)

DEFAULT_CDM_PROLATE_PRIOR = CDMPrior(
    depth=LogUniformPrior(3_000.0, 15_000.0),
    radius=UniformPrior(200.0, 1_000.0),     # a_z = radius; radius/depth < 0.35
    aspect=UniformPrior(0.2, 0.9),           # a_x = a_y = radius*aspect (equatorial < polar)
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=UniformPrior(-math.pi / 2, math.pi / 2),
    omega_z=UniformPrior(0.0, 2.0 * math.pi),
    style="prolate",
)

DEFAULT_CDM_OBLATE_PRIOR = CDMPrior(
    depth=LogUniformPrior(3_000.0, 15_000.0),
    radius=UniformPrior(200.0, 1_000.0),     # a_x = a_y = radius
    aspect=UniformPrior(0.1, 0.8),           # a_z = radius*aspect (flattened)
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=UniformPrior(-math.pi / 2, math.pi / 2),
    omega_z=UniformPrior(0.0, 2.0 * math.pi),
    style="oblate",
)

DEFAULT_CDM_DYKE_PRIOR = CDMPrior(
    depth=LogUniformPrior(5_000.0, 15_000.0),
    radius=UniformPrior(500.0, 3_000.0),     # a_y (along-strike half-length)
    aspect=UniformPrior(0.3, 1.5),           # a_z/a_y: vert extent <= 4500 < min depth; a_y/a_z > 0.25
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=ConstantPrior(0.0),              # vertical sheet
    omega_z=UniformPrior(0.0, math.pi),      # strike (pi symmetry)
    style="dyke",
)

DEFAULT_CDM_SILL_PRIOR = CDMPrior(
    depth=LogUniformPrior(5_000.0, 15_000.0),
    radius=UniformPrior(500.0, 3_000.0),     # a_x
    aspect=UniformPrior(0.5, 1.5),           # a_y/a_x: in-plane extent <= 4500 < min depth
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=ConstantPrior(0.0),              # horizontal sheet
    omega_z=UniformPrior(0.0, math.pi),
    style="sill",
)

DEFAULT_CDM_SILL_SYMMETRIC_PRIOR = CDMPrior(
    depth=LogUniformPrior(5_000.0, 15_000.0),
    radius=UniformPrior(500.0, 3_000.0),
    aspect=ConstantPrior(1.0),               # circular horizontal sheet
    dv=LogUniformPrior(1e5, 1e7),
    omega_x=ConstantPrior(0.0),
    omega_z=ConstantPrior(0.0),
    style="sill",
)

#: Default finite-CDM prior per magmatic style, keyed by name. The
#: ``sphere`` preset covers both the symmetric and "simple" sphere setups (a
#: sphere is isotropic, so orientation is irrelevant).
DEFAULT_CDM_PRIORS: dict[str, CDMPrior] = {
    "dyke": DEFAULT_CDM_DYKE_PRIOR,
    "sill": DEFAULT_CDM_SILL_PRIOR,
    "sill_symmetric": DEFAULT_CDM_SILL_SYMMETRIC_PRIOR,
    "sphere": DEFAULT_CDM_SPHERE_PRIOR,
    "prolate": DEFAULT_CDM_PROLATE_PRIOR,
    "oblate": DEFAULT_CDM_OBLATE_PRIOR,
}

#: Sentinel-1 IW acquisition geometry: bimodal heading (ascending/descending),
#: IW incidence range, right-looking. Plug into ``los_vector(**prior.sample(n))``.
#: The functional one-shot equivalent is
#: :func:`~torchdeform.observation.sample_s1_geometry` (returns a ``(heading,
#: incidence)`` tuple); this composable form additionally yields ``look_side``,
#: lets you reweight the ascending/descending split via the ``MultimodalPrior``
#: weights, and can be combined with :class:`PriorMixture`.
DEFAULT_S1_GEOMETRY_PRIOR = GeometryPrior(
    heading_deg=MultimodalPrior([
        UniformPrior(*S1_HEADING_ASCENDING_DEG),
        UniformPrior(*S1_HEADING_DESCENDING_DEG),
    ]),
    incidence_deg=UniformPrior(*S1_INCIDENCE_RANGE_DEG),
    look_side=ConstantPrior(float(S1_LOOK_SIDE)),
)

#: Single-pass Sentinel-1 geometry priors (right-looking): the ascending and
#: descending headings split out into separate unimodal :class:`GeometryPrior`
#: instances. Unlike :data:`DEFAULT_S1_GEOMETRY_PRIOR` (one bimodal prior that
#: samples *either* pass per item), these name each pass, so they can be combined
#: into a multi-track set that observes the *same* deformation from *both*
#: geometries -- e.g. ``MultiGeometryGenerator({"asc": DEFAULT_S1_ASCENDING_PRIOR,
#: "desc": DEFAULT_S1_DESCENDING_PRIOR})``.
DEFAULT_S1_ASCENDING_PRIOR = GeometryPrior(
    heading_deg=UniformPrior(*S1_HEADING_ASCENDING_DEG),
    incidence_deg=UniformPrior(*S1_INCIDENCE_RANGE_DEG),
    look_side=ConstantPrior(float(S1_LOOK_SIDE)),
)

DEFAULT_S1_DESCENDING_PRIOR = GeometryPrior(
    heading_deg=UniformPrior(*S1_HEADING_DESCENDING_DEG),
    incidence_deg=UniformPrior(*S1_INCIDENCE_RANGE_DEG),
    look_side=ConstantPrior(float(S1_LOOK_SIDE)),
)

#: Sentinel-1 ascending + descending as a named set, ready for
#: :class:`~torchdeform.simulation.MultiGeometryGenerator`.
DEFAULT_S1_ASC_DESC_PRIORS: dict[str, GeometryPrior] = {
    "asc": DEFAULT_S1_ASCENDING_PRIOR,
    "desc": DEFAULT_S1_DESCENDING_PRIOR,
}


# --------------------------------------------------------------------------- #
# Mixture: weighted choice over source types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class MixtureSample:
    """Result of :meth:`SourceMixture.sample` for a batch.

    Attributes
    ----------
    types : list[str]
        The source type chosen for each of the ``B`` batch items, in order.
    params : dict[str, dict[str, Tensor]]
        For each source type that occurs in the batch, the sampled parameters
        for its items, keyed ``type -> {param_name: tensor[n_type]}``.
    index : dict[str, Tensor]
        For each source type that occurs, a ``LongTensor`` of the batch indices
        of its items (aligned with ``params[type]``), so results can be scattered
        back into a ``[B, ...]`` tensor.
    """

    types: list[str]
    params: dict[str, dict[str, Tensor]]
    index: dict[str, Tensor]


@dataclass
class PriorMixture:
    """Weighted mixture over named :class:`PriorBundle` objects.

    Models a dataset's choice of which bundle to draw: each item picks a key from
    a categorical distribution given by ``weights`` (normalised internally), then
    that bundle's parameters are sampled. The same :class:`PriorBundle` can appear
    in several mixtures with different weights -- which is exactly why the weight
    lives here and not on the bundle. Works for any bundle (source types via
    :class:`MogiPrior`/..., acquisition geometry via :class:`GeometryPrior`, ...).

    ``SourceMixture`` is a backwards-compatible alias for this class.

    Parameters
    ----------
    priors : dict[str, PriorBundle]
        Prior bundles keyed by name (e.g. ``"mogi"``, ``"asc"``).
    weights : dict[str, float], optional
        Relative (unnormalised) sampling weight per key; must use the same keys
        as ``priors`` and be strictly positive. Defaults to uniform.
    """

    priors: dict[str, PriorBundle]
    weights: Optional[dict[str, float]] = None

    _names: tuple[str, ...] = field(init=False, repr=False)
    _probs: Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if not self.priors:
            raise ValueError("priors must be a non-empty mapping")
        names = tuple(self.priors)

        if self.weights is None:
            w = [1.0] * len(names)
        else:
            if set(self.weights) != set(names):
                raise ValueError(
                    "weights keys must match priors keys: "
                    f"{sorted(self.weights)} vs {sorted(names)}"
                )
            w = [float(self.weights[n]) for n in names]
            if any(x <= 0.0 for x in w):
                raise ValueError("weights must be strictly positive")

        total = math.fsum(w)
        self._names = names
        self._probs = torch.tensor([x / total for x in w], dtype=torch.float64)

    @property
    def names(self) -> tuple[str, ...]:
        """Source type names, in a fixed order aligned with :attr:`probabilities`."""
        return self._names

    @property
    def probabilities(self) -> Tensor:
        """Normalised selection probabilities, aligned with :attr:`names`."""
        return self._probs

    def sample_types(
            self,
            batch: int,
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
    ) -> list[str]:
        """Draw a source type name for each of ``batch`` items (weighted)."""
        probs = self._probs.to(device=device) if device is not None else self._probs
        idx = torch.multinomial(probs, batch, replacement=True, generator=generator)
        return [self._names[i] for i in idx.tolist()]

    def sample(
            self,
            batch: int,
            generator: torch.Generator | None = None,
            device: Optional[DeviceLikeType] = None,
            dtype: torch.dtype = torch.float64,
    ) -> MixtureSample:
        """Draw a type per item, then that type's parameters, grouped by type.

        Items are grouped so each prior is sampled once for all its items. See
        :class:`MixtureSample` for the returned structure.
        """
        probs = self._probs.to(device=device) if device is not None else self._probs
        choice = torch.multinomial(probs, batch, replacement=True, generator=generator)
        types = [self._names[i] for i in choice.tolist()]

        params: dict[str, dict[str, Tensor]] = {}
        index: dict[str, Tensor] = {}
        for k, name in enumerate(self._names):
            sel = (choice == k).nonzero(as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                continue
            index[name] = sel
            params[name] = self.priors[name].sample(
                (int(sel.numel()),), generator=generator, device=device, dtype=dtype
            )
        return MixtureSample(types=types, params=params, index=index)


#: Backwards-compatible alias (the mixture used to be source-specific).
SourceMixture = PriorMixture