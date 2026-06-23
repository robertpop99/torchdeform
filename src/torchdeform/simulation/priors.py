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
  scale parameters spanning orders of magnitude.
* :class:`SignedLogUniformPrior` -- log-uniform magnitude with a random sign;
  use for signed scale parameters (e.g. inflation/deflation).
* :class:`ConstantPrior` -- always returns a fixed value (to pin a parameter).
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
class SignedLogUniformPrior(Prior):
    """Symmetric signed log-uniform prior.

    Draws a log-uniform magnitude on ``[low, high]`` and multiplies it by a
    random sign (+1/-1 with equal probability). Use for signed scale parameters
    such as volume change (inflation vs deflation) or slip direction.

    Parameters
    ----------
    low, high : float
        Magnitude bounds; both must be ``> 0`` and ``high > low``. The realised
        values fall in ``[-high, -low] u [low, high]``.
    """

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
        """Draw a tensor of shape ``size`` of signed log-uniform values.

        Also callable directly. Sign and magnitude are drawn independently.
        """
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


# --------------------------------------------------------------------------- #
# Bridge: build a scalar prior from a (low, high, mode) spec
# --------------------------------------------------------------------------- #
_MODES: dict[str, type[Prior]] = {
    "uniform": UniformPrior,
    "log": LogUniformPrior,            # log-uniform
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
    mode : {'uniform', 'log', 'signed_log'}, default 'uniform'
        Which prior to construct.

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