"""
Composition layer: turn priors + models + atmosphere into batched physical output.

The primitives (source models, ``los_vector``, the atmosphere generators, ...)
live in their own modules; this module only *composes* them into higher-level
generators that a dataset can drive:

* :class:`ObservationGrid` -- the fixed East/North pixel grid (built once).
* :class:`SourceGenerator` -- one source type: prior + model (+ optional
  param->forward bridge) -> :class:`~torchdeform.core.Displacement`.
* :class:`DeformationGenerator` -- a weighted mixture of source types: pick a
  type per item, sample its parameters and a shared location, render, and
  scatter every group back into a ``[B, N]`` batch.
* :class:`GeometryGenerator` -- a :class:`GeometryPrior` (one sensor) or a
  :class:`PriorMixture` of them -> per-item :class:`~torchdeform.core.LOSVector`.
* :class:`AtmosphereGenerator` -- orbital ramp + stratified + turbulent screen.
* :class:`InterferogramGenerator` -- the full pipeline: deformation -> LOS ->
  phase (+ atmosphere) -> :class:`InterferogramSample`.

Everything stored in the samples is the *unwrapped* physical signal; the wrapped
interferogram (the observable / network input) is produced on demand via
:meth:`InterferogramSample.wrapped`, so there is never a wrapped/unwrapped
mismatch and the user picks the representation they need.

Mapping samples to an ML target space (normalisation, encodings, ...) is left to
the application -- see :mod:`torchdeform.simulation.datasets` for the thin
``Dataset`` wrappers with a ``transform`` hook.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from ..core import Displacement, LOSVector, DeviceLikeType
from ..sources.base import SourceModel
from ..observation.los import los_vector, S1_LOOK_SIDE
from ..observation.insar import to_phase, wrap_phase, S1_C_BAND_WAVELENGTH
from ..atmosphere import turbulent_aps, stratified_aps, orbital_ramp
from .priors import Prior, PriorBundle, PriorMixture, UniformPrior, LocationPrior


# --------------------------------------------------------------------------- #
# Observation grid
# --------------------------------------------------------------------------- #
@dataclass
class ObservationGrid:
    """Fixed East/North observation grid; the ``[N]`` coordinates are cached once.

    The grid is centred on ``(0, 0)``; ``psizex``/``psizey`` set the ground
    spacing. To use a different grid, build a new instance.

    Parameters
    ----------
    rows, cols : int
        Grid shape (``N = rows * cols`` observation points, row-major).
    psizex, psizey : float
        East/North pixel spacing (metres).
    device, dtype
        Placement of the cached coordinate tensors.
    """

    rows: int
    cols: int
    psizex: float = 1.0
    psizey: float = 1.0
    device: Optional[DeviceLikeType] = "cpu"
    dtype: torch.dtype = torch.float64

    _x: Tensor = field(init=False, repr=False)
    _y: Tensor = field(init=False, repr=False)

    def __post_init__(self):
        ax = (torch.arange(self.cols, device=self.device, dtype=self.dtype)
              - (self.cols - 1) / 2.0) * self.psizex
        ay = (torch.arange(self.rows, device=self.device, dtype=self.dtype)
              - (self.rows - 1) / 2.0) * self.psizey
        yy, xx = torch.meshgrid(ay, ax, indexing="ij")
        self._x = xx.reshape(-1)            # [N], row-major
        self._y = yy.reshape(-1)

    @property
    def n(self) -> int:
        """Number of observation points (``rows * cols``)."""
        return self.rows * self.cols

    @property
    def extent(self) -> tuple[float, float]:
        """Coordinate span ``(width, height)`` in metres."""
        return (self.cols - 1) * self.psizex, (self.rows - 1) * self.psizey

    def coords(self, batch: int) -> tuple[Tensor, Tensor]:
        """Return ``(x_obs, y_obs)``, each ``[batch, N]`` (a broadcast view)."""
        x = self._x.unsqueeze(0).expand(batch, -1)
        y = self._y.unsqueeze(0).expand(batch, -1)
        return x, y


def centered_location(grid: ObservationGrid, frac: float = 0.25) -> LocationPrior:
    """A :class:`LocationPrior` of uniform jitter within ``+/-frac/2`` of the grid."""
    ex, ey = grid.extent
    return LocationPrior(
        source_x=UniformPrior(-frac * ex / 2.0, frac * ex / 2.0),
        source_y=UniformPrior(-frac * ey / 2.0, frac * ey / 2.0),
    )


# --------------------------------------------------------------------------- #
# Source generation
# --------------------------------------------------------------------------- #
@dataclass
class SourceGenerator:
    """Generate deformation for a single source type (prior + model).

    Parameters
    ----------
    model : SourceModel
        The source displacement model (e.g. ``MogiSource``).
    prior : PriorBundle
        Prior over the source parameters (e.g. ``MogiPrior``).
    to_forward : callable, optional
        Maps the prior's sampled dict to the model's ``forward`` kwargs (e.g.
        ``okada_params_from_fault``). ``None`` (default) means the prior's keys
        already match ``forward`` (Mogi/Penny/pCDM).
    location_keys : tuple[str, str]
        Names of the model's source-location arguments. All built-in sources use
        the default ``("source_x", "source_y")``; override only for a custom
        model whose location arguments are named differently.
    """

    model: SourceModel
    prior: PriorBundle
    to_forward: Optional[Callable[[dict], dict]] = None
    location_keys: tuple[str, str] = ("source_x", "source_y")

    def generate(
        self,
        x_obs: Tensor,
        y_obs: Tensor,
        source_x: Tensor,
        source_y: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Displacement, dict]:
        """Sample parameters and render the displacement for ``[B, N]`` coords.

        Returns ``(displacement, params)`` where ``params`` is the dict of
        sampled physical parameters (the per-item labels; location excluded).
        """
        batch = x_obs.shape[0]
        params = self.prior.sample((batch,), generator=generator,
                                   device=x_obs.device, dtype=x_obs.dtype)
        fwd = self.to_forward(params) if self.to_forward is not None else dict(params)
        kx, ky = self.location_keys
        fwd[kx] = source_x
        fwd[ky] = source_y
        disp = self.model(x_obs, y_obs, **fwd)
        return disp, params


@dataclass
class DeformationSample:
    """Output of :meth:`DeformationGenerator.generate` for a batch.

    Attributes
    ----------
    displacement : Displacement
        ENU surface displacement ``[B, N]`` (assembled across source types).
    source_type : list[str]
        The source type chosen for each of the ``B`` items, in order.
    source_x, source_y : Tensor
        Source map location ``[B]`` (metres).
    params : dict[str, dict[str, Tensor]]
        Sampled physical parameters, grouped by source type
        (``type -> {name: [n_type]}``).
    index : dict[str, Tensor]
        Batch indices of each type's items (aligned with ``params[type]``).
    grid : ObservationGrid
        The grid the displacement was evaluated on.
    """

    displacement: Displacement
    source_type: list[str]
    source_x: Tensor
    source_y: Tensor
    params: dict[str, dict[str, Tensor]]
    index: dict[str, Tensor]
    grid: ObservationGrid


@dataclass
class DeformationGenerator:
    """Weighted mixture of source types -> per-item deformation on a fixed grid.

    Each item draws a source type (by ``weights``), a shared location (from
    ``location``), and that type's parameters; the per-type displacements are
    scattered back into one ``[B, N]`` batch.

    Parameters
    ----------
    grid : ObservationGrid
        Observation grid (cached).
    sources : dict[str, SourceGenerator]
        Source generators keyed by type name (e.g. ``"mogi"``).
    location : LocationPrior, optional
        Source-location prior; defaults to :func:`centered_location` of the grid.
    weights : dict[str, float], optional
        Relative selection weight per type (same keys as ``sources``, > 0);
        defaults to uniform.
    """

    grid: ObservationGrid
    sources: dict[str, SourceGenerator]
    location: Optional[LocationPrior] = None
    weights: Optional[dict[str, float]] = None

    _names: tuple[str, ...] = field(init=False, repr=False)
    _probs: Tensor = field(init=False, repr=False)
    _location: LocationPrior = field(init=False, repr=False)

    def __post_init__(self):
        if not self.sources:
            raise ValueError("sources must be a non-empty mapping")
        names = tuple(self.sources)
        if self.weights is None:
            w = [1.0] * len(names)
        else:
            if set(self.weights) != set(names):
                raise ValueError(
                    f"weights keys must match sources keys: "
                    f"{sorted(self.weights)} vs {sorted(names)}"
                )
            w = [float(self.weights[n]) for n in names]
            if any(x <= 0.0 for x in w):
                raise ValueError("weights must be strictly positive")
        total = sum(w)
        self._names = names
        self._probs = torch.tensor([x / total for x in w],
                                   dtype=torch.float64, device=self.grid.device)
        self._location = self.location or centered_location(self.grid)

    def generate(self, batch: int, *,
                 generator: Optional[torch.Generator] = None) -> DeformationSample:
        """Generate ``batch`` deformation fields (mixed source types)."""
        grid = self.grid
        device, dtype = grid.device, grid.dtype
        x_obs, y_obs = grid.coords(batch)

        loc = self._location.sample((batch,), generator=generator,
                                    device=device, dtype=dtype)
        source_x, source_y = loc["source_x"], loc["source_y"]

        choice = torch.multinomial(self._probs, batch, replacement=True,
                                   generator=generator)            # [B]
        types = [self._names[i] for i in choice.tolist()]

        ue = torch.zeros(batch, grid.n, device=device, dtype=dtype)
        un = torch.zeros(batch, grid.n, device=device, dtype=dtype)
        uv = torch.zeros(batch, grid.n, device=device, dtype=dtype)
        params: dict[str, dict[str, Tensor]] = {}
        index: dict[str, Tensor] = {}

        for k, name in enumerate(self._names):
            sel = (choice == k).nonzero(as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                continue
            disp_g, p_g = self.sources[name].generate(
                x_obs[sel], y_obs[sel], source_x[sel], source_y[sel],
                generator=generator,
            )
            ue[sel] = disp_g.e.to(dtype)
            un[sel] = disp_g.n.to(dtype)
            uv[sel] = disp_g.u.to(dtype)
            params[name] = p_g
            index[name] = sel

        return DeformationSample(
            displacement=Displacement(e=ue, n=un, u=uv),
            source_type=types, source_x=source_x, source_y=source_y,
            params=params, index=index, grid=grid,
        )


# --------------------------------------------------------------------------- #
# Geometry generation
# --------------------------------------------------------------------------- #
@dataclass
class GeometryGenerator:
    """Sample acquisition geometry per item and build the LOS vectors.

    ``geometry`` is a single :class:`GeometryPrior` (one sensor) or a
    :class:`PriorMixture` of them (several sensors). A mixture is flattened back
    to per-item ``[B]`` tensors -- valid because all geometry bundles share the
    same fields -- then projected with ``los_vector`` (one vector per image).
    """

    geometry: PriorBundle | PriorMixture

    def generate(
        self,
        batch: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: Optional[DeviceLikeType] = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> tuple[LOSVector, dict]:
        """Return ``(los, geometry)``: an ``[B, 1]`` LOS vector and the ``[B]`` labels."""
        if isinstance(self.geometry, PriorMixture):
            ms = self.geometry.sample(batch, generator=generator,
                                      device=device, dtype=dtype)
            keys = sorted({k for p in ms.params.values() for k in p})
            g = {k: torch.empty(batch, device=device, dtype=dtype) for k in keys}
            for name, idx in ms.index.items():
                for k in keys:
                    g[k][idx] = ms.params[name][k]
        else:
            g = self.geometry.sample((batch,), generator=generator,
                                     device=device, dtype=dtype)

        los = los_vector(
            g["heading_deg"], g["incidence_deg"],
            device=device, dtype=dtype,
            look_side=g.get("look_side", S1_LOOK_SIDE),
        )
        return los, g


# --------------------------------------------------------------------------- #
# Atmosphere generation
# --------------------------------------------------------------------------- #
@dataclass
class AtmosphereGenerator:
    """Build a ``[B, rows, cols]`` background = orbital ramp + stratified + turbulent.

    Each component is optional and enabled by giving it an RMS/coefficient
    :class:`Prior` (so its strength randomises per image). The stratified term
    additionally needs a ``dem`` source. This is the in-library canonical
    composition; for anything fancier, build your own using it as a template.

    Parameters
    ----------
    grid : ObservationGrid
        Provides ``rows``/``cols`` and pixel spacing (for the turbulent model).
    orbital_rms : Prior, optional
        Per-image RMS of the long-wavelength orbital ramp.
    orbital_order : int
        Ramp polynomial order (1 planar, 2 quadratic).
    strat_coeff : Prior, optional
        Per-image stratification coefficient (needs ``dem``).
    strat_model : {'linear', 'exponential'}
        Stratified phase-elevation relation.
    dem : callable, optional
        ``(batch, generator) -> [B, rows, cols]`` elevation source (e.g. wrapping
        ``synthetic_dem``); required for the stratified term. The ``generator`` is
        forwarded so the DEM is reproducible alongside the rest of the sample.
    turbulent_rms : Prior, optional
        Per-image RMS of the turbulent screen.
    turbulent_model : {'powerlaw', 'exponential'}
        Turbulent power-spectrum model.
    turbulent_beta : float
        Power-law slope (``turbulent_model='powerlaw'``).
    turbulent_correlation_length : float, optional
        Correlation length (``turbulent_model='exponential'``).
    """

    grid: ObservationGrid
    orbital_rms: Optional[Prior] = None
    orbital_order: int = 1
    strat_coeff: Optional[Prior] = None
    strat_model: str = "linear"
    dem: Optional[Callable[[int, Optional[torch.Generator]], Tensor]] = None
    turbulent_rms: Optional[Prior] = None
    turbulent_model: str = "powerlaw"
    turbulent_beta: float = 8.0 / 3.0
    turbulent_correlation_length: Optional[float] = None

    def generate(self, batch: int, *,
                 generator: Optional[torch.Generator] = None) -> Tensor:
        """Return the summed atmospheric screen ``[B, rows, cols]``."""
        grid = self.grid
        rows, cols = grid.rows, grid.cols
        device, dtype = grid.device, grid.dtype
        total = torch.zeros(batch, rows, cols, device=device, dtype=dtype)

        if self.orbital_rms is not None:
            rms = self.orbital_rms.sample((batch,), generator, device, dtype)
            total = total + orbital_ramp(batch, rows, cols, rms=rms,
                                         order=self.orbital_order,
                                         generator=generator, device=device, dtype=dtype)

        if self.strat_coeff is not None:
            if self.dem is None:
                raise ValueError("strat_coeff is set but no dem source was given")
            dem = self.dem(batch, generator)
            coeff = self.strat_coeff.sample((batch,), generator, device, dtype)
            total = total + stratified_aps(dem, coeff, model=self.strat_model,
                                           device=device, dtype=dtype)

        if self.turbulent_rms is not None:
            rms = self.turbulent_rms.sample((batch,), generator, device, dtype)
            total = total + turbulent_aps(
                batch, rows, cols, rms=rms, model=self.turbulent_model,
                beta=self.turbulent_beta,
                correlation_length=self.turbulent_correlation_length,
                psizex=grid.psizex, psizey=grid.psizey,
                generator=generator, device=device, dtype=dtype,
            )

        return total


# --------------------------------------------------------------------------- #
# Full interferogram pipeline
# --------------------------------------------------------------------------- #
@dataclass
class InterferogramSample:
    """Output of :meth:`InterferogramGenerator.generate` for a batch.

    All stored fields are the *unwrapped* physical signal; the observable
    (wrapped) interferogram is produced on demand by :meth:`wrapped`.

    Attributes
    ----------
    deformation_phase : Tensor
        Unwrapped deformation phase ``[B, rows, cols]`` (signal only).
    atmosphere : Tensor or None
        Unwrapped atmospheric screen ``[B, rows, cols]``, or ``None``.
    los : LOSVector
        Line-of-sight vectors ``[B, 1]``.
    geometry : dict
        Acquisition geometry labels (``heading_deg``/``incidence_deg``/
        ``look_side``, each ``[B]``).
    deformation : DeformationSample
        Source labels (type, parameters, location) and the displacement field.
    """

    deformation_phase: Tensor
    atmosphere: Optional[Tensor]
    los: LOSVector
    geometry: dict
    deformation: DeformationSample

    @property
    def phase(self) -> Tensor:
        """Total unwrapped phase = ``deformation_phase + atmosphere``."""
        if self.atmosphere is None:
            return self.deformation_phase
        return self.deformation_phase + self.atmosphere

    def wrapped(self) -> Tensor:
        """The observable interferogram: ``wrap_phase(self.phase)`` in ``[-pi, pi]``."""
        return wrap_phase(self.phase)


@dataclass
class InterferogramGenerator:
    """Full pipeline: deformation -> LOS -> phase (+ atmosphere) -> sample.

    Parameters
    ----------
    deformation : DeformationGenerator
        Produces the surface displacement (and source labels).
    geometry : GeometryGenerator
        Produces the LOS vectors.
    atmosphere : AtmosphereGenerator, optional
        Adds an atmospheric screen; ``None`` for a clean deformation phase.
    wavelength : float
        Radar wavelength for the displacement->phase conversion.
    """

    deformation: DeformationGenerator
    geometry: GeometryGenerator
    atmosphere: Optional[AtmosphereGenerator] = None
    wavelength: float = S1_C_BAND_WAVELENGTH

    def generate(self, batch: int, *,
                 generator: Optional[torch.Generator] = None) -> InterferogramSample:
        """Generate ``batch`` interferogram samples."""
        grid = self.deformation.grid
        defo = self.deformation.generate(batch, generator=generator)
        los, geo = self.geometry.generate(batch, generator=generator,
                                          device=grid.device, dtype=grid.dtype)

        d_los = defo.displacement.to_los(los)                     # [B, N]
        phase = to_phase(d_los, self.wavelength).reshape(batch, grid.rows, grid.cols)

        aps = self.atmosphere.generate(batch, generator=generator) if self.atmosphere else None

        return InterferogramSample(
            deformation_phase=phase, atmosphere=aps,
            los=los, geometry=geo, deformation=defo,
        )
