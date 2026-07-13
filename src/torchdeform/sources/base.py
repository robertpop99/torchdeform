"""
Abstract base class for analytic deformation source models.

Every concrete source (Mogi, Okada, penny-shaped crack, ...) subclasses
:class:`SourceModel`. They are :class:`torch.nn.Module` subclasses whose
``forward`` takes batched observation coordinates plus source parameters and
returns a :class:`~torchdeform.core.Displacement` -- the ground surface
displacement predicted by the source's closed-form (analytic, differentiable)
solution. See the individual subclasses for their specific parameter signatures.
"""
from abc import abstractmethod, ABC

import torch
from torch import Tensor, nn

from ..core import Displacement


# Library-wide default elastic medium, set here in one place so the two properties
# that define it live together; each model still exposes them as override knobs.
#   nu = 0.25 is the "Poisson solid" (lambda = mu), the standard crustal reference,
#   and the shared default of every source model.
#   mu = 30 GPa is a standard crustal rigidity (Pa); it sets the absolute
#   displacement scale for the pressure-parametrised sources (penny, pecm) -- the
#   potency/slip-parametrised ones are independent of mu, so only those two use it.
DEFAULT_POISSON_RATIO = 0.25
DEFAULT_SHEAR_MODULUS = 3e10


# Canonical numerical-guard floors (denominator/sqrt/log), by precision, and the
# single source of truth for the guard magnitude across every source model. A
# fixed 1e-12 is right for float64 but lies below float32's machine epsilon
# (~1e-7), so it is lifted for coarser dtypes (see default_num_eps). Modules that
# need a *standalone* (non-dtype-aware) default import NUM_EPS_F64 from here rather
# than re-spelling the literal.
NUM_EPS_F64 = 1e-12
NUM_EPS_F32 = 1e-6
NUM_EPS_F16 = 1e-3   # float16 / bfloat16 and anything coarser


def default_num_eps(dtype: torch.dtype) -> float:
    """Numerical guard scale (denominator/sqrt/log floor) appropriate for ``dtype``.

    The source models add a small ``num_eps`` to radii/denominators to keep the
    forward *and its gradient* finite at the deformation singularities (on the
    source, at fault edges, ...). A fixed ``1e-12`` is right for ``float64`` but
    lies *below* ``float32``'s machine epsilon (~1e-7), so in ``float32`` it
    silently vanishes and the guard stops biting -- reintroducing ``inf``/``NaN``
    (e.g. ``1/r**4`` overflow past ``3.4e38``) near the singular manifolds. Scale
    the floor with the dtype's precision so the guard keeps working.
    """
    if dtype == torch.float64:
        return NUM_EPS_F64
    if dtype == torch.float32:
        return NUM_EPS_F32
    return NUM_EPS_F16


class SourceModel(nn.Module, ABC):
    """
    Generic source displacement model.

    Conventions
    -----------
    - All distances in meters
    - Returns ENU Displacement object in meters
    """

    @staticmethod
    def _validate_inputs(
        x_obs: Tensor,
        y_obs: Tensor,
        batched: dict[str, Tensor] = {},
    ) -> int:
        """Validate observation grids and per-source parameter batch sizes.

        Gives a clear error up front instead of a confusing deep-stack broadcast
        failure later. Checks that ``x_obs``/``y_obs`` are 2-D ``[B, N]`` of equal
        shape, and that each tensor in ``batched`` has a leading dimension equal
        to ``B`` (or 1, to allow broadcasting a single source over the batch).

        Parameters
        ----------
        x_obs, y_obs : Tensor
            Observation coordinates, expected shape ``[B, N]``.
        batched : dict[str, Tensor]
            Per-source parameters keyed by name (for the error message), each
            expected to have leading dimension ``B`` or ``1``.

        Returns
        -------
        int
            The batch size ``B``.

        Raises
        ------
        ValueError
            If the observation grids are not matching 2-D tensors, or a parameter
            has an incompatible batch dimension.
        """
        if x_obs.ndim != 2 or y_obs.ndim != 2:
            raise ValueError(
                f"x_obs and y_obs must be 2-D [B, N]; got {tuple(x_obs.shape)} "
                f"and {tuple(y_obs.shape)}"
            )
        if x_obs.shape != y_obs.shape:
            raise ValueError(
                f"x_obs and y_obs must have the same shape; got "
                f"{tuple(x_obs.shape)} vs {tuple(y_obs.shape)}"
            )
        batch = x_obs.shape[0]
        for name, t in batched.items():
            if t.ndim >= 1 and t.shape[0] not in (batch, 1):
                raise ValueError(
                    f"{name} has batch dimension {t.shape[0]}, expected "
                    f"{batch} (or 1) to match x_obs"
                )
        return batch

    def _resolve_num_eps(self) -> float:
        """Concrete ``num_eps``: the user's value, or a dtype-appropriate default.

        Subclasses accept ``num_eps=None`` (the default) meaning "pick a floor
        matched to ``internal_dtype``" via :func:`default_num_eps`; an explicit
        value overrides. Resolved on each call from ``self.internal_dtype`` so it
        stays correct even if the compute dtype is changed after construction.
        """
        eps = getattr(self, "num_eps", None)
        if eps is not None:
            return eps
        return default_num_eps(getattr(self, "internal_dtype", torch.float64))

    @abstractmethod
    def forward(self, *args, **kwargs) -> Displacement:
        """Compute surface displacement for this source. See subclass signatures.

        Returns
        -------
        Displacement
            ENU displacement (metres) at the observation points.
        """
        pass