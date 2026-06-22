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

from torch import Tensor, nn

from ..core import Displacement


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

    @abstractmethod
    def forward(self, *args, **kwargs) -> Displacement:
        """Compute surface displacement for this source. See subclass signatures.

        Returns
        -------
        Displacement
            ENU displacement (metres) at the observation points.
        """
        pass