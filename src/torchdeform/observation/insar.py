import torch
from torch import Tensor, nn

from ..core import Displacement, LOSVector


S1_C_BAND_WAVELENGTH = 0.0555


def to_los(
    displacement: Displacement,
    los: LOSVector,
) -> Tensor:
    return (
        displacement.e * los.e
        + displacement.n * los.n
        + displacement.u * los.u
    )


def to_phase(
    los_displacement: Tensor,
    wavelength: float | Tensor = S1_C_BAND_WAVELENGTH,
) -> Tensor:
    return (
        -4.0
        * torch.pi
        * los_displacement
        / wavelength
    )


def phase_to_los(
    phase: Tensor,
    wavelength: float | Tensor = S1_C_BAND_WAVELENGTH,
) -> Tensor:
    return (
        phase
        * wavelength
        / (
            -4.0
            * torch.pi
        )
    )


def wrap_phase(phase: Tensor) -> Tensor:
    """
    Wrap phase to [-π, π].

    Parameters
    ----------
    phase : Tensor
        Wrapped or unwrapped phase [rad].

    Returns
    -------
    Tensor
        Wrapped phase [rad].
    """
    return torch.atan2(
        torch.sin(phase),
        torch.cos(phase),
    )


def add_wrap(phase1: Tensor, phase2: Tensor) -> Tensor:
    """
    Add two phase fields and wrap the result.

    Inputs may be wrapped or unwrapped.
    Output is wrapped to the principal interval [-π, π].

    Parameters
    ----------
    phase1 : Tensor
        First phase field [rad].

    phase2 : Tensor
        Second phase field [rad].

    Returns
    -------
    Tensor
        Wrapped phase sum [rad].
    """
    return wrap_phase(phase1 + phase2)


def subtract_wrap(phase1: Tensor, phase2: Tensor) -> Tensor:
    """
    Subtract two phase fields and wrap the result.

    Inputs may be wrapped or unwrapped.
    Output is wrapped to the principal interval [-π, π].

    Parameters
    ----------
    phase1 : Tensor
        First phase field [rad].

    phase2 : Tensor
        Second phase field [rad].

    Returns
    -------
    Tensor
        Wrapped phase sum [rad].
    """
    return wrap_phase(phase1 - phase2)


def phase_to_complex(phase: Tensor) -> Tensor:
    return torch.exp(1j * phase)


def phase_to_unit_circle(
    phase: Tensor,
    channel_dim: int = 1,
) -> Tensor:
    """
    Convert wrapped or unwrapped phase to
    a unit-circle representation.

    Returns
    -------
    Tensor
        [..., 2, ...]
        containing cos(phase) and sin(phase).
    """
    return torch.stack(
        (
            torch.cos(phase),
            torch.sin(phase),
        ),
        dim=channel_dim,
    )


def wrapped_phase_loss(
    pred_phase: Tensor,
    target_phase: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    res = 2 * torch.sin((pred_phase - target_phase) / 2) ** 2
    if reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)
    elif reduction == 'none':
        return res
    else:
        raise ValueError


class WrappedPhaseLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return wrapped_phase_loss(pred, target, self.reduction)
