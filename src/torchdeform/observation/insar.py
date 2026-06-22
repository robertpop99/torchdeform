"""
InSAR observation operators: displacement <-> LOS <-> interferometric phase.

Differentiable helpers that turn a 3-D ground :class:`~torchdeform.core.Displacement`
into the quantity a radar interferogram actually measures and back again:

* projection onto the line of sight (:func:`to_los`),
* the LOS-displacement <-> phase relation (:func:`to_phase`, :func:`phase_to_los`),
* phase wrapping / wrapped arithmetic (:func:`wrap_phase`, :func:`add_wrap`,
  :func:`subtract_wrap`), and
* representations and losses for learning on wrapped phase
  (:func:`phase_to_complex`, :func:`phase_to_unit_circle` /
  :func:`unit_circle_to_phase`, :func:`wrapped_phase_loss` /
  :class:`WrappedPhaseLoss`).

Phase sign convention
---------------------
``phase = -4*pi/lambda * d_los`` where ``d_los`` is LOS displacement (positive
toward the satellite, see :mod:`torchdeform.observation.los`). The default
``wavelength`` is the Sentinel-1 C-band value, ``S1_C_BAND_WAVELENGTH``.
"""
import torch
from torch import Tensor, nn

from ..core import Displacement, LOSVector


S1_C_BAND_WAVELENGTH = 0.0555   # Sentinel-1 C-band radar wavelength (metres)


def to_los(
    displacement: Displacement,
    los: LOSVector,
) -> Tensor:
    """Project a displacement field onto a line-of-sight vector.

    Thin functional wrapper over the canonical :meth:`Displacement.to_los`;
    returns the scalar LOS displacement ``e*los.e + n*los.n + u*los.u`` [B, N]
    (positive toward the satellite).
    """
    return displacement.to_los(los)


def to_phase(
    los_displacement: Tensor,
    wavelength: float | Tensor = S1_C_BAND_WAVELENGTH,
) -> Tensor:
    """Convert LOS displacement (metres) to interferometric phase (radians).

    Applies ``phase = -4*pi * d_los / wavelength``. The result is unwrapped
    phase; use :func:`wrap_phase` to wrap it to ``[-pi, pi]``.
    """
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
    """Inverse of :func:`to_phase`: convert phase (radians) to LOS metres."""
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
        Wrapped phase difference [rad].
    """
    return wrap_phase(phase1 - phase2)


def phase_to_complex(phase: Tensor) -> Tensor:
    """Map phase to the complex unit circle, ``exp(i * phase)``.

    Returns a complex tensor; the result is invariant to 2*pi wrapping of the
    input, which makes it a convenient wrap-agnostic representation.
    """
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


def unit_circle_to_phase(
    unit_circle: Tensor,
    channel_dim: int = 1,
) -> Tensor:
    """Recover wrapped phase from a ``(cos, sin)`` unit-circle representation.

    Inverse of :func:`phase_to_unit_circle`: takes a tensor with a length-2 axis
    at ``channel_dim`` holding ``(cos(phase), sin(phase))`` and returns the
    wrapped phase in ``[-pi, pi]`` via ``atan2(sin, cos)``. The magnitude of the
    input is irrelevant (only its direction), so it also works on the raw,
    un-normalised two-channel output of a network.

    Parameters
    ----------
    unit_circle : Tensor
        Tensor with ``unit_circle.shape[channel_dim] == 2``; channel 0 is the
        cosine component, channel 1 the sine component.
    channel_dim : int, default 1
        Axis holding the two components.

    Returns
    -------
    Tensor
        Wrapped phase [rad], with the channel axis removed.
    """
    if unit_circle.shape[channel_dim] != 2:
        raise ValueError(
            f"expected size 2 along channel_dim={channel_dim}, got "
            f"{unit_circle.shape[channel_dim]}"
        )
    cos = unit_circle.select(channel_dim, 0)
    sin = unit_circle.select(channel_dim, 1)
    return torch.atan2(sin, cos)


def wrapped_phase_loss(
    pred_phase: Tensor,
    target_phase: Tensor,
    reduction: str = 'mean',
    period: float = 2.0 * torch.pi,
) -> Tensor:
    """Phase-wrap-invariant loss between predicted and target phase.

    Computes ``2 * sin(pi * (pred - target) / period)**2``, equivalently
    ``1 - cos(2*pi*(pred - target)/period)``. It depends only on the phase
    difference modulo one full cycle (``period``), so it is insensitive to
    integer-cycle ambiguities. It is zero when the phases agree and smooth
    everywhere (good for gradient-based training).

    The ``period`` makes the unit explicit: use the default ``2*pi`` for phase in
    **radians**, or set it to match whatever scaling you train on -- e.g.
    ``period=2.0`` for phase normalised to ``[-1, 1]`` (one cycle spans 2 units),
    or ``period=1.0`` for phase in cycles/fringes. The inputs need not lie within
    a single period; the loss is cycle-invariant regardless.

    Parameters
    ----------
    pred_phase, target_phase : Tensor
        Predicted and target phase, broadcastable to a common shape, in the same
        unit as ``period``.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How to reduce the per-element loss.
    period : float, default ``2*pi``
        Length of one full phase cycle in the unit of the inputs.

    Returns
    -------
    Tensor
        Scalar for ``'mean'``/``'sum'``, otherwise the per-element loss.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of the accepted values.
    """
    res = 2 * torch.sin(torch.pi * (pred_phase - target_phase) / period) ** 2
    if reduction == 'mean':
        return torch.mean(res)
    elif reduction == 'sum':
        return torch.sum(res)
    elif reduction == 'none':
        return res
    else:
        raise ValueError(
            f"unknown reduction {reduction!r} (use 'mean', 'sum' or 'none')"
        )


class WrappedPhaseLoss(nn.Module):
    """:class:`~torch.nn.Module` wrapper around :func:`wrapped_phase_loss`.

    Parameters
    ----------
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Reduction applied by the underlying :func:`wrapped_phase_loss`.
    period : float, default ``2*pi``
        Length of one full phase cycle in the unit of the inputs (e.g. ``2*pi``
        for radians, ``2.0`` for phase normalised to ``[-1, 1]``).
    """

    def __init__(self, reduction='mean', period: float = 2.0 * torch.pi):
        super().__init__()
        self.reduction = reduction
        self.period = period

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Return the wrapped-phase loss between ``pred`` and ``target``.

        ``pred`` and ``target`` are in the unit implied by ``self.period``
        (radians by default).
        """
        return wrapped_phase_loss(pred, target, self.reduction, self.period)
