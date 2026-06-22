"""
Tests for the InSAR observation operators (``observation/insar.py``).

These ops sit between the source models and the loss, so differentiability is
the property that matters most:

* **Correctness** - ``to_phase`` / ``phase_to_los`` are exact inverses;
  ``wrap_phase`` maps into ``[-pi, pi]``; ``wrapped_phase_loss`` is zero for
  equal phase and invariant to 2*pi cycle shifts.
* **Differentiability** - gradients flow (and ``gradcheck`` passes) through the
  LOS projection, the phase conversions and the wrapped-phase loss.

Run with::

    pytest test_insar.py -v
"""
import math

import pytest
import torch

from torchdeform import Displacement, LOSVector
from torchdeform.observation.insar import (
    to_los,
    to_phase,
    phase_to_los,
    wrap_phase,
    add_wrap,
    subtract_wrap,
    phase_to_complex,
    phase_to_unit_circle,
    unit_circle_to_phase,
    wrapped_phase_loss,
    WrappedPhaseLoss,
    S1_C_BAND_WAVELENGTH,
)


DTYPE = torch.float64


def _rand_displacement(B, N, *, requires_grad=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    e, n, u = (torch.randn(B, N, generator=g, dtype=DTYPE) for _ in range(3))
    for t in (e, n, u):
        t.requires_grad_(requires_grad)
    return Displacement(e=e, n=n, u=u)


def _rand_los(B, N, *, requires_grad=False, seed=1):
    g = torch.Generator().manual_seed(seed)
    e, n, u = (torch.randn(B, N, generator=g, dtype=DTYPE) for _ in range(3))
    for t in (e, n, u):
        t.requires_grad_(requires_grad)
    return LOSVector(e=e, n=n, u=u)


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def test_to_los_matches_manual_dot_product():
    disp = _rand_displacement(2, 5)
    los = _rand_los(2, 5)
    expected = disp.e * los.e + disp.n * los.n + disp.u * los.u
    assert torch.allclose(to_los(disp, los), expected)


def test_to_los_agrees_with_methods():
    """Free function, Displacement.to_los and LOSVector.project must agree."""
    disp = _rand_displacement(3, 4)
    los = _rand_los(3, 4)
    a = to_los(disp, los)
    assert torch.allclose(a, disp.to_los(los))
    assert torch.allclose(a, los.project(disp))


def test_phase_los_round_trip():
    d = torch.randn(4, 7, dtype=DTYPE)
    recovered = phase_to_los(to_phase(d))
    assert torch.allclose(recovered, d, atol=1e-12)


def test_to_phase_sign_and_scale():
    """Default wavelength is Sentinel-1 C-band; sign follows -4*pi/lambda."""
    d = torch.tensor([1.0], dtype=DTYPE)
    phase = to_phase(d)
    assert torch.allclose(phase, torch.tensor([-4.0 * math.pi / S1_C_BAND_WAVELENGTH],
                                              dtype=DTYPE))


def test_wrap_phase_in_principal_interval():
    phase = torch.linspace(-20.0, 20.0, 200, dtype=DTYPE)
    wrapped = wrap_phase(phase)
    assert torch.all(wrapped <= math.pi + 1e-9)
    assert torch.all(wrapped >= -math.pi - 1e-9)
    # wrapping is idempotent on already-wrapped values
    assert torch.allclose(wrap_phase(wrapped), wrapped, atol=1e-12)


def test_wrap_phase_invariant_to_cycles():
    phase = torch.rand(50, dtype=DTYPE) * 2.0 * math.pi - math.pi
    shifted = phase + 2.0 * math.pi * torch.randint(-3, 4, (50,)).to(DTYPE)
    assert torch.allclose(wrap_phase(shifted), phase, atol=1e-9)


def test_add_and_subtract_wrap():
    a = torch.tensor([3.0], dtype=DTYPE)
    b = torch.tensor([2.0], dtype=DTYPE)
    assert torch.allclose(add_wrap(a, b), wrap_phase(a + b))
    assert torch.allclose(subtract_wrap(a, b), wrap_phase(a - b))


def test_phase_to_unit_circle_shape_and_values():
    phase = torch.randn(2, 8, dtype=DTYPE)
    uc = phase_to_unit_circle(phase, channel_dim=1)
    assert uc.shape == (2, 2, 8)
    assert torch.allclose(uc[:, 0], torch.cos(phase))
    assert torch.allclose(uc[:, 1], torch.sin(phase))
    # lies on the unit circle
    assert torch.allclose((uc ** 2).sum(dim=1), torch.ones(2, 8, dtype=DTYPE))


def test_unit_circle_to_phase_inverts_phase_to_unit_circle():
    phase = torch.rand(2, 8, dtype=DTYPE) * 2.0 * math.pi - math.pi   # in [-pi, pi]
    uc = phase_to_unit_circle(phase, channel_dim=1)
    assert torch.allclose(unit_circle_to_phase(uc, channel_dim=1), phase, atol=1e-12)


def test_unit_circle_to_phase_ignores_magnitude():
    """Only the direction of (cos, sin) matters, not its norm."""
    phase = torch.linspace(-3.0, 3.0, 25, dtype=DTYPE)
    uc = phase_to_unit_circle(phase, channel_dim=0)
    scaled = uc * torch.rand(25, dtype=DTYPE).clamp_min(0.1)
    assert torch.allclose(unit_circle_to_phase(scaled, channel_dim=0), phase, atol=1e-12)


def test_unit_circle_to_phase_bad_shape_raises():
    with pytest.raises(ValueError, match="size 2"):
        unit_circle_to_phase(torch.randn(2, 3, 4, dtype=DTYPE), channel_dim=1)


def test_phase_to_complex_is_wrap_invariant():
    phase = torch.randn(10, dtype=DTYPE)
    assert torch.allclose(phase_to_complex(phase),
                          phase_to_complex(phase + 2.0 * math.pi), atol=1e-12)


# --------------------------------------------------------------------------- #
# wrapped_phase_loss
# --------------------------------------------------------------------------- #
def test_wrapped_phase_loss_zero_when_equal():
    p = torch.randn(3, 6, dtype=DTYPE)
    assert wrapped_phase_loss(p, p.clone()).item() == pytest.approx(0.0, abs=1e-12)


def test_wrapped_phase_loss_cycle_invariant():
    pred = torch.randn(4, 5, dtype=DTYPE)
    target = pred + 2.0 * math.pi * torch.randint(-2, 3, (4, 5)).to(DTYPE)
    assert wrapped_phase_loss(pred, target).item() == pytest.approx(0.0, abs=1e-9)


def test_wrapped_phase_loss_reductions():
    pred = torch.randn(2, 3, dtype=DTYPE)
    target = torch.randn(2, 3, dtype=DTYPE)
    none = wrapped_phase_loss(pred, target, reduction="none")
    assert none.shape == (2, 3)
    assert wrapped_phase_loss(pred, target, "sum") == pytest.approx(float(none.sum()))
    assert wrapped_phase_loss(pred, target, "mean") == pytest.approx(float(none.mean()))


def test_wrapped_phase_loss_bad_reduction_raises():
    pred = torch.zeros(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="unknown reduction"):
        wrapped_phase_loss(pred, pred, reduction="median")


def test_wrapped_phase_loss_module_matches_function():
    pred = torch.randn(2, 4, dtype=DTYPE)
    target = torch.randn(2, 4, dtype=DTYPE)
    mod = WrappedPhaseLoss(reduction="sum")
    assert torch.allclose(mod(pred, target),
                          wrapped_phase_loss(pred, target, "sum"))


def test_wrapped_phase_loss_default_period_is_radians():
    """The default period reproduces the original radians formula."""
    pred = torch.randn(3, 5, dtype=DTYPE)
    target = torch.randn(3, 5, dtype=DTYPE)
    expected = 2 * torch.sin((pred - target) / 2) ** 2
    assert torch.allclose(
        wrapped_phase_loss(pred, target, "none"), expected, atol=1e-12)


def test_wrapped_phase_loss_normalized_period_cycle_invariant():
    """With phase normalised to [-1, 1] (period=2), a 2-unit shift is one cycle."""
    pred = torch.rand(4, 6, dtype=DTYPE) * 2.0 - 1.0          # in [-1, 1]
    target = pred + 2.0 * torch.randint(-2, 3, (4, 6)).to(DTYPE)
    assert wrapped_phase_loss(pred, target, period=2.0).item() == pytest.approx(0.0, abs=1e-9)
    # the default (radians) period would NOT see these as equal
    assert wrapped_phase_loss(pred, target) > 1e-3


def test_wrapped_phase_loss_period_matches_rescaled_radians():
    """Loss with period=2 on [-1,1] equals the radians loss on pi-scaled inputs."""
    pred = torch.rand(2, 4, dtype=DTYPE) * 2.0 - 1.0
    target = torch.rand(2, 4, dtype=DTYPE) * 2.0 - 1.0
    norm = wrapped_phase_loss(pred, target, "none", period=2.0)
    rad = wrapped_phase_loss(pred * torch.pi, target * torch.pi, "none")
    assert torch.allclose(norm, rad, atol=1e-12)


def test_wrapped_phase_loss_module_forwards_period():
    pred = torch.rand(2, 3, dtype=DTYPE) * 2.0 - 1.0
    target = torch.rand(2, 3, dtype=DTYPE) * 2.0 - 1.0
    mod = WrappedPhaseLoss(reduction="mean", period=2.0)
    assert torch.allclose(mod(pred, target),
                          wrapped_phase_loss(pred, target, "mean", period=2.0))


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
def test_gradcheck_to_los():
    disp = _rand_displacement(2, 3, requires_grad=True, seed=2)
    los = _rand_los(2, 3, requires_grad=True, seed=3)

    def f(de, dn, du, le, ln, lu):
        return to_los(Displacement(de, dn, du), LOSVector(le, ln, lu))

    assert torch.autograd.gradcheck(
        f, (disp.e, disp.n, disp.u, los.e, los.n, los.u)
    )


def test_gradcheck_to_phase_and_back():
    d = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: to_phase(x), (d,))
    p = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: phase_to_los(x), (p,))


def test_gradcheck_wrapped_phase_loss():
    pred = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    target = torch.randn(2, 3, dtype=DTYPE)
    assert torch.autograd.gradcheck(
        lambda x: wrapped_phase_loss(x, target, "mean"), (pred,)
    )


def test_gradients_finite_and_nonzero():
    disp = _rand_displacement(2, 4, requires_grad=True, seed=4)
    los = _rand_los(2, 4, seed=5)
    loss = to_phase(to_los(disp, los)).pow(2).mean()
    loss.backward()
    for t in (disp.e, disp.n, disp.u):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0
