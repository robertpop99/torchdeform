"""
End-to-end differentiability of the full synthesis pipeline.

The individual building blocks (sources, LOS, phase, atmosphere) each have their
own gradient tests; this module checks that gradients survive when they are
*composed* the way a real training setup would use them:

    source params -> Displacement -> LOS projection -> interferometric phase
                  -> (+ turbulent & stratified atmosphere) -> wrapped-phase loss

Two flavours:

* a deterministic sub-pipeline (source -> LOS -> phase -> loss) run through
  ``torch.autograd.gradcheck``, and
* the complete pipeline including a (deterministic, fixed-noise) atmospheric
  screen, checked for finite, non-zero gradients to both the source parameters
  and the atmosphere coefficient.

Run with::

    pytest test_pipeline_gradients.py -v
"""
import pytest
import torch

from torchdeform import MogiSource, Displacement, los_vector
from torchdeform.observation.insar import to_phase, wrapped_phase_loss
from torchdeform.atmosphere import turbulent_aps, stratified_aps
from torchdeform.simulation import synthetic_dem


DTYPE = torch.float64


def _obs_grid(B, rows, cols, *, extent=20_000.0):
    """Flattened [B, N] east/north observation grid centred on the origin."""
    ax = torch.linspace(-extent / 2, extent / 2, cols, dtype=DTYPE)
    ay = torch.linspace(-extent / 2, extent / 2, rows, dtype=DTYPE)
    yy, xx = torch.meshgrid(ay, ax, indexing="ij")
    x = xx.reshape(1, -1).expand(B, -1).contiguous()
    y = yy.reshape(1, -1).expand(B, -1).contiguous()
    return x, y


def _mogi_phase(model, x_obs, y_obs, sx, sy, depth, dv):
    disp = model(x_obs, y_obs, sx, sy, depth, dv)
    heading = torch.full_like(sx, -13.0)
    incidence = torch.full_like(sx, 39.0)
    los = los_vector(heading, incidence)
    return to_phase(disp.to_los(los))   # [B, N]


# --------------------------------------------------------------------------- #
# Deterministic sub-pipeline: gradcheck through to the loss
# --------------------------------------------------------------------------- #
def test_gradcheck_source_to_phase_loss():
    model = MogiSource()
    B, rows, cols = 1, 3, 3
    x_obs, y_obs = _obs_grid(B, rows, cols)

    sx = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    sy = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    depth = torch.full((B,), 3000.0, dtype=DTYPE, requires_grad=True)
    dv = torch.full((B,), 1e6, dtype=DTYPE, requires_grad=True)

    target = torch.zeros(B, rows * cols, dtype=DTYPE)

    def f(sx_, sy_, depth_, dv_):
        phase = _mogi_phase(model, x_obs, y_obs, sx_, sy_, depth_, dv_)
        return wrapped_phase_loss(phase, target, "mean")

    assert torch.autograd.gradcheck(f, (sx, sy, depth, dv))


# --------------------------------------------------------------------------- #
# Full pipeline incl. atmosphere: finite, non-zero gradients
# --------------------------------------------------------------------------- #
def test_full_pipeline_gradients_finite_and_nonzero():
    model = MogiSource()
    B, rows, cols = 2, 8, 8
    N = rows * cols
    x_obs, y_obs = _obs_grid(B, rows, cols)

    sx = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    sy = torch.zeros(B, dtype=DTYPE, requires_grad=True)
    depth = torch.full((B,), 4000.0, dtype=DTYPE, requires_grad=True)
    dv = torch.full((B,), 2e6, dtype=DTYPE, requires_grad=True)

    phase = _mogi_phase(model, x_obs, y_obs, sx, sy, depth, dv).reshape(B, rows, cols)

    # Deterministic atmosphere: fixed white noise so the screen is reproducible
    # and the gradient w.r.t. the stratification coefficient is well defined.
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(B, rows, cols, generator=g, dtype=DTYPE)
    turb = turbulent_aps(B, rows, cols, rms=1.0, psizex=2500.0, psizey=2500.0,
                         noise=noise)

    dem = synthetic_dem(B, rows, cols, relief=800.0, generator=g)
    coeff = torch.full((B,), 5e-3, dtype=DTYPE, requires_grad=True)
    strat = stratified_aps(dem, coeff, model="linear")

    total = phase + turb + strat
    target = torch.zeros_like(total)
    loss = wrapped_phase_loss(total, target, "mean")
    loss.backward()

    for name, t in [("sx", sx), ("sy", sy), ("depth", depth), ("dv", dv),
                    ("coeff", coeff)]:
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"
        assert t.grad.abs().sum() > 0, f"zero grad for {name}"
