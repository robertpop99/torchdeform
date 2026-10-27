"""
Tests for the finite-spheroid priors: ``SpheroidPrior`` and the
``DEFAULT_SPHEROID_*`` presets.

Covers:

* **Prior** - ``sample`` returns exactly the ``SpheroidSource.forward`` kwargs
  (minus the location) and is reproducible.
* **Presets** - each keeps its shape class (prolate ``a > b`` / oblate
  ``a < b``) and the cavity buried at any orientation, drives ``SpheroidSource``
  to a finite field, and round-trips through ``SourceGenerator``.

Run with::

    pytest test_spheroid_prior.py -v
"""
import pytest
import torch

from torchdeform import SpheroidSource
from torchdeform.simulation import (
    SpheroidPrior,
    SourceGenerator,
    DEFAULT_SPHEROID_PRIORS,
)

DTYPE = torch.float64
FORWARD_KEYS = {"depth", "polar_radius", "equatorial_radius", "dip", "strike",
                "strength"}


def _grid(B, n=9, extent=40_000.0):
    ax = torch.linspace(-extent / 2, extent / 2, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    return (xx.reshape(1, -1).expand(B, -1).contiguous(),
            yy.reshape(1, -1).expand(B, -1).contiguous())


def test_sample_keys_match_forward():
    out = DEFAULT_SPHEROID_PRIORS["prolate"].sample((4,))
    assert set(out) == FORWARD_KEYS
    assert isinstance(DEFAULT_SPHEROID_PRIORS["prolate"], SpheroidPrior)


def test_sampling_reproducible():
    p = DEFAULT_SPHEROID_PRIORS["oblate"]
    a = p.sample((16,), generator=torch.Generator().manual_seed(3))
    b = p.sample((16,), generator=torch.Generator().manual_seed(3))
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


@pytest.mark.parametrize("shape", list(DEFAULT_SPHEROID_PRIORS))
def test_preset_shape_class_buried_and_finite(shape):
    B = 256
    out = DEFAULT_SPHEROID_PRIORS[shape].sample((B,), generator=torch.Generator().manual_seed(0))
    a, b = out["polar_radius"], out["equatorial_radius"]
    assert ((a > b) if shape == "prolate" else (a < b)).all()
    # buried at any orientation, within the compactness bound
    assert (torch.maximum(a, b) < 0.35 * out["depth"] + 1e-9).all()
    assert (out["strength"] > 0).any() and (out["strength"] < 0).any()
    x, y = _grid(B)
    z0 = torch.zeros(B, dtype=DTYPE)
    d = SpheroidSource()(x, y, z0, z0, **out)
    for t in (d.e, d.n, d.u):
        assert torch.isfinite(t).all()


def test_round_trip_through_source_generator():
    B = 8
    x, y = _grid(B)
    gen = SourceGenerator(model=SpheroidSource(), prior=DEFAULT_SPHEROID_PRIORS["prolate"])
    z0 = torch.zeros(B, dtype=DTYPE)
    disp, params = gen.generate(x, y, z0, z0, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(disp.u).all()
    assert set(params) == FORWARD_KEYS
