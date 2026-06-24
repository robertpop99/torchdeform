"""
Tests for the finite-CDM priors: ``cdm_params_from_shape``, ``CDMPrior`` and the
``DEFAULT_CDM_*`` magmatic-style presets.

Covers:

* **Adapter** - each style builds the semi-axes as specified and converts the
  potency ``dv`` to ``opening`` via the CDM relation; the dyke/sill degenerate
  axis is a thin (non-zero) sheet that converges as ``flat_axis_ratio -> 0``.
* **Prior** - ``sample`` returns exactly the ``CDMSource.forward`` kwargs, the
  shared ``+/-1`` sign is applied to ``dv`` only when ``signed``, sampling is
  reproducible, and a bad style is rejected.
* **Presets** - every ``DEFAULT_CDM_*`` drives ``CDMSource`` to a finite field,
  keeps the source submerged, and round-trips through ``SourceGenerator``.

Run with::

    pytest test_cdm_prior.py -v
"""
import math

import pytest
import torch

from torchdeform import CDMSource, Displacement
from torchdeform.simulation import (
    CDMPrior,
    cdm_params_from_shape,
    CDM_STYLES,
    ConstantPrior,
    LogUniformPrior,
    UniformPrior,
    LocationPrior,
    SourceGenerator,
    DEFAULT_CDM_PRIORS,
    DEFAULT_CDM_SPHERE_PRIOR,
)

DTYPE = torch.float64
FORWARD_KEYS = {"depth", "omega_x", "omega_y", "omega_z",
                "a_x", "a_y", "a_z", "opening"}


def _t(v, n=1):
    return torch.full((n,), float(v), dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Adapter: axis construction + opening
# --------------------------------------------------------------------------- #
def _shape(style, depth=8000., radius=600., aspect=0.5, dv=1e6, ox=0.0, oz=0.0,
           ratio=1e-3):
    return cdm_params_from_shape(style, _t(depth), _t(radius), _t(aspect),
                                 _t(dv), _t(ox), _t(oz), flat_axis_ratio=ratio)


def test_adapter_axis_construction():
    r, c = 600.0, 0.5
    sph = _shape("sphere", radius=r, aspect=c)
    assert sph["a_x"] == r and sph["a_y"] == r and sph["a_z"] == r   # isotropic
    pro = _shape("prolate", radius=r, aspect=c)
    assert pro["a_x"] == r * c and pro["a_y"] == r * c and pro["a_z"] == r
    obl = _shape("oblate", radius=r, aspect=c)
    assert obl["a_x"] == r and obl["a_y"] == r and obl["a_z"] == r * c
    dyk = _shape("dyke", radius=r, aspect=c, ratio=1e-3)
    assert dyk["a_y"] == r and dyk["a_z"] == r * c
    assert torch.isclose(dyk["a_x"], _t(1e-3 * r))     # thin, non-zero
    sil = _shape("sill", radius=r, aspect=c, ratio=1e-3)
    assert sil["a_x"] == r and sil["a_y"] == r * c
    assert torch.isclose(sil["a_z"], _t(1e-3 * r))
    # omega_y is always zero
    for st in CDM_STYLES:
        assert _shape(st)["omega_y"] == 0.0


def test_adapter_opening_matches_cdm_relation():
    p = _shape("oblate", radius=700.0, aspect=0.4, dv=5e6)
    ax, ay, az = p["a_x"], p["a_y"], p["a_z"]
    denom = 4.0 * (ax * ay + ax * az + ay * az)        # (2ax)(2ay)+...
    assert torch.allclose(p["opening"] * denom, _t(5e6))


def test_adapter_rejects_bad_style():
    with pytest.raises(ValueError, match="unknown CDM style"):
        _shape("banana")


def test_dyke_flat_axis_converges():
    """The thin-sheet dyke converges to the zero-thickness limit ~ O(ratio)."""
    B = 1
    n = 31
    ax = torch.linspace(-15000, 15000, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    X, Y = xx.reshape(1, -1), yy.reshape(1, -1)
    z0 = torch.zeros(B, dtype=DTYPE)
    src = CDMSource()

    def field(ratio):
        p = _shape("dyke", depth=8000., radius=2000., aspect=1.0, dv=2e7,
                   oz=0.6, ratio=ratio)
        return src(X, Y, z0, z0, **p).u

    ref = field(1e-5)
    rel_coarse = (field(1e-2) - ref).abs().max() / ref.abs().max()
    rel_fine = (field(1e-3) - ref).abs().max() / ref.abs().max()
    assert rel_fine < rel_coarse                     # finer floor -> closer
    assert rel_fine < 1e-2                            # default is within ~1%


# --------------------------------------------------------------------------- #
# CDMPrior
# --------------------------------------------------------------------------- #
def test_sample_returns_forward_keys():
    out = DEFAULT_CDM_SPHERE_PRIOR.sample((16,))
    assert set(out) == FORWARD_KEYS
    for v in out.values():
        assert v.shape == (16,) and torch.isfinite(v).all()


def test_sample_drives_cdmsource():
    B, n = 8, 9
    ax = torch.linspace(-20000, 20000, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    X = xx.reshape(1, -1).expand(B, -1).contiguous()
    Y = yy.reshape(1, -1).expand(B, -1).contiguous()
    z0 = torch.zeros(B, dtype=DTYPE)
    g = torch.Generator().manual_seed(0)
    out = DEFAULT_CDM_PRIORS["prolate"].sample((B,), generator=g)
    d = CDMSource()(X, Y, z0, z0, **out)
    assert isinstance(d, Displacement) and torch.isfinite(d.u).all()


def test_signed_applies_shared_sign_to_dv_only():
    g = torch.Generator().manual_seed(1)
    # signed: opening (sign of dv) takes both signs
    s = DEFAULT_CDM_SPHERE_PRIOR.sample((4000,), generator=g)
    assert (s["opening"] > 0).any() and (s["opening"] < 0).any()
    # semi-axes always positive regardless of the dv sign
    for k in ("a_x", "a_y", "a_z"):
        assert (s[k] > 0).all()


def test_unsigned_keeps_opening_positive():
    pr = CDMPrior(
        depth=ConstantPrior(8000.0), radius=ConstantPrior(500.0),
        aspect=ConstantPrior(1.0), dv=LogUniformPrior(1e5, 1e7),
        omega_x=ConstantPrior(0.0), omega_z=ConstantPrior(0.0),
        style="sphere", signed=False,
    )
    s = pr.sample((1000,), generator=torch.Generator().manual_seed(2))
    assert (s["opening"] > 0).all()


def test_sample_is_reproducible():
    a = DEFAULT_CDM_PRIORS["oblate"].sample((32,), generator=torch.Generator().manual_seed(7))
    b = DEFAULT_CDM_PRIORS["oblate"].sample((32,), generator=torch.Generator().manual_seed(7))
    for k in FORWARD_KEYS:
        assert torch.equal(a[k], b[k])


def test_prior_rejects_bad_style():
    pr = CDMPrior(
        depth=ConstantPrior(8000.0), radius=ConstantPrior(500.0),
        aspect=ConstantPrior(1.0), dv=ConstantPrior(1e6),
        omega_x=ConstantPrior(0.0), omega_z=ConstantPrior(0.0), style="wedge",
    )
    with pytest.raises(ValueError, match="unknown CDM style"):
        pr.sample((4,))


# --------------------------------------------------------------------------- #
# Presets: finite, submerged, generator round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", list(DEFAULT_CDM_PRIORS))
def test_preset_finite_and_submerged(style):
    B, n = 256, 7
    ax = torch.linspace(-25000, 25000, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    X = xx.reshape(1, -1).expand(B, -1).contiguous()
    Y = yy.reshape(1, -1).expand(B, -1).contiguous()
    z0 = torch.zeros(B, dtype=DTYPE)
    g = torch.Generator().manual_seed(0)
    out = DEFAULT_CDM_PRIORS[style].sample((B,), generator=g)
    # the cited geometric constraint: vertical extent stays below the surface
    assert (out["a_z"] < out["depth"]).all()
    d = CDMSource()(X, Y, z0, z0, **out)
    assert torch.isfinite(d.u).all() and torch.isfinite(d.e).all()


def test_round_trip_through_source_generator():
    B, n = 8, 9
    ax = torch.linspace(-20000, 20000, n, dtype=DTYPE)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    X = xx.reshape(1, -1).expand(B, -1).contiguous()
    Y = yy.reshape(1, -1).expand(B, -1).contiguous()
    gen = SourceGenerator(model=CDMSource(), prior=DEFAULT_CDM_PRIORS["sphere"])
    z0 = torch.zeros(B, dtype=DTYPE)
    g = torch.Generator().manual_seed(0)
    disp, params = gen.generate(X, Y, z0, z0, generator=g)
    assert torch.isfinite(disp.u).all()
    assert set(params) == FORWARD_KEYS
