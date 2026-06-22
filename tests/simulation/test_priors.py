"""
Tests for parameter priors (scalar priors + typed per-source bundles).

Covers the :class:`Prior` interface, the three scalar distributions, the
``make_prior`` config bridge, and the :class:`SourcePrior` bundles (generic
field-introspecting ``sample`` returning a dict, ``weight`` excluded).

Run with::

    pytest test_priors.py -v
"""
import pytest
import torch

from torchdeform import (
    OkadaSourceSimple,
    okada_params_from_fault,
)
from torchdeform.simulation import (
    Prior,
    UniformPrior,
    LogUniformPrior,
    SignedLogUniformPrior,
    ConstantPrior,
    make_prior,
    SourcePrior,
    MogiPrior,
    PennyPrior,
    OkadaPrior,
    DEFAULT_MOGI_PRIOR,
    DEFAULT_PRIORS,
    DEFAULT_EARTHQUAKE_PRIOR,
    DEFAULT_DYKE_PRIOR,
    SourceMixture,
    MixtureSample,
)


DTYPE = torch.float64


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------- #
# Scalar priors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [UniformPrior, LogUniformPrior, SignedLogUniformPrior])
def test_scalar_priors_are_prior_instances(cls):
    assert isinstance(cls(1.0, 10.0), Prior)


def test_uniform_in_range_and_shape():
    p = UniformPrior(2.0, 5.0)
    x = p.sample((5000,), _gen(), dtype=DTYPE)
    assert x.shape == (5000,)
    assert x.min() >= 2.0 and x.max() <= 5.0


def test_call_matches_sample_and_is_deterministic():
    p = UniformPrior(-1.0, 1.0)
    a = p.sample((10,), _gen(7), dtype=DTYPE)
    b = p((10,), _gen(7), dtype=DTYPE)        # __call__ from the base
    assert torch.allclose(a, b)


def test_log_uniform_positive_and_log_spaced():
    p = LogUniformPrior(1e2, 1e6)
    x = p.sample((10000,), _gen(1), dtype=DTYPE)
    assert (x > 0).all()
    assert x.min() >= 1e2 * (1 - 1e-9) and x.max() <= 1e6 * (1 + 1e-9)
    # uniform in log10 => median near the geometric mean (10**4)
    assert 1e3 < x.median() < 1e5


def test_signed_log_has_both_signs_and_right_magnitude():
    p = SignedLogUniformPrior(1e3, 1e5)
    x = p.sample((10000,), _gen(2), dtype=DTYPE)
    assert (x > 0).any() and (x < 0).any()
    mag = x.abs()
    assert mag.min() >= 1e3 * (1 - 1e-9) and mag.max() <= 1e5 * (1 + 1e-9)


@pytest.mark.parametrize("cls", [UniformPrior, LogUniformPrior, SignedLogUniformPrior])
def test_invalid_bounds_raise(cls):
    with pytest.raises(ValueError):
        cls(5.0, 1.0)            # high <= low


def test_log_priors_require_positive_low():
    with pytest.raises(ValueError):
        LogUniformPrior(-1.0, 10.0)


# --------------------------------------------------------------------------- #
# make_prior bridge
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,cls", [
    ("uniform", UniformPrior),
    ("log", LogUniformPrior),
    ("signed_log", SignedLogUniformPrior),
])
def test_make_prior_dispatch(mode, cls):
    assert isinstance(make_prior(1.0, 10.0, mode), cls)


def test_make_prior_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown prior mode"):
        make_prior(1.0, 10.0, "median")


# --------------------------------------------------------------------------- #
# Source priors
# --------------------------------------------------------------------------- #
def test_source_prior_sample_returns_named_dict():
    out = DEFAULT_MOGI_PRIOR.sample((8,), _gen(0), dtype=DTYPE)
    assert set(out) == {"depth", "delta_v"}
    assert all(v.shape == (8,) for v in out.values())
    assert (out["depth"] > 0).all()                       # log prior


def test_source_prior_fields_match_mogi_source_forward():
    """MogiPrior fields are exactly the kwargs MogiSource.forward consumes."""
    out = MogiPrior(
        depth=UniformPrior(1.0, 2.0),
        delta_v=UniformPrior(1.0, 2.0),
    ).sample((4,))
    assert set(out) == {"depth", "delta_v"}


def test_source_prior_call_matches_sample():
    p = PennyPrior(depth=UniformPrior(1.0, 2.0), radius=UniformPrior(1.0, 2.0),
                   pressure=UniformPrior(1.0, 2.0))
    a = p.sample((6,), _gen(3), dtype=DTYPE)
    b = p((6,), _gen(3), dtype=DTYPE)
    assert a.keys() == b.keys()
    assert all(torch.allclose(a[k], b[k]) for k in a)


def test_defaults_are_source_priors():
    assert isinstance(DEFAULT_MOGI_PRIOR, SourcePrior)
    assert set(DEFAULT_PRIORS) == {"earthquake", "dyke", "sill", "mogi", "penny"}
    # the Okada defaults are OkadaPrior presets, not distinct types
    assert isinstance(DEFAULT_EARTHQUAKE_PRIOR, OkadaPrior)
    assert type(DEFAULT_PRIORS["dyke"]) is OkadaPrior


def test_source_prior_sample_dtype_propagates():
    out = OkadaPrior(
        strike=UniformPrior(0.0, 360.0),
        dip=UniformPrior(30.0, 90.0),
        rake=UniformPrior(-180.0, 180.0),
        slip=UniformPrior(0.1, 5.0),
        opening=ConstantPrior(0.0),
        top_depth=UniformPrior(1.0, 2.0),
        length=UniformPrior(1.0, 2.0),
        width=UniformPrior(1.0, 2.0),
    ).sample((3,), dtype=torch.float32)
    assert all(v.dtype == torch.float32 for v in out.values())


# --------------------------------------------------------------------------- #
# ConstantPrior
# --------------------------------------------------------------------------- #
def test_constant_prior_returns_fixed_value():
    p = ConstantPrior(2.5)
    assert isinstance(p, Prior)
    x = p.sample((7,), dtype=DTYPE)
    assert x.shape == (7,)
    assert torch.all(x == 2.5)


# --------------------------------------------------------------------------- #
# OkadaPrior + okada_params_from_fault
# --------------------------------------------------------------------------- #
def test_okada_prior_presets_pin_irrelevant_slip():
    """Earthquake = pure shear (opening 0); dyke = pure opening (slip 0)."""
    eq = DEFAULT_EARTHQUAKE_PRIOR.sample((16,), _gen(0))
    assert torch.all(eq["opening"] == 0.0)
    assert (eq["slip"] > 0).all()

    dyke = DEFAULT_DYKE_PRIOR.sample((16,), _gen(1))
    assert torch.all(dyke["slip"] == 0.0)
    assert (dyke["opening"] > 0).all()


def test_okada_params_from_fault_conversion():
    params = {
        "strike": torch.tensor([90.0], dtype=DTYPE),
        "dip": torch.tensor([30.0], dtype=DTYPE),
        "rake": torch.tensor([0.0], dtype=DTYPE),    # pure strike-slip
        "slip": torch.tensor([2.0], dtype=DTYPE),
        "opening": torch.tensor([0.5], dtype=DTYPE),
        "top_depth": torch.tensor([1000.0], dtype=DTYPE),
        "length": torch.tensor([5000.0], dtype=DTYPE),
        "width": torch.tensor([4000.0], dtype=DTYPE),
        "fault_x": torch.tensor([123.0], dtype=DTYPE),   # passthrough
    }
    out = okada_params_from_fault(params)

    assert torch.allclose(out["strike"], torch.deg2rad(params["strike"]))
    assert torch.allclose(out["dip"], torch.deg2rad(params["dip"]))
    # rake 0 -> pure strike slip
    assert torch.allclose(out["disl1"], torch.tensor([2.0], dtype=DTYPE))
    assert torch.allclose(out["disl2"], torch.zeros(1, dtype=DTYPE), atol=1e-12)
    assert torch.allclose(out["disl3"], torch.tensor([0.5], dtype=DTYPE))
    # centroid = top + 0.5*width*sin(dip) = 1000 + 0.5*4000*0.5 = 2000
    assert torch.allclose(out["centroid_depth"], torch.tensor([2000.0], dtype=DTYPE))
    assert "fault_x" in out and torch.allclose(out["fault_x"], params["fault_x"])
    # raw fault keys are consumed, not passed through
    assert "slip" not in out and "top_depth" not in out


def test_okada_prior_plugs_into_okada_source():
    """End-to-end: sample -> convert -> OkadaSourceSimple.forward runs."""
    B, N = 4, 9
    prior = DEFAULT_EARTHQUAKE_PRIOR
    params = prior.sample((B,), _gen(0))
    fwd = okada_params_from_fault(params)

    x = torch.linspace(-10_000, 10_000, N, dtype=DTYPE).expand(B, -1).contiguous()
    y = x.clone()
    disp = OkadaSourceSimple()(
        x, y,
        fault_x=torch.zeros(B, dtype=DTYPE), fault_y=torch.zeros(B, dtype=DTYPE),
        **fwd,
    )
    assert disp.e.shape == (B, N)
    assert torch.isfinite(disp.e).all()
    assert torch.isfinite(disp.u).all()


# --------------------------------------------------------------------------- #
# SourceMixture
# --------------------------------------------------------------------------- #
def test_mixture_uniform_probabilities_by_default():
    mix = SourceMixture(DEFAULT_PRIORS)
    assert mix.names == tuple(DEFAULT_PRIORS)
    assert torch.allclose(mix.probabilities, torch.full((5,), 0.2, dtype=torch.float64))


def test_mixture_normalizes_weights():
    mix = SourceMixture(DEFAULT_PRIORS,
                        weights={"earthquake": 3, "dyke": 1, "sill": 1,
                                 "mogi": 1, "penny": 4})
    assert torch.isclose(mix.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    # order matches names; earthquake weight 3/10, penny 4/10
    p = dict(zip(mix.names, mix.probabilities.tolist()))
    assert p["earthquake"] == pytest.approx(0.3)
    assert p["penny"] == pytest.approx(0.4)


def test_mixture_sample_groups_by_type():
    mix = SourceMixture({"mogi": DEFAULT_PRIORS["mogi"],
                         "penny": DEFAULT_PRIORS["penny"]})
    res = mix.sample(64, generator=_gen(0))
    assert isinstance(res, MixtureSample)
    assert len(res.types) == 64
    # every item is accounted for exactly once across the per-type index tensors
    all_idx = torch.cat([res.index[t] for t in res.index])
    assert torch.equal(all_idx.sort().values, torch.arange(64))
    # params group sizes match index sizes and carry the right keys
    for t, p in res.params.items():
        n = res.index[t].numel()
        assert set(p) == set(DEFAULT_PRIORS[t].sample((1,)).keys())
        assert all(v.shape == (n,) for v in p.values())
        # the recorded types agree with the index grouping
        assert all(res.types[i] == t for i in res.index[t].tolist())


def test_mixture_weighting_is_respected_statistically():
    mix = SourceMixture({"mogi": DEFAULT_PRIORS["mogi"],
                         "penny": DEFAULT_PRIORS["penny"]},
                        weights={"mogi": 9.0, "penny": 1.0})
    types = mix.sample_types(20000, generator=_gen(1))
    frac_mogi = sum(t == "mogi" for t in types) / len(types)
    assert 0.85 < frac_mogi < 0.95          # ~0.9


def test_mixture_validation():
    with pytest.raises(ValueError):
        SourceMixture({})                                   # empty
    with pytest.raises(ValueError):
        SourceMixture(DEFAULT_PRIORS, weights={"mogi": 1.0})  # keys mismatch
    with pytest.raises(ValueError):
        SourceMixture({"mogi": DEFAULT_PRIORS["mogi"]},
                      weights={"mogi": 0.0})                # non-positive
