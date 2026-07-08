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
    PCDMSource,
    los_vector,
)
from torchdeform.simulation import (
    Prior,
    UniformPrior,
    LogUniformPrior,
    ReverseLogUniformPrior,
    SignedPrior,
    SignedLogUniformPrior,
    NormalPrior,
    TruncatedNormalPrior,
    LogNormalPrior,
    SignedLogNormalPrior,
    PowerLawPrior,
    VonMisesPrior,
    ConstantPrior,
    ChoicePrior,
    MultimodalPrior,
    make_prior,
    PriorBundle,
    SourcePrior,
    MogiPrior,
    PennyPrior,
    OkadaPrior,
    PCDMPrior,
    GeometryPrior,
    DEFAULT_MOGI_PRIOR,
    DEFAULT_PRIORS,
    DEFAULT_EARTHQUAKE_PRIOR,
    DEFAULT_DYKE_PRIOR,
    DEFAULT_PCDM_PRIOR,
    DEFAULT_S1_GEOMETRY_PRIOR,
    PriorMixture,
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
# ReverseLogUniformPrior -- mirror of LogUniformPrior (mass near high)
# --------------------------------------------------------------------------- #
def test_reverse_log_in_range_and_biased_high():
    p = ReverseLogUniformPrior(1e2, 1e6)
    x = p.sample((10000,), _gen(0), dtype=DTYPE)
    assert isinstance(p, Prior)
    assert x.min() >= 1e2 * (1 - 1e-9) and x.max() <= 1e6 * (1 + 1e-9)
    # exact mirror of the log-uniform draw about the interval
    xr = (1e2 + 1e6) - LogUniformPrior(1e2, 1e6).sample((10000,), _gen(0), dtype=DTYPE)
    assert torch.allclose(x, xr)
    # mass piles up near high: median sits above the linear midpoint
    assert x.median() > 0.5 * (1e2 + 1e6)


def test_reverse_log_requires_positive_low_and_ordered_bounds():
    with pytest.raises(ValueError):
        ReverseLogUniformPrior(-1.0, 10.0)
    with pytest.raises(ValueError):
        ReverseLogUniformPrior(10.0, 1.0)


# --------------------------------------------------------------------------- #
# NormalPrior / TruncatedNormalPrior
# --------------------------------------------------------------------------- #
def test_normal_mean_and_std():
    p = NormalPrior(3.0, 2.0)
    x = p.sample((50000,), _gen(0), dtype=DTYPE)
    assert isinstance(p, Prior)
    assert abs(x.mean().item() - 3.0) < 0.05
    assert abs(x.std().item() - 2.0) < 0.05


def test_normal_requires_positive_std():
    with pytest.raises(ValueError):
        NormalPrior(0.0, 0.0)


def test_truncated_normal_stays_in_bounds_and_peaks():
    p = TruncatedNormalPrior(mean=5.0, std=3.0, low=0.0, high=10.0)
    x = p.sample((20000,), _gen(1), dtype=DTYPE)
    assert x.min() >= 0.0 and x.max() <= 10.0
    assert abs(x.mean().item() - 5.0) < 0.1        # symmetric truncation
    # heavy truncation on one side shifts the mass, never escapes the bounds
    q = TruncatedNormalPrior(mean=-5.0, std=2.0, low=0.0, high=1.0)
    y = q.sample((5000,), _gen(2), dtype=DTYPE)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_truncated_normal_validation():
    with pytest.raises(ValueError):
        TruncatedNormalPrior(0.0, 0.0, 0.0, 1.0)   # std <= 0
    with pytest.raises(ValueError):
        TruncatedNormalPrior(0.0, 1.0, 1.0, 0.0)   # high <= low


# --------------------------------------------------------------------------- #
# LogNormalPrior / SignedLogNormalPrior
# --------------------------------------------------------------------------- #
def test_lognormal_positive_and_median():
    p = LogNormalPrior(median=1e4, sigma=1.0)
    x = p.sample((50000,), _gen(0), dtype=DTYPE)
    assert (x > 0).all()
    assert 0.9e4 < x.median() < 1.1e4              # median = the `median` arg
    # ln(x) is normal with std ~ sigma
    assert abs(x.log().std().item() - 1.0) < 0.05


def test_lognormal_validation():
    with pytest.raises(ValueError):
        LogNormalPrior(-1.0, 1.0)
    with pytest.raises(ValueError):
        LogNormalPrior(1.0, 0.0)


def test_signed_lognormal_has_both_signs_and_right_median():
    p = SignedLogNormalPrior(median=1e4, sigma=0.8)
    x = p.sample((50000,), _gen(3), dtype=DTYPE)
    assert (x > 0).any() and (x < 0).any()
    assert 0.9e4 < x.abs().median() < 1.1e4


# --------------------------------------------------------------------------- #
# SignedPrior wrapper + preset aliases
# --------------------------------------------------------------------------- #
def test_signed_prior_wraps_any_magnitude():
    p = SignedPrior(ReverseLogUniformPrior(1e5, 1e8))
    x = p.sample((10000,), _gen(0), dtype=DTYPE)
    assert isinstance(p, Prior)
    assert (x > 0).any() and (x < 0).any()
    mag = x.abs()
    assert mag.min() >= 1e5 * (1 - 1e-9) and mag.max() <= 1e8 * (1 + 1e-9)


def test_signed_presets_are_signed_prior_instances():
    assert issubclass(SignedLogUniformPrior, SignedPrior)
    assert issubclass(SignedLogNormalPrior, SignedPrior)
    assert isinstance(SignedLogUniformPrior(1.0, 10.0), Prior)


def test_signed_preset_matches_explicit_wrapper_exactly():
    """The alias reproduces SignedPrior(LogUniformPrior(...)) draw-for-draw."""
    a = SignedLogUniformPrior(1e3, 1e5).sample((5000,), _gen(4), dtype=DTYPE)
    b = SignedPrior(LogUniformPrior(1e3, 1e5)).sample((5000,), _gen(4), dtype=DTYPE)
    assert torch.allclose(a, b)
    c = SignedLogNormalPrior(1e4, 0.7).sample((5000,), _gen(5), dtype=DTYPE)
    d = SignedPrior(LogNormalPrior(1e4, 0.7)).sample((5000,), _gen(5), dtype=DTYPE)
    assert torch.allclose(c, d)


def test_signed_prior_rejects_non_prior_magnitude():
    with pytest.raises(TypeError):
        SignedPrior(1.0)


# --------------------------------------------------------------------------- #
# PowerLawPrior
# --------------------------------------------------------------------------- #
def test_power_law_in_range_and_alpha_one_is_log_uniform():
    p = PowerLawPrior(1.0, 1e4, alpha=1.6)
    x = p.sample((20000,), _gen(0), dtype=DTYPE)
    assert x.min() >= 1.0 * (1 - 1e-9) and x.max() <= 1e4 * (1 + 1e-9)
    # alpha=1 exactly reproduces a log-uniform draw
    a = PowerLawPrior(1.0, 1e4, alpha=1.0).sample((5000,), _gen(7), dtype=DTYPE)
    b = LogUniformPrior(1.0, 1e4).sample((5000,), _gen(7), dtype=DTYPE)
    assert torch.allclose(a, b)


def test_power_law_alpha_controls_concentration():
    # larger alpha => steeper falloff => more mass near low => smaller median
    lo = PowerLawPrior(1.0, 1e4, alpha=0.5).sample((20000,), _gen(1), dtype=DTYPE)
    hi = PowerLawPrior(1.0, 1e4, alpha=2.5).sample((20000,), _gen(1), dtype=DTYPE)
    assert hi.median() < lo.median()


def test_power_law_validation():
    with pytest.raises(ValueError):
        PowerLawPrior(-1.0, 10.0, alpha=1.5)
    with pytest.raises(ValueError):
        PowerLawPrior(10.0, 1.0, alpha=1.5)


# --------------------------------------------------------------------------- #
# VonMisesPrior
# --------------------------------------------------------------------------- #
def test_von_mises_wraps_and_concentrates():
    import math
    p = VonMisesPrior(loc=170.0, concentration=8.0, degrees=True)
    x = p.sample((20000,), _gen(0), dtype=DTYPE)
    assert x.min() >= -180.0 and x.max() <= 180.0
    # circular mean near loc despite the 180/-180 wrap (no seam bias)
    ang = torch.deg2rad(x)
    cbar = math.degrees(math.atan2(ang.sin().mean().item(), ang.cos().mean().item()))
    assert abs(((cbar - 170.0 + 180.0) % 360.0) - 180.0) < 3.0
    # resultant length is high for kappa=8
    rlen = (ang.cos().mean() ** 2 + ang.sin().mean() ** 2).sqrt().item()
    assert rlen > 0.8


def test_von_mises_zero_concentration_is_uniform_circle():
    p = VonMisesPrior(loc=0.0, concentration=0.0)
    x = p.sample((20000,), _gen(1), dtype=DTYPE)
    assert x.min() >= -3.1416 and x.max() <= 3.1416
    ang = x
    rlen = (ang.cos().mean() ** 2 + ang.sin().mean() ** 2).sqrt().item()
    assert rlen < 0.05                             # no preferred direction


def test_von_mises_deterministic_and_validation():
    p = VonMisesPrior(0.5, 4.0)
    a = p.sample((32,), _gen(5), dtype=DTYPE)
    b = p.sample((32,), _gen(5), dtype=DTYPE)
    assert torch.allclose(a, b)                    # generator-respecting
    with pytest.raises(ValueError):
        VonMisesPrior(0.0, -1.0)


# --------------------------------------------------------------------------- #
# ChoicePrior
# --------------------------------------------------------------------------- #
def test_choice_prior_only_emits_given_values():
    p = ChoicePrior([-1.0, 1.0])
    x = p.sample((5000,), _gen(0), dtype=DTYPE)
    assert isinstance(p, Prior)
    assert set(x.unique().tolist()) <= {-1.0, 1.0}
    assert (x == -1.0).any() and (x == 1.0).any()


def test_choice_prior_weighting():
    p = ChoicePrior([0.0, 1.0], weights=[1.0, 3.0])
    x = p.sample((20000,), _gen(1), dtype=DTYPE)
    frac_one = (x == 1.0).float().mean().item()
    assert 0.70 < frac_one < 0.80                  # ~0.75


def test_choice_prior_validation():
    with pytest.raises(ValueError):
        ChoicePrior([])
    with pytest.raises(ValueError):
        ChoicePrior([1.0, 2.0], weights=[1.0])     # length mismatch
    with pytest.raises(ValueError):
        ChoicePrior([1.0], weights=[0.0])          # non-positive


# --------------------------------------------------------------------------- #
# make_prior bridge
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,cls", [
    ("uniform", UniformPrior),
    ("log", LogUniformPrior),
    ("reverse_log", ReverseLogUniformPrior),
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
    assert set(DEFAULT_PRIORS) == {"earthquake", "dyke", "sill", "mogi", "penny", "pcdm"}
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
        "source_x": torch.tensor([123.0], dtype=DTYPE),   # passthrough
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
    assert "source_x" in out and torch.allclose(out["source_x"], params["source_x"])
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
        source_x=torch.zeros(B, dtype=DTYPE), source_y=torch.zeros(B, dtype=DTYPE),
        **fwd,
    )
    assert disp.e.shape == (B, N)
    assert torch.isfinite(disp.e).all()
    assert torch.isfinite(disp.u).all()


# --------------------------------------------------------------------------- #
# PCDMPrior
# --------------------------------------------------------------------------- #
def test_pcdm_prior_fields_match_source():
    out = DEFAULT_PCDM_PRIOR.sample((5,), _gen(0))
    assert set(out) == {"depth", "omega_x", "omega_y", "omega_z",
                        "dv_x", "dv_y", "dv_z"}
    assert all(v.shape == (5,) for v in out.values())


def test_pcdm_prior_potencies_share_sign_per_item():
    out = DEFAULT_PCDM_PRIOR.sample((200,), _gen(1))
    sx, sy, sz = out["dv_x"].sign(), out["dv_y"].sign(), out["dv_z"].sign()
    assert torch.equal(sx, sy) and torch.equal(sy, sz)
    # and both polarities actually occur
    assert (sx > 0).any() and (sx < 0).any()


def test_pcdm_prior_unsigned_is_positive():
    p = PCDMPrior(
        depth=LogUniformPrior(1e3, 2e4),
        omega_x=UniformPrior(-1.0, 1.0), omega_y=UniformPrior(-1.0, 1.0),
        omega_z=UniformPrior(0.0, 6.0),
        dv_x=LogUniformPrior(1e5, 1e8), dv_y=LogUniformPrior(1e5, 1e8),
        dv_z=LogUniformPrior(1e5, 1e8),
        signed=False,
    )
    out = p.sample((100,), _gen(2))
    assert (out["dv_x"] > 0).all() and (out["dv_y"] > 0).all() and (out["dv_z"] > 0).all()


def test_pcdm_prior_plugs_into_source():
    B, N = 4, 9
    params = DEFAULT_PCDM_PRIOR.sample((B,), _gen(0))
    x = torch.linspace(-10_000, 10_000, N, dtype=DTYPE).expand(B, -1).contiguous()
    y = x.clone()
    disp = PCDMSource()(
        x, y, source_x=torch.zeros(B, dtype=DTYPE), source_y=torch.zeros(B, dtype=DTYPE),
        **params,
    )
    assert disp.u.shape == (B, N)
    assert torch.isfinite(disp.u).all()


def test_pcdm_in_default_priors():
    assert "pcdm" in DEFAULT_PRIORS
    assert isinstance(DEFAULT_PRIORS["pcdm"], PCDMPrior)


# --------------------------------------------------------------------------- #
# SourceMixture
# --------------------------------------------------------------------------- #
def test_mixture_uniform_probabilities_by_default():
    mix = SourceMixture(DEFAULT_PRIORS)
    k = len(DEFAULT_PRIORS)
    assert mix.names == tuple(DEFAULT_PRIORS)
    assert torch.allclose(mix.probabilities, torch.full((k,), 1.0 / k, dtype=torch.float64))


def test_mixture_normalizes_weights():
    mix = SourceMixture(
        {"earthquake": DEFAULT_PRIORS["earthquake"], "penny": DEFAULT_PRIORS["penny"]},
        weights={"earthquake": 3, "penny": 7},
    )
    assert torch.isclose(mix.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    p = dict(zip(mix.names, mix.probabilities.tolist()))
    assert p["earthquake"] == pytest.approx(0.3)
    assert p["penny"] == pytest.approx(0.7)


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


# --------------------------------------------------------------------------- #
# Generic aliases (renamed from Source* -> Prior*)
# --------------------------------------------------------------------------- #
def test_aliases_are_identical_objects():
    assert SourcePrior is PriorBundle
    assert SourceMixture is PriorMixture
    # existing bundles are PriorBundle instances
    assert isinstance(DEFAULT_MOGI_PRIOR, PriorBundle)


# --------------------------------------------------------------------------- #
# MultimodalPrior
# --------------------------------------------------------------------------- #
def test_multimodal_is_prior_and_covers_components():
    mm = MultimodalPrior([UniformPrior(-15.0, -13.0), UniformPrior(193.0, 195.0)])
    assert isinstance(mm, Prior)
    x = mm.sample((4000,), _gen(0), dtype=DTYPE)
    in_asc = (x >= -15.0) & (x <= -13.0)
    in_desc = (x >= 193.0) & (x <= 195.0)
    assert bool((in_asc | in_desc).all())     # every draw in one component
    assert in_asc.any() and in_desc.any()     # both components used


def test_multimodal_weighting():
    mm = MultimodalPrior([ConstantPrior(0.0), ConstantPrior(1.0)], weights=[1.0, 3.0])
    x = mm.sample((10000,), _gen(1))
    frac_one = (x == 1.0).float().mean().item()
    assert 0.70 < frac_one < 0.80           # ~0.75


def test_multimodal_validation():
    with pytest.raises(ValueError):
        MultimodalPrior([])
    with pytest.raises(ValueError):
        MultimodalPrior([UniformPrior(0, 1)], weights=[1.0, 2.0])    # length mismatch
    with pytest.raises(ValueError):
        MultimodalPrior([UniformPrior(0, 1)], weights=[0.0])         # non-positive


def test_multimodal_call_and_dtype():
    mm = MultimodalPrior([UniformPrior(0.0, 1.0), UniformPrior(2.0, 3.0)])
    a = mm.sample((5,), _gen(2), dtype=torch.float32)
    assert a.shape == (5,) and a.dtype == torch.float32


# --------------------------------------------------------------------------- #
# GeometryPrior
# --------------------------------------------------------------------------- #
def test_geometry_prior_fields_and_default_look_side():
    geom = GeometryPrior(UniformPrior(-15.0, -13.0), UniformPrior(29.0, 46.0))
    out = geom.sample((6,), _gen(0))
    assert set(out) == {"heading_deg", "incidence_deg", "look_side"}
    assert all(v.shape == (6,) for v in out.values())
    assert torch.all(out["look_side"] == 1.0)      # default right-looking


def test_geometry_prior_left_looking():
    geom = GeometryPrior(UniformPrior(-15.0, -13.0), UniformPrior(29.0, 46.0),
                         look_side=ConstantPrior(-1.0))
    out = geom.sample((4,))
    assert torch.all(out["look_side"] == -1.0)


def test_geometry_prior_plugs_into_los_vector():
    out = DEFAULT_S1_GEOMETRY_PRIOR.sample((8,), _gen(0))
    los = los_vector(**out)
    assert torch.allclose(los.norm, torch.ones_like(los.norm), atol=1e-12)


def test_s1_default_heading_is_bimodal():
    from torchdeform.observation import (S1_HEADING_ASCENDING_DEG,
                                         S1_HEADING_DESCENDING_DEG)
    alo, ahi = S1_HEADING_ASCENDING_DEG
    dlo, dhi = S1_HEADING_DESCENDING_DEG
    out = DEFAULT_S1_GEOMETRY_PRIOR.sample((4000,), _gen(0))
    h = out["heading_deg"]
    in_asc = (h >= alo) & (h <= ahi)
    in_desc = (h >= dlo) & (h <= dhi)
    assert bool((in_asc | in_desc).all())
    assert in_asc.any() and in_desc.any()
    # incidence in the IW range
    assert bool(((out["incidence_deg"] >= 29.0) & (out["incidence_deg"] <= 46.0)).all())


def test_prior_mixture_over_geometry_bundles():
    asc = GeometryPrior(UniformPrior(-15.0, -13.0), UniformPrior(29.0, 46.0))
    desc = GeometryPrior(UniformPrior(193.0, 195.0), UniformPrior(29.0, 46.0))
    mix = PriorMixture({"asc": asc, "desc": desc}, weights={"asc": 1.0, "desc": 3.0})
    res = mix.sample(2000, generator=_gen(0))
    assert set(res.index) <= {"asc", "desc"}
    # the heavier "desc" bundle is chosen ~3x more often
    n_desc = res.index["desc"].numel()
    assert 0.70 < n_desc / 2000 < 0.80
