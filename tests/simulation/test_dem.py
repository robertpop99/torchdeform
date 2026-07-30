"""
Tests for the synthetic fractal DEM generator.

Checks:
* shape, exact per-image relief (std) and base elevation (mean);
* fractal roughness -- the 2-D PSD slope tracks ``-(2*hurst+2)`` and the surface
  gets smoother as Hurst rises; ``beta`` overrides ``hurst``;
* the regional ``ramp`` (peak-to-peak) and the ``positive`` shift (min == base);
* generator determinism, dtype, device;
* ``method="spim"``: exact unit tests of the flow-routing / drainage-area /
  pit-filling helpers, plus slope-area scaling and gradient isotropy of the
  evolved terrain (no gradcheck -- spim is a data generator, not a gradient path);
* it feeds ``stratified_aps`` (skipped if the atmosphere module isn't importable).

Run with::

    pytest test_dem.py -v

"""
import math

import pytest
import torch

from torchdeform.simulation import synthetic_dem
from torchdeform.simulation.dem import (SPIM_FINE_RES_CAP, _EROSION_NEIGHBOURS,
                                        _drainage_area, _fill_pits, _flow_weights,
                                        _normalize, _pin_ring, _shift)
from torchdeform.atmosphere import stratified_aps


DTYPE = torch.float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _psd_slope(field, fit=(0.02, 0.2)):
    B, H, W = field.shape
    f = field - field.mean(dim=(-2, -1), keepdim=True)
    Fs = torch.fft.fftshift(torch.fft.fft2(f), dim=(-2, -1))
    Pk = (Fs.abs() ** 2).mean(0)
    ky = torch.fft.fftshift(torch.fft.fftfreq(H, dtype=DTYPE))
    kx = torch.fft.fftshift(torch.fft.fftfreq(W, dtype=DTYPE))
    K = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    m = (K > fit[0]) & (K < fit[1])
    A = torch.stack([torch.log(K[m]), torch.ones(int(m.sum()), dtype=DTYPE)], dim=1)
    return torch.linalg.lstsq(A, torch.log(Pk[m] + 1e-30)[:, None]).solution[0, 0].item()


def _lag1(field):
    f = field - field.mean(dim=(-2, -1), keepdim=True)
    v = f.var(dim=(-2, -1))
    return ((f[:, 1:, :] * f[:, :-1, :]).mean(dim=(-2, -1)) / v).mean().item()


# --------------------------------------------------------------------------- #
# Shape & amplitude
# --------------------------------------------------------------------------- #
class TestShapeAndAmplitude:
    def test_shape(self):
        assert synthetic_dem(5, 32, 48, generator=_gen()).shape == (5, 32, 48)

    def test_relief_is_exact_std(self):
        dem = synthetic_dem(6, 64, 64, relief=250.0, ramp=0.0, generator=_gen())
        torch.testing.assert_close(dem.std(dim=(-2, -1)),
                                   torch.full((6,), 250.0, dtype=DTYPE),
                                   rtol=1e-6, atol=1e-6)

    def test_mean_is_base_elevation(self):
        dem = synthetic_dem(4, 64, 64, relief=200.0, base_elevation=750.0,
                            generator=_gen())
        torch.testing.assert_close(dem.mean(dim=(-2, -1)),
                                   torch.full((4,), 750.0, dtype=DTYPE),
                                   rtol=0, atol=1e-7)

    def test_zero_relief_is_flat(self):
        dem = synthetic_dem(2, 32, 32, relief=0.0, base_elevation=100.0,
                            generator=_gen())
        assert torch.allclose(dem, torch.full_like(dem, 100.0))


# --------------------------------------------------------------------------- #
# Roughness
# --------------------------------------------------------------------------- #
class TestRoughness:
    # roughness knobs (hurst/beta/PSD slope) are specific to the fBm surface
    @pytest.mark.parametrize("hurst", [0.2, 0.5, 0.8])
    def test_psd_slope_tracks_hurst(self, hurst):
        torch.manual_seed(0)
        dem = synthetic_dem(64, 128, 128, relief=300.0, hurst=hurst,
                            method="fbm", generator=_gen())
        assert abs(_psd_slope(dem) - (-(2 * hurst + 2))) < 0.2

    def test_smoother_with_higher_hurst(self):
        def ac(h):
            return _lag1(synthetic_dem(48, 128, 128, hurst=h,
                                       method="fbm", generator=_gen(1)))
        assert ac(0.2) < ac(0.5) < ac(0.8)

    def test_beta_overrides_hurst(self):
        a = synthetic_dem(2, 64, 64, beta=2.5, hurst=0.1, method="fbm", generator=_gen(3))
        b = synthetic_dem(2, 64, 64, beta=2.5, hurst=0.9, method="fbm", generator=_gen(3))
        torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------- #
# Ramp & positivity
# --------------------------------------------------------------------------- #
class TestRampAndPositive:
    def test_ramp_sets_peak_to_peak(self):
        # tiny relief -> the planar tilt dominates the range
        dem = synthetic_dem(4, 64, 80, relief=1e-9, ramp=500.0, generator=_gen())
        ptp = dem.amax(dim=(-2, -1)) - dem.amin(dim=(-2, -1))
        torch.testing.assert_close(ptp, torch.full((4,), 500.0, dtype=DTYPE),
                                   rtol=1e-3, atol=1e-3)

    def test_positive_sets_min_to_base(self):
        dem = synthetic_dem(3, 64, 64, relief=200.0, positive=True,
                            base_elevation=300.0, generator=_gen())
        torch.testing.assert_close(dem.amin(dim=(-2, -1)),
                                   torch.full((3,), 300.0, dtype=DTYPE),
                                   rtol=0, atol=1e-6)

    def test_positive_dem_is_nonnegative(self):
        dem = synthetic_dem(3, 48, 48, relief=400.0, positive=True,
                            base_elevation=0.0, generator=_gen())
        assert dem.min().item() >= 0.0

    def test_fold_min_is_base_and_nonnegative(self):
        dem = synthetic_dem(3, 64, 64, relief=400.0, fold=True,
                            base_elevation=200.0, generator=_gen())
        torch.testing.assert_close(dem.amin(dim=(-2, -1)),
                                   torch.full((3,), 200.0, dtype=DTYPE),
                                   rtol=0, atol=1e-6)
        assert dem.min().item() >= 200.0 - 1e-9

    def test_fold_differs_from_shift(self):
        # abs-fold is a different shape than a plain min-shift for the same field
        kw = dict(relief=400.0, base_elevation=0.0)
        shifted = synthetic_dem(2, 64, 64, positive=True, generator=_gen(9), **kw)
        folded = synthetic_dem(2, 64, 64, fold=True, generator=_gen(9), **kw)
        assert not torch.allclose(shifted, folded)


# --------------------------------------------------------------------------- #
# Determinism / dtype / device
# --------------------------------------------------------------------------- #
class TestDeterminismDtypeDevice:
    def test_generator_deterministic(self):
        a = synthetic_dem(3, 64, 64, ramp=200.0, generator=_gen(7))
        b = synthetic_dem(3, 64, 64, ramp=200.0, generator=_gen(7))
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        a = synthetic_dem(2, 64, 64, generator=_gen(0))
        b = synthetic_dem(2, 64, 64, generator=_gen(1))
        assert not torch.allclose(a, b)

    def test_dtype(self):
        assert synthetic_dem(2, 32, 32, dtype=torch.float32,
                             generator=_gen()).dtype == torch.float32

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        dem = synthetic_dem(4, 64, 64, relief=300.0, ramp=200.0,
                            positive=True, base_elevation=100.0, device="cuda")
        assert dem.device.type == "cuda" and torch.isfinite(dem).all()


# --------------------------------------------------------------------------- #
# Realistic ("ridged") terrain
# --------------------------------------------------------------------------- #
class TestRidgedMethod:
    def test_shape_and_finite(self):
        dem = synthetic_dem(3, 96, 128, method="ridged", warp=0.2,
                            erosion_iters=30, generator=_gen())
        assert dem.shape == (3, 96, 128) and torch.isfinite(dem).all()

    def test_shares_output_semantics(self):
        # relief is still the exact per-image std, base_elevation the mean
        dem = synthetic_dem(4, 96, 96, relief=250.0, base_elevation=800.0,
                            method="ridged", warp=0.2, erosion_iters=20,
                            generator=_gen())
        torch.testing.assert_close(dem.std(dim=(-2, -1)),
                                   torch.full((4,), 250.0, dtype=DTYPE),
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(dem.mean(dim=(-2, -1)),
                                   torch.full((4,), 800.0, dtype=DTYPE),
                                   rtol=0, atol=1e-6)

    def test_deterministic(self):
        kw = dict(method="ridged", warp=0.2, erosion_iters=25)
        torch.testing.assert_close(synthetic_dem(2, 96, 96, generator=_gen(4), **kw),
                                   synthetic_dem(2, 96, 96, generator=_gen(4), **kw))

    def test_asymmetric_unlike_fbm(self):
        # sharp peaks + broad valleys make ridged terrain skewed, whereas an fBm
        # surface is (near-)symmetric Gaussian -> ~zero skew. Skewness is the
        # structural fingerprint fBm lacks.
        def skew(dem):
            f = dem - dem.mean(dim=(-2, -1), keepdim=True)
            s = f.std(dim=(-2, -1), keepdim=True)
            return ((f / s) ** 3).mean(dim=(-2, -1)).abs().mean().item()
        r = skew(synthetic_dem(16, 128, 128, method="ridged", warp=0.2,
                               generator=_gen(2)))
        f = skew(synthetic_dem(16, 128, 128, method="fbm", generator=_gen(2)))
        assert r > 0.3 and r > f

    def test_erosion_smooths(self):
        # thermal erosion rounds ridges -> smoother (higher lag-1 autocorr)
        rough = _lag1(synthetic_dem(8, 128, 128, method="ridged", warp=0.2,
                                    erosion_iters=0, generator=_gen(5)))
        eroded = _lag1(synthetic_dem(8, 128, 128, method="ridged", warp=0.2,
                                     erosion_iters=80, generator=_gen(5)))
        assert eroded > rough

    def test_bad_method_raises(self):
        with pytest.raises(ValueError):
            synthetic_dem(1, 32, 32, method="nope", generator=_gen())

    def test_ridged_is_the_default(self):
        torch.testing.assert_close(synthetic_dem(2, 64, 64, generator=_gen(11)),
                                   synthetic_dem(2, 64, 64, method="ridged",
                                                 generator=_gen(11)))


# --------------------------------------------------------------------------- #
# Stream-power ("spim") terrain
# --------------------------------------------------------------------------- #
# The routing/drainage/fill helpers are the valuable unit tests -- they are exact
# on hand-built surfaces, and a sign error in any of them only shows up as soup
# after a few hundred evolution steps. Deliberately no gradcheck: "spim" is a data
# generator, not a gradient path (see the module docstring).

# small but still network-forming; the default 400+200 steps at 112/176 is far
# too slow for CI
_SPIM_FAST = dict(method="spim", spim_res=64, spim_fine_res=96,
                  spim_steps=120, spim_fine_steps=60)


def _plane(n, dtype=DTYPE):
    """Inclined plane dropping toward +y, ringed by base level."""
    yy = torch.arange(n, dtype=dtype)
    z = (n - 1 - yy)[None, :, None].repeat(1, 1, n).contiguous()
    return _pin_ring(z, 0.0)


def _orientation_stats(dem, nbins=72):
    """Magnitude-weighted histogram of gradient orientations (mod 180 deg),
    normalised to mean 1. Returns ``(peak, aligned/off-axis)``: both ~1 for an
    isotropic field, both large when structure locks onto the grid."""
    gy, gx = torch.gradient(dem, dim=(-2, -1))
    mag = (gx ** 2 + gy ** 2).sqrt()
    ang = torch.atan2(gy, gx) % math.pi
    idx = (ang / math.pi * nbins).long().clamp(0, nbins - 1)
    hist = torch.zeros(nbins, dtype=dem.dtype)
    hist.scatter_add_(0, idx.reshape(-1), mag.reshape(-1))
    hist = hist / hist.mean()

    centres = torch.arange(nbins) * 180.0 / nbins
    aligned = torch.zeros(nbins, dtype=torch.bool)
    for a in (0.0, 45.0, 90.0, 135.0):
        off = (centres - a).abs() % 180.0
        aligned |= (off < 3.0) | (off > 177.0)
    ratio = hist[aligned].mean() / hist[~aligned].mean().clamp_min(1e-12)
    return hist.max().item(), ratio.item()


def _interior_pits(z):
    """Count interior cells lying below all eight of their neighbours."""
    nb = torch.stack([_shift(z, dy, dx, mode="constant")
                      for dy, dx, _ in _EROSION_NEIGHBOURS], dim=0)
    inner = torch.zeros_like(z, dtype=torch.bool)
    inner[:, 1:-1, 1:-1] = True
    return int(((z < nb.amin(dim=0)) & inner).sum())


class TestSpimRouting:
    def test_flow_points_downhill(self):
        w, _ = _flow_weights(_plane(24), 8.0)
        k = int(w[:, 0, 5, 5].argmax())
        assert _EROSION_NEIGHBOURS[k][:2] == (1, 0)      # +y is downslope

    def test_drainage_grows_downslope(self):
        z = _plane(24)
        w, _ = _flow_weights(z, 8.0)
        col = _drainage_area(w, torch.ones_like(z), 60)[0, :, 12]
        assert torch.all(col[2:-1] >= col[1:-2] - 1e-9)

    def test_drainage_area_counts_upslope_cells(self):
        # with near-D8 routing on a plane, the cell above the outlet drains the
        # whole column above it: area == number of upslope interior cells
        n = 24
        z = _plane(n)
        w, _ = _flow_weights(z, 8.0)
        area = _drainage_area(w, torch.ones_like(z), 4 * n)
        assert abs(area[0, -2, n // 2].item() - (n - 3)) < 0.5

    def test_fill_removes_interior_pits(self):
        n = 24
        yy = torch.arange(n, dtype=DTYPE)
        z = (n - 1 - yy)[None, :, None].repeat(1, 1, n).contiguous()
        z[0, 12, 12] = -50.0
        assert _interior_pits(z) == 1
        zf = _fill_pits(z, None, n, z.new_tensor(1e-6))
        assert _interior_pits(zf) == 0
        assert torch.all(zf >= z - 1e-12)               # filling only ever raises

    def test_fill_is_a_no_op_on_a_monotone_surface(self):
        z = _plane(24)
        zf = _fill_pits(z, None, 24, z.new_tensor(1e-9))
        assert (zf - z).abs().max().item() < 1e-6

    def test_pits_route_nowhere(self):
        # a cell with no downhill neighbour keeps its water instead of sending it
        # uphill: all-zero weights, not a uniform 1/8 split
        z = torch.zeros(1, 5, 5, dtype=DTYPE)
        z[0, 2, 2] = -1.0
        w, _ = _flow_weights(z, 4.0)
        assert w[:, 0, 2, 2].abs().max().item() == 0.0


class TestSpimTerrain:
    def test_shape_and_finite(self):
        dem = synthetic_dem(3, 96, 128, generator=_gen(), **_SPIM_FAST)
        assert dem.shape == (3, 96, 128) and torch.isfinite(dem).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtypes_finite(self, dtype):
        dem = synthetic_dem(2, 64, 64, dtype=dtype, generator=_gen(), **_SPIM_FAST)
        assert dem.dtype == dtype and torch.isfinite(dem).all()

    def test_shares_output_semantics(self):
        dem = synthetic_dem(4, 64, 64, relief=250.0, base_elevation=800.0,
                            generator=_gen(), **_SPIM_FAST)
        torch.testing.assert_close(dem.std(dim=(-2, -1)),
                                   torch.full((4,), 250.0, dtype=DTYPE),
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(dem.mean(dim=(-2, -1)),
                                   torch.full((4,), 800.0, dtype=DTYPE),
                                   rtol=0, atol=1e-6)

    def test_positive_and_fold(self):
        p = synthetic_dem(2, 64, 64, positive=True, base_elevation=7.0,
                          generator=_gen(), **_SPIM_FAST)
        torch.testing.assert_close(p.amin(dim=(-2, -1)),
                                   torch.full((2,), 7.0, dtype=DTYPE))
        f = synthetic_dem(2, 64, 64, fold=True, generator=_gen(), **_SPIM_FAST)
        assert (f >= 0).all()

    def test_deterministic_and_batch_varies(self):
        a = synthetic_dem(2, 64, 64, generator=_gen(3), **_SPIM_FAST)
        b = synthetic_dem(2, 64, 64, generator=_gen(3), **_SPIM_FAST)
        torch.testing.assert_close(a, b)
        assert not torch.allclose(a[0], a[1])

    def test_runs_without_the_refinement_pass(self):
        dem = synthetic_dem(2, 64, 64, generator=_gen(),
                            **{**_SPIM_FAST, "spim_fine_steps": 0})
        assert torch.isfinite(dem).all()

    def test_no_autograd_graph(self):
        # forward-only by construction: nothing to differentiate, nothing retained
        dem = synthetic_dem(1, 64, 64, generator=_gen(), **_SPIM_FAST)
        assert not dem.requires_grad and dem.grad_fn is None

    def test_default_fine_res_tracks_output_then_caps(self):
        # spim_fine_res=None sizes the refinement grid to the output so fidelity does
        # not silently decay as the DEM grows, but stops at SPIM_FINE_RES_CAP because
        # cost is quadratic in it. Asserted on the grid the sizing logic picks rather
        # than on runtime, which is far too noisy to gate a test on.
        def fine_grid(out):
            want = math.ceil(out / (1.0 - 2.0 * 0.12))       # the default spim_margin
            return min(SPIM_FINE_RES_CAP, max(48, want))

        assert fine_grid(64) < fine_grid(256) <= SPIM_FINE_RES_CAP   # tracks the output
        assert fine_grid(512) == fine_grid(4096) == SPIM_FINE_RES_CAP  # then plateaus

        # small outputs are unaffected by the cap, so they must be bit-identical to
        # pinning the grid explicitly -- the change is a no-op below it.
        small = dict(_SPIM_FAST); small.pop("spim_fine_res")
        auto = synthetic_dem(2, 64, 64, generator=_gen(3), **small)
        want = min(SPIM_FINE_RES_CAP, max(48, math.ceil(64 / 0.76)))
        pinned = synthetic_dem(2, 64, 64, generator=_gen(3),
                               **{**small, "spim_fine_res": want})
        torch.testing.assert_close(auto, pinned)

    def test_slope_area_scaling(self):
        # the standard fingerprint of a fluvially-organised landscape: on channel
        # cells, S ~ A^(-m/n). Measured with detail off (the added roughness biases
        # S upward) and lakes excluded (filled depressions have S ~ 0 by
        # construction and are exactly the high-A cells). Wide band on purpose --
        # S ~ A^(-m/n) is a steady-state result and a few hundred steps from noise
        # need not have got there.
        dem = _normalize(synthetic_dem(2, 128, 128, generator=_gen(0),
                                       **{**_SPIM_FAST, "spim_detail": 0.0}))
        n = max(dem.shape[-2:])
        zf = _fill_pits(dem, None, n, dem.new_tensor(1e-9))
        w, _ = _flow_weights(zf, 4.0)
        area = _drainage_area(w, torch.ones_like(dem), 2 * n)
        _, drops = _flow_weights(dem, 4.0)               # slope on the real surface
        slope = (w * drops).sum(dim=0)

        lake = (zf - dem) > 1e-9 * (dem.amax() - dem.amin())
        keep = (area >= torch.quantile(area.reshape(-1), 0.90)) & ~lake & (slope > 0)
        keep[:, :3, :] = keep[:, -3:, :] = False
        keep[:, :, :3] = keep[:, :, -3:] = False
        assert int(keep.sum()) > 200

        la, ls = torch.log(area[keep]), torch.log(slope[keep])
        design = torch.stack([la, torch.ones_like(la)], dim=1)
        exponent = torch.linalg.lstsq(design, ls[:, None]).solution[0, 0].item()
        assert -0.8 < exponent < -0.2, exponent

    def test_not_grid_aligned(self):
        # Routing on a grid risks locking drainage onto the 8 neighbour
        # directions. Bin gradient orientations (mod 180 deg) weighted by
        # magnitude: striations show up as energy piled onto 0/45/90/135. The
        # ratio is ~0.87 for spim and unbounded for genuinely striped fields.
        s_peak, s_ratio = _orientation_stats(
            synthetic_dem(4, 128, 128, generator=_gen(0), **_SPIM_FAST))
        assert s_ratio < 1.15, s_ratio          # no preference for grid axes
        assert s_peak < 1.8, s_peak             # no single dominant orientation

    def test_orientation_metric_catches_striations(self):
        # guards the test above: a deliberately striped field must fail it
        n = 128
        stripes = torch.sin(1.2 * torch.arange(n, dtype=DTYPE))
        peak, ratio = _orientation_stats(stripes[None, :, None].repeat(1, 1, n))
        assert peak > 10.0 and ratio > 5.0


class TestOtherMethodsUnaffected:
    @pytest.mark.parametrize("kw", [
        dict(method="fbm"),
        dict(method="ridged", warp=0.2, erosion_iters=25),
        dict(method="ridged", ramp=500.0, positive=True),
    ])
    def test_spim_kwargs_do_not_touch_other_methods(self, kw):
        # the spim_* knobs must be inert outside method="spim"
        a = synthetic_dem(2, 48, 48, generator=_gen(5), **kw)
        b = synthetic_dem(2, 48, 48, generator=_gen(5), spim_steps=7,
                          spim_res=32, spim_detail=0.9, spim_margin=0.3, **kw)
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Integration with the stratified atmosphere
# --------------------------------------------------------------------------- #
class TestIntegration:
    @pytest.mark.skipif(stratified_aps is None, reason="atm module not importable")
    def test_feeds_stratified_aps(self):
        dem = synthetic_dem(2, 48, 48, relief=400.0, base_elevation=800.0,
                            positive=True, generator=_gen())
        s = stratified_aps(dem, torch.tensor([3e-3, -2e-3]), model="linear")
        assert s.shape == (2, 48, 48) and torch.isfinite(s).all()
        # linear screen is exactly affine in the DEM
        c = torch.corrcoef(torch.stack([s[0].reshape(-1), dem[0].reshape(-1)]))[0, 1]
        assert c.item() > 0.999999


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
