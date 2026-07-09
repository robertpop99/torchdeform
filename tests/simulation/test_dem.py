"""
Tests for the synthetic fractal DEM generator.

Checks:
* shape, exact per-image relief (std) and base elevation (mean);
* fractal roughness -- the 2-D PSD slope tracks ``-(2*hurst+2)`` and the surface
  gets smoother as Hurst rises; ``beta`` overrides ``hurst``;
* the regional ``ramp`` (peak-to-peak) and the ``positive`` shift (min == base);
* generator determinism, dtype, device;
* it feeds ``stratified_aps`` (skipped if the atmosphere module isn't importable).

Run with::

    pytest test_dem.py -v

"""
import pytest
import torch

from torchdeform.simulation import synthetic_dem
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
