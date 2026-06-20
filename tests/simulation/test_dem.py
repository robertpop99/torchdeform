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

----------------------------------------------------------------------------
IMPORT NOTE - adjust the imports below to your package layout.
----------------------------------------------------------------------------
"""
import pytest
import torch

try:
    from .dem import synthetic_dem
except ImportError:                               # pragma: no cover
    try:
        from deform.dem import synthetic_dem
    except ImportError:
        from dem import synthetic_dem

# stratified_aps is only needed for the integration test; skip if unavailable.
try:
    from .atm import stratified_aps
except ImportError:                               # pragma: no cover
    try:
        from atm import stratified_aps
    except ImportError:
        stratified_aps = None


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
    @pytest.mark.parametrize("hurst", [0.2, 0.5, 0.8])
    def test_psd_slope_tracks_hurst(self, hurst):
        torch.manual_seed(0)
        dem = synthetic_dem(64, 128, 128, relief=300.0, hurst=hurst, generator=_gen())
        assert abs(_psd_slope(dem) - (-(2 * hurst + 2))) < 0.2

    def test_smoother_with_higher_hurst(self):
        def ac(h):
            return _lag1(synthetic_dem(48, 128, 128, hurst=h, generator=_gen(1)))
        assert ac(0.2) < ac(0.5) < ac(0.8)

    def test_beta_overrides_hurst(self):
        a = synthetic_dem(2, 64, 64, beta=2.5, hurst=0.1, generator=_gen(3))
        b = synthetic_dem(2, 64, 64, beta=2.5, hurst=0.9, generator=_gen(3))
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
