"""
Tests for the synthetic atmospheric phase screen module (turbulent + stratified).

What is checked
---------------
Turbulent (spectral synthesis):
* power-law screens have a 2-D PSD slope ~ -beta; exponential screens get
  smoother as the correlation length grows;
* per-image spatial RMS is set exactly and the field is zero-mean;
* batching, generator determinism, and the user-supplied-noise path;
* the dense Cholesky reference reproduces ``maxvar*exp(-alpha r)``.

Stratified (topography-correlated):
* linear screen is exactly affine in the DEM (corr = +/-1, zero mean, invariant
  to a constant elevation offset); exponential screen anti-correlates with
  elevation and reduces to the linear model for large scale height;
* shapes / batching / coefficient sampler.

Plus differentiability (gradients flow through both components) and dtype/device.

Run with::

    pytest test_atm.py -v

"""

import math

import pytest
import torch


from torchdeform.atmosphere import (TurbulentAPS, turbulent_aps, spectral_filter,
                 correlated_noise_cholesky, stratified_aps, StratifiedAPS,
                 sample_stratified_coeff, orbital_ramp, atmospheric_phase_screen,
                 covariance_vs_distance, fit_exponential_covariance)



DTYPE = torch.float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _radial_psd_slope(field, fit=(0.02, 0.2)):
    """Fit the slope of the radially-averaged 2-D power spectrum (log-log)."""
    B, H, W = field.shape
    f = field - field.mean(dim=(-2, -1), keepdim=True)
    Fs = torch.fft.fftshift(torch.fft.fft2(f), dim=(-2, -1))
    Pk = (Fs.abs() ** 2).mean(0)
    ky = torch.fft.fftshift(torch.fft.fftfreq(H, dtype=DTYPE))
    kx = torch.fft.fftshift(torch.fft.fftfreq(W, dtype=DTYPE))
    K = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    m = (K > fit[0]) & (K < fit[1])
    A = torch.stack([torch.log(K[m]), torch.ones(int(m.sum()), dtype=DTYPE)], dim=1)
    sol = torch.linalg.lstsq(A, torch.log(Pk[m] + 1e-30)[:, None]).solution
    return sol[0, 0].item()


def _lag1_autocorr(field):
    f = field - field.mean(dim=(-2, -1), keepdim=True)
    v = f.var(dim=(-2, -1))
    a = (f[:, 1:, :] * f[:, :-1, :]).mean(dim=(-2, -1)) / v
    return a.mean().item()


def _random_dem(B=2, H=48, W=48, seed=0, relief=1500.0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, H, W, generator=g, dtype=DTYPE) * relief + 200.0


# --------------------------------------------------------------------------- #
# Turbulent: spectrum
# --------------------------------------------------------------------------- #
class TestTurbulentSpectrum:
    @pytest.mark.parametrize("beta", [2.0, 8.0 / 3.0, 3.0])
    def test_powerlaw_psd_slope(self, beta):
        torch.manual_seed(0)
        f = turbulent_aps(96, 128, 128, rms=1.0, model="powerlaw", beta=beta)
        assert abs(_radial_psd_slope(f) - (-beta)) < 0.2

    def test_exponential_smoother_with_longer_correlation(self):
        # same white noise, only the filter changes -> isolates the spectrum
        noise = torch.randn(32, 96, 96,
                            generator=torch.Generator().manual_seed(0), dtype=DTYPE)

        def ac(L):
            m = TurbulentAPS(96, 96, model="exponential", correlation_length=L)
            return _lag1_autocorr(m(32, noise=noise))

        a3, a10, a30 = ac(3.0), ac(10.0), ac(30.0)
        assert a3 < a10 < a30

    def test_spectral_filter_zeroes_dc(self):
        h = spectral_filter(32, 32, model="powerlaw")
        assert h[0, 0].item() == 0.0           # zero-mean field
        assert torch.isfinite(h).all()


# --------------------------------------------------------------------------- #
# Turbulent: amplitude
# --------------------------------------------------------------------------- #
class TestTurbulentAmplitude:
    def test_rms_scalar_is_exact(self):
        f = turbulent_aps(8, 64, 64, rms=2.5,
                          generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(f.std(dim=(-2, -1)),
                                   torch.full((8,), 2.5, dtype=DTYPE),
                                   rtol=1e-6, atol=1e-6)

    def test_rms_per_image_tensor(self):
        rms = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=DTYPE)
        f = turbulent_aps(4, 48, 48, rms=rms,
                          generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(f.std(dim=(-2, -1)), rms, rtol=1e-6, atol=1e-6)

    def test_zero_mean(self):
        f = turbulent_aps(6, 64, 64, rms=1.0,
                          generator=torch.Generator().manual_seed(0))
        assert f.mean(dim=(-2, -1)).abs().max().item() < 1e-10


# --------------------------------------------------------------------------- #
# Turbulent: batching / determinism
# --------------------------------------------------------------------------- #
class TestTurbulentBatching:
    def test_shape(self):
        assert turbulent_aps(5, 32, 48).shape == (5, 32, 48)

    def test_generator_deterministic(self):
        f1 = turbulent_aps(4, 64, 64, generator=torch.Generator().manual_seed(7))
        f2 = turbulent_aps(4, 64, 64, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(f1, f2)

    def test_supplied_noise_is_reproducible(self):
        noise = torch.randn(3, 64, 64,
                            generator=torch.Generator().manual_seed(1), dtype=DTYPE)
        m = TurbulentAPS(64, 64)
        torch.testing.assert_close(m(3, noise=noise), m(3, noise=noise))

    def test_realizations_in_a_batch_differ(self):
        f = turbulent_aps(4, 64, 64, generator=torch.Generator().manual_seed(0))
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(f[i], f[j])


# --------------------------------------------------------------------------- #
# Cholesky reference
# --------------------------------------------------------------------------- #
class TestCholeskyReference:
    def test_reproduces_exponential_covariance(self):
        rows = cols = 20
        maxvar, alpha, N = 2.0, 1.0 / 4.0, 4000
        x = correlated_noise_cholesky(rows, cols, maxvar, alpha, N=N,
                                      generator=torch.Generator().manual_seed(0))
        assert x.shape == (N, rows, cols)

        n = rows * cols
        emp = torch.cov(x.reshape(N, n).T)             # [n, n]
        yy, xx = torch.meshgrid(torch.arange(rows, dtype=DTYPE),
                                torch.arange(cols, dtype=DTYPE), indexing="ij")
        xv, yv = xx.reshape(n), yy.reshape(n)
        r = torch.sqrt((xv[:, None] - xv[None, :]) ** 2
                       + (yv[:, None] - yv[None, :]) ** 2)
        vcm = maxvar * torch.exp(-alpha * r)
        assert (emp - vcm).abs().max().item() / maxvar < 0.15


# --------------------------------------------------------------------------- #
# Stratified
# --------------------------------------------------------------------------- #
class TestStratified:
    def test_linear_proportional_to_coeff(self):
        dem = _random_dem()
        s1 = stratified_aps(dem, torch.tensor([3e-3, -2e-3]), model="linear")
        s2 = stratified_aps(dem, torch.tensor([6e-3, -4e-3]), model="linear")
        torch.testing.assert_close(s2, 2 * s1)

    def test_linear_zero_coeff_is_zero(self):
        s = stratified_aps(_random_dem(), torch.zeros(2, dtype=DTYPE), model="linear")
        assert s.abs().max().item() == 0.0

    def test_linear_perfectly_correlated_with_dem(self):
        dem = _random_dem(B=1)
        pos = stratified_aps(dem, torch.tensor([4e-3]), model="linear")
        neg = stratified_aps(dem, torch.tensor([-4e-3]), model="linear")
        cp = torch.corrcoef(torch.stack([pos[0].reshape(-1), dem[0].reshape(-1)]))[0, 1]
        cn = torch.corrcoef(torch.stack([neg[0].reshape(-1), dem[0].reshape(-1)]))[0, 1]
        assert cp.item() > 0.999999 and cn.item() < -0.999999

    def test_linear_zero_mean(self):
        s = stratified_aps(_random_dem(), torch.tensor([3e-3, -2e-3]), model="linear")
        assert s.mean(dim=(-2, -1)).abs().max().item() < 1e-8

    def test_linear_invariant_to_constant_dem_offset(self):
        dem = _random_dem()
        coeff = torch.tensor([3e-3, -2e-3])
        s0 = stratified_aps(dem, coeff, model="linear")
        s1 = stratified_aps(dem + 500.0, coeff, model="linear")   # reference="mean"
        torch.testing.assert_close(s0, s1)

    def test_exponential_proportional_to_coeff(self):
        dem = _random_dem()
        s1 = stratified_aps(dem, torch.tensor([2.0, -1.0]), model="exponential")
        s2 = stratified_aps(dem, torch.tensor([4.0, -2.0]), model="exponential")
        torch.testing.assert_close(s2, 2 * s1)

    def test_exponential_anticorrelated_with_elevation(self):
        dem = _random_dem(B=1)
        s = stratified_aps(dem, torch.tensor([2.0]), model="exponential",
                           scale_height=6000.0)
        c = torch.corrcoef(torch.stack([s[0].reshape(-1), dem[0].reshape(-1)]))[0, 1]
        assert c.item() < 0          # higher topography -> less delay

    def test_exponential_reduces_to_linear_for_large_scale_height(self):
        dem = _random_dem(B=1)
        Hs, A = 1e9, 3.0       # exp(-h/Hs) ~ 1 - h/Hs  =>  slope -A/Hs
        s_exp = stratified_aps(dem, torch.tensor([A]), model="exponential",
                               scale_height=Hs)
        s_lin = stratified_aps(dem, torch.tensor([-A / Hs]), model="linear")
        torch.testing.assert_close(s_exp, s_lin, rtol=1e-3, atol=1e-9)


# --------------------------------------------------------------------------- #
# Stratified: shapes / batching
# --------------------------------------------------------------------------- #
class TestStratifiedShapes:
    def test_dem_2d_broadcasts_with_coeff(self):
        dem = torch.rand(40, 50, dtype=DTYPE) * 1000
        s = stratified_aps(dem, torch.tensor([1e-3, -1e-3, 2e-3]), model="linear")
        assert s.shape == (3, 40, 50)

    def test_dem_3d_per_image(self):
        dem = torch.rand(3, 40, 50, dtype=DTYPE) * 1000
        s = stratified_aps(dem, torch.tensor([1e-3, -1e-3, 2e-3]), model="linear")
        assert s.shape == (3, 40, 50)

    def test_scalar_coeff(self):
        s = stratified_aps(torch.rand(2, 30, 30, dtype=DTYPE) * 1000, 2e-3,
                           model="linear")
        assert s.shape == (2, 30, 30)

    def test_module_wrapper_matches_function(self):
        dem = _random_dem()
        coeff = torch.tensor([2e-3, -1e-3])
        mod = StratifiedAPS(model="linear")
        torch.testing.assert_close(mod(dem, coeff),
                                   stratified_aps(dem, coeff, model="linear"))

    def test_batch_mismatch_raises(self):
        dem = torch.rand(3, 20, 20, dtype=DTYPE) * 1000
        with pytest.raises(ValueError):
            stratified_aps(dem, torch.tensor([1e-3, 2e-3]), model="linear")


# --------------------------------------------------------------------------- #
# Sampler & combination
# --------------------------------------------------------------------------- #
class TestSamplerAndCombine:
    def test_sampler_linear_range_signed(self):
        c = sample_stratified_coeff(2000, model="linear", k_range=(-5e-3, 5e-3),
                                    generator=torch.Generator().manual_seed(0))
        assert c.shape == (2000,) and c.dtype == DTYPE
        assert c.min().item() >= -5e-3 and c.max().item() <= 5e-3
        assert (c < 0).any() and (c > 0).any()

    def test_sampler_exponential_range(self):
        c = sample_stratified_coeff(2000, model="exponential", a_range=(-3.0, 3.0),
                                    generator=torch.Generator().manual_seed(0))
        assert c.min().item() >= -3.0 and c.max().item() <= 3.0

    def test_sampler_deterministic(self):
        a = sample_stratified_coeff(64, generator=torch.Generator().manual_seed(1))
        b = sample_stratified_coeff(64, generator=torch.Generator().manual_seed(1))
        assert torch.equal(a, b)

    def test_combine_is_sum(self):
        turb = turbulent_aps(2, 40, 40, rms=1.0,
                             generator=torch.Generator().manual_seed(0))
        dem = torch.rand(2, 40, 40, dtype=DTYPE) * 1000
        coeff = torch.tensor([2e-3, -1e-3])
        full = atmospheric_phase_screen(turb, dem, coeff, model="linear")
        torch.testing.assert_close(full, turb + stratified_aps(dem, coeff, model="linear"))
        assert full.shape == (2, 40, 40)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiability:
    def test_gradcheck_turbulent_wrt_noise_and_rms(self):
        m = TurbulentAPS(8, 8, model="powerlaw")
        noise = torch.randn(1, 8, 8, dtype=DTYPE, requires_grad=True)
        rms = torch.tensor([1.5], dtype=DTYPE, requires_grad=True)

        def f(noise, rms):
            return m(1, rms=rms, noise=noise)

        assert torch.autograd.gradcheck(f, (noise, rms))

    def test_gradcheck_stratified_linear(self):
        dem = (torch.rand(2, 5, 5, dtype=DTYPE) * 1000).requires_grad_(True)
        coeff = torch.tensor([3e-3, -2e-3], dtype=DTYPE, requires_grad=True)

        def f(dem, coeff):
            return stratified_aps(dem, coeff, model="linear")

        assert torch.autograd.gradcheck(f, (dem, coeff))

    def test_gradcheck_stratified_exponential_coeff(self):
        dem = torch.rand(2, 5, 5, dtype=DTYPE) * 1000
        coeff = torch.tensor([2.0, -1.0], dtype=DTYPE, requires_grad=True)

        def f(coeff):
            return stratified_aps(dem, coeff, model="exponential")

        assert torch.autograd.gradcheck(f, (coeff,))

    def test_grad_flows_through_combined_screen(self):
        noise = torch.randn(2, 16, 16, dtype=DTYPE, requires_grad=True)
        rms = torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True)
        dem = (torch.rand(2, 16, 16, dtype=DTYPE) * 1000).requires_grad_(True)
        coeff = torch.tensor([2e-3, -1e-3], dtype=DTYPE, requires_grad=True)
        m = TurbulentAPS(16, 16)
        full = atmospheric_phase_screen(m(2, rms=rms, noise=noise), dem, coeff,
                                        model="linear")
        full.pow(2).sum().backward()
        for name, p in (("noise", noise), ("rms", rms), ("dem", dem), ("coeff", coeff)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert coeff.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
class TestDtypeAndDevice:
    def test_turbulent_dtype(self):
        assert turbulent_aps(2, 32, 32, dtype=torch.float32).dtype == torch.float32

    def test_stratified_dtype(self):
        dem = torch.rand(2, 16, 16, dtype=torch.float32) * 1000
        s = stratified_aps(dem, torch.tensor([1e-3, -1e-3]), model="linear",
                           dtype=torch.float32)
        assert s.dtype == torch.float32

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        m = TurbulentAPS(64, 64, device="cuda")
        f = m(4, rms=1.0)
        assert f.device.type == "cuda" and torch.isfinite(f).all()
        dem = torch.rand(4, 64, 64, dtype=DTYPE, device="cuda") * 1000
        s = stratified_aps(dem, torch.tensor([1e-3], device="cuda"), model="linear")
        assert s.device.type == "cuda" and torch.isfinite(s).all()


# --------------------------------------------------------------------------- #
# orbital ramp
# --------------------------------------------------------------------------- #
class TestOrbitalRamp:
    def test_shape_zero_mean_and_rms(self):
        g = torch.Generator().manual_seed(0)
        f = orbital_ramp(4, 32, 48, rms=2.0, generator=g, dtype=DTYPE)
        assert f.shape == (4, 32, 48)
        assert torch.allclose(f.mean(dim=(-2, -1)),
                              torch.zeros(4, dtype=DTYPE), atol=1e-9)
        std = f.std(dim=(-2, -1))
        assert torch.allclose(std, torch.full((4,), 2.0, dtype=DTYPE), atol=1e-6)

    def test_per_image_rms_tensor(self):
        rms = torch.tensor([0.5, 3.0], dtype=DTYPE)
        f = orbital_ramp(2, 24, 24, rms=rms, generator=torch.Generator().manual_seed(1))
        assert torch.allclose(f.std(dim=(-2, -1)), rms, atol=1e-6)

    def test_order1_is_planar(self):
        """order=1 -> affine field: second differences vanish along both axes."""
        f = orbital_ramp(3, 20, 20, order=1, generator=torch.Generator().manual_seed(2),
                         dtype=DTYPE)
        d2x = f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]
        d2y = f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]
        assert d2x.abs().max() < 1e-9
        assert d2y.abs().max() < 1e-9

    def test_order2_has_curvature(self):
        f = orbital_ramp(8, 20, 20, order=2, generator=torch.Generator().manual_seed(3),
                         dtype=DTYPE)
        d2x = f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]
        assert d2x.abs().max() > 1e-6           # some image is genuinely curved

    def test_generator_determinism(self):
        a = orbital_ramp(2, 16, 16, generator=torch.Generator().manual_seed(7))
        b = orbital_ramp(2, 16, 16, generator=torch.Generator().manual_seed(7))
        assert torch.allclose(a, b)

    def test_explicit_coeff_is_used(self):
        coeff = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)   # x-ramp, y-ramp
        f = orbital_ramp(2, 16, 16, coeff=coeff)
        # image 0 varies along x only, image 1 along y only
        assert f[0].std(dim=0).max() < 1e-9      # constant down columns
        assert f[1].std(dim=1).max() < 1e-9      # constant along rows

    def test_bad_order_and_coeff_shape_raise(self):
        with pytest.raises(ValueError, match="order must be"):
            orbital_ramp(1, 8, 8, order=3)
        with pytest.raises(ValueError, match="coeff must have shape"):
            orbital_ramp(2, 8, 8, order=1, coeff=torch.zeros(2, 5))

    def test_differentiable_in_rms(self):
        rms = torch.tensor([1.5], dtype=DTYPE, requires_grad=True)
        f = orbital_ramp(1, 12, 12, rms=rms, generator=torch.Generator().manual_seed(4))
        f.pow(2).mean().backward()
        assert rms.grad is not None and torch.isfinite(rms.grad).all()

    def test_folds_into_atmospheric_phase_screen(self):
        g = torch.Generator().manual_seed(0)
        turb = turbulent_aps(2, 32, 32, rms=1.0, generator=g)
        dem = torch.rand(2, 32, 32, dtype=DTYPE, generator=g) * 1000
        coeff = torch.tensor([2e-3, -2e-3], dtype=DTYPE)
        ramp = orbital_ramp(2, 32, 32, rms=3.0, generator=g)

        without = atmospheric_phase_screen(turb, dem, coeff, model="linear")
        with_ramp = atmospheric_phase_screen(turb, dem, coeff, model="linear",
                                             orbital=ramp)
        assert torch.allclose(with_ramp, without + ramp)


# --------------------------------------------------------------------------- #
# hole-effect covariance models (Cholesky path)
# --------------------------------------------------------------------------- #
class TestHoleEffectModels:
    def test_expcos_runs(self):
        # PD only for a gentle oscillation: beta <= alpha (period >= 2*pi/alpha)
        g = torch.Generator().manual_seed(0)
        f = correlated_noise_cholesky(20, 20, 3.0, 1 / 3000.0, N=4, model="expcos",
                                      beta=2 * math.pi / 25000.0, psizex=500.0,
                                      psizey=500.0, generator=g, jitter=1e-4)
        assert torch.isfinite(f).all()

    def test_ebessel_runs(self):
        g = torch.Generator().manual_seed(0)
        f = correlated_noise_cholesky(24, 24, 3.0, 1 / 5000.0, N=4, model="ebessel",
                                      bessel_w=30000.0, psizex=500.0, psizey=500.0,
                                      generator=g, jitter=1e-3)
        assert torch.isfinite(f).all()

    def test_hole_effect_differs_from_exponential(self):
        g = lambda: torch.Generator().manual_seed(0)
        exp = correlated_noise_cholesky(20, 20, 3.0, 1 / 5000.0, N=4, psizex=500.0,
                                        psizey=500.0, generator=g())
        eb = correlated_noise_cholesky(20, 20, 3.0, 1 / 5000.0, N=4, model="ebessel",
                                       bessel_w=30000.0, psizex=500.0, psizey=500.0,
                                       generator=g(), jitter=1e-3)
        assert not torch.allclose(exp, eb)

    def test_missing_params_and_unknown_model_raise(self):
        with pytest.raises(ValueError, match="expcos"):
            correlated_noise_cholesky(16, 16, 1.0, 1 / 1000.0, N=1, model="expcos")
        with pytest.raises(ValueError, match="ebessel"):
            correlated_noise_cholesky(16, 16, 1.0, 1 / 1000.0, N=1, model="ebessel")
        with pytest.raises(ValueError, match="unknown model"):
            correlated_noise_cholesky(16, 16, 1.0, 1 / 1000.0, N=1, model="bogus")


# --------------------------------------------------------------------------- #
# covariance estimation
# --------------------------------------------------------------------------- #
class TestCovarianceEstimation:
    def _field(self, cl, N=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        return correlated_noise_cholesky(48, 48, 4.0, 1 / cl, N=N,
                                         psizex=300.0, psizey=300.0, generator=g)

    def test_covariance_curve_shape(self):
        f = self._field(3000.0)
        d, cov = covariance_vs_distance(f, psizex=300.0, psizey=300.0)
        assert d.shape[0] == cov.shape[1]
        assert cov.shape[0] == f.shape[0]
        # zero-lag bin ~ variance, and covariance decreases with distance
        cov_m = cov.mean(0)
        var = (f - f.mean(dim=(-2, -1), keepdim=True)).pow(2).mean()
        assert cov_m[0].item() == pytest.approx(var.item(), rel=0.05)
        assert cov_m[0] > cov_m[5] > cov_m[-1]

    def test_roundtrip_recovers_parameters(self):
        """demean=False on a zero-mean exponential field recovers (var, corr_len)."""
        f = self._field(3000.0, N=32)
        var, cl = fit_exponential_covariance(f, psizex=300.0, psizey=300.0, demean=False)
        assert var.mean().item() == pytest.approx(4.0, rel=0.3)
        assert cl.mean().item() == pytest.approx(3000.0, rel=0.3)

    def test_estimate_is_monotonic_in_true_correlation_length(self):
        cls = [1500.0, 3000.0, 6000.0]
        ests = []
        for cl in cls:
            _, est = fit_exponential_covariance(self._field(cl, N=32), psizex=300.0,
                                                psizey=300.0, demean=False)
            ests.append(est.mean().item())
        assert ests[0] < ests[1] < ests[2]

    def test_accepts_2d_input(self):
        f = self._field(3000.0, N=1)[0]            # [H, W]
        d, cov = covariance_vs_distance(f, psizex=300.0, psizey=300.0)
        assert cov.shape == d.shape                 # 1-D, no batch axis
        var, cl = fit_exponential_covariance(f, psizex=300.0, psizey=300.0)
        assert var.ndim == 0 and cl.ndim == 0

    def test_nls_is_more_accurate_than_loglinear(self):
        """The default NLS fit recovers corr_len far closer than log-linear."""
        true_cl = 3000.0
        f = self._field(true_cl, N=32)
        _, cl_nls = fit_exponential_covariance(
            f, psizex=300.0, psizey=300.0, demean=False, method="nls")
        _, cl_log = fit_exponential_covariance(
            f, psizex=300.0, psizey=300.0, demean=False, method="loglinear")
        err_nls = abs(cl_nls.mean().item() - true_cl) / true_cl
        err_log = abs(cl_log.mean().item() - true_cl) / true_cl
        assert err_nls < 0.1            # within ~10% on the exact-covariance ref
        assert err_nls < err_log        # and strictly better than the legacy fit

    def test_nls_variance_is_zero_lag_sample_variance(self):
        """NLS reports the unbiased sill, not the fitted nuisance amplitude."""
        f = self._field(3000.0, N=16)
        var, _ = fit_exponential_covariance(
            f, psizex=300.0, psizey=300.0, demean=False, method="nls")
        sample_var = f.pow(2).mean(dim=(-2, -1))
        assert torch.allclose(var, sample_var.clamp_min(1e-12))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            fit_exponential_covariance(self._field(3000.0, N=1), method="bogus")

    def test_nls_fit_is_differentiable(self):
        f = self._field(3000.0, N=2).clone().requires_grad_(True)
        var, cl = fit_exponential_covariance(f, psizex=300.0, psizey=300.0,
                                             demean=False, method="nls")
        (var.sum() + cl.sum()).backward()
        assert f.grad is not None and torch.isfinite(f.grad).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
