"""
Tests for ``PennySource`` (the Fialko penny-shaped crack / sill source).

Unlike the Mogi point source there is no elementary closed form, so correctness
is pinned down three ways:

* **Quadrature units** – ``build_quadrature`` reproduces 16-pt Gauss-Legendre and
  integrates polynomials on [0, 1] to machine precision.
* **Physical invariants** (exact, framework-internal) – axisymmetry, horizontal
  displacement is purely radial, linearity in pressure, proportionality to
  ``(1-nu)`` and ``1/mu``, and length self-similarity (scaling all geometry by L
  scales displacement by L).
* **Independent physics** – directly above a *deep* crack the vertical
  displacement converges to the Mogi point source with the known penny-crack
  volume change ``dV = 16(1-nu) a^3 P / (3 mu)``; plus golden values produced by
  a separate NumPy port (itself cross-checked against that Mogi limit) guard
  against silent numerical regressions.

Plus the library's core requirements: batchability and differentiability
(gradients flow through the Fredholm linear solve).

Run with::

    pytest test_penny_source.py -v

"""
import math

import pytest
import torch

from torchdeform import PennySource, Displacement
from torchdeform.sources.penny import WEIGHT16, ROOT16, build_quadrature


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPE = torch.float64


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _t(data, device="cpu", dtype=DTYPE):
    return torch.tensor(data, dtype=dtype, device=device)


def make_inputs(B, N, *, device="cpu", dtype=DTYPE, scale="real",
                requires_grad=False, seed=0):
    """Random, well-posed problem.

    Observation points are scattered at dimensionless radial distances in
    roughly [0.3, 4] crack radii around each source, so they avoid the
    on-axis r->0 region. ``scale='unit'`` keeps every quantity O(1) for
    well-conditioned finite-difference gradient checks (pair with a
    ``shear_modulus=1.0`` model so displacements are O(1) too).
    """
    g = torch.Generator().manual_seed(seed)

    def u(shape, lo, hi):
        return torch.rand(shape, generator=g, dtype=DTYPE) * (hi - lo) + lo

    if scale == "real":
        radius = u((B,), 500.0, 1500.0)
        depth = u((B,), 1000.0, 4000.0)
        source_x = u((B,), -500.0, 500.0)
        source_y = u((B,), -500.0, 500.0)
        pressure = (u((B,), 1.0, 5.0) * torch.sign(u((B,), -1, 1))) * 1e6
    else:  # unit
        radius = u((B,), 0.8, 1.2)
        depth = u((B,), 1.0, 2.0)
        source_x = u((B,), -0.3, 0.3)
        source_y = u((B,), -0.3, 0.3)
        pressure = u((B,), 0.5, 2.0) * torch.sign(u((B,), -1, 1))

    rho = u((B, N), 0.3, 4.0) * radius[:, None]      # metric radial distance
    theta = u((B, N), 0.0, 2 * math.pi)
    x_obs = source_x[:, None] + rho * torch.cos(theta)
    y_obs = source_y[:, None] + rho * torch.sin(theta)

    out = [x_obs, y_obs, source_x, source_y, depth, radius, pressure]
    out = [o.to(device=device, dtype=dtype) for o in out]
    if requires_grad:
        for o in out:
            o.requires_grad_(True)
    return out


# Golden values from an independent NumPy implementation of the same model,
# cross-validated against the deep-crack Mogi limit. Config A: h=d/a~2.78,
# inflation. Config B: shallow (h=0.4) deflation. (poisson=0.25, mu=3e10, nis=2)
GOLDEN = {
    "A": {
        "inputs": dict(x=[[0.0, 1500.0, 4000.0]], y=[[0.0, 0.0, 3000.0]],
                       sx=[100.0], sy=[-50.0], depth=[2500.0],
                       radius=[900.0], P=[5e6]),
        "e": [-0.0005927754970410022, 0.004779776795032082, 0.0005618252515434544],
        "n": [0.0002963877485205011, 0.00017070631410828863, 0.00043937615825834263],
        "u": [0.016726580851621632, 0.009392070165268739, 0.0003698510838914162],
    },
    "B": {
        "inputs": dict(x=[[0.0, 800.0, 2500.0]], y=[[0.0, 600.0, 0.0]],
                       sx=[0.0], sy=[0.0], depth=[400.0],
                       radius=[1000.0], P=[-2e6]),
        "e": [-0.0, -0.02731738010165471, -0.000714619272318414],
        "n": [-0.0, -0.020488035076241033, -0.0],
        "u": [-0.23535867461976673, -0.032466156269246864, -3.260596794711587e-05],
    },
}


# --------------------------------------------------------------------------- #
# Quadrature units
# --------------------------------------------------------------------------- #
class TestQuadrature:
    def test_gl16_nodes_and_weights(self):
        # symmetric nodes, weights sum to 2 on [-1, 1]
        assert torch.allclose(ROOT16, -ROOT16.flip(0), atol=1e-15)
        assert abs(float(WEIGHT16.sum()) - 2.0) < 1e-13
        # optional cross-check against numpy's reference GL16 nodes/weights
        np = pytest.importorskip("numpy")
        ref_r, ref_w = (torch.as_tensor(v, dtype=DTYPE)
                        for v in np.polynomial.legendre.leggauss(16))
        assert torch.allclose(ROOT16, ref_r, atol=1e-14)
        assert torch.allclose(WEIGHT16, ref_w, atol=1e-14)

    @pytest.mark.parametrize("nis", [1, 2, 4])
    def test_build_quadrature_integrates_polynomials(self, nis):
        t, Wt = build_quadrature(nis, ROOT16, WEIGHT16, device="cpu", dtype=DTYPE)
        assert t.shape == (16 * nis,) and Wt.shape == (16 * nis,)
        assert torch.all((t > 0) & (t < 1))            # nodes inside (0, 1)
        assert abs(float(Wt.sum()) - 1.0) < 1e-13      # |[0,1]| = 1
        for k in range(8):                             # exact for x^k
            approx = float((Wt * t**k).sum())
            assert abs(approx - 1.0 / (k + 1)) < 1e-12


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
class TestCorrectness:
    @pytest.mark.parametrize("cfg", ["A", "B"])
    def test_golden_regression(self, cfg):
        g = GOLDEN[cfg]; ins = g["inputs"]
        model = PennySource()        # poisson=0.25, mu=3e10, nis=2 (defaults)
        out = model(_t(ins["x"]), _t(ins["y"]), _t(ins["sx"]), _t(ins["sy"]),
                    _t(ins["depth"]), _t(ins["radius"]), _t(ins["P"]))
        torch.testing.assert_close(out.e[0], _t(g["e"]), rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(out.n[0], _t(g["n"]), rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(out.u[0], _t(g["u"]), rtol=1e-6, atol=1e-10)

    def test_returns_dataclass_with_shapes(self):
        B, N = 3, 12
        out = PennySource()(*make_inputs(B, N))
        assert isinstance(out, Displacement)
        for comp in (out.e, out.n, out.u):
            assert comp.shape == (B, N)
        assert torch.isfinite(out.e).all() and torch.isfinite(out.u).all()

    def test_axisymmetry(self):
        # three points at identical radial distance, different azimuth
        model = PennySource()
        x = _t([[1000.0, 0.0, -700.0]])
        y = _t([[0.0, 1000.0, math.sqrt(1000.0**2 - 700.0**2)]])
        z = _t([0.0])
        out = model(x, y, z, z.clone(), _t([2000.0]), _t([800.0]), _t([1e6]))
        torch.testing.assert_close(out.u, torch.full_like(out.u, out.u[0, 0].item()))
        horiz = torch.sqrt(out.e**2 + out.n**2)
        torch.testing.assert_close(horiz, torch.full_like(horiz, horiz[0, 0].item()))

    def test_horizontal_is_radial(self):
        # horizontal displacement vector parallel to (dx, dy): cross product ~ 0
        model = PennySource()
        x = _t([[1200.0, -800.0, 300.0]]); y = _t([[400.0, 900.0, -1500.0]])
        sx, sy = _t([100.0]), _t([-50.0])
        out = model(x, y, sx, sy, _t([1500.0]), _t([700.0]), _t([3e6]))
        dx = x - sx[:, None]; dy = y - sy[:, None]
        cross = out.e * dy - out.n * dx
        assert cross.abs().max() < 1e-9 * (out.e.abs().max() + 1e-12)

    def test_inflation_uplift_deflation_subsidence(self):
        model = PennySource()
        z = _t([0.0]); ctr_x = _t([[0.0]]); ctr_y = _t([[0.0]])
        infl = model(ctr_x, ctr_y, z, z.clone(), _t([2000.0]), _t([800.0]), _t([1e6]))
        defl = model(ctr_x, ctr_y, z, z.clone(), _t([2000.0]), _t([800.0]), _t([-1e6]))
        assert infl.u.item() > 0          # inflation lifts the centre
        assert defl.u.item() < 0          # deflation drops it
        # and inflation pushes outward: east point moves east
        east = model(_t([[500.0]]), ctr_y, z, z.clone(), _t([2000.0]), _t([800.0]), _t([1e6]))
        assert east.e.item() > 0

    def test_deep_crack_converges_to_mogi_on_axis(self):
        # Directly above a deep horizontal crack, uu -> Mogi point source with
        # dV = 16(1-nu) a^3 P / (3 mu). (Off-axis it differs: a sill carries a
        # CLVD component, so this equivalence is on-axis only.)
        v, mu, a, P = 0.25, 3e10, 500.0, 1e6
        model = PennySource(poisson_ratio=v, shear_modulus=mu)
        dV = 16.0 * (1 - v) * a**3 * P / (3.0 * mu)
        ratios = []
        for d in (5000.0, 10000.0, 20000.0, 40000.0):
            out = model(_t([[0.0]]), _t([[0.0]]), _t([0.0]), _t([0.0]),
                        _t([d]), _t([a]), _t([P]))
            mogi = (1 - v) / math.pi * dV / d**2     # on-axis: R = d
            ratios.append(out.u.item() / mogi)
        assert all(0.0 < r < 1.0 for r in ratios)            # approaches from below
        assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
        assert ratios[-1] > 0.999                            # ~0.9998 at h=80


# --------------------------------------------------------------------------- #
# Linearity & scaling invariants
# --------------------------------------------------------------------------- #
class TestLinearityAndScaling:
    def test_linear_in_pressure(self):
        model = PennySource()
        x, y, sx, sy, d, a, P = make_inputs(3, 15)
        single = model(x, y, sx, sy, d, a, P)
        double = model(x, y, sx, sy, d, a, 2 * P)
        torch.testing.assert_close(double.e, 2 * single.e)
        torch.testing.assert_close(double.u, 2 * single.u)
        zero = model(x, y, sx, sy, d, a, torch.zeros_like(P))
        assert zero.e.abs().max() == 0 and zero.u.abs().max() == 0

    @pytest.mark.parametrize("nu", [0.0, 0.1, 0.3, 0.49])
    def test_proportional_to_one_minus_nu(self, nu):
        args = make_inputs(2, 10)
        base = PennySource(poisson_ratio=0.25)(*args)
        other = PennySource(poisson_ratio=nu)(*args)
        ratio = (1.0 - nu) / (1.0 - 0.25)
        torch.testing.assert_close(other.u, ratio * base.u)
        torch.testing.assert_close(other.e, ratio * base.e)

    def test_inversely_proportional_to_shear_modulus(self):
        args = make_inputs(2, 10)
        base = PennySource(shear_modulus=3e10)(*args)
        stiff = PennySource(shear_modulus=6e10)(*args)
        torch.testing.assert_close(stiff.u, 0.5 * base.u)
        torch.testing.assert_close(stiff.e, 0.5 * base.e)

    def test_length_self_similarity(self):
        # scale ALL geometry by L (depth, radius, source, obs) at fixed pressure
        # => displacement scales by L. Exercises the whole dimensionless pipeline.
        model = PennySource()
        x, y, sx, sy, d, a, P = make_inputs(3, 12)
        base = model(x, y, sx, sy, d, a, P)
        L = 3.7
        scaled = model(L * x, L * y, L * sx, L * sy, L * d, L * a, P)
        torch.testing.assert_close(scaled.u, L * base.u, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(scaled.e, L * base.e, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
class TestBatching:
    @pytest.mark.parametrize("B", [1, 2, 5])
    def test_batched_equals_single_loop(self, B):
        model = PennySource()
        x, y, sx, sy, d, a, P = make_inputs(B, 20)
        batched = model(x, y, sx, sy, d, a, P)
        for b in range(B):
            one = model(x[b:b + 1], y[b:b + 1], sx[b:b + 1], sy[b:b + 1],
                        d[b:b + 1], a[b:b + 1], P[b:b + 1])
            torch.testing.assert_close(one.e, batched.e[b:b + 1])
            torch.testing.assert_close(one.n, batched.n[b:b + 1])
            torch.testing.assert_close(one.u, batched.u[b:b + 1])

    def test_heterogeneous_params_across_batch(self):
        # very different h = depth/radius per element solved together
        model = PennySource()
        x, y, _, _, _, _, _ = make_inputs(3, 10)
        sx = sy = torch.zeros(3, dtype=DTYPE)
        depth = _t([400.0, 2500.0, 9000.0])
        radius = _t([1000.0, 900.0, 450.0])     # h = 0.4, 2.78, 20
        P = _t([1e6, -3e6, 2e6])
        batched = model(x, y, sx, sy, depth, radius, P)
        for b in range(3):
            one = model(x[b:b + 1], y[b:b + 1], sx[b:b + 1], sy[b:b + 1],
                        depth[b:b + 1], radius[b:b + 1], P[b:b + 1])
            torch.testing.assert_close(one.u, batched.u[b:b + 1])

    def test_batch_independence(self):
        model = PennySource()
        x, y, sx, sy, d, a, P = make_inputs(4, 15)
        base = model(x, y, sx, sy, d, a, P)
        P2 = P.clone(); P2[2] *= -5.0           # perturb only row 2
        out2 = model(x, y, sx, sy, d, a, P2)
        for b in range(4):
            changed = not torch.allclose(out2.u[b], base.u[b])
            assert changed == (b == 2)

    @pytest.mark.parametrize("N", [1, 50])
    def test_various_n(self, N):
        out = PennySource()(*make_inputs(2, N))
        assert out.e.shape == (2, N)


# --------------------------------------------------------------------------- #
# Differentiability  (gradients flow through the Fredholm linear solve)
# --------------------------------------------------------------------------- #
class TestDifferentiability:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_source_params(self, device):
        # O(1) geometry + mu=1 so outputs are O(1): well-conditioned FD jacobian.
        model = PennySource(shear_modulus=1.0)
        x, y, sx, sy, d, a, P = make_inputs(2, 3, device=device, scale="unit")
        for t in (sx, sy, d, a, P):
            t.requires_grad_(True)

        def f(sx, sy, d, a, P):
            o = model(x, y, sx, sy, d, a, P)
            return o.e, o.n, o.u

        assert torch.autograd.gradcheck(f, (sx, sy, d, a, P), eps=1e-6,
                                        atol=1e-5, rtol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_observation_coords(self, device):
        model = PennySource(shear_modulus=1.0)
        x, y, sx, sy, d, a, P = make_inputs(2, 3, device=device, scale="unit")
        x.requires_grad_(True); y.requires_grad_(True)

        def f(x, y):
            o = model(x, y, sx, sy, d, a, P)
            return o.e, o.n, o.u

        assert torch.autograd.gradcheck(f, (x, y), eps=1e-6, atol=1e-5, rtol=1e-3)

    def test_gradgradcheck_small(self):
        # second order through the linear solve (small system, nis=1).
        # If your torch build lacks double-backward for linalg.solve, drop this.
        model = PennySource(shear_modulus=1.0, nis=1)
        x, y, sx, sy, d, a, P = make_inputs(1, 2, scale="unit")
        d.requires_grad_(True); a.requires_grad_(True)

        def f(d, a):
            return model(x, y, sx, sy, d, a, P).u

        assert torch.autograd.gradgradcheck(f, (d, a), eps=1e-6, atol=1e-4, rtol=1e-2)

    def test_grad_flows_to_all_inputs(self):
        model = PennySource()
        x, y, sx, sy, d, a, P = make_inputs(2, 8, requires_grad=True)
        out = model(x, y, sx, sy, d, a, P)
        (out.e.sum() + out.n.sum() + out.u.sum()).backward()
        for name, t in [("x_obs", x), ("y_obs", y), ("source_x", sx),
                        ("source_y", sy), ("depth", d), ("radius", a),
                        ("pressure", P)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"
            assert t.grad.abs().sum() > 0, f"zero grad for {name}"

    def test_output_requires_grad_when_input_does(self):
        x, y, sx, sy, d, a, P = make_inputs(2, 5)
        P.requires_grad_(True)
        out = PennySource()(x, y, sx, sy, d, a, P)
        assert out.e.requires_grad and out.u.requires_grad


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
class TestDtypeAndDevice:
    def test_float32_input_promoted_to_internal_dtype(self):
        # default internal dtype is float64: float32 inputs come back float64.
        # Reference = the SAME float32 values cast up to float64 (which is exactly
        # what the model does internally), so promotion should be lossless.
        args32 = make_inputs(2, 8, dtype=torch.float32)
        args_up = [a.double() for a in args32]
        model = PennySource()
        out32 = model(*args32)
        out_ref = model(*args_up)
        assert out32.e.dtype == torch.float64
        torch.testing.assert_close(out32.u, out_ref.u, rtol=1e-10, atol=1e-13)
        torch.testing.assert_close(out32.e, out_ref.e, rtol=1e-10, atol=1e-13)

    def test_internal_dtype_float32_runs(self):
        model = PennySource(internal_dtype=torch.float32)
        out = model(*make_inputs(2, 8, dtype=torch.float32))
        assert out.e.dtype == torch.float32
        assert torch.isfinite(out.u).all()

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        model = PennySource().to("cuda")
        out = model(*make_inputs(2, 8, device="cuda"))
        assert out.u.device.type == "cuda"
        assert torch.isfinite(out.u).all()


# --------------------------------------------------------------------------- #
# External reference: original Fialko et al. (2001) MATLAB penny-crack code
# --------------------------------------------------------------------------- #
import json
from pathlib import Path

_GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "data" / "fialko_golden.json").read_text()
)


class TestFialkoReference:
    """Golden values from the original Fialko, Khazan & Simons (2001) code.

    Ground truth produced by ``tests/sources/reference/gen_fialko.m`` (run via
    MATLAB). Observation points lie on the +East radius (``y = 0``), so the
    horizontal field is purely radial: ``ue == Ur`` and ``uu == Uz``. Inputs use
    crack radius ``a`` and pressure ``P`` from the data file; ``nu``/``mu`` enter
    only through the scale ``Pf = 2(1-nu) a P / mu``.

    NOTE: the public GeodMod mirror of Fialko's ``intgr.m`` has a bug in the
    vertical displacement (its vectorized line factors ``fi`` over the ``psi``
    terms); ``gen_fialko.m`` uses the *original* loop formula, which is what
    ``PennySource`` implements. See ``reference/gen_fialko.m`` for details.
    """

    meta = _GOLDEN["meta"]

    @pytest.mark.parametrize(
        "case", _GOLDEN["cases"], ids=[f"h={c['h']}" for c in _GOLDEN["cases"]]
    )
    def test_penny_matches_matlab(self, case):
        m = self.meta
        a, P = float(m["a"]), float(m["P"])
        r = torch.tensor(m["r"], dtype=DTYPE)
        x = (r * a).reshape(1, -1)          # along +East
        y = torch.zeros_like(x)
        zero = torch.zeros(1, dtype=DTYPE)
        out = PennySource(poisson_ratio=m["nu"], shear_modulus=m["mu"])(
            x, y, zero, zero,
            torch.tensor([case["depth"]], dtype=DTYPE),
            torch.tensor([a], dtype=DTYPE),
            torch.tensor([P], dtype=DTYPE),
        )
        # +East radial line: horizontal displacement is the radial component.
        torch.testing.assert_close(out.e[0], torch.tensor(case["ur"], dtype=DTYPE), rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(out.u[0], torch.tensor(case["uz"], dtype=DTYPE), rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
