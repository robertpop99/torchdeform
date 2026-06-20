"""
Tests for ``MogiSource`` (the point-source Mogi displacement model).

Focus areas, matching the design goals of the library:

* **Correctness** – output matches an independent, un-vectorised reference and
  the closed-form Mogi solution; physical sign / symmetry / decay properties hold.
* **Batchability** – a batched call equals a loop over single sources, batch
  rows are independent, and per-point depth ``[B, N]`` broadcasting works.
* **Differentiability** – gradients flow to every continuous input, agree with
  finite differences (``torch.autograd.gradcheck``), and stay finite at the
  regularised singularity.

Run with::

    pytest test_mogi_source.py -v
"""
import math

import pytest
import torch

from torchdeform import MogiSource, Displacement


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
DTYPE = torch.float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def model():
    """Default float64 source (no learnable params, pure function)."""
    return MogiSource()


def reference_mogi_loop(x_obs, y_obs, source_x, source_y, depth, delta_v, nu=0.25):
    """Independent, fully un-vectorised reference implementation.

    Plain Python loops over ``[B, N]`` so it shares no code path with the
    vectorised/broadcast implementation under test. Returns float64 tensors.
    """
    x_obs = x_obs.double(); y_obs = y_obs.double()
    source_x = source_x.double(); source_y = source_y.double()
    delta_v = delta_v.double(); depth = depth.double()
    B, N = x_obs.shape
    ue = torch.zeros(B, N, dtype=DTYPE)
    un = torch.zeros(B, N, dtype=DTYPE)
    uu = torch.zeros(B, N, dtype=DTYPE)
    for b in range(B):
        C = (1.0 - nu) / math.pi * float(delta_v[b])
        for n in range(N):
            d = float(depth[b, n]) if depth.ndim == 2 else float(depth[b])
            dx = float(x_obs[b, n]) - float(source_x[b])
            dy = float(y_obs[b, n]) - float(source_y[b])
            r3 = (dx * dx + dy * dy + d * d) ** 1.5
            ue[b, n] = C * dx / r3
            un[b, n] = C * dy / r3
            uu[b, n] = C * d / r3
    return ue, un, uu


def make_inputs(B, N, *, device="cpu", dtype=DTYPE, depth_per_point=False,
                requires_grad=False, scale="real", seed=0):
    """Build a random, physically-plausible problem.

    ``scale='real'`` uses metre-magnitude values; ``scale='unit'`` keeps
    everything O(1) so finite-difference gradient checks stay well-conditioned.
    Depth is always strictly positive and observation points never coincide
    with the source, so inputs sit away from the (regularised) singularity.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if scale == "real":
        xy_sd, src_sd, d_lo, d_hi, dv_sd = 5_000.0, 1_000.0, 1_000.0, 5_000.0, 1e6
    else:
        xy_sd, src_sd, d_lo, d_hi, dv_sd = 1.0, 1.0, 1.0, 2.0, 1.0

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE)

    def ru(*shape):
        return torch.rand(*shape, generator=g, dtype=DTYPE)

    x_obs = (rn(B, N) * xy_sd).to(device=device, dtype=dtype)
    y_obs = (rn(B, N) * xy_sd).to(device=device, dtype=dtype)
    source_x = (rn(B) * src_sd).to(device=device, dtype=dtype)
    source_y = (rn(B) * src_sd).to(device=device, dtype=dtype)
    if depth_per_point:
        depth = (ru(B, N) * (d_hi - d_lo) + d_lo).to(device=device, dtype=dtype)
    else:
        depth = (ru(B) * (d_hi - d_lo) + d_lo).to(device=device, dtype=dtype)
    delta_v = ((ru(B) * 2 - 1) * dv_sd).to(device=device, dtype=dtype)

    tensors = [x_obs, y_obs, source_x, source_y, depth, delta_v]
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)
    return tensors


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
class TestCorrectness:
    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_reference(self, model, device):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(4, 50, device=device)
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        ue, un, uu = reference_mogi_loop(x_obs, y_obs, sx, sy, depth, dv)
        ue, un, uu = (t.to(device) for t in (ue, un, uu))
        torch.testing.assert_close(out.e, ue, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(out.n, un, rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(out.u, uu, rtol=1e-9, atol=1e-12)

    def test_returns_displacement_dataclass_with_shapes(self, model):
        B, N = 3, 17
        out = model(*make_inputs(B, N))
        assert isinstance(out, Displacement)
        for comp in (out.e, out.n, out.u):
            assert comp.shape == (B, N)

    def test_point_directly_above_source(self, model):
        # dx = dy = 0  =>  ue = un = 0 and uu = (1-nu)/pi * dV / d^2
        d, dv, nu = 2_000.0, 1e6, 0.25
        x = torch.zeros(1, 1, dtype=DTYPE)
        out = model(x, x.clone(), torch.zeros(1, dtype=DTYPE),
                    torch.zeros(1, dtype=DTYPE),
                    torch.tensor([d]), torch.tensor([dv]))
        expected_uu = (1.0 - nu) / math.pi * dv / d**2
        assert out.e.abs().item() < 1e-9
        assert out.n.abs().item() < 1e-9
        torch.testing.assert_close(out.u.item(), expected_uu, rtol=1e-9, atol=0.0)

    def test_radial_symmetry(self, model):
        # points (dx, dy) and (-dx, -dy): horizontals flip sign, vertical equal
        x = torch.tensor([[1000.0, -1000.0]])
        y = torch.tensor([[500.0, -500.0]])
        z = torch.zeros(1, dtype=DTYPE)
        out = model(x, y, z, z.clone(), torch.tensor([1500.0]), torch.tensor([1e6]))
        torch.testing.assert_close(out.e[0, 0], -out.e[0, 1])
        torch.testing.assert_close(out.n[0, 0], -out.n[0, 1])
        torch.testing.assert_close(out.u[0, 0], out.u[0, 1])

    def test_inflation_and_deflation_signs(self, model):
        # East observation point, source below origin.
        x = torch.tensor([[1000.0]]); y = torch.zeros(1, 1, dtype=DTYPE)
        z = torch.zeros(1, dtype=DTYPE); d = torch.tensor([2000.0])
        infl = model(x, y, z, z.clone(), d, torch.tensor([1e6]))
        defl = model(x, y, z, z.clone(), d, torch.tensor([-1e6]))
        # inflation: uplift (uu>0) and outward (east point moves east, ue>0)
        assert infl.u.item() > 0 and infl.e.item() > 0
        # deflation: subsidence and inward
        assert defl.u.item() < 0 and defl.e.item() < 0

    def test_linearity_in_delta_v(self, model):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(3, 20)
        single = model(x_obs, y_obs, sx, sy, depth, dv)
        double = model(x_obs, y_obs, sx, sy, depth, 2 * dv)
        torch.testing.assert_close(double.e, 2 * single.e)
        torch.testing.assert_close(double.u, 2 * single.u)
        zero = model(x_obs, y_obs, sx, sy, depth, torch.zeros_like(dv))
        assert zero.e.abs().max() == 0 and zero.u.abs().max() == 0

    def test_decay_with_distance(self, model):
        x = torch.tensor([[100.0, 1000.0, 5000.0, 20000.0]])
        y = torch.zeros_like(x)
        z = torch.zeros(1, dtype=DTYPE)
        out = model(x, y, z, z.clone(), torch.tensor([2000.0]), torch.tensor([1e6]))
        mag = torch.sqrt(out.e**2 + out.n**2 + out.u**2)[0]
        assert torch.all(mag[1:] < mag[:-1])

    @pytest.mark.parametrize("nu", [0.0, 0.25, 0.3, 0.49])
    def test_poisson_ratio_scaling(self, nu):
        # displacement is proportional to (1 - nu)
        args = make_inputs(2, 10)
        base = MogiSource(poisson_ratio=0.25)(*args)
        other = MogiSource(poisson_ratio=nu)(*args)
        ratio = (1.0 - nu) / (1.0 - 0.25)
        torch.testing.assert_close(other.e, ratio * base.e)
        torch.testing.assert_close(other.u, ratio * base.u)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
class TestBatching:
    @pytest.mark.parametrize("B", [1, 2, 8])
    def test_batched_equals_single_loop(self, model, B):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(B, 30)
        batched = model(x_obs, y_obs, sx, sy, depth, dv)
        for b in range(B):
            one = model(x_obs[b:b + 1], y_obs[b:b + 1], sx[b:b + 1],
                        sy[b:b + 1], depth[b:b + 1], dv[b:b + 1])
            torch.testing.assert_close(one.e, batched.e[b:b + 1])
            torch.testing.assert_close(one.n, batched.n[b:b + 1])
            torch.testing.assert_close(one.u, batched.u[b:b + 1])

    def test_batch_independence(self, model):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(4, 25)
        base = model(x_obs, y_obs, sx, sy, depth, dv)
        dv2 = dv.clone()
        dv2[1] *= 3.0                       # perturb only row 1
        out2 = model(x_obs, y_obs, sx, sy, depth, dv2)
        for b in range(4):
            changed = not torch.allclose(out2.e[b], base.e[b])
            assert changed == (b == 1)

    def test_per_batch_depth(self, model):
        # different scalar depth per batch element
        x_obs, y_obs, sx, sy, _, dv = make_inputs(3, 12)
        depth = torch.tensor([1000.0, 2500.0, 4000.0])
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        ref = reference_mogi_loop(x_obs, y_obs, sx, sy, depth, dv)
        torch.testing.assert_close(out.u, ref[2], rtol=1e-9, atol=1e-12)

    def test_depth_per_point_broadcasts(self, model):
        # depth given as [B, N]
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(3, 20, depth_per_point=True)
        assert depth.shape == (3, 20)
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        ref = reference_mogi_loop(x_obs, y_obs, sx, sy, depth, dv)
        torch.testing.assert_close(out.u, ref[2], rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(out.e, ref[0], rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("N", [1, 100])
    def test_various_n(self, model, N):
        out = model(*make_inputs(2, N))
        assert out.e.shape == (2, N)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiability:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_source_params(self, model, device):
        # O(1) magnitudes so the finite-difference jacobian is well conditioned.
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(
            2, 6, device=device, scale="unit")
        for t in (sx, sy, depth, dv):
            t.requires_grad_(True)

        def f(sx, sy, depth, dv):
            o = model(x_obs, y_obs, sx, sy, depth, dv)
            return o.e, o.n, o.u

        assert torch.autograd.gradcheck(f, (sx, sy, depth, dv))

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_observation_coords(self, model, device):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(
            2, 6, device=device, scale="unit")
        for t in (x_obs, y_obs):
            t.requires_grad_(True)

        def f(x_obs, y_obs):
            o = model(x_obs, y_obs, sx, sy, depth, dv)
            return o.e, o.n, o.u

        assert torch.autograd.gradcheck(f, (x_obs, y_obs))

    def test_gradcheck_per_point_depth(self, model):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(
            2, 5, depth_per_point=True, scale="unit")
        depth.requires_grad_(True)

        def f(depth):
            o = model(x_obs, y_obs, sx, sy, depth, dv)
            return o.u

        assert torch.autograd.gradcheck(f, (depth,))

    def test_gradgradcheck_source_params(self, model):
        # second-order check: useful if you do Hessian / Gauss-Newton inversion.
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(2, 4, scale="unit")
        for t in (sx, sy, depth, dv):
            t.requires_grad_(True)

        def f(sx, sy, depth, dv):
            o = model(x_obs, y_obs, sx, sy, depth, dv)
            return o.e, o.n, o.u

        assert torch.autograd.gradgradcheck(f, (sx, sy, depth, dv))

    def test_grad_flows_to_all_inputs(self, model):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(
            2, 8, requires_grad=True)
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        (out.e.sum() + out.n.sum() + out.u.sum()).backward()
        for name, t in [("x_obs", x_obs), ("y_obs", y_obs), ("source_x", sx),
                        ("source_y", sy), ("depth", depth), ("delta_v", dv)]:
            assert t.grad is not None, f"no grad for {name}"
            assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"
            assert t.grad.abs().sum() > 0, f"zero grad for {name}"

    def test_output_requires_grad_when_input_does(self, model):
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(2, 5)
        dv.requires_grad_(True)
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        assert out.e.requires_grad and out.n.requires_grad and out.u.requires_grad

    def test_gradient_finite_at_regularised_singularity(self, model):
        # observation point exactly on the source, zero depth: the num_eps term
        # must keep both forward and backward finite (no NaN / Inf).
        x = torch.zeros(1, 1, dtype=DTYPE, requires_grad=True)
        y = torch.zeros(1, 1, dtype=DTYPE, requires_grad=True)
        sx = torch.zeros(1, dtype=DTYPE, requires_grad=True)
        sy = torch.zeros(1, dtype=DTYPE, requires_grad=True)
        depth = torch.zeros(1, dtype=DTYPE, requires_grad=True)
        dv = torch.tensor([1.0], dtype=DTYPE, requires_grad=True)
        out = model(x, y, sx, sy, depth, dv)
        # forward is finite (and zero, since every numerator is zero)
        for comp in (out.e, out.n, out.u):
            assert torch.isfinite(comp).all()
        (out.e.sum() + out.n.sum() + out.u.sum()).backward()
        for t in (x, y, sx, sy, depth, dv):
            assert torch.isfinite(t.grad).all()


# --------------------------------------------------------------------------- #
# dtype / device handling
# --------------------------------------------------------------------------- #
class TestDtypeAndDevice:
    def test_float32_input_promoted_to_internal_dtype(self, model):
        # default internal_dtype is float64: float32 inputs come back as float64
        x_obs, y_obs, sx, sy, depth, dv = make_inputs(2, 10, dtype=torch.float32)
        out = model(x_obs, y_obs, sx, sy, depth, dv)
        assert out.e.dtype == torch.float64
        ref = reference_mogi_loop(x_obs, y_obs, sx, sy, depth, dv)
        torch.testing.assert_close(out.u, ref[2], rtol=1e-9, atol=1e-12)

    def test_internal_dtype_float32(self):
        m = MogiSource(internal_dtype=torch.float32)
        out = m(*make_inputs(2, 10, dtype=torch.float32))
        assert out.e.dtype == torch.float32
        ref = reference_mogi_loop(*make_inputs(2, 10, dtype=torch.float32))
        torch.testing.assert_close(out.u.double(), ref[2], rtol=1e-4, atol=1e-6)

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_output_on_input_device(self, model):
        out = model(*make_inputs(2, 8, device="cuda"))
        assert out.e.device.type == "cuda"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
