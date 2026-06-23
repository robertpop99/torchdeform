"""
Tests for the line-of-sight (LOS) projection module (Sentinel-1 InSAR).

What is checked
---------------
* **Convention** - the unit LOS vector (ground -> satellite, ENU) matches the
  documented Sentinel-1 ascending/descending reference vectors, points along
  ``heading + look_side*90`` in the horizontal, and has ``u = cos(incidence)``.
* **Unit norm** - ``e^2 + n^2 + u^2 == 1`` for every entry point and geometry.
* **Entry-point consistency** - per-image ([B,1]) broadcasts to per-pixel
  ([B,N]); ``los_vector_from_center`` reduces to the per-image value at the
  scene centroid (and exactly so for a single pixel).
* **from_center geometry** - reconstructed incidence varies monotonically with
  range offset, spans ~2 deg across a 30 km scene, and keeps the heading fixed.
* **Sampler** - ``sample_s1_geometry`` respects the IW incidence/heading bands,
  the ascending/descending split, dtype/device, and is generator-deterministic.
* **Differentiability** - gradients flow (and gradcheck passes) through the
  angle inputs and the observation coordinates, since LOS sits inside the
  differentiable forward pipeline.

Run with::

    pytest test_los.py -v
"""
import math

import pytest
import torch

from torchdeform import (los_vector, los_vector_per_pixel, los_vector_from_center,
                         LOSVector)
from torchdeform.observation import (
    sample_s1_geometry,
    S1_INCIDENCE_RANGE_DEG, S1_HEADING_ASCENDING_DEG,
    S1_HEADING_DESCENDING_DEG, S1_LOOK_SIDE)


DTYPE = torch.float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_D2R = math.pi / 180.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _t(x, device="cpu", dtype=DTYPE):
    return torch.as_tensor(x, dtype=dtype, device=device)


def _ref_los(heading_deg, incidence_deg, look_side=S1_LOOK_SIDE, device="cpu"):
    """Independent re-statement of the LOS formula for cross-checking."""
    h = _t(heading_deg, device) * _D2R
    i = _t(incidence_deg, device) * _D2R
    az = h + look_side * (math.pi / 2.0)
    sin_i = torch.sin(i)
    return sin_i * torch.sin(az), sin_i * torch.cos(az), torch.cos(i)


def _norm(v):
    return torch.sqrt(v.e ** 2 + v.n ** 2 + v.u ** 2)


# --------------------------------------------------------------------------- #
# Convention
# --------------------------------------------------------------------------- #
class TestConvention:
    @pytest.mark.parametrize("name,heading,want", [
        ("ascending", -12.0, (0.62, 0.13, 0.78)),
        ("descending", -168.0, (-0.62, 0.13, 0.78)),
    ])
    def test_documented_sentinel1_vectors(self, name, heading, want):
        # Reference vectors quoted in the module docstring (inc=39, right-look)
        v = los_vector([heading], [39.0])
        got = (v.e.item(), v.n.item(), v.u.item())
        assert all(abs(a - b) < 0.02 for a, b in zip(got, want)), (name, got, want)

    def test_matches_independent_reference(self):
        torch.manual_seed(0)
        h = (torch.rand(64, dtype=DTYPE) * 360 - 180)
        i = (torch.rand(64, dtype=DTYPE) * 17 + 29)
        v = los_vector_per_pixel(h[None, :], i[None, :])
        re, rn, ru = _ref_los(h[None, :], i[None, :])
        torch.testing.assert_close(v.e, re)
        torch.testing.assert_close(v.n, rn)
        torch.testing.assert_close(v.u, ru)

    def test_unit_norm(self):
        h, i = sample_s1_geometry(500, generator=torch.Generator().manual_seed(1))
        assert torch.allclose(_norm(los_vector(h, i)),
                              torch.ones(500, 1, dtype=DTYPE), atol=1e-12)

    def test_vertical_is_cos_incidence(self):
        i = _t([[29.0, 35.0, 39.0, 46.0]])
        h = _t([[-13.0, -13.0, -13.0, -13.0]])
        v = los_vector_per_pixel(h, i)
        torch.testing.assert_close(v.u, torch.cos(i * _D2R))
        # incidence is recoverable from u
        torch.testing.assert_close(torch.acos(v.u) / _D2R, i, atol=1e-9, rtol=0)

    def test_horizontal_points_along_heading_plus_look(self):
        # atan2(e, n) == heading + look_side*90 (compare as unit vectors)
        h = _t([-13.0, -167.0, 30.0]); i = _t([39.0, 33.0, 40.0])
        v = los_vector(h, i)
        az = torch.atan2(v.e[:, 0], v.n[:, 0])
        want = (h + S1_LOOK_SIDE * 90.0) * _D2R
        torch.testing.assert_close(torch.sin(az), torch.sin(want), atol=1e-9, rtol=0)
        torch.testing.assert_close(torch.cos(az), torch.cos(want), atol=1e-9, rtol=0)

    def test_ascending_east_descending_west(self):
        assert los_vector([-13.0], [39.0]).e.item() > 0   # ascending looks East
        assert los_vector([-167.0], [39.0]).e.item() < 0  # descending looks West

    def test_look_side_flips_horizontal_keeps_vertical(self):
        h, i = sample_s1_geometry(50, generator=torch.Generator().manual_seed(2))
        right = los_vector(h, i, look_side=+1)
        left = los_vector(h, i, look_side=-1)
        torch.testing.assert_close(left.e, -right.e)
        torch.testing.assert_close(left.n, -right.n)
        torch.testing.assert_close(left.u, right.u)

    def test_nadir_look_at_zero_incidence(self):
        v = los_vector([-13.0], [0.0])
        assert v.e.abs().item() < 1e-12 and v.n.abs().item() < 1e-12
        torch.testing.assert_close(v.u, torch.ones_like(v.u))


# --------------------------------------------------------------------------- #
# Entry-point shapes & consistency
# --------------------------------------------------------------------------- #
class TestEntryPoints:
    def test_returns_losvector(self):
        assert isinstance(los_vector([-13.0], [39.0]), LOSVector)
        assert isinstance(los_vector_per_pixel(_t([[-13.0]]), _t([[39.0]])), LOSVector)
        assert isinstance(
            los_vector_from_center([-13.0], [39.0], _t([[0.0]]), _t([[0.0]])),
            LOSVector)

    def test_per_image_shape_is_B1(self):
        v = los_vector(_t([-13.0, -167.0]), _t([39.0, 33.0]))
        assert v.e.shape == (2, 1) and v.n.shape == (2, 1) and v.u.shape == (2, 1)

    def test_per_pixel_shape_is_BN(self):
        v = los_vector_per_pixel(_t([[-13.0] * 5, [-167.0] * 5]),
                                 _t([[39.0] * 5, [33.0] * 5]))
        assert v.e.shape == (2, 5)

    def test_per_image_broadcasts_to_per_pixel(self):
        h = _t([-13.0, -167.0]); i = _t([39.0, 33.0])
        vi = los_vector(h, i)                       # [B,1]
        vp = los_vector_per_pixel(h[:, None].expand(2, 5),
                                  i[:, None].expand(2, 5))
        torch.testing.assert_close(vp.e, vi.e.expand(2, 5))
        torch.testing.assert_close(vp.u, vi.u.expand(2, 5))

    def test_from_center_single_pixel_equals_per_image(self):
        # N=1: range offset is identically zero, so incidence == center
        h = _t([-13.0]); ic = _t([39.0])
        vc = los_vector_from_center(h, ic, _t([[1234.0]]), _t([[-567.0]]))
        vi = los_vector(h, ic)
        torch.testing.assert_close(vc.e, vi.e)
        torch.testing.assert_close(vc.n, vi.n)
        torch.testing.assert_close(vc.u, vi.u)

    def test_from_center_matches_center_incidence_at_centroid(self):
        g = torch.linspace(-15e3, 15e3, 9, dtype=DTYPE)
        X, Y = torch.meshgrid(g, g, indexing="xy")
        x = X.reshape(1, -1); y = Y.reshape(1, -1)
        vc = los_vector_from_center(_t([-13.0]), _t([39.0]), x, y)
        inc_impl = torch.acos(vc.u) / _D2R
        ctr = int(torch.argmin(x[0] ** 2 + y[0] ** 2))   # the (0,0) pixel
        assert abs(inc_impl[0, ctr].item() - 39.0) < 1e-9


# --------------------------------------------------------------------------- #
# from_center geometry
# --------------------------------------------------------------------------- #
class TestFromCenterGeometry:
    def _grid(self):
        g = torch.linspace(-15e3, 15e3, 9, dtype=DTYPE)
        X, Y = torch.meshgrid(g, g, indexing="xy")
        return X.reshape(1, -1), Y.reshape(1, -1)

    def test_incidence_spread_is_a_few_degrees(self):
        x, y = self._grid()
        vc = los_vector_from_center(_t([-13.0]), _t([39.0]), x, y)
        inc = torch.acos(vc.u) / _D2R
        spread = (inc.max() - inc.min()).item()
        assert 1.0 < spread < 4.0          # ~2 deg over a 30 km scene

    def test_incidence_monotone_in_range_offset(self):
        x, y = self._grid()
        h = -13.0
        vc = los_vector_from_center(_t([h]), _t([39.0]), x, y)
        inc = torch.acos(vc.u)[0] / _D2R
        look_az = h * _D2R + S1_LOOK_SIDE * math.pi / 2
        roff = x[0] * math.sin(look_az) + y[0] * math.cos(look_az)
        order = torch.argsort(roff)
        assert torch.all(torch.diff(inc[order]) >= -1e-9)

    def test_heading_constant_across_scene(self):
        # only incidence varies across the scene -> horizontal azimuth is fixed
        x, y = self._grid()
        vc = los_vector_from_center(_t([-13.0]), _t([39.0]), x, y)
        az = torch.atan2(vc.e[0], vc.n[0])
        assert (az.max() - az.min()).abs().item() < 1e-9

    def test_from_center_unit_norm(self):
        x, y = self._grid()
        vc = los_vector_from_center(_t([-13.0]), _t([39.0]), x, y)
        assert torch.allclose(_norm(vc), torch.ones_like(vc.u), atol=1e-12)

    def test_centroid_offset_removed(self):
        # shifting the whole grid by a constant must not change the LOS field,
        # because the centroid is subtracted off internally
        x, y = self._grid()
        v0 = los_vector_from_center(_t([-13.0]), _t([39.0]), x, y)
        v1 = los_vector_from_center(_t([-13.0]), _t([39.0]), x + 50e3, y - 20e3)
        torch.testing.assert_close(v0.u, v1.u)
        torch.testing.assert_close(v0.e, v1.e)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class TestSampler:
    def test_shapes_and_default_dtype(self):
        h, i = sample_s1_geometry(16, generator=torch.Generator().manual_seed(0))
        assert h.shape == (16,) and i.shape == (16,)
        assert h.dtype == DTYPE and i.dtype == DTYPE

    def test_incidence_within_iw_swath(self):
        h, i = sample_s1_geometry(4000, generator=torch.Generator().manual_seed(0))
        lo, hi = S1_INCIDENCE_RANGE_DEG
        assert torch.all(i >= lo) and torch.all(i <= hi)

    def test_p_ascending_edges(self):
        g = torch.Generator().manual_seed(0)
        alo, ahi = S1_HEADING_ASCENDING_DEG
        dlo, dhi = S1_HEADING_DESCENDING_DEG
        h_a, _ = sample_s1_geometry(1000, generator=g, p_ascending=1.0)
        assert torch.all(h_a >= alo) and torch.all(h_a <= ahi)
        h_d, _ = sample_s1_geometry(1000, generator=g, p_ascending=0.0)
        assert torch.all(h_d >= dlo) and torch.all(h_d <= dhi)

    def test_mixed_falls_in_one_of_two_bands(self):
        h, _ = sample_s1_geometry(4000, generator=torch.Generator().manual_seed(3),
                                  p_ascending=0.5)
        alo, ahi = S1_HEADING_ASCENDING_DEG
        dlo, dhi = S1_HEADING_DESCENDING_DEG
        in_asc = (h >= alo) & (h <= ahi)
        in_desc = (h >= dlo) & (h <= dhi)
        assert torch.all(in_asc | in_desc)
        assert in_asc.any() and in_desc.any()      # both passes represented

    def test_generator_deterministic(self):
        h1, i1 = sample_s1_geometry(64, generator=torch.Generator().manual_seed(7))
        h2, i2 = sample_s1_geometry(64, generator=torch.Generator().manual_seed(7))
        assert torch.equal(h1, h2) and torch.equal(i1, i2)

    def test_sampled_geometry_is_unit_norm(self):
        h, i = sample_s1_geometry(500, generator=torch.Generator().manual_seed(9))
        assert torch.allclose(_norm(los_vector(h, i)),
                              torch.ones(500, 1, dtype=DTYPE), atol=1e-12)

    def test_dtype_respected(self):
        h, i = sample_s1_geometry(8, generator=torch.Generator().manual_seed(0),
                                  dtype=torch.float32)
        assert h.dtype == torch.float32 and i.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
class TestDifferentiability:
    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_los_vector(self, device):
        heading = torch.tensor([-13.0, -167.0], dtype=DTYPE, device=device,
                               requires_grad=True)
        inc = torch.tensor([39.0, 33.0], dtype=DTYPE, device=device,
                            requires_grad=True)

        def f(h, i):
            v = los_vector(h, i, device=device)
            return v.e, v.n, v.u

        assert torch.autograd.gradcheck(f, (heading, inc))

    def test_gradcheck_per_pixel(self):
        h = torch.tensor([[-13.0, -12.0, -14.0]], dtype=DTYPE, requires_grad=True)
        i = torch.tensor([[39.0, 40.0, 38.0]], dtype=DTYPE, requires_grad=True)

        def f(h, i):
            v = los_vector_per_pixel(h, i)
            return v.e, v.n, v.u

        assert torch.autograd.gradcheck(f, (h, i))

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_from_center_angles(self, device):
        # gradcheck over the angle inputs (well-conditioned); x/y handled below
        x = torch.tensor([[-6e3, 0.0, 8e3]], dtype=DTYPE, device=device)
        y = torch.tensor([[3e3, -2e3, 5e3]], dtype=DTYPE, device=device)
        heading = torch.tensor([-13.0], dtype=DTYPE, device=device, requires_grad=True)
        inc_c = torch.tensor([39.0], dtype=DTYPE, device=device, requires_grad=True)

        def f(h, ic):
            v = los_vector_from_center(h, ic, x, y, device=device)
            return v.e, v.n, v.u

        assert torch.autograd.gradcheck(f, (heading, inc_c))

    def test_grad_flows_to_observation_coords(self):
        # incidence in from_center depends on x_obs/y_obs through the geometry,
        # so they must receive finite, non-zero gradients.
        x = torch.tensor([[-6e3, 0.0, 8e3, 12e3]], dtype=DTYPE, requires_grad=True)
        y = torch.tensor([[3e3, -2e3, 5e3, -9e3]], dtype=DTYPE, requires_grad=True)
        heading = torch.tensor([-13.0], dtype=DTYPE, requires_grad=True)
        inc_c = torch.tensor([39.0], dtype=DTYPE, requires_grad=True)
        v = los_vector_from_center(heading, inc_c, x, y)
        (v.e.sum() + v.n.sum() + v.u.sum()).backward()
        for name, p in (("x_obs", x), ("y_obs", y),
                        ("heading", heading), ("inc_center", inc_c)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert (x.grad.abs().sum() + y.grad.abs().sum()).item() > 0


# --------------------------------------------------------------------------- #
# dtype / device
# --------------------------------------------------------------------------- #
class TestDtypeAndDevice:
    def test_default_dtype_is_float64(self):
        v = los_vector([-13.0], [39.0])
        assert v.e.dtype == torch.float64

    def test_dtype_argument_respected(self):
        v = los_vector(_t([-13.0], dtype=torch.float32), _t([39.0], dtype=torch.float32),
                       dtype=torch.float32)
        assert v.e.dtype == torch.float32
        # still a unit vector at single precision
        assert abs(float(_norm(v)) - 1.0) < 1e-5

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        h, i = sample_s1_geometry(8, generator=torch.Generator().manual_seed(0))
        v = los_vector(h.to("cuda"), i.to("cuda"), device="cuda")
        assert v.e.device.type == "cuda"
        assert torch.allclose(_norm(v), torch.ones_like(v.u), atol=1e-12)
        x = torch.linspace(-1e4, 1e4, 9, dtype=DTYPE, device="cuda")[None, :]
        vc = los_vector_from_center(h[:1].to("cuda"), i[:1].to("cuda"),
                                    x, x.clone(), device="cuda")
        assert vc.u.device.type == "cuda" and torch.isfinite(vc.u).all()


# --------------------------------------------------------------------------- #
# batchable look_side
# --------------------------------------------------------------------------- #
class TestBatchableLookSide:
    def test_per_image_matches_scalar(self):
        heading = _t([-13.0, -13.0])
        inc = _t([39.0, 39.0])
        batched = los_vector(heading, inc, look_side=_t([1.0, -1.0]))
        right = los_vector(heading[:1], inc[:1], look_side=1)
        left = los_vector(heading[1:], inc[1:], look_side=-1)
        assert torch.allclose(batched.e[0], right.e[0])
        assert torch.allclose(batched.e[1], left.e[0])

    def test_flipping_look_side_flips_horizontal_keeps_vertical(self):
        heading = _t([-13.0, -13.0])
        inc = _t([39.0, 39.0])
        v = los_vector(heading, inc, look_side=_t([1.0, -1.0]))
        assert torch.allclose(v.e[0], -v.e[1])
        assert torch.allclose(v.n[0], -v.n[1])
        assert torch.allclose(v.u[0], v.u[1])
        assert torch.allclose(_norm(v), torch.ones_like(_norm(v)))

    def test_scalar_default_still_works(self):
        v = los_vector(_t([-13.0]), _t([39.0]))     # int default look_side
        assert torch.allclose(_norm(v), torch.ones_like(_norm(v)))

    def test_per_pixel_batchable(self):
        h = torch.full((2, 4), -13.0, dtype=DTYPE)
        i = torch.full((2, 4), 39.0, dtype=DTYPE)
        v = los_vector_per_pixel(h, i, look_side=_t([1.0, -1.0]))
        assert v.e.shape == (2, 4)
        assert torch.allclose(v.e[0], -v.e[1])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
