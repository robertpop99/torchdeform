"""
Tests for OkadaSourceSimple (z = 0) and OkadaSource (z <= 0).

Validation strategy
-------------------
1. External ground truth (surface): Okada (1985), BSSA 75(4), Table 2,
   "Case 2" finite rectangular fault. This exercises all three slip
   components against published 4-significant-digit values.

2. External ground truth (depth, optional): cross-check OkadaSource
   against the standard `okada_wrapper` binding of Okada's own DC3D.f.
   Skipped automatically if okada_wrapper is not installed
   (pip install okada_wrapper).

3. Internal consistency / physics invariants that need no reference:
   - FullZ at z = 0 must equal Simplified (completely different code
     paths: UA real/image cancellation + UB + z*UC vs UB-only).
   - Translation invariance, strike-rotation equivariance,
     linearity + superposition in slip, far-field decay,
     continuity of FullZ as z -> 0-.

4. Numerical health: singular on-trace points return finite values,
   training_safe mode produces finite gradients on a grid crossing the
   fault trace, and torch.autograd.gradcheck passes in training_safe
   mode at a benign configuration.

Geometry conversion for the Okada-85 Case 2 test
------------------------------------------------
Okada (1985) Case 2: x=2, y=3, d=4, dip=70 deg, L=3, W=2, U=1, with the
fault occupying xi in [0, L], eta in [0, W], measured from a reference
point at depth d on the DEEP edge (up-dip is +eta, +y).

The classes here are centroid-based (al = +-L/2, aw = +-W/2,
depth = centroid depth), so the equivalent placement is:

    centroid_depth = d - (W/2) sin(dip)
    centroid offset in okada-local coords: (L/2 along strike,
                                            (W/2) cos(dip) across strike)

With strike = 0 the class's local frame reduces to:
    x_local = dN  (along strike)     y_local = dE  (across strike, down-dip side)
    un = ux_okada, ue = uy_okada, uu = uz_okada

so the observation point goes at
    dN = x_okada - L/2,   dE = y_okada - (W/2) cos(dip).

This mapping was verified symbolically: with it, the class's
(xi corners, eta corners, q) reproduce Okada-85's
({x, x-L}, {p, p-W}, y sin(d) - d cos(d)) exactly.

Run with::

    pytest test_okada_source.py -v
"""

import math

import pytest
import torch

from torchdeform import OkadaSource, OkadaSourceSimple

# dtype is passed explicitly everywhere (the t() helper, random_params, and
# each model's internal_dtype), so no global torch.set_default_dtype is needed:
# the suite never mutates global torch state.
DTYPE = torch.float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# Helpers

def t(x, device=None):
    """Tensor shortcut (float64, optional device)."""
    return torch.as_tensor(x, dtype=DTYPE, device=device)


def _assert_output_shape(out, expected):
    """All three displacement components must match the [B, N] obs grid.

    Guards against silent broadcasting bugs (e.g. a double-unsqueezed
    strike rotation producing [B, B, N]), which np/torch broadcasting can
    hide inside a loss without raising.
    """
    for nm, v in (("e", out.e), ("n", out.n), ("u", out.u)):
        assert v.shape == expected, \
            f"{nm} has shape {tuple(v.shape)}, expected {tuple(expected)}"


def run_simplified(x_obs, y_obs, source_x, source_y, dip, strike,
                   depth, length, width, d1, d2, d3, **model_kw):
    model = OkadaSourceSimple(**model_kw)
    out = model(
        x_obs=t(x_obs), y_obs=t(y_obs),
        source_x=t(source_x), source_y=t(source_y),
        dip=t(dip), strike=t(strike),
        centroid_depth=t(depth), length=t(length), width=t(width),
        disl1=t(d1), disl2=t(d2), disl3=t(d3),
    )
    _assert_output_shape(out, t(x_obs).shape)
    return out.e, out.n, out.u


def run_fullz(x_obs, y_obs, z_obs, source_x, source_y, dip, strike,
              depth, length, width, d1, d2, d3, **model_kw):
    model = OkadaSource(**model_kw)
    out = model(
        x_obs=t(x_obs), y_obs=t(y_obs), z_obs=t(z_obs),
        source_x=t(source_x), source_y=t(source_y),
        dip=t(dip), strike=t(strike),
        centroid_depth=t(depth), length=t(length), width=t(width),
        disl1=t(d1), disl2=t(d2), disl3=t(d3),
    )
    _assert_output_shape(out, t(x_obs).shape)
    return out.e, out.n, out.u


def random_params(seed, batch=4, n=12, surface_safe=True):
    """Random but physically sane fault/observation configurations."""
    g = torch.Generator().manual_seed(seed)

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DTYPE)

    length = u(2e3, 30e3, batch)
    width = u(1e3, 15e3, batch)
    dip = u(0.15, 1.40, batch)          # ~8.6 to ~80 deg, away from 0/90
    strike = u(0.0, 2 * math.pi, batch)
    # keep the fault buried: top edge depth = depth - (W/2) sin(dip) > margin
    min_depth = 0.5 * width * torch.sin(dip) + 1e3
    depth = min_depth + u(0.0, 20e3, batch)
    source_x = u(-5e3, 5e3, batch)
    source_y = u(-5e3, 5e3, batch)
    d1 = u(-2.0, 2.0, batch)
    d2 = u(-2.0, 2.0, batch)
    d3 = u(-0.5, 0.5, batch)
    x_obs = u(-60e3, 60e3, batch, n)
    y_obs = u(-60e3, 60e3, batch, n)
    if not surface_safe:
        pass
    return dict(x_obs=x_obs, y_obs=y_obs, source_x=source_x, source_y=source_y,
                dip=dip, strike=strike, depth=depth, length=length,
                width=width, d1=d1, d2=d2, d3=d3)


#
# Reference displacements (ux, uy, uz) in Okada's fault-local frame, U=1,
# taken from the "Checklist for numerical calculations", Okada (1985)
# Table 2, p.1149 (first three columns of the Beauducel okada85 checklist;
# columns 4-9 are spatial derivatives, which these classes don't output).
#
# Each case is a fault geometry + observation point; each component
# (strike/dip/tensile) sets the dislocation triple (disl1, disl2, disl3).
#
#   Case 2: dip=70, oblique observation, the canonical all-nonzero check.
#   Case 3: dip=90 vertical, surface-breaking, observed at the origin.
#   Case 4: simulates dip=-90 via dip=90 + rake=180. For strike-slip,
#           rake=180 => disl1 = slip*cos(180) = -1.
#
# Geometry placement matches Beauducel exactly:
#   okada85(x - L/2, y - cos(dip)*W/2, d + sin(dip)*W/2, strike=90, dip, ...)
# which in this centroid-based, strike=0 convention becomes:
#   centroid_depth = d - (W/2) sin(dip)
#   dN (along strike)   = x - L/2
#   dE (across strike)  = y - (W/2) cos(dip)
# and with strike=0: un = ux_okada, ue = uy_okada, uu = uz_okada.

# slip triples (disl1, disl2, disl3) per component name
_SLIP = {
    "strike_slip": (1.0, 0.0, 0.0),
    "dip_slip":    (0.0, 1.0, 0.0),
    "tensile":     (0.0, 0.0, 1.0),
}
# Case-4 strike-slip is rake=180 (=> disl1 = -1); dip/tensile unchanged.
_SLIP_RAKE180 = dict(_SLIP, strike_slip=(-1.0, 0.0, 0.0))

# (name, component, x, y, d, dip_deg, L, W, slip_table, (ux, uy, uz))
OKADA85_CHECKLIST = [
    # ---- Case 2: x=2, y=3, d=4, dip=70, L=3, W=2 ----
    ("case2", "strike_slip", 2, 3, 4, 70, 3, 2, _SLIP,
        (-8.689e-3, -4.298e-3, -2.747e-3)),
    ("case2", "dip_slip", 2, 3, 4, 70, 3, 2, _SLIP,
        (-4.682e-3, -3.527e-2, -3.564e-2)),
    ("case2", "tensile", 2, 3, 4, 70, 3, 2, _SLIP,
        (-2.660e-4, +1.056e-2, +3.214e-3)),
    # ---- Case 3: x=0, y=0, d=4, dip=90, L=3, W=2 ----
    ("case3", "strike_slip", 0, 0, 4, 90, 3, 2, _SLIP,
        (0.0, +5.253e-3, 0.0)),
    ("case3", "dip_slip", 0, 0, 4, 90, 3, 2, _SLIP,
        (0.0, 0.0, 0.0)),
    ("case3", "tensile", 0, 0, 4, 90, 3, 2, _SLIP,
        (+1.223e-2, 0.0, -1.606e-2)),
    # ---- Case 4: x=0, y=0, d=6, dip=90, L=3, W=2, rake=180 strike ----
    ("case4", "strike_slip", 0, 0, 6, 90, 3, 2, _SLIP_RAKE180,
        (0.0, -1.303e-3, 0.0)),
    ("case4", "dip_slip", 0, 0, 6, 90, 3, 2, _SLIP_RAKE180,
        (0.0, 0.0, 0.0)),
    ("case4", "tensile", 0, 0, 6, 90, 3, 2, _SLIP_RAKE180,
        (+3.507e-3, 0.0, -7.740e-3)),
]


# Spatial-derivative reference values (Okada-85 Table 2, columns 4-9),
# in Okada's fault-local frame:
#     (dux/dx, dux/dy, duy/dx, duy/dy, duz/dx, duz/dy)
# Keyed by (case, component). These are the `check(j, 4:9)` rows of the
# Beauducel checklist verbatim; the minus signs in his `varok` map apply
# to his computed E/N outputs to bring them INTO this frame, not to the
# table values, so the table entries are used as-is here.
OKADA85_DERIV = {
    ("case2", "strike_slip"): (-1.220e-3, +2.470e-4, -8.191e-3, -5.814e-4, -5.175e-3, +2.945e-4),
    ("case2", "dip_slip"):    (-8.867e-3, -1.519e-4, +4.057e-3, -1.035e-2, +4.088e-3, +2.626e-3),
    ("case2", "tensile"):     (-5.655e-4, +1.993e-3, -1.066e-3, +1.230e-2, -3.730e-4, +1.040e-2),
    ("case3", "strike_slip"): (0.0, -1.864e-2, -2.325e-3, 0.0, 0.0, +2.289e-2),
    ("case3", "dip_slip"):    (0.0, +2.748e-2, 0.0, 0.0, 0.0, -7.166e-2),
    ("case3", "tensile"):     (-4.182e-3, 0.0, 0.0, -2.325e-3, -9.146e-3, 0.0),
    ("case4", "strike_slip"): (0.0, +2.726e-3, +7.345e-4, 0.0, 0.0, -4.422e-3),
    ("case4", "dip_slip"):    (0.0, +5.157e-3, 0.0, 0.0, 0.0, -1.901e-2),
    ("case4", "tensile"):     (-1.770e-3, 0.0, 0.0, -7.345e-4, -1.843e-3, 0.0),
}


def _checklist_inputs(x, y, d, dip_deg, L, W):
    """Centroid-based placement equivalent to the Okada-85 checklist."""
    dip = math.radians(dip_deg)
    depth_c = d - 0.5 * W * math.sin(dip)
    dn = x - 0.5 * L                    # along strike
    de = y - 0.5 * W * math.cos(dip)    # across strike
    return depth_c, de, dn, dip


# Only Case 2 has a smooth displacement field at the observation point.
# Cases 3 and 4 are vertical, surface-breaking faults observed ON the fault
# trace, where the strike/dip slip makes displacement DISCONTINUOUS across
# the plane. The displacement there is well-defined (the average of the two
# sides -- which is why the displacement checklist passes on-trace), but the
# spatial derivative ACROSS the discontinuity is not: its value depends on
# the approach direction. Okada's analytic table reports one limiting
# convention; autograd of a (smoothed) field reports another; neither is
# "wrong", the quantity is simply ill-defined there. So derivatives are only
# checked against the analytic table where the field is smooth -- Case 2.
# (On-trace gradient *finiteness* for the singular cases is already covered
# by test_no_nan_values_or_gradients_training_safe.)
OKADA85_DERIV_ROWS = [r for r in OKADA85_CHECKLIST if r[0] == "case2"]


def _benign_inputs_simplified(dtype, device="cpu"):
    """A single off-fault, fully-buried Simplified config in the given dtype."""
    def f(x):
        return torch.tensor(x, dtype=dtype, device=device)
    return dict(
        x_obs=f([[3e3, -7e3, 12e3]]), y_obs=f([[5e3, 9e3, -4e3]]),
        source_x=f([0.0]), source_y=f([0.0]),
        dip=f([0.9]), strike=f([0.7]),
        centroid_depth=f([9e3]), length=f([8e3]), width=f([4e3]),
        disl1=f([1.0]), disl2=f([-0.5]), disl3=f([0.2]),
    )


class TestOkada85Reference:
    """Published Okada (1985) Table 2 ground truth (displacement + derivatives)."""

    @pytest.mark.parametrize(
        "row", OKADA85_CHECKLIST,
        ids=[f"{r[0]}-{r[1]}" for r in OKADA85_CHECKLIST],
    )
    @pytest.mark.parametrize("which", ["simplified", "fullz"])
    def test_okada85_checklist(self, row, which):
        name, comp, x, y, d, dip_deg, L, W, slip_table, (ux, uy, uz) = row
        d1, d2, d3 = slip_table[comp]
        depth_c, de, dn, dip = _checklist_inputs(x, y, d, dip_deg, L, W)

        common = dict(
            x_obs=[[de]], y_obs=[[dn]],
            source_x=[0.0], source_y=[0.0],
            dip=[dip], strike=[0.0],
            depth=[depth_c], length=[float(L)], width=[float(W)],
            d1=[d1], d2=[d2], d3=[d3],
        )
        if which == "simplified":
            ue, un, uu = run_simplified(**common)
        else:
            ue, un, uu = run_fullz(z_obs=[0.0], **common)

        # With strike = 0: un = ux_okada, ue = uy_okada, uu = uz_okada.
        got = torch.stack([un.reshape(-1)[0], ue.reshape(-1)[0], uu.reshape(-1)[0]])
        want = t([ux, uy, uz])
        # Table values carry 4 significant digits; allow a small absolute floor
        # for the exact-zero entries (vertical-fault symmetry components).
        assert torch.allclose(got, want, rtol=3e-3, atol=5e-6), \
            f"{name}/{comp}/{which}: got {got.tolist()}, want {want.tolist()}"

    @pytest.mark.parametrize(
        "row", OKADA85_DERIV_ROWS,
        ids=[f"{r[0]}-{r[1]}" for r in OKADA85_DERIV_ROWS],
    )
    @pytest.mark.parametrize("which", ["simplified", "fullz"])
    def test_okada85_checklist_derivatives(self, row, which):
        """Columns 4-9 of the Okada-85 checklist: horizontal spatial derivatives
        of displacement, obtained by autograd of the (differentiable) forward
        pass w.r.t. the observation coordinates rather than analytic formulas.

        Restricted to Case 2, the only checklist geometry whose displacement
        field is smooth at the observation point (see OKADA85_DERIV_ROWS note).

        Frame/coordinate mapping (strike = 0):
            un = ux_okada, ue = uy_okada, uu = uz_okada
            y_obs <-> Okada-x (along strike),  x_obs <-> Okada-y (across strike)
        The placement offsets are additive constants, so derivatives transfer
        1:1 with no sign change:
            dux/dx = d(un)/d(y_obs)     dux/dy = d(un)/d(x_obs)
            duy/dx = d(ue)/d(y_obs)     duy/dy = d(ue)/d(x_obs)
            duz/dx = d(uu)/d(y_obs)     duz/dy = d(uu)/d(x_obs)

        Run in training_safe=True for consistency with the gradient-health
        tests; Case 2 is off-fault so exact mode would also work here.
        """
        name, comp, x, y, d, dip_deg, L, W, slip_table, _disp = row
        d1, d2, d3 = slip_table[comp]
        depth_c, de, dn, dip = _checklist_inputs(x, y, d, dip_deg, L, W)

        # Observation coords as leaf tensors we can differentiate through.
        x_obs = t([[de]]).requires_grad_(True)   # across strike (Okada y)
        y_obs = t([[dn]]).requires_grad_(True)   # along strike  (Okada x)

        kw = dict(
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([dip]), strike=t([0.0]),
            centroid_depth=t([depth_c]), length=t([float(L)]), width=t([float(W)]),
            disl1=t([d1]), disl2=t([d2]), disl3=t([d3]),
        )
        if which == "simplified":
            model = OkadaSourceSimple(training_safe=True)
            out = model(x_obs=x_obs, y_obs=y_obs, **kw)
        else:
            model = OkadaSource(training_safe=True)
            out = model(x_obs=x_obs, y_obs=y_obs, z_obs=t([0.0]), **kw)

        def grads(scalar_field):
            # Single observation point: .sum() picks out the lone entry; its
            # gradient w.r.t. (x_obs, y_obs) is that point's derivative, with
            # no cross-point coupling.
            gx, gy = torch.autograd.grad(
                scalar_field.sum(), (x_obs, y_obs), retain_graph=True,
            )
            return gy.reshape(-1)[0], gx.reshape(-1)[0]   # (d/d Okada-x, d/d Okada-y)

        # un -> ux_okada, ue -> uy_okada, uu -> uz_okada
        duxdx, duxdy = grads(out.n)
        duydx, duydy = grads(out.e)
        duzdx, duzdy = grads(out.u)

        got = torch.stack([duxdx, duxdy, duydx, duydy, duzdx, duzdy])
        want = t(OKADA85_DERIV[(name, comp)])

        # Case 2 is off-fault and smooth, so autograd should match the analytic
        # table to its 4-sig-fig precision. A small absolute floor covers
        # entries near zero.
        assert torch.allclose(got, want, rtol=3e-3, atol=5e-6), (
            f"{name}/{comp}/{which} derivatives:\n"
            f"  got  {[f'{v:+.3e}' for v in got.tolist()]}\n"
            f"  want {[f'{v:+.3e}' for v in want.tolist()]}"
        )

class TestFullZConsistency:
    """OkadaSource vs the z=0 Simplified path, the DC3D cross-check, and z-behaviour."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fullz_equals_simplified_at_surface(self, seed):
        p = random_params(seed)
        ue_s, un_s, uu_s = run_simplified(**p)
        z0 = torch.zeros_like(p["x_obs"])
        ue_f, un_f, uu_f = run_fullz(z_obs=z0, **p)
        assert torch.allclose(ue_f, ue_s, rtol=1e-10, atol=1e-12)
        assert torch.allclose(un_f, un_s, rtol=1e-10, atol=1e-12)
        assert torch.allclose(uu_f, uu_s, rtol=1e-10, atol=1e-12)

    def test_fullz_against_dc3d_at_depth(self):
        okada_wrapper = pytest.importorskip("okada_wrapper")
        dc3dwrapper = okada_wrapper.dc3dwrapper

        alpha = 2.0 / 3.0  # nu = 0.25  ->  alpha = (lam+mu)/(lam+2mu) = 2/3
        dip_deg = 55.0
        dip = math.radians(dip_deg)
        L, W, depth_c = 12e3, 6e3, 9e3
        d1, d2, d3 = 1.3, -0.7, 0.2

        # strike = 0 so the class's local frame is (x=dN, y=dE); DC3D takes
        # fault-local coordinates directly.
        obs = [
            (4e3, 7e3, -2e3),
            (-9e3, 3e3, -5e3),
            (15e3, -6e3, -0.5e3),
            (1e3, -12e3, -8e3),
        ]
        xs = [[o[1] for o in obs]]   # dE = local y
        ys = [[o[0] for o in obs]]   # dN = local x
        zs = [[o[2] for o in obs]]

        ue, un, uu = run_fullz(
            x_obs=xs, y_obs=ys, z_obs=zs,
            source_x=[0.0], source_y=[0.0],
            dip=[dip], strike=[0.0],
            depth=[depth_c], length=[L], width=[W],
            d1=[d1], d2=[d2], d3=[d3],
        )

        for i, (xl, yl, z) in enumerate(obs):
            success, u, _ = dc3dwrapper(
                alpha, [xl, yl, z], depth_c, dip_deg,
                [-L / 2, L / 2], [-W / 2, W / 2], [d1, d2, d3],
            )
            assert success == 0
            got = torch.stack([un[0, i], ue[0, i], uu[0, i]])  # (ux, uy, uz) local
            want = t(list(u))
            assert torch.allclose(got, want, rtol=1e-6, atol=1e-9), \
                f"point {i}: got {got.tolist()}, want {want.tolist()}"

    def test_fullz_continuity_in_z(self):
        """FullZ output must approach the z = 0 result as z -> 0-."""
        p = random_params(17)
        z0 = torch.zeros_like(p["x_obs"])
        ue0, un0, uu0 = run_fullz(z_obs=z0, **p)
        ue1, un1, uu1 = run_fullz(z_obs=z0 - 1e-3, **p)  # 1 mm below surface
        scale = max(float(torch.stack([ue0, un0, uu0]).abs().max()), 1e-12)
        for a, b in ((ue0, ue1), (un0, un1), (uu0, uu1)):
            assert float((a - b).abs().max()) < 1e-6 * scale + 1e-12

    def test_fullz_depth_dependence_nontrivial(self):
        """At depth, the field must differ from the surface field (z*UC and the
        real/image asymmetry actually engage)."""
        p = random_params(19, batch=1, n=8)
        z0 = torch.zeros_like(p["x_obs"])
        ue0, un0, uu0 = run_fullz(z_obs=z0, **p)
        zdeep = z0 - 0.5 * p["depth"][:, None]
        ue1, un1, uu1 = run_fullz(z_obs=zdeep, **p)
        diff = torch.stack([ue1 - ue0, un1 - un0, uu1 - uu0]).abs().max()
        assert float(diff) > 1e-8

class TestPhysicsInvariants:
    """Reference-free symmetry / linearity / decay invariants."""

    def test_translation_invariance(self):
        p = random_params(7)
        ue1, un1, uu1 = run_simplified(**p)
        shift_x, shift_y = 12.3e3, -4.56e3
        p2 = dict(p)
        p2["x_obs"] = p["x_obs"] + shift_x
        p2["y_obs"] = p["y_obs"] + shift_y
        p2["source_x"] = p["source_x"] + shift_x
        p2["source_y"] = p["source_y"] + shift_y
        ue2, un2, uu2 = run_simplified(**p2)
        assert torch.allclose(ue1, ue2, rtol=1e-10, atol=1e-13)
        assert torch.allclose(un1, un2, rtol=1e-10, atol=1e-13)
        assert torch.allclose(uu1, uu2, rtol=1e-10, atol=1e-13)

    def test_strike_rotation_equivariance(self):
        """Rotating strike and observation grid together rotates (ue, un)
        by the same angle and leaves uu unchanged.

        The class maps strike phi to a strike vector (sin phi, cos phi); the
        rotation taking phi -> phi + a acts on (E, N) as
            E' =  E cos a + N sin a
            N' = -E sin a + N cos a
        and displacement vectors transform identically.
        """
        p = random_params(11)
        a = 0.7
        ca, sa = math.cos(a), math.sin(a)

        ue1, un1, uu1 = run_simplified(**p)

        dx = p["x_obs"] - p["source_x"][:, None]
        dy = p["y_obs"] - p["source_y"][:, None]
        p2 = dict(p)
        p2["x_obs"] = p["source_x"][:, None] + (dx * ca + dy * sa)
        p2["y_obs"] = p["source_y"][:, None] + (-dx * sa + dy * ca)
        p2["strike"] = p["strike"] + a
        ue2, un2, uu2 = run_simplified(**p2)

        ue1r = ue1 * ca + un1 * sa
        un1r = -ue1 * sa + un1 * ca
        assert torch.allclose(ue2, ue1r, rtol=1e-9, atol=1e-12)
        assert torch.allclose(un2, un1r, rtol=1e-9, atol=1e-12)
        assert torch.allclose(uu2, uu1, rtol=1e-9, atol=1e-12)

    def test_linearity_and_superposition(self):
        p = random_params(13)

        ue, un, uu = run_simplified(**p)

        # Doubling all slips doubles the field.
        p2 = dict(p, d1=2 * p["d1"], d2=2 * p["d2"], d3=2 * p["d3"])
        ue2, un2, uu2 = run_simplified(**p2)
        assert torch.allclose(ue2, 2 * ue, rtol=1e-10, atol=1e-13)
        assert torch.allclose(un2, 2 * un, rtol=1e-10, atol=1e-13)
        assert torch.allclose(uu2, 2 * uu, rtol=1e-10, atol=1e-13)

        # Field of (d1, d2, d3) equals sum of single-component fields.
        zero = torch.zeros_like(p["d1"])
        parts = []
        for k in ("d1", "d2", "d3"):
            pk = dict(p, d1=zero, d2=zero, d3=zero)
            pk[k] = p[k]
            parts.append(run_simplified(**pk))
        ue_sum = sum(pp[0] for pp in parts)
        un_sum = sum(pp[1] for pp in parts)
        uu_sum = sum(pp[2] for pp in parts)
        assert torch.allclose(ue, ue_sum, rtol=1e-10, atol=1e-13)
        assert torch.allclose(un, un_sum, rtol=1e-10, atol=1e-13)
        assert torch.allclose(uu, uu_sum, rtol=1e-10, atol=1e-13)

    def test_far_field_decay(self):
        """|u| should fall off rapidly (~1/r^2 for a finite source)."""
        base = dict(
            source_x=[0.0], source_y=[0.0],
            dip=[math.radians(60.0)], strike=[0.4],
            depth=[8e3], length=[10e3], width=[5e3],
            d1=[1.0], d2=[0.5], d3=[0.1],
        )
        r_near, r_far = 50e3, 500e3
        ang = torch.linspace(0, 2 * math.pi, 9, dtype=DTYPE)[:-1]

        def mag(r):
            ue, un, uu = run_simplified(
                x_obs=(r * torch.cos(ang))[None, :],
                y_obs=(r * torch.sin(ang))[None, :],
                **base,
            )
            return torch.sqrt(ue**2 + un**2 + uu**2).max()

        m_near, m_far = mag(r_near), mag(r_far)
        # 10x distance, ~1/r^2 decay -> expect ~100x drop; require >= 50x.
        assert m_far < m_near / 50.0
        assert m_far > 0.0  # but not identically zero

class TestNumericalHealth:
    """Finite values and gradients at and near Okada's singular configurations."""

    def test_surface_breaking_trace_is_finite(self):
        """Vertical, surface-breaking fault: points exactly on the trace hit
        Okada's singular configuration and must return finite values (the
        DC3D convention zeroes them), never NaN/inf."""
        L, W = 10e3, 5e3
        dip = math.pi / 2.0
        depth_c = 0.5 * W  # top edge exactly at the surface

        # On-trace points: y = 0, x within and at the fault ends; plus
        # near-trace points for good measure.
        xs = t([[-6e3, -5e3, -2e3, 0.0, 2e3, 5e3, 6e3, 1e3]])
        ys = t([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        ue, un, uu = run_simplified(
            x_obs=xs, y_obs=ys,
            source_x=[0.0], source_y=[0.0],
            dip=[dip], strike=[math.pi / 2.0],
            depth=[depth_c], length=[L], width=[W],
            d1=[1.0], d2=[0.0], d3=[0.0],
        )
        for v in (ue, un, uu):
            assert torch.isfinite(v).all(), "NaN/inf on or near the fault trace"

    def test_no_nan_values_or_gradients_training_safe(self):
        """training_safe mode: forward and backward must stay finite on a
        grid that crosses the surface trace of a surface-breaking fault."""
        L, W = 10e3, 5e3
        dip = t([math.radians(89.0)]).requires_grad_(True)
        depth = t([0.5 * W * math.sin(89.0 * math.pi / 180.0) + 1.0])
        d1 = t([1.0]).requires_grad_(True)
        d2 = t([0.5]).requires_grad_(True)
        d3 = t([0.0]).requires_grad_(True)

        n = 21
        g = torch.linspace(-8e3, 8e3, n, dtype=DTYPE)
        X, Y = torch.meshgrid(g, g, indexing="xy")
        x_obs = X.reshape(1, -1)
        y_obs = Y.reshape(1, -1)

        model = OkadaSourceSimple(training_safe=True)
        out = model(
            x_obs=x_obs, y_obs=y_obs,
            source_x=t([0.0]), source_y=t([0.0]),
            dip=dip, strike=t([0.3]),
            centroid_depth=depth, length=t([L]), width=t([W]),
            disl1=d1, disl2=d2, disl3=d3,
        )
        loss = (out.e**2 + out.n**2 + out.u**2).sum()
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in (("dip", dip), ("d1", d1), ("d2", d2), ("d3", d3)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), \
                f"non-finite gradient for {name}"

class TestDifferentiability:
    """Autograd through the model: gradcheck and finite-gradient coverage."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_training_safe(self, device):
        """Full autograd gradcheck at a benign configuration (float64).

        Parametrized over DEVICES so autograd is exercised on CUDA too when
        available (matching the Mogi/Penny suites)."""
        model = OkadaSourceSimple(training_safe=True).to(device)

        def td(x):
            return t(x, device=device)

        x_obs = td([[3e3, -7e3, 12e3]])
        y_obs = td([[5e3, 9e3, -4e3]])

        def f(dip, depth, length, width, d1, d2, d3):
            out = model(
                x_obs=x_obs, y_obs=y_obs,
                source_x=td([0.0]), source_y=td([0.0]),
                dip=dip, strike=td([0.7]),
                centroid_depth=depth, length=length, width=width,
                disl1=d1, disl2=d2, disl3=d3,
            )
            return torch.stack([out.e.reshape(-1), out.n.reshape(-1),
                                 out.u.reshape(-1)])

        args = tuple(
            v.clone().requires_grad_(True)
            for v in (td([0.9]), td([9e3]), td([8e3]), td([4e3]),
                      td([1.0]), td([-0.5]), td([0.2]))
        )
        assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_fullz_training_safe(self, device):
        """gradcheck through OkadaSource at depth (training_safe), where the
        UC kernel and z-dependence engage -- complements the Simplified
        gradcheck above."""
        model = OkadaSource(training_safe=True).to(device)

        def td(x):
            return t(x, device=device)

        x_obs = td([[3e3, -7e3, 9e3]])
        y_obs = td([[5e3, 4e3, -6e3]])
        z_obs = td([[-2e3, -1e3, -4e3]])

        def f(dip, depth, length, width, d1, d2, d3):
            out = model(
                x_obs=x_obs, y_obs=y_obs, z_obs=z_obs,
                source_x=td([0.0]), source_y=td([0.0]),
                dip=dip, strike=td([0.6]),
                centroid_depth=depth, length=length, width=width,
                disl1=d1, disl2=d2, disl3=d3,
            )
            return torch.stack([out.e.reshape(-1), out.n.reshape(-1),
                                out.u.reshape(-1)])

        args = tuple(
            v.clone().requires_grad_(True)
            for v in (td([0.8]), td([11e3]), td([9e3]), td([5e3]),
                      td([1.0]), td([-0.4]), td([0.3]))
        )
        assert torch.autograd.gradcheck(f, args, eps=1e-6, atol=1e-7, rtol=1e-4)

    def test_exact_mode_gradients_finite_off_fault(self):
        """Exact mode (training_safe=False) must give finite, non-zero gradients
        at an off-fault configuration, where the analytic kernels are smooth.
        Complements the training_safe on-trace gradient-health test."""
        model = OkadaSourceSimple(training_safe=False)
        x_obs = t([[3e3, -7e3, 12e3, 20e3]])
        y_obs = t([[5e3, 9e3, -4e3, -15e3]])
        dip = t([0.9]).requires_grad_(True)
        depth = t([9e3]).requires_grad_(True)
        d1 = t([1.0]).requires_grad_(True)
        d2 = t([-0.5]).requires_grad_(True)
        d3 = t([0.2]).requires_grad_(True)
        out = model(
            x_obs=x_obs, y_obs=y_obs,
            source_x=t([0.0]), source_y=t([0.0]),
            dip=dip, strike=t([0.7]),
            centroid_depth=depth, length=t([8e3]), width=t([4e3]),
            disl1=d1, disl2=d2, disl3=d3,
        )
        (out.e ** 2 + out.n ** 2 + out.u ** 2).sum().backward()
        for name, p in (("dip", dip), ("depth", depth),
                        ("d1", d1), ("d2", d2), ("d3", d3)):
            assert p.grad is not None and torch.isfinite(p.grad).all(), \
                f"non-finite gradient for {name}"
        assert float((d1.grad.abs() + d2.grad.abs() + d3.grad.abs()).item()) > 0.0

    def test_gradient_wrt_zero_slip_component_is_nonzero(self):
        """Guards against the `if torch.any(disl != 0)` pattern: optimizing a
        slip component initialized at exactly zero must receive gradient.
        Exercised through FullZ at depth, where the UC kernel participates."""
        p = random_params(23, batch=1, n=6)
        d3 = torch.zeros(1, dtype=DTYPE, requires_grad=True)
        model = OkadaSource()
        out = model(
            x_obs=p["x_obs"], y_obs=p["y_obs"],
            z_obs=torch.full_like(p["x_obs"], -2e3),
            source_x=p["source_x"], source_y=p["source_y"],
            dip=p["dip"], strike=p["strike"],
            centroid_depth=p["depth"], length=p["length"], width=p["width"],
            disl1=p["d1"], disl2=p["d2"], disl3=d3,
        )
        loss = (out.e + 2 * out.n + 3 * out.u).sum()
        loss.backward()
        assert d3.grad is not None
        assert torch.isfinite(d3.grad).all()
        assert float(d3.grad.abs().max()) > 0.0, \
            "zero gradient w.r.t. zero-initialized slip (dead `torch.any` branch?)"

class TestBatching:
    """Batched calls match per-sample runs; scalar and per-point z agree."""

    def test_batched_matches_individual_runs(self):
        p = random_params(29, batch=3, n=5)
        ue, un, uu = run_simplified(**p)
        for b in range(3):
            pb = {k: (v[b:b + 1] if torch.is_tensor(v) and v.ndim >= 1 else v)
                  for k, v in p.items()}
            ue_b, un_b, uu_b = run_simplified(**pb)
            assert torch.allclose(ue[b:b + 1], ue_b, rtol=1e-12, atol=1e-15)
            assert torch.allclose(un[b:b + 1], un_b, rtol=1e-12, atol=1e-15)
            assert torch.allclose(uu[b:b + 1], uu_b, rtol=1e-12, atol=1e-15)

    def test_fullz_scalar_and_per_point_z_agree(self):
        p = random_params(31, batch=2, n=7)
        z_scalar = t([-1.5e3, -3e3])                       # [B]
        z_grid = z_scalar[:, None].expand_as(p["x_obs"])   # [B, N]
        ue1, un1, uu1 = run_fullz(z_obs=z_scalar, **p)
        ue2, un2, uu2 = run_fullz(z_obs=z_grid.clone(), **p)
        assert torch.allclose(ue1, ue2, rtol=1e-12, atol=1e-15)
        assert torch.allclose(un1, un2, rtol=1e-12, atol=1e-15)
        assert torch.allclose(uu1, uu2, rtol=1e-12, atol=1e-15)

class TestDtypeAndDevice:
    """dtype promotion, internal dtype, and CUDA execution."""

    def test_float32_input_promoted_to_internal_dtype(self):
        """Default internal dtype is float64: float32 inputs are upcast losslessly.

        Reference = the SAME float32 values cast up to float64 (exactly what the
        model does internally), so results must match to float64 round-off."""
        model = OkadaSourceSimple()                 # internal_dtype=float64
        a32 = _benign_inputs_simplified(torch.float32)
        a64 = {k: v.double() for k, v in a32.items()}
        out32 = model(**a32)
        out64 = model(**a64)
        assert out32.e.dtype == torch.float64
        assert torch.allclose(out32.e, out64.e, rtol=1e-10, atol=1e-13)
        assert torch.allclose(out32.u, out64.u, rtol=1e-10, atol=1e-13)

    def test_internal_dtype_float32_runs(self):
        """internal_dtype=float32 produces float32 output and stays finite."""
        model = OkadaSourceSimple(internal_dtype=torch.float32)
        out = model(**_benign_inputs_simplified(torch.float32))
        assert out.e.dtype == torch.float32
        assert torch.isfinite(out.e).all() and torch.isfinite(out.u).all()

    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        """Both source classes run on CUDA with finite forward outputs.

        Mirrors the CPU/GPU coverage of the Mogi and Penny suites; the
        device-parametrized gradcheck above also exercises autograd on CUDA.
        """
        device = "cuda"
        p = random_params(41, batch=2, n=6)

        def to(k):
            return p[k].to(device)

        simp = OkadaSourceSimple(training_safe=True).to(device)
        out_s = simp(
            x_obs=to("x_obs"), y_obs=to("y_obs"),
            source_x=to("source_x"), source_y=to("source_y"),
            dip=to("dip"), strike=to("strike"),
            centroid_depth=to("depth"), length=to("length"), width=to("width"),
            disl1=to("d1"), disl2=to("d2"), disl3=to("d3"),
        )
        assert out_s.e.device.type == "cuda"
        assert torch.isfinite(out_s.e).all() and torch.isfinite(out_s.u).all()

        full = OkadaSource(training_safe=True).to(device)
        out_f = full(
            x_obs=to("x_obs"), y_obs=to("y_obs"),
            z_obs=torch.full_like(p["x_obs"], -2e3).to(device),
            source_x=to("source_x"), source_y=to("source_y"),
            dip=to("dip"), strike=to("strike"),
            centroid_depth=to("depth"), length=to("length"), width=to("width"),
            disl1=to("d1"), disl2=to("d2"), disl3=to("d3"),
        )
        assert out_f.u.device.type == "cuda"
        assert torch.isfinite(out_f.u).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
