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
   smooth_grad mode produces finite gradients on a grid crossing the
   fault trace, and torch.autograd.gradcheck passes in smooth_grad
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

import json
import math
from pathlib import Path

import pytest
import torch

from torchdeform import OkadaSource, OkadaSourceSimple
from torchdeform.sources.okada import F32_VERTICAL_BAND

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
#   Case 4: dip=-90 vertical, the paper's geometry used directly (the model
#           handles negative dip; no dip=90 + rake=180 emulation needed).
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
    # ---- Case 4: x=0, y=0, d=4, dip=-90, L=3, W=2 (paper-faithful) ----
    # Okada's Table 2 Case 4 is the same fault as Case 3 with dip = -90. The
    # model handles negative dip directly, so the paper's geometry is used as-is
    # (d=4, dip=-90), with no dip=+90 / rake=180 emulation. The placement
    # centroid_depth = d - (W/2) sin(dip) = 4 - (1)(-1) = 5.
    ("case4", "strike_slip", 0, 0, 4, -90, 3, 2, _SLIP,
        (0.0, -1.303e-3, 0.0)),
    ("case4", "dip_slip", 0, 0, 4, -90, 3, 2, _SLIP,
        (0.0, 0.0, 0.0)),
    ("case4", "tensile", 0, 0, 4, -90, 3, 2, _SLIP,
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


# ---------------------------------------------------------------------------
# Okada (1985) Table 2, Case 1 -- the POINT source (x=2, y=3, d=4, dip=70).
#
# NOT exercised above: Case 1 uses Okada's point-source formulas (the `DC3D0`
# routine), a different model from the finite rectangular dislocation (`DC3D`)
# that OkadaSource / OkadaSourceSimple implement. torchdeform has no point-source
# Okada class, so there is nothing to check these rows against -- they are
# recorded here verbatim from the paper so the full table is captured. If a
# DC3D0 forward model is ever added, promote this into OKADA85_CHECKLIST /
# OKADA85_DERIV. Same fault-local frame and column order as the tables above:
# displacement (ux, uy, uz); derivatives (dux/dx, dux/dy, duy/dx, duy/dy,
# duz/dx, duz/dy).
#
# OKADA85_CASE1_POINT = {
#     # (ux, uy, uz)
#     "displacement": {
#         "strike_slip": (-9.447e-4, -1.023e-3, -7.420e-4),
#         "dip_slip":    (-1.172e-3, -2.082e-3, -2.532e-3),
#         "tensile":     (-3.572e-4, +3.531e-4, -2.007e-4),
#     },
#     # (dux/dx, dux/dy, duy/dx, duy/dy, duz/dx, duz/dy)
#     "derivatives": {
#         "strike_slip": (-2.286e-4, -1.425e-4, -2.051e-4, -3.007e-4, -6.259e-5, -1.693e-4),
#         "dip_slip":    (-1.526e-4, -3.544e-4, +6.983e-4, -1.154e-3, +8.707e-4, -6.345e-4),
#         "tensile":     (-1.360e-4, +5.073e-4, -6.773e-5, +6.811e-4, +7.541e-5, +8.104e-4),
#     },
# }
# ---------------------------------------------------------------------------


def _checklist_inputs(x, y, d, dip_deg, L, W):
    """Centroid-based placement equivalent to the Okada-85 checklist."""
    dip = math.radians(dip_deg)
    depth_c = d - 0.5 * W * math.sin(dip)
    dn = x - 0.5 * L                    # along strike
    de = y - 0.5 * W * math.cos(dip)    # across strike
    return depth_c, de, dn, dip


# All three checklist cases have a well-defined horizontal spatial derivative.
# Cases 3 and 4 sit directly above a *buried* vertical fault (top edge at depth
# 2 and 4 respectively, since centroid_depth = d - (W/2) sin(dip)), so the
# displacement field is continuous there and the cross-trace derivative has a
# single value -- equal to Okada's Table-2 entry, confirmed by symmetric finite
# differences. The reason these rows previously could not be checked by autograd
# is the `smooth_grad` SMOOTHING: at a vertical dip (cd = cos(dip) = 0) the
# smoothed kernel's gradient drifts far from the analytic value. The analytic
# backend (OkadaSourceSimple(analytic_grad=True)) returns the closed-form Okada strain
# instead and reproduces the table for every case.
OKADA85_DERIV_ROWS = list(OKADA85_CHECKLIST)


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
    def test_okada85_checklist_derivatives(self, row):
        """Columns 4-9 of the Okada-85 checklist: horizontal spatial derivatives
        of displacement, obtained by autograd of OkadaSourceSimple(analytic_grad=True) w.r.t.
        the observation coordinates. Its backward returns the closed-form Okada
        strain, so it reproduces the table for *every* checklist case -- including
        the vertical-dip Cases 3 and 4 where autograd of the smoothed forward
        drifts (see OKADA85_DERIV_ROWS note).

        Frame/coordinate mapping (strike = 0):
            un = ux_okada, ue = uy_okada, uu = uz_okada
            y_obs <-> Okada-x (along strike),  x_obs <-> Okada-y (across strike)
        The placement offsets are additive constants, so derivatives transfer
        1:1 with no sign change:
            dux/dx = d(un)/d(y_obs)     dux/dy = d(un)/d(x_obs)
            duy/dx = d(ue)/d(y_obs)     duy/dy = d(ue)/d(x_obs)
            duz/dx = d(uu)/d(y_obs)     duz/dy = d(uu)/d(x_obs)
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
        model = OkadaSourceSimple(analytic_grad=True)
        out = model(x_obs=x_obs, y_obs=y_obs, **kw)

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

        # The analytic strain matches the table to its 4-sig-fig precision for
        # every case. A small absolute floor covers entries near zero.
        assert torch.allclose(got, want, rtol=3e-3, atol=5e-6), (
            f"{name}/{comp} derivatives:\n"
            f"  got  {[f'{v:+.3e}' for v in got.tolist()]}\n"
            f"  want {[f'{v:+.3e}' for v in want.tolist()]}"
        )

_DC3D_GOLDEN = Path(__file__).resolve().parent / "data" / "dc3d_golden.json"


def _dc3d_map_jacobian(d):
    """DC3D fault-frame derivatives -> map-frame Jacobian J[..., i, j] = d ENU_i /
    d coord_j, coords (E, N, z). The stored 9-vector is column-major, so a
    row-major reshape gives J_fault^T; transpose, then rotate both legs by the
    strike matrix C (its own inverse): J_map = C @ J_fault @ C."""
    b, n = len(d["strike"]), d["n_points"]
    j_fault = t(d["derivatives_fault_frame"]).reshape(b, n, 3, 3).transpose(-1, -2)
    s, c = torch.sin(t(d["strike"])), torch.cos(t(d["strike"]))
    cmat = torch.zeros(b, n, 3, 3, dtype=DTYPE)
    cmat[..., 0, 0] = s[:, None]; cmat[..., 0, 1] = c[:, None]
    cmat[..., 1, 0] = c[:, None]; cmat[..., 1, 1] = -s[:, None]
    cmat[..., 2, 2] = 1.0
    return cmat @ j_fault @ cmat


def _dc3d_model_run(d, grad_params):
    """Run OkadaSource(analytic_grad=True) on the fixture, with the named inputs
    made leaves that require grad. Returns (Displacement, {name: leaf})."""
    leaves = {}

    def leaf(name):
        v = t(d[name])
        if name in grad_params:
            v = v.clone().requires_grad_(True)
            leaves[name] = v
        return v

    model = OkadaSource(poisson_ratio=d["poisson_ratio"], analytic_grad=True)
    out = model(
        x_obs=leaf("x_obs"), y_obs=leaf("y_obs"), z_obs=leaf("z_obs"),
        source_x=leaf("source_x"), source_y=leaf("source_y"),
        dip=leaf("dip"), strike=t(d["strike"]),
        centroid_depth=leaf("centroid_depth"),
        length=leaf("length"), width=leaf("width"),
        disl1=leaf("disl1"), disl2=leaf("disl2"), disl3=leaf("disl3"),
    )
    return out, leaves


# Points per fault at which the source-parameter gradients are checked. A
# systematic gradient error shows at any point, so a subset over all 24 faults is
# ample; it keeps the (custom-Function, non-vmap-able) one-hot backwards cheap.
_N_GRAD_POINTS = 8


def _dc3d_surface_simplified_run(d, grad_params):
    """OkadaSourceSimple(analytic_grad=True) on the fixture's z = 0 points, each
    flattened to its own batch element. With one point per element, a per-image
    parameter feeds exactly one output point, so ``grad(out_i.sum(), param)`` is
    the elementwise per-point gradient -- no one-hot loop needed. Returns
    (Displacement, {name: leaf [K]}, surface mask [B, N])."""
    z = t(d["z_obs"])
    mask = z == 0.0
    b, n = z.shape
    leaves = {}

    def leaf(name):
        v = t(d[name])[:, None].expand(b, n)[mask]      # [K]
        if name in grad_params:
            v = v.clone().requires_grad_(True)
            leaves[name] = v
        return v

    model = OkadaSourceSimple(poisson_ratio=d["poisson_ratio"], analytic_grad=True)
    out = model(
        x_obs=t(d["x_obs"])[mask][:, None], y_obs=t(d["y_obs"])[mask][:, None],
        source_x=leaf("source_x"), source_y=leaf("source_y"),
        dip=leaf("dip"), strike=t(d["strike"])[:, None].expand(b, n)[mask],
        centroid_depth=leaf("centroid_depth"),
        length=leaf("length"), width=leaf("width"),
        disl1=leaf("disl1"), disl2=leaf("disl2"), disl3=leaf("disl3"),
    )
    return out, leaves, mask


def _elementwise_source_grad(out, param):
    """d(ENU_i) / d param as [K, 3] when each param element feeds one output
    point (the flattened surface layout)."""
    g = torch.zeros(param.shape[0], 3, dtype=DTYPE)
    for i, comp in enumerate((out.e, out.n, out.u)):
        g[:, i] = torch.autograd.grad(comp.sum(), param, retain_graph=True)[0]
    return g


def _per_point_source_grad(out, param, b, n, k=_N_GRAD_POINTS):
    """d(ENU_i)[b, j] / d param[b] as [b, k, 3] for the first ``k`` points. Each
    per-image parameter feeds all N of its points, so one one-hot backward per
    (component, point index) isolates that point's gradient. The custom analytic
    Function is not vmap-able, so this stays an explicit loop."""
    k = min(k, n)
    g = torch.zeros(b, k, 3, dtype=DTYPE)
    for i, comp in enumerate((out.e, out.n, out.u)):
        for j in range(k):
            sel = torch.zeros(b, n, dtype=DTYPE)
            sel[:, j] = 1.0
            g[:, j, i] = torch.autograd.grad(comp, param, grad_outputs=sel,
                                             retain_graph=True)[0]
    return g


class TestOkadaDC3DVolume:
    """OkadaSource against Okada's own DC3D over a random volume of buried faults.

    External ground truth at depth, complementing the surface-only Okada-85
    Table 2 cases. The fixture ``data/dc3d_golden.json`` freezes the output of
    Okada's ``DC3D`` (DC3D.f90) for 24 random buried faults -- each with a
    *non-zero* strike, so the full map->fault assembly is exercised, not just
    the kernel -- observed at 32 points each, ~3/4 of them *below* the surface
    (z < 0). Regenerate with ``reference/gen_dc3d.py`` (needs gfortran + a local
    DC3D.f90); the committed JSON is all this test needs. See reference/README.md.
    """

    def _load(self):
        assert _DC3D_GOLDEN.is_file(), (
            f"{_DC3D_GOLDEN} missing; regenerate with reference/gen_dc3d.py"
        )
        return json.loads(_DC3D_GOLDEN.read_text())

    def test_dc3d_volume_displacement(self):
        d = self._load()
        # Material must match the fixture (nu = 0.25 -> alpha = 2/3).
        assert d["poisson_ratio"] == 0.25

        model = OkadaSource(poisson_ratio=d["poisson_ratio"])
        out = model(
            x_obs=t(d["x_obs"]), y_obs=t(d["y_obs"]), z_obs=t(d["z_obs"]),
            source_x=t(d["source_x"]), source_y=t(d["source_y"]),
            dip=t(d["dip"]), strike=t(d["strike"]),
            centroid_depth=t(d["centroid_depth"]),
            length=t(d["length"]), width=t(d["width"]),
            disl1=t(d["disl1"]), disl2=t(d["disl2"]), disl3=t(d["disl3"]),
        )
        got = torch.stack([out.e, out.n, out.u], dim=-1)   # [B, N, 3] ENU
        want = t(d["u_enu"])

        # Both sides are float64: the only gap is summation order / the kernel
        # refactor, ~1e-11 m on displacements up to ~0.4 m. atol covers the
        # handful of points where |u| ~ 1e-7 m (relative noise, not error).
        assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), (
            "OkadaSource disagrees with DC3D at depth: max abs diff "
            f"{(got - want).abs().max().item():.3e} m"
        )

    def test_dc3d_surface_simplified(self):
        """OkadaSourceSimple (surface-only fast path) against DC3D ground truth
        at the fixture's z = 0 points.

        The Simplified model is otherwise checked against Okada-85 Table 2 (three
        surface points, strike = 0) and by equality with OkadaSource at z = 0.
        This adds independent DC3D ground truth at the surface with a *non-zero*
        strike. Each surface point is flattened to its own batch element so the
        per-fault parameters line up with the ragged count of z = 0 points.
        """
        d = self._load()
        z = t(d["z_obs"])
        mask = z == 0.0                       # exact: generator sets 0.0
        assert int(mask.sum()) > 50, "fixture has too few surface points"

        b, n = z.shape

        def per_point(name):                  # [B] source param -> [K] surface pts
            return t(d[name])[:, None].expand(b, n)[mask]

        model = OkadaSourceSimple(poisson_ratio=d["poisson_ratio"])
        out = model(
            x_obs=t(d["x_obs"])[mask][:, None], y_obs=t(d["y_obs"])[mask][:, None],
            source_x=per_point("source_x"), source_y=per_point("source_y"),
            dip=per_point("dip"), strike=per_point("strike"),
            centroid_depth=per_point("centroid_depth"),
            length=per_point("length"), width=per_point("width"),
            disl1=per_point("disl1"), disl2=per_point("disl2"), disl3=per_point("disl3"),
        )
        got = torch.stack([out.e, out.n, out.u], dim=-1).reshape(-1, 3)  # [K, 3]
        want = t(d["u_enu"])[mask]                                       # [K, 3]

        assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), (
            "OkadaSourceSimple disagrees with DC3D at the surface: max abs diff "
            f"{(got - want).abs().max().item():.3e} m"
        )

    def test_dc3d_surface_simplified_analytic_grad_strain(self):
        """OkadaSourceSimple(analytic_grad=True)'s closed-form backward vs DC3D's
        exact derivatives, at the fixture's z = 0 points.

        The Simplified model takes no z_obs, so autograd reaches only the six
        *horizontal* components d(ENU)/d(E, N) -- the map-frame Jacobian's first
        two columns. Those depend solely on DC3D's fault-frame x/y derivatives
        (the strike rotation C is block-diagonal in the horizontal plane), so
        ``(C @ J_fault @ C)[:, :, :2]`` is the reference. The z-derivative column
        is only checkable through the full OkadaSource (see the volume test).
        """
        d = self._load()
        z = t(d["z_obs"])
        mask = z == 0.0
        b, n = z.shape
        k = int(mask.sum())
        assert k > 50, "fixture has too few surface points"

        def per_point(name):
            return t(d[name])[:, None].expand(b, n)[mask]

        x = t(d["x_obs"])[mask][:, None].clone().requires_grad_(True)   # [K, 1]
        y = t(d["y_obs"])[mask][:, None].clone().requires_grad_(True)
        strike = per_point("strike")

        model = OkadaSourceSimple(poisson_ratio=d["poisson_ratio"], analytic_grad=True)
        out = model(
            x_obs=x, y_obs=y,
            source_x=per_point("source_x"), source_y=per_point("source_y"),
            dip=per_point("dip"), strike=strike,
            centroid_depth=per_point("centroid_depth"),
            length=per_point("length"), width=per_point("width"),
            disl1=per_point("disl1"), disl2=per_point("disl2"), disl3=per_point("disl3"),
        )

        # Horizontal map-frame Jacobian d(ENU_i)/d(E, N) -> [K, 1, 3, 2].
        j_map_h = torch.zeros(k, 1, 3, 2, dtype=DTYPE)
        for i, comp in enumerate((out.e, out.n, out.u)):
            for j, g in enumerate(torch.autograd.grad(comp.sum(), (x, y), retain_graph=True)):
                j_map_h[..., i, j] = g

        j_fault = t(d["derivatives_fault_frame"])[mask].reshape(k, 3, 3).transpose(-1, -2)
        s, c = torch.sin(strike), torch.cos(strike)
        cmat = torch.zeros(k, 3, 3, dtype=DTYPE)
        cmat[:, 0, 0] = s; cmat[:, 0, 1] = c
        cmat[:, 1, 0] = c; cmat[:, 1, 1] = -s
        cmat[:, 2, 2] = 1.0
        j_map_ref = (cmat @ j_fault @ cmat)[:, :, :2].unsqueeze(1)   # [K, 1, 3, 2]

        assert torch.allclose(j_map_h, j_map_ref, rtol=1e-9, atol=1e-12), (
            "OkadaSourceSimple analytic_grad strain disagrees with DC3D at the "
            f"surface: max abs diff {(j_map_h - j_map_ref).abs().max().item():.3e} /m"
        )

    def test_dc3d_volume_analytic_grad_strain(self):
        """OkadaSource(analytic_grad=True)'s closed-form backward vs DC3D's exact
        spatial derivatives, over the same at-depth volume.

        DC3D returns the nine fault-frame derivatives ``d u_i / d x_j`` (stored
        column-major in ``derivatives_fault_frame``). The analytic backend's
        gradient w.r.t. the *map-frame* observation coordinates is the map-frame
        Jacobian ``J_map = d(ENU) / d(E, N, z)``; the two are related by the same
        strike rotation ``C`` used for displacement (``C`` is its own inverse):

            J_map = C @ J_fault @ C.

        Because each output point depends only on its own observation coordinate,
        one ``grad(out_i.sum(), coord_j)`` per (i, j) recovers the whole batched
        Jacobian in nine backward passes. This is the only external check of the
        analytic strain in the *interior* (z < 0) and for all nine components --
        Okada-85 Table 2 covers three surface points and horizontal derivatives.
        """
        d = self._load()

        x = t(d["x_obs"]).requires_grad_(True)
        y = t(d["y_obs"]).requires_grad_(True)
        z = t(d["z_obs"]).requires_grad_(True)
        strike = t(d["strike"])

        model = OkadaSource(poisson_ratio=d["poisson_ratio"], analytic_grad=True)
        out = model(
            x_obs=x, y_obs=y, z_obs=z,
            source_x=t(d["source_x"]), source_y=t(d["source_y"]),
            dip=t(d["dip"]), strike=strike,
            centroid_depth=t(d["centroid_depth"]),
            length=t(d["length"]), width=t(d["width"]),
            disl1=t(d["disl1"]), disl2=t(d["disl2"]), disl3=t(d["disl3"]),
        )

        # J_map[..., i, j] = d(ENU_i) / d(coord_j), coords (E=x_obs, N=y_obs, z).
        comps, coords = (out.e, out.n, out.u), (x, y, z)
        j_map = torch.zeros(*x.shape, 3, 3, dtype=DTYPE)
        for i, comp in enumerate(comps):
            grads = torch.autograd.grad(comp.sum(), coords, retain_graph=True)
            for j, g in enumerate(grads):
                j_map[..., i, j] = g

        j_map_ref = _dc3d_map_jacobian(d)

        # The analytic backward returns the closed-form Okada strain, so this is
        # a port-vs-Fortran check of the same formulas: agreement is ~1e-15 on
        # entries up to ~1e-4 /m. atol covers the near-zero components.
        assert torch.allclose(j_map, j_map_ref, rtol=1e-9, atol=1e-12), (
            "analytic_grad strain disagrees with DC3D at depth: max abs diff "
            f"{(j_map - j_map_ref).abs().max().item():.3e} /m"
        )

    # -- Source-parameter gradients ------------------------------------------
    # The three tests below check OkadaSource(analytic_grad=True)'s gradients
    # w.r.t. the *source* parameters (not the observation coords above) against
    # DC3D. Two are exact identities; the rest are DC3D finite differences.

    def test_dc3d_slip_gradient(self):
        """Exact: displacement is linear in the three dislocations, so
        d u / d disl_k is the unit-slip response G_k, which the fixture stores
        (from DC3D with disl = e_k). Matches to forward precision (~1e-11)."""
        d = self._load()
        b, n = d["n_faults"], d["n_points"]
        out, leaves = _dc3d_model_run(d, {"disl1", "disl2", "disl3"})
        for name in ("disl1", "disl2", "disl3"):
            got = _per_point_source_grad(out, leaves[name], b, n)
            want = t(d["param_gradients"][name])[:, :got.shape[1]]
            assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), (
                f"d u / d {name} disagrees with DC3D: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

    def test_dc3d_source_position_gradient(self):
        """Exact: a half-space is horizontally homogeneous, so the field depends
        only on (x_obs - source_x, y_obs - source_y). Hence
        d u / d source_x = -d u / d x_obs (and likewise for y) -- the negated
        horizontal spatial strain. Matches the closed-form backward to ~1e-15."""
        d = self._load()
        b, n = d["n_faults"], d["n_points"]
        out, leaves = _dc3d_model_run(d, {"source_x", "source_y"})
        j_map_ref = _dc3d_map_jacobian(d)
        for name, col_idx in (("source_x", 0), ("source_y", 1)):
            got = _per_point_source_grad(out, leaves[name], b, n)
            want = -j_map_ref[..., :got.shape[1], :, col_idx]   # -d(ENU)/d(coord)
            assert torch.allclose(got, want, rtol=1e-9, atol=1e-12), (
                f"d u / d {name} != -horizontal strain: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

    def test_dc3d_geometry_gradient_finite_diff(self):
        """length / width / centroid_depth gradients vs DC3D central differences
        (stored in the fixture). These flow through autograd of the exact forward,
        a different path from the closed-form strain. The gradient magnitudes are
        small (~1e-5 /m), so the finite-difference truncation is negligible and
        agreement is ~1e-11."""
        d = self._load()
        b, n = d["n_faults"], d["n_points"]
        names = ("length", "width", "centroid_depth")
        out, leaves = _dc3d_model_run(d, set(names))
        for name in names:
            got = _per_point_source_grad(out, leaves[name], b, n)
            want = t(d["param_gradients"][name])[:, :got.shape[1]]
            assert torch.allclose(got, want, rtol=1e-6, atol=1e-9), (
                f"d u / d {name} disagrees with DC3D finite diff: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

    def test_dc3d_dip_gradient_finite_diff(self):
        """dip gradient vs a DC3D central difference. The weakest of the set:
        analytic_grad computes dip by its *own* wide central difference, so this
        is finite-diff vs finite-diff -- it confirms sign and scale (agreement
        ~1e-6 on a gradient of magnitude ~0.5), not machine precision. The exact
        dip sensitivity is covered by the gradcheck tests below."""
        d = self._load()
        b, n = d["n_faults"], d["n_points"]
        out, leaves = _dc3d_model_run(d, {"dip"})
        got = _per_point_source_grad(out, leaves["dip"], b, n)
        want = t(d["param_gradients"]["dip"])[:, :got.shape[1]]
        assert torch.allclose(got, want, rtol=1e-3, atol=1e-5), (
            "d u / d dip disagrees with DC3D finite diff: max abs diff "
            f"{(got - want).abs().max().item():.3e}"
        )

    # -- Same source-parameter gradients for the surface-only Simplified model --
    # OkadaSourceSimple takes no z_obs, but its *source*-parameter gradients are
    # all well-defined at the surface; only the observation z-derivative is out
    # of reach. Checked at the fixture's z = 0 points against the same DC3D
    # references.

    def test_dc3d_surface_simplified_exact_gradients(self):
        """OkadaSourceSimple exact source-parameter gradients at z = 0: slips
        (unit-slip responses) and source position (negated horizontal strain)."""
        d = self._load()
        out, leaves, mask = _dc3d_surface_simplified_run(
            d, {"disl1", "disl2", "disl3", "source_x", "source_y"})

        for name in ("disl1", "disl2", "disl3"):
            got = _elementwise_source_grad(out, leaves[name])
            want = t(d["param_gradients"][name])[mask]
            assert torch.allclose(got, want, rtol=1e-9, atol=1e-11), (
                f"Simplified d u / d {name} disagrees with DC3D: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

        j_map_ref = _dc3d_map_jacobian(d)[mask]        # [K, 3, 3]
        for name, col_idx in (("source_x", 0), ("source_y", 1)):
            got = _elementwise_source_grad(out, leaves[name])
            want = -j_map_ref[..., :, col_idx]
            assert torch.allclose(got, want, rtol=1e-9, atol=1e-12), (
                f"Simplified d u / d {name} != -horizontal strain: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

    def test_dc3d_surface_simplified_finite_diff_gradients(self):
        """OkadaSourceSimple finite-difference source-parameter gradients at
        z = 0: dip / length / width / centroid_depth vs the DC3D references.
        Same tolerances as the volume model (dip is the loose one)."""
        d = self._load()
        tol = {
            "length": (1e-6, 1e-9), "width": (1e-6, 1e-9),
            "centroid_depth": (1e-6, 1e-9), "dip": (1e-3, 1e-5),
        }
        out, leaves, mask = _dc3d_surface_simplified_run(d, set(tol))
        for name, (rtol, atol) in tol.items():
            got = _elementwise_source_grad(out, leaves[name])
            want = t(d["param_gradients"][name])[mask]
            assert torch.allclose(got, want, rtol=rtol, atol=atol), (
                f"Simplified d u / d {name} disagrees with DC3D: max abs diff "
                f"{(got - want).abs().max().item():.3e}"
            )

    def test_dc3d_volume_has_at_depth_coverage(self):
        """Guard the fixture itself: it must actually probe the volume (z < 0),
        not just the surface, or the checks above would silently weaken."""
        d = self._load()
        z = t(d["z_obs"])
        assert (z < 0).float().mean() > 0.5      # majority below surface
        assert float(z.min()) < -1e3             # genuinely deep points
        # And the strikes are non-trivial, so the rotation path is exercised.
        strike = t(d["strike"]).abs()
        assert float(strike.min()) > 0.1
        # The derivative fixture must be present and correctly shaped.
        deriv = t(d["derivatives_fault_frame"])
        assert deriv.shape == (z.shape[0], z.shape[1], 9)
        # Source-parameter gradient references, each [B, N, 3].
        for name in ("disl1", "disl2", "disl3",
                     "dip", "length", "width", "centroid_depth"):
            assert t(d["param_gradients"][name]).shape == (z.shape[0], z.shape[1], 3)


class TestAnalyticBackend:
    """OkadaSourceSimple(analytic_grad=True): exact forward + closed-form Okada-strain backward.

    Gradients w.r.t. observation coords, source location, length, width,
    centroid_depth and strike are analytic (closed-form Okada strain); dip and
    the slips are delegated to an exact-forward pass.
    """

    def test_forward_matches_exact_simplified(self):
        """Forward values are identical to the exact OkadaSourceSimple (drop-in)."""
        p = random_params(5, batch=3, n=7)
        ref = OkadaSourceSimple(smooth_grad=False)
        ana = OkadaSourceSimple(analytic_grad=True)
        kw = dict(
            x_obs=p["x_obs"], y_obs=p["y_obs"],
            source_x=p["source_x"], source_y=p["source_y"],
            dip=p["dip"], strike=p["strike"], centroid_depth=p["depth"],
            length=p["length"], width=p["width"],
            disl1=p["d1"], disl2=p["d2"], disl3=p["d3"],
        )
        a, b = ana(**kw), ref(**kw)
        for nm in ("e", "n", "u"):
            assert torch.allclose(getattr(a, nm), getattr(b, nm),
                                  rtol=1e-12, atol=1e-15)

    def test_obs_gradient_matches_finite_difference(self):
        """Analytic observation-coordinate gradient matches central differences
        of the exact forward at a benign (smooth) configuration."""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        kw = dict(
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([0.9]), strike=t([0.7]), centroid_depth=t([9e3]),
            length=t([8e3]), width=t([4e3]),
            disl1=t([1.0]), disl2=t([-0.5]), disl3=t([0.2]),
        )
        x0, y0 = t([[3e3, -7e3]]), t([[5e3, 9e3]])
        xg = x0.clone().requires_grad_(True)
        out = ana(x_obs=xg, y_obs=y0.clone(), **kw)
        (gx,) = torch.autograd.grad(out.u.sum(), xg)

        h = 1.0
        up = ref(x_obs=x0 + h, y_obs=y0, **kw).u
        dn = ref(x_obs=x0 - h, y_obs=y0, **kw).u
        fd = (up - dn) / (2 * h)
        assert torch.allclose(gx, fd, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize(
        "param,h",
        [("centroid_depth", 1.0), ("length", 1.0), ("width", 1.0),
         ("source_x", 1.0), ("source_y", 1.0), ("strike", 1e-6)],
    )
    def test_geometry_gradient_matches_finite_difference(self, param, h):
        """Each analytic geometry/location gradient (everything except dip and
        the slips) matches central differences of the exact forward."""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        base = dict(
            x_obs=t([[3e3, -7e3, 12e3]]), y_obs=t([[5e3, 9e3, -4e3]]),
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([0.9]), strike=t([0.7]), centroid_depth=t([9e3]),
            length=t([8e3]), width=t([4e3]),
            disl1=t([1.0]), disl2=t([-0.5]), disl3=t([0.2]),
        )

        def loss(model, **kw):
            o = model(**kw)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        kw = {k: (v.clone().requires_grad_(True) if k == param else v.clone())
              for k, v in base.items()}
        loss(ana, **kw).backward()
        analytic = kw[param].grad

        kp = dict(base); kp[param] = base[param] + h
        km = dict(base); km[param] = base[param] - h
        fd = (loss(ref, **kp) - loss(ref, **km)) / (2 * h)
        # FD of a scalar loss -> gradient summed over the param's entries.
        assert torch.allclose(analytic.sum(), fd, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("dip_deg", [45.0, 89.0, 89.99, 90.0, 90.001])
    def test_dip_gradient_accurate_through_vertical(self, dip_deg):
        """dip gradient matches the exact forward's derivative right through the
        vertical manifold -- including exactly dip = 90 deg -- where the smoothed
        path drifts by orders of magnitude and naive autograd is ill-conditioned.

        Reference is a wide (1e-3 rad) central difference of the exact forward:
        the kernel's 1/cos(dip)^2 terms lose precision for |cos(dip)| <~ 1e-3, so
        a narrow-step FD is itself unreliable near vertical (this is exactly why
        the module uses a wide step internally)."""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        base = dict(
            x_obs=t([[3e3, -7e3, 12e3]]), y_obs=t([[5e3, 9e3, -4e3]]),
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([math.radians(dip_deg)]), strike=t([0.7]),
            centroid_depth=t([6e3]), length=t([8e3]), width=t([4e3]),
            disl1=t([1.0]), disl2=t([-0.5]), disl3=t([0.2]),
        )

        def loss(model, **kw):
            o = model(**kw)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        kw = {k: (v.clone().requires_grad_(True) if k == "dip" else v.clone())
              for k, v in base.items()}
        loss(ana, **kw).backward()
        g = kw["dip"].grad

        h = 1e-3
        kp = dict(base); kp["dip"] = base["dip"] + h
        km = dict(base); km["dip"] = base["dip"] - h
        fd = (loss(ref, **kp) - loss(ref, **km)) / (2 * h)
        assert torch.isfinite(g).all()
        assert torch.allclose(g.sum(), fd, rtol=1e-3, atol=1e-9)

    @pytest.mark.parametrize("dip_deg,rtol", [(60.0, 1e-9), (89.5, 1e-6),
                                              (90.0, 2e-3)])
    def test_dip_gradient_matches_richardson_reference(self, dip_deg, rtol):
        """dip gradient converges to the *true* derivative -- not merely to the
        module's own wide-step FD -- right through the vertical manifold.

        ``test_dip_gradient_accurate_through_vertical`` compares the internal
        1e-3 FD against a 1e-3 FD of the reference forward, which only pins the
        gradient to the truncation scale of that step. Here the reference is an
        O(h^4) Richardson extrapolation ``(4*D(h/2) - D(h)) / 3`` of the exact
        forward, i.e. genuine ground truth. Off-vertical (60, 89.5 deg) the dip
        gradient comes from autograd of the exact forward, which agrees with the
        reference to ~machine precision -- limited only by the reference itself
        (~1e-12 at 60 deg, ~1e-7 at 89.5 deg where the forward starts to
        stiffen), hence the tight tolerances. At exactly 90 deg autograd is
        ill-conditioned so the module falls back to the wide FD, and the forward
        itself loses precision (|cos(dip)| collapses), so both the module and the
        reference are limited to ~1e-3 and the tolerance is relaxed accordingly."""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        base = dict(
            x_obs=t([[3e3, -7e3, 12e3]]), y_obs=t([[5e3, 9e3, -4e3]]),
            source_x=t([0.0]), source_y=t([0.0]),
            strike=t([0.7]), centroid_depth=t([6e3]), length=t([8e3]),
            width=t([4e3]), disl1=t([1.0]), disl2=t([-0.5]), disl3=t([0.2]),
        )
        dip0 = t([math.radians(dip_deg)])

        def loss(model, dip):
            o = model(dip=dip, **base)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        dipg = dip0.clone().requires_grad_(True)
        loss(ana, dipg).backward()
        g = dipg.grad.sum()

        h = 1e-3
        cd = lambda step: (loss(ref, dip0 + step) - loss(ref, dip0 - step)) / (2 * step)
        richardson = (4.0 * cd(h / 2) - cd(h)) / 3.0
        assert torch.allclose(g, richardson, rtol=rtol, atol=1e-9)

    @pytest.mark.parametrize("dip_deg", [89.9, 89.95, 89.97, 90.05, 90.1])
    def test_dip_gradient_inside_fd_band_no_resonance(self, dip_deg):
        """Inside the near-vertical FD band, at dips *offset* from exactly 90 deg.

        These are the points a single fixed-step central difference silently
        fails: for any step h, some in-band dip puts one sample on the removable
        singularity at 90 deg and the FD relative error spikes to ~1e-2 (worse
        than the autograd it replaces). The module's Richardson fallback uses wide
        steps that clear vertical for every in-band dip, so it stays ~1e-6.

        Ground truth is a *wide-base* Richardson (base step 1.5e-2 rad) whose
        samples all sit outside the ill-conditioned band -- validated to ~1e-7 by
        its agreement across base steps 1e-2..3e-2 rad. (The 1e-3-base Richardson
        used in the sibling test is itself unreliable at these offsets, which is
        exactly the failure being guarded against.)"""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        base = dict(
            x_obs=t([[3e3, -7e3, 12e3]]), y_obs=t([[5e3, 9e3, -4e3]]),
            source_x=t([0.0]), source_y=t([0.0]), strike=t([0.7]),
            centroid_depth=t([6e3]), length=t([8e3]), width=t([4e3]),
            disl1=t([1.0]), disl2=t([-0.5]), disl3=t([0.2]),
        )

        def loss(model, dip):
            o = model(dip=dip, **base)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        dip0 = t([math.radians(dip_deg)])
        dipg = dip0.clone().requires_grad_(True)
        loss(ana, dipg).backward()
        g = dipg.grad.sum()

        H = 1.5e-2
        cd = lambda s: (loss(ref, dip0 + s) - loss(ref, dip0 - s)) / (2 * s)
        wide_ref = (4.0 * cd(H / 2) - cd(H)) / 3.0
        assert torch.isfinite(g).all()
        assert torch.allclose(g, wide_ref, rtol=1e-4, atol=1e-9)

    def test_dip_gradient_mixed_batch_autograd_and_fd(self):
        """Per-element hand-off: in one batch, an off-vertical element takes the
        autograd path (tight) and a vertical element the wide-FD fallback, and
        both are correct. Guards the ``torch.where(near_vert, ...)`` selection --
        a NaN/ill-conditioned autograd value at the vertical element must not
        leak into the off-vertical one, and vice versa."""
        ana = OkadaSourceSimple(analytic_grad=True)
        ref = OkadaSourceSimple(smooth_grad=False)
        dips = t([math.radians(45.0), math.radians(90.0)])
        base = dict(
            x_obs=t([[3e3, -7e3, 12e3], [3e3, -7e3, 12e3]]),
            y_obs=t([[5e3, 9e3, -4e3], [5e3, 9e3, -4e3]]),
            source_x=t([0.0, 0.0]), source_y=t([0.0, 0.0]),
            strike=t([0.7, 0.7]), centroid_depth=t([6e3, 6e3]),
            length=t([8e3, 8e3]), width=t([4e3, 4e3]),
            disl1=t([1.0, 1.0]), disl2=t([-0.5, -0.5]), disl3=t([0.2, 0.2]),
        )

        def loss(model, dip):
            o = model(dip=dip, **base)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        dipg = dips.clone().requires_grad_(True)
        loss(ana, dipg).backward()
        g = dipg.grad
        assert torch.isfinite(g).all()

        # Per-element wide-FD reference of the exact forward (matches the module's
        # own scheme at the vertical element; the tight autograd path at 45 deg is
        # far inside its trust region so a wide FD still pins it to ~1e-4).
        h = 1e-3
        kp = dips + h
        km = dips - h
        op = ref(dip=kp, **base)
        om = ref(dip=km, **base)
        lp = (op.e**2 + op.n**2 + op.u**2).sum(dim=1)
        lm = (om.e**2 + om.n**2 + om.u**2).sum(dim=1)
        fd = (lp - lm) / (2 * h)
        assert torch.allclose(g, fd, rtol=1e-3, atol=1e-9)

    @staticmethod
    def _f32_base(dtype):
        tt = lambda x: torch.as_tensor(x, dtype=dtype)
        return dict(
            x_obs=tt([[3e3, -7e3, 12e3]]), y_obs=tt([[5e3, 9e3, -4e3]]),
            source_x=tt([0.0]), source_y=tt([0.0]),
            strike=tt([0.7]), centroid_depth=tt([6e3]), length=tt([8e3]),
            width=tt([4e3]), disl1=tt([1.0]), disl2=tt([-0.5]), disl3=tt([0.2]),
        )

    @pytest.mark.parametrize("mode", [{}, {"smooth_grad": True}],
                             ids=["exact", "smooth_grad"])
    @pytest.mark.parametrize("dip_deg", [60.0, 89.5, 89.99, 90.0, 90.001])
    def test_forward_float32_accurate_near_vertical(self, dip_deg, mode):
        """The float32 *forward* is accurate through the near-vertical band.

        The general-dip ("_a") Okada terms carry ``1/cos(dip)^2`` factors whose
        numerators vanish like ``cos(dip)^2``; evaluated naively in float32 the
        subtraction cancels to noise and the output is wrong by *orders of
        magnitude* for ``0 < |cos(dip)| < ~1e-2`` (rel. err up to ~7 at
        89.99 deg). The model promotes that band to float64 internally and casts
        back, so the float32 output tracks float64 -- and the returned dtype is
        still float32 (the promotion is invisible).

        ``smooth_grad`` shares the same numerator cancellation (its blends/safe
        divisions guard the *denominator*, not the ``big - big`` numerator, and
        it only swaps toward the dip=90 form for ``|cos| < ~1e-4``), so it is
        covered too. Each mode is compared float32-vs-float64 in the *same* mode,
        so smooth_grad's intended near-singularity perturbation cancels out and
        only the float32 conditioning error is under test."""
        m32 = OkadaSourceSimple(internal_dtype=torch.float32, **mode)
        m64 = OkadaSourceSimple(internal_dtype=torch.float64, **mode)
        dip = math.radians(dip_deg)
        o32 = m32(dip=t([dip]).float(), **self._f32_base(torch.float32))
        o64 = m64(dip=t([dip]), **self._f32_base(torch.float64))
        assert o32.e.dtype == torch.float32          # promotion cast back
        v32 = torch.cat([o32.e, o32.n, o32.u]).double()
        v64 = torch.cat([o64.e, o64.n, o64.u])
        assert (v32 - v64).abs().max() <= 1e-3 * v64.abs().max()

    @pytest.mark.parametrize("mode", [{}, {"smooth_grad": True}],
                             ids=["exact", "smooth_grad"])
    @pytest.mark.parametrize("dip_deg", [78.0, 82.0, 84.0])
    def test_forward_float32_band_edge_margin(self, dip_deg, mode):
        """Characterise pure-float32 accuracy just *outside* the promotion band.

        These dips have ``|cos(dip)| = 0.21/0.14/0.105 > F32_VERTICAL_BAND = 0.1``,
        so the model does NOT promote and runs in pure float32 -- this is the edge
        the 0.1 threshold leaves to float32. It exposes that 0.1 is *optimistic*:
        error grows steeply toward the band and, right at the edge (84 deg), the
        exact mode reaches ~1.3e-3 -- over the 1e-3 contract that the promoted band
        satisfies (see the ``F32_VERTICAL_BAND`` note; the un-promoted edge is the
        one regime that misses it). Widening the band to 0.2 would promote these
        dips and restore 1e-3; it is left at 0.1 for speed for now. This test pins
        the edge to a 2e-3 bound so the small overshoot cannot silently grow (via a
        narrowed band or numeric drift) without a failure here.

        The 78/82 deg cases stay well under 1e-3 (~2-4e-4); 84 deg is the outlier.
        """
        m32 = OkadaSourceSimple(internal_dtype=torch.float32, **mode)
        m64 = OkadaSourceSimple(internal_dtype=torch.float64, **mode)
        dip = math.radians(dip_deg)
        assert abs(math.cos(dip)) > F32_VERTICAL_BAND     # genuinely un-promoted
        o32 = m32(dip=t([dip]).float(), **self._f32_base(torch.float32))
        o64 = m64(dip=t([dip]), **self._f32_base(torch.float64))
        v32 = torch.cat([o32.e, o32.n, o32.u]).double()
        v64 = torch.cat([o64.e, o64.n, o64.u])
        assert (v32 - v64).abs().max() <= 2e-3 * v64.abs().max()

    def test_f32_vertical_band_constructor_override(self):
        """The ``f32_vertical_band`` knob widens the float64-promotion band.

        At dip 83 deg (``|cos| = 0.122``) the default band (0.1) leaves the batch
        in float32 -- near the edge, so ~1e-3 error -- while a model built with
        ``f32_vertical_band=0.2`` promotes it to float64 and tracks the float64
        reference to float32-cast precision. Verifies the constructor arg is
        threaded through ``_compute_dtype`` per instance."""
        dip = t([math.radians(83.0)])
        assert abs(math.cos(float(dip))) > F32_VERTICAL_BAND     # default: un-promoted

        default = OkadaSourceSimple(internal_dtype=torch.float32)
        widened = OkadaSourceSimple(internal_dtype=torch.float32, f32_vertical_band=0.2)
        assert default._compute_dtype(dip) == torch.float32      # stays float32
        assert widened._compute_dtype(dip) == torch.float64      # promoted

        m64 = OkadaSourceSimple(internal_dtype=torch.float64)
        ow = widened(dip=dip.float(), **self._f32_base(torch.float32))
        o64 = m64(dip=dip, **self._f32_base(torch.float64))
        assert ow.e.dtype == torch.float32                       # promotion cast back
        vw = torch.cat([ow.e, ow.n, ow.u]).double()
        v64 = torch.cat([o64.e, o64.n, o64.u])
        # promoted -> float64-accurate up to the final float32 cast (~1e-6)
        assert (vw - v64).abs().max() <= 1e-5 * v64.abs().max()

    @pytest.mark.parametrize("dip_deg", [60.0, 89.5, 89.99, 90.0, 90.001])
    def test_dip_gradient_float32_accurate_through_vertical(self, dip_deg):
        """float32 dip gradient tracks the float64 gradient right through the
        vertical manifold -- not merely finite there.

        Both the ill-conditioned forward *and* the wide-step FD subtraction of
        two nearly-equal losses cancel catastrophically in float32 near vertical
        (relative error ~1e3-1e4 before the fix). The analytic backend promotes
        the near-vertical forward and runs the whole dip FD in float64, casting
        only the resulting gradient back, so float32 matches float64."""
        ana32 = OkadaSourceSimple(analytic_grad=True, internal_dtype=torch.float32)
        ana64 = OkadaSourceSimple(analytic_grad=True, internal_dtype=torch.float64)

        def dip_grad(model, dtype):
            dip = torch.as_tensor([math.radians(dip_deg)],
                                  dtype=dtype).requires_grad_(True)
            o = model(dip=dip, **self._f32_base(dtype))
            (o.e**2 + o.n**2 + o.u**2).sum().backward()
            return dip.grad

        g32 = dip_grad(ana32, torch.float32)
        g64 = dip_grad(ana64, torch.float64)
        assert g32.dtype == torch.float32
        assert torch.isfinite(g32).all()
        assert torch.allclose(g32.double(), g64, rtol=1e-3, atol=1e-9)

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradcheck_all_params(self, device):
        """gradcheck through the analytic module for every continuous input:
        analytic obs/source gradients + delegated geometry/slip gradients."""
        m = OkadaSourceSimple(analytic_grad=True).to(device)

        def td(x):
            return t(x, device=device)

        base = dict(
            x_obs=td([[3e3, -7e3, 12e3]]), y_obs=td([[5e3, 9e3, -4e3]]),
            source_x=td([0.0]), source_y=td([0.0]),
            dip=td([0.9]), strike=td([0.7]), centroid_depth=td([9e3]),
            length=td([8e3]), width=td([4e3]),
            disl1=td([1.0]), disl2=td([-0.5]), disl3=td([0.2]),
        )
        names = ("x_obs", "y_obs", "source_x", "source_y", "dip",
                 "centroid_depth", "length", "width", "disl1", "disl2", "disl3")
        leaves = {k: base[k].clone().requires_grad_(True) for k in names}

        def f(*vals):
            kw = dict(base)
            kw.update({n: v for n, v in zip(names, vals)})
            out = m(**kw)
            return torch.cat([out.e.reshape(-1), out.n.reshape(-1),
                              out.u.reshape(-1)])

        assert torch.autograd.gradcheck(
            f, tuple(leaves[k] for k in names), eps=1e-6, atol=1e-6, rtol=1e-3)

    def test_obs_gradient_finite_on_trace(self):
        """On a surface-crossing grid the analytic obs-gradient stays finite
        (the closed-form strain has no NaN trap that plain autograd of the exact
        forward hits at cd = 0)."""
        L, W = 10e3, 5e3
        n = 21
        g = torch.linspace(-8e3, 8e3, n, dtype=DTYPE)
        X, Y = torch.meshgrid(g, g, indexing="xy")
        x_obs = X.reshape(1, -1).requires_grad_(True)
        y_obs = Y.reshape(1, -1)
        out = OkadaSourceSimple(analytic_grad=True)(
            x_obs=x_obs, y_obs=y_obs,
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([math.pi / 2.0]), strike=t([0.3]),   # vertical dip: cd = 0
            centroid_depth=t([0.5 * W]), length=t([L]), width=t([W]),
            disl1=t([1.0]), disl2=t([0.5]), disl3=t([0.0]),
        )
        (gx,) = torch.autograd.grad((out.e**2 + out.n**2 + out.u**2).sum(), x_obs)
        assert torch.isfinite(gx).all()


class TestFullAnalyticBackend:
    """OkadaSource(analytic_grad=True) (z != 0): exact forward + closed-form DC3D-strain backward.

    Observation-coordinate gradients (x/y/z_obs, source) are analytic; geometry
    and slips via autograd of the exact forward; dip via a wide central difference.
    """

    def _base(self, dip=0.9, z=None):
        if z is None:
            z = t([[-2e3, -5e3, -0.5e3]])
        return dict(
            x_obs=t([[4e3, -9e3, 15e3]]), y_obs=t([[7e3, 3e3, -6e3]]), z_obs=z,
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([dip]), strike=t([0.6]), centroid_depth=t([9e3]),
            length=t([12e3]), width=t([6e3]),
            disl1=t([1.3]), disl2=t([-0.7]), disl3=t([0.2]),
        )

    def test_forward_matches_exact(self):
        a = OkadaSource(analytic_grad=True)(**self._base())
        b = OkadaSource(smooth_grad=False)(**self._base())
        for nm in ("e", "n", "u"):
            assert torch.allclose(getattr(a, nm), getattr(b, nm),
                                  rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("coord", ["x_obs", "y_obs", "z_obs"])
    def test_obs_strain_matches_finite_difference(self, coord):
        """Analytic d(e,n,u)/d(coord) matches central differences of the exact
        forward (the full real+image+UC strain assembly)."""
        ana = OkadaSource(analytic_grad=True)
        ref = OkadaSource(smooth_grad=False)
        base = self._base()
        cg = {k: (v.clone().requires_grad_(True) if k == coord else v.clone())
              for k, v in base.items()}
        out = ana(**cg)
        # accumulate the analytic gradient of a fixed linear functional
        w = (1.3, -0.4, 0.8)
        (w[0] * out.e.sum() + w[1] * out.n.sum() + w[2] * out.u.sum()).backward()
        g = cg[coord].grad

        h = 1.0
        kp = dict(base); kp[coord] = base[coord] + h
        km = dict(base); km[coord] = base[coord] - h
        op, om = ref(**kp), ref(**km)
        fd = (w[0] * (op.e - om.e) + w[1] * (op.n - om.n)
              + w[2] * (op.u - om.u)) / (2 * h)
        assert torch.allclose(g, fd, rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("z_kind", ["per_pixel", "scalar"])
    def test_gradcheck_all_params(self, z_kind):
        m = OkadaSource(analytic_grad=True)
        z = t([[-2e3, -5e3, -0.5e3]]) if z_kind == "per_pixel" else t([-3e3])
        base = self._base(z=z)
        names = ("x_obs", "y_obs", "z_obs", "source_x", "source_y", "dip",
                 "strike", "centroid_depth", "length", "width",
                 "disl1", "disl2", "disl3")
        leaves = {k: base[k].clone().requires_grad_(True) for k in names}

        def f(*vals):
            kw = dict(base)
            kw.update({n: v for n, v in zip(names, vals)})
            o = m(**kw)
            return torch.cat([o.e.reshape(-1), o.n.reshape(-1), o.u.reshape(-1)])

        assert torch.autograd.gradcheck(
            f, tuple(leaves[k] for k in names), eps=1e-6, atol=1e-6, rtol=1e-3)

    @pytest.mark.parametrize("dip_deg", [45.0, 89.99, 90.0, 90.001])
    def test_dip_gradient_through_vertical(self, dip_deg):
        ana = OkadaSource(analytic_grad=True)
        ref = OkadaSource(smooth_grad=False)
        base = self._base(dip=math.radians(dip_deg))

        def loss(model, **kw):
            o = model(**kw)
            return (o.e**2 + o.n**2 + o.u**2).sum()

        kw = {k: (v.clone().requires_grad_(True) if k == "dip" else v.clone())
              for k, v in base.items()}
        loss(ana, **kw).backward()
        g = kw["dip"].grad

        h = 1e-3
        kp = dict(base); kp["dip"] = base["dip"] + h
        km = dict(base); km["dip"] = base["dip"] - h
        fd = (loss(ref, **kp) - loss(ref, **km)) / (2 * h)
        assert torch.isfinite(g).all()
        assert torch.allclose(g.sum(), fd, rtol=1e-3, atol=1e-9)

    def test_obs_gradient_finite_on_fault_plane(self):
        """Observation points on the fault plane hit the singular configuration;
        the analytic obs-gradient must stay finite."""
        n = 21
        g = torch.linspace(-10e3, 10e3, n, dtype=DTYPE)
        x_obs = g.reshape(1, -1).requires_grad_(True)
        out = OkadaSource(analytic_grad=True)(
            x_obs=x_obs, y_obs=torch.zeros(1, n, dtype=DTYPE),
            z_obs=torch.full((1, n), -3e3, dtype=DTYPE),
            source_x=t([0.0]), source_y=t([0.0]),
            dip=t([math.pi / 2.0]), strike=t([0.0]),
            centroid_depth=t([5e3]), length=t([12e3]), width=t([6e3]),
            disl1=t([1.0]), disl2=t([0.5]), disl3=t([0.0]),
        )
        (gx,) = torch.autograd.grad((out.e**2 + out.n**2 + out.u**2).sum(), x_obs)
        assert torch.isfinite(gx).all()


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

    def test_no_nan_values_or_gradients_smooth_grad(self):
        """smooth_grad mode: forward and backward must stay finite on a
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

        model = OkadaSourceSimple(smooth_grad=True)
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
    def test_gradcheck_smooth_grad(self, device):
        """Full autograd gradcheck at a benign configuration (float64).

        Parametrized over DEVICES so autograd is exercised on CUDA too when
        available (matching the Mogi/Penny suites)."""
        model = OkadaSourceSimple(smooth_grad=True).to(device)

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
    def test_gradcheck_fullz_smooth_grad(self, device):
        """gradcheck through OkadaSource at depth (smooth_grad), where the
        UC kernel and z-dependence engage -- complements the Simplified
        gradcheck above."""
        model = OkadaSource(smooth_grad=True).to(device)

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
        """Exact mode (smooth_grad=False) must give finite, non-zero gradients
        at an off-fault configuration, where the analytic kernels are smooth.
        Complements the smooth_grad on-trace gradient-health test."""
        model = OkadaSourceSimple(smooth_grad=False)
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

class TestInputValidation:
    """Unphysical fault dimensions must raise, not silently misbehave.

    A *negative* ``length``/``width`` flips the sign of the Chinnery corner
    subtraction and returns a plausible but sign-flipped displacement (a zero
    dimension collapses the fault to zero output); ``centroid_depth <= 0`` puts
    the centroid at/above the free surface, outside the buried-fault half-space.
    None of these raise on their own, so guard them for both Okada classes.
    """

    # base geometry known to be valid; each test overrides one dimension.
    _base = dict(x_obs=[[1e3, 5e3]], y_obs=[[2e3, -3e3]],
                 source_x=[0.0], source_y=[0.0], dip=[0.7], strike=[0.3],
                 depth=[8e3], length=[10e3], width=[4e3],
                 d1=[1.0], d2=[0.0], d3=[0.0])

    @pytest.mark.parametrize("field,bad,msg", [
        ("length", 0.0, "length"),
        ("length", -10e3, "length"),
        ("width", 0.0, "width"),
        ("width", -4e3, "width"),
        ("depth", 0.0, "centroid_depth"),
        ("depth", -8e3, "centroid_depth"),
    ])
    def test_simple_rejects_degenerate_geometry(self, field, bad, msg):
        p = dict(self._base); p[field] = [bad]
        with pytest.raises(ValueError, match=msg):
            run_simplified(**p)

    @pytest.mark.parametrize("field,bad,msg", [
        ("length", -10e3, "length"),
        ("width", 0.0, "width"),
        ("depth", -8e3, "centroid_depth"),
    ])
    def test_fullz_rejects_degenerate_geometry(self, field, bad, msg):
        p = dict(self._base); p[field] = [bad]
        with pytest.raises(ValueError, match=msg):
            run_fullz(z_obs=[[0.0, 0.0]], **p)

    def test_rejects_if_any_batch_element_degenerate(self):
        # a single bad element in an otherwise valid batch must still raise.
        p = random_params(7, batch=3, n=4)
        p["width"] = p["width"].clone(); p["width"][1] = -1.0
        with pytest.raises(ValueError, match="width"):
            run_simplified(**p)

    # dip = 90 deg vertical; top-edge depth = centroid_depth - width/2.
    _vert = dict(_base, dip=[math.pi / 2], width=[4e3])

    @pytest.mark.parametrize("runner", [run_simplified, run_fullz])
    def test_protruding_fault_warns(self, runner):
        # top edge above the free surface (centroid 1 km, half-width 2 km) is a
        # soft error: it warns but still returns a result.
        p = dict(self._vert); p["depth"] = [1e3]
        if runner is run_fullz:
            p = dict(p, z_obs=[[0.0, 0.0]])
        with pytest.warns(UserWarning, match="top edge is above the free surface"):
            runner(**p)

    def test_surface_rupturing_fault_does_not_warn(self):
        # top edge exactly at the surface (centroid = half-width) is a valid
        # surface-rupturing fault and must stay silent.
        import warnings
        p = dict(self._vert); p["depth"] = [2e3]      # 2 km == width/2
        with warnings.catch_warnings():
            warnings.simplefilter("error")            # any warning -> failure
            run_simplified(**p)


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

        simp = OkadaSourceSimple(smooth_grad=True).to(device)
        out_s = simp(
            x_obs=to("x_obs"), y_obs=to("y_obs"),
            source_x=to("source_x"), source_y=to("source_y"),
            dip=to("dip"), strike=to("strike"),
            centroid_depth=to("depth"), length=to("length"), width=to("width"),
            disl1=to("d1"), disl2=to("d2"), disl3=to("d3"),
        )
        assert out_s.e.device.type == "cuda"
        assert torch.isfinite(out_s.e).all() and torch.isfinite(out_s.u).all()

        full = OkadaSource(smooth_grad=True).to(device)
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


class TestDtypeNumericalFloors:
    """The ``num_eps`` guard scales with ``internal_dtype``.

    A fixed ``1e-12`` is right for float64 but lies below float32's ~1e-7
    epsilon, so in float32 that floor is effectively inert -- it no longer does
    what "numerical guard for denominators/sqrt" claims. ``num_eps`` now defaults
    to ``None`` -> a dtype-appropriate floor. These tests pin that contract and
    assert forward + gradient stay finite in float32 on the singular manifolds.

    (Note: with the current kxi/ket edge-masking the float64->float32 floor change
    does not by itself flip any of these configs from non-finite to finite -- the
    masks already protect them. The tests are a finiteness/contract guard, not a
    regression reproducer.)
    """

    def test_default_num_eps_scales_with_dtype(self):
        from torchdeform.sources.base import default_num_eps
        assert default_num_eps(torch.float64) == 1e-12
        assert default_num_eps(torch.float32) > default_num_eps(torch.float64)
        assert default_num_eps(torch.float16) >= default_num_eps(torch.float32)

    def test_float64_num_eps_unchanged(self):
        """Regression: the float64 path keeps its validated 1e-12 floor."""
        assert OkadaSource()._resolve_num_eps() == 1e-12
        assert OkadaSourceSimple()._resolve_num_eps() == 1e-12

    def test_explicit_num_eps_overrides(self):
        assert OkadaSource(num_eps=1e-9)._resolve_num_eps() == 1e-9
        assert OkadaSource(internal_dtype=torch.float32,
                           num_eps=1e-9)._resolve_num_eps() == 1e-9

    @pytest.mark.parametrize("grad_mode", ["analytic_grad", "smooth_grad"])
    def test_simplified_float32_finite_on_trace(self, grad_mode):
        """float32 forward + obs-gradient stay finite on a fault-trace-crossing
        grid at vertical dip (cd = 0), for both gradient backends."""
        f32 = torch.float32
        L, W = 10e3, 5e3
        g = torch.linspace(-8e3, 8e3, 21, dtype=f32)
        X, Y = torch.meshgrid(g, g, indexing="xy")
        x_obs = X.reshape(1, -1).clone().requires_grad_(True)
        y_obs = Y.reshape(1, -1)
        model = OkadaSourceSimple(internal_dtype=f32, **{grad_mode: True})
        out = model(
            x_obs=x_obs, y_obs=y_obs,
            source_x=torch.zeros(1, dtype=f32), source_y=torch.zeros(1, dtype=f32),
            dip=torch.tensor([math.pi / 2.0], dtype=f32),
            strike=torch.tensor([0.3], dtype=f32),
            centroid_depth=torch.tensor([0.5 * W], dtype=f32),
            length=torch.tensor([L], dtype=f32), width=torch.tensor([W], dtype=f32),
            disl1=torch.tensor([1.0], dtype=f32), disl2=torch.tensor([0.5], dtype=f32),
            disl3=torch.zeros(1, dtype=f32),
        )
        assert torch.isfinite(out.e).all() and torch.isfinite(out.u).all()
        (gx,) = torch.autograd.grad((out.e**2 + out.n**2 + out.u**2).sum(), x_obs)
        assert torch.isfinite(gx).all()

    def test_fullz_float32_finite_on_fault_plane(self):
        """float32 obs-points on the fault plane (analytic backend) stay finite."""
        f32 = torch.float32
        g = torch.linspace(-10e3, 10e3, 21, dtype=f32)
        x_obs = g.reshape(1, -1).clone().requires_grad_(True)
        n = x_obs.shape[1]
        out = OkadaSource(internal_dtype=f32, analytic_grad=True)(
            x_obs=x_obs, y_obs=torch.zeros(1, n, dtype=f32),
            z_obs=torch.full((1, n), -3e3, dtype=f32),
            source_x=torch.zeros(1, dtype=f32), source_y=torch.zeros(1, dtype=f32),
            dip=torch.tensor([math.pi / 2.0], dtype=f32),
            strike=torch.zeros(1, dtype=f32),
            centroid_depth=torch.tensor([5e3], dtype=f32),
            length=torch.tensor([12e3], dtype=f32), width=torch.tensor([6e3], dtype=f32),
            disl1=torch.tensor([1.0], dtype=f32), disl2=torch.tensor([0.5], dtype=f32),
            disl3=torch.zeros(1, dtype=f32),
        )
        assert torch.isfinite(out.e).all() and torch.isfinite(out.u).all()
        (gx,) = torch.autograd.grad((out.e**2 + out.n**2 + out.u**2).sum(), x_obs)
        assert torch.isfinite(gx).all()

    def test_mogi_float32_finite_near_source(self):
        """Mogi stays finite in float32 for a grid passing over a shallow source
        (same dtype-aware num_eps convention as the Okada sources)."""
        from torchdeform import MogiSource
        f32 = torch.float32
        # a grid passing directly over an epicentre at (0, 0)
        g = torch.linspace(-10.0, 10.0, 21, dtype=f32)
        x_obs = g.reshape(1, -1).clone().requires_grad_(True)
        n = x_obs.shape[1]
        out = MogiSource(internal_dtype=f32)(
            x_obs=x_obs, y_obs=torch.zeros(1, n, dtype=f32),
            source_x=torch.zeros(1, dtype=f32), source_y=torch.zeros(1, dtype=f32),
            depth=torch.tensor([1e-3], dtype=f32),   # shallow -> near-singular over epicentre
            delta_v=torch.tensor([1e5], dtype=f32),
        )
        assert torch.isfinite(out.e).all() and torch.isfinite(out.u).all()
        (gx,) = torch.autograd.grad((out.e**2 + out.n**2 + out.u**2).sum(), x_obs)
        assert torch.isfinite(gx).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
