"""
Fialko, Khazan & Simons (2001) pressurized horizontal circular ("penny-shaped")
crack in an elastic half-space.

This is a clean-room implementation derived **solely** from the published paper:

    Y. Fialko, Y. Khazan & M. Simons (2001), "Deformation due to a pressurized
    horizontal circular crack in an elastic half-space, with applications to
    volcano geodesy", Geophys. J. Int. 146, 181-190.

A horizontal circular crack of radius ``R`` is buried at depth ``H`` in an
elastic half-space and loaded by a uniform (hydrostatic) excess pressure
``dP``. The free surface displacement has no elementary closed form: it follows
from a pair of coupled Fredholm integral equations of the second kind for two
auxiliary "image" functions phi(t), psi(t) (the paper's eq 27 / Appendix A),
after which the surface displacements are recovered from a second integral with
closed-form kernels (eq 30 / Appendix B).

Pipeline (all dimensionless lengths normalised by the crack radius ``R`` and all
stresses by the shear modulus ``mu``, paper section 3):

1. ``build_quadrature`` -- composite 16-point Gauss-Legendre grid on [0, 1].
2. ``_fredholm_kernels`` -- the smooth bounded kernels ``T1..T4`` (eqs A2-A5),
   built from ``R1..R3`` (eq 25) and ``P1..P4`` (Appendix A).
3. ``_solve_fredholm`` -- discretise the coupled system (eq A1) as a *single*
   dense linear system ``M x = b`` and solve once with ``torch.linalg.solve``
   (no successive-approximation iteration), so gradients flow through the solve
   and the whole batch is solved together.
4. ``_surface_kernels`` -- the closed-form ``S/C`` improper-integral kernels
   (eqs B5-B13).
5. ``_surface_displacement`` -- assemble eq 30 (from eqs B1-B2) to get the
   dimensionless vertical / radial surface displacements, then scale by
   ``Pf = 2(1-nu) R dP / mu`` (the dimensional factor of eqs B1-B2, C12) to get
   metres.

All steps are batched over ``B`` and differentiable in the source parameters.

Conventions (matching the rest of torchdeform):
- ``z`` positive downward, crack centred at ``depth`` with ``radius``.
- Output vertical ``u`` is positive **upward** (uplift) for positive pressure;
  the paper's ``Uz`` points downward (its ``z`` axis), so we negate it.
- The horizontal surface field is purely radial; we decompose it into E/N along
  the unit vector from the crack centre to each observation point.
"""
import torch
from torch import Tensor

from .base import SourceModel
from ..core import Displacement


# --------------------------------------------------------------------------- #
# 16-point Gauss-Legendre nodes / weights on [-1, 1] (Abramowitz & Stegun).
# Used because Appendix A states the numerical integration is performed with
# 16-point Gaussian quadrature on each subinterval.
# --------------------------------------------------------------------------- #
_GL16_NODES = (
    0.0950125098376374401853193354250,
    0.2816035507792589132304605014605,
    0.4580167776572273863424194429513,
    0.6178762444026437484466717640413,
    0.7554044083550030338951011948474,
    0.8656312023878317438804678977124,
    0.9445750230732325760779884155343,
    0.9894009349916499325961541734504,
)
_GL16_WEIGHTS = (
    0.1894506104550684962853967232083,
    0.1826034150449235888667636679692,
    0.1691565193950025381893120790304,
    0.1495959888165767320815017305474,
    0.1246289712555338720524762821920,
    0.0951585116824927848099251076022,
    0.0622535239386478928628438369944,
    0.0271524594117540948517805724560,
)

# Full symmetric 16-node / 16-weight vectors (negative half mirrors the positive
# half; nodes ordered ascending to match numpy.polynomial.legendre.leggauss).
ROOT16 = torch.tensor(
    [-n for n in reversed(_GL16_NODES)] + list(_GL16_NODES),
    dtype=torch.float64,
)
WEIGHT16 = torch.tensor(
    list(reversed(_GL16_WEIGHTS)) + list(_GL16_WEIGHTS),
    dtype=torch.float64,
)


def build_quadrature(
    nis: int,
    root: Tensor,
    weight: Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Composite Gauss-Legendre quadrature on [0, 1].

    Subdivide [0, 1] into ``nis`` equal panels and map the reference nodes /
    weights (``root``/``weight`` on [-1, 1]) into each panel (Appendix A:
    "Numerical integration is performed by subdividing the interval of
    integration into ... equal intervals and using the 16-point Gaussian
    quadrature on each interval").

    Parameters
    ----------
    nis : int
        Number of equal panels on [0, 1].
    root, weight : Tensor
        Gauss-Legendre nodes and weights on [-1, 1] (e.g. ``ROOT16``/``WEIGHT16``).

    Returns
    -------
    (t, Wt) : tuple of Tensor
        Quadrature nodes ``t`` in (0, 1) and weights ``Wt`` scaled so that
        ``Wt.sum() == 1`` (the length of [0, 1]); each of length ``nis * len(root)``.
    """
    root = root.to(device=device, dtype=dtype)
    weight = weight.to(device=device, dtype=dtype)

    panel = 1.0 / nis                                  # panel width
    # left edges of the nis panels: 0, panel, 2*panel, ...
    edges = torch.arange(nis, device=device, dtype=dtype) * panel  # [nis]
    centres = edges + 0.5 * panel                                  # [nis]
    half = 0.5 * panel

    # map reference nodes/weights into every panel -> [nis, len(root)]
    t = centres[:, None] + half * root[None, :]
    Wt = half * weight[None, :].expand(nis, -1)

    return t.reshape(-1), Wt.reshape(-1)


class PennySource(SourceModel):
    """Fialko et al. (2001) pressurized penny-shaped (horizontal circular) crack.

    Surface displacement of an elastic half-space due to a horizontal circular
    crack of ``radius`` at ``depth`` loaded by uniform excess ``pressure``.

    Conventions
    -----------
    - All distances in metres, pressure in Pa.
    - Depth positive downward; crack centred at ``(source_x, source_y, depth)``.
    - Returns ENU :class:`~torchdeform.core.Displacement` in metres, vertical
      positive upward (uplift) for positive pressure.
    """

    def __init__(
        self,
        poisson_ratio: float = 0.25,
        shear_modulus: float = 3e10,
        internal_dtype: torch.dtype = torch.float64,
        nis: int = 2,
        num_eps: float | None = None,
    ):
        super().__init__()
        self.v = poisson_ratio
        self.mu = shear_modulus
        self.internal_dtype = internal_dtype
        self.nis = nis
        # None -> dtype-appropriate floor resolved per call (see _resolve_num_eps);
        # 1e-12 underflows float32, so the default must track internal_dtype.
        self.num_eps = num_eps

    # ------------------------------------------------------------------ #
    # Appendix A kernels T1..T4  (eqs A2-A5)
    # ------------------------------------------------------------------ #
    def _fredholm_kernels(self, t: Tensor, r: Tensor, h: Tensor) -> tuple[Tensor, ...]:
        """Closed-form Fredholm kernels ``T1, T2, T3, T4`` (eqs A2-A5).

        Parameters
        ----------
        t, r : Tensor
            Quadrature node vectors (length ``M``), forming the [M, M] grids
            ``t`` (rows, the equation index) and ``r`` (cols, the integration
            variable).
        h : Tensor
            Per-batch dimensionless depth ``h = depth / radius``, shape [B, 1, 1].

        Returns
        -------
        (T1, T2, T3, T4) : tuple of Tensor
            Each shaped [B, M, M].
        """
        eps = self._resolve_num_eps()
        # outer grids: t varies down rows, r across cols  -> [M, M]
        tt = t[:, None]
        rr = r[None, :]
        # broadcast against batch dimension via h [B,1,1]
        h2 = h * h
        h3 = h2 * h
        h4 = h2 * h2

        # P1..P4 (Appendix A), functions of x = (t -/+ r):
        #   P1(x) = (12 h^2 - x^2) / (4 h^2 + x^2)^3
        #   P2(x) = ln(4 h^2 + x^2) + (8 h^4 + 2 x^2 h^2 - x^4) / (4 h^2 + x^2)^2
        #   P3(x) = 2 (8 h^4 - 2 x^2 h^2 + x^4) / (4 h^2 + x^2)^3
        #   P4(x) = (4 h^2 - x^2) / (4 h^2 + x^2)^2
        def _P(x):
            d = 4.0 * h2 + x * x                       # 4h^2 + x^2 (>0)
            d2 = d * d
            d3 = d2 * d
            x2 = x * x
            x4 = x2 * x2
            P1 = (12.0 * h2 - x2) / d3
            P2 = torch.log(d) + (8.0 * h4 + 2.0 * x2 * h2 - x4) / d2
            P3 = 2.0 * (8.0 * h4 - 2.0 * x2 * h2 + x4) / d3
            P4 = (4.0 * h2 - x2) / d2
            return P1, P2, P3, P4

        xm = tt - rr      # (t - r)
        xp = tt + rr      # (t + r)

        P1m, P2m, P3m, P4m = _P(xm)
        P1p, P2p, P3p, P4p = _P(xp)

        # eq (A2): T1(t,r) = 4 h^3 [ P1(t-r) - P1(t+r) ]
        T1 = 4.0 * h3 * (P1m - P1p)

        # eq (A3): T2(t,r) = (h/(t r)) [ P2(t-r) - P2(t+r) ] + h [ P3(t-r) + P3(t+r) ]
        tr = (tt * rr) + eps
        T2 = (h / tr) * (P2m - P2p) + h * (P3m + P3p)

        # eq (A4): T3(t,r) = (h^2 / r) [ P4(t-r) - P4(t+r)
        #                                - 2 r ( (t-r) P1(t-r) + (t+r) P1(t+r) ) ]
        rsafe = rr + eps
        T3 = (h2 / rsafe) * (
            P4m - P4p - 2.0 * rsafe * (xm * P1m + xp * P1p)
        )

        # eq (A5): T4(t,r) = T3(r,t)  -> swap the roles of t and r.
        # Rebuild T3 with t<->r: r/t arguments swapped.
        xm_s = rr - tt    # (r - t)  (note P depends on x^2 so sign is irrelevant,
        xp_s = rr + tt    # (r + t)   but kept explicit for clarity)
        P1ms, _, _, P4ms = _P(xm_s)
        P1ps, _, _, P4ps = _P(xp_s)
        tsafe = tt + eps
        T4 = (h2 / tsafe) * (
            P4ms - P4ps - 2.0 * tsafe * (xm_s * P1ms + xp_s * P1ps)
        )

        return T1, T2, T3, T4

    # ------------------------------------------------------------------ #
    # Fredholm solve (eq A1) as a single dense linear system.
    # ------------------------------------------------------------------ #
    def _solve_fredholm(
        self, t: Tensor, Wt: Tensor, h: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Solve the coupled Fredholm system (eq A1) on the quadrature grid.

        For a uniformly (hydrostatic) pressurized crack, eq (A1) reads (with
        normalised functions ``phibar = phi/p0``, ``psibar = psi/p0``):

            phibar(t) = -2 t / pi + (2/pi) integral[ T1 phibar + T3 psibar ] dr
            psibar(t) =             (2/pi) integral[ T2 psibar + T4 phibar ] dr

        Discretising the integrals with the quadrature weights ``Wt`` and
        collocating at the nodes ``t`` turns this into a 2M x 2M linear system

            (I - K) [phibar; psibar] = b,

        which we solve once per batch with ``torch.linalg.solve`` (no iteration),
        keeping the whole thing autograd-differentiable.

        Returns
        -------
        (phibar, psibar) : tuple of Tensor
            Node values of the auxiliary functions, each [B, M].
        """
        dtype = t.dtype
        device = t.device
        B = h.shape[0]
        M = t.shape[0]
        c = 2.0 / torch.pi

        # kernels on the [M, M] grid, batched [B, M, M]
        T1, T2, T3, T4 = self._fredholm_kernels(t, r=t, h=h)

        # Quadrature: integral f(r) dr ~ sum_j Wt_j f(r_j).  Fold the weights
        # into the kernel columns so each block is (c * T * Wt).
        Wcol = (c * Wt)[None, None, :]            # [1, 1, M]
        K11 = T1 * Wcol                           # phibar <- phibar (eq A1, T1)
        K12 = T3 * Wcol                           # phibar <- psibar (T3)
        K22 = T2 * Wcol                           # psibar <- psibar (T2)
        K21 = T4 * Wcol                           # psibar <- phibar (T4)

        eye = torch.eye(M, dtype=dtype, device=device).expand(B, M, M)
        # Assemble the 2M x 2M operator  A = I - K  with block layout
        #   [ I - K11 ,   -K12 ]   [phibar]   [ -2 t / pi ]
        #   [   -K21  , I - K22 ]   [psibar] = [    0      ]
        top = torch.cat([eye - K11, -K12], dim=2)        # [B, M, 2M]
        bot = torch.cat([-K21, eye - K22], dim=2)        # [B, M, 2M]
        A = torch.cat([top, bot], dim=1)                 # [B, 2M, 2M]

        # rhs: phibar forcing -2 t / pi (eq A1), psibar forcing 0.
        b_phi = (-c * t).expand(B, M)                    # [B, M]
        b_psi = torch.zeros(B, M, dtype=dtype, device=device)
        b = torch.cat([b_phi, b_psi], dim=1)[..., None]  # [B, 2M, 1]

        sol = torch.linalg.solve(A, b)[..., 0]           # [B, 2M]
        phibar = sol[:, :M]
        psibar = sol[:, M:]
        return phibar, psibar

    # ------------------------------------------------------------------ #
    # Appendix B surface kernels (eqs B5-B13)
    # ------------------------------------------------------------------ #
    def _surface_kernels(self, r: Tensor, t: Tensor, h: Tensor) -> tuple[Tensor, ...]:
        """Closed-form surface-displacement kernels (eqs B5-B13).

        Parameters
        ----------
        r : Tensor
            Dimensionless observation radius, shape [B, N, 1].
        t : Tensor
            Quadrature nodes, shape [1, 1, M].
        h : Tensor
            Dimensionless depth, shape [B, 1, 1].

        Returns
        -------
        (S0_0, S0_1, C0_1, S1_m1, S1_0, C1_0, C1_1, S1_1) : tuple of Tensor
            The eight ``S/C`` kernels of eqs B5-B12, each broadcast to [B, N, M].
        """
        eps = self._resolve_num_eps()
        # eq (B13): X1 = h^2 + r^2 - t^2 ,  X2 = sqrt(X1^2 + 4 h^2 t^2)
        h2 = h * h
        X1 = h2 + r * r - t * t
        X2 = torch.sqrt(X1 * X1 + 4.0 * h2 * t * t + eps)

        # Common radicals sqrt((X2 +/- X1)/2)  (so that, e.g., these are the real
        # and imaginary parts arising from the improper integrals B3-B4).
        sp = torch.sqrt(torch.clamp(0.5 * (X2 + X1), min=0.0) + eps)  # ~ sqrt((X2+X1)/2)
        sm = torch.sqrt(torch.clamp(0.5 * (X2 - X1), min=0.0) + eps)  # ~ sqrt((X2-X1)/2)
        # In eqs B5-B12 these appear as sqrt(X2+X1)/sqrt(2) and sqrt(X2-X1)/sqrt(2).
        Sp = sp        # = sqrt(X2 + X1) / sqrt(2)
        Sm = sm        # = sqrt(X2 - X1) / sqrt(2)

        rsafe = r + eps
        X2_3 = X2 * X2 * X2

        # eq (B5):  S0^0 = sqrt(2) h t / (X2 sqrt(X2 + X1))
        #         = h t / (X2 * Sp)            since sqrt(X2+X1) = sqrt(2) Sp
        S0_0 = h * t / (X2 * Sp)

        # eq (B6):  S0^1 = [ h sqrt(X2 - X1)(2 X1 + X2) - t sqrt(X2 + X1)(2 X1 - X2) ]
        #                   / (sqrt(2) X2^3)
        #         = [ h Sm (2 X1 + X2) - t Sp (2 X1 - X2) ] / X2^3
        S0_1 = (h * Sm * (2.0 * X1 + X2) - t * Sp * (2.0 * X1 - X2)) / X2_3

        # eq (B7):  C0^1 = [ h sqrt(X2 + X1)(2 X1 - X2) + t sqrt(X2 - X1)(2 X1 + X2) ]
        #                   / (sqrt(2) X2^3)
        #         = [ h Sp (2 X1 - X2) + t Sm (2 X1 + X2) ] / X2^3
        C0_1 = (h * Sp * (2.0 * X1 - X2) + t * Sm * (2.0 * X1 + X2)) / X2_3

        # eq (B8):  S1^-1 = t / r - sqrt(X2 - X1) / (sqrt(2) r)
        #         = t / r - Sm / r
        S1_m1 = t / rsafe - Sm / rsafe

        # eq (B9):  S1^0 = [ t sqrt(X2 + X1) - h sqrt(X2 - X1) ] / (sqrt(2) r X2)
        #         = ( t Sp - h Sm ) / (r X2)
        S1_0 = (t * Sp - h * Sm) / (rsafe * X2)

        # eq (B10): C1^0 = 1/r - [ h sqrt(X2 + X1) + t sqrt(X2 - X1) ] / (sqrt(2) r X2)
        #         = 1/r - ( h Sp + t Sm ) / (r X2)
        C1_0 = 1.0 / rsafe - (h * Sp + t * Sm) / (rsafe * X2)

        # eq (B11): C1^1 = r sqrt(X2 + X1)(2 X1 - X2) / (sqrt(2) X2^3)
        #         = r Sp (2 X1 - X2) / X2^3
        C1_1 = r * Sp * (2.0 * X1 - X2) / X2_3

        # eq (B12): S1^1 = r sqrt(X2 - X1)(2 X1 + X2) / (sqrt(2) X2^3)
        #         = r Sm (2 X1 + X2) / X2^3
        S1_1 = r * Sm * (2.0 * X1 + X2) / X2_3

        return S0_0, S0_1, C0_1, S1_m1, S1_0, C1_0, C1_1, S1_1

    # ------------------------------------------------------------------ #
    # Surface displacement (eq 30 via Appendix B, eqs B1-B2)
    # ------------------------------------------------------------------ #
    def _surface_displacement(
        self,
        r: Tensor,            # [B, N] dimensionless observation radius
        t: Tensor,            # [M] quadrature nodes
        Wt: Tensor,           # [M] quadrature weights
        phibar: Tensor,       # [B, M]
        psibar: Tensor,       # [B, M]
        h: Tensor,            # [B] dimensionless depth
    ) -> tuple[Tensor, Tensor]:
        """Dimensionless vertical / radial surface displacement (eqs B1-B2).

        From eqs (28)-(29) with the auxiliary representations (24), (26) and the
        closed-form ``S/C`` improper integrals (B3-B12), the surface
        displacements (in units of ``2(1-nu) p0`` with ``p0`` the dimensionless
        pressure) are

            Uz / [2(1-nu) p0] = integral[ (S0^0 + h S0^1) phibar ] dt        # B1
                              + integral[ (S0^0 / t - C0^1) psibar ] dt
            Ur / [2(1-nu) p0] = integral[ (S1^-1/t - C1^0
                                           - (h/t) S1^0 + h C1^1) psibar ] dt # B2
                              - h integral[ S1^1 phibar ] dt

        (The integrals are over t in [0, 1]; we evaluate them with the composite
        Gauss-Legendre weights ``Wt``.)

        Returns
        -------
        (uz, ur) : tuple of Tensor
            Dimensionless displacements, each [B, N], with ``uz`` in the paper's
            downward-positive convention and ``ur`` radially outward positive.
            (Multiply by ``Pf`` and negate ``uz`` for upward-positive metres.)
        """
        eps = self._resolve_num_eps()

        # reshape for [B, N, M] broadcasting
        r_ = r[:, :, None]                       # [B, N, 1]
        t_ = t[None, None, :]                    # [1, 1, M]
        h_ = h[:, None, None]                    # [B, 1, 1]
        phib = phibar[:, None, :]                # [B, 1, M]
        psib = psibar[:, None, :]                # [B, 1, M]
        W = Wt[None, None, :]                    # [1, 1, M]

        (S0_0, S0_1, C0_1, S1_m1,
         S1_0, C1_0, C1_1, S1_1) = self._surface_kernels(r_, t_, h_)

        tsafe = t_ + eps

        # eq (B1): integrand for Uz
        #   [ S0^0 + h S0^1 ] phibar + [ S0^0 / t - C0^1 ] psibar
        integ_z = (S0_0 + h_ * S0_1) * phib + (S0_0 / tsafe - C0_1) * psib

        # eq (B2): integrand for Ur, obtained by substituting eqs (24), (26) into
        # eq (29).  With Psi(xi) = int (sin(xi t)/(xi t) - cos(xi t)) psi(t) dt and
        # Phi(xi) = int sin(xi t) phi(t) dt, eq (29) is
        #     Ur^s/[2(1-nu) p0] = int [ (1 - xi h) Psi(xi) - xi h Phi(xi) ]
        #                              e^{-xi h} J1(xi r) dxi ,
        # and expanding via the S/C improper integrals (B3-B12) gives
        #   psi(t) coefficient:  S1^-1/t - C1^0 - (h/t) S1^0 + h C1^1
        #   phi(t) coefficient:  -h S1^1
        integ_r = (
            (S1_m1 / tsafe - C1_0 - (h_ / tsafe) * S1_0 + h_ * C1_1) * psib
            - h_ * S1_1 * phib
        )

        uz = (integ_z * W).sum(dim=-1)           # [B, N]
        ur = (integ_r * W).sum(dim=-1)           # [B, N]
        return uz, ur

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_obs: Tensor,      # [B, N] east observation coordinates (m)
        y_obs: Tensor,      # [B, N] north observation coordinates (m)
        source_x: Tensor,   # [B] crack centre east (m)
        source_y: Tensor,   # [B] crack centre north (m)
        depth: Tensor,      # [B] crack depth, positive downward (m)
        radius: Tensor,     # [B] crack radius (m)
        pressure: Tensor,   # [B] uniform excess pressure (Pa)
    ) -> Displacement:
        """Surface displacement from a Fialko penny-shaped crack.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            Crack centre [B] in metres.
        depth : Tensor
            Crack depth [B], positive downward, metres (must be > 0; the crack
            must be buried in the half-space).
        radius : Tensor
            Crack radius [B] in metres (must be > 0).
        pressure : Tensor
            Uniform excess pressure [B] in Pa (positive = inflation).

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres (``u`` positive upward).
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y,
             "depth": depth, "radius": radius, "pressure": pressure},
        )

        dtype = self.internal_dtype
        device = x_obs.device

        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        source_x = source_x.to(dtype)
        source_y = source_y.to(dtype)
        depth = depth.to(dtype)
        radius = radius.to(dtype)
        pressure = pressure.to(dtype)

        if bool((radius <= 0).any()):
            raise ValueError("radius must be strictly positive")
        if bool((depth <= 0).any()):
            # h = depth / radius = 0 makes the P2 kernel log(0) -> -inf and the
            # solution collapses to NaN; the crack must be buried (half-space).
            raise ValueError("depth must be strictly positive")

        # --- dimensionless geometry (paper section 3) ---------------------- #
        # normalise lengths by the crack radius R.
        R = radius                                       # [B]
        h = depth / R                                    # [B] dimensionless depth

        dx = x_obs - source_x[:, None]                   # [B, N] east offset
        dy = y_obs - source_y[:, None]                   # [B, N] north offset
        rho = torch.sqrt(dx * dx + dy * dy + self._resolve_num_eps())  # metric radial distance
        r = rho / R[:, None]                             # [B, N] dimensionless radius

        # --- quadrature grid ------------------------------------------------ #
        t, Wt = build_quadrature(self.nis, ROOT16, WEIGHT16,
                                 device=device, dtype=dtype)

        # --- solve the Fredholm system (eq A1) ----------------------------- #
        h3 = h[:, None, None]                             # [B, 1, 1]
        phibar, psibar = self._solve_fredholm(t, Wt, h3)

        # --- assemble surface displacements (eq 30 / Appendix B) ----------- #
        uz_dimless, ur_dimless = self._surface_displacement(
            r, t, Wt, phibar, psibar, h
        )

        # --- dimensional scaling (paper section 3 / eqs B1-B2, C12) -------- #
        # The dimensionless displacements are in units of 2(1-nu) p0, with p0 the
        # dimensionless excess pressure dP / mu and lengths in units of R. The
        # dimensional displacement is therefore multiplied by Pf = 2(1-nu) R dP / mu.
        Pf = 2.0 * (1.0 - self.v) * R * pressure / self.mu      # [B]
        Pf = Pf[:, None]                                        # [B, 1]

        # paper's Uz is positive downward; flip sign so u is positive upward.
        uu = -uz_dimless * Pf                                   # [B, N] up (m)
        ur = ur_dimless * Pf                                    # [B, N] radial (m)

        # decompose radial horizontal displacement into E/N along (dx, dy).
        inv_rho = 1.0 / rho                                     # rho already eps-guarded
        ue = ur * dx * inv_rho
        un = ur * dy * inv_rho

        return Displacement(e=ue, n=un, u=uu)
