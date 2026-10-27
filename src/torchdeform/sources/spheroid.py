"""
Finite pressurised spheroid source (Yang et al. 1988, as corrected by Cervelli).

A uniformly pressurised spheroidal cavity of arbitrary orientation in an elastic
half-space, after Yang, Davis & Dieterich (1988). The cavity is represented by a
line of centres of dilatation and double forces distributed between its two foci
(``Z = +/- c`` along the polar axis, ``c^2 = a^2 - b^2``), plus the image terms that
free the surface of traction. Unlike the point sources
(:class:`~torchdeform.sources.MogiSource`, :class:`~torchdeform.sources.PECMSource`)
it keeps the finite length of the cavity, which is what matters for an elongated,
dipping reservoir or conduit observed in the near field.

This is a differentiable, batched PyTorch port of the surface-displacement part of
Peter Cervelli's ``spheroid.m`` (USGS, v1.11), which corrects errors in the
original paper. Beyond a straight port it is reformulated for the free surface so
that the forward *and its gradient* stay smooth everywhere a buried cavity can be
observed:

* The two foci contribute ``U(c) - U(-c)``, so any term that is independent of
  ``Z`` cancels exactly. The ``log(R1 + r3bar)`` and ``1/(R1 + r3bar)`` terms,
  singular where the polar axis pierces the surface, are rewritten via
  ``R1 + r3bar = rho^2 / (R1 - r3bar)`` and their ``Z``-independent singular
  parts dropped. (At ``x3 = 0`` the remainder equals the image terms.) This
  replaces ``spheroid.m``'s thresholded limit branches and is more accurate
  than them.
* The ``atan(.../(cos(dip) y2))`` angle is replaced by ``atan(cos(dip) y2 / N)``,
  where ``N > 0`` for any buried cavity. The two differ by a ``Z``-independent
  constant that cancels, and the new form needs no division by ``y2`` on the
  strike line.
* The general expressions carry ``1/cos(dip)^2`` cancellations that lose
  precision as the axis turns vertical (relative error ~``3e-13 / cos(dip)^2``).
  Within ``VERTICAL_BAND`` of vertical the field is the quartic interpolant in
  dip through the exact vertical closed form and four general-branch nodes on
  either side. The same is done within ``SPHERE_BAND`` of a sphere (``a = b``,
  where ``c -> 0`` makes ``U(c) - U(-c)`` a cancelling difference), with the
  Mogi closed form as the centre node. Both keep the error near ``1e-9`` of the
  peak through the singular limits.
* Oblate cavities (``a < b``) follow Cervelli's analytic continuation: ``c`` is
  imaginary and the field is the real part of the complex evaluation. Batches
  with no oblate element stay in real arithmetic.

The model is accurate when the cavity is small compared with its depth (the
half-space correction only approximately preserves uniform pressure on the
cavity wall); it reduces exactly to Mogi for a sphere.

Attribution
-----------
Ported from ``spheroid.m`` by Peter F. Cervelli (U.S. Geological Survey), a work
of the U.S. Government in the public domain (17 U.S.C. 105); see the project
``NOTICE`` file.

References:
Yang, X.-M., Davis, P. M., Dieterich, J. H. (1988), "Deformation from inflation of
a dipping finite prolate spheroid in an elastic half-space as a model for volcanic
stressing", Journal of Geophysical Research, 93(B5), 4249-4257.
Cervelli, P. F. (2013), "Analytical expressions for deformation from an
arbitrarily oriented spheroid in a half-space", Eos Trans. AGU (abstract).
"""
import math

import torch
from torch import Tensor

from .base import SourceModel, DEFAULT_POISSON_RATIO, DEFAULT_SHEAR_MODULUS
from ..core import Displacement

STRENGTH_TYPES = ("pressure", "volume")

# Half-width (rad, ~0.86 deg) of the near-vertical band |dip - pi/2| interpolated
# through the exact vertical form. Balances the general branch's round-off
# (~3e-13 / delta^2, at the band edge and the nodes) against the quartic's
# truncation (~h^5); measured worst case ~3e-9 of the peak vs a 50-digit reference.
VERTICAL_BAND = 1.5e-2
# Half-width of the near-spherical band |a/b - 1| interpolated through Mogi
# (round-off ~3e-12 / e^2 at the nodes vs truncation): ~3e-10 at 0.01.
SPHERE_BAND = 1e-2
_NODES = (-2, -1, 0, 1, 2)


def _quartic_weights(t: Tensor) -> list[Tensor]:
    """Lagrange weights of the quartic through ``_NODES`` evaluated at ``t``."""
    weights = []
    for i in _NODES:
        w = torch.ones_like(t)
        for j in _NODES:
            if j != i:
                w = w * (t - j) / (i - j)
        weights.append(w)
    return weights


def _by_rows(mask: Tensor, f_true, f_false, tensors: tuple[Tensor, ...]):
    """Evaluate ``f_true`` on the batch rows where ``mask`` holds and ``f_false`` on
    the rest, reassembling the ``(u1, u2, u3)`` results in the original order.

    Each branch sees only its own rows, so neither has to tolerate (or be fed safe
    stand-ins for) the inputs it cannot handle.
    """
    parts, order = [], []
    for idx, f in ((mask.nonzero(as_tuple=True)[0], f_true),
                   ((~mask).nonzero(as_tuple=True)[0], f_false)):
        if idx.numel():
            parts.append(f(*(t[idx] for t in tensors)))
            order.append(idx)
    if len(parts) == 1:
        return parts[0]
    inv = torch.argsort(torch.cat(order))
    return tuple(torch.cat([p[k] for p in parts])[inv] for k in range(3))


def _yang_surface(x1, x2, d, a, b, dip, strength, nu, mu, volume, vertical=False):
    """Surface displacement of a Yang spheroid in the axis frame.

    ``x1``/``x2`` ``[B, N]`` are the observation offsets along / across the
    azimuth of the polar axis; ``d, a, b, dip, strength`` are ``[B, 1]``.
    ``vertical=True`` evaluates the exact ``dip = pi/2`` closed form (``dip`` is
    then ignored). Returns ``(u1, u2, u3)``, positive strength inflating.
    """
    if vertical:
        ct = torch.zeros_like(dip)
        st = torch.ones_like(dip)
    else:
        ct, st = torch.cos(dip), torch.sin(dip)
    C0 = 4.0 * (1.0 - nu)
    C1 = C0 * (1.0 - 2.0 * nu)
    C2 = C0 - 1.0
    C3 = C0 + 1.0

    b2 = b * b
    c2 = a * a - b2
    if bool((c2 < 0).any()):   # oblate: imaginary foci, take the real part
        c2 = c2.to(torch.promote_types(c2.dtype, torch.complex64))
    c = torch.sqrt(c2)
    L0 = torch.log((a - c) / (a + c))
    den = (b2 * c * (2.0 * b2 - C0 * a * a) * L0 - a * c * c * (4.0 * b2 + C0 * a * a)
           + a * b2 * b2 * (1.0 + nu) * L0 * L0)
    a1 = b2 * (2.0 * c * (a * a + 2.0 * b2) + 3.0 * a * b2 * L0) / den
    b1 = (a * a * c * C0 + b2 * (c + C3 * (c + a * L0))) / den
    # spheroid.m's sign convention has negative strength inflating; flip it.
    if volume:
        C5 = -3.0 * strength / (8.0 * math.pi * b2 * (
            -2.0 * b1 * c ** 3 - a1 * (a * L0 * (1.0 - 2.0 * nu) + c * C3)))
    else:
        C5 = -strength / (4.0 * mu)

    # free surface: x3 = 0, so x3hat = d and x3bar = X3 = -d
    r2 = x1 * st + d * ct          # = q2 at the surface
    r3 = x1 * ct - d * st          # = -q3 at the surface
    q2, q3 = r2, -r3

    def terms(Z):
        y1 = x1 - Z * ct
        y2 = x2
        y3bar = d + Z * st
        r3bar = r3 - Z
        q3bar = q3 + Z
        R = torch.sqrt(y1 * y1 + y2 * y2 + y3bar * y3bar)   # R1 = R2 at the surface
        R2q3 = R - r3bar                                    # = R2 + q3bar
        R2y3 = R + y3bar
        L2 = torch.log(R2q3)
        L3 = torch.log(R2y3)
        # L1 and A1star with their Z-independent singular parts removed (they
        # cancel between the foci); at the surface they equal -L2 and A1starbar.
        L1 = -L2
        A1starbar = -a1 / (R * R2q3) - b1 * (L2 + (q3 - Z) / R2q3)
        A1star = A1starbar
        A1 = Z / R + L1
        A1bar = Z / R - L2
        A2 = R - r3 * L1
        A2bar = R - q3 * L2
        A3 = Z * r3bar / R + R
        A3bar = Z * q3bar / R - R

        if vertical:
            # spheroid.m's ct == 0 branch, its Z-independent 1/rho^2 part removed
            g = L2 / 2.0 + (d - Z) / (2.0 * R2y3)
            f1 = C1 * y1 * g
            f2 = C1 * y2 * g
            f3 = C1 * (R - d * L2)
        else:
            phi = torch.atan(ct * y2 / (ct * q2 + (1.0 + st) * R2q3))
            f1 = C1 * (Z * y1 / R2y3 - 6.0 * st * y2 * phi / ct ** 2
                       + 3.0 / ct ** 2 * (q2 * (st * L2 - L3) + ct * (R - y3bar))
                       - 2.0 * ct * A2bar + 2.0 / ct * (d * L3 - q3 * L2))
            f2 = C1 * (Z * y2 / R2y3 - 6.0 * q2 * phi / ct ** 2
                       - 3.0 * y2 / ct ** 2 * (L2 - st * L3) + 2.0 * y2 * L2
                       + 4.0 * d / ct * phi)
            f3 = C1 * (-2.0 * y2 * phi / ct + 2.0 * st * A2bar + q3 * L3 - Z
                       + q2 / ct * (L2 - st * L3))

        # at x3 = 0 the F1star, F2star and B4 terms vanish identically
        F1 = -C2 * A1bar
        F2 = -C2 * A3bar
        Bstar = 2.0 * A2 * b1 + a1 / R + C2 * (2.0 * A2bar * b1 + a1 / R)
        B1 = -2.0 * C2 * (-d * (Z / R - L2) + st * (A2bar - Z * Z / R))
        B2 = A3 + C0 * (A2 + A2bar)
        B3 = A1star * r2 + C2 * q2 * A1starbar

        U1 = b2 * (st * B3 + ct * Bstar) + a1 * (st * (q2 * F1 - A1 * r2)
                                                - ct * (F2 + B2) + f1)
        U2 = b2 * y2 * (A1star + C2 * A1starbar) - a1 * (y2 * (A1 - F1) - f2)
        U3 = b2 * (ct * B3 - st * Bstar) + a1 * (-ct * (q2 * F1 + A1 * r2)
                                                - st * (F2 - B2) - B1 + f3)
        return U1, U2, U3

    up, um = terms(c), terms(-c)
    u = tuple(C5 * (p - m) for p, m in zip(up, um))
    if u[0].is_complex():
        u = tuple(t.real for t in u)
    return u


def _mogi_surface(x1, x2, d, b, strength, nu, mu, volume):
    """Exact sphere (``a = b``): the Mogi field in the axis frame."""
    k = ((1.0 - nu) * strength / math.pi) if volume else ((1.0 - nu) * b ** 3 * strength / mu)
    s3 = (x1 * x1 + x2 * x2 + d * d) ** 1.5
    return k * x1 / s3, k * x2 / s3, k * d / s3


def _yang_any_dip(x1, x2, d, a, b, dip, strength, nu, mu, volume):
    """:func:`_yang_surface` for any dip in ``[0, pi)``, interpolating the
    near-vertical band through the exact vertical form."""
    h = VERTICAL_BAND

    def band(x1, x2, d, a, b, dip, strength):
        out = None
        for k, w in zip(_NODES, _quartic_weights((dip - math.pi / 2.0) / h)):
            node = torch.full_like(dip, math.pi / 2.0 + k * h)
            uk = _yang_surface(x1, x2, d, a, b, node, strength, nu, mu, volume,
                               vertical=(k == 0))
            out = (tuple(w * u for u in uk) if out is None
                   else tuple(o + w * u for o, u in zip(out, uk)))
        return out

    def general(x1, x2, d, a, b, dip, strength):
        return _yang_surface(x1, x2, d, a, b, dip, strength, nu, mu, volume)

    near_vertical = (dip[:, 0] - math.pi / 2.0).abs() < h
    return _by_rows(near_vertical, band, general, (x1, x2, d, a, b, dip, strength))


def _spheroid_surface(x1, x2, d, a, b, dip, strength, nu, mu, volume):
    """Spheroid field for any shape, interpolating the near-spherical band
    through the Mogi form (nodes vary ``a`` at fixed ``b``)."""
    h = SPHERE_BAND

    def band(x1, x2, d, a, b, dip, strength):
        out = None
        for k, w in zip(_NODES, _quartic_weights((a / b - 1.0) / h)):
            if k == 0:
                uk = _mogi_surface(x1, x2, d, b, strength, nu, mu, volume)
            else:
                uk = _yang_any_dip(x1, x2, d, b * (1.0 + k * h), b, dip, strength,
                                   nu, mu, volume)
            out = (tuple(w * u for u in uk) if out is None
                   else tuple(o + w * u for o, u in zip(out, uk)))
        return out

    def spheroid(x1, x2, d, a, b, dip, strength):
        return _yang_any_dip(x1, x2, d, a, b, dip, strength, nu, mu, volume)

    near_sphere = (a[:, 0] / b[:, 0] - 1.0).abs() < h
    return _by_rows(near_sphere, band, spheroid, (x1, x2, d, a, b, dip, strength))


class SpheroidSource(SourceModel):
    """
    Finite pressurised spheroid (Yang et al. 1988) of arbitrary orientation.

    A spheroidal cavity with polar semi-axis ``a`` (along its symmetry axis) and
    equatorial semi-axis ``b``: prolate for ``a > b`` (an elongated reservoir or
    conduit), oblate for ``a < b`` (a flattened, sill-like body), and exactly a
    Mogi source for ``a = b``. The polar axis plunges at ``dip`` below horizontal
    toward the azimuth ``strike``.

    Conventions
    -----------
    - All distances in metres; depth (of the centre) positive downward. The
      cavity must lie wholly below the free surface.
    - ``dip`` and ``strike`` in radians. ``strike`` is clockwise from north and
      is the azimuth toward which the polar axis plunges; ``dip = 0`` is a
      horizontal axis, ``dip = pi/2`` a vertical one. ``dip`` is taken modulo
      ``pi`` (``(dip, strike)`` and ``(pi - dip, strike + pi)`` are the same
      cavity).
    - ``strength`` is the cavity-wall pressure change (Pa) for
      ``strength_type="pressure"`` (default) or the cavity volume change (m^3)
      for ``strength_type="volume"``; positive inflates and uplifts. In volume
      mode the field is independent of ``shear_modulus`` and a sphere equals
      :class:`~torchdeform.sources.MogiSource` with the same ``delta_v``.
    - Returns ENU surface displacement in metres, linear in ``strength``.

    Accuracy is ~``1e-10`` of the peak against ``spheroid.m`` in ``float64``
    (better than ``spheroid.m`` itself against a 50-digit evaluation), and
    ~``1e-9`` through the near-vertical / near-spherical interpolation bands.
    The model itself is the small-cavity approximation of Yang et al., best
    when the cavity is small compared with its depth. Use ``float64``: the
    cancellations between the two foci leave ``float32`` with few correct
    digits.
    """

    def __init__(
        self,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        shear_modulus: float = DEFAULT_SHEAR_MODULUS,
        strength_type: str = "pressure",
        internal_dtype: torch.dtype = torch.float64,
    ):
        """
        Parameters
        ----------
        poisson_ratio : float, default 0.25
            Poisson's ratio of the elastic half-space.
        shear_modulus : float, default 3e10
            Shear modulus ``mu`` (Pa). Sets the displacement scale in pressure
            mode; unused in volume mode.
        strength_type : {"pressure", "volume"}, default "pressure"
            Whether ``forward``'s ``strength`` is a pressure change (Pa) or a
            cavity volume change (m^3).
        internal_dtype : torch.dtype, default torch.float64
            Dtype used for the internal computation; inputs are cast to it.
        """
        super().__init__()
        if strength_type not in STRENGTH_TYPES:
            raise ValueError(
                f"unknown strength_type {strength_type!r}; expected one of {STRENGTH_TYPES}"
            )
        self.v = poisson_ratio
        self.mu = shear_modulus
        self.strength_type = strength_type
        self.internal_dtype = internal_dtype

    def forward(
        self,
        x_obs: Tensor,              # [B, N] east coordinates (m)
        y_obs: Tensor,              # [B, N] north coordinates (m)
        source_x: Tensor,           # [B] centre east (m)
        source_y: Tensor,           # [B] centre north (m)
        depth: Tensor,              # [B] centre depth, positive down (m)
        polar_radius: Tensor,       # [B] semi-axis a along the symmetry axis (m)
        equatorial_radius: Tensor,  # [B] semi-axis b perpendicular to it (m)
        dip: Tensor,                # [B] plunge of the polar axis (rad)
        strike: Tensor,             # [B] azimuth of the plunge, clockwise from N (rad)
        strength: Tensor,           # [B] pressure (Pa) or volume (m^3) change
    ) -> Displacement:
        """Surface displacement from a pressurised spheroidal cavity.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the cavity centre [B] in metres.
        depth : Tensor
            Depth of the centre [B] in metres (positive down).
        polar_radius, equatorial_radius : Tensor
            Semi-axes ``a`` (along the symmetry axis) and ``b`` [B] in metres.
        dip : Tensor
            Plunge of the polar axis below horizontal [B] in radians.
        strike : Tensor
            Azimuth (clockwise from north) toward which the axis plunges [B],
            radians.
        strength : Tensor
            Pressure change (Pa) or volume change (m^3) [B], per
            ``strength_type``; positive inflates.

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.

        Raises
        ------
        ValueError
            If ``depth`` or a semi-axis is not strictly positive, or the cavity
            reaches the free surface.
        """
        params = {"source_x": source_x, "source_y": source_y, "depth": depth,
                  "polar_radius": polar_radius, "equatorial_radius": equatorial_radius,
                  "dip": dip, "strike": strike, "strength": strength}
        batch = self._validate_inputs(x_obs, y_obs, params)
        for name in ("depth", "polar_radius", "equatorial_radius"):
            if torch.any(params[name] <= 0):
                raise ValueError(f"{name} must be strictly positive")

        dtype = self.internal_dtype
        x_obs = x_obs.to(dtype)
        y_obs = y_obs.to(dtype)
        p = {k: v.to(dtype).expand(batch)[:, None] for k, v in params.items()}
        dip = torch.remainder(p["dip"], math.pi)
        a, b, d = p["polar_radius"], p["equatorial_radius"], p["depth"]

        # vertical half-extent of the cavity must stay below the surface
        half_height = torch.sqrt((a * torch.sin(dip)) ** 2 + (b * torch.cos(dip)) ** 2)
        if torch.any(half_height >= d):
            raise ValueError(
                "spheroid extends above the free surface: its vertical half-extent "
                "must be smaller than depth. Increase depth or shrink the semi-axes."
            )

        ss, cs = torch.sin(p["strike"]), torch.cos(p["strike"])
        dx = x_obs - p["source_x"]
        dy = y_obs - p["source_y"]
        x1 = ss * dx + cs * dy       # along the axis azimuth
        x2 = -cs * dx + ss * dy      # across it

        u1, u2, u3 = _spheroid_surface(
            x1, x2, d, a, b, dip, p["strength"], self.v, self.mu,
            self.strength_type == "volume")

        return Displacement(e=ss * u1 - cs * u2, n=cs * u1 + ss * u2, u=u3)
