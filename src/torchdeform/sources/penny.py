"""
Penny-shaped (circular) crack source, differentiable.

Surface displacement from a pressurised horizontal penny-shaped crack in an
elastic half-space (Fialko et al., 2001) -- a sill-like analogue to the Mogi
point source that captures the flatter, broader signal of a shallow horizontal
intrusion. There is no closed form: the solution reduces to a Fredholm integral
equation, which is discretised with Gauss-Legendre quadrature, solved as a dense
linear system, and integrated over the crack. Every step is implemented with
batched tensor ops and a linear solve, so the whole model is differentiable in
the source parameters.

Pipeline (all batched over ``B`` source instances)
-------------------------------------------------
1. ``build_quadrature`` -- Gauss-Legendre nodes/weights on ``[0, 1]`` (16-point
   rule, ``ROOT16``/``WEIGHT16``, subdivided into ``nis`` panels).
2. ``fpkernel_vec`` / ``kg`` / ``kern`` -- the integral-equation kernels.
3. ``fredholm_solve_differentiable`` -- solves for the auxiliary functions
   ``fi``, ``psi`` via :func:`torch.linalg.solve`.
4. ``q_all`` + ``intgr_batched`` -- integrate to get dimensionless vertical and
   radial displacement, which :class:`PennySource` rescales to metres.
"""
import math
import torch
from torch import Tensor

from .base import SourceModel
from ..core import Displacement

NUM_EPS = 1e-12  # float64 denominator/log/sqrt safety


ROOT16 = torch.tensor([
    -0.989400934991649932596,
    -0.944575023073232576078,
    -0.865631202387831743880,
    -0.755404408355003033895,
    -0.617876244402643748447,
    -0.458016777657227386342,
    -0.281603550779258913230,
    -0.095012509837637440185,
     0.095012509837637440185,
     0.281603550779258913230,
     0.458016777657227386342,
     0.617876244402643748447,
     0.755404408355003033895,
     0.865631202387831743880,
     0.944575023073232576078,
     0.989400934991649932596,
], dtype=torch.float64)

WEIGHT16 = torch.tensor([
    0.027152459411754094852,
    0.062253523938647892863,
    0.095158511682492784810,
    0.124628971255533872052,
    0.149595988816576732081,
    0.169156519395002538189,
    0.182603415044923588867,
    0.189450610455068496285,
    0.189450610455068496285,
    0.182603415044923588867,
    0.169156519395002538189,
    0.149595988816576732081,
    0.124628971255533872052,
    0.095158511682492784810,
    0.062253523938647892863,
    0.027152459411754094852,
], dtype=torch.float64)


def kg(s, p):
    """Elementary kernel term ``(3p - s^2) / (p + s^2)^3`` used by ``fpkernel_vec``.

    ``s`` and ``p`` are broadcast-compatible tensors (``p = 4 h^2``).
    """
    z = s * s
    y = p + z
    return (3.0 * p - z) / (y ** 3)


def kern(w, p):
    """Elementary kernel term ``2 (p^2/2 + w^4 - p w^2/2) / (p + w^2)^3``.

    Companion to :func:`kg`, used to assemble the ``n = 2`` (KN1) kernel.
    ``w`` and ``p`` are broadcast-compatible tensors.
    """
    u = (p + w * w) ** 3
    return 2.0 * (0.5 * p * p + w**4 - 0.5 * p * w * w) / u


def fpkernel_vec(h, t, r, n, *, eps=NUM_EPS, dlt=1e-6):
    """
    Fully tensorized, differentiable version of fpkernel.

    Parameters
    ----------
    h : tensor broadcastable with t and r
    t : tensor
    r : tensor
    n : int in {1,2,3,4}

    Returns
    -------
    tensor broadcast over h, t, r
    """
    # Floor p (= 4h²) away from 0. At depth 0 (h == 0, p == 0) the kernel
    # building blocks kg/kern and the log argument all collapse to 0/0 → NaN on
    # the diagonal (where t == r, so t ± r and the log numerator vanish). Every
    # kernel carries an overall factor of p or h, so it still → 0 as h → 0; the
    # floor only keeps the intermediates finite. clamp is inert for any
    # realistic depth (4h² ≫ eps), so existing results are unchanged.
    p = torch.clamp(4.0 * h * h, min=eps)

    if n == 1:  # KN
        return p * h * (kg(t - r, p) - kg(t + r, p))

    elif n == 2:  # KN1
        a = t + r
        b = t - r
        y = a * a
        z = b * b

        g = 2.0 * p * h * (p * p + 6.0 * p * (t * t + r * r) + 5.0 * (a * b) ** 2)
        s = g / (((p + z) * (p + y)) ** 2)

        safe_t = torch.where(torch.abs(t) < eps, torch.full_like(t, eps), t)
        safe_r = torch.where(torch.abs(r) < eps, torch.full_like(r, eps), r)

        trbl_general = h / (safe_t * safe_r) * torch.log((p + z) / (p + y))

        trbl_t0 = -4.0 * h / (p + r * r)
        trbl_r0 = -4.0 * h / (p + t * t)

        # Match original branch order:
        # if |t| < dlt: use trbl_t0
        # elif r > dlt: use general expression
        # else: use trbl_r0
        trbl = torch.where(
            torch.abs(t) < dlt,
            trbl_t0,
            torch.where(r > dlt, trbl_general, trbl_r0)
        )

        return trbl + s + h * (kern(b, p) + kern(a, p))

    elif n == 3:  # KM
        y = (t + r) ** 2
        z = (t - r) ** 2
        a = ((p + y) * (p + z)) ** 2
        c = t + r
        d = t - r
        b = p * t * ((3.0 * p * p - (c * d) ** 2 + 2.0 * p * (t * t + r * r)) / a)
        a2 = p / 2.0 * (c * kg(c, p) + d * kg(d, p))
        return b - a2

    elif n == 4:  # KM1(t,r)=KM(r,t)
        y = (t + r) ** 2
        z = (t - r) ** 2
        a = ((p + y) * (p + z)) ** 2
        c = t + r
        d = -t + r
        b = p * r * ((3.0 * p * p - (c * d) ** 2 + 2.0 * p * (t * t + r * r)) / a)
        a2 = p / 2.0 * (c * kg(c, p) + d * kg(d, p))
        return b - a2

    else:
        raise ValueError(f"Unknown kernel index n={n}")


def build_quadrature(nis, root, weight, *, device, dtype):
    """
    Build t and Wt without Python loops.
    """
    root = root.to(device=device, dtype=dtype)
    weight = weight.to(device=device, dtype=dtype)

    k = torch.arange(nis, device=device, dtype=dtype)  # [0, ..., nis-1]
    d1 = 1.0 / nis

    t_left = d1 * k[:, None]         # [nis, 1]
    t_right = d1 * (k + 1)[:, None]  # [nis, 1]

    rr = root[None, :]               # [1, 16]
    ww = weight[None, :]             # [1, 16]

    t = (rr * (t_right - t_left) * 0.5 + (t_right + t_left) * 0.5).reshape(-1)
    Wt = (0.5 / nis * ww).expand(nis, -1).reshape(-1)

    return t, Wt


def fredholm_solve_differentiable(h, t, Wt):
    """
    Solve the Fredholm system directly as a linear system.

    Parameters
    ----------
    h : tensor [B] or scalar tensor
        Dimensionless crack depth.
    t : tensor [M]
    Wt : tensor [M]

    Returns
    -------
    fi, psi : tensors [B, M] if h is batched, otherwise [1, M]
    """
    lamda = 2.0 / math.pi

    if h.ndim == 0:
        h = h[None]

    B = h.shape[0]
    M = t.numel()
    dtype = t.dtype
    device = t.device

    H = h[:, None, None]          # [B,1,1]
    T = t[None, :, None]          # [1,M,1]
    R = t[None, None, :]          # [1,1,M]

    # Kernels [B, M, M]
    K1 = fpkernel_vec(H, T, R, 1)
    K2 = fpkernel_vec(H, T, R, 2)
    K3 = fpkernel_vec(H, T, R, 3)
    K4 = fpkernel_vec(H, T, R, 4)

    # Fold in quadrature weights on the "integration" dimension (columns)
    W = Wt[None, None, :]         # [1,1,M]
    K1w = K1 * W
    K2w = K2 * W
    K3w = K3 * W
    K4w = K4 * W

    I = torch.eye(M, dtype=dtype, device=device)[None, :, :]  # [1,M,M]

    A11 = I - lamda * K1w
    A12 = -lamda * K3w
    A21 = -lamda * K4w
    A22 = I - lamda * K2w

    A_top = torch.cat([A11, A12], dim=-1)   # [B,M,2M]
    A_bot = torch.cat([A21, A22], dim=-1)   # [B,M,2M]
    A = torch.cat([A_top, A_bot], dim=-2)   # [B,2M,2M]

    rhs_f = -lamda * t[None, :]             # [1,M]
    rhs_p = torch.zeros((B, M), dtype=dtype, device=device)
    rhs = torch.cat([rhs_f.expand(B, -1), rhs_p], dim=-1)  # [B,2M]

    # A = A + 1e-12 * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)[None] # mitigation
    sol = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)  # [B,2M]

    fi = sol[:, :M]
    psi = sol[:, M:]
    return fi, psi


def q_all(h, t, r, *, eps=NUM_EPS):
    """
    Compute Q1..Q8 in one differentiable pass.

    Parameters
    ----------
    h : tensor broadcastable with t and r
    t : tensor
    r : tensor

    Returns
    -------
    q1..q8 : tensors of broadcast shape
    """
    e = h * h + r * r - t * t
    d = torch.sqrt(e * e + 4.0 * h * h * t * t + eps)
    d3 = d * d * d

    sqrt2 = math.sqrt(2.0)
    sqrt_dp = torch.sqrt(d + e + eps)
    sqrt_dm = torch.sqrt(d - e + eps)

    safe_r = torch.where(torch.abs(r) < eps, torch.full_like(r, eps), r)

    q1 = sqrt2 * h * t / (d * sqrt_dp)

    q2 = (
        h * sqrt_dm * (2.0 * e + d)
        - t * sqrt_dp * (2.0 * e - d)
    ) / (sqrt2 * d3)

    q3 = (
        h * sqrt_dp * (2.0 * e - d)
        + t * sqrt_dm * (2.0 * e + d)
    ) / (sqrt2 * d3)

    q4 = t / safe_r - sqrt_dm / (safe_r * sqrt2)

    q5 = -(h * sqrt_dm - t * sqrt_dp) / (d * safe_r * sqrt2)

    q6 = 1.0 / safe_r - (h * sqrt_dp + t * sqrt_dm) / (d * safe_r * sqrt2)

    q7 = safe_r * sqrt_dp * (2.0 * e - d) / (d3 * sqrt2)

    q8 = safe_r * sqrt_dm * (2.0 * e + d) / (d3 * sqrt2)

    return q1, q2, q3, q4, q5, q6, q7, q8


def intgr_batched(r, fi, psi, h, Wt, t, *, eps=NUM_EPS):
    """
    Batched/vectorized version of intgr.

    Parameters
    ----------
    r : tensor [B, N]
    fi, psi : tensors [B, M]
    h : tensor [B]
    Wt, t : tensors [M]

    Returns
    -------
    Uz, Ur : tensors [B, N]
    """
    B, N = r.shape
    M = t.numel()

    H = h[:, None, None]       # [B,1,1]
    R = r[:, :, None]          # [B,N,1]
    T = t[None, None, :]       # [1,1,M]

    q1, q2, q3, q4, q5, q6, q7, q8 = q_all(H, T, R, eps=eps)  # [B,N,M]

    inv_t = 1.0 / torch.clamp(T, min=eps)   # [1,1,M]
    W = Wt[None, None, :]                   # [1,1,M]
    FI = fi[:, None, :]                     # [B,1,M]
    PSI = psi[:, None, :]                   # [B,1,M]

    Uz = torch.sum(
        W * (
            FI * (q1 + H * q2)
            + PSI * (q1 * inv_t - q3)
        ),
        dim=-1
    )  # [B,N]

    Ur = torch.sum(
        W * (
            PSI * (
                (q4 - H * q5) * inv_t
                - q6
                + H * q7
            )
            - H * FI * q8
        ),
        dim=-1
    )  # [B,N]

    return Uz, Ur


class PennySource(SourceModel):
    """
    Differentiable, vectorized penny-shaped crack source.
    """

    def __init__(
        self,
        poisson_ratio: float = 0.25,
        shear_modulus: float = 3e10,
        internal_dtype: torch.dtype = torch.float64,
        nis:int=2,
        num_eps:float = NUM_EPS,
    ):
        super().__init__()
        self.v = poisson_ratio
        self.mu = shear_modulus
        self.internal_dtype = internal_dtype
        self.nis = nis
        self.num_eps = num_eps

        self.register_buffer("root16", ROOT16.to(internal_dtype))
        self.register_buffer("weight16", WEIGHT16.to(internal_dtype))

    def forward(
        self,
        x_obs: Tensor,      # [B, N]
        y_obs: Tensor,      # [B, N]
        source_x: Tensor,   # [B]
        source_y: Tensor,   # [B]
        depth: Tensor,      # [B]
        radius: Tensor,     # [B]
        pressure: Tensor,   # [B]
    ) -> Displacement:
        """Surface displacement from a pressurised penny-shaped crack.

        Parameters
        ----------
        x_obs, y_obs : Tensor
            East/north observation coordinates [B, N] in metres.
        source_x, source_y : Tensor
            East/north position of the crack centre [B] in metres.
        depth : Tensor
            Depth of the crack centre [B] in metres (positive down).
        radius : Tensor
            Crack radius [B] in metres.
        pressure : Tensor
            Excess (boundary) pressure [B] in Pa; scaled by the configured shear
            modulus and Poisson ratio.

        Returns
        -------
        Displacement
            ENU surface displacement [B, N] in metres.
        """
        self._validate_inputs(
            x_obs, y_obs,
            {"source_x": source_x, "source_y": source_y, "depth": depth,
             "radius": radius, "pressure": pressure},
        )

        # radius == 0 collapses the penny to a point: dx, dy, h and Pf all divide
        # by it, poisoning the whole batch with NaN (forward and gradient) with no
        # clean way to recover. Fail loudly instead.
        if torch.any(radius <= 0):
            raise ValueError("radius must be strictly positive (> 0)")

        dtype = self.internal_dtype
        device = x_obs.device

        x_obs = x_obs.to(dtype=dtype, device=device)
        y_obs = y_obs.to(dtype=dtype, device=device)
        source_x = source_x.to(dtype=dtype, device=device)
        source_y = source_y.to(dtype=dtype, device=device)
        depth = depth.to(dtype=dtype, device=device)
        radius = radius.to(dtype=dtype, device=device)
        pressure = pressure.to(dtype=dtype, device=device)

        # Dimensionless coordinates relative to source center
        dx = (x_obs - source_x[:, None]) / radius[:, None]   # [B,N]
        dy = (y_obs - source_y[:, None]) / radius[:, None]   # [B,N]
        r = torch.sqrt(dx * dx + dy * dy + self.num_eps)     # [B,N]

        # Dimensionless crack depth
        h = depth / radius                                   # [B]

        # Pressure scaling
        Pf = 2.0 * (1.0 - self.v) * radius * pressure / self.mu   # [B]

        # Quadrature nodes
        t, Wt = build_quadrature(
            self.nis,
            self.root16,
            self.weight16,
            device=device,
            dtype=dtype,
        )

        # Solve Fredholm system for every batch item
        fi, psi = fredholm_solve_differentiable(h, t, Wt)    # [B,M], [B,M]

        # Integrate for all observation points
        Uz_dimless, Ur_dimless = intgr_batched(r, fi, psi, h, Wt, t, eps=self.num_eps)

        Uz = -Uz_dimless * Pf[:, None]
        Ur =  Ur_dimless * Pf[:, None]

        Nx = dx / r
        Ny = dy / r

        ue = Ur * Nx
        un = Ur * Ny
        uu = Uz

        return Displacement(e=ue, n=un, u=uu)
