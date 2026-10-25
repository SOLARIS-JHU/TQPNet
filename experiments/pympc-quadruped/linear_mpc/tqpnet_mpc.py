"""TQPNet for pympc-quadruped's SRBD-MPC.

Strategy
--------
The MPC's Q matrix depends on yaw and per-leg foot positions, which makes the
QP non-parametric across robots/timesteps. To use TQPNet (which assumes a
*parametric* QP family with fixed Q0/A0), we approximate:

    Q ≈ Q0   (built at yaw=0 and nominal stand-pose foot positions)
    A ≈ A0   (friction cone is exact, fully fixed)

Under this approximation, q and c become linear in a low-dim parameter:

    q(p) = q0 + Cq · [x_t ; X_ref]   ∈ R^(n)        (n = 12·H)
    c(p) = c0 + Cc · gait_table       ∈ R^(m)        (m = 24·H, qpth form)

This matches the parametric-QP setup of the QP experiments and lets us pre-compute the
matmul matrices (M1, M2, H_inv, M_beta) for fast K-step ADMM-ReLU refinement
on GPU.  In closed-loop the actual yaw drifts a bit during a trot, so the
refine is on the *nominal* Q—approximate but cheap (no factorization).

Layers
------
- ``NominalQPConfig`` : builds Q0, A0, q0, Cq, c0, Cc + ADMM matmul tensors.
- ``TQPNet``          : MLP(p) → (x_warm, beta_warm).
- ``build_problem_data(P, cfg)`` : (B, np_dim) → (Q_b, q_b, A_b, c_b).
- ``admm_refine(...)``           : K-step matmul ADMM-ReLU refinement (no grad).
- ``admm_unrolled_loss(...)``    : differentiable K_unroll-step loss.
- ``BatchTQPNetController``      : drop-in counterpart of BatchMPC.

The QP is in qpth form  min 1/2 u^T Q u + q^T u  s.t. A u <= c
(translated from MPC's C_lb <= C u <= C_ub by stacking [C; -C]).
"""
from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import expm

from linear_mpc_configs import LinearMpcConfig
from robot_configs import RobotConfig


# ── Problem config (mirror of the problem config in experiments/base.py) ────────

@dataclass
class NominalQPConfig:
    n: int                # decision dim = 12·H
    m: int                # constraint dim = 24·H (qpth form)
    np_dim: int           # parameter dim
    rho_refine: float = 1.0

    Q0:        torch.Tensor = field(default=None, repr=False)   # (n,n)
    A0:        torch.Tensor = field(default=None, repr=False)   # (m,n)
    q0:        torch.Tensor = field(default=None, repr=False)   # (n,)
    Cq:        torch.Tensor = field(default=None, repr=False)   # (n, np_dim)
    c0_vec:    torch.Tensor = field(default=None, repr=False)   # (m,)
    Cc:        torch.Tensor = field(default=None, repr=False)   # (m, np_dim)

    _D_U:      torch.Tensor = field(default=None, repr=False)   # (m,)
    _D_V:      torch.Tensor = field(default=None, repr=False)   # (n,)
    _inv_D_U:  torch.Tensor = field(default=None, repr=False)   # (m,)
    _H_V_chol: torch.Tensor = field(default=None, repr=False)   # (n,n)
    _M1:       torch.Tensor = field(default=None, repr=False)   # (n,n)
    _M2:       torch.Tensor = field(default=None, repr=False)   # (n,m)
    _H_inv:    torch.Tensor = field(default=None, repr=False)   # (n,n)
    _M_beta:   torch.Tensor = field(default=None, repr=False)   # (m,n)


def _chol(M: torch.Tensor) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    return torch.linalg.cholesky(M)


def _normalize_cmd_shapes(base_vel_base_des, yaw_turn_rate_des, B: int):
    """Coerce user-supplied cmds to canonical (B, 3) and (B,) numpy arrays.

    Accepts:
        base_vel_base_des : (3,) | (1, 3) | (B, 3)
        yaw_turn_rate_des : scalar | (1,) | (B,) | (1, 1) | (B, 1)
    """
    v = np.asarray(base_vel_base_des, dtype=np.float32)
    if v.ndim == 1:
        v = np.tile(v, (B, 1))
    elif v.shape[0] == 1:
        v = np.tile(v.reshape(1, 3), (B, 1))
    elif v.shape != (B, 3):
        v = v.reshape(B, 3)

    y = np.asarray(yaw_turn_rate_des, dtype=np.float32).reshape(-1)
    if y.size == 1:
        y = np.full(B, y.item(), dtype=np.float32)
    elif y.size != B:
        raise ValueError(f"yaw_turn_rate_des has {y.size} elements; expected 1 or {B}")
    return v.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


def _yaw_R(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _vec2so3(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float32)


# ── Nominal QP build ────────────────────────────────────────────────────────

def build_nominal_qp_config(mpc_cfg: LinearMpcConfig,
                             robot_cfg: RobotConfig,
                             nominal_pos_base_feet: np.ndarray = None,
                             rho_refine: float = 1.0,
                             device: Union[str, torch.device] = 'cuda:0',
                             ) -> NominalQPConfig:
    """Build (Q0, A0, q0, Cq, c0, Cc) with yaw=0 and nominal foot positions.

    Parameter p packs [x_t (13) ; X_ref (13·H) ; gait_table (4·H)],
    so np_dim = 13 + 13·H + 4·H.

    QP form (qpth):  min 1/2 u^T Q u + q^T u  s.t.  A u <= c.
    """
    H        = mpc_cfg.horizon
    dt       = 0.05                          # mpc.py's hard-coded dt
    g        = mpc_cfg.gravity
    mu       = mpc_cfg.friction_coef
    fz_max   = robot_cfg.fz_max
    mass     = robot_cfg.mass_base
    I_base   = np.asarray(robot_cfg.base_inertia_base, dtype=np.float32)

    n = 12 * H
    m = 24 * H        # 6 ineq rows per leg per horizon (×4 legs)
    np_dim = 13 + 13 * H + 4 * H

    # Default nominal foot positions in base frame (rough A1/Aliengo stand pose)
    if nominal_pos_base_feet is None:
        nominal_pos_base_feet = np.array([
            [+0.183, +0.13, -0.30],   # FL
            [+0.183, -0.13, -0.30],   # FR
            [-0.183, +0.13, -0.30],   # RL
            [-0.183, -0.13, -0.30],   # RR
        ], dtype=np.float32)

    # ── 1. Continuous-time SRBD at yaw=0
    Rz = np.eye(3, dtype=np.float32)         # yaw=0
    world_I = Rz @ I_base @ Rz.T
    world_I_inv = np.linalg.inv(world_I)

    Ac = np.zeros((13, 13), dtype=np.float32)
    Bc = np.zeros((13, 12), dtype=np.float32)
    Ac[0:3, 6:9] = Rz.T
    Ac[3:6, 9:12] = np.eye(3, dtype=np.float32)
    Ac[11, 12] = 1.0
    for i in range(4):
        Bc[6:9, 3*i:3*(i+1)] = world_I_inv @ _vec2so3(nominal_pos_base_feet[i])
        Bc[9:12, 3*i:3*(i+1)] = np.eye(3, dtype=np.float32) / mass

    # ZOH discretization
    dim = 25
    sq = np.zeros((dim, dim), dtype=np.float32)
    sq[:13, :13] = Ac * dt
    sq[:13, 13:] = Bc * dt
    E = expm(sq)
    Ad, Bd = E[:13, :13], E[:13, 13:]

    # ── 2. Sx, Su (state and input prediction matrices)
    n_state, n_input = 13, 12
    Sx = np.zeros((n_state * H, n_state), dtype=np.float32)
    Su = np.zeros((n_state * H, n_input * H), dtype=np.float32)
    Apow = [np.eye(n_state, dtype=np.float32)]
    for _ in range(H):
        Apow.append(Apow[-1] @ Ad)
    for i in range(H):
        Sx[n_state*i:n_state*(i+1), :] = Apow[i+1]
        for j in range(H):
            if i >= j:
                Su[n_state*i:n_state*(i+1), n_input*j:n_input*(j+1)] = Apow[i-j] @ Bd

    # ── 3. Q_blk, R_blk
    Qbar = np.kron(np.eye(H), np.asarray(mpc_cfg.Q, dtype=np.float32))
    Rbar = np.kron(np.eye(H), np.asarray(mpc_cfg.R, dtype=np.float32))

    # ── 4. qpH = 2(Su^T Qbar Su + Rbar). PSD by construction.
    # Stronger regularization (1e-3) to make Q strictly PD and eliminate
    # null-space drift in NN warm starts (Fix B).
    H_qp = 2.0 * (Su.T @ Qbar @ Su + Rbar)
    H_qp = 0.5 * (H_qp + H_qp.T) + 1e-3 * np.eye(n, dtype=np.float32)

    # qpg(x_t, X_ref) = 2 Su^T Qbar (Sx · x_t - X_ref)
    #               = (2 Su^T Qbar Sx) · x_t  -  (2 Su^T Qbar) · X_ref
    SuT_Q = Su.T @ Qbar                             # (n, n_state·H)
    M_xt   = 2.0 * SuT_Q @ Sx                       # (n, 13)
    M_xref = -2.0 * SuT_Q                           # (n, 13·H)

    # ── 5. Friction cone constraints (qpth form  Gz <= h)
    # Per leg block (6 rows × 3 cols):
    G_leg = np.array([
        [-1.0,  0.0, -mu],   # -fx - μfz <= 0
        [+1.0,  0.0, -mu],   # +fx - μfz <= 0
        [ 0.0, -1.0, -mu],
        [ 0.0, +1.0, -mu],
        [ 0.0,  0.0, +1.0],  # +fz <= gait·fz_max
        [ 0.0,  0.0, -1.0],  # -fz <= 0
    ], dtype=np.float32)

    A0 = np.zeros((m, n), dtype=np.float32)
    for h in range(H):
        for leg in range(4):
            row = (h * 4 + leg) * 6
            col = h * 12 + leg * 3
            A0[row:row+6, col:col+3] = G_leg

    # h vector: only the 5th row (fz_upper) depends on gait·fz_max; others = 0
    c0_vec = np.zeros(m, dtype=np.float32)
    Cc = np.zeros((m, 4 * H), dtype=np.float32)
    for h in range(H):
        for leg in range(4):
            row_fz_upper = (h * 4 + leg) * 6 + 4
            gait_idx = 4 * h + leg
            Cc[row_fz_upper, gait_idx] = fz_max

    # ── 6. Pack into NominalQPConfig (parameter layout: [x_t(13) | X_ref(13H) | gait(4H)])
    cfg = NominalQPConfig(n=n, m=m, np_dim=np_dim, rho_refine=rho_refine)

    # Cq has structure [M_xt | M_xref | 0]
    Cq = np.zeros((n, np_dim), dtype=np.float32)
    Cq[:, 0:13]            = M_xt
    Cq[:, 13:13 + 13*H]    = M_xref
    q0 = np.zeros(n, dtype=np.float32)

    # Cc operates only on the trailing gait_table (4H) slice of p
    Cc_full = np.zeros((m, np_dim), dtype=np.float32)
    Cc_full[:, 13 + 13*H:] = Cc

    cfg.Q0      = torch.tensor(H_qp,    dtype=torch.float32, device=device) #+ 0.01 * torch.eye(n, device=device)  # regularize for ADMM
    cfg.A0      = torch.tensor(A0,      dtype=torch.float32, device=device)
    cfg.q0      = torch.tensor(q0,      dtype=torch.float32, device=device)
    cfg.Cq      = torch.tensor(Cq,      dtype=torch.float32, device=device)
    cfg.c0_vec  = torch.tensor(c0_vec,  dtype=torch.float32, device=device)
    cfg.Cc      = torch.tensor(Cc_full, dtype=torch.float32, device=device)

    # Pre-compute matmul tensors for *proper* ADMM (full A^T A, no D_V approx).
    # Linear system per iteration:  H_V x = ρ A^T A x_old + ρ A^T β  - q
    # Allowed because Q0 / A0 are FIXED (nominal QP family, parameter only enters q,c).
    A0    = cfg.A0
    AtA   = A0.T @ A0                                                # (n,n)
    H_V   = cfg.Q0 + rho_refine * AtA                                 # (n,n)  PSD
    cfg._H_V_chol = _chol(H_V)
    H_inv = torch.linalg.solve(H_V, torch.eye(n, device=device))
    # x update :  x = M1 x_old + M2 β + bias_x
    cfg._M1    = rho_refine * H_inv @ AtA
    cfg._M2    = rho_refine * H_inv @ A0.T
    cfg._H_inv = H_inv
    # β update :  β = max(0, β + ρ (A x - c))  - standard inequality dual update
    # encoded as M_beta @ x then add bias_beta = -ρ c
    cfg._M_beta = rho_refine * A0                                     # (m,n)
    # legacy fields kept for any external code
    cfg._D_U     = (A0 ** 2).sum(dim=1)
    cfg._D_V     = (A0.abs() > 1e-12).to(A0.dtype).sum(dim=0)
    cfg._inv_D_U = 1.0 / cfg._D_U.clamp(min=1e-12)
    return cfg


# ── Build per-batch QP data from packed parameter ──────────────────────────

def build_problem_data(P: torch.Tensor, cfg: NominalQPConfig):
    """P : (B, np_dim) → (Q_b, q_b, A_b, c_b)."""
    b = P.shape[0]
    return (
        cfg.Q0.unsqueeze(0).expand(b, -1, -1),
        cfg.q0.unsqueeze(0) + P @ cfg.Cq.T,
        cfg.A0.unsqueeze(0).expand(b, -1, -1),
        cfg.c0_vec.unsqueeze(0) + P @ cfg.Cc.T,
    )


# ── KKT residual + losses ──────────────────────────────────────────────────

def kkt_raw_residual(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg):
    """KKT residual for inequality QP  min ½xᵀQx + qᵀx  s.t.  Ax ≤ c, β ≥ 0.

    Stationarity   : r1 = Qx + q + Aᵀβ
    Compl. slack   : r2 = β - max(0, β + Ax - c)
    """
    Qx  = torch.bmm(Q_b, x_pred.unsqueeze(-1)).squeeze(-1)
    Ax  = torch.bmm(A_b, x_pred.unsqueeze(-1)).squeeze(-1)
    ATb = torch.bmm(A_b.transpose(1, 2), beta_pred.unsqueeze(-1)).squeeze(-1)
    r1  = Qx + q_b + ATb
    r2  = beta_pred - (beta_pred + Ax - c_b).clamp(min=0.0)
    return (r1**2).sum(1) / cfg.n + (r2**2).sum(1) / cfg.m


def admm_informed_loss(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg):
    return torch.log1p(kkt_raw_residual(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg)).mean()


def admm_unrolled_loss(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg, K_unroll: int = 10):
    """OSQP-style ADMM for inequality QP, K_unroll differentiable steps.

    Variables:  primal x; slack z = clip(Ax + u, -inf, c); scaled dual u = β/ρ.
    Update:
        (Q + ρ A^T A) x = ρ A^T (z - u) - q          (Cholesky pre-computed)
        z              = min(c, A x + u)
        u              = u + A x - z
    """
    rho   = cfg.rho_refine
    x     = x_pred
    u     = (beta_pred.clamp(min=0)) / rho
    A_t   = A_b.transpose(1, 2)
    chol  = cfg._H_V_chol.unsqueeze(0)
    Ax    = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
    z     = torch.minimum(c_b, Ax + u)

    for _ in range(K_unroll):
        rhs = rho * torch.bmm(A_t, (z - u).unsqueeze(-1)).squeeze(-1) - q_b
        x   = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        Ax  = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
        z   = torch.minimum(c_b, Ax + u)
        u   = u + Ax - z
    beta = rho * u
    return admm_informed_loss(x, beta, Q_b, q_b, A_b, c_b, cfg)


# ── ADMM-ReLU refinement (inference) ────────────────────────────────────────

@torch.no_grad()
def admm_refine(x0, Q_b, q_b, A_b, c_b, cfg: NominalQPConfig,
                K: int = 10, beta0=None):
    """OSQP-style ADMM for inequality QP   min ½xᵀQx + qᵀx  s.t.  Ax ≤ c.

    Per iteration:
        rhs = ρ Aᵀ(z − u) − q              (uses A_b == cfg.A0, q_b)
        x   = (Q + ρ AᵀA)⁻¹ rhs            ← **Q is used here** (via cfg._H_V_chol)
        z   = min(c, A x + u)
        u   = u + (A x − z)

    Q does not appear *explicitly* in the loop body because we pre-factor
    (Q + ρ AᵀA) once into cfg._H_V_chol (Cholesky) at config build time.  The
    cholesky_solve then implicitly multiplies by (Q + ρ AᵀA)⁻¹, i.e. uses Q.

    Constraint: Q_b, A_b are required to equal cfg.Q0, cfg.A0 (nominal QP
    family) — otherwise chol is inconsistent.  build_problem_data guarantees
    this by returning batched views of cfg.Q0/A0.
    """
    batch, dev, dtype = x0.shape[0], x0.device, x0.dtype
    rho = cfg.rho_refine

    # Sanity check: assert nominal-QP consistency.
    assert Q_b.shape[1:] == cfg.Q0.shape and A_b.shape[1:] == cfg.A0.shape, \
        "admm_refine: Q_b/A_b must match cfg.Q0/cfg.A0 shape (nominal QP family)."

    x   = x0.clone()
    u   = (beta0.clamp(min=0) / rho).clone() if beta0 is not None \
          else torch.zeros(batch, cfg.m, device=dev, dtype=dtype)
    Ax  = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
    z   = torch.minimum(c_b, Ax + u)

    A_t  = A_b.transpose(1, 2)
    chol = cfg._H_V_chol.unsqueeze(0)         # Cholesky(Q0 + ρ A0ᵀA0) — Q baked in

    for k in range(K):
        # x update :  (Q + ρ AᵀA) x = ρ Aᵀ(z − u) − q
        rhs = rho * torch.bmm(A_t, (z - u).unsqueeze(-1)).squeeze(-1) - q_b
        x   = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        # NaN/Inf guard — if any sample diverged (large K + ill-conditioned),
        # break early and fall back to last sane (x, u).  Otherwise NaN will
        # propagate through subsequent iterations and out to the actuators.
        if not torch.isfinite(x).all():
            print(f"[admm_refine] non-finite x at iter {k}/{K}; breaking early.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            break
        # z update
        Ax  = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
        z   = torch.minimum(c_b, Ax + u)
        # u update
        u   = u + Ax - z
    return x, rho * u


# ── Network ────────────────────────────────────────────────────────────────

def _xavier_init(mod):
    if isinstance(mod, nn.Linear):
        nn.init.xavier_uniform_(mod.weight)
        nn.init.zeros_(mod.bias)


class TQPNet(nn.Module):
    """MLP(p) → (x_warm, beta_warm)."""
    def __init__(self, np_dim: int, n: int, m: int, W: int = 512, L: int = 4):
        super().__init__()
        self.n, self.m = n, m
        layers = [nn.Linear(np_dim, W), nn.ReLU()]
        for _ in range(L - 1):
            layers += [nn.Linear(W, W), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head_x   = nn.Linear(W, n)
        self.head_b   = nn.Sequential(nn.Linear(W, m), nn.Softplus())
        self.apply(_xavier_init)

    def forward(self, p):
        h = self.backbone(p)
        return self.head_x(h), self.head_b(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class ScaledTQPNet(TQPNet):
    """TQPNet with output ×x_scale (helps when force magnitudes ~ 100 N)."""
    def __init__(self, np_dim, n, m, W=512, L=4, x_scale: float = 100.0):
        super().__init__(np_dim, n, m, W=W, L=L)
        self.x_scale = float(x_scale)

    def forward(self, p):
        x, beta = super().forward(p)
        return x * self.x_scale, beta * self.x_scale


class NormalizedTQPNet(TQPNet):
    """TQPNet with input z-score normalization + per-dim output scaling.

    Buffers (saved with state_dict):
      input_mean, input_std       : (np_dim,)   parameter normalization
      output_x_scale              : (n,)        per-dim primal scale
      output_b_scale              : (m,)        per-dim dual scale

    forward:
      p_norm = (p - mean) / std
      h = MLP(p_norm)
      return  head_x(h) ⊙ x_scale,   softplus(head_b(h)) ⊙ b_scale
    """
    def __init__(self, np_dim: int, n: int, m: int, W: int = 512, L: int = 4):
        super().__init__(np_dim, n, m, W=W, L=L)
        self.register_buffer('input_mean',     torch.zeros(np_dim))
        self.register_buffer('input_std',      torch.ones(np_dim))
        self.register_buffer('output_x_scale', torch.ones(n))
        self.register_buffer('output_b_scale', torch.ones(m))

    def set_normalization(self,
                           input_mean: torch.Tensor,
                           input_std: torch.Tensor,
                           output_x_scale: torch.Tensor,
                           output_b_scale: torch.Tensor):
        self.input_mean.data.copy_(input_mean.to(self.input_mean))
        self.input_std.data.copy_(input_std.to(self.input_std).clamp(min=1e-3))
        self.output_x_scale.data.copy_(output_x_scale.to(self.output_x_scale).clamp(min=1e-3))
        self.output_b_scale.data.copy_(output_b_scale.to(self.output_b_scale).clamp(min=1e-3))

    def forward(self, p):
        p_norm = (p - self.input_mean) / self.input_std
        x, beta = super().forward(p_norm)
        return x * self.output_x_scale, beta * self.output_b_scale


# ── Batch controller (drop-in counterpart of BatchModelPredictiveController) ─

class BatchTQPNetController:
    """N-robot MPC using TQPNet warm start + K-step ADMM-ReLU refinement.

    Same public surface as `BatchModelPredictiveController` so the script can
    swap implementations behind `--solver tqpnet`.
    """
    def __init__(self,
                 mpc_config: LinearMpcConfig,
                 robot_config: RobotConfig,
                 num_robots: int,
                 model_path: str,
                 device: Union[str, torch.device] = 'cuda:0',
                 K_refine: int = 10,
                 net_width: int = 512,
                 net_depth: int = 4,
                 x_scale: float = 100.0,
                 use_compact_input: bool = True,
                 disable_nn: bool = False,
                 rho_refine: float = 1.0):
        """rho_refine MUST match the value used during training (default 1.0,
        same as train_tqpnet.py default).  Mismatched ρ → chol factorizes the
        wrong (Q+ρAᵀA) and ADMM converges to a non-optimal fixed point."""
        self.disable_nn = disable_nn
        self.B = num_robots
        self.device = torch.device(device)
        self.K_refine = K_refine
        self.use_compact_input = use_compact_input
        self.mpc_config = mpc_config
        self.robot_config = robot_config

        # Load nominal QP family.  Crucially pass rho_refine here so that the
        # pre-computed Cholesky factor matches the value used during training.
        self.cfg = build_nominal_qp_config(mpc_config, robot_config,
                                            rho_refine=rho_refine,
                                            device=self.device)
        self.np_dim_full = self.cfg.np_dim
        self.np_dim_nn   = COMPACT_DIM if use_compact_input else self.cfg.np_dim
        self.n_dec  = self.cfg.n
        self.m_con  = self.cfg.m

        # MPC scheduling parameters used by the script
        self.dt_control = mpc_config.dt_control
        self.iterations_between_mpc = mpc_config.iteration_between_mpc
        self.dt = 0.05
        self.horizon = mpc_config.horizon
        self.gravity = mpc_config.gravity
        self.com_height_des = robot_config.base_height_des
        self.mass = robot_config.mass_base

        # If disable_nn: skip NN, use cold-start (x=β=0) then ADMM K refine.
        if disable_nn:
            self.model = None
            self.np_dim_nn = 0
            print(f"[ColdADMM] cold-start ADMM (K={K_refine}) — no NN")
        # Network — try NormalizedTQPNet first (preferred), fallback to ScaledTQPNet
        elif model_path is not None and os.path.isfile(model_path):
            sd = torch.load(model_path, map_location=self.device)
            # Auto-detect input dimension from saved weights
            backbone_in_dim = sd.get('backbone.0.weight', sd.get('input_mean', None))
            if 'input_mean' in sd:
                detected_np_dim = sd['input_mean'].numel()
            elif 'backbone.0.weight' in sd:
                detected_np_dim = sd['backbone.0.weight'].shape[1]
            else:
                detected_np_dim = self.np_dim_nn
            self.np_dim_nn = detected_np_dim
            self.use_compact_input = (detected_np_dim == COMPACT_DIM)

            if 'input_mean' in sd:
                self.model = NormalizedTQPNet(detected_np_dim, self.n_dec, self.m_con,
                                               W=net_width, L=net_depth).to(self.device)
                kind = 'NormalizedTQPNet'
            else:
                self.model = ScaledTQPNet(detected_np_dim, self.n_dec, self.m_con,
                                           W=net_width, L=net_depth,
                                           x_scale=x_scale).to(self.device)
                kind = 'ScaledTQPNet'
            self.model.load_state_dict(sd)
            print(f"[TQPNet] loaded weights from {model_path}  "
                  f"({self.model.count_params()/1e6:.2f} M params, {kind}, "
                  f"np_dim={detected_np_dim}, "
                  f"input={'compact' if self.use_compact_input else 'full'})")
        else:
            self.model = ScaledTQPNet(self.np_dim_nn, self.n_dec, self.m_con,
                                       W=net_width, L=net_depth, x_scale=x_scale).to(self.device)
            print(f"[TQPNet] WARNING: weights not found at {model_path}; "
                  f"using random-init ScaledTQPNet (np_dim={self.np_dim_nn}).")
        if self.model is not None:
            self.model.eval()

        # Per-robot stateful trajectory bookkeeping (mirrors single MPC)
        self.is_first_run = np.ones(num_robots, dtype=bool)
        self.current_state = np.zeros((num_robots, 13), dtype=np.float32)
        self.yaw = np.zeros(num_robots, dtype=np.float32)
        # Stacked rotation matrices (B,3,3); maintained in update_robot_state
        # so downstream batch ops avoid a Python loop.
        self.R_base = np.zeros((num_robots, 3, 3), dtype=np.float32)
        self.pos_base_feet = np.zeros((num_robots, 4, 3), dtype=np.float32)
        self.roll_init = np.zeros(num_robots, dtype=np.float32)
        self.pitch_init = np.zeros(num_robots, dtype=np.float32)
        self.xpos_base_desired = np.zeros(num_robots, dtype=np.float32)
        self.ypos_base_desired = np.zeros(num_robots, dtype=np.float32)
        self.yaw_desired = np.zeros(num_robots, dtype=np.float32)
        self._robot_data_list: list = [None] * num_robots
        self.__contact_forces = np.zeros((num_robots, self.n_dec), dtype=np.float32)

        # Static QP terms shared by the actual-Q (cold-admm) path so that the
        # construction matches BatchModelPredictiveController exactly.
        self._build_static_actual_qp_terms()

    # ──────────────────────────────────────────────────────────────────────

    def update_robot_state(self, robot_idx: int, robot_data):
        from kinematics import quat2ZYXangle
        rd = robot_data
        rpy = quat2ZYXangle(rd.quat_base)
        cs = self.current_state[robot_idx]
        cs[0:3] = rpy
        cs[3:6] = np.asarray(rd.pos_base, dtype=np.float32)
        cs[6:9] = np.asarray(rd.ang_vel_base, dtype=np.float32)
        cs[9:12] = np.asarray(rd.lin_vel_base, dtype=np.float32)
        cs[12] = -self.gravity
        self.yaw[robot_idx] = rpy[2]
        # Real per-leg foot positions in base frame; needed by the actual-Q path.
        self.pos_base_feet[robot_idx] = np.asarray(rd.pos_base_feet, dtype=np.float32)
        self.R_base[robot_idx] = np.asarray(rd.R_base, dtype=np.float32)
        self._robot_data_list[robot_idx] = rd

    def update_robot_state_batch(self,
                                  rpy_b: np.ndarray,
                                  pos_base_b: np.ndarray,
                                  lin_vel_b: np.ndarray,
                                  ang_vel_b: np.ndarray,
                                  pos_base_feet_b: np.ndarray,
                                  R_base_b: np.ndarray,
                                  robot_data_list=None) -> None:
        """Batched state intake — replaces NUM_ROBOTS× per-robot Python calls.

        Caller passes ALREADY-STACKED arrays (no Python loops here):
            rpy_b           : (B, 3)     ZYX angles
            pos_base_b      : (B, 3)
            lin_vel_b       : (B, 3)     world frame
            ang_vel_b       : (B, 3)     world frame
            pos_base_feet_b : (B, 4, 3)  per-leg feet in world frame (from FK)
            R_base_b        : (B, 3, 3)
            robot_data_list : optional list of RobotData (kept for back-compat;
                              the controller itself no longer reads from it)
        """
        cs = self.current_state
        cs[:, 0:3]  = rpy_b
        cs[:, 3:6]  = pos_base_b
        cs[:, 6:9]  = ang_vel_b
        cs[:, 9:12] = lin_vel_b
        cs[:, 12]   = -self.gravity
        self.yaw[:]            = rpy_b[:, 2]
        self.pos_base_feet[:]  = pos_base_feet_b
        self.R_base[:]         = R_base_b

        # NaN guard (vectorized)
        bad = ~np.all(np.isfinite(cs), axis=1)
        if bad.any():
            idxs = np.where(bad)[0]
            print(f"[BatchMPC] WARN: NaN/Inf state for {bad.sum()} robots "
                  f"(first: {idxs[:5].tolist()}); resetting to stand pose.")
            cs[bad] = 0.0
            cs[bad, 5]  = self.com_height_des
            cs[bad, 12] = -self.gravity
            self.yaw[bad] = 0.0
        np.clip(cs, -50.0, 50.0, out=cs)

        if robot_data_list is not None:
            self._robot_data_list = robot_data_list
        # Sanity-clamp extreme values (e.g., velocity > 50 m/s = sim explosion).
        np.clip(cs, -50.0, 50.0, out=cs)

    # ──────────────────────────────────────────────────────────────────────

    def update_mpc_if_needed_batch(self,
                                    iter_counter: int,
                                    base_vel_base_des: np.ndarray,
                                    yaw_turn_rate_des: np.ndarray,
                                    gait_tables: np.ndarray) -> np.ndarray:
        """Solve all N MPC QPs via TQPNet warm + ADMM refine."""
        B = self.B
        H = self.horizon
        # Normalize cmd shapes to (B, 3) and (B,).  Accepts scalar / (1,) /
        # (B,) / (B,1) for yaw rate and (3,) / (1,3) / (B,3) for base vel —
        # so the script can pass per-robot or shared commands interchangeably.
        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)

        # Vectorized tracker accumulator (replaces per-robot Python loop).
        v_des_world_all = np.einsum('bij,bj->bi', self.R_base,
                                     base_vel_base_des).astype(np.float32)        # (B,3)
        first = self.is_first_run
        new_xpos = self.xpos_base_desired + self.dt_control * v_des_world_all[:, 0]
        new_ypos = self.ypos_base_desired + self.dt_control * v_des_world_all[:, 1]
        new_yaw  = self.yaw                + self.dt_control * yaw_turn_rate_des
        self.xpos_base_desired = np.where(first, 0.0, new_xpos).astype(np.float32)
        self.ypos_base_desired = np.where(first, 0.0, new_ypos).astype(np.float32)
        self.yaw_desired       = np.where(first, self.yaw, new_yaw).astype(np.float32)
        self.is_first_run[:] = False

        if iter_counter % self.iterations_between_mpc == 0:
            t0 = time.time()

            # ── Deployment ALWAYS solves the actual MPC QP (live yaw + live
            # foot positions + real ref trajectory). Only the warm start
            # differs:
            #   disable_nn=True  → cold start (x=β=0)
            #   disable_nn=False → TQPNet(p_compact) provides (x_warm, β_warm)
            # The NN was trained on the nominal QP family but is asked here to
            # warm-start the *actual* QP — the parametric family approximates
            # the actual one, so the warm start remains useful.
            ref_traj = self._generate_reference_trajectory_batch(
                base_vel_base_des, yaw_turn_rate_des)
            ref_traj_t = torch.tensor(ref_traj, dtype=torch.float32, device=self.device)
            gait_t     = torch.tensor(gait_tables.astype(np.float32),
                                        dtype=torch.float32, device=self.device)
            Q_b, q_b, A_b, c_b = self._build_actual_qp_batch(ref_traj_t, gait_t)
            if not (torch.isfinite(q_b).all() and torch.isfinite(c_b).all()):
                print("[BatchMPC] WARN: QP q/c contains NaN/Inf; reusing previous forces.")
                return self.__contact_forces[:, :12]

            if self.disable_nn:
                x_warm = torch.zeros(B, self.n_dec, device=self.device)
                b_warm = torch.zeros(B, self.m_con, device=self.device)
                tag = 'cold-ADMM(actual-Q)'
            else:
                # NN takes the same input shape it was trained on (compact or full).
                if self.use_compact_input:
                    p_compact = self._build_compact_p(base_vel_base_des,
                                                       yaw_turn_rate_des, iter_counter)
                    p_nn = torch.tensor(p_compact, dtype=torch.float32, device=self.device)
                else:
                    p_np = np.concatenate([
                        self.current_state, ref_traj, gait_tables.astype(np.float32)
                    ], axis=1)
                    p_nn = torch.tensor(p_np, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    x_warm, b_warm = self.model(p_nn)
                tag = f"TQPNet(actual-Q, input={'compact' if self.use_compact_input else 'full'})"

            with torch.no_grad():
                x_ref, _ = self._admm_refine_per_batch(
                    x_warm, Q_b, q_b, A_b, c_b, K=self.K_refine, beta0=b_warm)
            x_ref = torch.nan_to_num(x_ref, nan=0.0, posinf=0.0, neginf=0.0)

            print(f"Batch {tag} ({B} robots, K={self.K_refine}) solved in "
                  f"{(time.time()-t0)*1000:.2f} ms.")
            self.__contact_forces = x_ref.detach().cpu().numpy()

        return self.__contact_forces[:, :12]

    # ──────────────────────────────────────────────────────────────────────

    # ─── Actual-Q (cold-admm) path ───────────────────────────────────────
    # The cold-admm path solves the SAME QP as `--solver osqp`:
    #   Ac/Bc built from real yaw + real foot positions per robot,
    #   Q = 2 (Suᵀ Q̄ Su + R̄), q = 2 Suᵀ Q̄ (Sx xt − Xref),
    #   A,c = friction cone + fz bounds (gait·fz_max).
    # Only the inner solver differs (K-step ADMM cold-start vs qpth/OSQP).

    def _build_static_actual_qp_terms(self):
        """Cache shared (yaw/feet-independent) tensors used by every step."""
        H = self.horizon
        Qi = np.asarray(self.mpc_config.Q, dtype=np.float32)
        Ri = np.asarray(self.mpc_config.R, dtype=np.float32)
        self._Qbar_t = torch.tensor(np.kron(np.eye(H), Qi), dtype=torch.float32,
                                     device=self.device)                      # (13H, 13H)
        self._Rbar_t = torch.tensor(np.kron(np.eye(H), Ri), dtype=torch.float32,
                                     device=self.device)                      # (12H, 12H)
        self._I_base_t = torch.tensor(self.robot_config.base_inertia_base,
                                       dtype=torch.float32, device=self.device)  # (3,3)
        self._mu = float(self.mpc_config.friction_coef)
        self._fz_max = float(self.robot_config.fz_max)

        # Friction cone block, per-leg (6 rows × 3 cols), shared across H,B
        mu = self._mu
        G_leg = torch.tensor([
            [-1.0,  0.0, -mu],
            [+1.0,  0.0, -mu],
            [ 0.0, -1.0, -mu],
            [ 0.0,  0.0, -mu],   # placeholder — replaced below
            [ 0.0,  0.0, +1.0],
            [ 0.0,  0.0, -1.0],
        ], dtype=torch.float32, device=self.device)
        # fix the +fy row (was placeholder)
        G_leg[3] = torch.tensor([0.0, +1.0, -mu], dtype=torch.float32, device=self.device)

        zeros3 = torch.zeros(6, 3, device=self.device)
        leg_block_rows = []
        for leg_in_step in range(4):
            row = [zeros3, zeros3, zeros3, zeros3]
            row[leg_in_step] = G_leg
            leg_block_rows.append(torch.cat(row, dim=1))                       # (6, 12)
        G_step = torch.cat(leg_block_rows, dim=0)                             # (24, 12)

        G_full = torch.zeros(24 * H, 12 * H, device=self.device)
        for h in range(H):
            G_full[24 * h:24 * (h + 1), 12 * h:12 * (h + 1)] = G_step
        self._G_template = G_full                                             # (24H, 12H)
        self._G_batch    = G_full.unsqueeze(0).expand(self.B, -1, -1).contiguous()
        self._GtG        = G_full.T @ G_full                                  # (12H, 12H)

    def _build_actual_qp_batch(self,
                                ref_traj_t: torch.Tensor,
                                gait_t: torch.Tensor):
        """Build the SAME (Q_b, q_b, A_b, c_b) that BatchModelPredictiveController solves.

        Inputs (all on self.device):
            ref_traj_t : (B, 13H)    reference trajectory (yaw-aware, full tracker)
            gait_t     : (B, 4H)     per-leg stance schedule (1 = stance)
        Returns (Q_b, q_b, A_b, c_b) for qpth-form  min ½uᵀQu + qᵀu  s.t. Au ≤ c.
        """
        B = self.B
        H = self.horizon
        n_state, n_input = 13, 12
        dev = self.device

        # 1. dynamics (Ac, Bc) per robot — yaw and foot positions are the only batch axes
        yaw_t = torch.tensor(self.yaw, dtype=torch.float32, device=dev)
        cy, sy = torch.cos(yaw_t), torch.sin(yaw_t)
        zero, one = torch.zeros_like(cy), torch.ones_like(cy)
        Rz = torch.stack([
            torch.stack([cy, -sy, zero], dim=-1),
            torch.stack([sy,  cy, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)                                                         # (B, 3, 3)

        I_base = self._I_base_t.unsqueeze(0).expand(B, -1, -1)
        world_I = Rz @ I_base @ Rz.transpose(-1, -2)
        world_I_inv = torch.linalg.inv(world_I)

        Ac = torch.zeros(B, 13, 13, device=dev)
        Bc = torch.zeros(B, 13, 12, device=dev)
        Ac[:, 0:3, 6:9] = Rz.transpose(-1, -2)
        Ac[:, 3:6, 9:12] = torch.eye(3, device=dev).unsqueeze(0).expand(B, -1, -1)
        Ac[:, 11, 12] = 1.0

        feet = torch.tensor(self.pos_base_feet, dtype=torch.float32, device=dev)  # (B,4,3)
        rx, ry, rz = feet[..., 0], feet[..., 1], feet[..., 2]
        zero_b = torch.zeros_like(rx)
        skew = torch.stack([
            torch.stack([zero_b, -rz, ry], dim=-1),
            torch.stack([rz, zero_b, -rx], dim=-1),
            torch.stack([-ry, rx, zero_b], dim=-1),
        ], dim=-2)                                                         # (B, 4, 3, 3)
        inv_m_eye3 = (torch.eye(3, device=dev) / self.mass).unsqueeze(0).expand(B, -1, -1)
        for i in range(4):
            Bc[:, 6:9, 3*i:3*(i+1)] = world_I_inv @ skew[:, i]
            Bc[:, 9:12, 3*i:3*(i+1)] = inv_m_eye3

        # 2. ZOH discretization via matrix exponential (batched)
        dim = n_state + n_input
        sq = torch.zeros(B, dim, dim, device=dev)
        sq[:, :n_state, :n_state] = Ac * self.dt
        sq[:, :n_state, n_state:] = Bc * self.dt
        E = torch.linalg.matrix_exp(sq)
        Ad = E[:, :n_state, :n_state]
        Bd = E[:, :n_state, n_state:]

        # 3. condense (Sx, Su)
        Sx = torch.zeros(B, n_state * H, n_state, device=dev)
        Su = torch.zeros(B, n_state * H, n_input * H, device=dev)
        Apow = [torch.eye(n_state, device=dev).unsqueeze(0).expand(B, -1, -1)]
        for _ in range(H):
            Apow.append(Apow[-1] @ Ad)
        for i in range(H):
            Sx[:, n_state*i:n_state*(i+1), :] = Apow[i + 1]
            for j in range(H):
                if i >= j:
                    Su[:, n_state*i:n_state*(i+1), n_input*j:n_input*(j+1)] = Apow[i - j] @ Bd

        # 4. cost: qpH = 2 (Suᵀ Q̄ Su + R̄), qpg = 2 Suᵀ Q̄ (Sx xt − Xref)
        xt = torch.tensor(self.current_state, dtype=torch.float32, device=dev)  # (B, 13)
        Q  = self._Qbar_t.unsqueeze(0).expand(B, -1, -1)
        R  = self._Rbar_t.unsqueeze(0).expand(B, -1, -1)
        SuT   = Su.transpose(-1, -2)
        qpH   = 2.0 * (SuT @ Q @ Su + R)
        qpH   = qpH + 1e-3 * torch.eye(qpH.shape[-1], device=dev).unsqueeze(0)
        delta = (Sx @ xt.unsqueeze(-1)).squeeze(-1) - ref_traj_t            # (B, 13H)
        qpg   = 2.0 * (SuT @ Q @ delta.unsqueeze(-1)).squeeze(-1)           # (B, 12H)

        # 5. inequality bounds c_b (h vector): only fz_upper rows depend on gait
        c_b = torch.zeros(B, 24 * H, device=dev)
        # row index (h, leg, 4) = 24h + 6·leg + 4
        for h_idx in range(H):
            for leg_idx in range(4):
                c_b[:, 24 * h_idx + 6 * leg_idx + 4] = (
                    gait_t[:, 4 * h_idx + leg_idx] * self._fz_max)

        return qpH, qpg, self._G_batch, c_b

    @torch.no_grad()
    def _admm_refine_per_batch(self,
                                x0: torch.Tensor,
                                Q_b: torch.Tensor,
                                q_b: torch.Tensor,
                                A_b: torch.Tensor,
                                c_b: torch.Tensor,
                                K: int,
                                beta0: torch.Tensor = None):
        """OSQP-style ADMM with PER-BATCH Cholesky factorization.

        Use this when Q_b varies across the batch (actual MPC QP).  Each step:
            (Q_i + ρ AᵀA) x_i = ρ Aᵀ(z − u) − q_i           (per-batch chol)
            z = min(c, A x + u)
            u = u + A x − z
        """
        rho = self.cfg.rho_refine
        B   = x0.shape[0]
        dev = x0.device
        # Per-batch (Q + ρ AᵀA).  AᵀA is shared (A == G_template), so reuse self._GtG.
        H_qp = Q_b + rho * self._GtG.unsqueeze(0)                              # (B, n, n)
        H_qp = 0.5 * (H_qp + H_qp.transpose(-1, -2))                           # numerical sym
        chol = torch.linalg.cholesky(H_qp)                                     # (B, n, n)

        x = x0.clone()
        u = (beta0.clamp(min=0) / rho).clone() if beta0 is not None \
            else torch.zeros(B, A_b.shape[1], device=dev)
        A_t = A_b.transpose(-1, -2)
        Ax  = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
        z   = torch.minimum(c_b, Ax + u)

        for k in range(K):
            rhs = rho * torch.bmm(A_t, (z - u).unsqueeze(-1)).squeeze(-1) - q_b
            x   = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
            if not torch.isfinite(x).all():
                print(f"[admm_actual] non-finite x at iter {k}/{K}; breaking early.")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                break
            Ax  = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
            z   = torch.minimum(c_b, Ax + u)
            u   = u + Ax - z
        return x, rho * u

    def _build_compact_p(self, base_vel_base_des, yaw_turn_rate_des, iter_counter):
        """Vectorized build of compact p (B, 18) — no Python loop over robots."""
        B = self.B
        z_des = self.com_height_des
        period = 0.5  # trot period (matches Gait.TROTTING10)

        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)

        v_des_world = np.einsum('bij,bj->bi', self.R_base,
                                 base_vel_base_des).astype(np.float32)        # (B, 3)
        gait_phase_scalar = (iter_counter * self.dt_control / period) % 1.0

        cs = self.current_state                                               # (B, 13)
        p = np.zeros((B, COMPACT_DIM), dtype=np.float32)
        p[:, 0:3]  = cs[:, 0:3]                                  # rpy
        p[:, 3]    = self.xpos_base_desired - cs[:, 3]           # xpos_drift
        p[:, 4]    = self.ypos_base_desired - cs[:, 4]           # ypos_drift
        p[:, 5]    = cs[:, 5] - z_des                             # z_drift
        p[:, 6:9]  = cs[:, 6:9]                                   # ω
        p[:, 9:12] = cs[:, 9:12]                                  # v_world
        p[:, 12]   = v_des_world[:, 0]                            # cmd_vx_world
        p[:, 13]   = v_des_world[:, 1]                            # cmd_vy_world
        p[:, 14]   = yaw_turn_rate_des                            # cmd_wz
        p[:, 15]   = self.roll_init
        p[:, 16]   = self.pitch_init
        p[:, 17]   = gait_phase_scalar
        return p

    # ──────────────────────────────────────────────────────────────────────

    def _generate_reference_trajectory_batch(self, base_vel_base_des, yaw_turn_rate_des):
        """Fully vectorized over robots (replaces per-robot Python loop).

        Mutates self.xpos_base_desired/ypos_base_desired (clamped to ±0.1m) and
        self.roll_init/pitch_init (terrain compensation), then builds the
        (B, 13·H) reference trajectory.
        """
        B, H = self.B, self.horizon
        cs = self.current_state                                              # (B, 13)

        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)

        # 1. world-frame desired velocity: v_des = R_base @ v_des_base
        v_des_world = np.einsum('bij,bj->bi', self.R_base,
                                 base_vel_base_des).astype(np.float32)        # (B, 3)

        # 2. clamp xpos/ypos_base_desired to current pos ± 0.1 m
        max_pos_err = 0.1
        np.clip(self.xpos_base_desired,
                cs[:, 3] - max_pos_err, cs[:, 3] + max_pos_err,
                out=self.xpos_base_desired)
        np.clip(self.ypos_base_desired,
                cs[:, 4] - max_pos_err, cs[:, 4] + max_pos_err,
                out=self.ypos_base_desired)
        cur_x = self.xpos_base_desired                                        # (B,)
        cur_y = self.ypos_base_desired                                        # (B,)

        # 3. roll/pitch_init updates with conditional masks (vx>0.2, vy>0.1)
        vx, vy = cs[:, 9], cs[:, 10]
        pitch_mask = np.abs(vx) > 0.2
        if pitch_mask.any():
            self.pitch_init[pitch_mask] += (self.dt
                                              * (-cs[pitch_mask, 1])
                                              / vx[pitch_mask])
        roll_mask = np.abs(vy) > 0.1
        if roll_mask.any():
            self.roll_init[roll_mask] += (self.dt
                                            * (-cs[roll_mask, 0])
                                            / vy[roll_mask])
        np.clip(self.roll_init,  -0.25, 0.25, out=self.roll_init)
        np.clip(self.pitch_init, -0.25, 0.25, out=self.pitch_init)
        roll_comp  = vy * self.roll_init                                      # (B,)
        pitch_comp = vx * self.pitch_init                                     # (B,)

        # 4. build reference (B, 13·H) using strided fancy indexing
        ref = np.zeros((B, 13 * H), dtype=np.float32)
        k_arr = np.arange(H, dtype=np.float32)                                # (H,)
        # yaw[k] = yaw_desired + k·dt·yaw_rate
        ref[:, 2::13]  = self.yaw_desired[:, None] + k_arr[None, :] * self.dt * yaw_turn_rate_des[:, None]
        ref[:, 3::13]  = cur_x[:, None] + k_arr[None, :] * self.dt * v_des_world[:, 0:1]
        ref[:, 4::13]  = cur_y[:, None] + k_arr[None, :] * self.dt * v_des_world[:, 1:2]
        ref[:, 0::13]  = roll_comp[:, None]
        ref[:, 1::13]  = pitch_comp[:, None]
        ref[:, 5::13]  = self.com_height_des
        ref[:, 8::13]  = yaw_turn_rate_des[:, None]
        ref[:, 9::13]  = v_des_world[:, 0:1]
        ref[:, 10::13] = v_des_world[:, 1:2]
        ref[:, 12::13] = -self.gravity
        return ref


# ── Compact (18-dim) parameterization ──────────────────────────────────────

COMPACT_DIM = 18

def compact_to_full(p_compact: torch.Tensor,
                     mpc_cfg: LinearMpcConfig,
                     robot_cfg: RobotConfig,
                     horizon: int = None) -> torch.Tensor:
    """Reconstruct full 285-dim parameter from compact 18-dim parameter.

    Compact layout (18 dim):
        [0:3]   rpy                         (roll, pitch, yaw)
        [3]     xpos_drift   = X_ref[3] - xt[3]   (positive if des ahead of cur)
        [4]     ypos_drift
        [5]     z_drift      = z - z_des
        [6:9]   omega_body                  (ω_x, ω_y, ω_z)
        [9:12]  v_world                     (vx, vy, vz)
        [12:15] cmd_world                   (vx_des, vy_des, wz_des)
        [15:17] (roll_comp, pitch_comp)
        [17]    gait_phase ∈ [0, 1)

    Output full layout (285 dim):
        [0:13]            xt
        [13:13+13H]       X_ref (13 axes × H horizon steps)
        [13+13H:285]      gait_table (4 legs × H horizon steps)
    """
    B = p_compact.shape[0]
    H = horizon if horizon is not None else mpc_cfg.horizon
    dt = 0.05
    g = mpc_cfg.gravity
    z_des = robot_cfg.base_height_des
    dev = p_compact.device
    dtype = p_compact.dtype

    rpy        = p_compact[:, 0:3]
    xpos_drift = p_compact[:, 3]
    ypos_drift = p_compact[:, 4]
    z_drift    = p_compact[:, 5]
    omega      = p_compact[:, 6:9]
    v_world    = p_compact[:, 9:12]
    cmd_vx     = p_compact[:, 12]
    cmd_vy     = p_compact[:, 13]
    cmd_wz     = p_compact[:, 14]
    roll_comp  = p_compact[:, 15]
    pitch_comp = p_compact[:, 16]
    gait_phase = p_compact[:, 17]

    # xt: absolute xpos/ypos drop out of cost; set to 0 (drift carries the info)
    xt = torch.zeros(B, 13, device=dev, dtype=dtype)
    xt[:, 0:3]  = rpy
    xt[:, 5]    = z_des + z_drift
    xt[:, 6:9]  = omega
    xt[:, 9:12] = v_world
    xt[:, 12]   = -g

    k_arr = torch.arange(H, device=dev, dtype=dtype)
    k_dt  = (k_arr * dt).unsqueeze(0)                                    # (1, H)

    # X_ref (B, 13·H)
    X_ref = torch.zeros(B, 13 * H, device=dev, dtype=dtype)
    X_ref[:, 0::13]  = roll_comp.unsqueeze(-1).expand(-1, H)
    X_ref[:, 1::13]  = pitch_comp.unsqueeze(-1).expand(-1, H)
    X_ref[:, 2::13]  = rpy[:, 2:3] + cmd_wz.unsqueeze(-1) * k_dt
    X_ref[:, 3::13]  = xpos_drift.unsqueeze(-1) + cmd_vx.unsqueeze(-1) * k_dt
    X_ref[:, 4::13]  = ypos_drift.unsqueeze(-1) + cmd_vy.unsqueeze(-1) * k_dt
    X_ref[:, 5::13]  = z_des
    # axes 6, 7, 11 stay 0
    X_ref[:, 8::13]  = cmd_wz.unsqueeze(-1).expand(-1, H)
    X_ref[:, 9::13]  = cmd_vx.unsqueeze(-1).expand(-1, H)
    X_ref[:, 10::13] = cmd_vy.unsqueeze(-1).expand(-1, H)
    X_ref[:, 12::13] = -g

    # Gait table (B, 4·H) reconstructed from gait_phase (trot 50% duty, period 0.5s).
    period, duty = 0.5, 0.5
    phases   = (gait_phase.unsqueeze(-1) + k_arr.unsqueeze(0) * dt / period) % 1.0
    stance_A = (phases < duty).to(dtype)        # FL + RR
    stance_B = 1.0 - stance_A                    # FR + RL
    gait_full = torch.zeros(B, 4 * H, device=dev, dtype=dtype)
    gait_full[:, 0::4] = stance_A   # FL
    gait_full[:, 1::4] = stance_B   # FR
    gait_full[:, 2::4] = stance_B   # RL
    gait_full[:, 3::4] = stance_A   # RR

    return torch.cat([xt, X_ref, gait_full], dim=1)


def sample_training_params_compact(N: int,
                                    mpc_cfg: LinearMpcConfig,
                                    robot_cfg: RobotConfig,
                                    seed: int = 42) -> np.ndarray:
    """Sample N compact parameter vectors (N, 18) with closed-loop-like correlations.

    Key change vs IID sampling: v ≈ cmd + noise, ω_z ≈ cmd_wz + noise. In the
    deployed closed loop, the body velocity tracks the command modulo a small
    error; pure-IID samples ignore this correlation and produce (cmd=1.4, v=0)
    type pairs that rarely arise in steady-state walking.

    Mix of three regimes (33% each):
      - "near-tracking":   v close to cmd, drifts/comps small  (steady walking)
      - "transient":       v partially tracks cmd, drifts/comps moderate
      - "diverse":         independent v/cmd, large drifts (training tail)
    """
    rng = np.random.default_rng(seed)
    p = np.zeros((N, COMPACT_DIM), dtype=np.float32)

    # cmd
    p[:, 12] = rng.uniform(-1.6, 1.6, N)              # cmd_vx
    p[:, 13] = rng.uniform(-0.6, 0.6, N)              # cmd_vy
    p[:, 14] = rng.uniform(-1.2, 1.2, N)              # cmd_wz

    # Regime mask: split into 3 thirds
    third = N // 3
    near = slice(0, third)
    trans = slice(third, 2 * third)
    div  = slice(2 * third, N)

    # ── regime 1: near-tracking (steady walking)
    p[near, 9]  = p[near, 12] + rng.normal(0, 0.10, third)   # vx ≈ cmd_vx
    p[near, 10] = p[near, 13] + rng.normal(0, 0.10, third)
    p[near, 8]  = p[near, 14] + rng.normal(0, 0.15, third)   # ω_z ≈ cmd_wz
    p[near, 3]  = rng.normal(0, 0.05, third)                  # small drifts
    p[near, 4]  = rng.normal(0, 0.05, third)
    p[near, 5]  = rng.normal(0, 0.02, third)
    p[near, 0]  = rng.normal(0, 0.04, third)
    p[near, 1]  = rng.normal(0, 0.04, third)

    # ── regime 2: transient (start-up / cmd switch)
    p[trans, 9]  = p[trans, 12] * rng.uniform(0.3, 0.8, third) \
                    + rng.normal(0, 0.30, third)
    p[trans, 10] = p[trans, 13] * rng.uniform(0.3, 0.8, third) \
                    + rng.normal(0, 0.20, third)
    p[trans, 8]  = p[trans, 14] * rng.uniform(0.3, 0.8, third) \
                    + rng.normal(0, 0.30, third)
    p[trans, 3]  = rng.normal(0, 0.15, third)
    p[trans, 4]  = rng.normal(0, 0.15, third)
    p[trans, 5]  = rng.normal(0, 0.04, third)
    p[trans, 0]  = rng.normal(0, 0.08, third)
    p[trans, 1]  = rng.normal(0, 0.08, third)

    # ── regime 3: diverse (training tail / robustness)
    n_div = N - 2 * third
    p[div, 9]   = rng.normal(0, 1.00, n_div)
    p[div, 10]  = rng.normal(0, 0.60, n_div)
    p[div, 8]   = rng.normal(0, 0.80, n_div)
    p[div, 3]   = rng.normal(0, 0.25, n_div)
    p[div, 4]   = rng.normal(0, 0.25, n_div)
    p[div, 5]   = rng.normal(0, 0.06, n_div)
    p[div, 0]   = rng.normal(0, 0.12, n_div)
    p[div, 1]   = rng.normal(0, 0.12, n_div)

    # ── shared (regime-independent)
    p[:, 2]      = rng.uniform(-np.pi, np.pi, N)      # yaw: any
    p[:, 6]      = rng.normal(0, 0.30, N)             # ω_x
    p[:, 7]      = rng.normal(0, 0.30, N)             # ω_y
    p[:, 11]     = rng.normal(0, 0.15, N)             # vz
    p[:, 15]     = rng.normal(0, 0.04, N)             # roll_comp
    p[:, 16]     = rng.normal(0, 0.04, N)             # pitch_comp
    p[:, 17]     = rng.uniform(0, 1, N)               # gait_phase

    # shuffle so batches are mixed
    perm = rng.permutation(N)
    return p[perm]


# ── Training-data sampler (legacy 285-dim full layout) ─────────────────────

def sample_training_params(N: int,
                            cfg: NominalQPConfig,
                            mpc_cfg: LinearMpcConfig,
                            robot_cfg: RobotConfig,
                            seed: int = 42) -> np.ndarray:
    """Sample (x_t, X_ref, gait_table) tuples spanning realistic walking states."""
    H = mpc_cfg.horizon
    rng = np.random.default_rng(seed)
    P = np.zeros((N, cfg.np_dim), dtype=np.float32)

    z_des = robot_cfg.base_height_des
    g     = mpc_cfg.gravity

    for i in range(N):
        # x_t: random near stand pose
        xt = np.zeros(13, dtype=np.float32)
        xt[0:3] = rng.normal(0, [0.05, 0.05, 0.30], 3)             # rpy
        xt[3:5] = rng.normal(0, 0.10, 2)                            # x,y
        xt[5]   = z_des + rng.normal(0, 0.02)                       # z
        xt[6:9] = rng.normal(0, [0.20, 0.20, 0.50], 3)              # ω
        xt[9:11] = rng.normal(0, 0.40, 2)                            # vx,vy
        xt[11]  = rng.normal(0, 0.10)                                # vz
        xt[12]  = -g

        # cmd → reference trajectory
        cmd_vx = rng.uniform(-1.5, 1.5)
        cmd_vy = rng.uniform(-0.5, 0.5)
        cmd_wz = rng.uniform(-1.0, 1.0)

        Xref = np.zeros(13 * H, dtype=np.float32)
        Xref[2] = xt[0:3][2] + 0.0   # yaw_des (start near current yaw)
        Xref[3] = xt[3] + cmd_vx * 0.05
        Xref[4] = xt[4] + cmd_vy * 0.05
        Xref[5::13] = z_des
        Xref[8::13] = cmd_wz
        Xref[9::13] = cmd_vx
        Xref[10::13] = cmd_vy
        Xref[12::13] = -g
        for k in range(1, H):
            Xref[2 + 13*k] = Xref[2 + 13*(k-1)] + 0.05 * cmd_wz
            Xref[3 + 13*k] = Xref[3 + 13*(k-1)] + 0.05 * cmd_vx
            Xref[4 + 13*k] = Xref[4 + 13*(k-1)] + 0.05 * cmd_vy

        # gait_table: trot 50% duty, random phase
        phase_offset = rng.uniform(0, 1)
        gait = np.zeros(4 * H, dtype=np.float32)
        # FL+RR diagonal pair (legs 0, 3) vs FR+RL (legs 1, 2)
        for k in range(H):
            t_in_period = ((k * 0.05 / 0.5) + phase_offset) % 1.0
            stance_A = 1.0 if t_in_period < 0.5 else 0.0
            stance_B = 1.0 - stance_A
            gait[4*k + 0] = stance_A   # FL
            gait[4*k + 1] = stance_B   # FR
            gait[4*k + 2] = stance_B   # RL
            gait[4*k + 3] = stance_A   # RR

        P[i, 0:13]            = xt
        P[i, 13:13 + 13*H]    = Xref
        P[i, 13 + 13*H:]      = gait
    return P
