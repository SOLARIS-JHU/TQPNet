"""Problem base for the paper experiments.

Self-contained: all network classes, loss functions, ADMM routines, and
problem construction are defined here. No dependency on the core package.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Problem config ────────────────────────────────────────────────────────────

@dataclass
class ProblemConfig:
    n: int
    m: int
    np_dim: int
    rho_refine: float = 1.0
    Q0:        torch.Tensor = field(default=None, repr=False)
    A0:        torch.Tensor = field(default=None, repr=False)
    q0:        torch.Tensor = field(default=None, repr=False)
    Cq:        torch.Tensor = field(default=None, repr=False)
    c0_vec:    torch.Tensor = field(default=None, repr=False)
    Cc:        torch.Tensor = field(default=None, repr=False)
    _D_U:      torch.Tensor = field(default=None, repr=False)
    _D_V:      torch.Tensor = field(default=None, repr=False)
    _inv_D_U:  torch.Tensor = field(default=None, repr=False)
    _H_V_chol: torch.Tensor = field(default=None, repr=False)
    _M1:       torch.Tensor = field(default=None, repr=False)   # (Q+ρD_V)^{-1} ρD_V
    _M2:       torch.Tensor = field(default=None, repr=False)   # (Q+ρD_V)^{-1} ρA^T
    _H_inv:    torch.Tensor = field(default=None, repr=False)   # (Q+ρD_V)^{-1}
    _M_beta:   torch.Tensor = field(default=None, repr=False)   # -diag(D_U^{-1}) A


def _chol(M: torch.Tensor) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    return torch.linalg.cholesky(M)


def build_config(
    n, m, np_dim,
    dev=device,
    seed: int = 42,
    rho_refine: float = 1.0,
    density: float = 0.3,
    a_scale: float = 10.0,
    c_offset: float = -1.0,
    signed: bool = False,
) -> ProblemConfig:
    """Build parametric QP with sparse A.

    signed=False (default): A in [0, a_scale]  (positive sparse)
    signed=True:            A in [-a_scale, a_scale]  (mixed-sign sparse)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ProblemConfig(n=n, m=m, np_dim=np_dim, rho_refine=rho_refine)

    L0 = 0.3 * torch.randn(n, n, device=dev)
    cfg.Q0 = L0 @ L0.T / n + 0.1 * torch.eye(n, device=dev)
    cfg.Q0 = 0.5 * (cfg.Q0 + cfg.Q0.T)

    # Uniform column sparsity: each column has exactly k = ceil(m * density) nonzeros.
    # This makes D_V constant across columns, improving low-dim ADMM convergence.
    k_per_col = max(1, int(np.ceil(m * density)))
    mask = torch.zeros(m, n, device=dev)
    for j in range(n):
        rows = torch.randperm(m, device=dev)[:k_per_col]
        mask[rows, j] = 1.0
    # Ensure every row has at least one nonzero
    for i in range(m):
        if mask[i].sum() == 0:
            mask[i, torch.randint(0, n, (1,), device=dev)] = 1.0

    if signed:
        cfg.A0 = mask * a_scale * (-1.0 + 2.0 * torch.rand(m, n, device=dev))
    else:
        cfg.A0 = mask * a_scale * torch.rand(m, n, device=dev)

    cfg.q0     = torch.randn(n, device=dev)
    cfg.Cq     = 0.3 * torch.randn(n, np_dim, device=dev)
    cfg.c0_vec = cfg.A0 @ torch.randn(n, device=dev) + c_offset
    cfg.Cc     = 0.1 * torch.randn(m, np_dim, device=dev)

    cfg._D_U      = (cfg.A0 ** 2).sum(dim=1)
    cfg._D_V      = (cfg.A0.abs() > 0).to(cfg.A0.dtype).sum(dim=0)
    cfg._inv_D_U  = 1.0 / cfg._D_U.clamp(min=1e-12)

    # Pre-compute ADMM-ReLU weight matrices (matmul instead of cholesky_solve)
    H_V = cfg.Q0 + rho_refine * torch.diag(cfg._D_V)
    cfg._H_V_chol = _chol(H_V)  # kept for backward compat
    H_inv = torch.linalg.solve(H_V, torch.eye(n, device=dev))
    cfg._M1 = rho_refine * H_inv @ torch.diag(cfg._D_V)       # n×n
    cfg._M2 = rho_refine * H_inv @ cfg.A0.T                    # n×m
    cfg._H_inv = H_inv                                          # n×n
    cfg._M_beta = -torch.diag(cfg._inv_D_U) @ cfg.A0           # m×n
    return cfg


def build_problem_data(P: torch.Tensor, cfg: ProblemConfig):
    b = P.shape[0]
    return (
        cfg.Q0.unsqueeze(0).expand(b, -1, -1),
        cfg.q0.unsqueeze(0) + P @ cfg.Cq.T,
        cfg.A0.unsqueeze(0).expand(b, -1, -1),
        cfg.c0_vec.unsqueeze(0) + P @ cfg.Cc.T,
    )


# ── KKT residual and loss ────────────────────────────────────────────────────

def kkt_raw_residual(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg):
    Qx  = torch.bmm(Q_b, x_pred.unsqueeze(-1)).squeeze(-1)
    Ax  = torch.bmm(A_b, x_pred.unsqueeze(-1)).squeeze(-1)
    ATb = torch.bmm(A_b.transpose(1, 2), beta_pred.unsqueeze(-1)).squeeze(-1)
    r1  = Qx + q_b - ATb
    r2  = beta_pred - (beta_pred - Ax + c_b).clamp(min=0.0)
    return (r1**2).sum(1) / cfg.n + (r2**2).sum(1) / cfg.m


def admm_informed_loss(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg):
    return torch.log1p(kkt_raw_residual(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg)).mean()


def admm_unrolled_loss(x_pred, beta_pred, Q_b, q_b, A_b, c_b, cfg,
                       K_unroll: int = 5):
    """Unrolled ADMM loss: run K differentiable ADMM steps from (x_pred, beta_pred),
    then compute KKT residual. Gradients flow through all K steps.
    Uses pre-computed weight matrices (matmul, no cholesky_solve)."""
    x    = x_pred
    beta = beta_pred.clamp(min=0)
    dev, dtype = x.device, x.dtype

    # Training uses cholesky_solve for numerical stability in backward pass
    A_t    = A_b.transpose(1, 2)
    D_V    = cfg._D_V.to(dev, dtype).unsqueeze(0)
    inv_DU = cfg._inv_D_U.to(dev, dtype).unsqueeze(0)
    chol   = cfg._H_V_chol.to(dev, dtype).unsqueeze(0)
    rho    = cfg.rho_refine

    for _ in range(K_unroll):
        x_old = x
        rhs   = rho * D_V * x + rho * torch.bmm(A_t, beta.unsqueeze(-1)).squeeze(-1) - q_b
        x     = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        beta  = (beta
                 - inv_DU * torch.bmm(A_b, (2 * x - x_old).unsqueeze(-1)).squeeze(-1)
                 + inv_DU * c_b).clamp(min=0)

    return admm_informed_loss(x, beta, Q_b, q_b, A_b, c_b, cfg)


# ── ADMM-ReLU refinement (inference, no grad) ────────────────────────────────

@torch.no_grad()
def admm_refine(x0, Q_b, q_b, A_b, c_b, cfg: ProblemConfig,
                K: int = 200, tol=None, beta0=None):
    batch, dev, dtype = x0.shape[0], x0.device, x0.dtype
    x    = x0.clone()
    beta = beta0.clamp(min=0).clone() if beta0 is not None else \
           torch.zeros(batch, cfg.m, device=dev, dtype=dtype)

    # Pre-computed weight matrices → matmul instead of cholesky_solve
    M1     = cfg._M1.to(dev, dtype).unsqueeze(0).expand(batch, -1, -1)
    M2     = cfg._M2.to(dev, dtype).unsqueeze(0).expand(batch, -1, -1)
    H_inv  = cfg._H_inv.to(dev, dtype).unsqueeze(0).expand(batch, -1, -1)
    M_beta = cfg._M_beta.to(dev, dtype).unsqueeze(0).expand(batch, -1, -1)
    inv_DU = cfg._inv_D_U.to(dev, dtype).unsqueeze(0)

    # Per-instance bias (q and c vary with parameter p)
    bias_x    = -torch.bmm(H_inv, q_b.unsqueeze(-1)).squeeze(-1)
    bias_beta = inv_DU * c_b

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    step, converged = 0, False

    if tol is not None and kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).mean() < tol:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return x, beta, (time.perf_counter() - t0) * 1000, step, True

    for step in range(1, K + 1):
        x_old = x
        x     = (torch.bmm(M1, x.unsqueeze(-1)).squeeze(-1)
                 + torch.bmm(M2, beta.unsqueeze(-1)).squeeze(-1)
                 + bias_x)
        beta  = (beta
                 + torch.bmm(M_beta, (2 * x - x_old).unsqueeze(-1)).squeeze(-1)
                 + bias_beta).clamp(min=0)
        if tol is not None and kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).mean() < tol:
            converged = True; break

    if torch.cuda.is_available(): torch.cuda.synchronize()
    return x, beta, (time.perf_counter() - t0) * 1000, step, converged


# ── Network architectures ────────────────────────────────────────────────────

def _xavier_init(mod):
    if isinstance(mod, nn.Linear):
        nn.init.xavier_uniform_(mod.weight)
        nn.init.zeros_(mod.bias)


class TQPNet(nn.Module):
    """Unsupervised neural proxy: p -> (x*, beta*)."""
    def __init__(self, np_dim, n, m, W=256, L=4):
        super().__init__()
        self.n, self.m = n, m
        layers = [nn.Linear(np_dim, W), nn.ReLU()]
        for _ in range(L - 1):
            layers += [nn.Linear(W, W), nn.ReLU()]
        self.backbone  = nn.Sequential(*layers)
        self.head_x    = nn.Linear(W, n)
        self.head_beta = nn.Sequential(nn.Linear(W, m), nn.Softplus())
        self.apply(_xavier_init)

    def forward(self, p):
        h = self.backbone(p)
        return self.head_x(h), self.head_beta(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class LQPNet(nn.Module):
    """Supervised baseline: p -> x*."""
    def __init__(self, np_dim, n, W=256, L=4):
        super().__init__()
        layers = [nn.Linear(np_dim, W), nn.ReLU()]
        for _ in range(L - 1):
            layers += [nn.Linear(W, W), nn.ReLU()]
        layers.append(nn.Linear(W, n))
        self.net = nn.Sequential(*layers)
        self.apply(_xavier_init)

    def forward(self, p):
        return self.net(p)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── OSQP reference solver ────────────────────────────────────────────────────

def solve_osqp_reference(Q_np, q_np, A_np, c_np, n):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q_np) + q_np @ x),
                      [A_np @ x >= c_np])
    prob.solve(solver=cp.OSQP, verbose=False, polish=True, eps_abs=1e-7, eps_rel=1e-7)
    if x.value is None:
        return None, float("nan"), prob.status
    obj = float(0.5 * x.value @ Q_np @ x.value + q_np @ x.value)
    return x.value, obj, prob.status


def compute_metrics(x_pred_batch, Q_b, q_b, A_b, c_b, cfg, n_samples=50):
    rel_errors, obj_gaps, violations = [], [], []
    for i in range(min(n_samples, x_pred_batch.shape[0])):
        x_nn = x_pred_batch[i].detach().cpu().numpy()
        if np.isnan(x_nn).any():
            continue
        Q_np = Q_b[i].cpu().numpy()
        q_np = q_b[i].cpu().numpy()
        A_np = A_b[i].cpu().numpy()
        c_np = c_b[i].cpu().numpy()
        violations.append(float(np.maximum(c_np - A_np @ x_nn, 0).max()))
        x_star, obj_star, _ = solve_osqp_reference(Q_np, q_np, A_np, c_np, cfg.n)
        if x_star is None:
            continue
        rel_errors.append(float(np.linalg.norm(x_nn - x_star) ** 2 / (np.linalg.norm(x_star) ** 2 + 1e-10)))
        obj_nn = float(0.5 * x_nn @ Q_np @ x_nn + q_np @ x_nn)
        obj_gaps.append(abs(obj_nn - obj_star) / (abs(obj_star) + 1e-10))

    def stats(values):
        arr = np.array(values) if values else np.array([float("nan")])
        return {"mean": float(np.nanmean(arr)),
                "p15": float(np.nanpercentile(arr, 15)),
                "p85": float(np.nanpercentile(arr, 85))}

    return {"rel_error": stats(rel_errors), "obj_gap": stats(obj_gaps),
            "violation": stats(violations)}


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    "ProblemConfig", "build_config", "build_problem_data",
    "kkt_raw_residual", "admm_informed_loss", "admm_unrolled_loss",
    "admm_refine",
    "TQPNet", "LQPNet",
    "solve_osqp_reference", "compute_metrics",
    "device", "_chol",
]
