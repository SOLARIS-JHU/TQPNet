"""Fig 2 EXPERIMENT: convergence of full-dim 3-block ADMM vs low-dim 2-block ADMM(ReLU).

Runs 100 random QPs at three constraint magnitudes (`A_scale ∈ {2, 4, 6}`),
batched on GPU via PyTorch.  Records **three** convergence metrics per
iteration so plotting (`experiments/admm/plot_admm_convergence.py`) can switch between
them without re-running the experiment:

  • `kkt`  — KKT residual            ‖r₁‖²/n + ‖r₂‖²/m  (intrinsic)
              r₁ = Qx + d + Aᵀβ,  r₂ = β − max(0, β + Ax − b)
  • `abs`  — absolute error          ‖xᵏ − x⋆‖₂        (matches main_v1.m)
  • `rel`  — relative error          ‖xᵏ − x⋆‖₂ / ‖x⋆‖₂

`x⋆` is computed once per QP by OSQP at eps=1e-10.
β is the unscaled dual: β = ρ·y for 3-block ADMM, β directly for 2-block.

QP form (per instance, matches `experiments/admm/main_v1.m`):
    Q = GᵀG + 1e-6·I,  G ~ 𝒩(0, I)
    d ~ 𝒰(-1, 1)
    A = A_scale·sprand(m, n, density),  entries ~ 𝒰(0, 1)
    b = A_scale·𝒰(0, 1)

Data output (`experiments/runs/admm_convergence/data/admm_convergence_a{2,4,6}.npz`):
    keys:
      kkt3, kkt2 : (B, K)   KKT residual
      abs3, abs2 : (B, K)   absolute error
      rel3, rel2 : (B, K)   relative error
      x_star_norms : (B,)   ‖x⋆‖₂ per QP (for reference)
      A_scale, n, m, density, K, rho, seed

Usage:
    python -m experiments.admm.admm_convergence
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse
import torch


SCALE_TAG = "n500m1000"  # paper Fig 2: n=500, m=1000, density 0.1; "n10"/"n100"/"n500" match the ablation scales
_SCALES   = {"n10": (10, 20), "n100": (100, 200), "n500": (500, 500), "n500m1000": (500, 1000)}
# Per-scale (np_dim, density) matching the corresponding training_ablation setup.
_NP_DIM   = {"n10": 4,  "n100": 20, "n500": 50, "n500m1000": 50}
_DENSITY  = {"n10": 0.3, "n100": 0.2, "n500": 0.1, "n500m1000": 0.1}
DATA_DIR  = (Path(__file__).resolve().parent
             / "runs" / "admm_convergence" / f"data_{SCALE_TAG}")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── QP generation + reference solve ──────────────────────────────────────────

def gen_qps_with_xopt(num_qp: int, n: int, m: int, density: float,
                      A_scale: float, seed: int):
    """Generate `num_qp` QPs and solve each to high precision via OSQP."""
    rng = np.random.default_rng(seed)
    n_nonzero = int(round(m * n * density))

    Qs, ds, As, bs, xopts = [], [], [], [], []
    print(f"  generating {num_qp} QPs (n={n}, m={m}, density={density}, "
          f"A_scale={A_scale}) and solving x* via OSQP…", flush=True)
    t0 = time.time()
    for k in range(num_qp):
        if (k + 1) % 25 == 0:
            print(f"    QP {k+1}/{num_qp}  ({time.time()-t0:.1f}s)", flush=True)
        G = rng.standard_normal((n, n))
        Q = G.T @ G + 1e-6 * np.eye(n)
        d = (2 * rng.random(n) - 1)
        A_dense = np.zeros((m, n))
        flat_idx = rng.choice(m * n, n_nonzero, replace=False)
        A_dense.flat[flat_idx] = rng.random(n_nonzero)
        A = A_scale * A_dense
        b = A_scale * rng.random(m)
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(Q), q=d,
                   A=sparse.csc_matrix(A),
                   l=-np.inf * np.ones(m), u=b,
                   verbose=False, polish=True,
                   eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)
        res = prob.solve()
        if res.x is None or not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"OSQP failed on QP {k}: status={res.info.status}")
        Qs.append(Q.astype(np.float32))
        ds.append(d.astype(np.float32))
        As.append(A.astype(np.float32))
        bs.append(b.astype(np.float32))
        xopts.append(res.x.astype(np.float32))
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return (np.stack(Qs), np.stack(ds),
            np.stack(As), np.stack(bs), np.stack(xopts))


# ── Metrics (batched) ────────────────────────────────────────────────────────

def _kkt_batched(Q_b, d_b, A_b, b_b, x, beta):
    """KKT² per QP.  Returns (B,) tensor."""
    n = x.shape[1]; m = b_b.shape[1]
    Qx  = torch.einsum('bij,bj->bi', Q_b, x)
    Ax  = torch.einsum('bij,bj->bi', A_b, x)
    ATb = torch.einsum('bij,bi->bj', A_b, beta)
    r1  = Qx + d_b + ATb
    r2  = beta - (beta + Ax - b_b).clamp(min=0)
    return (r1 ** 2).sum(-1) / n + (r2 ** 2).sum(-1) / m


def _abs_err_batched(x, x_opt_b):
    """‖xᵏ − x⋆‖₂ per QP."""
    return (x - x_opt_b).norm(dim=-1)


# ── Batched ADMMs — record three metrics per iter ────────────────────────────

@torch.no_grad()
def admm3_full_batched(Q_b, d_b, A_b, b_b, x_opt_b, K: int, rho: float = 1.0):
    """Standard 3-block ADMM.  Returns (kkt, abs_err) of shape (B, K)."""
    B, n = d_b.shape
    m = b_b.shape[1]
    dev, dtype = Q_b.device, Q_b.dtype

    AtA = A_b.transpose(-1, -2) @ A_b
    K3 = Q_b + rho * AtA
    chol3 = torch.linalg.cholesky(0.5 * (K3 + K3.transpose(-1, -2)))

    x = torch.zeros(B, n, device=dev, dtype=dtype)
    z = torch.zeros(B, m, device=dev, dtype=dtype)
    y = torch.zeros(B, m, device=dev, dtype=dtype)
    kkt = torch.empty(B, K, device=dev, dtype=dtype)
    err = torch.empty(B, K, device=dev, dtype=dtype)
    for it in range(K):
        rhs = (-d_b + torch.einsum('bij,bi->bj', A_b, z - y)).unsqueeze(-1)
        x = torch.cholesky_solve(rhs, chol3).squeeze(-1)
        ax = torch.einsum('bij,bj->bi', A_b, x)
        z = torch.minimum(b_b, ax + y)
        y = y + ax - z
        beta = rho * y
        kkt[:, it] = _kkt_batched(Q_b, d_b, A_b, b_b, x, beta)
        err[:, it] = _abs_err_batched(x, x_opt_b)
    return kkt.cpu().numpy(), err.cpu().numpy()


@torch.no_grad()
def admm2_relu_batched(Q_b, d_b, A_b, b_b, x_opt_b, K: int, rho: float = 1.0):
    """Low-dim 2-block ADMM(ReLU).  Returns (kkt, abs_err) of shape (B, K)."""
    B, n = d_b.shape
    m = b_b.shape[1]
    dev, dtype = Q_b.device, Q_b.dtype

    DU = (A_b ** 2).sum(dim=-1).clamp_min(1e-12)
    invDU = 1.0 / DU
    DV = (A_b.abs() > 1e-12).to(dtype).sum(dim=-2).clamp_min(1e-12)

    K2 = Q_b + rho * torch.diag_embed(DV)
    chol2 = torch.linalg.cholesky(0.5 * (K2 + K2.transpose(-1, -2)))

    x = torch.zeros(B, n, device=dev, dtype=dtype)
    beta = torch.maximum(torch.zeros_like(b_b), -invDU * b_b)
    kkt = torch.empty(B, K, device=dev, dtype=dtype)
    err = torch.empty(B, K, device=dev, dtype=dtype)

    for it in range(K):
        rhs = (rho * DV * x
                - rho * torch.einsum('bji,bj->bi', A_b, beta)
                - d_b).unsqueeze(-1)
        x_next = torch.cholesky_solve(rhs, chol2).squeeze(-1)
        ax_relax = torch.einsum('bij,bj->bi', A_b, 2 * x_next - x)
        beta = torch.maximum(torch.zeros_like(beta),
                              beta + invDU * ax_relax - invDU * b_b)
        x = x_next
        kkt[:, it] = _kkt_batched(Q_b, d_b, A_b, b_b, x, beta)
        err[:, it] = _abs_err_batched(x, x_opt_b)
    return kkt.cpu().numpy(), err.cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

def gen_mpqp_with_xopt(cfg, num_qp: int, seed: int):
    """Generate `num_qp` mpQP instances by sampling p ~ U[-1,1]^np_dim, then
    OSQP-solve each for high-precision x*."""
    import osqp
    from scipy import sparse
    from experiments.base import build_problem_data
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    P = (-1.0 + 2.0 * torch.rand(num_qp, cfg.np_dim, generator=gen)).to(cfg.Q0.device)
    Q_b, q_b, A_b, c_b = build_problem_data(P, cfg)
    Q_np = Q_b.detach().cpu().numpy(); q_np = q_b.detach().cpu().numpy()
    A_np = A_b.detach().cpu().numpy(); c_np = c_b.detach().cpu().numpy()
    n = cfg.n
    x_opt = np.zeros((num_qp, n), dtype=np.float32)
    print(f"  solving OSQP x* on {num_qp} mpQP instances…", flush=True)
    t0 = time.time()
    for i in range(num_qp):
        if (i + 1) % 25 == 0:
            print(f"    QP {i+1}/{num_qp}  ({time.time()-t0:.1f}s)", flush=True)
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(Q_np[i]), q=q_np[i],
                   A=sparse.csc_matrix(A_np[i]),
                   l=c_np[i], u=np.inf * np.ones_like(c_np[i]),
                   verbose=False, polish=True,
                   eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)
        res = prob.solve()
        if res.x is None or not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"OSQP failed on QP {i}: status={res.info.status}")
        x_opt[i] = res.x.astype(np.float32)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return Q_np.astype(np.float32), q_np.astype(np.float32), \
           A_np.astype(np.float32), c_np.astype(np.float32), x_opt


def main():
    n, m = _SCALES[SCALE_TAG]
    density = _DENSITY[SCALE_TAG]
    rho = 1.0
    num_qp = 100
    K_iter = 10_000
    seed_base = 2025
    scales = [2.0, 4.0, 6.0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Generic random QPs (NOT mpQP) — Fig 2 tests the algorithmic claim of low-dim
    # ADMM(ReLU) on independent random QP instances. Fig 3 uses mpQP family for
    # the learning claim.
    print(f"ADMM convergence on generic random QPs — n={n}, m={m}, "
          f"density={density}, num_qp={num_qp}, K_iter={K_iter}, "
          f"scales={scales}, device={device}")

    for A_scale in scales:
        print(f"\n=== A_scale = {A_scale} ===")
        Q, d, A, b, x_opt = gen_qps_with_xopt(
            num_qp, n, m, density, A_scale,
            seed=int(seed_base + 1000 * A_scale))

        Q_t   = torch.tensor(Q,    device=device, dtype=dtype)
        d_t   = torch.tensor(d,    device=device, dtype=dtype)
        A_t   = torch.tensor(A,    device=device, dtype=dtype)
        b_t   = torch.tensor(b,    device=device, dtype=dtype)
        xopt_t = torch.tensor(x_opt, device=device, dtype=dtype)
        x_star_norms = np.linalg.norm(x_opt, axis=1).astype(np.float32)

        print("  3-block ADMM…", flush=True); t0 = time.time()
        kkt3, abs3 = admm3_full_batched(Q_t, d_t, A_t, b_t, xopt_t, K_iter, rho=rho)
        print(f"    {time.time()-t0:.1f}s")
        print("  2-block low-dim ADMM(ReLU)…", flush=True); t0 = time.time()
        kkt2, abs2 = admm2_relu_batched(Q_t, d_t, A_t, b_t, xopt_t, K_iter, rho=rho)
        print(f"    {time.time()-t0:.1f}s")

        denom = np.maximum(x_star_norms, 1e-12)[:, None]                       # (B, 1)
        rel3 = abs3 / denom
        rel2 = abs2 / denom

        out = DATA_DIR / f"admm_convergence_a{int(A_scale)}.npz"
        np.savez(out,
                  kkt3=kkt3, kkt2=kkt2,
                  abs3=abs3, abs2=abs2,
                  rel3=rel3, rel2=rel2,
                  x_star_norms=x_star_norms,
                  A_scale=A_scale,
                  n=n, m=m, density=density, K=K_iter, rho=rho, seed=seed_base)
        print(f"  saved {out}  "
              f"final medians: kkt2={np.median(kkt2[:, -1]):.2e}  "
              f"abs2={np.median(abs2[:, -1]):.2e}  "
              f"rel2={np.median(rel2[:, -1]):.2e}")


if __name__ == "__main__":
    main()
