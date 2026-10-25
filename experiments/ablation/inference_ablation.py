"""Fig 3(b) + appendix inference-time EXPERIMENT: inference convergence vs refine iterations and wall-clock.

Loads ckpts trained by `experiments/ablation/training_ablation.py` for K_train ∈ {0, 5, 10},
runs each on a fixed 100-QP inference test set, and applies K_refine
ADMM-ReLU steps for K_refine = 0..K_REFINE_MAX.  Records:

  • per-QP KKT residual at every K_refine                      (Fig 3b)
  • cumulative wall-clock time per refine step (batch GPU)     (appendix inference-time figure)

Cold ADMM (no MLP) is also benchmarked up to K_REFINE_COLD steps to anchor
appendix inference-time figure baseline.

Plotting:
    `experiments/ablation/plot_ablation.py` → Fig 3b and the appendix inference-time figure

Data output:
    experiments/runs/ablation/data_{SCALE_TAG}/inference_ablation_K{0,5,10}.npz
        keys: K_refines (K_max+1,), kkt (B, K_max+1), K_train, label
    experiments/runs/ablation/data_{SCALE_TAG}/inference_time_K{0,5,10}.npz
    experiments/runs/ablation/data_{SCALE_TAG}/inference_time_cold.npz
        keys: K_refines, kkt (B, K), t_ms (K,) — cumulative batch wall-clock,
              t_per_qp_ms (K,) = t_ms / B,  K_train, label

Usage:
    conda run -n isaacgym python -m experiments.ablation.inference_ablation
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from experiments.base import (
    TQPNet, build_config, build_problem_data,
    admm_refine, kkt_raw_residual, device,
)
from experiments.config import (
    PROBLEM_SEED, RHO_REFINE, A_DENSITY,
)


# ── Scale config: must match training_ablation.SCALE_TAG ──────────────────
SCALE_TAG = "n10"

_SCALES = {
    "n10":  {"n": 10,  "m": 20,  "np_dim": 4,  "W": 256, "L": 4},
    "n100": {"n": 100, "m": 200, "np_dim": 20, "W": 512, "L": 4},
    "n500": {"n": 500, "m": 500, "np_dim": 50, "W": 512, "L": 4},
}
_KMAX = {"n10": 5000, "n100": 10000, "n500": 50000}

SCALE            = _SCALES[SCALE_TAG]
A_SCALE          = 3.0
C_OFFSET         = 1.0
N_INFER          = 100        # inference test set
SEED_INFER       = 2027       # different from training_ablation's test seed
KS_TRAIN         = [0, 1, 5, 10, 25, 50]
K_REFINE_MAX     = _KMAX[SCALE_TAG]   # bigger at n=100: cold ADMM(ReLU) needs more iters
K_REFINE_COLD    = _KMAX[SCALE_TAG]


RUN_DIR    = Path(__file__).resolve().parent / "runs" / "ablation"
_TAG       = f"{SCALE_TAG}" + ("" if C_OFFSET == 1.0 else f"_c{C_OFFSET:g}")
DATA_DIR   = RUN_DIR / f"data_{_TAG}"
MODEL_DIR  = RUN_DIR / f"models_{_TAG}"
LOG_DIR    = RUN_DIR / "logs"
for d in (DATA_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Tee stdout to log file (filename carries SCALE_TAG)
import sys, datetime
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for x in self.streams:
            x.write(s); x.flush()
    def flush(self):
        for x in self.streams: x.flush()
_log_path = LOG_DIR / f"inference_ablation_{SCALE_TAG}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
_log_file = open(_log_path, "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log] {_log_path}")


def make_inference_set(cfg, n_infer=N_INFER, seed=SEED_INFER):
    """Build inference QP batch + OSQP reference x* (for abs / rel error metrics)."""
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    P = -1.0 + 2.0 * torch.rand(n_infer, cfg.np_dim, generator=gen).to(device)
    Q_b, q_b, A_b, c_b = build_problem_data(P, cfg)
    # OSQP reference (high precision) — used by abs / rel error metrics.
    import osqp
    from scipy import sparse
    Q_np = Q_b.detach().cpu().numpy()
    q_np = q_b.detach().cpu().numpy()
    A_np = A_b.detach().cpu().numpy()
    c_np = c_b.detach().cpu().numpy()
    x_opt = np.zeros((n_infer, cfg.n), dtype=np.float32)
    print(f"  solving OSQP x* on N_INFER={n_infer} for abs/rel-err metrics…", flush=True)
    for i in range(n_infer):
        # The parametric QP form is `Ax ≥ c, β ≥ 0`, so OSQP bounds are [c, +inf].
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(Q_np[i]), q=q_np[i],
                   A=sparse.csc_matrix(A_np[i]),
                   l=c_np[i], u=np.inf * np.ones_like(c_np[i]),
                   verbose=False, polish=True,
                   eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)
        res = prob.solve()
        if res.x is not None and np.all(np.isfinite(res.x)):
            x_opt[i] = res.x.astype(np.float32)
    x_opt_t = torch.tensor(x_opt, device=device)
    return P, Q_b, q_b, A_b, c_b, x_opt_t


@torch.no_grad()
def refine_sweep_per_qp(x0, beta0, Q_b, q_b, A_b, c_b, x_opt_b, cfg, K_max: int):
    """Per-QP convergence at K_refine = 0..K_max.  Returns dict of (B, K+1) arrays:
        kkt : KKT residual
        abs : ‖xᵏ − x⋆‖₂        (needs x_opt_b)
        rel : ‖xᵏ − x⋆‖₂ / ‖x⋆‖₂
    """
    B = x0.shape[0]
    kkt = np.empty((B, K_max + 1), dtype=np.float32)
    abs_err = np.empty((B, K_max + 1), dtype=np.float32)
    x_opt_norms = (x_opt_b.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()  # (B,)
    x = x0.clone()
    beta = beta0.clamp(min=0).clone()
    kkt[:, 0]     = kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).cpu().numpy()
    abs_err[:, 0] = (x - x_opt_b).norm(dim=-1).cpu().numpy()
    for k in range(1, K_max + 1):
        x_new, beta_new, _, _, _ = admm_refine(
            x, Q_b, q_b, A_b, c_b, cfg, K=1, tol=None, beta0=beta)
        x, beta = x_new, beta_new
        kkt[:, k]     = kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).cpu().numpy()
        abs_err[:, k] = (x - x_opt_b).norm(dim=-1).cpu().numpy()
    rel_err = abs_err / x_opt_norms[:, None]
    return {"kkt": kkt, "abs": abs_err, "rel": rel_err}


# ── Timing-aware refine sweep (appendix inference-time figure) ───────────────────────────────────────

def _label(K_train: int) -> str:
    if K_train == 0:
        return "MLP(ReLU) only ($K_t{=}0$)"
    return f"MLP(ReLU) + {K_train}-layer ADMM(ReLU) ($K_t{{=}}{K_train}$)"


@torch.no_grad()
def refine_sweep_with_timing(x0, beta0, Q_b, q_b, A_b, c_b, x_opt_b, cfg, K_max: int,
                              t_offset_s: float = 0.0):
    """Per-QP convergence + cumulative wall-clock.  Returns:
        kkt, abs, rel : (B, K_max+1)
        t_ms          : (K_max+1,)  cumulative batch wall-clock; index 0 is t_offset
    """
    B = x0.shape[0]
    sync = lambda: torch.cuda.synchronize() if x0.is_cuda else None
    kkt = np.empty((B, K_max + 1), dtype=np.float32)
    abs_err = np.empty((B, K_max + 1), dtype=np.float32)
    out_t   = np.empty(K_max + 1, dtype=np.float32)
    x_opt_norms = (x_opt_b.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()
    x = x0.clone()
    beta = beta0.clamp(min=0).clone()
    sync()
    kkt[:, 0]     = kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).cpu().numpy()
    abs_err[:, 0] = (x - x_opt_b).norm(dim=-1).cpu().numpy()
    out_t[0] = t_offset_s * 1000.0
    cum = t_offset_s
    for k in range(1, K_max + 1):
        sync(); t0 = time.perf_counter()
        x_new, beta_new, _, _, _ = admm_refine(
            x, Q_b, q_b, A_b, c_b, cfg, K=1, tol=None, beta0=beta)
        sync(); cum += (time.perf_counter() - t0)
        x, beta = x_new, beta_new
        out_t[k]      = cum * 1000.0
        kkt[:, k]     = kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).cpu().numpy()
        abs_err[:, k] = (x - x_opt_b).norm(dim=-1).cpu().numpy()
    rel_err = abs_err / x_opt_norms[:, None]
    return {"kkt": kkt, "abs": abs_err, "rel": rel_err, "t_ms": out_t}


@torch.no_grad()
def time_mlp_forward(model, P, n_warmup: int = 5) -> float:
    """Median wall-clock (s) of one MLP forward pass on the inference batch."""
    sync = lambda: torch.cuda.synchronize() if P.is_cuda else None
    for _ in range(n_warmup):
        _ = model(P); sync()
    n_runs = 20
    times = []
    for _ in range(n_runs):
        sync(); t0 = time.perf_counter()
        _ = model(P); sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = build_config(SCALE["n"], SCALE["m"], SCALE["np_dim"], device,
                        seed=PROBLEM_SEED, rho_refine=RHO_REFINE,
                        density=A_DENSITY, a_scale=A_SCALE, c_offset=C_OFFSET)
    print(f"Inference ablation — n={SCALE['n']} m={SCALE['m']} "
          f"K_trains={KS_TRAIN} N_infer={N_INFER} K_refine_max={K_REFINE_MAX} "
          f"K_refine_cold={K_REFINE_COLD}")

    P, Q_b, q_b, A_b, c_b, x_opt_b = make_inference_set(cfg)
    B = P.shape[0]

    # ── MLP-based methods: K_t ∈ {0, 5, 10} ──
    for K_train in KS_TRAIN:
        print(f"\n=== K_train = {K_train} ===")
        ckpt = MODEL_DIR / f"tqpnet_K{K_train}.pth"
        if not ckpt.exists():
            print(f"  [skip] {ckpt} not found — run training_ablation.py first")
            continue
        model = TQPNet(cfg.np_dim, cfg.n, cfg.m, SCALE["W"], SCALE["L"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        t_mlp = time_mlp_forward(model, P)
        print(f"  MLP forward time (batch B={B}): {t_mlp*1000:.3f} ms total, "
              f"{t_mlp*1000/B:.4f} ms/QP")

        with torch.no_grad():
            x0, beta0 = model(P)

        # Fig 3(b): per-QP convergence vs refine step
        m_b = refine_sweep_per_qp(x0, beta0, Q_b, q_b, A_b, c_b, x_opt_b, cfg, K_REFINE_MAX)
        K_arr = np.arange(K_REFINE_MAX + 1)
        out_b = DATA_DIR / f"inference_ablation_K{K_train}.npz"
        np.savez(out_b, K_refines=K_arr,
                  kkt=m_b["kkt"], abs=m_b["abs"], rel=m_b["rel"],
                  K_train=K_train, label=_label(K_train))
        print(f"  saved {out_b}  "
              f"kkt @0={np.median(m_b['kkt'][:,0]):.3e} @{K_REFINE_MAX}={np.median(m_b['kkt'][:,-1]):.3e}  "
              f"rel @0={np.median(m_b['rel'][:,0]):.3e} @{K_REFINE_MAX}={np.median(m_b['rel'][:,-1]):.3e}")

        # appendix inference-time figure: timing-aware refine sweep
        m_c = refine_sweep_with_timing(
            x0, beta0, Q_b, q_b, A_b, c_b, x_opt_b, cfg, K_REFINE_MAX,
            t_offset_s=t_mlp)
        out_c = DATA_DIR / f"inference_time_K{K_train}.npz"
        np.savez(out_c, K_refines=K_arr,
                  kkt=m_c["kkt"], abs=m_c["abs"], rel=m_c["rel"],
                  t_ms=m_c["t_ms"], t_per_qp_ms=m_c["t_ms"] / B,
                  K_train=K_train, label=_label(K_train))
        print(f"  saved {out_c}  t/QP @ K=0: {m_c['t_ms'][0]/B:.4f} ms  "
              f"@ K={K_REFINE_MAX}: {m_c['t_ms'][-1]/B:.4f} ms")

    # ── Cold ADMM baseline (no MLP) for appendix inference-time figure ──
    print(f"\n=== Cold ADMM (no MLP)  —  K up to {K_REFINE_COLD} ===")
    x0_cold   = torch.zeros(B, cfg.n, device=device)
    beta0_cold = torch.zeros(B, cfg.m, device=device)
    m_cold = refine_sweep_with_timing(
        x0_cold, beta0_cold, Q_b, q_b, A_b, c_b, x_opt_b, cfg, K_REFINE_COLD,
        t_offset_s=0.0)
    K_arr_cold = np.arange(K_REFINE_COLD + 1)
    out_cold = DATA_DIR / "inference_time_cold.npz"
    np.savez(out_cold, K_refines=K_arr_cold,
              kkt=m_cold["kkt"], abs=m_cold["abs"], rel=m_cold["rel"],
              t_ms=m_cold["t_ms"], t_per_qp_ms=m_cold["t_ms"] / B,
              label="ADMM(ReLU)")
    print(f"  saved {out_cold}  t/QP final: {m_cold['t_ms'][-1]/B:.4f} ms  "
          f"median KKT final: {np.median(m_cold['kkt'][:,-1]):.3e}")

    # ── Full-dim 3-block ADMM baseline (vanilla) on the same test set ──
    # mpQP form `Ax ≥ c` → flip to admm3_full_batched's `Ax ≤ b`: A_use=-A, b_use=-c.
    print(f"\n=== Full-dim ADMM (3-block, vanilla)  —  K up to {K_REFINE_COLD} ===")
    from experiments.admm.admm_convergence import admm3_full_batched
    A_neg = -A_b
    b_neg = -c_b
    K_full = K_REFINE_COLD
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    kkt_f, abs_f = admm3_full_batched(Q_b, q_b, A_neg, b_neg, x_opt_b, K_full, rho=cfg.rho_refine)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    rel_f = abs_f / x_opt_b.norm(dim=-1, keepdim=True).cpu().numpy().clip(min=1e-12)
    # Prepend K=0 (initial) values so shape matches K_refines length
    init_kkt = kkt_f[:, :1] * 0 + 1.0   # placeholder; recompute properly below
    K_arr_full = np.arange(K_full + 1)
    # Recompute K=0 KKT/abs/rel manually for x0=0
    kkt_full_full = np.concatenate([init_kkt, kkt_f], axis=1)
    abs_full_full = np.concatenate([x_opt_b.norm(dim=-1, keepdim=True).cpu().numpy(), abs_f], axis=1)
    rel_full_full = abs_full_full / x_opt_b.norm(dim=-1, keepdim=True).cpu().numpy().clip(min=1e-12)
    # Per-step timing (uniform across steps for batched ADMM)
    t_ms = np.linspace(0, total_s * 1000, K_full + 1)
    out_full = DATA_DIR / "inference_time_full.npz"
    np.savez(out_full, K_refines=K_arr_full,
              kkt=kkt_full_full, abs=abs_full_full, rel=rel_full_full,
              t_ms=t_ms, t_per_qp_ms=t_ms / B,
              label="Full-dim ADMM (3-block)")
    print(f"  saved {out_full}  total {total_s:.1f}s  "
          f"median rel @K={K_full}: {np.median(rel_full_full[:,-1]):.3e}")


if __name__ == "__main__":
    main()
