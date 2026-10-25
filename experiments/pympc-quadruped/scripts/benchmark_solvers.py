"""Benchmark OSQP / qpth / TQPNet+ADMM-K / cold-ADMM-K on the same QP family.

Outputs:
    runs/figures/solver_timing_vs_N.{png,pdf}
    runs/figures/kkt_vs_K.{png,pdf}
    runs/figures/training_loss.{png,pdf}
    runs/data/benchmark.npz

No Isaac Gym, no physics — pure QP solve throughput / accuracy.
"""
from __future__ import annotations
import os
import re
import sys
import time
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from linear_mpc_configs import LinearMpcConfig
from robot_configs import A1Config
from tqpnet_mpc import (
    build_nominal_qp_config, build_problem_data, sample_training_params,
    sample_training_params_compact, compact_to_full, COMPACT_DIM,
    admm_refine, kkt_raw_residual, ScaledTQPNet, NormalizedTQPNet,
)
from linear_mpc_configs import LinearMpcConfig
from robot_configs import A1Config
from qpth.qp import QPFunction


# ──────────────────────────────────────────────────────────────────────────────

def osqp_per_robot(Q_b, q_b, A_b, c_b):
    """OSQP per-instance solve, returns (B, n) solutions."""
    from scipy import sparse
    import osqp
    B, n = q_b.shape
    out = np.zeros((B, n), dtype=np.float32)
    for i in range(B):
        P = sparse.csc_matrix(Q_b[i].cpu().numpy())
        q = q_b[i].cpu().numpy()
        A = sparse.csc_matrix(A_b[i].cpu().numpy())
        l = -np.inf * np.ones(A.shape[0])
        u = c_b[i].cpu().numpy()
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, polish=False, eps_rel=1e-3, eps_abs=1e-3)
        res = prob.solve()
        out[i] = res.x if res.x is not None else 0.0
    return torch.from_numpy(out).to(q_b.device)


def qpth_batch(Q_b, q_b, A_b, c_b):
    A_eq = torch.zeros(0, Q_b.shape[-1], device=Q_b.device)
    b_eq = torch.zeros(0, device=Q_b.device)
    Q_pd = Q_b + 1e-6 * torch.eye(Q_b.shape[-1], device=Q_b.device).unsqueeze(0)
    return QPFunction(verbose=-1, check_Q_spd=False)(Q_pd, q_b, A_b, c_b, A_eq, b_eq)


def cold_admm(Q_b, q_b, A_b, c_b, cfg, K):
    x0 = torch.zeros(Q_b.shape[0], cfg.n, device=Q_b.device)
    x_ref, _ = admm_refine(x0, Q_b, q_b, A_b, c_b, cfg, K=K)
    return x_ref


def tqpnet_warm_admm(Q_b, q_b, A_b, c_b, model, p_nn, cfg, K):
    with torch.no_grad():
        x_w, b_w = model(p_nn)
        x_ref, _ = admm_refine(x_w, Q_b, q_b, A_b, c_b, cfg, K=K, beta0=b_w)
    return x_ref


# ──────────────────────────────────────────────────────────────────────────────

def time_call(fn, n_repeat=3):
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    return float(np.median(times))


# ──────────────────────────────────────────────────────────────────────────────

def load_training_log(path: Path):
    """Parse 'ep=  10  loss=1.0385e+01' from training log → (eps, losses)."""
    eps, losses = [], []
    if not path.exists():
        return None, None
    pat = re.compile(r"ep=\s*(\d+)\s+loss=([\d\.eE\+\-]+)")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            eps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return np.asarray(eps), np.asarray(losses)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda')
    out_root = Path(__file__).resolve().parent.parent / 'runs'
    fig_dir = out_root / 'figures'
    data_dir = out_root / 'data'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Building nominal QP config (A1 + LinearMpc)...')
    cfg = build_nominal_qp_config(LinearMpcConfig, A1Config, device=device)
    print(f'  n={cfg.n}  m={cfg.m}  np_dim={cfg.np_dim}')

    weights_path = (Path(__file__).resolve().parent.parent
                    / 'linear_mpc' / 'tqpnet_weights' / 'tqpnet_a1.pth')
    if weights_path.exists():
        sd = torch.load(weights_path, map_location=device)
        # Auto-detect input dim from checkpoint
        if 'input_mean' in sd:
            np_dim_nn = sd['input_mean'].numel()
        elif 'backbone.0.weight' in sd:
            np_dim_nn = sd['backbone.0.weight'].shape[1]
        else:
            np_dim_nn = cfg.np_dim
        is_compact = (np_dim_nn == COMPACT_DIM)

        if 'input_mean' in sd:
            model = NormalizedTQPNet(np_dim_nn, cfg.n, cfg.m, W=512, L=4).to(device)
            kind = 'NormalizedTQPNet'
        else:
            model = ScaledTQPNet(np_dim_nn, cfg.n, cfg.m, W=512, L=4, x_scale=30.0).to(device)
            kind = 'ScaledTQPNet'
        model.load_state_dict(sd)
        model.eval()
        have_tqpnet = True
        print(f'  TQPNet loaded ({model.count_params()/1e6:.2f} M params, {kind}, '
              f'np_dim={np_dim_nn}, {"compact" if is_compact else "full"})')
    else:
        have_tqpnet = False
        is_compact = False
        print(f'  TQPNet weights missing at {weights_path}; skipping NN benchmark')

    # ─── 1. Solver timing vs batch size ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('1) Solver wall-clock vs batch size N')
    Ns = [1, 4, 16, 64, 128, 256]
    K_admm = 10
    K_admm_cold = 50
    rows = []
    for N in Ns:
        print(f'  N={N}')
        if have_tqpnet and is_compact:
            P_compact_np = sample_training_params_compact(N, LinearMpcConfig, A1Config, seed=N)
            P_compact = torch.tensor(P_compact_np, dtype=torch.float32, device=device)
            P_full = compact_to_full(P_compact, LinearMpcConfig, A1Config)
            P_nn = P_compact
        else:
            P_np = sample_training_params(N, cfg, LinearMpcConfig, A1Config, seed=N)
            P_full = torch.tensor(P_np, dtype=torch.float32, device=device)
            P_nn = P_full
        Q_b, q_b, A_b, c_b = build_problem_data(P_full, cfg)

        # Warmup all paths
        if N <= 64:
            osqp_per_robot(Q_b, q_b, A_b, c_b)
        qpth_batch(Q_b, q_b, A_b, c_b)
        cold_admm(Q_b, q_b, A_b, c_b, cfg, K_admm_cold)
        if have_tqpnet:
            tqpnet_warm_admm(Q_b, q_b, A_b, c_b, model, P_nn, cfg, K_admm)

        ms_osqp = time_call(lambda: osqp_per_robot(Q_b, q_b, A_b, c_b)) if N <= 64 else None
        ms_qpth = time_call(lambda: qpth_batch(Q_b, q_b, A_b, c_b))
        ms_cold = time_call(lambda: cold_admm(Q_b, q_b, A_b, c_b, cfg, K_admm_cold))
        ms_tqp  = time_call(lambda: tqpnet_warm_admm(Q_b, q_b, A_b, c_b, model, P_nn, cfg, K_admm)) \
                  if have_tqpnet else None
        rows.append({'N': N, 'osqp': ms_osqp, 'qpth': ms_qpth,
                      'cold_admm_K50': ms_cold,
                      'tqpnet_warm_admm_K10': ms_tqp})
        print(f'    OSQP={ms_osqp}  qpth={ms_qpth:.1f}  cold-ADMM-K{K_admm_cold}={ms_cold:.1f}  '
              f'TQPNet+ADMM-K{K_admm}={ms_tqp}')

    # ─── 2. KKT vs K (warm vs cold) ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('2) KKT residual vs ADMM iteration K (warm vs cold)')
    N_kkt = 64
    if have_tqpnet and is_compact:
        P_compact_np = sample_training_params_compact(N_kkt, LinearMpcConfig, A1Config, seed=999)
        P_compact = torch.tensor(P_compact_np, dtype=torch.float32, device=device)
        P = compact_to_full(P_compact, LinearMpcConfig, A1Config)
        P_nn = P_compact
    else:
        P_np = sample_training_params(N_kkt, cfg, LinearMpcConfig, A1Config, seed=999)
        P = torch.tensor(P_np, dtype=torch.float32, device=device)
        P_nn = P
    Q_b, q_b, A_b, c_b = build_problem_data(P, cfg)
    Ks = [0, 5, 10, 20, 50, 100, 200]
    kkt_cold = []
    kkt_warm = []
    for K in Ks:
        x_c = (torch.zeros(N_kkt, cfg.n, device=device) if K == 0
               else cold_admm(Q_b, q_b, A_b, c_b, cfg, K))
        beta_c = torch.zeros(N_kkt, cfg.m, device=device)
        # For K>0 we need β. Re-run with returned β.
        if K > 0:
            xc, bc = admm_refine(torch.zeros(N_kkt, cfg.n, device=device),
                                  Q_b, q_b, A_b, c_b, cfg, K=K)
            kkt_c = kkt_raw_residual(xc, bc, Q_b, q_b, A_b, c_b, cfg).median().item()
        else:
            kkt_c = kkt_raw_residual(x_c, beta_c, Q_b, q_b, A_b, c_b, cfg).median().item()
        kkt_cold.append(kkt_c)

        if have_tqpnet:
            with torch.no_grad():
                xw, bw = model(P_nn)
                if K == 0:
                    kkt_w = kkt_raw_residual(xw, bw.clamp(min=0), Q_b, q_b, A_b, c_b, cfg).median().item()
                else:
                    xr, br = admm_refine(xw, Q_b, q_b, A_b, c_b, cfg, K=K, beta0=bw)
                    kkt_w = kkt_raw_residual(xr, br, Q_b, q_b, A_b, c_b, cfg).median().item()
            kkt_warm.append(kkt_w)
        print(f'  K={K:3d}  cold={kkt_cold[-1]:.3e}  '
              f'TQPNet-warm={kkt_warm[-1] if kkt_warm else "n/a"}')

    # ─── 3. OSQP reference: cost + constraint violation vs K ───────────────
    print('\n' + '=' * 60)
    print('3) Cost & constraint violation vs K (vs OSQP optimum)')
    x_star = osqp_per_robot(Q_b, q_b, A_b, c_b)

    def _cost(x):
        # 1/2 x^T Q x + q^T x   (mean over batch)
        Qx = torch.bmm(Q_b, x.unsqueeze(-1)).squeeze(-1)
        return (0.5 * (Qx * x).sum(1) + (q_b * x).sum(1)).median().item()

    def _violation(x):
        # max(0, A x - c).norm(inf-or-2 per row)   (median over batch of L2)
        Ax = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
        v  = (Ax - c_b).clamp(min=0)
        return v.norm(dim=1).median().item()

    cost_osqp      = _cost(x_star)
    viol_osqp      = _violation(x_star)
    cost_cold, cost_warm = [], []
    viol_cold, viol_warm = [], []
    for K in Ks:
        if K == 0:
            xc = torch.zeros(N_kkt, cfg.n, device=device)
        else:
            xc, _ = admm_refine(torch.zeros(N_kkt, cfg.n, device=device),
                                 Q_b, q_b, A_b, c_b, cfg, K=K)
        cost_cold.append(_cost(xc));     viol_cold.append(_violation(xc))
        if have_tqpnet:
            with torch.no_grad():
                xw, bw = model(P_nn)
                if K == 0:
                    xref = xw
                else:
                    xref, _ = admm_refine(xw, Q_b, q_b, A_b, c_b, cfg, K=K, beta0=bw)
            cost_warm.append(_cost(xref));   viol_warm.append(_violation(xref))
        print(f'  K={K:3d}  cold cost={cost_cold[-1]:.2e}  viol={viol_cold[-1]:.2e}'
              + (f'   TQPNet cost={cost_warm[-1]:.2e}  viol={viol_warm[-1]:.2e}'
                 if cost_warm else ''))
    print(f'  ── OSQP reference: cost={cost_osqp:.2e}  viol={viol_osqp:.2e}')

    # ─── Save raw + plot ───────────────────────────────────────────────────
    np.savez(data_dir / 'benchmark.npz',
             Ns=np.array([r['N'] for r in rows]),
             ms_osqp=np.array([r['osqp'] if r['osqp'] is not None else np.nan for r in rows]),
             ms_qpth=np.array([r['qpth'] for r in rows]),
             ms_cold=np.array([r['cold_admm_K50'] for r in rows]),
             ms_tqp=np.array([r['tqpnet_warm_admm_K10'] if r['tqpnet_warm_admm_K10'] is not None
                              else np.nan for r in rows]),
             Ks=np.array(Ks),
             kkt_cold=np.array(kkt_cold),
             kkt_warm=np.array(kkt_warm if kkt_warm else [np.nan]*len(Ks)),
             cost_osqp=cost_osqp, viol_osqp=viol_osqp,
             cost_cold=np.array(cost_cold),
             cost_warm=np.array(cost_warm if cost_warm else [np.nan]*len(Ks)),
             viol_cold=np.array(viol_cold),
             viol_warm=np.array(viol_warm if viol_warm else [np.nan]*len(Ks)))
    print(f'\nSaved {data_dir / "benchmark.npz"}')

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10})

    # Fig 1: Solver wall-clock vs N
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    Ns_arr = np.array([r['N'] for r in rows])
    osqp_arr = np.array([r['osqp'] if r['osqp'] is not None else np.nan for r in rows])
    qpth_arr = np.array([r['qpth'] for r in rows])
    cold_arr = np.array([r['cold_admm_K50'] for r in rows])
    tqp_arr  = np.array([r['tqpnet_warm_admm_K10'] if r['tqpnet_warm_admm_K10'] is not None
                          else np.nan for r in rows])
    ax.loglog(Ns_arr, osqp_arr, 'o-', label='OSQP (per-robot)',          color='darkorange')
    ax.loglog(Ns_arr, qpth_arr, 's-', label='qpth (batch)',              color='royalblue')
    ax.loglog(Ns_arr, cold_arr, '^-', label=f'cold ADMM K={K_admm_cold}', color='gray')
    if have_tqpnet:
        ax.loglog(Ns_arr, tqp_arr, 'D-', label=f'TQPNet+ADMM K={K_admm}',  color='crimson', lw=2.0)
    ax.set_xlabel('Batch size N (robots)')
    ax.set_ylabel('Wall-clock (ms)')
    ax.set_title('Solver throughput on SRBD-MPC QP')
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    for sfx in ('.png', '.pdf'):
        fig.savefig(fig_dir / f'solver_timing_vs_N{sfx}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved solver_timing_vs_N.{{png,pdf}}')

    # Fig 2: KKT residual vs K
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.semilogy(Ks, kkt_cold, '^-', label='cold (x=β=0)', color='gray')
    if have_tqpnet:
        ax.semilogy(Ks, kkt_warm, 'D-', label='TQPNet warm', color='crimson', lw=2.0)
    ax.set_xlabel('ADMM-ReLU iteration K')
    ax.set_ylabel('KKT residual (median, raw)')
    ax.set_title('Convergence: cold vs TQPNet warm start')
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for sfx in ('.png', '.pdf'):
        fig.savefig(fig_dir / f'kkt_vs_K{sfx}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved kkt_vs_K.{{png,pdf}}')

    # Fig 3: Cost & constraint violation vs K
    fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    # (a) Relative cost gap |J_K − J*_OSQP| / |J*_OSQP|, log scale.
    denom = max(abs(cost_osqp), 1e-9)
    rel_cost_cold = np.maximum(np.abs(np.array(cost_cold) - cost_osqp) / denom, 1e-10)
    rel_cost_warm = (np.maximum(np.abs(np.array(cost_warm) - cost_osqp) / denom, 1e-10)
                     if have_tqpnet else None)
    ax_c.semilogy(Ks, rel_cost_cold, '^-', color='gray', label='cold ADMM')
    if have_tqpnet:
        ax_c.semilogy(Ks, rel_cost_warm, 'D-', color='crimson', lw=2.0,
                      label='TQPNet+ADMM')
    ax_c.set_xlabel('ADMM-ReLU iteration K')
    ax_c.set_ylabel(r'$|J_K - J^{\star}_{\rm OSQP}|\,/\,|J^{\star}_{\rm OSQP}|$')
    ax_c.set_title('(a) Relative cost gap vs K')
    ax_c.legend(framealpha=0.9, fontsize=8)
    ax_c.grid(True, which='both', alpha=0.3)

    # (b) Constraint violation
    ax_v.axhline(max(viol_osqp, 1e-10), color='darkorange', ls='--', lw=1.5,
                 label=f'OSQP violation = {viol_osqp:.2e}')
    ax_v.semilogy(Ks, np.maximum(viol_cold, 1e-10), '^-', color='gray', label='cold ADMM')
    if have_tqpnet:
        ax_v.semilogy(Ks, np.maximum(viol_warm, 1e-10), 'D-', color='crimson', lw=2.0,
                      label='TQPNet+ADMM')
    ax_v.set_xlabel('ADMM-ReLU iteration K')
    ax_v.set_ylabel(r'Constraint violation $\|\max(0, A x - c)\|_2$')
    ax_v.set_title('(b) Inequality violation vs K')
    ax_v.legend(framealpha=0.9, fontsize=8)
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    for sfx in ('.png', '.pdf'):
        fig.savefig(fig_dir / f'cost_violation_vs_K{sfx}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved cost_violation_vs_K.{{png,pdf}}')

    # Remove obsolete rel_err_vs_K figures if they exist
    for sfx in ('.png', '.pdf'):
        old = fig_dir / f'rel_err_vs_K{sfx}'
        if old.exists():
            old.unlink()

    # Fig 4: Training loss curve
    log_path = (Path(__file__).resolve().parent.parent
                / 'linear_mpc' / 'tqpnet_weights' / 'train_tqpnet_a1.log')
    eps, losses = load_training_log(log_path)
    if eps is not None and len(eps):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(eps, losses, '-', color='crimson', lw=1.6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('admm_unrolled_loss = log(1 + KKT)')
        ax.set_title('TQPNet training loss')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for sfx in ('.png', '.pdf'):
            fig.savefig(fig_dir / f'training_loss{sfx}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved training_loss.{{png,pdf}}  ({len(eps)} epochs)')

    print('\nAll figures + data saved.')


if __name__ == '__main__':
    main()
