"""Generate the 4 paper figures for the A1 batch-MPC experiments.

Style follows experiments/plot_paper.py (publication palette + small figure sizes).
Outputs go to experiments/pympc-quadruped/runs/paper_figures/.

Plot 1: solver wall-clock vs batch size N
Plot 2: QP solution error vs ADMM-ReLU iteration K (fixed batch)
Plot 3: cost gap + constraint violation vs K (fixed batch)
Plot 4: closed-loop velocity tracking (vx_mean vs t) — needs CSVs from
        scripts/run_velocity_tracking.py (separate Isaac Gym runs).

  conda run -n isaacgym python scripts/make_paper_plots.py
"""
from __future__ import annotations

import os, sys, time, json
from pathlib import Path

# Import torch / numpy / matplotlib BEFORE adding utils/ to sys.path — utils/cmd.py
# shadows stdlib's cmd module which torch.import → pdb needs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'linear_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from linear_mpc_configs import LinearMpcConfig
from robot_configs import A1Config
from tqpnet_mpc import (
    build_nominal_qp_config, build_problem_data,
    sample_training_params, sample_training_params_compact,
    compact_to_full, COMPACT_DIM,
    admm_refine, kkt_raw_residual,
    NormalizedTQPNet, ScaledTQPNet,
)
from qpth.qp import QPFunction


# ── Style (mirrors plot_paper.py) ─────────────────────────────────────────────

COLORS = {
    "tqp_K0":   "lightcoral",   # raw NN output (no refine)
    "tqp_K10":  "crimson",
    "tqp_K20":  "firebrick",
    "cold":     "royalblue",
    "qpth":     "darkorange",
    "osqp":     "gray",
}
mk_kw = dict(markeredgecolor="white", markeredgewidth=0.4)


def _setup():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "grid.alpha": 0.15,
    })


def _save(fig, out_dir, name):
    for sfx in (".png", ".pdf"):
        fig.savefig(out_dir / f"{name}{sfx}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {name}{{.png,.pdf}}")


# ── Solver wrappers ───────────────────────────────────────────────────────────

def _osqp_per_robot(Q_b, q_b, A_b, c_b):
    from scipy import sparse
    import osqp
    B, n = q_b.shape
    out = np.zeros((B, n), dtype=np.float32)
    for i in range(B):
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(Q_b[i].cpu().numpy()),
                   q=q_b[i].cpu().numpy(),
                   A=sparse.csc_matrix(A_b[i].cpu().numpy()),
                   l=-np.inf * np.ones(A_b.shape[1]),
                   u=c_b[i].cpu().numpy(),
                   verbose=False, polish=True, eps_abs=1e-8, eps_rel=1e-8,
                   max_iter=50000)
        res = prob.solve()
        out[i] = res.x if res.x is not None else 0.0
    return torch.from_numpy(out).to(q_b.device)


def _qpth_batch(Q_b, q_b, A_b, c_b):
    e = torch.zeros(0, Q_b.shape[-1], device=Q_b.device)
    bv = torch.zeros(0, device=Q_b.device)
    Q_pd = Q_b + 1e-6 * torch.eye(Q_b.shape[-1], device=Q_b.device).unsqueeze(0)
    return QPFunction(eps=1e-10, verbose=-1, check_Q_spd=False, maxIter=200)(
        Q_pd.double(), q_b.double(), A_b.double(), c_b.double(), e, bv).float()


def _cold_admm(Q_b, q_b, A_b, c_b, cfg, K):
    x0 = torch.zeros(Q_b.shape[0], cfg.n, device=Q_b.device)
    x_ref, b_ref = admm_refine(x0, Q_b, q_b, A_b, c_b, cfg, K=K)
    return x_ref, b_ref


def _tqpnet_warm(model, P_nn, Q_b, q_b, A_b, c_b, cfg, K):
    with torch.no_grad():
        x0, b0 = model(P_nn)
        if K == 0:
            return x0, b0.clamp(min=0)
        x_ref, b_ref = admm_refine(x0, Q_b, q_b, A_b, c_b, cfg, K=K, beta0=b0)
    return x_ref, b_ref


def _time_call(fn, repeats=3):
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)
    return float(np.median(times))


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _build_setup(device='cuda'):
    cfg = build_nominal_qp_config(LinearMpcConfig, A1Config, device=device)
    weights = (Path(__file__).resolve().parent.parent
               / 'linear_mpc' / 'tqpnet_weights' / 'tqpnet_a1.pth')
    sd = torch.load(weights, map_location=device)
    np_dim = sd['input_mean'].numel() if 'input_mean' in sd else sd['backbone.0.weight'].shape[1]
    is_compact = (np_dim == COMPACT_DIM)
    cls = NormalizedTQPNet if 'input_mean' in sd else ScaledTQPNet
    if cls is NormalizedTQPNet:
        model = cls(np_dim, cfg.n, cfg.m, W=512, L=4).to(device)
    else:
        model = cls(np_dim, cfg.n, cfg.m, W=512, L=4, x_scale=30.0).to(device)
    model.load_state_dict(sd); model.eval()
    print(f"  loaded TQPNet ({model.count_params()/1e6:.2f}M params, "
          f"np_dim={np_dim}, {'compact' if is_compact else 'full'})")
    return cfg, model, is_compact


def _sample_problem(N, cfg, is_compact, seed):
    if is_compact:
        Pc_np = sample_training_params_compact(N, LinearMpcConfig, A1Config, seed=seed)
        Pc = torch.tensor(Pc_np, dtype=torch.float32, device=cfg.Q0.device)
        Pf = compact_to_full(Pc, LinearMpcConfig, A1Config)
        return Pc, Pf
    P_np = sample_training_params(N, cfg, LinearMpcConfig, A1Config, seed=seed)
    P = torch.tensor(P_np, dtype=torch.float32, device=cfg.Q0.device)
    return P, P


# ── Plot 1: solver throughput vs N ───────────────────────────────────────────

def plot_batch_qp_timing(out_dir, cfg, model, is_compact):
    print("\n[Plot 1] solver wall-clock vs N")
    Ns = [1, 10, 100, 500, 1000, 2000, 4000, 6000, 8000]
    K_TQP = [0, 10, 20]
    K_COLD = 50

    rows = {f"tqp_K{K}": [] for K in K_TQP}
    rows["cold_K50"] = []
    rows["qpth"]     = []

    for N in Ns:
        Pc, Pf = _sample_problem(N, cfg, is_compact, seed=N)
        P_nn = Pc if is_compact else Pf
        Q_b, q_b, A_b, c_b = build_problem_data(Pf, cfg)

        # Warmup each solver
        for K in K_TQP:
            _tqpnet_warm(model, P_nn, Q_b, q_b, A_b, c_b, cfg, K)
        _cold_admm(Q_b, q_b, A_b, c_b, cfg, K_COLD)
        try:
            _qpth_batch(Q_b, q_b, A_b, c_b)
        except Exception as e:
            print(f"    qpth warmup failed at N={N}: {e}")

        for K in K_TQP:
            ms = _time_call(lambda K=K: _tqpnet_warm(model, P_nn, Q_b, q_b, A_b, c_b, cfg, K))
            rows[f"tqp_K{K}"].append(ms)
        rows["cold_K50"].append(_time_call(lambda: _cold_admm(Q_b, q_b, A_b, c_b, cfg, K_COLD)))
        try:
            rows["qpth"].append(_time_call(lambda: _qpth_batch(Q_b, q_b, A_b, c_b)))
        except Exception:
            rows["qpth"].append(np.nan)
        print(f"  N={N:>4}: " + "  ".join(
            f"{k}={rows[k][-1]:.2f}ms" for k in rows))

    _setup()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    Ns_arr = np.array(Ns)
    ax.loglog(Ns_arr, rows["qpth"],     "D--", color=COLORS["qpth"], ms=5, **mk_kw,
              label="qpth (PDIPM)")
    ax.loglog(Ns_arr, rows["cold_K50"], "o:",  color=COLORS["cold"], lw=1.2, ms=5, **mk_kw,
              label="ADMM-ReLU cold ($K{=}50$)")
    ax.loglog(Ns_arr, rows["tqp_K0"],   "v-",  color=COLORS["tqp_K0"], ms=5, **mk_kw,
              label="TQPNet ($K{=}0$)")
    ax.loglog(Ns_arr, rows["tqp_K10"],  "s-",  color=COLORS["tqp_K10"], ms=5, lw=2.0, **mk_kw,
              label="TQPNet ($K{=}10$)")
    ax.loglog(Ns_arr, rows["tqp_K20"],  "^-",  color=COLORS["tqp_K20"], ms=5, **mk_kw,
              label="TQPNet ($K{=}20$)")
    ax.set_xlabel("Batch size $N$ (robots)")
    ax.set_ylabel("Solve Latency (ms)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.grid(True, which="both", alpha=0.15)
    _save(fig, out_dir, "batch_qp_timing")
    np.savez(out_dir / 'batch_qp_timing.npz', Ns=Ns_arr, **{k: np.array(v) for k, v in rows.items()})


# ── Plots 2 + 3: error / cost / violation vs K ───────────────────────────────

def plot_qp_convergence(out_dir, cfg, model, is_compact, B=100, K_max=100):
    print(f"\n[Plot 2+3] QP convergence vs K (B={B}, K up to {K_max})")
    Pc, Pf = _sample_problem(B, cfg, is_compact, seed=2026)
    P_nn = Pc if is_compact else Pf
    Q_b, q_b, A_b, c_b = build_problem_data(Pf, cfg)

    # Reference solution (OSQP @ 1e-8) for "true" optimum
    print("  computing OSQP reference (eps=1e-8)…")
    x_star = _osqp_per_robot(Q_b, q_b, A_b, c_b)

    def _cost(x):
        Qx = torch.bmm(Q_b, x.unsqueeze(-1)).squeeze(-1)
        return (0.5 * (Qx * x).sum(1) + (q_b * x).sum(1)).median().item()

    def _viol(x):
        Ax = torch.bmm(A_b, x.unsqueeze(-1)).squeeze(-1)
        return (Ax - c_b).clamp(min=0).norm(dim=1).median().item()

    cost_star = _cost(x_star)
    viol_star = _viol(x_star)
    norm_star = x_star.norm(dim=1) + 1e-12

    # qpth: single-shot solve (PDIPM converges internally) — show as horizontal lines
    print("  qpth solve…")
    try:
        x_qpth = _qpth_batch(Q_b, q_b, A_b, c_b)
        qpth_err  = ((x_qpth - x_star).norm(dim=1) / norm_star).median().item()
        qpth_cost = _cost(x_qpth)
        qpth_viol = _viol(x_qpth)
    except Exception as e:
        print(f"    qpth failed: {e}")
        qpth_err = qpth_cost = qpth_viol = float('nan')

    Ks = list(range(0, K_max + 1, 5))
    if 1 not in Ks: Ks = [0, 1] + [k for k in Ks if k > 1]
    Ks = sorted(set(Ks))
    print(f"  Ks = {Ks}")

    err_tqp, err_cold = [], []
    cost_tqp, cost_cold = [], []
    viol_tqp, viol_cold = [], []

    for K in Ks:
        # TQPNet warm + K-step refine (K=0 → raw NN output)
        x_tqp, _ = _tqpnet_warm(model, P_nn, Q_b, q_b, A_b, c_b, cfg, K)
        err_tqp.append(((x_tqp - x_star).norm(dim=1) / norm_star).median().item())
        cost_tqp.append(_cost(x_tqp))
        viol_tqp.append(_viol(x_tqp))

        # Cold ADMM-ReLU (K=0 → zero start, undefined solution; we report large)
        if K == 0:
            x_cold = torch.zeros_like(x_star)
        else:
            x_cold, _ = _cold_admm(Q_b, q_b, A_b, c_b, cfg, K)
        err_cold.append(((x_cold - x_star).norm(dim=1) / norm_star).median().item())
        cost_cold.append(_cost(x_cold))
        viol_cold.append(_viol(x_cold))

    Ks_arr = np.array(Ks)
    eps = 1e-12
    rel_cost_tqp  = np.maximum(np.abs(np.array(cost_tqp)  - cost_star) / max(abs(cost_star), eps), eps)
    rel_cost_cold = np.maximum(np.abs(np.array(cost_cold) - cost_star) / max(abs(cost_star), eps), eps)

    np.savez(out_dir / 'qp_convergence.npz',
             Ks=Ks_arr, err_tqp=err_tqp, err_cold=err_cold,
             cost_tqp=cost_tqp, cost_cold=cost_cold,
             viol_tqp=viol_tqp, viol_cold=viol_cold,
             cost_star=cost_star, viol_star=viol_star,
             qpth_err=qpth_err, qpth_cost=qpth_cost, qpth_viol=qpth_viol)

    _setup()
    # Plot 2 — solution error
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.semilogy(Ks_arr, np.maximum(err_cold, eps), "o:", color=COLORS["cold"], lw=1.2,
                ms=5, **mk_kw, label="ADMM-ReLU cold")
    ax.semilogy(Ks_arr, np.maximum(err_tqp,  eps), "s-", color=COLORS["tqp_K10"], lw=2.0,
                ms=5, **mk_kw, label="TQPNet + ADMM-ReLU")
    ax.axhline(max(qpth_err, eps), color=COLORS["qpth"], ls="--", lw=1.5,
               label=f"qpth ({qpth_err:.1e})")
    ax.set_xlabel("$K$ (refine steps)")
    ax.set_ylabel(r"$\|x_K - x^{\star}_{\rm OSQP}\| / \|x^{\star}_{\rm OSQP}\|$")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(True, which="both", alpha=0.15)
    _save(fig, out_dir, "qp_solution_error_vs_K")

    # Plot 3 — cost + violation
    fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax_c.semilogy(Ks_arr, rel_cost_cold, "o:", color=COLORS["cold"], lw=1.2, ms=5, **mk_kw,
                   label="ADMM-ReLU cold")
    ax_c.semilogy(Ks_arr, rel_cost_tqp,  "s-", color=COLORS["tqp_K10"], lw=2.0, ms=5, **mk_kw,
                   label="TQPNet + ADMM-ReLU")
    ax_c.axhline(max(abs(qpth_cost - cost_star) / max(abs(cost_star), eps), eps),
                  color=COLORS["qpth"], ls="--", lw=1.5, label="qpth")
    ax_c.set_xlabel("$K$ (refine steps)")
    ax_c.set_ylabel(r"$|J_K - J^{\star}_{\rm OSQP}| / |J^{\star}_{\rm OSQP}|$")
    ax_c.set_title("(a) Cost gap")
    ax_c.legend(loc="upper right", fontsize=6.5)
    ax_c.grid(True, which="both", alpha=0.15)

    ax_v.semilogy(Ks_arr, np.maximum(viol_cold, eps), "o:", color=COLORS["cold"], lw=1.2,
                   ms=5, **mk_kw, label="ADMM-ReLU cold")
    ax_v.semilogy(Ks_arr, np.maximum(viol_tqp,  eps), "s-", color=COLORS["tqp_K10"], lw=2.0,
                   ms=5, **mk_kw, label="TQPNet + ADMM-ReLU")
    ax_v.axhline(max(qpth_viol, eps), color=COLORS["qpth"], ls="--", lw=1.5,
                  label=f"qpth")
    ax_v.axhline(max(viol_star, eps), color=COLORS["osqp"], ls=":", lw=1.0,
                  label="OSQP ref")
    ax_v.set_xlabel("$K$ (refine steps)")
    ax_v.set_ylabel(r"$\|\max(0, A x - c)\|_2$")
    ax_v.set_title("(b) Constraint violation")
    ax_v.legend(loc="upper right", fontsize=6.5)
    ax_v.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    _save(fig, out_dir, "qp_cost_violation_vs_K")


# ── Plot 4: closed-loop velocity tracking ────────────────────────────────────

def plot_velocity_tracking(out_dir, log_dir):
    """Read per-method vx CSV logs (produced by run_velocity_tracking.py) and plot."""
    print("\n[Plot 4] velocity tracking vs time")
    if not log_dir.exists():
        print(f"  [skip] log dir {log_dir} not found — run scripts/run_velocity_tracking.py first")
        return

    methods = [
        ("qpth.csv",         "qpth (PDIPM)",                COLORS["qpth"],   "--"),
        ("cold_K50.csv",     "ADMM-ReLU cold ($K{=}50$)",   COLORS["cold"],   ":"),
        ("tqpnet_K0.csv",    "TQPNet ($K{=}0$)",            COLORS["tqp_K0"], "-"),
        ("tqpnet_K10.csv",   "TQPNet ($K{=}10$)",           COLORS["tqp_K10"],"-"),
        ("tqpnet_K20.csv",   "TQPNet ($K{=}20$)",           COLORS["tqp_K20"],"-"),
    ]
    cmd_vx = None
    found = []
    for fname, label, color, ls in methods:
        path = log_dir / fname
        if not path.exists():
            print(f"  [miss] {fname}")
            continue
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        # Columns: t_iter, t_seconds, vx_mean, cmd_vx
        found.append((data, label, color, ls))
        if cmd_vx is None and data.shape[1] >= 4:
            cmd_vx = data[:, 3]

    if not found:
        print("  [skip] no logs to plot")
        return

    _setup()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for data, label, color, ls in found:
        t = data[:, 1]                  # seconds
        vx = data[:, 2]
        lw = 2.0 if "K=10" in label or "K{=}10" in label else 1.5
        ax.plot(t, vx, ls, color=color, lw=lw, label=label)
    if cmd_vx is not None:
        t_cmd = found[0][0][:, 1]
        ax.plot(t_cmd, cmd_vx, "k:", lw=1.0, alpha=0.6, label="command")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\bar v_x$ (m/s)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(True, alpha=0.15)
    _save(fig, out_dir, "velocity_tracking")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = (Path(__file__).resolve().parent.parent / 'runs' / 'paper_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, model, is_compact = _build_setup(device)

    plot_batch_qp_timing(out_dir, cfg, model, is_compact)
    plot_qp_convergence(out_dir, cfg, model, is_compact, B=100, K_max=100)

    log_dir = out_dir.parent / 'velocity_tracking_logs'
    plot_velocity_tracking(out_dir, log_dir)

    print(f"\nAll figures saved to {out_dir}")
