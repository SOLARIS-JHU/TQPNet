"""Experiment 3: UAV MPC with TQPNet  (N_H=20, time-aware parametrisation).

Fix vs v1: r_qp(t) is time-varying.  We decompose it into its exact 5-component
Fourier basis  [1, sin(ωt), cos(ωt), sin(2ωt), cos(2ωt)]  and extend the QP
parameter from  p = x0 (13-dim)  to  p = [x0, sin(ωt), cos(ωt), sin(2ωt), cos(2ωt)]
(17-dim).  The TQPNet model now sees the current reference phase, eliminating the
open-loop drift that caused the v1 infer trajectory to diverge.

Three methods compared (all at N_H=20, n=80):
  1. TQPNet infer only  – fast (~0.5 ms/step), approximate
  2. TQPNet + K refine  – accurate (K ADMM steps from TQPNet warm start)
  3. OSQP (N_H=20)      – accurate reference solver

Output: experiments/runs/uav_mpc/
  - figures/uav_trajectory.{png,pdf}
  - uav_results.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import UAV_DIR, ensure_dirs, setup_logger, subdirs, write_json
from experiments.config import (
    ADMM_REFINE_KMAX, ADMM_REFINE_TOL,
    BATCH_SIZE, LOSS_THRESHOLD, LOG_EVERY_EPOCH, LR,
    MAX_EPOCHS, WEIGHT_DECAY,
)
from experiments.base import (
    ProblemConfig, TQPNet, admm_informed_loss, admm_unrolled_loss, admm_refine,
    build_problem_data, device, kkt_raw_residual, _chol,
)
from experiments.uav.uav_physics import (
    N_S, N_U, DT, LEMN_Z0, LEMN_A, LEMN_W,
    DU_MIN, DU_MAX,
    build_linear_model, condense_mpc_matrices, make_ref_seq,
    lemn_point, lemn_velocity, qp_from_x0,
)

N_H      = 25
OUT_DIR  = UAV_DIR
N_TRAIN  = 8000
N_REFINE = 50
T_SIM    = 600

# Extended parameter dimension: x0 (13) + [sin(wt), cos(wt), sin(2wt), cos(2wt)] (4)
NP_DIM_V2 = N_S + 4   # 17


# ─── Fourier decomposition of r_qp(t) ────────────────────────────────────────

def _fourier_rqp(Ad, Bd):
    """Fit r_qp(t) = coefs @ [1, sin(wt), cos(wt), sin(2wt), cos(2wt)].

    Returns coefs of shape (5, n_qp=200).  Error is ~1e-12 (exact).
    """
    w     = LEMN_W
    T_per = 2.0 * np.pi / w
    n_pts = 32
    t_vals = np.linspace(0, T_per, n_pts, endpoint=False)
    R = np.array([condense_mpc_matrices(Ad, Bd, make_ref_seq(t, N_H), N_H)[2]
                  for t in t_vals])           # (n_pts, 200)
    Phi = np.column_stack([
        np.ones(n_pts),
        np.sin(w * t_vals), np.cos(w * t_vals),
        np.sin(2 * w * t_vals), np.cos(2 * w * t_vals),
    ])                                        # (n_pts, 5)
    coefs, _, _, _ = np.linalg.lstsq(Phi, R, rcond=None)  # (5, 200)
    return coefs


def _trig_features(t: float) -> np.ndarray:
    """Return [sin(wt), cos(wt), sin(2wt), cos(2wt)]."""
    w = LEMN_W
    return np.array([np.sin(w * t), np.cos(w * t),
                     np.sin(2 * w * t), np.cos(2 * w * t)], dtype=np.float32)


def _trig_features_batch(t_vals: np.ndarray) -> np.ndarray:
    w = LEMN_W
    return np.column_stack([
        np.sin(w * t_vals), np.cos(w * t_vals),
        np.sin(2 * w * t_vals), np.cos(2 * w * t_vals),
    ]).astype(np.float32)


# ─── Build UAV parametric QP config (v2, 17-dim param) ───────────────────────

def build_uav_cfg(rho_refine: float = 1.0) -> ProblemConfig:
    """ProblemConfig for the N_H=20 UAV MPC condensed QP.

    Parameter p = [x0 (13), sin(wt), cos(wt), sin(2wt), cos(2wt)] (17-dim).

    q(p) = q0 + p @ Cq.T   where Cq = [M_qp | col_rs | col_rc | col_rs2 | col_rc2]
    c(p) = c0 + p @ Cc.T   where Cc[:, :13] = -M_st (same as before, trig cols = 0)
    """
    Ad, Bd = build_linear_model()

    # Condensed matrices at t=0 (for A_u, A_st, M_st, off which don't depend on t)
    ref0 = make_ref_seq(0, N_H)
    md0  = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    Q_qp, M_qp = md0[0], md0[1]
    A_u, c_u   = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]

    n      = N_H * N_U          # 100  (N_H=25)
    m_ctrl = A_u.shape[0]       # 200
    m_st   = A_st.shape[0]      # 100
    m      = m_ctrl + m_st      # 300

    # Fourier decomposition of r_qp(t)
    fourier = _fourier_rqp(Ad, Bd)  # (5, 200): [r_const, rs, rc, rs2, rc2]
    r_const = fourier[0]             # (200,)
    rs, rc, rs2, rc2 = fourier[1], fourier[2], fourier[3], fourier[4]

    cfg        = ProblemConfig(n=n, m=m, np_dim=NP_DIM_V2, rho_refine=rho_refine)
    cfg.Q0     = torch.tensor(Q_qp + 1e-9 * np.eye(n), dtype=torch.float32, device=device)
    cfg.Q0     = 0.5 * (cfg.Q0 + cfg.Q0.T)

    A_full = np.vstack([A_u, A_st])
    cfg.A0 = torch.tensor(A_full, dtype=torch.float32, device=device)

    # q(p) = r_const + x0 @ M_qp.T + s1*rs + c1*rc + s2*rs2 + c2*rc2
    cfg.q0  = torch.tensor(r_const, dtype=torch.float32, device=device)
    # Cq shape: (n, np_dim) = (200, 17)
    Cq_np   = np.column_stack([M_qp,                        # (200, 13)
                                rs.reshape(-1, 1),           # (200, 1)
                                rc.reshape(-1, 1),
                                rs2.reshape(-1, 1),
                                rc2.reshape(-1, 1)])         # → (200, 17)
    cfg.Cq  = torch.tensor(Cq_np, dtype=torch.float32, device=device)

    # c(p): control bounds fixed; state constraints depend on x0 (not t)
    c0      = np.hstack([c_u, off])
    Cc_np   = np.zeros((m, NP_DIM_V2), dtype=np.float64)
    Cc_np[m_ctrl:, :N_S] = -M_st   # trig columns stay 0
    cfg.c0_vec = torch.tensor(c0,    dtype=torch.float32, device=device)
    cfg.Cc     = torch.tensor(Cc_np, dtype=torch.float32, device=device)

    cfg._D_U      = (cfg.A0 ** 2).sum(dim=1)
    cfg._D_V      = (cfg.A0.abs() > 1e-12).to(cfg.A0.dtype).sum(dim=0)
    cfg._inv_D_U  = 1.0 / cfg._D_U.clamp(min=1e-12)
    H_V           = cfg.Q0 + rho_refine * torch.diag(cfg._D_V)
    cfg._H_V_chol = _chol(H_V)

    # Pre-compute ADMM-ReLU weight matrices for matmul inference
    H_inv      = torch.linalg.solve(H_V, torch.eye(n, device=device))
    cfg._M1     = rho_refine * H_inv @ torch.diag(cfg._D_V)
    cfg._M2     = rho_refine * H_inv @ cfg.A0.T
    cfg._H_inv  = H_inv
    cfg._M_beta = -torch.diag(cfg._inv_D_U) @ cfg.A0
    return cfg


# ─── Training data ────────────────────────────────────────────────────────────

def sample_training_params(n_samples: int, sigma: float = 0.3) -> np.ndarray:
    """Sample p = [x0 (perturbed from lemniscate), sin(wt), cos(wt), sin(2wt), cos(2wt)]."""
    rng    = np.random.default_rng(42)
    T_per  = 2.0 * np.pi / LEMN_W
    t_vals = rng.uniform(0, T_per, n_samples)
    params = []
    for t in t_vals:
        s        = np.zeros(N_S)
        s[3:6]   = lemn_point(t)
        s[9:12]  = lemn_velocity(t)   # include correct velocity reference
        s        = s + sigma * rng.standard_normal(N_S)
        p        = np.concatenate([s, _trig_features(t)])   # (17,)
        params.append(p)
    return np.array(params, dtype=np.float32)


# ─── TQPNet training ──────────────────────────────────────────────────────────

def train_tqpnet_uav(cfg: ProblemConfig, model_path: Path,
                     logger, force_train: bool = False) -> TQPNet:
    if not force_train and model_path.exists():
        logger.info("Loading cached TQPNet from %s", model_path)
        model = TQPNet(NP_DIM_V2, cfg.n, cfg.m, W=512, L=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    logger.info("Training TQPNet | np_dim=%d n=%d m=%d N_H=%d N_TRAIN=%d",
                NP_DIM_V2, cfg.n, cfg.m, N_H, N_TRAIN)
    model   = TQPNet(NP_DIM_V2, cfg.n, cfg.m, W=512, L=4).to(device)
    logger.info("params=%.3fM", model.count_params() / 1e6)
    P_train = torch.tensor(sample_training_params(N_TRAIN, sigma=0.3), device=device)
    loader  = DataLoader(TensorDataset(P_train), batch_size=BATCH_SIZE, shuffle=True)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for (p_b,) in loader:
            Q_b, q_b, A_b, c_b = build_problem_data(p_b, cfg)
            opt.zero_grad()
            x_p, beta_p = model(p_b)
            loss = admm_unrolled_loss(x_p, beta_p, Q_b, q_b, A_b, c_b, cfg,
                                     K_unroll=30)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step(); ep_loss += loss.item() * p_b.shape[0]
        sched.step(); ep_loss /= N_TRAIN
        if ep % LOG_EVERY_EPOCH == 0 or ep == MAX_EPOCHS:
            logger.info("ep=%4d loss=%.4e", ep, ep_loss)
        if ep_loss < LOSS_THRESHOLD:
            logger.info("converged ep=%d loss=%.3e", ep, ep_loss); break
    logger.info("Training done | %.1fs", time.time() - t0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    model.eval()
    return model


# ─── OSQP solver ─────────────────────────────────────────────────────────────

def solve_osqp(Q_qp, q_qp, A_ineq, c_ineq,
               eps_abs: float = 1e-2, eps_rel: float = 1e-2,
               polish: bool = False):
    z    = cp.Variable(Q_qp.shape[0])
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(z, Q_qp) + q_qp @ z),
        [A_ineq @ z >= c_ineq])
    prob.solve(solver=cp.OSQP, verbose=False,
               warm_starting=True,
               eps_abs=eps_abs, eps_rel=eps_rel, polish=polish)
    return z.value


# ─── Simulation loop ──────────────────────────────────────────────────────────

def run_sim(method: str, cfg: ProblemConfig, model,
            Ad, Bd, t_sim: int, logger) -> dict:
    traj          = np.zeros((t_sim + 1, N_S))
    traj[0, 3:6]  = lemn_point(0)
    traj[0, 9:12] = lemn_velocity(0)

    ref0         = make_ref_seq(0, N_H)
    md0          = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    Q_qp         = md0[0]
    A_u, c_u     = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]
    A_ineq       = np.vstack([A_u, A_st])

    Q_torch = cfg.Q0.unsqueeze(0)
    A_torch = cfg.A0.unsqueeze(0)

    err, z_errs    = np.zeros(t_sim), np.zeros(t_sim)
    total_ms       = 0.0
    ref_steps_list = []

    # Raw OSQP setup (P, A constant across timesteps) — set up once when method=="osqp"
    osqp_prob = None
    if method == "osqp":
        import osqp
        from scipy import sparse as sp
        n_dec, m_con = Q_qp.shape[0], A_ineq.shape[0]
        osqp_prob = osqp.OSQP()
        osqp_prob.setup(P=sp.csc_matrix(Q_qp), q=np.zeros(n_dec),
                        A=sp.csc_matrix(A_ineq),
                        l=-np.inf * np.ones(m_con), u=np.inf * np.ones(m_con),
                        eps_abs=1e-3, eps_rel=1e-3, polish=False,
                        warm_start=True, verbose=False)

    for t in range(t_sim):
        # Per-step QP: current-time cost vector and constraint RHS
        ref_t             = make_ref_seq(t, N_H)
        md_t              = condense_mpc_matrices(Ad, Bd, ref_t, N_H)
        M_qp_t, r_qp_t   = md_t[1], md_t[2]
        q_qp_t, c_ineq_t = qp_from_x0(M_qp_t, r_qp_t, M_st, c_u, off, traj[t], A_u)

        t0 = time.perf_counter()

        if method == "osqp":
            osqp_prob.update(q=q_qp_t.astype(np.float64),
                              l=c_ineq_t.astype(np.float64))
            res = osqp_prob.solve()
            sol = res.x if (res.info.status_val == 1 and res.x is not None
                             and not np.isnan(res.x).any()) else None
            du  = sol[:N_U] if sol is not None else np.zeros(N_U)
            ref_steps_list.append(int(res.info.iter))

        else:
            # Build 17-dim parameter: [x0, sin(wt), cos(wt), sin(2wt), cos(2wt)]
            trig = _trig_features(float(t))
            p17  = np.concatenate([traj[t].astype(np.float32), trig])
            p_t  = torch.tensor(p17, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                u_warm, beta_warm = model(p_t)

            if method == "tqpnet_infer":
                du = u_warm[0, :N_U].cpu().numpy()
                ref_steps_list.append(0)

            else:  # tqpnet_refine
                q_torch = torch.tensor(q_qp_t, dtype=torch.float32, device=device).unsqueeze(0)
                c_torch = torch.tensor(c_ineq_t, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    x_ref, _, _, n_steps, _ = admm_refine(
                        u_warm, Q_torch, q_torch, A_torch, c_torch, cfg,
                        K=N_REFINE, tol=None, beta0=beta_warm)
                du = x_ref[0, :N_U].cpu().numpy()
                ref_steps_list.append(n_steps)

        total_ms       += (time.perf_counter() - t0) * 1000
        traj[t + 1]     = Ad @ traj[t] + Bd @ du
        traj[t + 1, 12]-= DT * LEMN_Z0

        ref_pos    = lemn_point(t)
        err[t]     = np.linalg.norm(traj[t + 1, 3:6] - ref_pos)
        z_errs[t]  = traj[t + 1, 5] - LEMN_Z0

        if (t + 1) % 100 == 0 or t + 1 == t_sim:
            logger.info("[%s] step=%04d err=%.4fm z_err=%+.5fm ms/step=%.3f",
                        method, t + 1, err[t], z_errs[t], total_ms / (t + 1))

    return {
        "method":           method, "t_sim": t_sim, "N_H": N_H,
        "mean_rmse":        float(err.mean()),
        "max_error":        float(err.max()),
        "mean_abs_z_err":   float(np.abs(z_errs).mean()),
        "ms_per_step":      float(total_ms / t_sim),
        "median_ref_steps": float(np.median(ref_steps_list)),
        "traj": traj, "err": err,
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_uav(results: dict, dirs: dict, logger, suffix: str = "") -> None:
    t_ref = np.linspace(0, T_SIM, 1000)
    rx    = LEMN_A * np.sin(LEMN_W * t_ref)
    ry    = LEMN_A * np.sin(LEMN_W * t_ref) * np.cos(LEMN_W * t_ref)

    colors = {
        "osqp":          "chocolate",
        "tqpnet_infer":  "royalblue",
        "tqpnet_refine": "seagreen",
    }
    labels = {
        "osqp":         (f"OSQP N_H={N_H} "
                         f"({results['osqp']['ms_per_step']:.1f}ms/step, "
                         f"RMSE={results['osqp']['mean_rmse']:.3f}m)"),
        "tqpnet_infer": (f"TQPNet infer "
                         f"({results['tqpnet_infer']['ms_per_step']:.2f}ms/step, "
                         f"RMSE={results['tqpnet_infer']['mean_rmse']:.3f}m)"),
        "tqpnet_refine":(f"TQPNet+{N_REFINE} refine "
                         f"({results['tqpnet_refine']['ms_per_step']:.2f}ms/step, "
                         f"RMSE={results['tqpnet_refine']['mean_rmse']:.3f}m)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    T_ax = np.arange(1, T_SIM + 1) * DT

    # XY trajectory
    ax = axes[0]
    ax.plot(rx, ry, "k--", lw=2, alpha=0.6, zorder=5, label="Reference")
    for m in ["tqpnet_infer", "tqpnet_refine", "osqp"]:
        traj = results[m]["traj"]
        ax.plot(traj[:, 3], traj[:, 4], color=colors[m], lw=1.5, label=labels[m])
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("XY Trajectory"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.6, 0.6)

    # Tracking error (log)
    ax = axes[1]
    for m in ["tqpnet_infer", "tqpnet_refine", "osqp"]:
        ax.plot(T_ax, results[m]["err"], color=colors[m], lw=1.5, label=labels[m])
    ax.set_yscale("log"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Pos error (m)")
    ax.set_title("Tracking Error (log)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bar: RMSE + ms/step
    ax = axes[2]
    methods = ["tqpnet_infer", "tqpnet_refine", "osqp"]
    short   = ["TQPNet\ninfer", f"TQPNet\n+{N_REFINE} refine", f"OSQP\nN_H={N_H}"]
    rmse    = [results[m]["mean_rmse"] for m in methods]
    ms_s    = [results[m]["ms_per_step"] for m in methods]
    x_pos   = np.arange(len(methods))
    bars = ax.bar(x_pos, rmse, color=[colors[m] for m in methods], alpha=0.8, zorder=3)
    ax.set_xticks(x_pos); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Mean pos error (m)"); ax.set_yscale("log")
    ax.set_title("Accuracy vs Speed"); ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax2 = ax.twinx()
    ax2.plot(x_pos, ms_s, "D--k", ms=7, lw=2, label="ms/step", zorder=4)
    ax2.axhline(1000 * DT, color="red", ls=":", lw=1.5, label=f"RT ({1000*DT:.1f}ms)")
    ax2.set_ylabel("ms/step"); ax2.legend(fontsize=7)
    for bar, ms in zip(bars, ms_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                f"{ms:.1f}ms", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    fig.suptitle(
        f"UAV MPC: TQPNet (time-aware p) vs OSQP  |  N_H={N_H}, n={N_H*N_U}, T={T_SIM} steps",
        fontsize=11)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(dirs["figures"] / f"uav_trajectory{suffix}{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figures to %s", dirs["figures"])


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_uav_tqpnet(t_sim: int = T_SIM, force_train: bool = False) -> None:
    ensure_dirs()
    dirs   = subdirs(OUT_DIR)
    logger = setup_logger("uav_train", dirs["logs"] / "uav_tqpnet.log")
    logger.info("UAV TQPNet (time-aware) | N_H=%d n=%d np_dim=%d T_SIM=%d",
                N_H, N_H * N_U, NP_DIM_V2, t_sim)

    Ad, Bd = build_linear_model()
    cfg    = build_uav_cfg(rho_refine=1.0)
    logger.info("UAV cfg | n=%d m=%d np_dim=%d", cfg.n, cfg.m, cfg.np_dim)

    model_path = OUT_DIR / "models" / "tqpnet_uav.pth"
    model      = train_tqpnet_uav(cfg, model_path, logger, force_train=force_train)

    results = {}
    for method in ["osqp", "tqpnet_infer", "tqpnet_refine"]:
        logger.info("--- %s ---", method)
        res = run_sim(method, cfg, model, Ad, Bd, t_sim, logger)
        results[method] = res
        logger.info("DONE %s | RMSE=%.4f | z_err=%.5f | %.3fms/step",
                    method, res["mean_rmse"], res["mean_abs_z_err"], res["ms_per_step"])

    summary = {m: {k: v for k, v in r.items() if k not in ("traj", "err")}
               for m, r in results.items()}
    write_json(OUT_DIR / "uav_results.json", summary)
    plot_uav(results, dirs, logger, suffix="")

    logger.info("Done | out=%s", OUT_DIR)
    logger.info("=" * 60)
    logger.info("SUMMARY")
    for m, r in results.items():
        logger.info("  %-20s  RMSE=%10.4fm  ms/step=%8.3f", m, r["mean_rmse"], r["ms_per_step"])
    logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t-sim",       type=int, default=T_SIM)
    p.add_argument("--force-train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_uav_tqpnet(t_sim=args.t_sim, force_train=args.force_train)


if __name__ == "__main__":
    main()
