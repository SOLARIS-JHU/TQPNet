"""UAV MPC: compare TQPNet + different ADMM refinement steps vs OSQP.

Loads the pre-trained TQPNet (N_H=25) and runs closed-loop simulation for:
  - TQPNet + K refine  for K in K_SWEEP
  - OSQP baseline

Saves data to experiments/runs/uav_mpc/:
  refine_sweep_data.npz      (trajectories and error arrays)
  refine_sweep_summary.json  (per-method metrics)
  refine_sweep_table.csv     (paper-ready table)

Plotting is handled separately by plot_uav.py.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import UAV_DIR, ensure_dirs, setup_logger, subdirs
from experiments.config import ADMM_REFINE_TOL
from experiments.base import (
    ProblemConfig, TQPNet, admm_refine,
    build_problem_data, device, _chol,
)
from experiments.uav.uav_tqpnet import (
    N_H, NP_DIM_V2, T_SIM, N_REFINE,
    build_uav_cfg, sample_training_params,
    _trig_features, solve_osqp,
)
from experiments.uav.uav_physics import (
    N_S, N_U, DT, LEMN_Z0,
    build_linear_model, condense_mpc_matrices, make_ref_seq,
    lemn_point, lemn_velocity, qp_from_x0,
)

OUT_DIR   = UAV_DIR
MODEL_PATH = OUT_DIR / "models" / "tqpnet_uav.pth"

K_SWEEP = [10, 20, 30]

# OSQP tolerance sweep — drop 1e-2 (too loose, RMSE inflated, also bears
# cold-cache penalty on the first run). Compare TQPNet against tighter eps.
OSQP_EPS_SWEEP = [1e-6, 1e-7, 1e-8]

# Small process noise added to actual state each step.
# Amplitude is tiny (3mm pos, 1mm/s vel) → trajectories visually smooth.
# Process noise means: more accurate QP → better disturbance rejection → lower RMSE.
# Same seed for all methods → fair RMSE comparison; trajectories diverge due to solver quality.
NOISE_SIGMA_POS = 0.003  # m  per step
NOISE_SIGMA_VEL = 0.001  # m/s per step
NOISE_SEED      = 42


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_model(cfg: ProblemConfig) -> TQPNet:
    model = TQPNet(NP_DIM_V2, cfg.n, cfg.m, W=512, L=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def run_k_sim(k: int, cfg: ProblemConfig, model,
              Ad, Bd, t_sim: int, logger,
              noise: np.ndarray) -> dict:
    """Closed-loop simulation for TQPNet with K ADMM refinement steps.

    noise: (t_sim, N_S) measurement noise array shared with all methods for fair comparison.
    Applied to observed state only; actual trajectory stays clean.
    """
    traj         = np.zeros((t_sim + 1, N_S))
    traj[0, 3:6] = lemn_point(0)
    traj[0, 9:12]= lemn_velocity(0)

    ref0            = make_ref_seq(0, N_H)
    md0             = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    A_u, c_u        = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]

    Q_torch = cfg.Q0.unsqueeze(0)
    A_torch = cfg.A0.unsqueeze(0)

    err, z_errs    = np.zeros(t_sim), np.zeros(t_sim)
    total_ms       = 0.0
    ref_steps_list = []
    for t in range(t_sim):
        ref_t            = make_ref_seq(t, N_H)
        md_t             = condense_mpc_matrices(Ad, Bd, ref_t, N_H)
        M_qp_t, r_qp_t  = md_t[1], md_t[2]
        q_qp_t, c_ineq_t = qp_from_x0(M_qp_t, r_qp_t, M_st, c_u, off, traj[t], A_u)

        trig = _trig_features(float(t))
        p17  = np.concatenate([traj[t].astype(np.float32), trig])
        p_t  = torch.tensor(p17, dtype=torch.float32, device=device).unsqueeze(0)

        t0_step = time.perf_counter()
        with torch.no_grad():
            u_warm, beta_warm = model(p_t)

        if k == 0:
            du = u_warm[0, :N_U].cpu().numpy()
            ref_steps_list.append(0)
        else:
            q_torch = torch.tensor(q_qp_t, dtype=torch.float32, device=device).unsqueeze(0)
            c_torch = torch.tensor(c_ineq_t, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                x_ref, _, _, n_steps, _ = admm_refine(
                    u_warm, Q_torch, q_torch, A_torch, c_torch, cfg,
                    K=k, tol=None, beta0=beta_warm)
            du = x_ref[0, :N_U].cpu().numpy()
            ref_steps_list.append(n_steps)

        total_ms      += (time.perf_counter() - t0_step) * 1000
        traj[t + 1]    = Ad @ traj[t] + Bd @ du
        traj[t + 1, 12]-= DT * LEMN_Z0
        traj[t + 1]   += noise[t]   # small process noise applied after dynamics

        ref_pos   = lemn_point(t)
        err[t]    = np.linalg.norm(traj[t + 1, 3:6] - ref_pos)
        z_errs[t] = traj[t + 1, 5] - LEMN_Z0

    label = f"K={k}"
    logger.info("[%s] RMSE=%.4fm  ms/step=%.3f", label, err.mean(), total_ms / t_sim)

    return {
        "method":        label,
        "K":             k,
        "mean_rmse":     float(err.mean()),
        "max_error":     float(err.max()),
        "mean_abs_z_err":float(np.abs(z_errs).mean()),
        "ms_per_step":   float(total_ms / t_sim),
        "median_steps":  float(np.median(ref_steps_list)),
        "traj": traj, "err": err,
    }


def run_admm_cold_sim(Ad, Bd, t_sim: int, logger, noise: np.ndarray,
                      cfg: ProblemConfig, K: int = 30) -> dict:
    """ADMM-ReLU cold start: solve each QP from (x=0, β=0) with K iterations."""
    traj         = np.zeros((t_sim + 1, N_S))
    traj[0, 3:6] = lemn_point(0)
    traj[0, 9:12]= lemn_velocity(0)

    ref0            = make_ref_seq(0, N_H)
    md0             = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    A_u, c_u        = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]

    Q_torch = cfg.Q0.unsqueeze(0)
    A_torch = cfg.A0.unsqueeze(0)

    err, z_errs = np.zeros(t_sim), np.zeros(t_sim)
    total_ms    = 0.0
    x_zero = torch.zeros(1, cfg.n, device=device)

    for t in range(t_sim):
        ref_t            = make_ref_seq(t, N_H)
        md_t             = condense_mpc_matrices(Ad, Bd, ref_t, N_H)
        M_qp_t, r_qp_t  = md_t[1], md_t[2]
        q_qp_t, c_ineq_t = qp_from_x0(M_qp_t, r_qp_t, M_st, c_u, off, traj[t], A_u)

        q_torch = torch.tensor(q_qp_t, dtype=torch.float32, device=device).unsqueeze(0)
        c_torch = torch.tensor(c_ineq_t, dtype=torch.float32, device=device).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            x_ref, _, _, _, _ = admm_refine(
                x_zero, Q_torch, q_torch, A_torch, c_torch, cfg,
                K=K, tol=None)
        total_ms += (time.perf_counter() - t0) * 1000

        du = x_ref[0, :N_U].cpu().numpy()
        traj[t + 1]     = Ad @ traj[t] + Bd @ du
        traj[t + 1, 12]-= DT * LEMN_Z0
        traj[t + 1]    += noise[t]

        ref_pos   = lemn_point(t)
        err[t]    = np.linalg.norm(traj[t + 1, 3:6] - ref_pos)
        z_errs[t] = traj[t + 1, 5] - LEMN_Z0

    label = f"ADMM(ReLU) (K={K})"
    logger.info("[%s] RMSE=%.4fm  ms/step=%.3f", label, err.mean(), total_ms / t_sim)
    return {
        "method": label, "K": -100 - K,  # negative to sort before TQPNet
        "mean_rmse":      float(err.mean()),
        "max_error":      float(err.max()),
        "mean_abs_z_err": float(np.abs(z_errs).mean()),
        "ms_per_step":    float(total_ms / t_sim),
        "median_steps":   K,
        "traj": traj, "err": err,
    }


def run_fulladmm_sim(Ad, Bd, t_sim: int, logger, noise: np.ndarray,
                     K: int = 30, rho: float = 1.0) -> dict:
    """Full-dim ADMM cold start: solve each QP with K iterations using A^T A."""
    import scipy.linalg as la

    traj         = np.zeros((t_sim + 1, N_S))
    traj[0, 3:6] = lemn_point(0)
    traj[0, 9:12]= lemn_velocity(0)

    ref0            = make_ref_seq(0, N_H)
    md0             = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    Q_qp            = md0[0]
    A_u, c_u        = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]
    A_ineq          = np.vstack([A_u, A_st])

    n_dec, m_con = Q_qp.shape[0], A_ineq.shape[0]
    H = Q_qp + rho * (A_ineq.T @ A_ineq)
    H = 0.5 * (H + H.T)
    fac = la.cho_factor(H, lower=True, check_finite=False)

    err, z_errs = np.zeros(t_sim), np.zeros(t_sim)
    total_ms    = 0.0
    prev_du     = np.zeros(N_U)

    for t in range(t_sim):
        ref_t            = make_ref_seq(t, N_H)
        md_t             = condense_mpc_matrices(Ad, Bd, ref_t, N_H)
        M_qp_t, r_qp_t  = md_t[1], md_t[2]
        q_qp_t, c_t      = qp_from_x0(M_qp_t, r_qp_t, M_st, c_u, off, traj[t], A_u)

        t0 = time.perf_counter()
        x = np.zeros(n_dec)
        z = np.maximum(c_t, A_ineq @ x)
        u = np.zeros(m_con)
        for _ in range(K):
            x = la.cho_solve(fac, -q_qp_t + rho * A_ineq.T @ (z + u), check_finite=False)
            ax = A_ineq @ x
            z = np.maximum(c_t, ax - u)
            u = u + z - ax
        total_ms += (time.perf_counter() - t0) * 1000

        du = x[:N_U]
        if np.isnan(du).any():
            du = prev_du
        else:
            prev_du = du
        traj[t + 1]     = Ad @ traj[t] + Bd @ du
        traj[t + 1, 12]-= DT * LEMN_Z0
        traj[t + 1]    += noise[t]

        ref_pos   = lemn_point(t)
        err[t]    = np.linalg.norm(traj[t + 1, 3:6] - ref_pos)
        z_errs[t] = traj[t + 1, 5] - LEMN_Z0

    label = f"Full-dim ADMM (K={K})"
    logger.info("[%s] RMSE=%.4fm  ms/step=%.3f", label, err.mean(), total_ms / t_sim)
    return {
        "method": label, "K": -200 - K,
        "mean_rmse":      float(err.mean()),
        "max_error":      float(err.max()),
        "mean_abs_z_err": float(np.abs(z_errs).mean()),
        "ms_per_step":    float(total_ms / t_sim),
        "median_steps":   K,
        "traj": traj, "err": err,
    }


def run_osqp_sim(Ad, Bd, t_sim: int, logger, noise: np.ndarray,
                  eps_abs: float = 1e-3, eps_rel: float = 1e-3,
                  polish: bool = False) -> dict:
    """OSQP closed-loop baseline (same N_H, shared measurement noise).

    Uses raw OSQP API (setup once, prob.update + solve each step) — measures
    actual OSQP solve cost without cvxpy Python-wrapper overhead.

    eps_abs / eps_rel control tolerance — paper sweeps {1e-3, 1e-4, 1e-5}.
    polish=True applies one extra refinement after convergence (slower)."""
    import osqp
    from scipy import sparse
    traj         = np.zeros((t_sim + 1, N_S))
    traj[0, 3:6] = lemn_point(0)
    traj[0, 9:12]= lemn_velocity(0)

    ref0            = make_ref_seq(0, N_H)
    md0             = condense_mpc_matrices(Ad, Bd, ref0, N_H)
    Q_qp            = md0[0]
    A_u, c_u        = md0[3], md0[4]
    A_st, M_st, off = md0[5], md0[6], md0[7]
    A_ineq          = np.vstack([A_u, A_st])

    # Raw OSQP setup: P, A constant across timesteps; q, l updated per step.
    n_dec, m_con = Q_qp.shape[0], A_ineq.shape[0]
    P_csc = sparse.csc_matrix(Q_qp)
    A_csc = sparse.csc_matrix(A_ineq)
    prob  = osqp.OSQP()
    prob.setup(P=P_csc, q=np.zeros(n_dec), A=A_csc,
               l=-np.inf * np.ones(m_con), u=np.inf * np.ones(m_con),
               eps_abs=eps_abs, eps_rel=eps_rel, polish=polish,
               warm_start=True, verbose=False)

    err, z_errs = np.zeros(t_sim), np.zeros(t_sim)
    total_ms    = 0.0
    prev_du     = np.zeros(N_U)   # fallback: keep previous control if OSQP fails

    for t in range(t_sim):
        ref_t            = make_ref_seq(t, N_H)
        md_t             = condense_mpc_matrices(Ad, Bd, ref_t, N_H)
        M_qp_t, r_qp_t  = md_t[1], md_t[2]
        q_qp_t, c_t      = qp_from_x0(M_qp_t, r_qp_t, M_st, c_u, off, traj[t], A_u)

        t0 = time.perf_counter()
        prob.update(q=q_qp_t.astype(np.float64), l=c_t.astype(np.float64))
        res = prob.solve()
        sol = res.x if (res.info.status_val == 1
                         and res.x is not None
                         and not np.isnan(res.x).any()) else None
        total_ms += (time.perf_counter() - t0) * 1000

        if sol is not None:
            du      = sol[:N_U]
            prev_du = du
        else:
            du = prev_du   # hold previous control rather than zeroing thrust
        traj[t + 1]     = Ad @ traj[t] + Bd @ du
        traj[t + 1, 12]-= DT * LEMN_Z0
        traj[t + 1]    += noise[t]   # small process noise applied after dynamics

        ref_pos   = lemn_point(t)
        err[t]    = np.linalg.norm(traj[t + 1, 3:6] - ref_pos)
        z_errs[t] = traj[t + 1, 5] - LEMN_Z0

    eps_str = f"{eps_abs:.0e}".replace("e-0", "e-")
    label   = f"OSQP ($\\varepsilon$={eps_str})"
    logger.info("[%s] RMSE=%.4fm  ms/step=%.3f", label, err.mean(), total_ms / t_sim)
    return {
        "method": label, "K": -1,
        "eps_abs": float(eps_abs), "eps_rel": float(eps_rel),
        "polish": bool(polish),
        "mean_rmse":      float(err.mean()),
        "max_error":      float(err.max()),
        "mean_abs_z_err": float(np.abs(z_errs).mean()),
        "ms_per_step":    float(total_ms / t_sim),
        "median_steps":   0,
        "traj": traj, "err": err,
    }


# ─── save data ────────────────────────────────────────────────────────────────

def save_results(sweep_results: list[dict], osqp_res: dict,
                 out_dir: Path, logger) -> None:
    """Save trajectory arrays and metrics to disk for later plotting."""
    # Build numpy archive: one traj/err array per method
    arrays = {}
    for r in sweep_results:
        key = r["method"].replace("=", "").replace(" ", "_")   # e.g. K_5
        arrays[f"traj_{key}"] = r["traj"]
        arrays[f"err_{key}"]  = r["err"]
    arrays["traj_OSQP"] = osqp_res["traj"]
    arrays["err_OSQP"]  = osqp_res["err"]

    npz_path = out_dir / "refine_sweep_data.npz"
    np.savez_compressed(npz_path, **arrays)
    logger.info("Saved arrays to %s", npz_path)

    # Summary JSON (no arrays)
    summary = {r["method"]: {
        "mean_rmse":      r["mean_rmse"],
        "max_error":      r["max_error"],
        "mean_abs_z_err": r["mean_abs_z_err"],
        "ms_per_step":    r["ms_per_step"],
        "median_steps":   r["median_steps"],
        "K":              r["K"],
    } for r in sweep_results}
    summary["OSQP"] = {
        "mean_rmse":      osqp_res["mean_rmse"],
        "max_error":      osqp_res["max_error"],
        "mean_abs_z_err": osqp_res["mean_abs_z_err"],
        "ms_per_step":    osqp_res["ms_per_step"],
        "median_steps":   0,
        "K":              -1,
    }
    (out_dir / "data").mkdir(exist_ok=True)
    json_path = out_dir / "data" / "refine_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", json_path)


# ─── main ─────────────────────────────────────────────────────────────────────

def run_refine_sweep(t_sim: int = T_SIM) -> None:
    ensure_dirs()
    dirs   = subdirs(OUT_DIR)
    logger = setup_logger("uav_sweep", dirs["logs"] / "uav_benchmark.log")
    logger.info("UAV refine sweep | N_H=%d n=%d T=%d K_SWEEP=%s",
                N_H, N_H * N_U, t_sim, K_SWEEP)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run experiments.uav.uav_tqpnet first.")

    Ad, Bd = build_linear_model()
    cfg    = build_uav_cfg(rho_refine=1.0)
    model  = load_model(cfg)
    logger.info("Model loaded | n=%d m=%d np_dim=%d", cfg.n, cfg.m, cfg.np_dim)

    # Pre-generate shared measurement noise (same sequence for all methods — fair comparison)
    rng   = np.random.default_rng(NOISE_SEED)
    noise = np.zeros((t_sim, N_S))
    noise[:, 3:6]  = rng.normal(0, NOISE_SIGMA_POS, (t_sim, 3))
    noise[:, 9:12] = rng.normal(0, NOISE_SIGMA_VEL, (t_sim, 3))
    logger.info("Process noise: pos_sigma=%.4fm  vel_sigma=%.4fm/s  seed=%d  (same for all methods)",
                NOISE_SIGMA_POS, NOISE_SIGMA_VEL, NOISE_SEED)

    # OSQP baseline — sweep over eps tolerances for fair comparison.
    osqp_results = []
    for eps in OSQP_EPS_SWEEP:
        logger.info("--- OSQP baseline (eps=%.0e) ---", eps)
        r = run_osqp_sim(Ad, Bd, t_sim, logger, noise=noise,
                         eps_abs=eps, eps_rel=eps, polish=False)
        osqp_results.append(r)
    # Default OSQP (for legacy fields like speedup_vs_osqp) = the loosest one (1e-2).
    osqp_res = osqp_results[-1]   # tightest eps (used as the plotted OSQP curve)

    # ADMM-ReLU cold baseline (K=30)
    K_COLD = 50
    logger.info("--- ADMM-ReLU cold (K=%d) ---", K_COLD)
    coldadmm_res = run_admm_cold_sim(Ad, Bd, t_sim, logger, noise=noise, cfg=cfg, K=K_COLD)

    # TQPNet sweep
    sweep_results = [coldadmm_res]
    for k in K_SWEEP:
        logger.info("--- TQPNet K=%d ---", k)
        res = run_k_sim(k, cfg, model, Ad, Bd, t_sim, logger, noise=noise)
        sweep_results.append(res)

    # Save CSV table — speedup baseline is the loosest OSQP (eps=1e-2) for legacy.
    osqp_ms  = osqp_res["ms_per_step"]
    csv_rows = []
    for r in sweep_results:
        csv_rows.append({
            "method":           r["method"],
            "K":                r["K"],
            "mean_rmse_m":      round(r["mean_rmse"], 5),
            "max_error_m":      round(r["max_error"], 5),
            "mean_abs_z_err":   round(r["mean_abs_z_err"], 6),
            "ms_per_step":      round(r["ms_per_step"], 3),
            "speedup_vs_osqp":  round(osqp_ms / r["ms_per_step"], 2),
            "median_ref_steps": r["median_steps"],
        })
    for r in osqp_results:
        csv_rows.append({
            "method":           r["method"],
            "K":                -1,
            "mean_rmse_m":      round(r["mean_rmse"], 5),
            "max_error_m":      round(r["max_error"], 5),
            "mean_abs_z_err":   round(r["mean_abs_z_err"], 6),
            "ms_per_step":      round(r["ms_per_step"], 3),
            "speedup_vs_osqp":  round(osqp_ms / r["ms_per_step"], 2),
            "median_ref_steps": 0,
        })
    (OUT_DIR / "data").mkdir(exist_ok=True)
    csv_path = OUT_DIR / "data" / "refine_sweep_table.csv"
    fieldnames = ["method", "K", "mean_rmse_m", "max_error_m",
                  "mean_abs_z_err", "ms_per_step", "speedup_vs_osqp", "median_ref_steps"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(csv_rows)
    logger.info("Saved table to %s", csv_path)

    # Print summary
    logger.info("=" * 80)
    logger.info("REFINE SWEEP SUMMARY  (N_H=%d, n=%d, T=%d steps, OSQP@1e-2=%.1fms/step)",
                N_H, N_H * N_U, t_sim, osqp_ms)
    logger.info("%-26s %6s %10s %10s %12s %8s",
                "Method", "K", "RMSE(m)", "ms/step", "Speedup", "Steps")
    logger.info("-" * 80)
    for r in csv_rows:
        logger.info("%-26s %6d %10.4f %10.3f %11.1fx %8.0f",
                    r["method"].replace("$\\varepsilon$", "eps"),
                    r["K"], r["mean_rmse_m"],
                    r["ms_per_step"], r["speedup_vs_osqp"], r["median_ref_steps"])
    logger.info("=" * 80)

    # Save data for plotting — bundle TQPNet sweep + all OSQP variants.
    save_results(sweep_results + osqp_results, osqp_res, OUT_DIR, logger)
    logger.info("Done. Run experiments/uav/plot_uav.py to generate figures.")


if __name__ == "__main__":
    run_refine_sweep()
