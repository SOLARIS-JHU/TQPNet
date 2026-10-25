"""Scalability benchmark on randomly generated MPC problems.

System generation follows Wang & Boyd (2010): A, B ~ N(0,1), A scaled to
unit spectral radius, |x_i| <= 10, |u_i| <= 1, Q = I, R = I.

Closed-loop MPC simulation at each scale: compare tracking RMSE and solve
time between TQPNet + K ADMM-ReLU and OSQP.

Output: experiments/runs/scalability_mpc/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.base import (
    ProblemConfig, TQPNet, LQPNet, admm_unrolled_loss, admm_refine,
    kkt_raw_residual, solve_osqp_reference, device, _chol,
)
from experiments.common import RUN_DIR, ensure_dirs, setup_logger, subdirs, write_json
from experiments.config import (
    BATCH_SIZE, LR, WEIGHT_DECAY,
    LOG_EVERY_EPOCH, LOSS_THRESHOLD,
)

OUT_DIR = RUN_DIR / "scalability_mpc"

# Problem scales: (n_state, n_input, N_horizon)
MPC_SCALES = [
    {"name": "S1",  "ns": 4,  "nu": 2,  "N": 10},
    {"name": "S2",  "ns": 6,  "nu": 3,  "N": 15},
    {"name": "S3",  "ns": 8,  "nu": 4,  "N": 20},
    {"name": "S4",  "ns": 10, "nu": 5,  "N": 25},
    {"name": "S5",  "ns": 12, "nu": 6,  "N": 25},
    {"name": "S6",  "ns": 16, "nu": 8,  "N": 25},
    {"name": "S7",  "ns": 24, "nu": 12, "N": 25},
    {"name": "S8",  "ns": 32, "nu": 16, "N": 25},
]

T_SIM        = 200    # closed-loop simulation steps
K_TRAIN      = 10
K_EVAL       = [10]          # TQPNet/LQPNet use K=10 (matches K_train)
K_COLD       = 50            # ADMM-ReLU cold gets more steps

# Override training params for faster scalability runs
N_TRAIN      = 2000
MAX_EPOCHS   = 1000


# ── Random MPC system generation (Wang & Boyd 2010) ──────────────────────────

def generate_mpc_system(ns, nu, N, seed=42):
    rng = np.random.default_rng(seed)
    A_sys = rng.standard_normal((ns, ns))
    A_sys = A_sys / (np.max(np.abs(np.linalg.eigvals(A_sys))) + 1e-6)
    B_sys = rng.standard_normal((ns, nu))
    Q_s, R_s, P_s = np.eye(ns), np.eye(nu), np.eye(ns)

    n_dec = N * nu
    A_bar = np.zeros((N * ns, ns))
    B_bar = np.zeros((N * ns, n_dec))
    Ap = np.eye(ns)
    for i in range(N):
        Ap = A_sys @ Ap
        A_bar[i*ns:(i+1)*ns, :] = Ap
        for j in range(i + 1):
            B_bar[i*ns:(i+1)*ns, j*nu:(j+1)*nu] = \
                np.linalg.matrix_power(A_sys, i - j) @ B_sys

    Q_blk = np.kron(np.eye(N), Q_s)
    Q_blk[-ns:, -ns:] = P_s
    R_blk = np.kron(np.eye(N), R_s)
    H = B_bar.T @ Q_blk @ B_bar + R_blk
    H = 0.5 * (H + H.T) + 1e-8 * np.eye(n_dec)
    M = B_bar.T @ Q_blk @ A_bar

    # Constraints: |u_i| <= 1, |x_i| <= 10
    I_u = np.eye(n_dec)
    A_input = np.vstack([I_u, -I_u])
    c_input = np.hstack([np.full(n_dec, -1.0), np.full(n_dec, -1.0)])
    A_state = np.vstack([B_bar, -B_bar])
    M_state = np.vstack([-A_bar, A_bar])
    off_state = np.full(2 * N * ns, -10.0)
    A_full = np.vstack([A_input, A_state])

    return {
        "H": H, "M": M, "A_full": A_full,
        "c_input": c_input, "M_state": M_state, "off_state": off_state,
        "A_sys": A_sys, "B_sys": B_sys, "A_bar": A_bar, "B_bar": B_bar,
        "ns": ns, "nu": nu, "N": N,
        "n_dec": n_dec, "m_con": A_full.shape[0],
    }


def build_mpc_cfg(sys_data, rho_refine=1.0):
    n = sys_data["n_dec"]
    m = sys_data["m_con"]
    ns = sys_data["ns"]
    np_dim = ns

    cfg = ProblemConfig(n=n, m=m, np_dim=np_dim, rho_refine=rho_refine)
    cfg.Q0 = torch.tensor(sys_data["H"], dtype=torch.float32, device=device)
    cfg.A0 = torch.tensor(sys_data["A_full"], dtype=torch.float32, device=device)
    cfg.q0 = torch.zeros(n, device=device)
    cfg.Cq = torch.tensor(sys_data["M"], dtype=torch.float32, device=device)

    c_fixed = np.hstack([sys_data["c_input"], sys_data["off_state"]])
    cfg.c0_vec = torch.tensor(c_fixed, dtype=torch.float32, device=device)

    Cc_np = np.zeros((m, np_dim))
    n_input_con = 2 * n
    cfg.Cc = torch.tensor(
        np.vstack([np.zeros((n_input_con, np_dim)), sys_data["M_state"]]),
        dtype=torch.float32, device=device)

    cfg._D_U = (cfg.A0 ** 2).sum(dim=1)
    cfg._D_V = (cfg.A0.abs() > 1e-12).to(cfg.A0.dtype).sum(dim=0)
    cfg._inv_D_U = 1.0 / cfg._D_U.clamp(min=1e-12)
    H_V = cfg.Q0 + rho_refine * torch.diag(cfg._D_V)
    cfg._H_V_chol = _chol(H_V)
    H_inv = torch.linalg.solve(H_V, torch.eye(n, device=device))
    cfg._M1 = rho_refine * H_inv @ torch.diag(cfg._D_V)
    cfg._M2 = rho_refine * H_inv @ cfg.A0.T
    cfg._H_inv = H_inv
    cfg._M_beta = -torch.diag(cfg._inv_D_U) @ cfg.A0
    return cfg


def build_problem_data(P, cfg):
    b = P.shape[0]
    return (
        cfg.Q0.unsqueeze(0).expand(b, -1, -1),
        cfg.q0.unsqueeze(0) + P @ cfg.Cq.T,
        cfg.A0.unsqueeze(0).expand(b, -1, -1),
        cfg.c0_vec.unsqueeze(0) + P @ cfg.Cc.T,
    )


def get_q_c(x0, sys_data, cfg):
    """Get QP vectors q(x0) and c(x0) for a single state."""
    M = sys_data["M"]
    q = M @ x0
    c_fixed = np.hstack([sys_data["c_input"], sys_data["off_state"]])
    M_state = sys_data["M_state"]
    c = np.concatenate([c_fixed[:2*sys_data["n_dec"]], c_fixed[2*sys_data["n_dec"]:] + M_state @ x0])
    return q, c


def _compute_metrics(traj, controls, t_sim, total_ms):
    """Compute closed-loop MPC metrics."""
    x_states = traj[1:]  # (T, ns)
    rmse = float(np.sqrt(np.mean(x_states ** 2)))
    cost = float(np.sum(x_states ** 2) + np.sum(controls ** 2))
    viol = float(np.mean(np.any(np.abs(x_states) > 10, axis=1)))
    ms_step = total_ms / t_sim
    return rmse, cost, viol, ms_step


# ── Closed-loop simulation ───────────────────────────────────────────────────

def run_closedloop_osqp(sys_data, cfg, t_sim, logger):
    ns, nu, N = sys_data["ns"], sys_data["nu"], sys_data["N"]
    A_sys, B_sys = sys_data["A_sys"], sys_data["B_sys"]
    n_dec, m_con = sys_data["n_dec"], sys_data["m_con"]
    Q_np, A_qp = cfg.Q0.cpu().numpy(), cfg.A0.cpu().numpy()

    x = np.zeros(ns)
    rng = np.random.default_rng(123)
    traj = np.zeros((t_sim + 1, ns))
    controls = np.zeros((t_sim, nu))
    total_ms = 0.0

    prob = osqp.OSQP()
    q0, c0 = get_q_c(x, sys_data, cfg)
    prob.setup(P=sp.triu(sp.csc_matrix(Q_np), format="csc"),
               q=q0, A=sp.csc_matrix(A_qp),
               l=c0, u=np.full(m_con, np.inf),
               warm_starting=False, verbose=False,
               eps_abs=1e-4, eps_rel=0, scaling=0, max_iter=200000, polish=False)
    prob.solve()

    for t in range(t_sim):
        q_t, c_t = get_q_c(x, sys_data, cfg)
        prob.update(q=q_t, l=c_t)
        t0 = time.perf_counter()
        res = prob.solve()
        total_ms += (time.perf_counter() - t0) * 1000

        if res.x is not None and not np.isnan(res.x).any():
            u = res.x[:nu]
        else:
            u = np.zeros(nu)
        u = np.clip(u, -1.0, 1.0)
        controls[t] = u

        w = rng.uniform(-0.3, 0.3, ns)
        x = A_sys @ x + B_sys @ u + w
        traj[t + 1] = x

    rmse, cost, viol, ms_step = _compute_metrics(traj, controls, t_sim, total_ms)
    return rmse, cost, viol, ms_step


def run_closedloop_tqpnet(sys_data, cfg, model, K, t_sim, logger):
    ns, nu, N = sys_data["ns"], sys_data["nu"], sys_data["N"]
    A_sys, B_sys = sys_data["A_sys"], sys_data["B_sys"]

    x = np.zeros(ns)
    rng = np.random.default_rng(123)
    traj = np.zeros((t_sim + 1, ns))
    controls = np.zeros((t_sim, nu))
    total_ms = 0.0

    for t in range(t_sim):
        p_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        Q_b, q_b, A_b, c_b = build_problem_data(p_t, cfg)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            xi, bi = model(p_t)
            if K > 0:
                xr, _, _, _, _ = admm_refine(xi, Q_b, q_b, A_b, c_b, cfg,
                                              K=K, tol=None, beta0=bi)
            else:
                xr = xi
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        u = xr[0, :nu].cpu().numpy()
        u = np.clip(u, -1.0, 1.0)
        controls[t] = u

        w = rng.uniform(-0.3, 0.3, ns)
        x = A_sys @ x + B_sys @ u + w
        traj[t + 1] = x

    rmse, cost, viol, ms_step = _compute_metrics(traj, controls, t_sim, total_ms)
    return rmse, cost, viol, ms_step


def run_closedloop_cold(sys_data, cfg, K, t_sim, logger):
    ns, nu = sys_data["ns"], sys_data["nu"]
    A_sys, B_sys = sys_data["A_sys"], sys_data["B_sys"]

    x = np.zeros(ns)
    rng = np.random.default_rng(123)
    traj = np.zeros((t_sim + 1, ns))
    controls = np.zeros((t_sim, nu))
    total_ms = 0.0

    for t in range(t_sim):
        p_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        Q_b, q_b, A_b, c_b = build_problem_data(p_t, cfg)
        x0 = torch.zeros(1, cfg.n, device=device)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            xr, _, _, _, _ = admm_refine(x0, Q_b, q_b, A_b, c_b, cfg,
                                          K=K, tol=None)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        u = xr[0, :nu].cpu().numpy()
        u = np.clip(u, -1.0, 1.0)
        controls[t] = u

        w = rng.uniform(-0.3, 0.3, ns)
        x = A_sys @ x + B_sys @ u + w
        traj[t + 1] = x

    rmse, cost, viol, ms_step = _compute_metrics(traj, controls, t_sim, total_ms)
    return rmse, cost, viol, ms_step


# ── Training ──────────────────────────────────────────────────────────────────

def train_tqpnet(scale, cfg, logger):
    np_dim, n, m = cfg.np_dim, cfg.n, cfg.m
    W, L = 512, 4

    torch.manual_seed(0)
    P_train = torch.randn(N_TRAIN, np_dim, device=device) * 0.5

    model = TQPNet(np_dim, n, m, W, L).to(device)
    loader = DataLoader(TensorDataset(P_train), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    logger.info("Training TQPNet %s | n=%d m=%d np=%d W=%d", scale["name"], n, m, np_dim, W)
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for (p_b,) in loader:
            Q_b, q_b, A_b, c_b = build_problem_data(p_b, cfg)
            opt.zero_grad()
            xp, bp = model(p_b)
            loss = admm_unrolled_loss(xp, bp, Q_b, q_b, A_b, c_b, cfg, K_unroll=K_TRAIN)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step(); ep_loss += loss.item() * p_b.shape[0]
        sched.step(); ep_loss /= N_TRAIN
        if ep % LOG_EVERY_EPOCH == 0 or ep == MAX_EPOCHS:
            logger.info("[%s] ep=%4d loss=%.4e", scale["name"], ep, ep_loss)
        if ep_loss < LOSS_THRESHOLD:
            logger.info("[%s] converged ep=%d", scale["name"], ep); break
    logger.info("[%s] train done | %.1fs", scale["name"], time.time() - t0)
    model.eval()
    return model


def train_lqpnet(scale, cfg, logger):
    np_dim, n, m = cfg.np_dim, cfg.n, cfg.m
    W, L = 512, 4

    torch.manual_seed(0)
    P_train = torch.randn(N_TRAIN, np_dim, device=device) * 0.5

    # Generate OSQP labels
    logger.info("Generating LQPNet labels (OSQP) for %s ...", scale["name"])
    t_label = time.time()
    Q_b, q_b, A_b, c_b = build_problem_data(P_train, cfg)
    X_star = torch.zeros(N_TRAIN, n); valid = []
    for i in range(N_TRAIN):
        xv, _, _ = solve_osqp_reference(
            Q_b[i].cpu().numpy(), q_b[i].cpu().numpy(),
            A_b[i].cpu().numpy(), c_b[i].cpu().numpy(), n)
        if xv is not None and not np.isnan(xv).any():
            X_star[i] = torch.tensor(xv, dtype=torch.float32); valid.append(i)
    label_time = time.time() - t_label
    logger.info("  Labels: %d/%d valid | %.1fs", len(valid), N_TRAIN, label_time)

    from torch.utils.data import Dataset
    class SD(Dataset):
        def __init__(self, P, X): self.P, self.X = P.float(), X.float()
        def __len__(self): return len(self.P)
        def __getitem__(self, i): return self.P[i], self.X[i]

    model = LQPNet(np_dim, n, W, L).to(device)
    loader = DataLoader(SD(P_train[valid].cpu(), X_star[valid]),
                        batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    logger.info("Training LQPNet %s | n=%d W=%d", scale["name"], n, W)
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); ep_loss = 0.0
        for p_b, x_b in loader:
            p_b, x_b = p_b.to(device), x_b.to(device)
            opt.zero_grad()
            loss = torch.log1p(((model(p_b) - x_b)**2).sum(1).mean() / n)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step(); ep_loss += loss.item() * p_b.shape[0]
        sched.step(); ep_loss /= max(len(valid), 1)
        if ep % LOG_EVERY_EPOCH == 0 or ep == MAX_EPOCHS:
            logger.info("[%s LQP] ep=%4d loss=%.4e", scale["name"], ep, ep_loss)
        if ep_loss < LOSS_THRESHOLD:
            logger.info("[%s LQP] converged ep=%d", scale["name"], ep); break
    train_time = time.time() - t0
    logger.info("[%s LQP] train done | %.1fs (labels=%.1fs total=%.1fs)",
                scale["name"], train_time, label_time, label_time + train_time)
    model.eval()
    return model, label_time + train_time


def run_closedloop_lqpnet(sys_data, cfg, model, K, t_sim, logger):
    ns, nu = sys_data["ns"], sys_data["nu"]
    A_sys, B_sys = sys_data["A_sys"], sys_data["B_sys"]

    x = np.zeros(ns)
    rng = np.random.default_rng(123)
    traj = np.zeros((t_sim + 1, ns))
    controls = np.zeros((t_sim, nu))
    total_ms = 0.0

    for t in range(t_sim):
        p_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        Q_b, q_b, A_b, c_b = build_problem_data(p_t, cfg)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            xi = model(p_t)
            if K > 0:
                xr, _, _, _, _ = admm_refine(xi, Q_b, q_b, A_b, c_b, cfg,
                                              K=K, tol=None)
            else:
                xr = xi
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000

        u = xr[0, :nu].cpu().numpy()
        u = np.clip(u, -1.0, 1.0)
        controls[t] = u

        w = rng.uniform(-0.3, 0.3, ns)
        x = A_sys @ x + B_sys @ u + w
        traj[t + 1] = x

    rmse, cost, viol, ms_step = _compute_metrics(traj, controls, t_sim, total_ms)
    return rmse, cost, viol, ms_step


# ── Main ──────────────────────────────────────────────────────────────────────

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def run_scalability_mpc():
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = subdirs(OUT_DIR)
    logger = setup_logger("scal_mpc", dirs["logs"] / "scalability_mpc.log")
    logger.info("Scalability MPC | K_TRAIN=%d K_EVAL=%s T_SIM=%d", K_TRAIN, K_EVAL, T_SIM)

    results = []
    for scale in MPC_SCALES:
        ns, nu, N = scale["ns"], scale["nu"], scale["N"]
        n_dec = N * nu
        logger.info("════ %s: ns=%d nu=%d N=%d → n=%d ════", scale["name"], ns, nu, N, n_dec)

        sys_data = generate_mpc_system(ns, nu, N, seed=42)
        cfg = build_mpc_cfg(sys_data)

        row = {"name": scale["name"], "ns": ns, "nu": nu, "N": N,
               "n": cfg.n, "m": cfg.m}

        # OSQP closed-loop
        rmse, cost, viol, ms = run_closedloop_osqp(sys_data, cfg, T_SIM, logger)
        row.update({"osqp_rmse": rmse, "osqp_cost": cost, "osqp_viol": viol, "osqp_ms": ms})
        logger.info("  OSQP: RMSE=%.4f cost=%.1f viol=%.2f ms=%.2f", rmse, cost, viol, ms)

        # ADMM-ReLU cold closed-loop (K=50)
        rmse, cost, viol, ms = run_closedloop_cold(sys_data, cfg, K_COLD, T_SIM, logger)
        row.update({"cold_rmse": rmse, "cold_cost": cost, "cold_viol": viol, "cold_ms": ms})
        logger.info("  ADMM cold K=%d: RMSE=%.4f cost=%.1f viol=%.2f ms=%.2f",
                    K_COLD, rmse, cost, viol, ms)

        # Train TQPNet (unsupervised)
        t0_tqp = time.time()
        model = train_tqpnet(scale, cfg, logger)
        row["tqp_offline_s"] = time.time() - t0_tqp
        torch.save(model.state_dict(), dirs["models"] / f"tqpnet_{scale['name']}.pth")

        # Train LQPNet (supervised, needs OSQP labels)
        lqp_model, lqp_offline = train_lqpnet(scale, cfg, logger)
        row["lqp_offline_s"] = lqp_offline
        torch.save(lqp_model.state_dict(), dirs["models"] / f"lqpnet_{scale['name']}.pth")
        logger.info("  Offline: TQPNet=%.0fs  LQPNet=%.0fs", row["tqp_offline_s"], lqp_offline)

        # TQPNet closed-loop
        for K in K_EVAL:
            rmse, cost, viol, ms = run_closedloop_tqpnet(sys_data, cfg, model, K, T_SIM, logger)
            row.update({f"tqp_K{K}_rmse": rmse, f"tqp_K{K}_cost": cost,
                        f"tqp_K{K}_viol": viol, f"tqp_K{K}_ms": ms})
            sp = row["osqp_ms"] / max(ms, 0.01)
            logger.info("  TQPNet K=%d: RMSE=%.4f cost=%.1f viol=%.2f ms=%.2f (%.1fx)",
                        K, rmse, cost, viol, ms, sp)

        # TQPNet infer only
        rmse, cost, viol, ms = run_closedloop_tqpnet(sys_data, cfg, model, 0, T_SIM, logger)
        row.update({"tqp_infer_rmse": rmse, "tqp_infer_cost": cost,
                    "tqp_infer_viol": viol, "tqp_infer_ms": ms})

        # LQPNet closed-loop (same K values)
        for K in K_EVAL:
            rmse, cost, viol, ms = run_closedloop_lqpnet(sys_data, cfg, lqp_model, K, T_SIM, logger)
            row.update({f"lqp_K{K}_rmse": rmse, f"lqp_K{K}_cost": cost,
                        f"lqp_K{K}_viol": viol, f"lqp_K{K}_ms": ms})
            logger.info("  LQPNet K=%d: RMSE=%.4f cost=%.1f ms=%.2f", K, rmse, cost, ms)

        logger.info("  Summary: OSQP=%.2fms(RMSE=%.4f cost=%.0f)  TQP(K=10)=%.2fms(RMSE=%.4f cost=%.0f)  speedup=%.1fx",
                    row["osqp_ms"], row["osqp_rmse"], row["osqp_cost"],
                    row.get("tqp_K10_ms", 0), row.get("tqp_K10_rmse", 0), row.get("tqp_K10_cost", 0),
                    row["osqp_ms"] / max(row.get("tqp_K10_ms", 1), 0.01))

        results.append(row)
        write_json(OUT_DIR / "scalability_mpc_results.json", results)

        # Update all outputs after each scale
        try:
            plot_scalability_mpc(results)
            make_csv(results)
            make_latex_table(results)
        except Exception as e:
            logger.warning("Output update failed: %s", e)

    logger.info("Done | %s", OUT_DIR)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_scalability_mpc(results=None):
    if results is None:
        path = OUT_DIR / "scalability_mpc_results.json"
        if not path.exists():
            return
        with open(path) as f:
            results = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})

    nm = [r["n"] + r["m"] for r in results]
    tqp_c, lqp_c, osqp_c, cold_c = "crimson", "seagreen", "darkorange", "royalblue"

    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        for suffix in (".png", ".pdf"):
            fig.savefig(FIG_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}{{.png,.pdf}}")

    mk_kw = dict(markeredgecolor="white", markeredgewidth=0.5)

    # Load explicit MPC results if available
    explicit_path = Path(__file__).resolve().parent / "explicit_mpc" / "explicit_mpc_results.json"
    explicit_data = []
    if explicit_path.exists():
        with open(explicit_path) as f:
            explicit_data = [r for r in json.load(f) if r.get("status") == "success"]
    ex_color = "purple"

    # ── (1) Solve latency ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(nm, [r["osqp_ms"] for r in results], "D-",
                color=osqp_c, lw=2, ms=7, **mk_kw, label="OSQP")
    ax.semilogy(nm, [r.get("cold_ms", 0) for r in results], "o:",
                color=cold_c, lw=1.5, ms=6, **mk_kw, label=f"ADMM(ReLU) ($K$={K_COLD})")
    ax.semilogy(nm, [r.get("tqp_K10_ms", 0) for r in results], "s-",
                color=tqp_c, lw=2, ms=7, **mk_kw, label="TQPNet ($K$=10)")
    if "lqp_K10_ms" in results[0]:
        ax.semilogy(nm, [r.get("lqp_K10_ms", 0) for r in results], "^--",
                    color=lqp_c, lw=1.8, ms=6, **mk_kw, label="LQPNet ($K$=10)")
    # Explicit MPC (from MATLAB results — only S1, S2)
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        ex_ms = [r["online_median_ms"] for r in explicit_data]
        ax.semilogy(ex_nm, ex_ms, "x-", color=ex_color, lw=1.5, ms=8,
                    markeredgewidth=2, label="Explicit MPC")
    ax.set_xlabel("$n + m$ (QP size)")
    ax.set_ylabel("Latency (ms/step)")
    ax.set_title("Solve Latency")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    _save(fig, "latency")

    # ── (2) RMSE (left) + Closed-loop cost (right) dual y-axis ──────────
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    rmse_c, cost_c = "steelblue", "indianred"

    # Left: RMSE (solid lines)
    l1, = ax1.plot(nm, [r["osqp_rmse"] for r in results], "D-",
                   color=osqp_c, lw=2.5, ms=9, **mk_kw, label="OSQP (RMSE)")
    l2, = ax1.plot(nm, [r.get("tqp_K10_rmse", 0) for r in results], "s-",
                   color=tqp_c, lw=2, ms=7, **mk_kw, label="TQPNet (RMSE)")
    l3, = ax1.plot(nm, [r.get("cold_rmse", 0) for r in results], "o-",
                   color=cold_c, lw=1.5, ms=6, **mk_kw, label="ADMM(ReLU) (RMSE)")
    lines_left = [l1, l2, l3]
    if "lqp_K10_rmse" in results[0]:
        l_lqp, = ax1.plot(nm, [r.get("lqp_K10_rmse", 0) for r in results], "^-",
                          color=lqp_c, lw=1.8, ms=6, **mk_kw, label="LQPNet (RMSE)")
        lines_left.append(l_lqp)
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        l_ex, = ax1.plot(ex_nm, [r["rmse"] for r in explicit_data], "x-",
                         color=ex_color, lw=1.5, ms=8, markeredgewidth=2, label="Explicit MPC (RMSE)")
        lines_left.append(l_ex)
    ax1.set_xlabel("$n + m$ (QP size)")
    ax1.set_ylabel("Tracking RMSE (solid ━)")
    ax1.tick_params(axis="y")

    # Right: Cost (dashed lines)
    ax2 = ax1.twinx()
    l4, = ax2.plot(nm, [r["osqp_cost"] for r in results], "D--",
                   color=osqp_c, lw=1.5, ms=6, alpha=0.5, label="OSQP (Cost)")
    l5, = ax2.plot(nm, [r.get("tqp_K10_cost", 0) for r in results], "s--",
                   color=tqp_c, lw=1.5, ms=5, alpha=0.5, label="TQPNet (Cost)")
    l6, = ax2.plot(nm, [r.get("cold_cost", 0) for r in results], "o--",
                   color=cold_c, lw=1.2, ms=4, alpha=0.4, label="ADMM(ReLU) (Cost)")
    lines_right = [l4, l5, l6]
    if "lqp_K10_cost" in results[0]:
        l_lqp_c, = ax2.plot(nm, [r.get("lqp_K10_cost", 0) for r in results], "^--",
                            color=lqp_c, lw=1.2, ms=4, alpha=0.4, label="LQPNet (Cost)")
        lines_right.append(l_lqp_c)
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        l_ex_c, = ax2.plot(ex_nm, [r["cost"] for r in explicit_data], "x--",
                           color=ex_color, lw=1.2, ms=6, alpha=0.5, markeredgewidth=2, label="Explicit MPC (Cost)")
        lines_right.append(l_ex_c)
    ax2.set_ylabel("Closed-loop cost $J$ (dashed ┅)")
    ax2.tick_params(axis="y")

    ax1.set_title("Control Quality  (solid: RMSE,  dashed: Cost)")
    method_labels = ["OSQP", f"TQPNet ($K$=10)", f"ADMM(ReLU) ($K$={K_COLD})"]
    if "lqp_K10_rmse" in results[0]:
        method_labels.append("LQPNet ($K$=10)")
    if explicit_data:
        method_labels.append("Explicit MPC")
    ax1.legend(lines_left, method_labels, loc="upper center", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.2)
    _save(fig, "control_quality")

    # ── (3) Offline preparation cost ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    tqp_off = [r.get("tqp_offline_s", 0) for r in results]
    lqp_off = [r.get("lqp_offline_s", 0) for r in results]
    if any(v > 0 for v in tqp_off):
        ax.semilogy(nm, tqp_off, "s-", color=tqp_c, lw=2, ms=7, **mk_kw,
                    label="TQPNet (train only)")
    if any(v > 0 for v in lqp_off):
        ax.semilogy(nm, lqp_off, "^-", color=lqp_c, lw=2, ms=7, **mk_kw,
                    label="LQPNet (train + labels)")
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        ex_off = [r["offline_s"] for r in explicit_data]
        ax.semilogy(ex_nm, ex_off, "x-", color=ex_color, lw=1.5, ms=8,
                    markeredgewidth=2, label="Explicit MPC")
    ax.set_xlabel("$n + m$ (QP size)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Offline Preparation Time")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    _save(fig, "offline_time")

    # ── (4) Speedup ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sp = [r["osqp_ms"] / max(r.get("tqp_K10_ms", 1), 0.01) for r in results]
    ax.plot(nm, sp, "s-", color=tqp_c, lw=2, ms=7, **mk_kw, label="TQPNet ($K$=10)")
    ax.axhline(1, color="gray", ls=":", lw=1, alpha=0.4, label="Parity")
    ax.set_xlabel("$n + m$ (QP size)")
    ax.set_ylabel("Speedup vs OSQP ($\\times$)")
    ax.set_title("Speedup over OSQP")
    ax.legend(); ax.grid(True, alpha=0.2)
    _save(fig, "speedup")


def make_csv(results=None):
    if results is None:
        path = OUT_DIR / "scalability_mpc_results.json"
        if not path.exists():
            return
        with open(path) as f:
            results = json.load(f)

    import csv
    DATA_DIR = OUT_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "scalability_mpc.csv"

    fields = ["name", "ns", "nu", "N", "n", "m",
              "osqp_ms", "osqp_rmse", "osqp_cost", "osqp_viol"]
    for K in K_EVAL:
        fields += [f"tqp_K{K}_ms", f"tqp_K{K}_rmse", f"tqp_K{K}_cost", f"tqp_K{K}_viol"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved {csv_path.name}")


def make_latex_table(results=None):
    if results is None:
        path = OUT_DIR / "scalability_mpc_results.json"
        if not path.exists():
            return
        with open(path) as f:
            results = json.load(f)

    # Load explicit MPC results if available
    explicit_path = Path(__file__).resolve().parent / "explicit_mpc" / "explicit_mpc_results.json"
    explicit_data = []
    if explicit_path.exists():
        with open(explicit_path) as f:
            explicit_data = [r for r in json.load(f) if r.get("status") == "success"]
    ex_lookup = {r["n_plus_m"]: r for r in explicit_data}

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Scalability on random MPC problems ($T=200$ steps).}",
        r"\label{tab:scalability_mpc}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{rc ccc ccc ccc cc c}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{OSQP} & \multicolumn{3}{c}{TQPNet ($K{=}10$)} & \multicolumn{3}{c}{LQPNet ($K{=}10$)} & \multicolumn{2}{c}{Offline (s)} & Explicit \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-13} \cmidrule(lr){14-14}",
        r"$n{+}m$ & Speedup & RMSE & Cost & Latency (ms) & RMSE & Cost & Latency (ms) & RMSE & Cost & Latency (ms) & TQP & LQP & Latency (ms) \\",
        r"\midrule",
    ]
    for r in results:
        nm = r['n'] + r['m']
        sp_val = r["osqp_ms"] / max(r.get("tqp_K10_ms", 1), 0.01)
        ex = ex_lookup.get(nm)
        ex_str = f"{ex['online_median_ms']:.2f}" if ex else "---"
        lines.append(
            f"  {nm}"
            f" & ${sp_val:.1f}\\times$"
            f" & {r['osqp_rmse']:.3f} & {r['osqp_cost']:.0f} & {r['osqp_ms']:.2f}"
            f" & {r.get('tqp_K10_rmse',0):.3f} & {r.get('tqp_K10_cost',0):.0f} & {r.get('tqp_K10_ms',0):.2f}"
            f" & {r.get('lqp_K10_rmse',0):.3f} & {r.get('lqp_K10_cost',0):.0f} & {r.get('lqp_K10_ms',0):.2f}"
            f" & {r.get('tqp_offline_s',0):.0f} & {r.get('lqp_offline_s',0):.0f}"
            f" & {ex_str}"
            f" \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = FIG_DIR / "scalability_mpc_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {tex_path.name}")


if __name__ == "__main__":
    run_scalability_mpc()
    make_latex_table()
