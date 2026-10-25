"""Generalizability experiment on random parametric QP (n=10, m=20).

Evaluate TQPNet and LQPNet on in-distribution, boundary, and OOD parameter
sets. Models trained on p ~ U[-1,1]^np_dim.

Metric: relative solution error ||x_K - x*|| / ||x*|| vs OSQP optimal.

Output: experiments/runs/generalizability/
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.base import (
    ProblemConfig, build_config, build_problem_data,
    TQPNet, LQPNet, admm_refine,
    solve_osqp_reference, admm_unrolled_loss,
    device,
)
from experiments.common import RUN_DIR, subdirs, setup_logger, write_json
from experiments.config import (
    A_DENSITY, A_SCALE, PROBLEM_SEED, RHO_REFINE,
    BATCH_SIZE, LR, WEIGHT_DECAY, LOSS_THRESHOLD, LOG_EVERY_EPOCH,
)

# ── Scale config ──────────────────────────────────────────────────────────
SCALE_TAG = "n100"
_SCALES   = {
    "n10":  (10,  20,  4,  256),
    "n100": (100, 200, 20, 512),
    "n500": (500, 500, 50, 512),
}
N_DEC, M_CON, NP_DIM, W_NET = _SCALES[SCALE_TAG]
L_NET = 4

OUT_DIR    = RUN_DIR / "generalizability" / f"data_{SCALE_TAG}"
MODEL_DIR  = RUN_DIR / "generalizability" / f"models_{SCALE_TAG}"

N_TRAIN    = 6000
MAX_EPOCHS = 2000
K_UNROLL   = 25

K_LIST = [10, 30, 50]
N_EVAL = 500

PARAM_SETS = [
    {"name": "Train",    "range": 1.0, "seed": 0},
    {"name": "Valid",    "range": 1.0, "seed": 999},
    {"name": "Boundary", "range": 2.0, "seed": 777},
    {"name": "OOD",      "range": 3.0, "seed": 555},
]


def train_models(cfg, logger):
    """Train TQPNet and LQPNet on p ~ U[-1,1]."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    P_train = -1.0 + 2.0 * torch.rand(N_TRAIN, NP_DIM, device=device)

    # TQPNet
    logger.info("Training TQPNet (n=%d m=%d N=%d)", N_DEC, M_CON, N_TRAIN)
    tqp = TQPNet(NP_DIM, N_DEC, M_CON, W_NET, L_NET).to(device)
    loader = DataLoader(TensorDataset(P_train), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(tqp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    for ep in range(1, MAX_EPOCHS + 1):
        tqp.train(); ep_loss = 0.0
        for (p_b,) in loader:
            Q_b, q_b, A_b, c_b = build_problem_data(p_b, cfg)
            opt.zero_grad()
            xp, bp = tqp(p_b)
            loss = admm_unrolled_loss(xp, bp, Q_b, q_b, A_b, c_b, cfg, K_unroll=K_UNROLL)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tqp.parameters(), 10.0)
                opt.step(); ep_loss += loss.item() * p_b.shape[0]
        sched.step(); ep_loss /= N_TRAIN
        if ep % LOG_EVERY_EPOCH == 0 or ep == MAX_EPOCHS:
            logger.info("  TQP ep=%4d loss=%.4e", ep, ep_loss)
        if ep_loss < LOSS_THRESHOLD:
            logger.info("  TQP converged ep=%d", ep); break
    tqp.eval()

    # LQPNet (supervised)
    logger.info("Generating LQPNet labels...")
    Q_b, q_b, A_b, c_b = build_problem_data(P_train, cfg)
    X_star = torch.zeros(N_TRAIN, N_DEC); valid = []
    for i in range(N_TRAIN):
        xv, _, _ = solve_osqp_reference(
            Q_b[i].cpu().numpy(), q_b[i].cpu().numpy(),
            A_b[i].cpu().numpy(), c_b[i].cpu().numpy(), N_DEC)
        if xv is not None and not np.isnan(xv).any():
            X_star[i] = torch.tensor(xv, dtype=torch.float32); valid.append(i)
    logger.info("  Labels: %d/%d valid", len(valid), N_TRAIN)

    from torch.utils.data import Dataset
    class SD(Dataset):
        def __init__(self, P, X): self.P, self.X = P.float(), X.float()
        def __len__(self): return len(self.P)
        def __getitem__(self, i): return self.P[i], self.X[i]

    lqp = LQPNet(NP_DIM, N_DEC, W_NET, L_NET).to(device)
    loader_l = DataLoader(SD(P_train[valid].cpu(), X_star[valid]),
                          batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(lqp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    logger.info("Training LQPNet")
    for ep in range(1, MAX_EPOCHS + 1):
        lqp.train(); ep_loss = 0.0
        for p_b, x_b in loader_l:
            p_b, x_b = p_b.to(device), x_b.to(device)
            opt.zero_grad()
            loss = torch.log1p(((lqp(p_b) - x_b)**2).sum(1).mean() / N_DEC)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lqp.parameters(), 10.0)
                opt.step(); ep_loss += loss.item() * p_b.shape[0]
        sched.step(); ep_loss /= max(len(valid), 1)
        if ep % LOG_EVERY_EPOCH == 0 or ep == MAX_EPOCHS:
            logger.info("  LQP ep=%4d loss=%.4e", ep, ep_loss)
        if ep_loss < LOSS_THRESHOLD:
            logger.info("  LQP converged ep=%d", ep); break
    lqp.eval()
    return tqp, lqp


def compute_osqp_solutions(P_set, cfg):
    """Solve each QP with OSQP to get x*."""
    Q_b, q_b, A_b, c_b = build_problem_data(P_set, cfg)
    X_star = np.full((len(P_set), cfg.n), np.nan)
    for i in range(len(P_set)):
        xv, _, _ = solve_osqp_reference(
            Q_b[i].cpu().numpy(), q_b[i].cpu().numpy(),
            A_b[i].cpu().numpy(), c_b[i].cpu().numpy(), cfg.n)
        if xv is not None and not np.isnan(xv).any():
            X_star[i] = xv
    return X_star


def evaluate_rel_error(model, cfg, P_set, K_list, has_beta, X_star):
    """Compute relative error ||x_K - x*|| / ||x*|| for each K.
    Returns dict: K -> array of per-instance relative errors.
    """
    Q_b, q_b, A_b, c_b = build_problem_data(P_set, cfg)
    valid = ~np.isnan(X_star[:, 0])
    x_star_norms = np.linalg.norm(X_star[valid], axis=1).clip(min=1e-10)

    results = {}
    for K in K_list:
        with torch.no_grad():
            if has_beta:
                xi, bi = model(P_set)
            else:
                xi = model(P_set)
                bi = None
            xr, _, _, _, _ = admm_refine(xi, Q_b, q_b, A_b, c_b, cfg,
                                          K=K, tol=None, beta0=bi)
        xr_np = xr.cpu().numpy()[valid]
        err = np.linalg.norm(xr_np - X_star[valid], axis=1) / x_star_norms
        results[K] = err
    return results


def run_generalizability():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = subdirs(OUT_DIR)
    logger = setup_logger("general", dirs["logs"] / "generalizability.log")
    logger.info("Generalizability | random QP n=%d m=%d np=%d", N_DEC, M_CON, NP_DIM)

    cfg = build_config(N_DEC, M_CON, NP_DIM, seed=PROBLEM_SEED, rho_refine=RHO_REFINE,
                       density=A_DENSITY, a_scale=A_SCALE, c_offset=1.0)

    # Train or load models
    model_dir = dirs["models"]
    tqp_path = model_dir / "tqpnet.pth"
    lqp_path = model_dir / "lqpnet.pth"

    if tqp_path.exists() and lqp_path.exists():
        logger.info("Loading cached models")
        tqp = TQPNet(NP_DIM, N_DEC, M_CON, W_NET, L_NET).to(device)
        lqp = LQPNet(NP_DIM, N_DEC, W_NET, L_NET).to(device)
        tqp.load_state_dict(torch.load(tqp_path, map_location=device)); tqp.eval()
        lqp.load_state_dict(torch.load(lqp_path, map_location=device)); lqp.eval()
    else:
        tqp, lqp = train_models(cfg, logger)
        torch.save(tqp.state_dict(), tqp_path)
        torch.save(lqp.state_dict(), lqp_path)

    results = []
    all_errors = {}  # for violin plot: {set_name: {method_K: array}}

    for pset in PARAM_SETS:
        rng = np.random.default_rng(pset["seed"])
        P_np = rng.uniform(-pset["range"], pset["range"],
                           (N_EVAL, NP_DIM)).astype(np.float32)
        P_t = torch.tensor(P_np, device=device)

        logger.info("═══ %s (range=%.1f, N=%d) ═══", pset["name"], pset["range"], N_EVAL)

        X_star = compute_osqp_solutions(P_t, cfg)
        n_valid = int(np.sum(~np.isnan(X_star[:, 0])))
        logger.info("  OSQP: %d/%d valid", n_valid, N_EVAL)

        tqp_errs = evaluate_rel_error(tqp, cfg, P_t, K_LIST, has_beta=True, X_star=X_star)
        lqp_errs = evaluate_rel_error(lqp, cfg, P_t, K_LIST, has_beta=False, X_star=X_star)

        row = {"name": pset["name"], "range": pset["range"]}
        set_errors = {}
        for K in K_LIST:
            tqp_med = float(np.median(tqp_errs[K]))
            lqp_med = float(np.median(lqp_errs[K]))
            tqp_mean = float(np.mean(tqp_errs[K]))
            lqp_mean = float(np.mean(lqp_errs[K]))
            logger.info("  K=%2d  TQP median=%.2e mean=%.2e  LQP median=%.2e mean=%.2e",
                        K, tqp_med, tqp_mean, lqp_med, lqp_mean)
            row[f"tqp_median_K{K}"] = tqp_med
            row[f"tqp_mean_K{K}"] = tqp_mean
            row[f"lqp_median_K{K}"] = lqp_med
            row[f"lqp_mean_K{K}"] = lqp_mean
            set_errors[f"tqp_K{K}"] = tqp_errs[K].tolist()
            set_errors[f"lqp_K{K}"] = lqp_errs[K].tolist()
        results.append(row)
        all_errors[pset["name"]] = set_errors

    output = {"n": N_DEC, "m": M_CON, "np_dim": NP_DIM,
              "K_list": K_LIST, "param_sets": results}
    write_json(OUT_DIR / "generalizability_results.json", output)
    # Save raw errors separately (large)
    write_json(OUT_DIR / "generalizability_errors.json", all_errors)
    logger.info("Saved results")

    plot_generalizability(output, all_errors)
    make_generalizability_table(output)
    logger.info("Done.")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_generalizability(data=None, all_errors=None):
    if data is None:
        path = OUT_DIR / "generalizability_results.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
    if all_errors is None:
        err_path = OUT_DIR / "generalizability_errors.json"
        if not err_path.exists():
            return
        with open(err_path) as f:
            all_errors = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8})

    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        for suffix in (".png", ".pdf"):
            fig.savefig(FIG_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}{{.png,.pdf}}")

    sets = data["param_sets"]
    set_names = [s["name"] for s in sets]
    K_list = data["K_list"]
    n_K = len(K_list)
    tqp_c, lqp_c = "crimson", "seagreen"
    # Darker alpha for larger K
    alphas = [0.35, 0.55, 0.8][:n_K]

    # Single figure: x-axis grouped by set, within each group K pairs
    # Layout: for each set, 2*n_K violins (TQP/LQP pairs per K)
    group_width = 2 * n_K + 1  # spacing between groups
    fig, ax = plt.subplots(figsize=(10, 5))

    group_centers = []
    for j, sname in enumerate(set_names):
        errs = all_errors[sname]
        base = j * group_width
        group_centers.append(base + n_K - 0.5)

        for i, K in enumerate(K_list):
            tqp_vals = np.array(errs[f"tqp_K{K}"])
            lqp_vals = np.array(errs[f"lqp_K{K}"])

            pos_t = base + i * 2
            pos_l = base + i * 2 + 1

            vp_t = ax.violinplot([tqp_vals], positions=[pos_t], showmedians=True,
                                  showextrema=False, widths=0.8)
            for body in vp_t['bodies']:
                body.set_facecolor(tqp_c)
                body.set_alpha(alphas[i])
            vp_t['cmedians'].set_color("darkred")
            vp_t['cmedians'].set_linewidth(1.5)

            vp_l = ax.violinplot([lqp_vals], positions=[pos_l], showmedians=True,
                                  showextrema=False, widths=0.8)
            for body in vp_l['bodies']:
                body.set_facecolor(lqp_c)
                body.set_alpha(alphas[i])
            vp_l['cmedians'].set_color("darkgreen")
            vp_l['cmedians'].set_linewidth(1.5)

    ax.set_yscale("log")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(set_names, fontsize=11)
    ax.set_ylabel("Relative Error $\\|x_K - x^*\\| / \\|x^*\\|$")
    ax.set_title("Generalizability: Relative Solution Error")
    ax.grid(True, which="both", alpha=0.2, axis="y")

    # Legend: method + K intensity
    from matplotlib.patches import Patch
    handles = []
    for i, K in enumerate(K_list):
        handles.append(Patch(facecolor=tqp_c, alpha=alphas[i], label=f"TQPNet $K$={K}"))
        handles.append(Patch(facecolor=lqp_c, alpha=alphas[i], label=f"LQPNet $K$={K}"))
    ax.legend(handles=handles, fontsize=8, ncol=n_K, loc="upper left")

    fig.tight_layout()
    _save(fig, "generalizability_violin")

    print("  generalizability plots done")


def make_generalizability_table(data=None):
    if data is None:
        path = OUT_DIR / "generalizability_results.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)

    sets = data["param_sets"]
    K_list = data["K_list"]
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_k = len(K_list)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Generalizability: median relative error ($n={data['n']}$, $m={data['m']}$). "
        r"Train on $p \sim \mathcal{U}[-1,1]$.}",
        r"\label{tab:generalizability}",
        r"\begin{tabular}{l" + "cc" * n_k + "}",
        r"\toprule",
    ]
    h1 = "Set"
    for K in K_list:
        h1 += rf" & \multicolumn{{2}}{{c}}{{$K={K}$}}"
    h1 += r" \\"
    lines.append(h1)
    for i, K in enumerate(K_list):
        col_s = 2 + i * 2
        lines.append(rf"\cmidrule(lr){{{col_s}-{col_s+1}}}")
    h2 = ""
    for _ in K_list:
        h2 += r" & TQPNet & LQPNet"
    h2 += r" \\"
    lines.append(h2)
    lines.append(r"\midrule")

    for s in sets:
        cols = [s["name"]]
        for K in K_list:
            cols.append(f"{s[f'tqp_median_K{K}']:.1e}")
            cols.append(f"{s[f'lqp_median_K{K}']:.1e}")
        lines.append(" & ".join(cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = FIG_DIR / "generalizability_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {tex_path.name}")


if __name__ == "__main__":
    run_generalizability()
