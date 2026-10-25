"""Generate publication-quality individual figures for NeurIPS paper.

Each subplot is a separate file. LaTeX handles subfigure layout.
Outputs to experiments/runs/paper_figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────

COLORS = {
    "tqp":      "crimson",
    "lqp":      "seagreen",
    "osqp":     "darkorange",
    "cold":     "royalblue",
    "fulldim":  "gray",
    "explicit": "purple",
}

EXP_DIR = Path(__file__).resolve().parent
RUN_DIR = EXP_DIR / "runs"
OUT_DIR = RUN_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def _save(fig, name):
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"{name}{suffix}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {name}{{.png,.pdf}}")


def _load(path):
    with open(path) as f:
        return json.load(f)


# ── Generalizability Violin ──────────────────────────────────────────────────

def plot_generalizability():
    _setup()
    data_path = RUN_DIR / "generalizability" / "data_n100" / "generalizability_results.json"
    err_path = RUN_DIR / "generalizability" / "data_n100" / "generalizability_errors.json"
    if not data_path.exists() or not err_path.exists():
        print("  [skip] generalizability not found"); return

    data = _load(data_path)
    all_errors = _load(err_path)

    sets = data["param_sets"]
    set_names = [s["name"] for s in sets]
    K_list = data["K_list"]
    n_K = len(K_list)
    tqp_c, lqp_c = COLORS["tqp"], COLORS["lqp"]
    alphas = [0.35, 0.55, 0.8][:n_K]

    group_width = 2 * n_K + 1
    fig, ax = plt.subplots(figsize=(6.75, 3.2))

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
                body.set_facecolor(tqp_c); body.set_alpha(alphas[i])
            vp_t['cmedians'].set_color("darkred"); vp_t['cmedians'].set_linewidth(1.5)

            vp_l = ax.violinplot([lqp_vals], positions=[pos_l], showmedians=True,
                                  showextrema=False, widths=0.8)
            for body in vp_l['bodies']:
                body.set_facecolor(lqp_c); body.set_alpha(alphas[i])
            vp_l['cmedians'].set_color("darkgreen"); vp_l['cmedians'].set_linewidth(1.5)

    ax.set_yscale("log")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(set_names, fontsize=10)
    ax.set_ylabel("Relative Error $\\|x_K - x^*\\| / \\|x^*\\|$")
    ax.grid(True, which="both", alpha=0.15, axis="y")

    from matplotlib.patches import Patch
    handles = []
    for i, K in enumerate(K_list):
        handles.append(Patch(facecolor=tqp_c, alpha=alphas[i], label=f"TQPNet ($K{{=}}{K}$)"))
        handles.append(Patch(facecolor=lqp_c, alpha=alphas[i], label=f"LQPNet ($K{{=}}{K}$)"))
    ax.legend(handles=handles, fontsize=7, ncol=n_K, loc="upper left")

    _save(fig, "generalizability_violin")


# ── Scalability: Latency ─────────────────────────────────────────────────────

def plot_scalability_latency():
    _setup()
    path = RUN_DIR / "scalability_mpc" / "scalability_mpc_results.json"
    if not path.exists():
        print("  [skip] scalability not found"); return
    results = _load(path)

    explicit_path = EXP_DIR / "explicit_mpc" / "explicit_mpc_results.json"
    explicit_data = []
    if explicit_path.exists():
        explicit_data = [r for r in _load(explicit_path) if r.get("status") == "success"]

    nm = [r["n"] + r["m"] for r in results]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.semilogy(nm, [r["osqp_ms"] for r in results], "D-",
                color=COLORS["osqp"], ms=5, **mk_kw, label="OSQP")
    ax.semilogy(nm, [r.get("cold_ms", 0) for r in results], "o:",
                color=COLORS["cold"], lw=1.2, ms=5, **mk_kw, label="ADMM(ReLU) ($K{=}50$)")
    ax.semilogy(nm, [r.get("tqp_K10_ms", 0) for r in results], "s-",
                color=COLORS["tqp"], ms=5, **mk_kw, label="TQPNet ($K{=}10$)")
    if "lqp_K10_ms" in results[0]:
        ax.semilogy(nm, [r.get("lqp_K10_ms", 0) for r in results], "^--",
                    color=COLORS["lqp"], lw=1.5, ms=5, **mk_kw, label="LQPNet ($K{=}10$)")
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        ex_ms = [r["online_median_ms"] for r in explicit_data]
        ax.semilogy(ex_nm, ex_ms, "x-", color=COLORS["explicit"], lw=1.5, ms=5,
                    markeredgewidth=2, label="Explicit MPC")
    ax.set_xlabel("$n + m$")
    ax.set_ylabel("Solve Latency (ms)")
    ax.legend(fontsize=5.5, loc="lower right")
    ax.grid(True, which="both", alpha=0.15)
    _save(fig, "scalability_latency")


# ── Scalability: Offline Time ─────────────────────────────────────────────────

def plot_scalability_offline():
    _setup()
    path = RUN_DIR / "scalability_mpc" / "scalability_mpc_results.json"
    if not path.exists():
        print("  [skip] scalability not found"); return
    results = _load(path)

    explicit_path = EXP_DIR / "explicit_mpc" / "explicit_mpc_results.json"
    explicit_data = []
    if explicit_path.exists():
        explicit_data = [r for r in _load(explicit_path) if r.get("status") == "success"]

    nm = [r["n"] + r["m"] for r in results]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.semilogy(nm, [r.get("tqp_offline_s", 0) for r in results], "s-",
                color=COLORS["tqp"], ms=5, **mk_kw, label="TQPNet (unsupervised)")
    ax.semilogy(nm, [r.get("lqp_offline_s", 0) for r in results], "^-",
                color=COLORS["lqp"], ms=5, **mk_kw, label="LQPNet (supervised)")
    if explicit_data:
        ex_nm = [r["n_plus_m"] for r in explicit_data]
        ex_off = [r["offline_s"] for r in explicit_data]
        ax.semilogy(ex_nm, ex_off, "x-", color=COLORS["explicit"], lw=1.5, ms=5,
                    markeredgewidth=2, label="Explicit MPC")
    ax.set_xlabel("$n + m$")
    ax.set_ylabel("Offline Time (s)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(True, which="both", alpha=0.15)
    _save(fig, "scalability_offline")


# ── UAV: Trajectory ──────────────────────────────────────────────────────────

def plot_uav_trajectory():
    _setup()
    npz_path = RUN_DIR / "uav_mpc" / "refine_sweep_data.npz"
    if not npz_path.exists():
        print("  [skip] UAV data not found"); return
    d = np.load(npz_path)

    methods = [
        ("traj_ADMM-ReLU_cold_(K50)", COLORS["cold"], ":", 1.2, "ADMM(ReLU) ($K{=}50$)"),
        ("traj_K10", "royalblue",  "-", 1.5, "TQPNet ($K{=}10$)"),
        ("traj_K20", "seagreen",   "-", 1.5, "TQPNet ($K{=}20$)"),
        ("traj_K30", COLORS["tqp"],"-", 2.0, "TQPNet ($K{=}30$)"),
        ("traj_OSQP", COLORS["osqp"], "--", 1.8, "OSQP"),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for traj_k, color, ls, lw, label in methods:
        traj = d[traj_k]
        ax.plot(traj[:, 3], traj[:, 4], ls, color=color, lw=lw, label=label)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_ylim(-0.7, 0.9)
    ax.set_aspect("equal")
    ax.legend(fontsize=5.5, ncol=2, loc="upper center")
    ax.grid(True, alpha=0.15)
    _save(fig, "uav_trajectory")


# ── UAV: Tracking Error ──────────────────────────────────────────────────────

def plot_uav_tracking_error():
    _setup()
    npz_path = RUN_DIR / "uav_mpc" / "refine_sweep_data.npz"
    if not npz_path.exists():
        print("  [skip] UAV data not found"); return
    d = np.load(npz_path)

    methods = [
        ("err_ADMM-ReLU_cold_(K50)", COLORS["cold"], ":", 1.2, "ADMM(ReLU) ($K{=}50$)"),
        ("err_K10", "royalblue",  "-", 1.5, "TQPNet ($K{=}10$)"),
        ("err_K20", "seagreen",   "-", 1.5, "TQPNet ($K{=}20$)"),
        ("err_K30", COLORS["tqp"],"-", 2.0, "TQPNet ($K{=}30$)"),
        ("err_OSQP", COLORS["osqp"], "--", 1.8, "OSQP"),
    ]

    dt = 0.071
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for err_k, color, ls, lw, label in methods:
        err = d[err_k]
        t = np.arange(len(err)) * dt
        ax.plot(t, err, ls, color=color, lw=lw, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tracking Error (m)")
    ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.15)
    _save(fig, "uav_tracking_error")


# ── Scalability: RMSE ─────────────────────────────────────────────────────────

def plot_scalability_rmse():
    _setup()
    path = RUN_DIR / "scalability_mpc" / "scalability_mpc_results.json"
    if not path.exists():
        print("  [skip] scalability not found"); return
    results = _load(path)
    nm = [r["n"] + r["m"] for r in results]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot(nm, [r["osqp_rmse"] for r in results], "D-",
            color=COLORS["osqp"], ms=5, **mk_kw, label="OSQP")
    ax.plot(nm, [r.get("cold_rmse", 0) for r in results], "o:",
            color=COLORS["cold"], lw=1.2, ms=5, **mk_kw, label="ADMM(ReLU) ($K{=}50$)")
    ax.plot(nm, [r.get("tqp_K10_rmse", 0) for r in results], "s-",
            color=COLORS["tqp"], ms=5, **mk_kw, label="TQPNet ($K{=}10$)")
    if "lqp_K10_rmse" in results[0]:
        ax.plot(nm, [r.get("lqp_K10_rmse", 0) for r in results], "^--",
                color=COLORS["lqp"], lw=1.5, ms=5, **mk_kw, label="LQPNet ($K{=}10$)")
    ax.set_xlabel("$n + m$")
    ax.set_ylabel("Tracking RMSE")
    ax.legend(fontsize=5.5, loc="upper right")
    ax.grid(True, alpha=0.15)
    _save(fig, "scalability_rmse")


# ── Scalability: Cost ─────────────────────────────────────────────────────────

def plot_scalability_cost():
    _setup()
    path = RUN_DIR / "scalability_mpc" / "scalability_mpc_results.json"
    if not path.exists():
        print("  [skip] scalability not found"); return
    results = _load(path)
    nm = [r["n"] + r["m"] for r in results]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.plot(nm, [r["osqp_cost"] for r in results], "D-",
            color=COLORS["osqp"], ms=5, **mk_kw, label="OSQP")
    ax.plot(nm, [r.get("cold_cost", 0) for r in results], "o:",
            color=COLORS["cold"], lw=1.2, ms=5, **mk_kw, label="ADMM(ReLU) ($K{=}50$)")
    ax.plot(nm, [r.get("tqp_K10_cost", 0) for r in results], "s-",
            color=COLORS["tqp"], ms=5, **mk_kw, label="TQPNet ($K{=}10$)")
    if "lqp_K10_cost" in results[0]:
        ax.plot(nm, [r.get("lqp_K10_cost", 0) for r in results], "^--",
                color=COLORS["lqp"], lw=1.5, ms=5, **mk_kw, label="LQPNet ($K{=}10$)")
    ax.set_xlabel("$n + m$")
    ax.set_ylabel("Closed-loop Cost")
    ax.legend(fontsize=5.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    _save(fig, "scalability_cost")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generating paper figures ===")
    plot_generalizability()
    plot_scalability_latency()
    plot_scalability_offline()
    plot_scalability_rmse()
    plot_scalability_cost()
    plot_uav_trajectory()
    plot_uav_tracking_error()
    print(f"\nAll saved to {OUT_DIR}")
