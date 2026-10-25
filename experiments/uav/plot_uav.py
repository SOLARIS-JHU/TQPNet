"""Plot UAV MPC closed-loop figures.

Reads:
  experiments/runs/uav_mpc/refine_sweep_data.npz
  experiments/runs/uav_mpc/refine_sweep_summary.json

Outputs:
  experiments/runs/uav_mpc/figures/uav_trajectory.{png,pdf}
  experiments/runs/uav_mpc/figures/uav_tracking_error.{png,pdf}
  experiments/runs/uav_mpc/figures/uav_tradeoff.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UAV_DIR = Path(__file__).resolve().parent / "runs" / "uav_mpc"
FIG_DIR = UAV_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["royalblue", "seagreen", "mediumpurple", "crimson"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def savefig(fig, stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem.name}{{.png,.pdf}}")


def _arr_key(method: str) -> str:
    """Convert method label to npz array key. e.g. 'K=5' → 'K5', 'OSQP' → 'OSQP'."""
    return method.replace("=", "").replace(" ", "_")


DATA_DIR = UAV_DIR / "data"


def make_latex_table():
    summary_path = DATA_DIR / "refine_sweep_summary.json"
    if not summary_path.exists():
        print(f"  [skip] {summary_path.name} not found")
        return
    summary = load_json(summary_path)

    # All methods: cold (K<-1), TQPNet (K>=0), OSQP
    cold  = sorted([(v["K"], m) for m, v in summary.items() if v["K"] < -1])
    tqp   = sorted([(v["K"], m) for m, v in summary.items() if v["K"] >= 0])
    ordered = [m for _, m in cold] + [m for _, m in tqp] + ["OSQP"]
    osqp_ms = summary["OSQP"]["ms_per_step"]

    disp = {}
    for m in ordered:
        if m.startswith("K="):
            disp[m] = f"TQPNet ({m})"
        else:
            disp[m] = m

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{UAV Lemniscate MPC: accuracy--speed trade-off ($N_H=25$, $n=100$, $T=600$)}",
        r"\label{tab:uav_mpc}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & RMSE (m) & Time (ms/step) & Speedup \\",
        r"\midrule",
    ]
    for m in ordered:
        r       = summary[m]
        speedup = osqp_ms / r["ms_per_step"]
        sp_str  = f"${speedup:.1f}\\times$"
        lines.append(
            f"{disp[m]} & {r['mean_rmse']:.4f} & {r['ms_per_step']:.2f} & {sp_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = FIG_DIR / "uav_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {tex_path.name}")


def plot_uav():
    npz_path     = UAV_DIR / "refine_sweep_data.npz"
    summary_path = DATA_DIR / "refine_sweep_summary.json"
    if not npz_path.exists():
        print(f"  [skip] {npz_path.name} not found — run uav_benchmark.py first")
        return

    data    = np.load(npz_path)
    summary = load_json(summary_path)

    # Trajectory + tracking-error figures: keep one representative per family.
    # Plot order (drawn back-to-front so TQPNet is on top in trajectory figure):
    #   ADMM(ReLU) cold  →  OSQP  →  TQPNet (K=30)
    cold = sorted([(v["K"], m) for m, v in summary.items() if v["K"] < -1])
    tqp_K30 = "K=30" if "K=30" in summary else None
    ordered = [m for _, m in cold] + ["OSQP"] + ([tqp_K30] if tqp_K30 else [])

    # Color convention (matches main paper):
    #   ADMM(ReLU) cold → royalblue (cool, baseline)
    #   TQPNet         → crimson    (warm, proposed)
    #   OSQP           → darkorange (warm-different, classical)
    palette = {"cold": "#1f77b4", "tqp": "#d62728", "osqp": "#ff7f0e"}
    def _color(m):
        if m.startswith("ADMM"):  return palette["cold"]
        if m.startswith("K="):    return palette["tqp"]
        if m == "OSQP":           return palette["osqp"]
        return "gray"
    colors = {m: _color(m) for m in ordered}

    disp = {}
    for m in ordered:
        if m.startswith("K="):
            disp[m] = f"TQPNet ({m})"
        else:
            disp[m] = m

    try:
        from experiments.uav.uav_physics import DT
        from experiments.uav.uav_tqpnet import T_SIM, N_H
        from experiments.uav.uav_physics import N_U
        n_dec = N_H * N_U
        title_suffix = f"  (N_H={N_H}, n={n_dec})"
    except Exception:
        err_arr      = data[f"err_{_arr_key(ordered[0])}"]
        T_SIM, DT    = len(err_arr), 0.071
        title_suffix = ""

    T_ax  = np.arange(1, T_SIM + 1) * DT
    t_ref = np.linspace(0, T_SIM, 1000)
    rx    = 1.0 * np.sin(0.03 * t_ref)
    ry    = 1.0 * np.sin(0.03 * t_ref) * np.cos(0.03 * t_ref)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8})

    # Line style per method for consistent B&W readability
    ls_map = {}; mk_map = {}
    markers_cycle = ["o", "s", "^", "D", "P"]
    ls_cycle = [":", "-", "-.", "-", "--"]
    for i, m in enumerate(ordered):
        ls_map[m] = ls_cycle[i % len(ls_cycle)]
        mk_map[m] = markers_cycle[i % len(markers_cycle)]

    # ── Figure 1: XY trajectory ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(rx, ry, "k--", lw=2.5, alpha=0.55, zorder=5, label="Reference")
    for method in ordered:
        traj = data[f"traj_{_arr_key(method)}"]
        c = colors[method]
        lw = 2.0 if method == "OSQP" else 1.8
        ax.plot(traj[:, 3], traj[:, 4], color=c, lw=lw, ls=ls_map[method],
                label=disp[method])
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.78, 0.78)
    ax.set_title("UAV Lemniscate Tracking")
    ax.legend(ncol=2, loc="upper center", fontsize=6.5)
    ax.grid(True, alpha=0.3)
    savefig(fig, FIG_DIR / "uav_trajectory")

    # ── Figure 2: Tracking error over time ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for method in ordered:
        err = data[f"err_{_arr_key(method)}"]
        c = colors[method]
        lw = 2.0 if method == "OSQP" else 1.8
        mark_every = max(1, len(T_ax) // 12)
        ax.plot(T_ax, err, color=c, lw=lw, ls=ls_map[method],
                marker=mk_map[method], markersize=4, markevery=mark_every,
                markeredgecolor="white", markeredgewidth=0.3,
                label=disp[method])
    ax.set_yscale("log")
    ax.set_xlabel("Time $t$ (s)"); ax.set_ylabel("Position error (m)")
    ax.set_title("Lemniscate Tracking Error")
    ax.legend(ncol=2, loc="lower center", fontsize=6.5)
    ax.grid(True, alpha=0.3, which="both")
    savefig(fig, FIG_DIR / "uav_tracking_error")

    # ── Figure 3: Accuracy–speed tradeoff ────────────────────────────────────
    markers = ["o", "s", "^", "P", "D", "v"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for i, method in enumerate(ordered):
        info = summary[method]
        ax.scatter(info["ms_per_step"], info["mean_rmse"],
                   color=colors[method], marker=markers[i % len(markers)],
                   s=130, zorder=5, label=disp[method])
        ax.annotate(disp[method], (info["ms_per_step"], info["mean_rmse"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    ax.set_xlabel("Solve time per step (ms)")
    ax.set_ylabel("Mean position RMSE (m)")
    ax.set_title("Accuracy–Speed Trade-off  (Lemniscate MPC)")
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, FIG_DIR / "uav_tradeoff")


if __name__ == "__main__":
    make_latex_table()
    plot_uav()
