"""Fig 2 PLOT: median + 5-95% percentile band of relative error over 100 random QPs.

Reads the .npz files dumped by `experiments/admm/admm_convergence.py` and produces
one PDF/PNG per A_scale.  Re-style here without re-running the experiment.

Usage:
    python -m experiments.admm.plot_admm_convergence
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Style (mirrors experiments/plot_paper.py) ────────────────────────────────────────

COLORS = {
    # Paper-wide convention: crimson = proposed, royalblue = baseline.
    # (Fig 3 K_t=10 → crimson; Fig 3 cold ADMM(ReLU) → royalblue)
    "admm":      "royalblue",   # full-dim 3-block ADMM (baseline)
    "admm_relu": "crimson",     # low-dim 2-block ADMM(ReLU) (proposed)
}

# Pick which convergence metric to plot — change here, no re-run needed.
#   "kkt"  : KKT residual         ‖r1‖²/n + ‖r2‖²/m
#   "abs"  : absolute error       ‖xᵏ − x⋆‖₂
#   "rel"  : relative error       ‖xᵏ − x⋆‖₂ / ‖x⋆‖₂
METRIC = "rel"
METRIC_LABELS = {
    "kkt": "KKT Residual",
    "abs": r"$\|x^{k} - x^{\star}\|_{2}$",
    "rel": r"$\|x^{k} - x^{\star}\|_{2} / \|x^{\star}\|_{2}$",
}

SCALE_TAG = "n100"   # match admm_convergence.py SCALE_TAG; "n10"/"n100"/"n500" supported
DATA_DIR  = (Path(__file__).resolve().parent
             / "runs" / "admm_convergence" / f"data_{SCALE_TAG}")
OUT_DIR  = (Path(__file__).resolve().parent
            / "runs" / "paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _setup():
    # Mirrors MATLAB main_v1.m: box on, all 4 spines at lw=1.2, FontSize=10.
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "axes.spines.top":   True,
        "axes.spines.right": True,
        "axes.linewidth":    1.2,
        "grid.alpha":        0.20,
    })


def _save(fig, name):
    for sfx in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"{name}{sfx}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {name}{{.png,.pdf}}")


def plot_one(npz_path: Path):
    d = np.load(npz_path)
    # New experiment dump records 3 metrics; legacy dumps used res3/res2.
    if f"{METRIC}3" in d.files:
        res3, res2 = d[f"{METRIC}3"], d[f"{METRIC}2"]
    else:
        res3, res2 = d["res3"], d["res2"]            # legacy fallback
    A_scale = float(d["A_scale"])
    K = res3.shape[1]
    iters = np.arange(K)

    p3 = np.percentile(res3, [5, 50, 95], axis=0)
    p2 = np.percentile(res2, [5, 50, 95], axis=0)
    floor = 1e-16
    p3 = np.maximum(p3, floor)
    p2 = np.maximum(p2, floor)

    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))   # MATLAB-like aspect, more room for xticks
    # MATLAB main_v1.m order: bands first then median lines on top.
    ax.fill_between(iters, p3[0], p3[2], alpha=0.18, color=COLORS["admm"],      lw=0)
    ax.fill_between(iters, p2[0], p2[2], alpha=0.18, color=COLORS["admm_relu"], lw=0)
    ax.semilogy(iters, p3[1], "-", color=COLORS["admm"],      lw=2.0)
    ax.semilogy(iters, p2[1], "-", color=COLORS["admm_relu"], lw=2.0)

    # MATLAB-style legend: bands listed first, then medians.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLORS["admm"],      alpha=0.18, label="Full-dim ADMM 5%–95%"),
        Patch(facecolor=COLORS["admm_relu"], alpha=0.18, label="Low-dim ADMM(ReLU) 5%–95%"),
        Line2D([0], [0], color=COLORS["admm"],      lw=2.0, label="Full-dim ADMM"),
        Line2D([0], [0], color=COLORS["admm_relu"], lw=2.0, label="Low-dim ADMM(ReLU)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.95,
              handlelength=1.8, edgecolor="0.4")

    ax.set_xlabel("ADMM Iteration")
    ax.set_ylabel(METRIC_LABELS.get(METRIC, "KKT Residual"))
    # X-axis: ticks every 1000, full integer labels.
    from matplotlib.ticker import MultipleLocator, LogLocator
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.tick_params(axis="x", labelsize=8)
    # Y-axis (log): minor ticks at 2,3,…,9 within each decade for fine grid lines.
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    # Axis bounds: x strictly [0, K], y log [1e-6, 1e1].
    ax.set_xlim(0, K)
    ax.set_ylim(1e-6, 1e1)
    ax.margins(x=0)
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    _save(fig, f"admm_convergence_a{int(A_scale)}")


def main():
    files = sorted(DATA_DIR.glob("admm_convergence_a*.npz"))
    if not files:
        print(f"[skip] no .npz files in {DATA_DIR}; run experiments.admm.admm_convergence first")
        return
    for f in files:
        plot_one(f)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
