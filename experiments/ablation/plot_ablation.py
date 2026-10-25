"""Fig 3 PLOT + summary table: training (a) / inference iter (b) / inference time (appendix) / table.

Reads:
    experiments/runs/ablation/data/training_ablation_K*.npz     (Fig 3a)
    experiments/runs/ablation/data/inference_ablation_K*.npz    (Fig 3b, table iter)
    experiments/runs/ablation/data/inference_time_K*.npz +
        experiments/runs/ablation/data/inference_time_cold.npz  (appendix inference-time figure, table time)

Outputs (to experiments/runs/paper_figures/):
    training_ablation.{png,pdf}
    inference_ablation.{png,pdf}
    inference_time.{png,pdf}
    ablation_table.tex

Usage:
    python -m experiments.ablation.plot_ablation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Style (mirrors experiments/plot_paper.py) ────────────────────────────────────────

# K_train → (color, linestyle, lw)
# K_t=10 uses crimson because subsequent experiments (scalability / uav) all use
# K_t=10 as the "TQPNet" setting — keeping the same color across the paper.
STYLE_K_TRAIN = {
    # All K_t curves: solid + color-only differentiation.
    # K_t=10 → crimson (matches TQPNet (K=10) elsewhere in paper).
    # K_t=25/50 use cool tones (green / navy) so they don't compete visually.
    # ADMM(ReLU) cold (royalblue + `:`) preserved separately to match plot_paper.py.
    0:  {"color": "gray",       "ls": "-", "lw": 1.8, "dashes": None},
    1:  {"color": "purple",     "ls": "-", "lw": 1.8, "dashes": None},
    5:  {"color": "darkorange", "ls": "-", "lw": 1.8, "dashes": None},
    10: {"color": "crimson",    "ls": "-", "lw": 2.0, "dashes": None},
    25: {"color": "darkgreen",  "ls": "-", "lw": 2.0, "dashes": None},
    50: {"color": "navy",       "ls": "-", "lw": 2.2, "dashes": None},   # candidate-best
}

COLOR_COLD = "royalblue"   # matches plot_paper.py "cold" convention
COLOR_FULL = "gray"        # full-dim 3-block ADMM (vanilla baseline)
# KKT residual target — n=10 reaches 1e-8; n=100 plateaus near 3e-8, so use 1e-7.
SCALE_TAG = "n100"    # aligned with Fig 2 admm_convergence; "n10"/"n100"/"n500"
TARGET    = 1e-8 if SCALE_TAG == "n10" else 1e-7
_SCALE_DESC = {"n10": r"$n{=}10$, $m{=}20$",
               "n100": r"$n{=}100$, $m{=}200$",
               "n500": r"$n{=}500$, $m{=}1000$"}

# Pick which convergence metric to plot for Fig 3(b) and the appendix inference-time figure — change here, no re-run needed.
#   "kkt"  : KKT residual         ‖r1‖²/n + ‖r2‖²/m
#   "abs"  : absolute error       ‖xᵏ − x⋆‖₂
#   "rel"  : relative error       ‖xᵏ − x⋆‖₂ / ‖x⋆‖₂
METRIC_INFER = "kkt"
METRIC_LABELS = {
    "kkt": "KKT Residual",
    "abs": r"$\|x^{k} - x^{\star}\|_{2}$",
    "rel": r"$\|x^{k} - x^{\star}\|_{2} / \|x^{\star}\|_{2}$",
}

DATA_DIR  = Path(__file__).resolve().parent / "runs" / "ablation" / f"data_{SCALE_TAG}"
OUT_DIR   = Path(__file__).resolve().parent / "runs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_SAVE_SUFFIX = ""   # appended to output stems; main() sets per scale


def _set_scale(tag: str):
    """Switch active scale: changes SCALE_TAG, TARGET, DATA_DIR, _SAVE_SUFFIX."""
    global SCALE_TAG, TARGET, DATA_DIR, _SAVE_SUFFIX
    SCALE_TAG = tag
    TARGET    = 1e-8 if tag == "n10" else 1e-7
    DATA_DIR  = Path(__file__).resolve().parent / "runs" / "ablation" / f"data_{tag}"
    _SAVE_SUFFIX = f"_{tag}"


def _setup():
    # Match plot_admm_convergence MATLAB-style polish: box on, fontsize 10,
    # axis linewidth 1.2, generous fontsize for paper visibility.
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
    # Fixed margins → all sub-figures share identical canvas dimensions.
    fig.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.18)
    stem = f"{name}{_SAVE_SUFFIX}"
    for sfx in (".png", ".pdf"):
        fig.savefig(OUT_DIR / f"{stem}{sfx}", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}{{.png,.pdf}}")


def _label(K_train: int) -> str:
    if K_train == 0:
        return r"MLP(ReLU) only"
    return rf"MLP(ReLU) + {K_train}-layer ADMM(ReLU)"


def _line_kw(st):
    """Build line kwargs, honoring custom `dashes` if present."""
    base = {"color": st["color"], "lw": st["lw"]}
    if st.get("dashes"):
        return {**base, "dashes": st["dashes"]}
    return {**base, "linestyle": st["ls"]}


# ── Fig 3(a) — training_ablation.pdf ─────────────────────────────────────────

def plot_training_ablation():
    """MAIN-PAPER training Fig 3(a): rel_eval (test rel error after fixed
    K_eval=10 ADMM refinement). Apples-to-apples comparison across K_train.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    plotted = 0
    epochs_ref = None
    for K_train, st in STYLE_K_TRAIN.items():
        npz_path = DATA_DIR / f"training_ablation_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        d = np.load(npz_path)
        if "rel_eval" not in d.files:
            print(f"  [skip] K={K_train}: no rel_eval (older npz)"); continue
        epochs = d["epochs"]; epochs_ref = epochs
        m = d["rel_eval"]
        p = np.nanpercentile(m, [25, 50, 75], axis=1)
        p = np.maximum(p, floor)
        ax.fill_between(epochs, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.semilogy(epochs, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1
    if plotted == 0:
        print("  [skip] training_ablation: no data"); plt.close(fig); return
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"$\|x^{K_{\mathrm{eval}}=10} - x^{\star}\|_{2} / \|x^{\star}\|_{2}$")
    ax.set_xscale("log")
    ax.set_xlim(max(1, int(epochs_ref[0])), int(epochs_ref[-1]))
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    _save(fig, "training_ablation")


def plot_training_unr():
    """Appendix variant: y-axis = rel_unr (each method evaluated at its own
    K_train inference depth). Less fair but shows training-objective dynamics.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    plotted = 0
    epochs_ref = None
    for K_train, st in STYLE_K_TRAIN.items():
        npz_path = DATA_DIR / f"training_ablation_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        d = np.load(npz_path)
        epochs = d["epochs"]; epochs_ref = epochs
        m = d["rel_unr"] if "rel_unr" in d.files else d["batch_losses"]
        p = np.nanpercentile(m, [25, 50, 75], axis=1)
        p = np.maximum(p, floor)
        ax.fill_between(epochs, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.semilogy(epochs, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1
    if plotted == 0:
        print("  [skip] training_unr: no data"); plt.close(fig); return
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"$\|x^{K_t} - x^{\star}\|_{2} / \|x^{\star}\|_{2}$")
    ax.set_xscale("log")
    ax.set_xlim(max(1, int(epochs_ref[0])), int(epochs_ref[-1]))
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    _save(fig, "training_unr")


def plot_training_loss():
    """Per-epoch training loss (log1p(KKT residual after K_t unrolls)).

    Same as Fig 3(a) but the y-axis is the SGD training objective rather than
    the test-set rel error. Useful for showing optimization-side dynamics.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    plotted = 0
    epochs_ref = None
    for K_train, st in STYLE_K_TRAIN.items():
        npz_path = DATA_DIR / f"training_ablation_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        d = np.load(npz_path)
        epochs = d["epochs"]; epochs_ref = epochs
        # batch_losses: (E, n_batches) — IQR band over within-epoch batch variance
        bl = d["batch_losses"]
        p = np.nanpercentile(bl, [25, 50, 75], axis=1)
        p = np.maximum(p, floor)
        ax.fill_between(epochs, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.semilogy(epochs, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1
    if plotted == 0:
        print("  [skip] training_loss: no data"); plt.close(fig); return
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"Training loss $\log\!\left(1 + \|r\|_{2}^{2}\right)$")
    ax.set_xscale("log")
    ax.set_xlim(max(1, int(epochs_ref[0])), int(epochs_ref[-1]))
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    _save(fig, "training_loss")


# ── Fig 3(b) — inference_ablation.pdf ────────────────────────────────────────

def plot_inference_ablation():
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    metric_key = METRIC_INFER
    plotted = 0

    # Full-dim 3-block ADMM (vanilla) — listed first so it appears on top of legend.
    full_npz = DATA_DIR / "inference_time_full.npz"
    if full_npz.exists():
        d = np.load(full_npz)
        K = d["K_refines"]
        m = d[metric_key] if metric_key in d.files else d["kkt"]
        p = np.percentile(m, [25, 50, 75], axis=0)
        p = np.maximum(p, floor)
        ax.fill_between(K, p[0], p[2], color=COLOR_FULL, alpha=0.20, lw=0)
        ax.plot(K, p[1], "--", color=COLOR_FULL, lw=2.0, label="Full-dim ADMM")
        plotted += 1

    cold_npz = DATA_DIR / "inference_time_cold.npz"
    if cold_npz.exists():
        d = np.load(cold_npz)
        K = d["K_refines"]
        m = d[metric_key] if metric_key in d.files else d["kkt"]
        p = np.percentile(m, [25, 50, 75], axis=0)
        p = np.maximum(p, floor)
        ax.fill_between(K, p[0], p[2], color=COLOR_COLD, alpha=0.20, lw=0)
        ax.plot(K, p[1], ":", color=COLOR_COLD, lw=2.0, label="Low-dim ADMM(ReLU)")
        plotted += 1

    for K_train in [0, 1, 5, 10, 25, 50]:
        npz_path = DATA_DIR / f"inference_ablation_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        st = STYLE_K_TRAIN[K_train]
        d = np.load(npz_path)
        K = d["K_refines"]
        m = d[metric_key] if metric_key in d.files else d["kkt"]
        p = np.percentile(m, [25, 50, 75], axis=0)
        p = np.maximum(p, floor)
        ax.fill_between(K, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.plot(K, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1

    if plotted == 0:
        print("  [skip] inference_ablation: no data"); plt.close(fig); return
    ax.axhline(TARGET, color="black", ls=":", lw=0.9, alpha=0.5)
    ax.set_yscale("log")
    if SCALE_TAG == "n10":
        ax.set_xlim(0, 500); xtick = 50; ylo = 1e-12
    else:
        ax.set_xlim(0, 5000); xtick = 1000; ylo = 1e-9
    ax.set_ylim(ylo, 1e2)
    ax.set_xlabel("Inference Refinement $K$")
    ax.set_ylabel(METRIC_LABELS.get(metric_key, "KKT Residual"))
    from matplotlib.ticker import MultipleLocator, LogLocator
    ax.xaxis.set_major_locator(MultipleLocator(xtick))
    ax.tick_params(axis="x", labelsize=8)
    # Y-axis log minor grid.
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    _save(fig, "inference_ablation")


# ── Appendix — inference_time.pdf ────────────────────────────────────────────

def plot_inference_time():
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    plotted = 0

    # Full-dim 3-block ADMM (vanilla) — slowest baseline.
    full_npz = DATA_DIR / "inference_time_full.npz"
    if full_npz.exists():
        d = np.load(full_npz)
        t  = d["t_per_qp_ms"]; kkt = d["kkt"]
        p  = np.percentile(kkt, [25, 50, 75], axis=0)
        p  = np.maximum(p, floor)
        ax.fill_between(t, p[0], p[2], color=COLOR_FULL, alpha=0.20, lw=0)
        ax.semilogy(t, p[1], "--", color=COLOR_FULL, lw=1.8, label="Full-dim ADMM")
        plotted += 1

    # ADMM(ReLU) baseline (no MLP warm start), if available
    cold_npz = DATA_DIR / "inference_time_cold.npz"
    if cold_npz.exists():
        d = np.load(cold_npz)
        t  = d["t_per_qp_ms"]; kkt = d["kkt"]
        p  = np.percentile(kkt, [25, 50, 75], axis=0)
        p  = np.maximum(p, floor)
        ax.fill_between(t, p[0], p[2], color=COLOR_COLD, alpha=0.20, lw=0)
        ax.semilogy(t, p[1], ":", color=COLOR_COLD, lw=1.8, label="Low-dim ADMM(ReLU)")
        plotted += 1
    else:
        print(f"  [info] {cold_npz} not found (cold-ADMM baseline)")

    # MLP curves
    for K_train in [0, 1, 5, 10, 25, 50]:
        npz_path = DATA_DIR / f"inference_time_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        st = STYLE_K_TRAIN[K_train]
        d = np.load(npz_path)
        t   = d["t_per_qp_ms"]; kkt = d["kkt"]
        p   = np.percentile(kkt, [25, 50, 75], axis=0)
        p   = np.maximum(p, floor)
        ax.fill_between(t, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.semilogy(t, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1

    if plotted == 0:
        print("  [skip] inference_time: no data"); plt.close(fig); return
    ax.axhline(TARGET, color="black", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, 1.2 if SCALE_TAG == "n10" else 15.0)
    ax.set_xlabel("Cumulative inference time per QP (ms)")
    ax.set_ylabel(r"KKT Residual")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    ax.grid(True, which="both", alpha=0.15)
    _save(fig, "inference_time")


# ── Summary table: iterations & wall-clock to TARGET ────────────────────────

KS_TRAIN_TABLE = [0, 1, 5, 10, 25, 50]


def _first_K_below(kkt: np.ndarray, t_ms: np.ndarray, target: float):
    """Per-QP, first K with kkt[K] ≤ target. Returns (K_star, t_star), NaN if never."""
    below = kkt <= target
    B = kkt.shape[0]
    K_star = np.full(B, np.nan); t_star = np.full(B, np.nan)
    for i in range(B):
        idx = np.flatnonzero(below[i])
        if idx.size > 0:
            k = int(idx[0])
            K_star[i] = k
            t_star[i] = t_ms[k]
    return K_star, t_star


def _med_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    if np.all(np.isnan(x)):
        return np.nan, np.nan, np.nan
    return (float(np.nanmedian(x)),
            float(np.nanpercentile(x, 25)),
            float(np.nanpercentile(x, 75)))


def _fmt_iter(med, *_):
    return "—" if np.isnan(med) else f"{med:.0f}"

def _fmt_time(med, *_):
    return "—" if np.isnan(med) else f"{med:.3f}"


def make_ablation_table():
    rows = []   # (label, K_med/lo/hi, t_med/lo/hi, dK_med/lo/hi)

    # Optional Full-dim ADMM (vanilla) row at top — slowest baseline.
    full = DATA_DIR / "inference_time_full.npz"
    full_note = ""   # appended to caption if Full-dim never reaches target
    full_override = None
    if full.exists():
        d = np.load(full)
        K_full, t_full = _first_K_below(d[METRIC_INFER], d["t_per_qp_ms"], TARGET)
        K_max_full   = int(d["K_refines"][-1])
        t_max_full   = float(d["t_per_qp_ms"][-1])
        kkt_max_med  = float(np.nanmedian(d[METRIC_INFER][:, -1]))
        full_reached = not np.all(np.isnan(K_full))
        rows.append(("Full-dim ADMM",
                     *_med_iqr(K_full), *_med_iqr(t_full),
                     np.nan, np.nan, np.nan))
        if not full_reached:
            full_override = {
                "K_str":  rf"$>\!10^{{{int(np.log10(K_max_full))}}}$",
                "t_str":  rf"$>\!{t_max_full:.1f}$",
            }
            full_note = (rf" Full-dim ADMM does not reach the target within "
                         rf"$K{{=}}{K_max_full}$ "
                         rf"(median KKT $\approx {kkt_max_med:.1e}$).")

    cold = DATA_DIR / "inference_time_cold.npz"
    if not cold.exists():
        print(f"  [skip] table: {cold} not found"); return
    d = np.load(cold)
    K_base, t_base = _first_K_below(d[METRIC_INFER], d["t_per_qp_ms"], TARGET)
    rows.append(("Low-dim ADMM(ReLU)",
                 *_med_iqr(K_base), *_med_iqr(t_base),
                 np.nan, np.nan, np.nan))    # baseline: ΔK undefined
    for K_t in KS_TRAIN_TABLE:
        npz_b = DATA_DIR / f"inference_ablation_K{K_t}.npz"
        npz_c = DATA_DIR / f"inference_time_K{K_t}.npz"
        if not (npz_b.exists() and npz_c.exists()):
            print(f"  [skip] table K_t={K_t}: data missing"); continue
        m = np.load(npz_b)[METRIC_INFER]
        t_ms = np.load(npz_c)["t_per_qp_ms"]
        K_star, t_star = _first_K_below(m, t_ms, TARGET)
        # per-QP pairwise iteration savings vs cold ADMM(ReLU) baseline
        dK = K_base - K_star
        # Compact labels — full names go in caption / Fig 3 legend.
        label = (r"MLP only" if K_t == 0
                 else rf"MLP + {K_t}-layer ADMM(ReLU)")
        if K_t == 10:
            label = rf"\textbf{{{label}}}"
        rows.append((label,
                     *_med_iqr(K_star), *_med_iqr(t_star), *_med_iqr(dK)))

    # Speedup baseline = Low-dim ADMM(ReLU) cold (matches paper convention).
    cold_idx = next(i for i, r in enumerate(rows) if r[0] == "Low-dim ADMM(ReLU)")
    base_t = rows[cold_idx][4]
    speedups = [np.nan if (np.isnan(r[4]) or r[4] <= 0) else base_t / r[4] for r in rows]

    tex = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Iterations $K^{{\star}}$ and wall-clock to KKT residual "
        rf"$\leq {TARGET:.0e}$, median over $100$ test mpQPs "
        rf"({_SCALE_DESC[SCALE_TAG]}). "
        rf"Speedup is relative to cold low-dim ADMM(ReLU); $\Delta K$ is the "
        rf"per-QP iteration reduction vs that baseline.{full_note}}}",
        r"\label{tab:ablation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $K^{\star}$ & Time (ms) & Speedup & $\Delta K$ \\",
        r"\midrule",
    ]
    for r, sp in zip(rows, speedups):
        label, K_med, K_lo, K_hi, t_med, t_lo, t_hi, dK_med, dK_lo, dK_hi = r
        sp_s = (f"{sp:.2f}$\\times$" if r is rows[cold_idx]
                else (f"{sp:.1f}$\\times$" if not np.isnan(sp) else "—"))
        if label == "Full-dim ADMM" and full_override is not None:
            K_str, t_str = full_override["K_str"], full_override["t_str"]
        else:
            K_str, t_str = _fmt_iter(K_med), _fmt_time(t_med)
        tex.append(f"{label} & {K_str} & {t_str} "
                   f"& {sp_s} & {_fmt_iter(dK_med)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_stem = f"ablation_table{_SAVE_SUFFIX}.tex"
    (OUT_DIR / table_stem).write_text("\n".join(tex) + "\n")
    print(f"  saved {table_stem}")
    print("\n" + "\n".join(tex))


# ── Main ─────────────────────────────────────────────────────────────────────

def plot_training_eval():
    """Per-epoch test-set rel error after FIXED K_EVAL=10 ADMM(ReLU) refinement.
    Apples-to-apples comparison: every K_t-trained model is evaluated at the
    same inference depth (10 refinement steps), so curves directly reflect
    warm-start quality independent of inference-time choice.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    floor = 1e-16
    plotted = 0
    epochs_ref = None
    for K_train, st in STYLE_K_TRAIN.items():
        npz_path = DATA_DIR / f"training_ablation_K{K_train}.npz"
        if not npz_path.exists():
            print(f"  [skip] {npz_path} not found"); continue
        d = np.load(npz_path)
        if "rel_eval" not in d.files:
            print(f"  [skip] K={K_train}: no rel_eval (older npz)"); continue
        epochs = d["epochs"]; epochs_ref = epochs
        m = d["rel_eval"]
        p = np.nanpercentile(m, [25, 50, 75], axis=1)
        p = np.maximum(p, floor)
        ax.fill_between(epochs, p[0], p[2], color=st["color"], alpha=0.20, lw=0)
        ax.semilogy(epochs, p[1], **_line_kw(st), label=_label(K_train))
        plotted += 1
    if plotted == 0:
        print("  [skip] training_eval: no data"); plt.close(fig); return
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"$\|x^{K_{\text{eval}}=10} - x^{\star}\|_{2} / \|x^{\star}\|_{2}$")
    ax.set_xscale("log")
    ax.set_xlim(max(1, int(epochs_ref[0])), int(epochs_ref[-1]))
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                          numticks=100))
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.15, lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9, edgecolor="0.4", handlelength=1.4)
    _save(fig, "training_eval")


def main():
    for scale in ("n10", "n100"):
        _set_scale(scale)
        if not DATA_DIR.exists():
            print(f"\n=== [{scale}] data dir missing, skipping ==="); continue
        print(f"\n=== Plotting scale {scale} → suffix '{_SAVE_SUFFIX}' ===")
        plot_training_ablation()
        plot_training_unr()
        plot_training_loss()
        plot_inference_ablation()
        plot_inference_time()
        make_ablation_table()
    print(f"\nAll ablation figures + tables saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
