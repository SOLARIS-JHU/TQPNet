"""Fig 3(a) EXPERIMENT: training residual descent for K_train ∈ {0, 1, 5, 10}.

Trains TQPNet on a single mpQP family with different unrolled-ADMM(ReLU)
depths.  Records EVERYTHING that could be useful for downstream plotting,
so changing the y-axis metric never requires re-running:

  • per-batch SGD loss within each epoch          (Fig 3(a) default — band from in-epoch variance)
  • per-epoch test-set evaluation, per QP:
      kkt_raw  — KKT residual at MLP raw output
      kkt_unr  — KKT residual after K_train ADMM(ReLU) unrolls (matches SGD objective)
      abs_raw, abs_unr  — ‖x − x⋆‖₂           (needs OSQP x*)
      rel_raw, rel_unr  — abs / ‖x⋆‖₂

Loss objective (per K_train):
    K_t = 0:  log(1 + KKT_raw)                                  (KKT loss)
    K_t = K:  log(1 + KKT_after_K_unrolls)                      (unrolled KKT)

Data output:
    experiments/runs/ablation/data/training_ablation_K{0,1,5,10}.npz
        keys:
            epochs               : (E,)            checkpoint epoch indices
            train_loss_mean      : (E,)            epoch-mean training loss
            batch_losses         : (E, n_batches)  per-batch training loss in each epoch
            kkt_raw, abs_raw, rel_raw : (E, N_test)  test-set per-QP at MLP raw output
            kkt_unr, abs_unr, rel_unr : (E, N_test)  test-set per-QP after K_t unrolls
            K_train              : scalar
    experiments/runs/ablation/models/tqpnet_K{0,1,5,10}.pth                (final ckpts)

Usage:
    conda run -n isaacgym python -m experiments.ablation.training_ablation
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.base import (
    TQPNet, build_config, build_problem_data,
    admm_informed_loss, admm_unrolled_loss, kkt_raw_residual,
    device,
)
from experiments.config import (
    PROBLEM_SEED, RHO_REFINE, A_DENSITY,
    N_TRAIN, BATCH_SIZE, LR, WEIGHT_DECAY,
)

MAX_EPOCHS = 2000   # K=10 shows U-shape past ep~500; 2000 enough to see plateau,
                    # uniform across K_train values for fair Fig 3(a).


# ── Scale config: switch SCALE_TAG to swap data/model directories ───────────
# "n10":  small mpQPs   (n=10,  m=20,  np_dim=4,  W=256) — appendix scale
# "n100": medium mpQPs  (n=100, m=200, np_dim=20, W=512)
# "n500": large mpQPs   (n=500, m=500, np_dim=50, W=512) — current paper scale
SCALE_TAG = "n10"

_SCALES = {
    "n10":  {"n": 10,  "m": 20,  "np_dim": 4,  "W": 256, "L": 4},
    "n100": {"n": 100, "m": 200, "np_dim": 20, "W": 512, "L": 4},
    "n500": {"n": 500, "m": 500, "np_dim": 50, "W": 512, "L": 4},
}
SCALE        = _SCALES[SCALE_TAG]
A_SCALE      = 3.0
C_OFFSET     = 1.0
N_TEST       = 100
SEED_TEST    = 2026
LOG_EVERY    = 10
KS_TRAIN     = [0, 1, 5, 10, 25, 50]   # script auto-skips entries with existing files
K_EVAL       = 10                # fixed-depth refinement for cross-K_t comparison


RUN_DIR    = Path(__file__).resolve().parent / "runs" / "ablation"
# Append c{C_OFFSET} to dir tag so c_offset=0 data doesn't overwrite c_offset=1 data.
_TAG       = f"{SCALE_TAG}" + ("" if C_OFFSET == 1.0 else f"_c{C_OFFSET:g}")
DATA_DIR   = RUN_DIR / f"data_{_TAG}"
MODEL_DIR  = RUN_DIR / f"models_{_TAG}"
LOG_DIR    = RUN_DIR / "logs"
for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Tee stdout to a timestamped log file (filename carries SCALE_TAG) ──────
import sys, datetime
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for x in self.streams:
            x.write(s); x.flush()
    def flush(self):
        for x in self.streams: x.flush()
_log_path = LOG_DIR / f"training_ablation_{SCALE_TAG}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
_log_file = open(_log_path, "w")
sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log] {_log_path}")


# ── Test set + OSQP reference ────────────────────────────────────────────────

def make_test_set(cfg, n_test=N_TEST, seed=SEED_TEST):
    """Build test QP batch + OSQP reference x* (for abs / rel error metrics)."""
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    P = -1.0 + 2.0 * torch.rand(n_test, cfg.np_dim, generator=gen).to(device)
    Q_b, q_b, A_b, c_b = build_problem_data(P, cfg)
    import osqp
    from scipy import sparse
    Q_np = Q_b.detach().cpu().numpy(); q_np = q_b.detach().cpu().numpy()
    A_np = A_b.detach().cpu().numpy(); c_np = c_b.detach().cpu().numpy()
    x_opt = np.zeros((n_test, cfg.n), dtype=np.float32)
    print(f"  solving OSQP x* on N_TEST={n_test}…", flush=True)
    for i in range(n_test):
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(Q_np[i]), q=q_np[i],
                   A=sparse.csc_matrix(A_np[i]),
                   l=c_np[i], u=np.inf * np.ones_like(c_np[i]),
                   verbose=False, polish=True,
                   eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)
        res = prob.solve()
        if res.x is not None and np.all(np.isfinite(res.x)):
            x_opt[i] = res.x.astype(np.float32)
    x_opt_t = torch.tensor(x_opt, device=device)
    return P, Q_b, q_b, A_b, c_b, x_opt_t


# ── Eval helpers (per-QP) ────────────────────────────────────────────────────

def _per_qp_metrics(x, beta, Q_b, q_b, A_b, c_b, x_opt_b, x_opt_norms_np, cfg):
    kkt = kkt_raw_residual(x, beta, Q_b, q_b, A_b, c_b, cfg).cpu().numpy()
    abs_err = (x - x_opt_b).norm(dim=-1).cpu().numpy()
    rel_err = abs_err / x_opt_norms_np
    return kkt, abs_err, rel_err


@torch.no_grad()
def eval_raw(model, P_test, Q_b, q_b, A_b, c_b, x_opt_b, x_opt_norms, cfg):
    """At MLP raw output."""
    x, beta = model(P_test)
    return _per_qp_metrics(x, beta.clamp(min=0),
                            Q_b, q_b, A_b, c_b, x_opt_b, x_opt_norms, cfg)


@torch.no_grad()
def eval_after_unroll(model, P_test, Q_b, q_b, A_b, c_b, x_opt_b, x_opt_norms,
                       cfg, K_unroll: int):
    """After K_unroll ADMM(ReLU) steps from MLP warm start."""
    x, beta = model(P_test)
    beta = beta.clamp(min=0)
    if K_unroll > 0:
        dev, dtype = x.device, x.dtype
        A_t    = A_b.transpose(1, 2)
        D_V    = cfg._D_V.to(dev, dtype).unsqueeze(0)
        inv_DU = cfg._inv_D_U.to(dev, dtype).unsqueeze(0)
        chol   = cfg._H_V_chol.to(dev, dtype).unsqueeze(0)
        rho    = cfg.rho_refine
        for _ in range(K_unroll):
            x_old = x
            rhs   = (rho * D_V * x
                      + rho * torch.bmm(A_t, beta.unsqueeze(-1)).squeeze(-1)
                      - q_b)
            x     = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
            beta  = (beta
                     - inv_DU * torch.bmm(A_b, (2 * x - x_old).unsqueeze(-1)).squeeze(-1)
                     + inv_DU * c_b).clamp(min=0)
    return _per_qp_metrics(x, beta, Q_b, q_b, A_b, c_b, x_opt_b, x_opt_norms, cfg)


# ── Training loop ────────────────────────────────────────────────────────────

def train_one(K_train: int, cfg, P_test, Q_test, q_test, A_test, c_test, x_opt_t,
              x_opt_norms_np):
    torch.manual_seed(0)
    P_train = -1.0 + 2.0 * torch.rand(N_TRAIN, cfg.np_dim, device=device)
    loader = DataLoader(TensorDataset(P_train), batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True)
    n_batches = len(loader)

    model = TQPNet(cfg.np_dim, cfg.n, cfg.m, SCALE["W"], SCALE["L"]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)

    epochs, mean_log, batch_log = [], [], []
    kkt_raw_log, abs_raw_log, rel_raw_log = [], [], []
    kkt_unr_log, abs_unr_log, rel_unr_log = [], [], []
    kkt_eval_log, abs_eval_log, rel_eval_log = [], [], []   # fixed K_EVAL refinement
    label = f"K_train={K_train}"
    print(f"  [{label}]  start training (n_batches/epoch={n_batches})", flush=True)
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        per_batch = []
        for (p_b,) in loader:
            Q_b, q_b, A_b, c_b = build_problem_data(p_b, cfg)
            opt.zero_grad()
            xp, bp = model(p_b)
            if K_train > 0:
                loss = admm_unrolled_loss(xp, bp, Q_b, q_b, A_b, c_b, cfg,
                                           K_unroll=K_train)
            else:
                loss = admm_informed_loss(xp, bp, Q_b, q_b, A_b, c_b, cfg)
            if torch.isnan(loss) or torch.isinf(loss):
                per_batch.append(np.nan)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            per_batch.append(loss.item())
        sched.step()

        if ep % LOG_EVERY == 0 or ep == 1 or ep == MAX_EPOCHS:
            arr = np.array(per_batch, dtype=np.float32)
            model.eval()
            kr, ar, rr = eval_raw(model, P_test, Q_test, q_test, A_test, c_test,
                                   x_opt_t, x_opt_norms_np, cfg)
            ku, au, ru = eval_after_unroll(model, P_test, Q_test, q_test, A_test, c_test,
                                            x_opt_t, x_opt_norms_np, cfg, K_train)
            # Fixed-depth refinement (same K_EVAL across all K_t) — fair Fig 3(a) y-axis.
            ke, ae, re = eval_after_unroll(model, P_test, Q_test, q_test, A_test, c_test,
                                            x_opt_t, x_opt_norms_np, cfg, K_EVAL)
            epochs.append(ep)
            mean_log.append(float(np.nanmean(arr)))
            batch_log.append(arr)
            kkt_raw_log.append(kr);  abs_raw_log.append(ar);  rel_raw_log.append(rr)
            kkt_unr_log.append(ku);  abs_unr_log.append(au);  rel_unr_log.append(ru)
            kkt_eval_log.append(ke); abs_eval_log.append(ae); rel_eval_log.append(re)
            print(f"  [{label}]  ep={ep:5d}  loss(mean)={mean_log[-1]:.3e}  "
                  f"rel_unr(med)={float(np.nanmedian(ru)):.3e}  "
                  f"rel_eval@K{K_EVAL}(med)={float(np.nanmedian(re)):.3e}  "
                  f"kkt_raw(med)={float(np.nanmedian(kr)):.3e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return dict(
        model      = model,
        epochs     = np.array(epochs),
        train_loss_mean = np.array(mean_log, dtype=np.float32),
        batch_losses    = np.stack(batch_log).astype(np.float32),
        kkt_raw    = np.stack(kkt_raw_log).astype(np.float32),
        abs_raw    = np.stack(abs_raw_log).astype(np.float32),
        rel_raw    = np.stack(rel_raw_log).astype(np.float32),
        kkt_unr    = np.stack(kkt_unr_log).astype(np.float32),
        abs_unr    = np.stack(abs_unr_log).astype(np.float32),
        rel_unr    = np.stack(rel_unr_log).astype(np.float32),
        kkt_eval   = np.stack(kkt_eval_log).astype(np.float32),     # FIXED K_EVAL refines
        abs_eval   = np.stack(abs_eval_log).astype(np.float32),
        rel_eval   = np.stack(rel_eval_log).astype(np.float32),
        K_EVAL     = K_EVAL,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = build_config(SCALE["n"], SCALE["m"], SCALE["np_dim"], device,
                        seed=PROBLEM_SEED, rho_refine=RHO_REFINE,
                        density=A_DENSITY, a_scale=A_SCALE, c_offset=C_OFFSET)
    print(f"Training ablation — n={SCALE['n']} m={SCALE['m']} "
          f"K_trains={KS_TRAIN} max_epochs={MAX_EPOCHS}")
    P_test, Q_test, q_test, A_test, c_test, x_opt_t = make_test_set(cfg)
    x_opt_norms_np = (x_opt_t.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()

    for K_train in KS_TRAIN:
        print(f"\n=== K_train = {K_train} ===")
        out  = DATA_DIR / f"training_ablation_K{K_train}.npz"
        ckpt = MODEL_DIR / f"tqpnet_K{K_train}.pth"
        if out.exists() and ckpt.exists():
            print(f"  [skip] {out} and {ckpt} already exist")
            continue
        out_d = train_one(K_train, cfg, P_test, Q_test, q_test, A_test, c_test,
                           x_opt_t, x_opt_norms_np)
        torch.save(out_d.pop("model").state_dict(), ckpt)
        np.savez(out, K_train=K_train, **out_d)
        print(f"  saved {out}  +  ckpt {ckpt}")


if __name__ == "__main__":
    main()
