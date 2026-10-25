"""Train TQPNet for pympc-quadruped's SRBD-MPC QP family.

Usage:
    conda run -n isaacgym python -m scripts.train_tqpnet \
        --robot a1 --n-train 5000 --epochs 500 --k-unroll 10
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from linear_mpc_configs import LinearMpcConfig
from robot_configs import A1Config, AliengoConfig
from tqpnet_mpc import (
    NormalizedTQPNet, build_nominal_qp_config, build_problem_data,
    admm_unrolled_loss, kkt_raw_residual, sample_training_params, admm_refine,
    sample_training_params_compact, compact_to_full, COMPACT_DIM,
)


def setup_logger(out_dir: Path, name: str = 'train_tqpnet'):
    import logging
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h); h.close()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')
    fh = logging.FileHandler(out_dir / f'{name}.log', mode='w'); fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',     choices=['a1', 'aliengo'], default='a1')
    parser.add_argument('--n-train',   type=int,   default=5000,
                        help='number of QP samples per epoch')
    parser.add_argument('--batch-size',type=int,   default=256)
    parser.add_argument('--epochs',    type=int,   default=500)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--k-unroll',  type=int,   default=10,
                        help='ADMM unroll steps in the loss')
    parser.add_argument('--rho-refine',type=float, default=1.0)
    parser.add_argument('--net-w',     type=int,   default=512)
    parser.add_argument('--net-l',     type=int,   default=4)
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--input',     choices=['compact', 'full'], default='compact',
                        help='compact (18-dim) or full (285-dim) NN input')
    parser.add_argument('--save-dir',  type=str,
                        default=str(Path(__file__).resolve().parent.parent
                                    / 'linear_mpc' / 'tqpnet_weights'))
    parser.add_argument('--log-every', type=int,   default=10)
    args = parser.parse_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(save_dir, name=f'train_tqpnet_{args.robot}')
    log.info(f'Args: {vars(args)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'device = {device}')

    robot_cfg = A1Config if args.robot == 'a1' else AliengoConfig
    cfg = build_nominal_qp_config(LinearMpcConfig, robot_cfg,
                                   rho_refine=args.rho_refine, device=device)
    log.info(f'Nominal QP: n={cfg.n}  m={cfg.m}  np_dim={cfg.np_dim}  rho={cfg.rho_refine}')

    # Sample training set
    log.info(f'Sampling {args.n_train} parameter vectors (input={args.input})...')
    if args.input == 'compact':
        P_compact_np = sample_training_params_compact(args.n_train, LinearMpcConfig,
                                                        robot_cfg, seed=args.seed)
        P_compact = torch.tensor(P_compact_np, dtype=torch.float32, device=device)
        # Expand to full p for QP solving (deterministic, fixed once)
        P_full = compact_to_full(P_compact, LinearMpcConfig, robot_cfg)
        P_nn   = P_compact
        np_dim_nn = COMPACT_DIM
        log.info(f'  P_compact shape = {tuple(P_compact.shape)}')
        log.info(f'  P_full    shape = {tuple(P_full.shape)}')
    else:
        P_full_np = sample_training_params(args.n_train, cfg, LinearMpcConfig, robot_cfg,
                                            seed=args.seed)
        P_full = torch.tensor(P_full_np, dtype=torch.float32, device=device)
        P_nn   = P_full
        np_dim_nn = cfg.np_dim
        log.info(f'  P_full shape = {tuple(P_full.shape)}')

    # ── Calibrate input/output normalization ─────────────────────────────────
    log.info('Calibrating normalization stats...')
    input_mean = P_nn.mean(dim=0)
    input_std  = P_nn.std(dim=0).clamp(min=1e-3)
    log.info(f'  input_mean: min={input_mean.min().item():.3f}  max={input_mean.max().item():.3f}')
    log.info(f'  input_std:  min={input_std.min().item():.3f}  max={input_std.max().item():.3f}')

    # Solve N_calib problems with OSQP to estimate output magnitudes
    from scripts.benchmark_solvers import osqp_per_robot
    N_calib = min(256, args.n_train)
    P_cal = P_full[:N_calib]
    Q_b, q_b, A_b, c_b = build_problem_data(P_cal, cfg)
    log.info(f'  Calibrating output scale via OSQP on {N_calib} problems...')
    x_star = osqp_per_robot(Q_b, q_b, A_b, c_b)
    _, beta_star = admm_refine(torch.zeros(N_calib, cfg.n, device=device),
                                Q_b, q_b, A_b, c_b, cfg, K=200)
    output_x_scale = x_star.abs().mean(dim=0).clamp(min=10.0)
    output_b_scale = beta_star.abs().mean(dim=0).clamp(min=1.0)
    log.info(f'  output_x_scale: median={output_x_scale.median().item():.2f}  '
             f'p99={output_x_scale.quantile(0.99).item():.2f}')
    log.info(f'  output_b_scale: median={output_b_scale.median().item():.2f}  '
             f'p99={output_b_scale.quantile(0.99).item():.2f}')

    model = NormalizedTQPNet(np_dim_nn, cfg.n, cfg.m,
                              W=args.net_w, L=args.net_l).to(device)
    model.set_normalization(input_mean, input_std, output_x_scale, output_b_scale)
    log.info(f'NormalizedTQPNet: {model.count_params()/1e6:.2f} M params  '
             f'(np_dim={np_dim_nn}, {args.input} input)')

    loader = DataLoader(TensorDataset(P_nn, P_full), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    best_loss = float('inf')
    save_path = save_dir / f'tqpnet_{args.robot}.pth'
    history = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for (p_nn_b, p_full_b) in loader:
            Q_b, q_b, A_b, c_b = build_problem_data(p_full_b, cfg)
            opt.zero_grad()
            x_pred, b_pred = model(p_nn_b)
            loss = admm_unrolled_loss(x_pred, b_pred, Q_b, q_b, A_b, c_b, cfg,
                                       K_unroll=args.k_unroll)
            if torch.isnan(loss) or torch.isinf(loss):
                log.warning(f'ep={ep}  NaN/Inf loss, skipping batch')
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ep_loss += loss.item() * p_nn_b.shape[0]
        sched.step()
        ep_loss /= args.n_train
        history.append(ep_loss)
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(model.state_dict(), save_path)

        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            log.info(f'ep={ep:4d}  loss={ep_loss:.4e}  best={best_loss:.4e}  '
                     f'lr={sched.get_last_lr()[0]:.2e}  ({elapsed:.1f}s)')

    log.info(f'Training done in {time.time()-t0:.1f}s.  Best loss = {best_loss:.4e}')
    log.info(f'Saved → {save_path}')

    # Quick eval
    model.eval()
    if args.input == 'compact':
        P_eval_compact = torch.tensor(
            sample_training_params_compact(200, LinearMpcConfig, robot_cfg,
                                             seed=args.seed + 1),
            dtype=torch.float32, device=device)
        P_eval_full = compact_to_full(P_eval_compact, LinearMpcConfig, robot_cfg)
        P_eval_nn = P_eval_compact
    else:
        P_eval_full = torch.tensor(
            sample_training_params(200, cfg, LinearMpcConfig, robot_cfg, seed=args.seed + 1),
            dtype=torch.float32, device=device)
        P_eval_nn = P_eval_full

    with torch.no_grad():
        Q_b, q_b, A_b, c_b = build_problem_data(P_eval_full, cfg)
        x_pred, b_pred = model(P_eval_nn)
        kkt_pre = kkt_raw_residual(x_pred, b_pred, Q_b, q_b, A_b, c_b, cfg)
        x_ref, b_ref = admm_refine(x_pred, Q_b, q_b, A_b, c_b, cfg, K=10, beta0=b_pred)
        kkt_post = kkt_raw_residual(x_ref, b_ref, Q_b, q_b, A_b, c_b, cfg)
        x_star_e = osqp_per_robot(Q_b, q_b, A_b, c_b)
        rel_pre  = ((x_pred - x_star_e).norm(dim=1) / (x_star_e.norm(dim=1) + 1e-9)).median().item()
        rel_post = ((x_ref  - x_star_e).norm(dim=1) / (x_star_e.norm(dim=1) + 1e-9)).median().item()

    log.info(f'Eval (200 samples):')
    log.info(f'  KKT before refine : mean={kkt_pre.mean().item():.3e}  median={kkt_pre.median().item():.3e}')
    log.info(f'  KKT after  K=10   : mean={kkt_post.mean().item():.3e}  median={kkt_post.median().item():.3e}')
    log.info(f'  rel-err vs OSQP (NN raw)  : median={rel_pre:.3e}')
    log.info(f'  rel-err vs OSQP (NN+K=10) : median={rel_post:.3e}')


if __name__ == '__main__':
    main()
