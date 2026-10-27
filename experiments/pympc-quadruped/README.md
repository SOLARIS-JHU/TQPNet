# pympc-quadruped (adapted)

Quadruped linear-MPC stack for the batch-MPC experiments, adapted from the open-source Unitree A1/Aliengo MPC implementation [yinghansun/pympc-quadruped](https://github.com/yinghansun/pympc-quadruped) (MIT, original license in `LICENSE`; robot assets under `robot/` retain their own licenses). Our additions: a batched TQPNet QP controller (`linear_mpc/tqpnet_mpc.py`, `linear_mpc/batch_mpc.py`, `linear_mpc/batch_kinematics_controller.py`) and the paper benchmark scripts under `scripts/`.

## Usage

- `scripts/train_tqpnet.py` — train the TQPNet controller on the quadruped QP distribution.
- `scripts/benchmark_solvers.py` — solver wall-clock vs batch size (paper plot 1) and QP error vs iteration (plot 2).
- `scripts/run_velocity_tracking.py` — closed-loop velocity tracking in simulation (requires Isaac Gym or MuJoCo; writes CSVs).
- `scripts/make_paper_plots.py` — regenerates the four paper figures into `runs/paper_figures/`.

This stack does not import the core `tqpnet` package. The solver benchmarks and paper plots need `qpth`, which pins `numpy<2` while current JAX releases need `numpy>=2`, so use a separate environment and install the dependencies directly:

```bash
pip install "numpy<2" "qpth==0.0.18" torch matplotlib scipy osqp cvxpy
```

Run the scripts from this directory (`python scripts/<name>.py`). Simulation extras (Isaac Gym, MuJoCo) and `pinocchio` (used by `utils/robot_data.py`) are only needed for the closed-loop runs.
