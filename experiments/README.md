# Paper experiments

All commands run from the repository root and write outputs under a gitignored `runs/` directory next to each experiment (`experiments/runs/`, `experiments/ablation/runs/`, `experiments/admm/runs/`, `experiments/pympc-quadruped/runs/`). Install dependencies first: `pip install -e ".[experiments]"`.

## Layout

- Root: shared modules (`base.py`, `common.py`, `config.py`) plus `plot_paper.py`, `plot_results.py`.
- `experiments/ablation/`: training and inference ablations (`training_ablation.py`, `inference_ablation.py`) and their plotting script (`plot_ablation.py`).
- `experiments/admm/`: ADMM convergence experiment and its plotting script; `main_v1.m` is the original MATLAB prototype of the same comparison.
- `experiments/uav/`: UAV MPC (TQPNet, benchmark, physics) and its plotting script.
- `experiments/scalability/`: scalability MPC experiment, plus the `explicit_mpc/` MATLAB baseline.
- `experiments/generalizability/`: generalizability experiment.
- `experiments/pympc-quadruped/`: vendored third-party quadruped batch-MPC stack.

| Paper artifact | Generate data | Plot |
|---|---|---|
| Fig 2 (ADMM convergence, scale=2/4/6; 100 QPs at n=500, m=1000, density 0.1, the default `SCALE_TAG = "n500m1000"`) | `python -m experiments.admm.admm_convergence` | `python -m experiments.admm.plot_admm_convergence` |
| Fig 3a (training ablation, n=100) | `python -m experiments.ablation.training_ablation` (set `SCALE_TAG = "n100"` in the script for the main-text figure; the default `"n10"` produces the appendix figure) | `python -m experiments.ablation.plot_ablation` |
| Fig 3b + Table + Appendix inference-time figure (inference ablation, n=100) | `python -m experiments.ablation.inference_ablation` (set `SCALE_TAG = "n100"` in the script, matching `training_ablation.py`) | `python -m experiments.ablation.plot_ablation` |
| Appendix: ablation figures at n=10 (training, inference, inference time) | same two scripts with the default `SCALE_TAG = "n10"` | `python -m experiments.ablation.plot_ablation` (plots both scales it finds) |
| Fig 4 + Table (generalizability) | `python -m experiments.generalizability.generalizability` | (plots built in) |
| Fig 5 + Table (scalability MPC, explicit-MPC baseline) | `python -m experiments.scalability.scalability_mpc` (also writes `scalability_mpc_table.tex`) | `python -m experiments.plot_paper` (latency and offline-time figures) |
| Appendix: quadrotor UAV MPC (trajectory, tracking error, table) | `python -m experiments.uav.uav_tqpnet` then `python -m experiments.uav.uav_benchmark` | `python -m experiments.uav.plot_uav` |
| Figs 6 and 7 (quadruped batch MPC) | see `pympc-quadruped/README.md` (separate environment with `qpth`) | `pympc-quadruped/scripts/make_paper_plots.py` |
| Explicit MPC baseline (MATLAB) | run `experiments/scalability/explicit_mpc/run_explicit_mpc.m` in MATLAB | (results JSON consumed by scalability table) |

`python -m experiments.plot_results` runs the ADMM, ablation, UAV and scalability plotting entry points in one go, and `python -m experiments.plot_paper` renders the paper-styled scalability and UAV figures, assuming the data-generation runs above have completed.

Shared modules: `base.py` (problem construction, ADMM-ReLU operators, full-dim 3-block ADMM baseline, neural proxy), `config.py` (scales, seeds, hyperparameters), `common.py` (run directories, logging).
