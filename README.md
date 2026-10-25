# TQPNet

Official code for the NeurIPS 2026 paper *Neural Compression of Long ADMM Trajectory for Multiparametric Quadratic Program* by Liang Wu, Bo Yang, Xu Yang, Honghui Zheng, Yilin Mo and Ján Drgoňa (Johns Hopkins University and Tsinghua University).

TQPNet learns a neural warm start for parametric quadratic programs and refines it with a low-dimensional 2-block ADMM (ReLU) scheme, giving fast batched GPU inference with solver-grade accuracy.

## Layout

- `src/tulip_qp/` — core JAX package: network, unsupervised ADMM loss, training, certified solve with backup correction (`tulip_qp` is the import name of the core library).
- `experiments/` — all paper experiments (PyTorch): ablations, ADMM convergence, generalizability, scalability, UAV MPC, plus the quadruped batch-MPC stack in `experiments/pympc-quadruped/`. See `experiments/README.md` for the figure-by-figure reproduction guide.
- `tests/` — unit tests for the core package.

## Install

```bash
pip install -e ".[dev]"           # core package + tests
pip install -e ".[experiments]"   # adds torch/matplotlib/scipy/osqp/cvxpy for experiments/
```

## Quickstart

```bash
pytest
```

## Reproducing the paper

Every figure and table maps to a script in `experiments/`; see `experiments/README.md`. No trained weights ship with this repo: each experiment trains its own model from scratch with fixed seeds (minutes on a single GPU for the ablations).

## Citation

```bibtex
@inproceedings{wu2026neural,
  title     = {Neural Compression of Long {ADMM} Trajectory for Multiparametric Quadratic Program},
  author    = {Wu, Liang and Yang, Bo and Yang, Xu and Zheng, Honghui and Mo, Yilin and Drgo{\v{n}}a, J{\'a}n},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache-2.0. Third-party components in `experiments/pympc-quadruped/` retain their original licenses (see `experiments/pympc-quadruped/LICENSE` and `robot/a1/a1_license.txt`).
