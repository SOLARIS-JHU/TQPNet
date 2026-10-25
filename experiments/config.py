"""Configuration for the paper experiments.

A_SCALE=3.0: sparse nonnegative constraint matrix A in [0, 3].
Uniform column sparsity (build_config) ensures D_V is constant,
improving ADMM-ReLU convergence.
"""
from __future__ import annotations

SCALES = [
    {"name": "XXS", "n": 5,    "m": 10,   "np_dim": 2,   "W": 256,  "L": 4},
    {"name": "XS",  "n": 10,   "m": 20,   "np_dim": 4,   "W": 256,  "L": 4},
    {"name": "S",   "n": 50,   "m": 100,  "np_dim": 10,  "W": 256,  "L": 4},
    {"name": "S+",  "n": 100,  "m": 200,  "np_dim": 20,  "W": 256,  "L": 4},
    {"name": "M",   "n": 200,  "m": 400,  "np_dim": 30,  "W": 512,  "L": 4},
    {"name": "M+",  "n": 350,  "m": 700,  "np_dim": 50,  "W": 512,  "L": 4},
    {"name": "L",   "n": 500,  "m": 1000, "np_dim": 60,  "W": 512,  "L": 4},
    {"name": "L+",  "n": 750,  "m": 1500, "np_dim": 80,  "W": 1024, "L": 4},
    {"name": "XL",  "n": 1000, "m": 2000, "np_dim": 100, "W": 1024, "L": 4},
]

PROBLEM_SEED = 42
A_DENSITY    = 0.3
A_SCALE      = 3.0    # A values in [0, A_SCALE] (positive sparse)
RHO_REFINE   = 1.0

LOSS_THRESHOLD  = 1e-8
MAX_EPOCHS      = 2000
N_TRAIN         = 6000
N_TEST          = 10
BATCH_SIZE      = 256
LR              = 1e-3
WEIGHT_DECAY    = 1e-5

ADMM_REFINE_TOL  = 1e-4
ADMM_REFINE_KMAX = 1000

LOG_EVERY_EPOCH  = 100
