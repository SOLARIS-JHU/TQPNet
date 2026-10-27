from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkParams:
    """MLP(ReLU) warm-start network size (M and L in the paper)."""
    width: int = 256   # hidden layer width M
    depth: int = 4     # number of hidden layers L


@dataclass
class TrainParams:
    """Offline training hyperparameters (Algorithm 2, offline phase).

    Defaults follow the paper's experiment scripts: ``experiments/config.py`` and the
    inline constants of the scripts (gradient clip 10.0 and StepLR(300, 0.5) in
    ``training_ablation.py`` / ``generalizability.py``; ``K_train=10`` as in the MPC
    scalability study, while the n=100 random-mpQP studies use K_train=25).
    """
    n_train:        int   = 6000   # size of the sampled training set D
    epochs:         int   = 2000   # N_epoch
    batch_size:     int   = 256    # B
    lr:             float = 1e-3   # AdamW learning rate η
    weight_decay:   float = 1e-5   # AdamW weight decay
    grad_clip:      float = 10.0   # global gradient-norm clipping threshold
    lr_step_epochs: int   = 300    # halve the learning rate every this many epochs
    lr_gamma:       float = 0.5    # learning-rate decay factor
    K_train:        int   = 10     # unrolled ADMM(ReLU) layers inside the loss
    rho:            float = 1.0    # ADMM penalty parameter ρ
    tol_off:        float = 1e-8   # early-stopping tolerance ε_off on the epoch-mean loss
    log_every:      int   = 100    # print progress every this many epochs


@dataclass
class SolveParams:
    """Online inference hyperparameters (Algorithm 2, online phase)."""
    K_infer: Optional[int] = None  # ADMM(ReLU) layers after the warm start; None → K_train
    tol_on:  float = 1e-4          # certificate tolerance ε_on
    K_max:   int   = 1000          # maximum extra ADMM(ReLU) layers for triggered refinement


@dataclass
class Params:
    Network: NetworkParams = field(default_factory=NetworkParams)
    Train:   TrainParams   = field(default_factory=TrainParams)
    Solve:   SolveParams   = field(default_factory=SolveParams)
