from dataclasses import dataclass, field


@dataclass
class NetworkParams:
    width: int = 256   # hidden layer width M
    depth: int = 4     # number of hidden layers L


@dataclass
class TrainParams:
    epochs:     int   = 1000
    batch_size: int   = 64
    lr:         float = 1e-3
    alpha:      float = 1.0    # QPslack slack penalization scalar α
    tol_off:    float = 1e-4   # early-stopping tolerance ε_off
    log_every:  int   = 50


@dataclass
class SolveParams:
    tol_on:       float = 1e-4   # certificate tolerance ε_on: trigger backup if ε > tol_on
    backup_steps: int   = 100    # max Adam steps for backup correction
    backup_lr:    float = 1e-3   # backup Adam learning rate


@dataclass
class Params:
    Network: NetworkParams = field(default_factory=NetworkParams)
    Train:   TrainParams   = field(default_factory=TrainParams)
    Solve:   SolveParams   = field(default_factory=SolveParams)