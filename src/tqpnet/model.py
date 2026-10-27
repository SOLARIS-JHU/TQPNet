"""User-facing two-phase API: define a QP family, train the proxy offline, solve online."""
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from tqpnet.admm import ADMMOperator, build_operator
from tqpnet.network import WarmStartNet, build_network
from tqpnet.params import Params
from tqpnet.solve import SolveResult
from tqpnet.solve import solve as _solve
from tqpnet.train import train as _train


class Model:
    """TQPNet proxy for the multiparametric QP family (paper, Eq. (1))

        min_x  ½ xᵀ Q x + q(p)ᵀ x   s.t.  A x ≥ c(p),

    with Q and A fixed and q(p), c(p) parameter dependent.

    Usage:
        model = Model("my_qp")
        model.set_problem(Q, A, q_fn=lambda p: q0 + Cq @ p, c_fn=lambda p: c0 + Cc @ p)
        model.set_sampler(box_sampler(lo, hi))
        model.Params.Train.epochs = 500
        model.train()
        result = model.solve(P)

    Hyperparameters live in ``model.Params.Network``, ``model.Params.Train`` and
    ``model.Params.Solve`` (defaults match the paper's experiments).
    """

    def __init__(self, name: str = "tqpnet"):
        self.name = name
        self.Params = Params()

        self._Q: Optional[jax.Array] = None
        self._A: Optional[jax.Array] = None
        self._q_fn: Optional[Callable] = None
        self._c_fn: Optional[Callable] = None
        self._sampler_fn: Optional[Callable] = None

        self._op: Optional[ADMMOperator] = None
        self._net: Optional[WarmStartNet] = None
        self._variables = None
        self._n_p: Optional[int] = None

    # ------------------------------------------------------------------
    # Problem definition
    # ------------------------------------------------------------------

    def set_problem(self, Q, A, q_fn: Callable, c_fn: Callable) -> None:
        """Define the QP family.

        Args:
            Q:    (n, n) symmetric positive semidefinite cost matrix (fixed).
            A:    (m, n) constraint matrix (fixed); every row and every column must
                  contain a non-zero entry (required by the paper's D_U, D_V lemma).
            q_fn: callable p ↦ q(p) of shape (n,), traceable by ``jax.vmap``.
            c_fn: callable p ↦ c(p) of shape (m,), traceable by ``jax.vmap``.
        """
        Q = jnp.asarray(Q)
        if not jnp.issubdtype(Q.dtype, jnp.floating):
            Q = Q.astype(jnp.asarray(1.0).dtype)
        A = jnp.asarray(A, dtype=Q.dtype)

        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be a square matrix, got shape {Q.shape}")
        n = Q.shape[0]
        if A.ndim != 2 or A.shape[1] != n:
            raise ValueError(f"A must have shape (m, {n}), got {A.shape}")
        asym = float(jnp.max(jnp.abs(Q - Q.T)))
        if asym > 1e-8 * max(1.0, float(jnp.max(jnp.abs(Q)))):
            raise ValueError(f"Q must be symmetric (max |Q - Q^T| = {asym:.3e})")
        if bool(jnp.any(jnp.all(A == 0, axis=1))):
            raise ValueError("every row of A must contain a non-zero entry")
        if bool(jnp.any(jnp.all(A == 0, axis=0))):
            raise ValueError("every column of A must contain a non-zero entry")

        self._Q, self._A = Q, A
        self._q_fn, self._c_fn = q_fn, c_fn
        self._op = self._net = self._variables = self._n_p = None

    def set_sampler(self, sampler_fn: Callable) -> None:
        """Set the parameter sampler ``sampler_fn(key, n_samples) -> (n_samples, n_p)``."""
        self._sampler_fn = sampler_fn

    @property
    def n(self) -> int:
        if self._Q is None:
            raise RuntimeError("Call set_problem() first.")
        return self._Q.shape[0]

    @property
    def m(self) -> int:
        if self._A is None:
            raise RuntimeError("Call set_problem() first.")
        return self._A.shape[0]

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def train(self, seed: int = 0) -> Dict:
        """Offline phase of Algorithm 2: sample a training set and train the warm start.

        Returns:
            History dict with per-epoch ``"loss"``, ``"residual"``, ``"lr"`` and the
            counters ``"epochs"``, ``"updates"``.
        """
        if self._Q is None:
            raise RuntimeError("Call set_problem() first.")
        if self._sampler_fn is None:
            raise RuntimeError("Call set_sampler() first.")
        tp, netp = self.Params.Train, self.Params.Network
        n, m = self.n, self.m

        key = jax.random.PRNGKey(seed)
        key, k_probe, k_data, k_net, k_train = jax.random.split(key, 5)

        probe = jnp.asarray(self._sampler_fn(k_probe, 1))
        if probe.ndim != 2 or probe.shape[0] != 1:
            raise ValueError(f"sampler(key, 1) must return shape (1, n_p), got {probe.shape}")
        n_p = probe.shape[1]
        p0 = probe[0]
        q_shape, c_shape = jnp.shape(self._q_fn(p0)), jnp.shape(self._c_fn(p0))
        if q_shape != (n,):
            raise ValueError(f"q_fn(p) must return shape ({n},), got {q_shape}")
        if c_shape != (m,):
            raise ValueError(f"c_fn(p) must return shape ({m},), got {c_shape}")

        op = build_operator(self._Q, self._A, tp.rho)
        dataset = jnp.asarray(self._sampler_fn(k_data, tp.n_train))
        net, variables = build_network(n, m, n_p, netp.width, netp.depth, k_net)

        variables, history = _train(
            net, variables, op, dataset, self._q_fn, self._c_fn,
            epochs=tp.epochs, batch_size=tp.batch_size, lr=tp.lr, weight_decay=tp.weight_decay,
            K_train=tp.K_train, tol_off=tp.tol_off, log_every=tp.log_every, key=k_train,
            grad_clip=tp.grad_clip, lr_step_epochs=tp.lr_step_epochs, lr_gamma=tp.lr_gamma,
        )
        self._op, self._net, self._variables, self._n_p = op, net, variables, n_p
        return history

    # ------------------------------------------------------------------
    # Online inference
    # ------------------------------------------------------------------

    def solve(self, P) -> SolveResult:
        """Online phase of Algorithm 2 for one parameter vector (n_p,) or a batch (batch, n_p)."""
        if self._variables is None:
            raise RuntimeError("Call train() before solve().")
        P = jnp.atleast_2d(jnp.asarray(P))
        if P.ndim != 2 or P.shape[1] != self._n_p:
            raise ValueError(f"P must have shape (batch, {self._n_p}) or ({self._n_p},), got {P.shape}")
        sp, tp = self.Params.Solve, self.Params.Train
        K_infer = tp.K_train if sp.K_infer is None else sp.K_infer
        return _solve(self._net, self._variables, self._op, P, self._q_fn, self._c_fn,
                      K_infer=K_infer, tol_on=sp.tol_on, K_max=sp.K_max)
