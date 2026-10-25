from typing import Callable, Optional
import jax
import jax.numpy as jnp

from tulip_qp.params import Params
from tulip_qp.network import TuQPNet, build_network
from tulip_qp.train import train as _train
from tulip_qp.solve import solve as _solve, SolveResult


class Model:
    """TuQP model — two-phase API for multiparametric QPs.

    Solves the parametric QP family (Eqn. 1 in the paper):
        min   (1/2) x^T Q(p) x + q(p)^T x
        s.t.  A(p) x >= c(p)

    via an unsupervised neural proxy trained on the ADMM residual loss.

    Usage:
        model = Model("my_qp")
        model.set_problem(Q_fn, q_fn, A_fn, c_fn, n=10, m=4)
        model.set_sampler(box_sampler(n=10, m=4))
        model.Params.Train.epochs = 500
        model.train(log_dir="runs")
        result = model.solve(P)
    """

    def __init__(self, name: str = "tuqp_model"):
        self.name = name
        self.Params = Params()

        self._n: Optional[int] = None
        self._m: Optional[int] = None
        self._Q_fn: Optional[Callable] = None
        self._q_fn: Optional[Callable] = None
        self._A_fn: Optional[Callable] = None
        self._c_fn: Optional[Callable] = None
        self._sampler_fn: Optional[Callable] = None
        self._net: Optional[TuQPNet] = None
        self._params: Optional[dict] = None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(P: jnp.ndarray) -> jnp.ndarray:
        """Row-wise L2 normalization. Applied in both train and solve."""
        norm = jnp.linalg.norm(P, axis=1, keepdims=True).clip(1e-8)
        return P / norm

    # ------------------------------------------------------------------
    # Problem definition
    # ------------------------------------------------------------------

    def set_problem(
        self,
        Q_fn: Callable,
        q_fn: Callable,
        A_fn: Callable,
        c_fn: Callable,
        n: int,
        m: int,
    ) -> None:
        """Define the parametric QP data.

        Args:
            Q_fn: callable p -> jnp.ndarray (n, n), positive semidefinite
            q_fn: callable p -> jnp.ndarray (n,)
            A_fn: callable p -> jnp.ndarray (m, n)
            c_fn: callable p -> jnp.ndarray (m,)
            n:    number of primal variables
            m:    number of constraints
        """
        self._Q_fn = Q_fn
        self._q_fn = q_fn
        self._A_fn = A_fn
        self._c_fn = c_fn
        self._n = n
        self._m = m

    def set_sampler(self, sampler_fn: Callable) -> None:
        """Set the parameter sampler.

        Args:
            sampler_fn: callable (key, n) -> jnp.ndarray (n, n_p)
        """
        self._sampler_fn = sampler_fn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        if self._n is None:
            raise RuntimeError("Call set_problem() first.")
        return self._n

    @property
    def m(self) -> int:
        if self._m is None:
            raise RuntimeError("Call set_problem() first.")
        return self._m

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, seed: int = 0, log_dir: Optional[str] = None) -> None:
        """Offline training phase (Algorithm 2, offline).

        Trains the neural proxy by minimising the ADMM residual loss
        over parameter samples drawn from the sampler.

        Args:
            seed:    PRNG seed for reproducibility
            log_dir: TensorBoard log directory, e.g. "runs/my_qp".
                     If None, no TensorBoard logging is performed.
                     Launch with: tensorboard --logdir=runs
        """
        self._check_ready()
        tp = self.Params.Train
        np_ = self.Params.Network

        # Determine n_p by probing the sampler
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        probe = self._sampler_fn(subkey, 1)
        n_p = probe.shape[1]

        # Build network
        key, subkey = jax.random.split(key)
        self._net, self._params = build_network(
            n=self._n, m=self._m, n_p=n_p,
            width=np_.width, depth=np_.depth,
            key=subkey,
        )

        resolved_log_dir = (
            f"{log_dir}/{self.name}" if log_dir is not None else None
        )
        if resolved_log_dir is not None:
            print(f"  TensorBoard: tensorboard --logdir={log_dir}")

        key, train_key = jax.random.split(key)
        self._params = _train(
            net=self._net,
            params=self._params,
            sampler_fn=self._sampler_fn,
            Q_fn=self._Q_fn,
            q_fn=self._q_fn,
            A_fn=self._A_fn,
            c_fn=self._c_fn,
            epochs=tp.epochs,
            batch_size=tp.batch_size,
            lr=tp.lr,
            alpha=tp.alpha,
            tol_off=tp.tol_off,
            log_every=tp.log_every,
            key=train_key,
            normalize_fn=self._normalize,
            log_dir=resolved_log_dir,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def solve(self, P: jnp.ndarray) -> SolveResult:
        """Online inference phase (Algorithm 2, online).

        Args:
            P: raw parameter vector(s), shape (n_p,) or (batch, n_p)

        Returns:
            SolveResult with fields x, feasible, certificate, used_backup
        """
        if self._params is None:
            raise RuntimeError("Call train() before solve().")
        sp = self.Params.Solve
        tp = self.Params.Train

        P = jnp.atleast_2d(P)
        P = self._normalize(P)

        return _solve(
            net=self._net,
            params=self._params,
            P=P,
            Q_fn=self._Q_fn,
            q_fn=self._q_fn,
            A_fn=self._A_fn,
            c_fn=self._c_fn,
            alpha=tp.alpha,
            tol_on=sp.tol_on,
            backup_steps=sp.backup_steps,
            backup_lr=sp.backup_lr,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self._Q_fn is None:
            raise RuntimeError("Call set_problem() first.")
        if self._sampler_fn is None:
            raise RuntimeError("Call set_sampler() first.")