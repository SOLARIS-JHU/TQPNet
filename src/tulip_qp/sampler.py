from typing import Callable, Literal
import jax
import jax.numpy as jnp


def box_sampler(
    n: int,
    m: int,
    A_lb: float = -1.0, A_ub: float =  1.0,
    b_lb: float = -1.0, b_ub: float =  1.0,
    q_lb: float =  0.0, q_ub: float =  2.0,
    Q_lb: float =  0.0, Q_ub: float =  1.0,
    x_lb: float =  0.0, x_ub: float =  2.0,
    slack_ub: float = 0.5,
    farkas_eps: float = 0.1,
    mode: Literal["feasible", "infeasible", "random"] = "feasible",
) -> Callable:
    """Return a sampler for the QP family:
        min  (1/2) x^T Q x + q^T x   s.t.  A x >= c,  x >= 0

    Pack order: p = [vec(Q_diag), vec(A), c, q]
    where Q is diagonal PSD (Q_diag are the diagonal entries).

    n_p = n + m*n + m + n = 2n + m*n + m

    Args:
        n, m:       problem dimensions
        A_lb/ub:    bounds for A entries
        b_lb/ub:    bounds for c (constraint rhs) in "random" mode
        q_lb/ub:    bounds for linear cost vector q
        Q_lb/ub:    bounds for diagonal entries of Q (ensures Q >= 0)
        x_lb/ub:    bounds for feasible point in "feasible" mode
        slack_ub:   slack upper bound in "feasible" mode
        farkas_eps: Farkas certificate margin in "infeasible" mode
        mode:       "feasible"   — strict feasibility via constructive sampling
                    "infeasible" — strict infeasibility via Farkas certificate
                    "random"     — pure uniform sampling, no guarantee

    Returns:
        sampler(key, n_samples) -> jnp.ndarray of shape (n_samples, n_p)
    """
    def sampler(key: jax.Array, n_samples: int) -> jnp.ndarray:
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

        # Q: diagonal PSD, shape (n_samples, n)
        Q_diag = jax.random.uniform(k1, (n_samples, n), minval=Q_lb, maxval=Q_ub)

        # q: linear cost, shape (n_samples, n)
        q = jax.random.uniform(k4, (n_samples, n), minval=q_lb, maxval=q_ub)

        if mode == "feasible":
            # A x >= c guaranteed: c = A @ x_feas - slack
            A      = jax.random.uniform(k2, (n_samples, m, n), minval=A_lb, maxval=A_ub)
            x_feas = jax.random.uniform(k3, (n_samples, n),    minval=x_lb, maxval=x_ub)
            slack  = jax.random.uniform(k5, (n_samples, m),    minval=0.0,  maxval=slack_ub)
            c      = jnp.einsum("ijk,ik->ij", A, x_feas) - slack

        elif mode == "infeasible":
            # Farkas certificate: y >= 0, A^T y <= 0, c^T y > 0
            y = jax.random.uniform(k2, (n_samples, m),    minval=0.1, maxval=1.0)
            Z = jax.random.uniform(k3, (n_samples, m, n), minval=0.0, maxval=1.0)
            A = -Z
            t = jax.random.uniform(k5, (n_samples, 1),    minval=0.0, maxval=1.0)
            c = t * y + farkas_eps

        else:  # random
            A = jax.random.uniform(k2, (n_samples, m, n), minval=A_lb, maxval=A_ub)
            c = jax.random.uniform(k3, (n_samples, m),    minval=b_lb, maxval=b_ub)

        return jnp.concatenate([
            Q_diag,
            A.reshape(n_samples, m * n),
            c,
            q,
        ], axis=1)

    return sampler