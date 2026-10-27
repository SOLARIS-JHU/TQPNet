"""Shared test problems: the paper's random mpQP generator and a cvxpy reference.

``random_qp`` mirrors ``experiments/base.py::build_config`` (same formulas, numpy
instead of torch, with ``a_scale=3.0`` from ``experiments/config.py`` and
``build_config``'s default ``c_offset=-1.0``; ``training_ablation.py`` overrides the
latter to +1.0) so the JAX package is tested on the family the paper uses.
"""
import cvxpy as cp
import jax.numpy as jnp
import numpy as np


def random_qp(n: int, m: int, n_p: int, seed: int = 0, density: float = 0.3,
              a_scale: float = 3.0, c_offset: float = -1.0) -> dict:
    rng = np.random.default_rng(seed)

    L = 0.3 * rng.standard_normal((n, n))
    Q = L @ L.T / n + 0.1 * np.eye(n)
    Q = 0.5 * (Q + Q.T)

    # Uniform column sparsity: exactly k non-zeros per column, then fill empty rows.
    k = max(1, int(np.ceil(m * density)))
    mask = np.zeros((m, n))
    for j in range(n):
        mask[rng.permutation(m)[:k], j] = 1.0
    for i in range(m):
        if mask[i].sum() == 0:
            mask[i, rng.integers(0, n)] = 1.0
    A = mask * a_scale * rng.random((m, n))

    q0 = rng.standard_normal(n)
    Cq = 0.3 * rng.standard_normal((n, n_p))
    c0 = A @ rng.standard_normal(n) + c_offset
    Cc = 0.1 * rng.standard_normal((m, n_p))
    return {"Q": Q, "A": A, "q0": q0, "Cq": Cq, "c0": c0, "Cc": Cc}


def q_fn_c_fn(prob: dict):
    q0, Cq = jnp.asarray(prob["q0"]), jnp.asarray(prob["Cq"])
    c0, Cc = jnp.asarray(prob["c0"]), jnp.asarray(prob["Cc"])
    return (lambda p: q0 + Cq @ p), (lambda p: c0 + Cc @ p)


def cvxpy_solve(Q, A, q, c):
    """Solve min ½xᵀQx + qᵀx s.t. Ax ≥ c with OSQP; return (x*, β*) with β* ≥ 0."""
    Q, A, q, c = (np.asarray(v, dtype=float) for v in (Q, A, q, c))
    x = cp.Variable(Q.shape[0])
    con = A @ x >= c
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q)) + q @ x), [con])
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000, polishing=True)
    if x.value is None:
        raise RuntimeError(f"cvxpy failed: {prob.status}")
    return np.asarray(x.value), np.asarray(con.dual_value).reshape(-1)
