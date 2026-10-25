from typing import Dict, Tuple
import jax.numpy as jnp


def build_qpslack(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    A: jnp.ndarray,
    c: jnp.ndarray,
    alpha: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build QPslack matrices (Q̄, q̄, Ā, c̄) from original QP data.

    The QPslack problem (Eqn. 2 in the paper) introduces slack variables
    ξ ∈ R^m to ensure feasibility for any parameter p:

        min   (1/2) x^T Q x + q^T x + (α/2)||ξ||²
        s.t.  A x + ξ >= c,  ξ >= 0

    Equivalently written as:
        min   (1/2) x̄^T Q̄ x̄ + q̄^T x̄
        s.t.  Ā x̄ >= c̄

    where x̄ = (x, ξ) ∈ R^{n+m}, and:
        Q̄ = diag(Q, αI)  ∈ R^{(n+m)×(n+m)}
        q̄ = (q, 0)       ∈ R^{n+m}
        Ā = [[A, I], [0, I]]  ∈ R^{2m×(n+m)}
        c̄ = (c, 0)       ∈ R^{2m}

    Args:
        Q:     shape (n, n), positive semidefinite
        q:     shape (n,)
        A:     shape (m, n)
        c:     shape (m,)
        alpha: slack penalization scalar (alpha > 0)

    Returns:
        Q̄: shape (n+m, n+m)
        q̄: shape (n+m,)
        Ā: shape (2m, n+m)
        c̄: shape (2m,)
    """
    n = q.shape[0]
    m = c.shape[0]

    Q_bar = jnp.block([
        [Q,                  jnp.zeros((n, m))],
        [jnp.zeros((m, n)),  alpha * jnp.eye(m)],
    ])
    q_bar = jnp.concatenate([q, jnp.zeros(m)])
    A_bar = jnp.block([
        [A,                  jnp.eye(m)],
        [jnp.zeros((m, n)),  jnp.eye(m)],
    ])
    c_bar = jnp.concatenate([c, jnp.zeros(m)])

    return Q_bar, q_bar, A_bar, c_bar


def admm_loss(
    xbar: jnp.ndarray,
    z:    jnp.ndarray,
    lam:  jnp.ndarray,
    Q:    jnp.ndarray,
    q:    jnp.ndarray,
    A:    jnp.ndarray,
    c:    jnp.ndarray,
    alpha: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """ADMM-informed residual loss for a single parameter instance.

    Computes (1/2)||r(x̄, z, λ; p)||² where r is defined in Eqn. (6):

        r1 = Q̄ x̄ + Ā^T Ā x̄ - Ā^T z - Ā^T λ + q̄   (stationarity)
        r2 = z - max(c̄, Ā x̄ - λ)                     (complementarity)
        r3 = z - Ā x̄                                   (primal feasibility)

    By Theorem 1, minimizing ||r||² drives (x̄, z, λ) toward the
    solution set of the QPslack problem.

    Args:
        xbar:  shape (n+m,) — primal variable x̄ = (x, ξ)
        z:     shape (2m,)  — auxiliary variable
        lam:   shape (2m,)  — dual variable λ
        Q:     shape (n, n) — QP cost matrix (positive semidefinite)
        q:     shape (n,)   — QP linear cost vector
        A:     shape (m, n) — constraint matrix
        c:     shape (m,)   — constraint rhs
        alpha: QPslack penalization scalar

    Returns:
        total:     scalar loss value (1/2)||r||²
        breakdown: dict with keys "stat", "comp", "prim"
    """
    Q_bar, q_bar, A_bar, c_bar = build_qpslack(Q, q, A, c, alpha)

    # r1: stationarity (Eqn. 6, first block)
    r1 = Q_bar @ xbar + A_bar.T @ A_bar @ xbar - A_bar.T @ z - A_bar.T @ lam + q_bar
    term1 = 0.5 * jnp.sum(r1 ** 2)

    # r2: complementarity (Eqn. 6, second block)
    r2 = z - jnp.maximum(c_bar, A_bar @ xbar - lam)
    term2 = 0.5 * jnp.sum(r2 ** 2)

    # r3: primal feasibility (Eqn. 6, third block)
    r3 = z - A_bar @ xbar
    term3 = 0.5 * jnp.sum(r3 ** 2)

    total = term1 + term2 + term3
    breakdown = {"stat": term1, "comp": term2, "prim": term3}
    return total, breakdown