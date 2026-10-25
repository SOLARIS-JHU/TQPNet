import jax.numpy as jnp
import pytest
from tulip_qp.loss import admm_loss, build_qpslack


def test_build_qpslack_shapes():
    """Q̄, q̄, Ā, c̄ have correct shapes."""
    n, m = 3, 2
    Q = jnp.eye(n)
    q = jnp.ones(n)
    A = jnp.ones((m, n))
    c = jnp.ones(m)
    Q_bar, q_bar, A_bar, c_bar = build_qpslack(Q, q, A, c, alpha=1.0)
    assert Q_bar.shape == (n + m, n + m)
    assert q_bar.shape == (n + m,)
    assert A_bar.shape == (2 * m, n + m)
    assert c_bar.shape == (2 * m,)


def test_build_qpslack_Q_bar_psd():
    """Q̄ = diag(Q, αI) is positive definite when Q >= 0 and α > 0."""
    n, m = 3, 2
    Q = jnp.eye(n)
    q = jnp.zeros(n)
    A = jnp.zeros((m, n))
    c = jnp.zeros(m)
    Q_bar, _, _, _ = build_qpslack(Q, q, A, c, alpha=1.0)
    eigvals = jnp.linalg.eigvalsh(Q_bar)
    assert jnp.all(eigvals >= 0.0)


def test_build_qpslack_c_bar_structure():
    """c̄ = (c, 0): first m entries are c, last m entries are 0."""
    n, m = 3, 2
    Q = jnp.eye(n)
    q = jnp.zeros(n)
    A = jnp.zeros((m, n))
    c = jnp.array([1.0, 2.0])
    _, _, _, c_bar = build_qpslack(Q, q, A, c, alpha=1.0)
    assert jnp.allclose(c_bar[:m], c)
    assert jnp.allclose(c_bar[m:], jnp.zeros(m))


def test_admm_loss_shapes():
    """admm_loss returns scalar total and breakdown dict."""
    n, m = 3, 2
    xbar = jnp.zeros(n + m)
    z    = jnp.zeros(2 * m)
    lam  = jnp.zeros(2 * m)
    Q    = jnp.eye(n)
    q    = jnp.ones(n)
    A    = jnp.ones((m, n))
    c    = jnp.zeros(m)
    total, breakdown = admm_loss(xbar, z, lam, Q, q, A, c, alpha=1.0)
    assert total.shape == ()
    assert set(breakdown.keys()) == {"stat", "comp", "prim"}


def test_admm_loss_nonnegative():
    """ADMM residual loss is always >= 0."""
    n, m = 3, 2
    xbar = jnp.ones(n + m)
    z    = jnp.ones(2 * m)
    lam  = jnp.ones(2 * m)
    Q    = jnp.eye(n)
    q    = jnp.ones(n)
    A    = jnp.ones((m, n))
    c    = jnp.ones(m)
    total, _ = admm_loss(xbar, z, lam, Q, q, A, c, alpha=1.0)
    assert float(total) >= 0.0


def test_admm_loss_at_zero_is_positive():
    """Loss > 0 when all variables are zero (not a valid solution)."""
    n, m = 2, 2
    xbar = jnp.zeros(n + m)
    z    = jnp.zeros(2 * m)
    lam  = jnp.zeros(2 * m)
    Q    = jnp.eye(n)
    q    = jnp.ones(n)
    A    = jnp.ones((m, n))
    c    = jnp.ones(m)
    total, _ = admm_loss(xbar, z, lam, Q, q, A, c, alpha=1.0)
    assert float(total) > 0.0


def test_admm_loss_breakdown_sums_to_total():
    """Sum of breakdown terms equals total loss."""
    n, m = 3, 2
    xbar = jnp.ones(n + m) * 0.5
    z    = jnp.ones(2 * m) * 0.3
    lam  = jnp.ones(2 * m) * 0.1
    Q    = jnp.eye(n)
    q    = jnp.ones(n)
    A    = jnp.ones((m, n))
    c    = jnp.zeros(m)
    total, breakdown = admm_loss(xbar, z, lam, Q, q, A, c, alpha=1.0)
    assert jnp.allclose(
        total,
        breakdown["stat"] + breakdown["comp"] + breakdown["prim"],
        atol=1e-6,
    )