"""Tests for tulip_qp/solve.py: backup correction, feasibility, SolveResult."""
import jax
import jax.numpy as jnp
import pytest

from tulip_qp.network import TuQPNet, build_network
from tulip_qp.solve import SolveResult, _backup_correction, solve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trivial_qp():
    """1-variable, 1-constraint QP: min (1/2)x² + x s.t. x >= 1, x >= 0.
    Optimal: x* = 1.
    """
    n, m = 1, 1
    Q = jnp.array([[1.0]])
    q = jnp.array([1.0])
    A = jnp.array([[1.0]])
    c = jnp.array([1.0])
    return n, m, Q, q, A, c


def _make_net_params(n, m, n_p, width=16, depth=1):
    net, params = build_network(n, m, n_p, width, depth, jax.random.PRNGKey(0))
    return net, params


# ---------------------------------------------------------------------------
# Backup correction
# ---------------------------------------------------------------------------

def test_backup_correction_returns_correct_shapes():
    """_backup_correction returns same-shape tensors as input."""
    n, m, Q, q, A, c = _trivial_qp()
    xbar = jnp.ones(n + m)
    z    = jnp.ones(2 * m)
    lam  = jnp.ones(2 * m)
    xb2, z2, l2 = _backup_correction(
        xbar, z, lam, Q, q, A, c, alpha=1.0,
        backup_steps=5, backup_lr=1e-3,
    )
    assert xb2.shape == xbar.shape
    assert z2.shape  == z.shape
    assert l2.shape  == lam.shape


def test_backup_correction_reduces_loss():
    """After backup correction, ADMM loss should be <= starting loss."""
    from tulip_qp.loss import admm_loss
    n, m, Q, q, A, c = _trivial_qp()
    xbar = jnp.zeros(n + m)
    z    = jnp.zeros(2 * m)
    lam  = jnp.zeros(2 * m)
    loss_before, _ = admm_loss(xbar, z, lam, Q, q, A, c, alpha=1.0)
    xb2, z2, l2 = _backup_correction(
        xbar, z, lam, Q, q, A, c, alpha=1.0,
        backup_steps=200, backup_lr=1e-3,
    )
    loss_after, _ = admm_loss(xb2, z2, l2, Q, q, A, c, alpha=1.0)
    assert float(loss_after) <= float(loss_before)


# ---------------------------------------------------------------------------
# Feasibility detection
# ---------------------------------------------------------------------------

def test_feasibility_flag_shape():
    """feasible has shape (batch,) and dtype bool."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 4
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((6, n_p))
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=1e20,
        backup_steps=0, backup_lr=1e-3,
    )
    assert result.feasible.shape == (6,)
    assert result.feasible.dtype == jnp.bool_


def test_feasibility_via_certificate():
    """Samples with ε <= tol_on are declared feasible."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 4
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((4, n_p))
    # With tol_on=inf, all samples should be feasible (ε <= inf always)
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=1e30,
        backup_steps=0, backup_lr=1e-3,
    )
    assert jnp.all(result.feasible)


def test_x_star_shape():
    """x* has shape (batch, n) — slack ξ is stripped."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 3
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((8, n_p))
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=1e20,
        backup_steps=0, backup_lr=1e-3,
    )
    assert result.x.shape == (8, n)


# ---------------------------------------------------------------------------
# Backup triggers
# ---------------------------------------------------------------------------

def test_used_backup_flag_shape():
    """used_backup has shape (batch,) and dtype bool."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 2
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((6, n_p))
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=0.0,   # force backup on all
        backup_steps=2, backup_lr=1e-3,
    )
    assert result.used_backup.shape == (6,)
    assert result.used_backup.dtype == jnp.bool_
    assert jnp.all(result.used_backup)


def test_no_backup_when_disabled():
    """When backup_steps=0, used_backup is all False."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 2
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((4, n_p))
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=0.0,
        backup_steps=0, backup_lr=1e-3,
    )
    assert not jnp.any(result.used_backup)


def test_certificate_shape_and_nonneg():
    """certificate has shape (batch,) and is non-negative."""
    n, m, Q, q, A, c = _trivial_qp()
    n_p = 2
    net, params = _make_net_params(n, m, n_p)
    P = jnp.ones((5, n_p))
    result = solve(
        net, params, P,
        Q_fn=lambda p: Q,
        q_fn=lambda p: q,
        A_fn=lambda p: A,
        c_fn=lambda p: c,
        alpha=1.0, tol_on=1e20,
        backup_steps=0, backup_lr=1e-3,
    )
    assert result.certificate.shape == (5,)
    assert jnp.all(result.certificate >= 0.0)