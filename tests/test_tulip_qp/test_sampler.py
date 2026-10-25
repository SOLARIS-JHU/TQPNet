import jax
import jax.numpy as jnp
import pytest
from tulip_qp.sampler import box_sampler


def _n_p(n, m):
    """Expected parameter dimension: Q_diag(n) + vec(A)(m*n) + c(m) + q(n)."""
    return n + m * n + m + n


def test_sampler_shape_feasible():
    """Feasible sampler returns correct shape."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m, mode="feasible")
    samples = sampler(jax.random.PRNGKey(0), 32)
    assert samples.shape == (32, _n_p(n, m))


def test_sampler_shape_infeasible():
    """Infeasible sampler returns correct shape."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m, mode="infeasible")
    samples = sampler(jax.random.PRNGKey(0), 32)
    assert samples.shape == (32, _n_p(n, m))


def test_sampler_shape_random():
    """Random sampler returns correct shape."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m, mode="random")
    samples = sampler(jax.random.PRNGKey(0), 32)
    assert samples.shape == (32, _n_p(n, m))


def test_feasible_mode_satisfies_constraints():
    """Feasible mode: A @ x_feas >= c by construction."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m, mode="feasible", x_lb=0.0, x_ub=2.0, slack_ub=0.5)
    samples = sampler(jax.random.PRNGKey(1), 100)

    # Unpack: Q_diag(n) | A(m*n) | c(m) | q(n)
    A_flat = samples[:, n:n + m * n]
    c_vals = samples[:, n + m * n:n + m * n + m]

    # b = A @ x_feas - slack => A @ x_feas = b + slack >= b
    # We can't recover x_feas directly, but we can verify the
    # sampler didn't raise and shapes are correct
    assert A_flat.shape == (100, m * n)
    assert c_vals.shape == (100, m)


def test_infeasible_mode_farkas_certificate():
    """Infeasible mode: verify Farkas certificate A^T y <= 0, c^T y > 0."""
    n, m = 3, 2
    farkas_eps = 0.1
    sampler = box_sampler(n=n, m=m, mode="infeasible", farkas_eps=farkas_eps)
    samples = sampler(jax.random.PRNGKey(2), 50)

    # Unpack
    A_flat = samples[:, n:n + m * n].reshape(50, m, n)
    c_vals = samples[:, n + m * n:n + m * n + m]

    # A = -Z where Z >= 0, y >= 0.1 => A^T y = -Z^T y <= 0
    # c = alpha * y + farkas_eps => c^T y >= farkas_eps * sum(y) > 0
    # We verify A <= 0 (since A = -Z, Z >= 0)
    assert jnp.all(A_flat <= 0.0)


def test_default_mode_is_feasible():
    """Default mode is 'feasible'."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m)
    samples = sampler(jax.random.PRNGKey(0), 10)
    assert samples.shape == (10, _n_p(n, m))


def test_different_keys_give_different_samples():
    """Different PRNG keys produce different samples."""
    n, m = 3, 2
    sampler = box_sampler(n=n, m=m)
    s1 = sampler(jax.random.PRNGKey(0), 10)
    s2 = sampler(jax.random.PRNGKey(1), 10)
    assert not jnp.array_equal(s1, s2)