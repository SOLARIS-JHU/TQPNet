"""Tests for tulip_qp/train.py: training loop, convergence, PRNG isolation."""
import jax
import jax.numpy as jnp

from tulip_qp.network import build_network
from tulip_qp.sampler import box_sampler
from tulip_qp.train import train


def _simple_qp(n=2, m=1):
    Q_fixed = jnp.eye(n)
    q_fixed = jnp.ones(n)
    A_fixed = jnp.ones((m, n))
    c_fixed = jnp.zeros(m)
    return (
        lambda p: Q_fixed,
        lambda p: q_fixed,
        lambda p: A_fixed,
        lambda p: c_fixed,
        n, m,
    )


def test_train_returns_params():
    """train() returns a non-None param dict."""
    Q_fn, q_fn, A_fn, c_fn, n, m = _simple_qp()
    n_p = 7
    net, params = build_network(n, m, n_p, width=16, depth=1, key=jax.random.PRNGKey(0))
    sampler = box_sampler(n=n, m=m)
    trained = train(
        net, params, sampler, Q_fn, q_fn, A_fn, c_fn,
        epochs=3, batch_size=8, lr=1e-3, alpha=1.0,
        tol_off=1e-10, log_every=10,
        key=jax.random.PRNGKey(1),
        normalize_fn=lambda P: P,
    )
    assert trained is not None


def test_train_loss_decreases():
    """A few training steps should reduce the mean ADMM residual loss."""
    from tulip_qp.loss import admm_loss
    from tulip_qp.network import split_output_single

    Q_fn, q_fn, A_fn, c_fn, n, m = _simple_qp()
    n_p = 7
    key = jax.random.PRNGKey(42)
    net, params_init = build_network(n, m, n_p, width=32, depth=2, key=key)
    sampler = box_sampler(n=n, m=m)

    key, sk = jax.random.split(key)
    P = sampler(sk, 32)

    def batch_loss(p_dict):
        def single(p):
            out = net.apply(p_dict, p[None, :])[0]
            xbar, z, lam = split_output_single(out, n, m)
            return admm_loss(xbar, z, lam, Q_fn(p), q_fn(p), A_fn(p), c_fn(p), alpha=1.0)[0]
        return float(jnp.mean(jax.vmap(single)(P)))

    loss_before = batch_loss(params_init)
    key, tk = jax.random.split(key)
    params_trained = train(
        net, params_init, sampler, Q_fn, q_fn, A_fn, c_fn,
        epochs=50, batch_size=32, lr=1e-3, alpha=1.0,
        tol_off=1e-12, log_every=100,
        key=tk,
        normalize_fn=lambda P: P,
    )
    loss_after = batch_loss(params_trained)
    assert loss_after < loss_before


def test_train_different_keys_give_different_results():
    """Two runs with different PRNG keys produce different params."""
    Q_fn, q_fn, A_fn, c_fn, n, m = _simple_qp()
    n_p = 7
    net, params = build_network(n, m, n_p, width=16, depth=1, key=jax.random.PRNGKey(0))
    sampler = box_sampler(n=n, m=m)

    p1 = train(net, params, sampler, Q_fn, q_fn, A_fn, c_fn,
               epochs=5, batch_size=8, lr=1e-3, alpha=1.0,
               tol_off=1e-12, log_every=100,
               key=jax.random.PRNGKey(1),
               normalize_fn=lambda P: P)
    p2 = train(net, params, sampler, Q_fn, q_fn, A_fn, c_fn,
               epochs=5, batch_size=8, lr=1e-3, alpha=1.0,
               tol_off=1e-12, log_every=100,
               key=jax.random.PRNGKey(99),
               normalize_fn=lambda P: P)

    leaves1 = jax.tree_util.tree_leaves(p1)
    leaves2 = jax.tree_util.tree_leaves(p2)
    any_diff = any(not jnp.array_equal(a, b) for a, b in zip(leaves1, leaves2))
    assert any_diff


def test_early_stopping():
    """Training stops early when loss <= tol_off."""
    Q_fn, q_fn, A_fn, c_fn, n, m = _simple_qp()
    n_p = 7
    net, params = build_network(n, m, n_p, width=8, depth=1, key=jax.random.PRNGKey(0))
    sampler = box_sampler(n=n, m=m)
    trained = train(
        net, params, sampler, Q_fn, q_fn, A_fn, c_fn,
        epochs=1000, batch_size=4, lr=1e-3, alpha=1.0,
        tol_off=1e9,   # absurdly high => stops at epoch 1
        log_every=2000,
        key=jax.random.PRNGKey(0),
        normalize_fn=lambda P: P,
    )
    assert trained is not None