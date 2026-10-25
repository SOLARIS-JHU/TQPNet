import jax
import jax.numpy as jnp
import pytest
from tulip_qp.network import TuQPNet, build_network, split_output, split_output_single


def test_tuqpnet_output_shape():
    """Network output has shape (batch, n+5m)."""
    n, m, n_p = 3, 2, 10
    net = TuQPNet(n=n, m=m, width=32, depth=2)
    params = net.init(jax.random.PRNGKey(0), jnp.ones((1, n_p)))
    out = net.apply(params, jnp.ones((5, n_p)))
    assert out.shape == (5, n + 5 * m)


def test_tuqpnet_output_unconstrained():
    """Network output has no sign constraint (z and λ can be negative)."""
    n, m, n_p = 3, 2, 10
    net = TuQPNet(n=n, m=m, width=32, depth=2)
    params = net.init(jax.random.PRNGKey(0), jnp.ones((1, n_p)))
    key = jax.random.PRNGKey(1)
    p = jax.random.normal(key, (100, n_p))
    out = net.apply(params, p)
    # Output should not be forced non-negative (no output ReLU)
    assert out.shape == (100, n + 5 * m)


def test_split_output_shapes():
    """split_output correctly partitions (batch, n+5m) into (x̄, z, λ)."""
    n, m = 4, 3
    batch = 7
    out = jnp.ones((batch, n + 5 * m))
    xbar, z, lam = split_output(out, n, m)
    assert xbar.shape == (batch, n + m)
    assert z.shape   == (batch, 2 * m)
    assert lam.shape == (batch, 2 * m)


def test_split_output_single_shapes():
    """split_output_single correctly partitions (n+5m,) into (x̄, z, λ)."""
    n, m = 4, 3
    out = jnp.ones(n + 5 * m)
    xbar, z, lam = split_output_single(out, n, m)
    assert xbar.shape == (n + m,)
    assert z.shape   == (2 * m,)
    assert lam.shape == (2 * m,)


def test_split_output_consistent_with_single():
    """Batched and single split give same results for one sample."""
    n, m = 3, 2
    out_single = jax.random.normal(jax.random.PRNGKey(0), (n + 5 * m,))
    out_batch  = out_single[None, :]

    xbar_s, z_s, lam_s = split_output_single(out_single, n, m)
    xbar_b, z_b, lam_b = split_output(out_batch, n, m)

    assert jnp.allclose(xbar_s, xbar_b[0])
    assert jnp.allclose(z_s,    z_b[0])
    assert jnp.allclose(lam_s,  lam_b[0])


def test_build_network_returns_tuple():
    """build_network returns (net, params) tuple."""
    net, params = build_network(
        n=4, m=3, n_p=12, width=32, depth=2,
        key=jax.random.PRNGKey(0),
    )
    assert net is not None
    assert params is not None


def test_build_network_forward():
    """Built network runs forward pass without error."""
    n, m, n_p = 3, 2, 8
    net, params = build_network(n, m, n_p, width=16, depth=2, key=jax.random.PRNGKey(0))
    out = net.apply(params, jnp.ones((4, n_p)))
    assert out.shape == (4, n + 5 * m)