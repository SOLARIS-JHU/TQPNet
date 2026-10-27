import jax
import jax.numpy as jnp

from tqpnet.network import WarmStartNet, build_network


def test_output_shapes_and_nonneg_beta():
    net, params = build_network(n=3, m=5, n_p=2, width=8, depth=2, key=jax.random.PRNGKey(0))
    assert isinstance(net, WarmStartNet)
    x0, beta0 = net.apply(params, jnp.ones((7, 2)))
    assert x0.shape == (7, 3) and beta0.shape == (7, 5)
    assert bool(jnp.all(beta0 >= 0))


def test_depth_and_init():
    net, params = build_network(n=3, m=5, n_p=2, width=8, depth=4, key=jax.random.PRNGKey(0))
    dense = [k for k in params["params"] if k.startswith("Dense")]
    assert len(dense) == 4 + 2
    assert all(bool(jnp.all(params["params"][k]["bias"] == 0)) for k in dense)


def test_different_keys_differ():
    _, p1 = build_network(3, 5, 2, 8, 2, jax.random.PRNGKey(0))
    _, p2 = build_network(3, 5, 2, 8, 2, jax.random.PRNGKey(1))
    assert not bool(jnp.array_equal(p1["params"]["Dense_0"]["kernel"], p2["params"]["Dense_0"]["kernel"]))
