import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tqpnet.admm import build_operator
from tqpnet.network import build_network
from tqpnet.train import batch_loss, train

from .problems import q_fn_c_fn, random_qp


def _setup(n=4, m=8, n_p=2, seed=0):
    prob = random_qp(n, m, n_p, seed=seed)
    q_fn, c_fn = q_fn_c_fn(prob)
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    net, var = build_network(n, m, n_p, width=32, depth=2, key=jax.random.PRNGKey(seed))
    data = jax.random.uniform(jax.random.PRNGKey(seed + 1), (64, n_p), minval=-1.0, maxval=1.0)
    return op, net, var, data, q_fn, c_fn


def test_loss_decreases():
    op, net, var, data, q_fn, c_fn = _setup()
    before = float(batch_loss(var, net, op, data, q_fn, c_fn, K=5))
    var2, hist = train(net, var, op, data, q_fn, c_fn, epochs=30, batch_size=16, lr=1e-3,
                       weight_decay=1e-5, K_train=5, tol_off=0.0, log_every=1000,
                       key=jax.random.PRNGKey(0))
    assert float(batch_loss(var2, net, op, data, q_fn, c_fn, K=5)) < before
    assert hist["epochs"] == 30 and len(hist["loss"]) == 30


def test_early_stop_and_history():
    op, net, var, data, q_fn, c_fn = _setup()
    _, hist = train(net, var, op, data, q_fn, c_fn, epochs=50, batch_size=16, lr=1e-3,
                    weight_decay=0.0, K_train=5, tol_off=1e9, log_every=1000,
                    key=jax.random.PRNGKey(0))
    assert hist["epochs"] == 1 and len(hist["loss"]) == 1


def test_dataset_smaller_than_batch_raises():
    op, net, var, data, q_fn, c_fn = _setup()
    with pytest.raises(ValueError):
        train(net, var, op, data[:8], q_fn, c_fn, epochs=1, batch_size=16, lr=1e-3,
              weight_decay=0.0, K_train=5, tol_off=0.0, log_every=1000, key=jax.random.PRNGKey(0))


def test_partial_batch_dropped_and_keys_matter():
    op, net, var, data, q_fn, c_fn = _setup()
    v1, h1 = train(net, var, op, data[:50], q_fn, c_fn, epochs=2, batch_size=16, lr=1e-3,
                   weight_decay=0.0, K_train=5, tol_off=0.0, log_every=1000, key=jax.random.PRNGKey(1))
    v2, _ = train(net, var, op, data[:50], q_fn, c_fn, epochs=2, batch_size=16, lr=1e-3,
                  weight_decay=0.0, K_train=5, tol_off=0.0, log_every=1000, key=jax.random.PRNGKey(2))
    assert not bool(jnp.array_equal(v1["params"]["Dense_0"]["kernel"], v2["params"]["Dense_0"]["kernel"]))
    assert h1["updates"] == 2 * (50 // 16)   # last partial batch dropped


def test_nan_does_not_early_stop_but_terminates():
    op, net, var, data, q_fn, c_fn = _setup()
    _, hist = train(net, var, op, data, lambda p: q_fn(p) * jnp.nan, c_fn, epochs=3, batch_size=16,
                    lr=1e-3, weight_decay=0.0, K_train=5, tol_off=1e-8, log_every=1000,
                    key=jax.random.PRNGKey(0))
    assert hist["epochs"] == 3 and all(np.isnan(hist["loss"])) and hist["updates"] == 0


def test_lr_schedule_halves_every_step_epochs():
    op, net, var, data, q_fn, c_fn = _setup()
    _, hist = train(net, var, op, data, q_fn, c_fn, epochs=5, batch_size=16, lr=1e-3,
                    weight_decay=0.0, K_train=5, tol_off=0.0, log_every=1000, key=jax.random.PRNGKey(0),
                    lr_step_epochs=2, lr_gamma=0.5)
    assert hist["lr"] == [1e-3, 1e-3, 5e-4, 5e-4, 2.5e-4]


def test_infinite_loss_batches_are_recorded_as_nan():
    op, net, var, data, q_fn, c_fn = _setup()
    # q offset of 1e200 makes the squared residual overflow to +inf without producing NaN
    _, hist = train(net, var, op, data, lambda p: q_fn(p) + 1e200, c_fn, epochs=2, batch_size=16,
                    lr=1e-3, weight_decay=0.0, K_train=5, tol_off=1e-8, log_every=1000,
                    key=jax.random.PRNGKey(0))
    assert all(np.isnan(hist["loss"])) and all(np.isnan(hist["residual"])) and hist["updates"] == 0
