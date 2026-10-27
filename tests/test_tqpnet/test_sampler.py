import jax
import jax.numpy as jnp

from tqpnet import Params, box_sampler


def test_box_sampler_shape_and_bounds():
    fn = box_sampler(lo=[-1.0, 0.0, 2.0], hi=[1.0, 0.5, 3.0])
    P = fn(jax.random.PRNGKey(0), 100)
    assert P.shape == (100, 3)
    assert bool(jnp.all(P >= jnp.array([-1.0, 0.0, 2.0])))
    assert bool(jnp.all(P <= jnp.array([1.0, 0.5, 3.0])))


def test_box_sampler_is_deterministic_per_key():
    fn = box_sampler([0.0], [1.0])
    assert bool(jnp.array_equal(fn(jax.random.PRNGKey(3), 5), fn(jax.random.PRNGKey(3), 5)))
    assert not bool(jnp.array_equal(fn(jax.random.PRNGKey(3), 5), fn(jax.random.PRNGKey(4), 5)))


def test_params_defaults():
    p = Params()
    assert (p.Network.width, p.Network.depth) == (256, 4)
    assert (p.Train.n_train, p.Train.epochs, p.Train.batch_size, p.Train.K_train, p.Train.rho) == (
        6000, 2000, 256, 10, 1.0)
    assert (p.Train.lr, p.Train.weight_decay, p.Train.tol_off, p.Train.log_every) == (1e-3, 1e-5, 1e-8, 100)
    assert (p.Train.grad_clip, p.Train.lr_step_epochs, p.Train.lr_gamma) == (10.0, 300, 0.5)
    assert (p.Solve.K_infer, p.Solve.tol_on, p.Solve.K_max) == (None, 1e-4, 1000)
