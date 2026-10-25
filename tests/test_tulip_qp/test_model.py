import jax
import jax.numpy as jnp
import pytest
import tulip_qp
from tulip_qp.model import Model
from tulip_qp.solve import SolveResult
from tulip_qp.sampler import box_sampler


def _make_model():
    """Helper: create a small configured model for testing."""
    model = Model("test_qp")
    n, m = 3, 2
    Q_fixed = jnp.eye(n)
    q_fixed = jnp.ones(n)
    A_fixed = jnp.ones((m, n))
    c_fixed = jnp.zeros(m)
    model.set_problem(
        Q_fn=lambda p: Q_fixed,
        q_fn=lambda p: q_fixed,
        A_fn=lambda p: A_fixed,
        c_fn=lambda p: c_fixed,
        n=n, m=m,
    )
    model.set_sampler(box_sampler(n=n, m=m))
    model.Params.Train.epochs     = 2
    model.Params.Train.batch_size = 8
    model.Params.Network.width    = 16
    model.Params.Network.depth    = 1
    return model


def test_model_set_problem():
    model = _make_model()
    assert model.n == 3
    assert model.m == 2


def test_model_train_runs():
    model = _make_model()
    model.train()   # should not raise


def test_model_solve_returns_result():
    model = _make_model()
    model.train()
    n_p = model._sampler_fn(jax.random.PRNGKey(0), 1).shape[1]
    P = jnp.ones((5, n_p))
    result = model.solve(P)
    assert isinstance(result, SolveResult)
    assert result.x.shape          == (5, 3)
    assert result.feasible.shape   == (5,)
    assert result.certificate.shape == (5,)


def test_model_solve_single_sample():
    model = _make_model()
    model.train()
    n_p = model._sampler_fn(jax.random.PRNGKey(0), 1).shape[1]
    p = jnp.ones(n_p)
    result = model.solve(p)
    assert result.x.shape == (1, 3)


def test_model_check_ready_raises():
    """Calling train() without set_problem() raises RuntimeError."""
    model = Model("empty")
    with pytest.raises(RuntimeError):
        model.train()


def test_model_solve_before_train_raises():
    """Calling solve() before train() raises RuntimeError."""
    model = _make_model()
    n_p = model._sampler_fn(jax.random.PRNGKey(0), 1).shape[1]
    with pytest.raises(RuntimeError):
        model.solve(jnp.ones((3, n_p)))

