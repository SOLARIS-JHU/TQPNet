import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tqpnet import Model, SolveResult, box_sampler

from .problems import cvxpy_solve, q_fn_c_fn, random_qp


def _model(n=4, m=8, n_p=2, seed=0):
    prob = random_qp(n, m, n_p, seed=seed)
    q_fn, c_fn = q_fn_c_fn(prob)
    model = Model("test")
    model.set_problem(prob["Q"], prob["A"], q_fn, c_fn)
    model.set_sampler(box_sampler([-1.0] * n_p, [1.0] * n_p))
    model.Params.Network.width, model.Params.Network.depth = 32, 2
    model.Params.Train.n_train, model.Params.Train.batch_size = 256, 64
    model.Params.Train.epochs, model.Params.Train.log_every = 100, 1000
    return model, prob, q_fn, c_fn


def test_set_problem_validation():
    prob = random_qp(4, 8, 2)
    q_fn, c_fn = q_fn_c_fn(prob)
    m = Model()
    with pytest.raises(ValueError):
        m.set_problem(np.ones((4, 3)), prob["A"], q_fn, c_fn)          # Q not square
    with pytest.raises(ValueError):
        m.set_problem(prob["Q"], np.ones((8, 5)), q_fn, c_fn)          # A columns != n
    Qa = prob["Q"].copy(); Qa[0, 1] += 1e-3
    with pytest.raises(ValueError):
        m.set_problem(Qa, prob["A"], q_fn, c_fn)                        # not symmetric
    Az = prob["A"].copy(); Az[0, :] = 0
    with pytest.raises(ValueError):
        m.set_problem(prob["Q"], Az, q_fn, c_fn)                        # zero row
    Ac = prob["A"].copy(); Ac[:, 0] = 0
    with pytest.raises(ValueError):
        m.set_problem(prob["Q"], Ac, q_fn, c_fn)                        # zero column
    m.set_problem(prob["Q"], prob["A"], q_fn, c_fn)
    assert (m.n, m.m) == (4, 8)


def test_runtime_errors_before_setup():
    m = Model()
    with pytest.raises(RuntimeError):
        m.train()
    model, *_ = _model()
    with pytest.raises(RuntimeError):
        model.solve(jnp.zeros((1, 2)))


def test_q_fn_wrong_shape_raises_at_train():
    model, prob, q_fn, c_fn = _model()
    model.set_problem(prob["Q"], prob["A"], lambda p: jnp.zeros(3), c_fn)
    with pytest.raises(ValueError):
        model.train()


def test_end_to_end_matches_cvxpy():
    model, prob, q_fn, c_fn = _model()
    hist = model.train(seed=0)
    assert hist["loss"][-1] < hist["loss"][0]
    model.Params.Solve.tol_on, model.Params.Solve.K_max = 1e-10, 5000
    P = jax.random.uniform(jax.random.PRNGKey(5), (5, 2), minval=-1.0, maxval=1.0)
    r = model.solve(P)
    assert isinstance(r, SolveResult) and r.x.shape == (5, 4) and bool(jnp.all(r.certified))
    for i in range(5):
        x_star, _ = cvxpy_solve(prob["Q"], prob["A"], np.asarray(q_fn(P[i])), np.asarray(c_fn(P[i])))
        assert np.max(np.abs(np.asarray(r.x[i]) - x_star)) < 1e-3
    # learned warm start: K_train layers alone must beat the untrained loss on fresh samples
    model.Params.Solve.tol_on, model.Params.Solve.K_max = 1e-4, 0
    assert float(jnp.mean(jnp.log1p(model.solve(P).certificate))) < hist["loss"][0]


def test_train_seed_changes_result():
    m1, *_ = _model()
    m2, *_ = _model()
    m1.Params.Train.epochs = m2.Params.Train.epochs = 2
    m1.train(seed=0)
    m2.train(seed=1)
    P = jnp.zeros((1, 2))
    assert not bool(jnp.allclose(m1.solve(P).x, m2.solve(P).x))


def test_single_sample_and_wrong_np():
    model, *_ = _model()
    model.Params.Train.epochs = 2
    model.train()
    r = model.solve(jnp.zeros(2))
    assert r.x.shape == (1, 4)
    with pytest.raises(ValueError, match=r"2.*\(3, 5\)"):
        model.solve(jnp.zeros((3, 5)))
    with pytest.raises(ValueError, match=r"\(5,\)"):      # the user's shape, not the atleast_2d one
        model.solve(jnp.zeros(5))
    with pytest.raises(ValueError):
        model.solve(jnp.zeros((2, 3, 2)))


def test_float32_q_with_rounding_asymmetry_is_accepted():
    # Q = Bᵀ D B + 0.1 I computed in float32 (default JAX precision) is symmetric up to rounding.
    rng = np.random.default_rng(0)
    B = jnp.asarray(rng.normal(size=(30, 30)), dtype=jnp.float32)
    D = jnp.asarray(rng.uniform(0.5, 2.0, size=30), dtype=jnp.float32)
    Q = B.T @ (D[:, None] * B) + 0.1 * jnp.eye(30, dtype=jnp.float32)
    assert float(jnp.max(jnp.abs(Q - Q.T))) > 0.0
    prob = random_qp(30, 40, 2)
    q_fn, c_fn = q_fn_c_fn(prob)
    m = Model()
    m.set_problem(Q, prob["A"], q_fn, c_fn)   # must not raise
    assert m.n == 30


def test_float32_inputs_under_x64_do_not_crash():
    model, prob, q_fn, c_fn = _model()
    model.Params.Train.epochs = 2
    model.set_sampler(lambda key, n: jax.random.uniform(key, (n, 2), dtype=jnp.float32, minval=-1.0, maxval=1.0))
    model.train()
    r = model.solve(np.zeros((2, 2), dtype=np.float32))
    assert r.x.shape == (2, 4) and bool(jnp.all(jnp.isfinite(r.x)))


def test_k_infer_defaults_to_the_k_train_used_in_training():
    model, *_ = _model()
    model.Params.Train.epochs, model.Params.Train.K_train = 2, 3
    model.train()
    model.Params.Train.K_train = 7          # changed after training: must not affect inference depth
    model.Params.Solve.K_max = 0
    assert bool(jnp.all(model.solve(jnp.zeros((2, 2))).steps == 3))


@pytest.mark.parametrize("field, value", [("K_infer", -1), ("K_max", -1)])
def test_negative_solve_depths_are_rejected(field, value):
    model, *_ = _model()
    model.Params.Train.epochs = 2
    model.train()
    setattr(model.Params.Solve, field, value)
    with pytest.raises(ValueError, match=field):
        model.solve(jnp.zeros((1, 2)))


def test_negative_k_train_is_rejected():
    model, *_ = _model()
    model.Params.Train.K_train = -1
    with pytest.raises(ValueError, match="K_train"):
        model.train()
