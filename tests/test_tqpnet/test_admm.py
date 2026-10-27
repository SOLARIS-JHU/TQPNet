import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tqpnet.admm import admm_relu, admm_relu_step, build_operator, residual, unrolled_loss

from .problems import cvxpy_solve, q_fn_c_fn, random_qp


def _instance(seed=0, n=10, m=20, n_p=4):
    prob = random_qp(n, m, n_p, seed=seed)
    q_fn, c_fn = q_fn_c_fn(prob)
    p = jnp.array(np.random.default_rng(seed + 100).uniform(-1, 1, n_p))
    return prob, q_fn(p), c_fn(p)


def test_operator_matrices_match_formulas():
    prob, q, c = _instance()
    Q, A = jnp.array(prob["Q"]), jnp.array(prob["A"])
    rho = 1.0
    op = build_operator(Q, A, rho)
    D_U = jnp.sum(A**2, axis=1)
    D_V = jnp.count_nonzero(A, axis=0).astype(Q.dtype)
    assert jnp.allclose(op.H_V @ (Q + rho * jnp.diag(D_V)), jnp.eye(10), atol=1e-10)
    assert jnp.allclose(op.M1, rho * op.H_V @ jnp.diag(D_V), atol=1e-12)
    assert jnp.allclose(op.M2, rho * op.H_V @ A.T, atol=1e-12)
    assert jnp.allclose(op.inv_DU, 1.0 / D_U, atol=1e-12)
    assert jnp.allclose(op.M_beta, -jnp.diag(1.0 / D_U) @ A, atol=1e-12)
    assert (op.n, op.m) == (10, 20)


def test_step_matches_paper_W2_b2():
    # Theorem "Low-dimensional ADMM-ReLU Layer": [x+; β+] = ReLU_β(W2 [x; β] + b2)
    prob, q, c = _instance(seed=1)
    Q, A = jnp.array(prob["Q"]), jnp.array(prob["A"])
    rho = 1.0
    op = build_operator(Q, A, rho)
    n, m = 10, 20
    D_V = jnp.diag(jnp.count_nonzero(A, axis=0).astype(Q.dtype))
    D_U_inv = jnp.diag(1.0 / jnp.sum(A**2, axis=1))
    H_V = op.H_V
    I = jnp.eye(n)
    W2 = jnp.block([
        [rho * H_V @ D_V, rho * H_V @ A.T],
        [-D_U_inv @ A @ (2 * rho * H_V @ D_V - I), jnp.eye(m) - 2 * rho * D_U_inv @ A @ H_V @ A.T],
    ])
    b2 = jnp.concatenate([-H_V @ q, 2 * D_U_inv @ A @ H_V @ q + D_U_inv @ c])
    x = jnp.array(np.random.default_rng(2).normal(size=n))
    beta = jnp.abs(jnp.array(np.random.default_rng(3).normal(size=m)))
    z = W2 @ jnp.concatenate([x, beta]) + b2
    x_ref, beta_ref = z[:n], jnp.maximum(z[n:], 0.0)
    x_new, beta_new = admm_relu_step(op, x, beta, q, c)
    assert jnp.allclose(x_new, x_ref, atol=1e-10)
    assert jnp.allclose(beta_new, beta_ref, atol=1e-10)


def test_residual_zero_at_optimum_positive_elsewhere():
    prob, q, c = _instance(seed=2)
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    x_star, beta_star = cvxpy_solve(prob["Q"], prob["A"], np.asarray(q), np.asarray(c))
    assert float(residual(op, jnp.array(x_star), jnp.array(beta_star), q, c)) < 1e-12
    assert float(residual(op, jnp.array(x_star) + 0.1, jnp.array(beta_star), q, c)) > 1e-4


@pytest.mark.parametrize("seed", [3, 5, 8])
def test_k_steps_converge_to_cvxpy(seed):
    prob, q, c = _instance(seed=seed)
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    x, beta = admm_relu(op, jnp.zeros(10), jnp.zeros(20), q, c, K=3000)
    x_star, _ = cvxpy_solve(prob["Q"], prob["A"], np.asarray(q), np.asarray(c))
    assert float(residual(op, x, beta, q, c)) < 1e-10
    assert float(jnp.max(jnp.abs(x - jnp.array(x_star)))) < 1e-6


def test_k_zero_is_identity_and_loss_is_log1p():
    prob, q, c = _instance(seed=4)
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    x0 = jnp.ones(10)
    beta0 = jnp.linspace(-1, 1, 20)
    x, beta = admm_relu(op, x0, beta0, q, c, K=0)
    assert jnp.array_equal(x, x0) and jnp.array_equal(beta, beta0)
    xk, bk = admm_relu(op, x0, jnp.maximum(beta0, 0), q, c, K=5)
    assert jnp.allclose(unrolled_loss(op, x0, beta0, q, c, K=5), jnp.log1p(residual(op, xk, bk, q, c)))
    g = jax.grad(lambda x0_: unrolled_loss(op, x0_, beta0, q, c, K=5))(x0)
    assert bool(jnp.all(jnp.isfinite(g)))
