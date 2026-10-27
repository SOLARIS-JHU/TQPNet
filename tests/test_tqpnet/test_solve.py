import jax
import jax.numpy as jnp

from tqpnet.admm import admm_relu, build_operator, residual
from tqpnet.network import build_network
from tqpnet.solve import SolveResult, solve

from .problems import q_fn_c_fn, random_qp


def _setup(n=4, m=8, n_p=2, batch=6):
    prob = random_qp(n, m, n_p, seed=7)
    q_fn, c_fn = q_fn_c_fn(prob)
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    net, var = build_network(n, m, n_p, 16, 2, jax.random.PRNGKey(0))
    P = jax.random.uniform(jax.random.PRNGKey(1), (batch, n_p), minval=-1.0, maxval=1.0)
    return op, net, var, P, q_fn, c_fn


def test_shapes_and_certificate_consistency():
    op, net, var, P, q_fn, c_fn = _setup()
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=10, tol_on=1e-4, K_max=0)
    assert isinstance(r, SolveResult)
    assert r.x.shape == (6, 4) and r.beta.shape == (6, 8) and r.certificate.shape == (6,)
    assert r.certified.dtype == jnp.bool_ and r.refined.dtype == jnp.bool_
    recomputed = jax.vmap(lambda x, b, p: residual(op, x, b, q_fn(p), c_fn(p)))(r.x, r.beta, P)
    assert jnp.allclose(r.certificate, recomputed)
    assert bool(jnp.all(r.certified == (r.certificate <= 1e-4)))
    assert bool(jnp.all(r.steps == 10)) and not bool(jnp.any(r.refined))


def test_no_refinement_when_already_certified():
    op, net, var, P, q_fn, c_fn = _setup()
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=5, tol_on=jnp.inf, K_max=100)
    assert bool(jnp.all(r.steps == 5)) and not bool(jnp.any(r.refined)) and bool(jnp.all(r.certified))


def test_refinement_reaches_tolerance_and_counts_steps():
    op, net, var, P, q_fn, c_fn = _setup()
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=5, tol_on=1e-8, K_max=5000)
    assert bool(jnp.all(r.certified))
    assert bool(jnp.all(r.steps >= 5)) and bool(jnp.any(r.refined))
    assert bool(jnp.all(r.max_violation < 1e-3))
    # untrained net: at least one instance must have needed extra steps
    assert int(jnp.max(r.steps)) > 5


def test_refined_instances_freeze_at_first_certified_layer():
    op, net, var, P, q_fn, c_fn = _setup()
    tol = 1e-8
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=5, tol_on=tol, K_max=5000)
    x0, beta0 = net.apply(var, P)
    beta0 = jnp.maximum(beta0, 0.0)
    for i in range(P.shape[0]):
        k = int(r.steps[i])
        q, c = q_fn(P[i]), c_fn(P[i])
        xk, bk = admm_relu(op, x0[i], beta0[i], q, c, k)
        assert jnp.allclose(r.x[i], xk, atol=1e-12) and jnp.allclose(r.beta[i], bk, atol=1e-12)
        assert float(residual(op, xk, bk, q, c)) <= tol
        if k > 5:   # the layer before was not yet certified
            xp, bp = admm_relu(op, x0[i], beta0[i], q, c, k - 1)
            assert float(residual(op, xp, bp, q, c)) > tol


def test_k_max_caps_steps():
    op, net, var, P, q_fn, c_fn = _setup()
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=5, tol_on=0.0, K_max=7)
    assert bool(jnp.all(r.steps == 12)) and bool(jnp.all(r.refined)) and not bool(jnp.any(r.certified))


def test_max_violation_matches_direct_computation():
    op, net, var, P, q_fn, c_fn = _setup()
    r = solve(net, var, op, P, q_fn, c_fn, K_infer=3, tol_on=1e-4, K_max=0)
    direct = jax.vmap(lambda x, p: jnp.max(jnp.maximum(c_fn(p) - op.A @ x, 0.0)))(r.x, P)
    assert jnp.allclose(r.max_violation, direct)
