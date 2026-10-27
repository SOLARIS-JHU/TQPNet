"""Pin the JAX package to the PyTorch implementation used by the paper's experiments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.base import ProblemConfig, _chol, admm_refine, admm_unrolled_loss, kkt_raw_residual  # noqa: E402
from tqpnet.admm import admm_relu, build_operator, residual, unrolled_loss  # noqa: E402

from .problems import q_fn_c_fn, random_qp  # noqa: E402


def _torch_cfg(prob, rho=1.0):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    n, m = prob["Q"].shape[0], prob["A"].shape[0]
    cfg = ProblemConfig(n=n, m=m, np_dim=prob["Cq"].shape[1], rho_refine=rho)
    cfg.Q0, cfg.A0 = t(prob["Q"]), t(prob["A"])
    cfg._D_U = (cfg.A0 ** 2).sum(dim=1)
    cfg._D_V = (cfg.A0.abs() > 0).to(cfg.A0.dtype).sum(dim=0)
    cfg._inv_D_U = 1.0 / cfg._D_U.clamp(min=1e-12)
    H_V = cfg.Q0 + rho * torch.diag(cfg._D_V)
    cfg._H_V_chol = _chol(H_V)
    H_inv = torch.linalg.solve(H_V, torch.eye(n, dtype=torch.float64))
    cfg._M1 = rho * H_inv @ torch.diag(cfg._D_V)
    cfg._M2 = rho * H_inv @ cfg.A0.T
    cfg._H_inv = H_inv
    cfg._M_beta = -torch.diag(cfg._inv_D_U) @ cfg.A0
    return cfg


def test_iterates_residual_and_loss_match_experiments_base():
    prob = random_qp(10, 20, 4, seed=11)
    q_fn, c_fn = q_fn_c_fn(prob)
    rng = np.random.default_rng(12)
    P = rng.uniform(-1, 1, (3, 4))
    X0 = rng.normal(size=(3, 10))
    B0 = rng.normal(size=(3, 20))

    # JAX
    op = build_operator(jnp.array(prob["Q"]), jnp.array(prob["A"]))
    qs, cs = jax.vmap(q_fn)(jnp.array(P)), jax.vmap(c_fn)(jnp.array(P))
    xj, bj = jax.vmap(lambda x0, b0, q, c: admm_relu(op, x0, jnp.maximum(b0, 0), q, c, 20))(
        jnp.array(X0), jnp.array(B0), qs, cs)
    rj = jax.vmap(lambda x, b, q, c: residual(op, x, b, q, c))(xj, bj, qs, cs)
    lj = jnp.mean(jax.vmap(lambda x0, b0, q, c: unrolled_loss(op, x0, b0, q, c, 20))(
        jnp.array(X0), jnp.array(B0), qs, cs))

    # Torch
    cfg = _torch_cfg(prob)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    Q_b, A_b = cfg.Q0.expand(3, -1, -1), cfg.A0.expand(3, -1, -1)
    xt, bt, *_ = admm_refine(t(X0), Q_b, t(qs), A_b, t(cs), cfg, K=20, beta0=t(B0))
    rt = kkt_raw_residual(xt, bt, Q_b, t(qs), A_b, t(cs), cfg)
    lt = admm_unrolled_loss(t(X0), t(B0), Q_b, t(qs), A_b, t(cs), cfg, K_unroll=20)

    assert np.allclose(np.asarray(xj), xt.numpy(), atol=1e-8)
    assert np.allclose(np.asarray(bj), bt.numpy(), atol=1e-8)
    assert np.allclose(np.asarray(rj), rt.numpy(), atol=1e-10)
    assert abs(float(lj) - float(lt)) < 1e-10
