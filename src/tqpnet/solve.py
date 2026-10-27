"""Online inference with triggered refinement (Algorithm 2, online phase).

For a batch of parameter vectors the warm-start network is evaluated once,
``K_infer`` ADMM(ReLU) layers are applied to every instance in parallel, and
the optimality-condition residual serves as a certificate: instances whose
residual exceeds ``tol_on`` keep iterating, one ADMM(ReLU) layer at a time,
until they are certified or ``K_max`` extra layers have been spent.
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from tqpnet.admm import ADMMOperator, admm_relu, admm_relu_step, residual
from tqpnet.network import WarmStartNet


@dataclass
class SolveResult:
    """Result of :func:`solve` for a batch of parameter vectors.

    Attributes:
        x:             primal solutions, shape (batch, n).
        beta:          dual solutions β ≥ 0, shape (batch, m).
        certificate:   the paper's optimality-condition residual in the weighted
                       form ‖r₁‖²/n + ‖r₂‖²/m, shape (batch,).
        certified:     ``certificate <= tol_on``, shape (batch,), bool. This is the
                       paper's online test; it bounds each constraint violation only
                       by about sqrt(m · tol_on), so check ``max_violation`` when
                       feasibility matters.
        max_violation: max_i max(c_i − (A x)_i, 0) per instance, shape (batch,).
        steps:         total ADMM(ReLU) layers applied per instance, including the
                       first ``K_infer``, shape (batch,), int32.
        refined:       ``steps > K_infer``, i.e. whether triggered refinement ran for
                       the instance, shape (batch,), bool.
    """
    x: jax.Array
    beta: jax.Array
    certificate: jax.Array
    certified: jax.Array
    max_violation: jax.Array
    steps: jax.Array
    refined: jax.Array


@partial(jax.jit, static_argnames=("net", "K_infer", "K_max"))
def _solve_kernel(net: WarmStartNet, variables, op: ADMMOperator, P, qs, cs, *,
                  K_infer: int, K_max: int, tol_on):
    x0, beta0 = net.apply(variables, P)
    beta0 = jnp.maximum(beta0, 0.0)

    run_k = jax.vmap(lambda x, b, q, c: admm_relu(op, x, b, q, c, K_infer))
    one_step = jax.vmap(lambda x, b, q, c: admm_relu_step(op, x, b, q, c))
    res = jax.vmap(lambda x, b, q, c: residual(op, x, b, q, c))

    x, beta = run_k(x0, beta0, qs, cs)
    cert = res(x, beta, qs, cs)
    steps = jnp.full(P.shape[0], K_infer, dtype=jnp.int32)

    def cond(state):
        _, _, cert, _, it = state
        return (it < K_max) & jnp.any(cert > tol_on)

    def body(state):
        x, beta, cert, steps, it = state
        active = cert > tol_on
        x_new, beta_new = one_step(x, beta, qs, cs)
        x = jnp.where(active[:, None], x_new, x)
        beta = jnp.where(active[:, None], beta_new, beta)
        cert = res(x, beta, qs, cs)
        steps = steps + active.astype(jnp.int32)
        return x, beta, cert, steps, it + 1

    x, beta, cert, steps, _ = jax.lax.while_loop(cond, body, (x, beta, cert, steps, 0))
    violation = jax.vmap(lambda x, c: jnp.max(jnp.maximum(c - op.A @ x, 0.0)))(x, cs)
    return x, beta, cert, steps, violation


def solve(net: WarmStartNet, variables, op: ADMMOperator, P: jax.Array,
          q_fn: Callable, c_fn: Callable, *, K_infer: int, tol_on: float, K_max: int) -> SolveResult:
    """Certified batch solve; see the module docstring and :class:`SolveResult`.

    Args:
        net, variables: trained warm-start network.
        op:             ADMM(ReLU) operator of the problem family.
        P:              parameter batch, shape (batch, n_p).
        q_fn, c_fn:     single-instance callables p ↦ q(p), p ↦ c(p).
        K_infer:        ADMM(ReLU) layers applied to every instance (static int).
        tol_on:         certificate tolerance ε_on.
        K_max:          maximum extra layers for triggered refinement (static int).
    """
    qs = jax.vmap(q_fn)(P)
    cs = jax.vmap(c_fn)(P)
    x, beta, cert, steps, violation = _solve_kernel(
        net, variables, op, P, qs, cs, K_infer=int(K_infer), K_max=int(K_max), tol_on=tol_on)
    return SolveResult(
        x=x,
        beta=beta,
        certificate=cert,
        certified=cert <= tol_on,
        max_violation=violation,
        steps=steps,
        refined=steps > K_infer,
    )
