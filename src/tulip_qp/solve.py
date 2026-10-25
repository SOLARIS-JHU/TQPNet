from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import optax

from tulip_qp.loss import admm_loss
from tulip_qp.network import TuQPNet, split_output


@dataclass
class SolveResult:
    """Result of online inference for a batch of parameter vectors.

    Attributes:
        x:           primal solutions x* (original variables, without slack ξ),
                     shape (batch, n).
        feasible:    per-sample feasibility flag (ε <= tol_on), shape (batch,)
        certificate: per-sample ADMM residual ε = ||r||², shape (batch,)
        used_backup: per-sample flag — True if backup correction was triggered
    """
    x:           jax.Array
    feasible:    jax.Array
    certificate: jax.Array
    used_backup: jax.Array


def _per_sample_loss(
    xbar: jnp.ndarray,
    z:    jnp.ndarray,
    lam:  jnp.ndarray,
    Q:    jnp.ndarray,
    q:    jnp.ndarray,
    A:    jnp.ndarray,
    c:    jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """admm_loss total for a single sample (unbatched)."""
    return admm_loss(xbar, z, lam, Q, q, A, c, alpha)[0]


def _backup_correction(
    xbar: jnp.ndarray,
    z:    jnp.ndarray,
    lam:  jnp.ndarray,
    Q:    jnp.ndarray,
    q:    jnp.ndarray,
    A:    jnp.ndarray,
    c:    jnp.ndarray,
    alpha: float,
    backup_steps: int,
    backup_lr: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Adam gradient descent on admm_loss, warm-started from neural prediction.

    Implements the backup correction in Algorithm 2 (online inference, step 5):
    runs ADMM-ReLU iterations for samples where ε > tol_on.

    Args:
        xbar, z, lam: initial point (neural prediction), all unbatched
        Q, q, A, c:   QP data for this sample
        alpha:        QPslack penalization scalar
        backup_steps: max number of Adam steps
        backup_lr:    Adam learning rate

    Returns:
        corrected (xbar, z, lam)
    """
    optimizer = optax.adam(backup_lr)
    variables = {"xbar": xbar, "z": z, "lam": lam}
    opt_state = optimizer.init(variables)

    def loss_fn(variables):
        return _per_sample_loss(
            variables["xbar"], variables["z"], variables["lam"],
            Q, q, A, c, alpha,
        )

    def step(carry, _):
        variables, opt_state = carry
        _, grads = jax.value_and_grad(loss_fn)(variables)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        return (variables, opt_state), None

    (variables, _), _ = jax.lax.scan(
        step, (variables, opt_state), None, length=backup_steps
    )
    return variables["xbar"], variables["z"], variables["lam"]


def solve(
    net: TuQPNet,
    params: dict,
    P: jax.Array,
    Q_fn: Callable,
    q_fn: Callable,
    A_fn: Callable,
    c_fn: Callable,
    alpha: float,
    tol_on: float,
    backup_steps: int,
    backup_lr: float,
) -> SolveResult:
    """Online inference for a batch of parameter vectors (Algorithm 2, online phase).

    Args:
        net:          trained TuQPNet
        params:       trained Flax parameter dict
        P:            normalized parameter batch, shape (batch, n_p)
        Q_fn:         callable p -> (n, n)
        q_fn:         callable p -> (n,)
        A_fn:         callable p -> (m, n)
        c_fn:         callable p -> (m,)
        alpha:        QPslack penalization scalar (must match training alpha)
        tol_on:       certificate tolerance ε_on — trigger backup if ε > tol_on
        backup_steps: max Adam steps for backup (0 = disabled)
        backup_lr:    backup Adam learning rate

    Returns:
        SolveResult
    """
    # --- Step 1: Batch forward pass (Algorithm 2, step 1) ---
    out = net.apply(params, P)   # (batch, n+5m)
    xbar, z, lam = split_output(out, net.n, net.m)

    # --- Step 2 & 3: Certificate + Backup (Algorithm 2, steps 2-5) ---
    if backup_steps > 0:
        def single_eps(p, xb, z_, l_):
            return admm_loss(xb, z_, l_, Q_fn(p), q_fn(p), A_fn(p), c_fn(p), alpha)[0]

        eps = jax.vmap(single_eps)(P, xbar, z, lam)   # (batch,)
        needs_backup = eps > tol_on

        def maybe_backup(p, xb, z_, l_, needs):
            return jax.lax.cond(
                needs,
                lambda: _backup_correction(
                    xb, z_, l_, Q_fn(p), q_fn(p), A_fn(p), c_fn(p),
                    alpha, backup_steps, backup_lr,
                ),
                lambda: (xb, z_, l_),
            )

        xbar, z, lam = jax.vmap(maybe_backup)(P, xbar, z, lam, needs_backup)
        eps = jax.vmap(single_eps)(P, xbar, z, lam)
    else:
        eps          = jnp.zeros(P.shape[0])
        needs_backup = jnp.zeros(P.shape[0], dtype=bool)

    # --- Step 4: Feasibility via ADMM residual certificate ---
    feasible = eps <= tol_on

    # --- Step 5: Extract primal solution x (strip slack ξ) ---
    x_star = xbar[:, :net.n]

    return SolveResult(
        x=x_star,
        feasible=feasible,
        certificate=eps,
        used_backup=needs_backup,
    )