"""Offline training of the warm-start network (Algorithm 2, offline phase).

The loss is the paper's self-supervised objective

    L(θ) = mean_p log(1 + ‖r(ADMM-ReLU_K(N(p; θ)); p)‖²),

evaluated on a fixed training set D drawn once from the user's sampler.

The optimisation loop is aligned with the scripts that produced the paper's
numbers (``experiments/ablation/training_ablation.py``,
``experiments/generalizability/generalizability.py``): AdamW, global
gradient-norm clipping (default 10), a step schedule that multiplies the
learning rate by ``lr_gamma`` every ``lr_step_epochs`` epochs (evaluated per
epoch), full mini-batches only (the last incomplete batch of an epoch is
dropped), and mini-batches whose loss is not finite are skipped without an
update and recorded as NaN. Early stopping compares the epoch mean of the
mini-batch losses with ``tol_off``.
"""
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from tqpnet.admm import ADMMOperator, unrolled_loss
from tqpnet.network import WarmStartNet
from tqpnet.utils.log import TableLogger


def _per_instance_losses(variables, net: WarmStartNet, op: ADMMOperator, P, q_fn, c_fn, K: int):
    x0, beta0 = net.apply(variables, P)
    qs = jax.vmap(q_fn)(P)
    cs = jax.vmap(c_fn)(P)
    return jax.vmap(lambda x, b, q, c: unrolled_loss(op, x, b, q, c, K))(x0, beta0, qs, cs)


def batch_loss(variables, net: WarmStartNet, op: ADMMOperator, P, q_fn, c_fn, K: int) -> jax.Array:
    """Mean log(1 + residual) over the batch ``P`` after ``K`` unrolled ADMM(ReLU) layers."""
    return jnp.mean(_per_instance_losses(variables, net, op, P, q_fn, c_fn, K))


def _loss_with_residual(variables, net, op, P, q_fn, c_fn, K):
    losses = _per_instance_losses(variables, net, op, P, q_fn, c_fn, K)
    return jnp.mean(losses), jnp.mean(jnp.expm1(losses))   # (loss, mean raw residual)


def train(
    net: WarmStartNet,
    variables,
    op: ADMMOperator,
    dataset: jax.Array,
    q_fn: Callable,
    c_fn: Callable,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    K_train: int,
    tol_off: float,
    log_every: int,
    key: jax.Array,
    grad_clip: float = 10.0,
    lr_step_epochs: int = 300,
    lr_gamma: float = 0.5,
) -> Tuple[dict, Dict]:
    """Train ``variables`` on ``dataset`` (shape ``(N, n_p)``); see the module docstring.

    Returns:
        ``(variables, history)`` with ``history = {"loss", "residual", "lr", "epochs", "updates"}``:
        per-epoch mean loss, mean raw residual and learning rate, the number of
        epochs run and the number of optimizer updates applied.
    """
    N = dataset.shape[0]
    if N < batch_size:
        raise ValueError(f"n_train={N} is smaller than batch_size={batch_size}; "
                         "training needs at least one full mini-batch")
    n_batches = N // batch_size

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(variables)
    logger = TableLogger(log_every=log_every)

    @jax.jit
    def step(variables, opt_state, lr_now, P):
        (loss, mean_res), grads = jax.value_and_grad(_loss_with_residual, has_aux=True)(
            variables, net, op, P, q_fn, c_fn, K_train)
        clip_state, inj_state = opt_state
        lr_arr = jnp.asarray(lr_now, dtype=inj_state.hyperparams["learning_rate"].dtype)
        inj_state = inj_state._replace(hyperparams={**inj_state.hyperparams, "learning_rate": lr_arr})
        opt_state = (clip_state, inj_state)

        def apply(_):
            updates, new_opt_state = optimizer.update(grads, opt_state, variables)
            return optax.apply_updates(variables, updates), new_opt_state

        finite = jnp.isfinite(loss)
        variables, opt_state = jax.lax.cond(finite, apply, lambda _: (variables, opt_state), None)
        return variables, opt_state, loss, mean_res, finite

    history: Dict = {"loss": [], "residual": [], "lr": [], "epochs": 0, "updates": 0}
    for epoch in range(1, epochs + 1):
        lr_now = lr * lr_gamma ** ((epoch - 1) // lr_step_epochs)
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        losses, residuals = [], []
        for b in range(n_batches):
            P = dataset[perm[b * batch_size:(b + 1) * batch_size]]
            variables, opt_state, loss, mean_res, finite = step(variables, opt_state, lr_now, P)
            losses.append(float(loss))
            residuals.append(float(mean_res))
            history["updates"] += int(finite)

        epoch_loss = float(jnp.nanmean(jnp.asarray(losses)))
        epoch_res = float(jnp.nanmean(jnp.asarray(residuals)))
        history["loss"].append(epoch_loss)
        history["residual"].append(epoch_res)
        history["lr"].append(lr_now)
        history["epochs"] = epoch
        logger.log(epoch, epochs, epoch_loss, epoch_res)

        if epoch_loss <= tol_off:
            print(f"  Early stop at epoch {epoch}: loss={epoch_loss:.2e} <= tol_off={tol_off:.2e}")
            break

    return variables, history
