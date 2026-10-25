from typing import Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import optax

from tulip_qp.loss import admm_loss
from tulip_qp.network import TuQPNet, split_output_single
from tulip_qp.utils.log import TableLogger


def _batch_loss(
    params: dict,
    net: TuQPNet,
    P: jnp.ndarray,
    Q_fn: Callable,
    q_fn: Callable,
    A_fn: Callable,
    c_fn: Callable,
    alpha: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute mean ADMM residual loss over a batch.

    Implements the mini-batch loss from Algorithm 2 (offline training):
        L_B(θ) = (1/|B|) Σ_{p∈B} ||r(N(p;θ); p)||²

    P is assumed to be already normalized by the caller (Model.train).

    Returns:
        total:     scalar mean loss
        breakdown: dict of mean per-term losses {"stat", "comp", "prim"}
    """
    def single(p):
        out = net.apply(params, p[None, :])[0]
        xbar, z, lam = split_output_single(out, net.n, net.m)
        return admm_loss(xbar, z, lam, Q_fn(p), q_fn(p), A_fn(p), c_fn(p), alpha)

    totals, breakdowns = jax.vmap(single)(P)
    mean_total = jnp.mean(totals)
    mean_bd = {k: jnp.mean(v) for k, v in breakdowns.items()}
    return mean_total, mean_bd


def _compute_grad_norm(grads: dict) -> jnp.ndarray:
    """Global L2 norm of all gradients."""
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))


def train(
    net: TuQPNet,
    params: dict,
    sampler_fn: Callable,
    Q_fn: Callable,
    q_fn: Callable,
    A_fn: Callable,
    c_fn: Callable,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha: float,
    tol_off: float,
    log_every: int,
    key: jax.Array,
    normalize_fn: Callable,
    log_dir: Optional[str] = None,
) -> dict:
    """Train TuQPNet via JAX gradient descent (Algorithm 2, offline phase).

    Args:
        net:          TuQPNet instance
        params:       initial Flax parameter dict
        sampler_fn:   callable(key, n) -> jnp.ndarray of shape (n, n_p)
        Q_fn:         callable p -> jnp.ndarray (n, n), positive semidefinite
        q_fn:         callable p -> jnp.ndarray (n,)
        A_fn:         callable p -> jnp.ndarray (m, n)
        c_fn:         callable p -> jnp.ndarray (m,)
        epochs:       number of training epochs N_epoch
        batch_size:   mini-batch size B
        lr:           Adam learning rate η
        alpha:        QPslack slack penalization scalar α
        tol_off:      early-stopping tolerance ε_off
        log_every:    print loss every this many epochs
        key:          JAX PRNG key
        normalize_fn: callable P -> P_normalized, injected by Model
        log_dir:      TensorBoard log directory (None = no logging)

    Returns:
        trained Flax parameter dict θ
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    logger = TableLogger(log_every=log_every)

    # --- TensorBoard writer (optional) ---
    writer = None
    if log_dir is not None:
        try:
            import tensorflow as tf
            writer = tf.summary.create_file_writer(log_dir)
            with writer.as_default():
                tf.summary.text(
                    "hparams",
                    f"epochs={epochs} batch_size={batch_size} "
                    f"lr={lr} alpha={alpha} tol_off={tol_off}",
                    step=0,
                )
        except ImportError:
            print("  [warn] tensorflow not installed — TensorBoard logging disabled.")
            writer = None

    @jax.jit
    def step(params, opt_state, P):
        (loss, breakdown), grads = jax.value_and_grad(
            _batch_loss, has_aux=True
        )(params, net, P, Q_fn, q_fn, A_fn, c_fn, alpha)
        grad_norm = _compute_grad_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, breakdown, grad_norm

    for epoch in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        P = sampler_fn(subkey, batch_size)
        P = normalize_fn(P)

        params, opt_state, loss, breakdown, grad_norm = step(params, opt_state, P)

        loss_val      = float(loss)
        grad_norm_val = float(grad_norm)
        stat_val      = float(breakdown["stat"])
        comp_val      = float(breakdown["comp"])
        prim_val      = float(breakdown["prim"])

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("loss/train",      loss_val,      step=epoch)
                tf.summary.scalar("loss/stat",       stat_val,      step=epoch)
                tf.summary.scalar("loss/comp",       comp_val,      step=epoch)
                tf.summary.scalar("loss/prim",       prim_val,      step=epoch)
                tf.summary.scalar("train/grad_norm", grad_norm_val, step=epoch)

        logger.log(epoch, epochs, loss_val,
                   breakdown={"stat": stat_val, "comp": comp_val, "prim": prim_val},
                   grad_norm=grad_norm_val)

        if loss_val <= tol_off:
            print(f"  Early stop at epoch {epoch}: loss={loss_val:.2e} <= tol_off={tol_off:.2e}")
            break

    if writer is not None:
        writer.flush()

    return params