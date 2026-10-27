from typing import Callable

import jax
import jax.numpy as jnp


def box_sampler(lo, hi) -> Callable[[jax.Array, int], jax.Array]:
    """Uniform sampler over the box [lo, hi] ⊂ R^{n_p}.

    The paper's parameter domain P is a box in every experiment, and the
    experiment scripts sample it uniformly; this helper does the same.

    Args:
        lo, hi: array-likes of shape (n_p,), lower and upper bounds.

    Returns:
        ``sampler(key, n_samples) -> jnp.ndarray`` of shape ``(n_samples, n_p)``.
    """
    lo = jnp.asarray(lo, dtype=float)
    hi = jnp.asarray(hi, dtype=float)
    if lo.shape != hi.shape or lo.ndim != 1:
        raise ValueError(f"lo and hi must be 1-D with the same shape, got {lo.shape} and {hi.shape}")
    n_p = lo.shape[0]

    def sampler(key: jax.Array, n_samples: int) -> jax.Array:
        return jax.random.uniform(key, (n_samples, n_p), minval=lo, maxval=hi)

    return sampler
