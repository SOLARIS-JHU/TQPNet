from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class TuQPNet(nn.Module):
    """Neural proxy mapping p -> concat([x̄, z, λ]).

    Implements the DNN parameterization of Eqn. (16) in the paper:
        (x̄*, z*, λ*) = N(p; θ, M, L)

    Output dimension: (n+m) + 2m + 2m = n + 5m
    where:
        x̄ ∈ R^{n+m}: primal variable of QPslack (x, ξ)
        z  ∈ R^{2m}:  auxiliary variable
        λ  ∈ R^{2m}:  dual variable

    No output ReLU — z and λ are unconstrained.
    """
    n: int
    m: int
    width: int = 256
    depth: int = 4

    @nn.compact
    def __call__(self, p: jnp.ndarray) -> jnp.ndarray:
        out_dim = (self.n + self.m) + 2 * self.m + 2 * self.m   # n + 5m
        x = p
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = nn.relu(x)
        x = nn.Dense(out_dim)(x)
        return x


def split_output(
    out: jnp.ndarray,
    n: int,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split batched network output into (x̄, z, λ).

    Args:
        out: shape (batch, n+5m)
        n:   number of primal variables
        m:   number of constraints

    Returns:
        xbar: (batch, n+m)
        z:    (batch, 2m)
        lam:  (batch, 2m)
    """
    nm = n + m
    xbar = out[:, :nm]
    z    = out[:, nm:nm + 2 * m]
    lam  = out[:, nm + 2 * m:]
    return xbar, z, lam


def split_output_single(
    out: jnp.ndarray,
    n: int,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split single (unbatched) network output into (x̄, z, λ).

    Args:
        out: shape (n+5m,)
        n:   number of primal variables
        m:   number of constraints

    Returns:
        xbar: (n+m,)
        z:    (2m,)
        lam:  (2m,)
    """
    nm = n + m
    xbar = out[:nm]
    z    = out[nm:nm + 2 * m]
    lam  = out[nm + 2 * m:]
    return xbar, z, lam


def build_network(
    n: int,
    m: int,
    n_p: int,
    width: int,
    depth: int,
    key: jax.Array,
) -> Tuple[TuQPNet, dict]:
    """Instantiate TuQPNet and initialise parameters.

    Args:
        n:     number of primal variables
        m:     number of constraints
        n_p:   parameter dimension
        width: hidden layer width (M in the paper)
        depth: number of hidden layers (L in the paper)
        key:   JAX PRNG key

    Returns:
        (net, params) tuple
    """
    net = TuQPNet(n=n, m=m, width=width, depth=depth)
    dummy = jnp.ones((1, n_p))
    params = net.init(key, dummy)
    return net, params