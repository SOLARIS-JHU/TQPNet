"""MLP(ReLU) warm-start network N(p; θ, M, L) of the paper (Eq. "DNN proxy").

The network maps a parameter vector p to an initial primal-dual pair
(x₀, β₀) ∈ R^n × R^m that the unrolled ADMM(ReLU) layers refine. The dual head
is passed through softplus so β₀ ≥ 0, as in ``experiments/base.py::TQPNet``.
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class WarmStartNet(nn.Module):
    """p ↦ (x₀, β₀) with ``depth`` hidden ReLU layers of width ``width``."""
    n: int
    m: int
    width: int = 256
    depth: int = 4

    @nn.compact
    def __call__(self, p: jax.Array) -> Tuple[jax.Array, jax.Array]:
        dense = dict(kernel_init=nn.initializers.xavier_uniform(),
                     bias_init=nn.initializers.zeros)
        h = p
        for _ in range(self.depth):
            h = nn.relu(nn.Dense(self.width, **dense)(h))
        x0 = nn.Dense(self.n, **dense)(h)
        beta0 = nn.softplus(nn.Dense(self.m, **dense)(h))
        return x0, beta0


def build_network(n: int, m: int, n_p: int, width: int, depth: int, key: jax.Array):
    """Instantiate :class:`WarmStartNet` and initialise its variables.

    Returns:
        ``(net, variables)`` where ``variables`` is the flax variable dict.
    """
    net = WarmStartNet(n=n, m=m, width=width, depth=depth)
    variables = net.init(key, jnp.ones((1, n_p)))
    return net, variables
