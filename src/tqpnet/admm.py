"""Low-dimensional ADMM(ReLU) for the parametric QP

    min_x  ½ xᵀ Q x + q(p)ᵀ x   s.t.  A x ≥ c(p),

with Q ∈ R^{n×n} (symmetric PSD) and A ∈ R^{m×n} fixed and only q(p), c(p)
parameter dependent (Eq. (1) of the paper).

One ADMM(ReLU) layer (paper, Theorem "Low-dimensional ADMM-ReLU Layer") maps
(x, β) ↦ (x⁺, β⁺) with H_V = (Q + ρ D_V)⁻¹:

    x⁺ = ρ H_V D_V x + ρ H_V Aᵀ β − H_V q
    β⁺ = max(0, β − D_U⁻¹ A (2 x⁺ − x) + D_U⁻¹ c)

where (D_U)_ii = ‖A_{i,:}‖² and (D_V)_jj = nnz(A_{:,j}). All matrices are
precomputed once per problem family by :func:`build_operator`, so a layer is
two matrix-vector products plus a ReLU.

The optimality-condition residual (paper, Eq. "r(x, β; p)") is

    r₁ = Q x − Aᵀ β + q,        r₂ = β − max(0, β − A x + c),

and this module reports it in the weighted form ‖r₁‖²/n + ‖r₂‖²/m, exactly as
the paper's experiment scripts do (``experiments/base.py::kkt_raw_residual``).

Every function here acts on a single instance; batch with ``jax.vmap``.
"""
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


class ADMMOperator(NamedTuple):
    """Precomputed ADMM(ReLU) matrices for one problem family (a JAX pytree of arrays)."""
    Q: jax.Array        # (n, n)
    A: jax.Array        # (m, n)
    rho: jax.Array      # 0-d
    H_V: jax.Array      # (n, n)   (Q + ρ D_V)⁻¹
    M1: jax.Array       # (n, n)   ρ H_V D_V
    M2: jax.Array       # (n, m)   ρ H_V Aᵀ
    M_beta: jax.Array   # (m, n)   −D_U⁻¹ A
    inv_DU: jax.Array   # (m,)     1 / D_U

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[0]


def build_operator(Q, A, rho: float = 1.0) -> ADMMOperator:
    """Precompute the ADMM(ReLU) matrices for fixed (Q, A) and penalty ρ.

    Derived arrays take the dtype of ``Q``. No validation is performed here;
    :class:`tqpnet.Model` checks the structural requirements on Q and A.
    """
    Q = jnp.asarray(Q)
    A = jnp.asarray(A, dtype=Q.dtype)
    n = Q.shape[0]
    rho_arr = jnp.asarray(rho, dtype=Q.dtype)

    D_U = jnp.sum(A ** 2, axis=1)                                   # (m,)
    D_V = jnp.count_nonzero(A, axis=0).astype(Q.dtype)              # (n,)
    Q_sym = 0.5 * (Q + Q.T)
    H_V = jnp.linalg.solve(Q_sym + rho_arr * jnp.diag(D_V), jnp.eye(n, dtype=Q.dtype))
    inv_DU = 1.0 / jnp.maximum(D_U, 1e-12)

    return ADMMOperator(
        Q=Q,
        A=A,
        rho=rho_arr,
        H_V=H_V,
        M1=rho_arr * H_V * D_V[None, :],          # ρ H_V diag(D_V)
        M2=rho_arr * H_V @ A.T,
        M_beta=-inv_DU[:, None] * A,              # −diag(D_U⁻¹) A
        inv_DU=inv_DU,
    )


def admm_relu_step(op: ADMMOperator, x, beta, q, c) -> Tuple[jax.Array, jax.Array]:
    """One ADMM(ReLU) layer on a single instance."""
    x_new = op.M1 @ x + op.M2 @ beta - op.H_V @ q
    beta_new = jnp.maximum(beta + op.M_beta @ (2.0 * x_new - x) + op.inv_DU * c, 0.0)
    return x_new, beta_new


def admm_relu(op: ADMMOperator, x0, beta0, q, c, K: int) -> Tuple[jax.Array, jax.Array]:
    """Apply ``K`` ADMM(ReLU) layers from (x0, beta0). ``K`` must be a Python int.

    ``beta0`` is used as given (callers clamp it to β ≥ 0 when needed).
    """
    def body(_, carry):
        x, beta = carry
        return admm_relu_step(op, x, beta, q, c)

    return jax.lax.fori_loop(0, K, body, (x0, beta0))


def residual(op: ADMMOperator, x, beta, q, c) -> jax.Array:
    """Weighted optimality-condition residual ‖r₁‖²/n + ‖r₂‖²/m for one instance."""
    r1 = op.Q @ x - op.A.T @ beta + q
    r2 = beta - jnp.maximum(beta - op.A @ x + c, 0.0)
    return jnp.sum(r1 ** 2) / op.n + jnp.sum(r2 ** 2) / op.m


def unrolled_loss(op: ADMMOperator, x0, beta0, q, c, K: int) -> jax.Array:
    """Self-supervised loss log(1 + residual) after ``K`` unrolled ADMM(ReLU) layers."""
    x, beta = admm_relu(op, x0, jnp.maximum(beta0, 0.0), q, c, K)
    return jnp.log1p(residual(op, x, beta, q, c))
