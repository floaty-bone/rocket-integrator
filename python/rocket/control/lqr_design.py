"""
lqr_design.py — Off-line LQR gain synthesis.

This module is the **design-time** side of the controller (slow):
  * Linearises ``F(x, u)`` at an operating point using ``jax.jacfwd``.
  * Projects the resulting 13×13 ``A`` and 13×9 ``B`` matrices down to the
    12-D tangent space at the current quaternion.
  * Solves the LQR Riccati equation via :mod:`python-control` to obtain ``K``.

The runtime feedback law that *uses* the gain ``K`` lives in
:mod:`rocket.control.attitude_law`.
"""

from __future__ import annotations

import control as ctrl
import jax
import numpy as np

from rocket.math.tangent_space import E, project_to_tangent_space


def make_lqr_computer(F, Q, R):
    """Build a JIT-compiled ``compute_lqr(xl, ul) → (K12, A12, B12)`` callable.

    The Jacobians of ``F`` are precompiled with ``jax.jacfwd`` and cached, so
    re-linearisation at a new operating point is fast enough to run inside the
    SIL loop.
    """
    jac_x_fn = jax.jit(jax.jacfwd(F, argnums=0))
    jac_u_fn = jax.jit(jax.jacfwd(F, argnums=1))
    Q_np = np.array(Q)
    R_np = np.array(R)

    def compute_lqr(xl, ul):
        res_x = jac_x_fn(xl, ul).block_until_ready()
        res_u = jac_u_fn(xl, ul).block_until_ready()

        A13 = np.array(res_x)
        B13 = np.array(res_u)

        # Project A13, B13 to the 12-D tangent space at xl's quaternion.
        A12, B12 = project_to_tangent_space(A13, B13, xl[3:7])

        # Project Q13 to 12-D as well if the caller supplied a 13×13 weight.
        if Q_np.shape == (13, 13):
            E_mat = E(xl[3:7], scale=0.5)
            Q12 = E_mat.T @ Q_np @ E_mat
        else:
            Q12 = Q_np

        # Symmetrise to satisfy control's strict positive-definite check.
        Q12 = (Q12 + Q12.T) / 2.0

        K12, S12, E12 = ctrl.lqr(A12, B12, Q12, R_np)
        return K12, A12, B12

    return compute_lqr


def rscale(A, B, C, D, K):
    """Feed-forward gain ``N`` such that ``u = N·r − K·x`` tracks ``r`` at steady state.

    Uses the pseudoinverse rather than a plain inverse for robustness when
    ``M`` is rank-deficient (e.g. fewer measured outputs than inputs).
    """
    Acl = A - B @ K
    Ccl = C - D @ K
    M = -Ccl @ np.linalg.inv(Acl) @ B + D
    N = np.linalg.pinv(M)
    return N


def design_lqr(A, B, C, D, Q, R):
    """One-shot LQR design from explicit ``(A, B, C, D, Q, R)`` matrices.

    Returns ``(K, N, S, E)``: feedback gain, feed-forward gain, Riccati
    solution, and closed-loop eigenvalues.
    """
    K, S, E = ctrl.lqr(A, B, Q, R)
    N = rscale(A, B, C, D, K)
    return K, N, S, E
