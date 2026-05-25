"""
attitude_law.py — Runtime LQR feedback law with Rodrigues attitude error.

This is the *hot-path* side of the controller (fast):
  * Compute the attitude error in Rodrigues parameters (Gibbs vector) so the
    discontinuous 4-D quaternion error collapses to a smooth 3-D vector.
  * Stack ``δp, φ, δv, δω`` into the 12-D error state ``δx``.
  * Apply the precomputed gain ``K`` and bias by the nominal control ``u_hover``.

Companion to the design-time gain synthesis in
:mod:`rocket.control.lqr_design`.
"""

from __future__ import annotations

import numpy as np

from rocket.math.tangent_space import qtorp


def controller(
    x: np.ndarray,
    x_setpoint: np.ndarray,
    u_hover: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """12-D LQR feedback law using Rodrigues parameters for the attitude error.

    Args:
        x:          Current 13-element state.
        x_setpoint: Reference 13-element state.
        u_hover:    Nominal control vector at the operating point.
        K:          12×n_u LQR gain matrix from :func:`make_lqr_computer`.

    Returns:
        ``u = u_hover − K · δx`` where ``δx`` is the 12-D tangent-space error.
    """
    # Error quaternion δq = L(q_setpoint)ᵀ @ q  (L inlined — no 4×4 allocation)
    #
    # L(q) = [[ qw, -qx, -qy, -qz],
    #          [ qx,  qw, -qz,  qy],
    #          [ qy,  qz,  qw, -qx],
    #          [ qz, -qy,  qx,  qw]]
    #
    # L(q)ᵀ = [[ qw,  qx,  qy,  qz],
    #           [-qx,  qw,  qz, -qy],
    #           [-qy, -qz,  qw,  qx],
    #           [-qz,  qy, -qx,  qw]]
    sw = x_setpoint[3];  sx = x_setpoint[4];  sy = x_setpoint[5];  sz = x_setpoint[6]
    cw = x[3];           cx = x[4];           cy = x[5];           cz = x[6]

    eq = np.array([
         sw*cw + sx*cx + sy*cy + sz*cz,
        -sx*cw + sw*cx + sz*cy - sy*cz,
        -sy*cw - sz*cx + sw*cy + sx*cz,
        -sz*cw + sy*cx - sx*cy + sw*cz,
    ], dtype=np.float64)

    phi = qtorp(eq, scale=2.0)

    dx = np.empty(12, dtype=np.float64)
    dx[0:3]  = x[0:3]   - x_setpoint[0:3]
    dx[3:6]  = phi
    dx[6:9]  = x[7:10]  - x_setpoint[7:10]
    dx[9:12] = x[10:13] - x_setpoint[10:13]

    return u_hover - K @ dx
