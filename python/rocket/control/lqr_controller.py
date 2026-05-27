"""
lqr_controller.py — Stateful LQR controller bundling design + runtime.

``LQRController`` ties together:
  * The off-line gain synthesis from :mod:`rocket.control.lqr_design` (a
    single ``make_lqr_computer`` factory is built once at construction time).
  * The runtime feedback law from :mod:`rocket.control.attitude_law`.
  * A current operating point ``(x_op, u_op)`` and the corresponding gain ``K``.

Typical usage:

    controller = LQRController(F, Q=Q, R=R)
    controller.update_linearization(x_op, u_op)   # slow, e.g. 30 Hz
    u = controller.update(state, setpoint, u_nominal)   # fast, e.g. 5 kHz
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rocket.control.attitude_law import controller as _attitude_law
from rocket.control.lqr_design import make_lqr_computer


class LQRController:
    """Stateful LQR controller with re-linearisation support."""

    def __init__(self, F, Q=None, R=None):
        if Q is None:
            Q = np.diag([1e3] * 13)
        if R is None:
            R = np.diag([1e3] * 9)

        self.compute_lqr = make_lqr_computer(F, Q, R)

        self.K = None
        self.x_op = None
        self.u_op = None

        # Bootstrap linearisation at hover (upright body, +X thrust only).
        xl = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)
        ul = jnp.array([200000, 0, 0, 200000, 0, 0, 200000, 0, 0], dtype=jnp.float64)
        self.update_linearization(xl, ul)

    def update_linearization(self, xl, ul):
        """Re-linearise the dynamics around ``(xl, ul)`` and refresh ``K``.

        Slow path — invoke at a low rate (≈ 10–30 Hz).
        """
        K, A, B = self.compute_lqr(xl, ul)
        self.K    = K
        self.x_op = xl
        self.u_op = ul

    def _compute_u(self, state, setpoint, u_nominal):
        return _attitude_law(
            np.asarray(state),
            np.asarray(setpoint),
            np.asarray(u_nominal),
            self.K,
        )

    def update(self, state, setpoint=None, u_nominal=None, **kwargs):
        """Compute the control vector ``u`` (9,) from the current state.

        Fast path — invoke at the control loop rate (≈ 1–5 kHz).
        """
        if setpoint is None:
            setpoint = self.x_op
        if u_nominal is None:
            u_nominal = self.u_op

        return self._compute_u(state, setpoint, u_nominal)
