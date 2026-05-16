import numpy as np
import jax.numpy as jnp
from rocket.dynamics_function import make_lqr_computer
from rocket.helper import controller


class LQRController:
    def __init__(self, F, Q=None, R=None):
        # 1. Setup weights
        if Q is None:
            Q = np.diag([1e3] * 13)
        if R is None:
            R = np.diag([1e3] * 9)

        # 2. Build the JAX-compiled LQR computer (computes K, A, B at any op-point)
        self.compute_lqr = make_lqr_computer(F, Q, R)

        self.K = None
        self.x_op = None
        self.u_op = None

        # 3. Initial linearization at hover (body +X thrust only)
        xl = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)
        ul = jnp.array([200000, 0, 0, 200000, 0, 0, 200000, 0, 0], dtype=jnp.float64)
        self.update_linearization(xl, ul)

    def update_linearization(self, xl, ul):
        """
        Runs at a lower rate (e.g. 10 Hz).
        Re-linearises the dynamics around (xl, ul) and recomputes K.
        """
        K, A, B = self.compute_lqr(xl, ul)
        self.K    = K
        self.x_op = xl
        self.u_op = ul

    def _compute_u(self, state, setpoint, u_nominal):
        """
        LQR regulation law using the tangent-space controller:
        
        Computes the control action using Rodrigues parameter attitude error and the 12D gain matrix K.
        """
        return controller(np.asarray(state), np.asarray(setpoint), np.asarray(u_nominal), self.K)

    def update(self, state, setpoint=None, u_nominal=None, **kwargs):
        """
        Runs at a high rate (e.g. 1 kHz).
        Computes and returns the control vector u (9,).
        """
        if setpoint is None:
            setpoint = self.x_op
        if u_nominal is None:
            u_nominal = self.u_op

        return self._compute_u(state, setpoint, u_nominal)
