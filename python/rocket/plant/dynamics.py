"""
dynamics.py — Continuous-time 6-DOF rigid-body dynamics ``ẋ = F(x, u)``.

The :func:`make_F` factory pre-computes inertia inverses and closes over the
vehicle parameters, returning a JAX-friendly callable suitable for
``jax.jacfwd`` (used by :mod:`rocket.control.lqr_design` to build A/B matrices).

State vector ``x`` (13):
    [px, py, pz,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]

Control vector ``u`` (9):
    [X1, Y1, Z1,  X2, Y2, Z2,  X3, Y3, Z3]  — Cartesian body-frame thrust per engine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rocket.math.quaternion import construct_omega_matrix_jax, quat_to_rotmat_jax
from rocket.plant.thrust import compute_thrust_forces_and_moments_cartesian_jax

# Enable 64-bit precision for JAX so the Jacobians match the NumPy integrator.
jax.config.update("jax_enable_x64", True)


def make_F(
    mass: float,
    inertia_matrix: np.ndarray,
    gravity_force: np.ndarray,
    a: float = 1.5,
    l: float = 18.0,
):
    """Factory that closes over vehicle constants and returns a fast ``F(X, U)``
    callable suitable for autodiff.

    Args:
        mass:           Vehicle mass (kg).
        inertia_matrix: 3×3 inertia tensor in the body frame (kg·m²).
        gravity_force:  3-element gravity force in the inertial frame (N).
        a:              Engine cluster radius (m).
        l:              Distance from CoM to the engine plane along X (m).

    Returns:
        Callable ``F(X, U) → Ẋ`` where both arrays are length-13 / length-9.
    """
    _mass    = mass
    _I       = np.asarray(inertia_matrix, dtype=np.float64)
    _I_inv   = np.linalg.inv(_I)
    _gravity = np.asarray(gravity_force, dtype=np.float64)
    _a, _l   = a, l

    def F(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        pos, quat, vel, omega = X[0:3], X[3:7], X[7:10], X[10:13]

        U_cart      = U.reshape(3, 3)
        body_wrench = compute_thrust_forces_and_moments_cartesian_jax(U_cart, a=_a, l=_l)
        R           = quat_to_rotmat_jax(quat)
        Iw          = _I @ omega

        dP = vel
        dq = 0.5 * construct_omega_matrix_jax(omega) @ quat
        dv = (R @ body_wrench[:3] + _gravity) / _mass
        dw = _I_inv @ (body_wrench[3:] - jnp.cross(omega, Iw))

        return jnp.concatenate([dP, dq, dv, dw])

    return F
