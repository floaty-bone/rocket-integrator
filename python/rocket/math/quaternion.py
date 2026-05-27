"""
quaternion.py — Quaternion utilities (NumPy + JAX backends).

Two numerical backends are exposed:
    * ``_np`` suffix — pure NumPy, used in the hot integrator loop.
    * ``_jax`` suffix — JAX, used by code that needs autodiff
                        (``rocket.plant.dynamics.make_F`` is differentiated by
                        ``jax.jacfwd`` to build the LQR Jacobians).

Both backends implement the same mathematical operations bit-for-bit; the only
reason they coexist is the JAX↔NumPy boundary overhead in tight loops.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from rocket.math.types import AngularVelocityVector, QuaternionVector


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def quat_to_rotmat_np(q: QuaternionVector) -> npt.NDArray[np.float64]:
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3×3 rotation matrix.

    The returned matrix R satisfies ``v_inertial = R @ v_body``.

    Args:
        q: Length-4 array [qw, qx, qy, qz].  Need not be pre-normalised.

    Returns:
        3×3 rotation matrix (body → inertial).
    """
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}.")

    q_norm = q / np.linalg.norm(q)
    w, x, y, z = q_norm

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2*(y2 + z2),   2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(x2 + z2),   2*(yz - wx)],
        [    2*(xz - wy),   2*(yz + wx),   1 - 2*(x2 + y2)],
    ], dtype=np.float64)


def construct_omega_matrix_np(w: AngularVelocityVector) -> npt.NDArray[np.float64]:
    """Build the 4×4 Omega matrix used in the quaternion kinematic equation
    ``dq/dt = ½ Ω(ω) q``.

    Args:
        w: Angular velocity vector [wx, wy, wz] in the body frame.

    Returns:
        4×4 skew-symmetric Omega matrix.
    """
    if w.shape != (3,):
        raise ValueError(f"Expected angular velocity of shape (3,), got {w.shape}.")

    wx, wy, wz = w
    return np.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0],
    ], dtype=np.float64)


def L(q: np.ndarray) -> np.ndarray:
    """Left-quaternion multiplication matrix L(q) such that ``q1 ⊗ q2 = L(q1) @ q2``.

    Used by the attitude-error computation in
    ``rocket.control.attitude_law.controller``.
    """
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q
    return np.array([
        [qw, -qx, -qy, -qz],
        [qx,  qw, -qz,  qy],
        [qy,  qz,  qw, -qx],
        [qz, -qy,  qx,  qw],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

def quat_to_rotmat_jax(q):
    """JAX version of :func:`quat_to_rotmat_np` — used inside ``make_F`` so the
    full dynamics function is autodifferentiable via ``jax.jacfwd``."""
    q = jnp.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}.")

    q = q / jnp.linalg.norm(q)
    w, x, y, z = q

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return jnp.array([
        [1 - 2*(y2 + z2),   2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(x2 + z2),   2*(yz - wx)],
        [    2*(xz - wy),   2*(yz + wx),   1 - 2*(x2 + y2)],
    ], dtype=np.float64)


def construct_omega_matrix_jax(w):
    """JAX version of :func:`construct_omega_matrix_np`."""
    w = jnp.asarray(w, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError(f"Expected angular velocity of shape (3,), got {w.shape}.")

    wx, wy, wz = w
    return jnp.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0],
    ], dtype=np.float64)
