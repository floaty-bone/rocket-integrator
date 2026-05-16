"""
helper_np.py — Pure NumPy implementation of math utilities for the rocket integrator.
This avoids JAX-to-NumPy conversion overhead in high-frequency loops.
"""

from __future__ import annotations
import numpy as np
from math import cos, pi
from typing import Annotated
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StateVector           = Annotated[npt.NDArray[np.float64], (13,)]
EnginesThrustArray    = Annotated[npt.NDArray[np.float64], (3, 3)]
BodyWrench            = Annotated[npt.NDArray[np.float64], (6,)]
InertialForceVector   = Annotated[npt.NDArray[np.float64], (3,)]
AngularVelocityVector = Annotated[npt.NDArray[np.float64], (3,)]
QuaternionVector      = Annotated[npt.NDArray[np.float64], (4,)]

# ---------------------------------------------------------------------------
# Quaternion utilities (NumPy version)
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3×3 rotation matrix using NumPy."""
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


def construct_omega_matrix(w: np.ndarray) -> np.ndarray:
    """Build the 4×4 Omega matrix for quaternion kinematics using NumPy."""
    if w.shape != (3,):
        raise ValueError(f"Expected angular velocity of shape (3,), got {w.shape}.")

    wx, wy, wz = w
    return np.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Engine thrust model (NumPy version)
# ---------------------------------------------------------------------------

def engine_spherical_to_body_force(
    spherical: np.ndarray,
) -> np.ndarray:
    """Convert a single engine's thrust from spherical to body-frame Cartesian using NumPy."""
    theta, phi, r = spherical[0], spherical[1], spherical[2]
    return np.array([
         r * np.cos(phi),
        +r * np.sin(phi) * np.sin(theta),
        +r * np.cos(theta) * np.sin(phi),
    ], dtype=np.float64)


def body_force_to_spherical(
    force: np.ndarray,
) -> np.ndarray:
    """Convert a single engine's thrust from body-frame Cartesian to spherical using NumPy."""
    fx, fy, fz = force[0], force[1], force[2]
    r = np.sqrt(fx**2 + fy**2 + fz**2)
    phi = np.arctan2(np.sqrt(fy**2 + fz**2), fx)
    theta = np.arctan2(fy, fz)
    return np.array([theta, phi, r], dtype=np.float64)


def compute_thrust_forces_and_moments_cartesian(
    engine_thrust_cartesian: np.ndarray,
    a: float,
    l: float,
) -> np.ndarray:
    """Compute the total force and moment from three engines using NumPy."""
    # engine_thrust_cartesian shape (3, 3)
    resultant = np.sum(engine_thrust_cartesian, axis=0)

    # Engine attachment points in body frame (equilateral triangle)
    GE = np.array([
        [-l, a * cos(pi / 6),  -a * cos(pi / 3)],
        [-l,  0.0,             a              ],
        [-l,  -a * cos(pi / 6),  -a * cos(pi / 3)],
    ], dtype=np.float64)

    total_moment = np.zeros(3, dtype=np.float64)
    for i in range(3):
        total_moment += np.cross(GE[i], engine_thrust_cartesian[i])

    return np.concatenate((resultant, total_moment))
