"""
helper.py — Math utilities for the rocket RK4 integrator.

Provides:
  - Type aliases for annotated NumPy arrays
  - Quaternion → rotation matrix conversion
  - Quaternion kinematic Omega matrix construction
  - Engine spherical-coordinate → body-frame force conversion
  - Multi-engine thrust and moment resultant computation
"""

from __future__ import annotations

from math import cos, pi
from typing import Annotated

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StateVector               = Annotated[npt.NDArray[np.float64], (13,)]
EnginesThrustArray        = Annotated[npt.NDArray[np.float64], (3, 3)]   # [theta, phi, thrust] per engine
ForceMomentVector         = Annotated[npt.NDArray[np.float64], (6,)]     # [Fx, Fy, Fz, Mx, My, Mz]
AngularVelocityVector     = Annotated[npt.NDArray[np.float64], (3,)]     # [wx, wy, wz]  (body frame)
QuaternionVector          = Annotated[npt.NDArray[np.float64], (4,)]     # [qw, qx, qy, qz]
GravityVector             = Annotated[npt.NDArray[np.float64], (3,)]     # [gx, gy, gz]  (inertial frame)


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: QuaternionVector) -> npt.NDArray[np.float64]:
    """Convert a unit quaternion [qw, qx, qy, qz] to a 3×3 rotation matrix.

    The returned matrix R satisfies:
        v_inertial = R @ v_body

    Args:
        q: Length-4 array [qw, qx, qy, qz].  Need not be pre-normalised.

    Returns:
        3×3 rotation matrix (body → inertial).

    Raises:
        ValueError: If q does not have exactly 4 elements.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}.")

    q = q / np.linalg.norm(q)
    w, x, y, z = q

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2*(y2 + z2),   2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(x2 + z2),   2*(yz - wx)],
        [    2*(xz - wy),   2*(yz + wx),   1 - 2*(x2 + y2)],
    ], dtype=np.float64)


def construct_omega_matrix(w: AngularVelocityVector) -> npt.NDArray[np.float64]:
    """Build the 4×4 Omega matrix for quaternion kinematics.

    Used in the differential equation:
        dq/dt = ½ Ω(ω) q

    Args:
        w: Angular velocity vector [wx, wy, wz] in the body frame.

    Returns:
        4×4 skew-symmetric Omega matrix.

    Raises:
        ValueError: If w does not have exactly 3 elements.
    """
    w = np.asarray(w, dtype=np.float64)
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
# Engine thrust model
# ---------------------------------------------------------------------------

def engine_spherical_to_body_force(
    spherical: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert a single engine's thrust from spherical to body-frame Cartesian.

    Args:
        spherical: Array [theta, phi, r] where theta and phi are gimbal angles
                   (radians) and r is the thrust magnitude (N).

    Returns:
        Force vector [Fx, Fy, Fz] in the body frame.
    """
    theta, phi, r = spherical[..., 0], spherical[..., 1], spherical[..., 2]
    return np.array([
         r * np.cos(phi),
        +r * np.sin(phi) * np.sin(theta),
        +r * np.cos(theta) * np.sin(phi),
    ], dtype=np.float64)


def compute_thrust_forces_and_moments(
    engine_thrust: EnginesThrustArray,
    a: float,
    l: float,
) -> ForceMomentVector:
    """Compute the total  force and moment from three gimballed engines expressed in the body frame.

    Engines are arranged in an equilateral triangle at the base of the rocket.
    Moments are taken about the centre of mass G.

    Args:
        engine_thrust: (3, 3) array; each row is [theta, phi, thrust] for one engine.
        a: Engine cluster radius — distance from rocket centreline to each engine (m).
        l: Distance from the centre of mass to the engine plane along the X axis (m).

    Returns:
        6-element vector [Fx, Fy, Fz, Mx, My, Mz] in the body frame.
    """
    theta  = engine_thrust[:, 0]
    phi    = engine_thrust[:, 1]
    thrust = engine_thrust[:, 2]

    resultant = np.array([
        +np.sum(thrust * np.cos(phi)),
        +np.sum(thrust * np.sin(phi) * np.sin(theta)),
        +np.sum(thrust * np.sin(phi) * np.cos(theta)),
    ], dtype=np.float64)

    # Engine attachment points in body frame (equilateral triangle)
    GE = np.array([
        [-l, a * cos(pi / 6),  -a * cos(pi / 3)],
        [-l,  0.0,             a              ],
        [-l,  -a * cos(pi / 6),  -a * cos(pi / 3)],
    ], dtype=np.float64)

    total_moment = sum(
        np.cross(GE[i], engine_spherical_to_body_force(engine_thrust[i]))
        for i in range(3)
    )

    return np.concatenate((resultant, total_moment))

def body_frame_to_inertial_frame_force(
    body_force: npt.NDArray[np.float64],
    quaternion: QuaternionVector,
) -> npt.NDArray[np.float64]:
    """Convert a force vector from the body frame to the inertial frame.

    Args:
        body_force: Force vector [Fx, Fy, Fz] in the body frame.
        quaternion:  Orientation of the body as a unit quaternion [qw, qx, qy, qz].

    Returns:
        Force vector in the inertial frame.
    """
    R = quat_to_rotmat(quaternion)
    return R @ body_force