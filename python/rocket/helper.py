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

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StateVector           = Annotated[npt.NDArray[np.float64], (13,)]
EnginesThrustArray    = Annotated[npt.NDArray[np.float64], (3, 3)]  # [theta, phi, thrust] per engine
BodyWrench            = Annotated[npt.NDArray[np.float64], (6,)]    # [Fx, Fy, Fz, Mx, My, Mz] — body frame
InertialForceVector   = Annotated[npt.NDArray[np.float64], (3,)]    # [Fx, Fy, Fz] — inertial frame (N)
AngularVelocityVector = Annotated[npt.NDArray[np.float64], (3,)]    # [wx, wy, wz]  (body frame)
QuaternionVector      = Annotated[npt.NDArray[np.float64], (4,)]    # [qw, qx, qy, qz]


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
    return jnp.array([
         r * jnp.cos(phi),
        +r * jnp.sin(phi) * jnp.sin(theta),
        +r * jnp.cos(theta) * jnp.sin(phi),
    ], dtype=np.float64)


def body_force_to_spherical(
    force: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert a single engine's thrust from body-frame Cartesian to spherical.

    Args:
        force: Force vector [Fx, Fy, Fz] in the body frame.

    Returns:
        Array [theta, phi, r] where theta and phi are gimbal angles (radians)
        and r is the thrust magnitude (N).
    """
    fx, fy, fz = force[..., 0], force[..., 1], force[..., 2]
    r = jnp.sqrt(fx**2 + fy**2 + fz**2)
    phi = jnp.arctan2(jnp.sqrt(fy**2 + fz**2), fx)
    theta = jnp.arctan2(fy, fz)
    return jnp.stack([theta, phi, r], axis=-1)


def compute_thrust_forces_and_moments_cartesian(
    engine_thrust_cartesian: EnginesThrustArray,
    a: float,
    l: float,
) -> BodyWrench:
    """Compute the total force and moment from three engines given their Cartesian body forces.

    Engines are arranged in an equilateral triangle at the base of the rocket.
    Moments are taken about the centre of mass G.

    Args:
        engine_thrust_cartesian: (3, 3) array; each row is [Fx, Fy, Fz] for one engine.
        a: Engine cluster radius — distance from rocket centreline to each engine (m).
        l: Distance from the centre of mass to the engine plane along the X axis (m).

    Returns:
        6-element vector [Fx, Fy, Fz, Mx, My, Mz] in the body frame.
    """
    resultant = jnp.sum(engine_thrust_cartesian, axis=0)

    # Engine attachment points in body frame (equilateral triangle)
    GE = jnp.array([
        [-l, a * cos(pi / 6),  -a * cos(pi / 3)],
        [-l,  0.0,             a              ],
        [-l,  -a * cos(pi / 6),  -a * cos(pi / 3)],
    ], dtype=np.float64)

    total_moment = sum(
        jnp.cross(GE[i], engine_thrust_cartesian[i])
        for i in range(3)
    )

    return jnp.concatenate((resultant, total_moment))


def compute_thrust_forces_and_moments(
    engine_thrust: EnginesThrustArray,
    a: float,
    l: float,
) -> BodyWrench:
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

    resultant = jnp.array([
        +jnp.sum(thrust * jnp.cos(phi)),
        +jnp.sum(thrust * jnp.sin(phi) * jnp.sin(theta)),
        +jnp.sum(thrust * jnp.sin(phi) * jnp.cos(theta)),
    ], dtype=np.float64)

    # Engine attachment points in body frame (equilateral triangle)
    GE = jnp.array([
        [-l, a * cos(pi / 6),  -a * cos(pi / 3)],
        [-l,  0.0,             a              ],
        [-l,  -a * cos(pi / 6),  -a * cos(pi / 3)],
    ], dtype=np.float64)

    total_moment = sum(
        jnp.cross(GE[i], engine_spherical_to_body_force(engine_thrust[i]))
        for i in range(3)
    )

    return jnp.concatenate((resultant, total_moment))

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


# ---------------------------------------------------------------------------
# Tangent space projection and attitude controller utilities
# ---------------------------------------------------------------------------

def L(q: np.ndarray) -> np.ndarray:
    """Left-quaternion multiplication matrix L(q) such that q1_multiply_q2 = L(q1) @ q2."""
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q
    return np.array([
        [qw, -qx, -qy, -qz],
        [qx,  qw, -qz,  qy],
        [qy,  qz,  qw, -qx],
        [qz, -qy,  qx,  qw]
    ], dtype=np.float64)

def E(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Tangent space projection matrix E(q) mapping 12D tangent perturbation to 13D state perturbation."""
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    G_T = scale * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw]
    ], dtype=np.float64)

    E_mat = np.zeros((13, 12), dtype=np.float64)
    E_mat[0:3, 0:3] = np.eye(3)
    E_mat[3:7, 3:6] = G_T
    E_mat[7:10, 6:9] = np.eye(3)
    E_mat[10:13, 9:12] = np.eye(3)
    return E_mat

def E_pinv(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Left pseudoinverse of the tangent space projection matrix E(q)."""
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    G_pinv = (1.0 / scale) * np.array([
        [-qx,  qw,  qz, -qy],
        [-qy, -qz,  qw,  qx],
        [-qz,  qy, -qx,  qw]
    ], dtype=np.float64)

    Ep = np.zeros((12, 13), dtype=np.float64)
    Ep[0:3, 0:3] = np.eye(3)
    Ep[3:6, 3:7] = G_pinv
    Ep[6:9, 7:10] = np.eye(3)
    Ep[9:12, 10:13] = np.eye(3)
    return Ep

def qtorp(q: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Convert a quaternion error q to Rodrigues parameters (Gibbs vector)."""
    q = np.asarray(q, dtype=np.float64)
    qw = q[0]
    qv = q[1:4]
    
    if qw < 0:
        qw = -qw
        qv = -qv
        
    if abs(qw) < 1e-8:
        return scale * qv / 1e-8
        
    return scale * qv / qw

def project_to_tangent_space(A_naive: np.ndarray, B_naive: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns A, B correctly projected to tangent space using the pseudoinverse of E(q)."""
    # Uses scale=0.5 and its corresponding pseudoinverse to project A and B
    E_mat = E(q, scale=0.5)
    Ep = E_pinv(q, scale=0.5)
    A = Ep @ A_naive @ E_mat
    B = Ep @ B_naive
    return A, B

def controller(x: np.ndarray, x_setpoint: np.ndarray, u_hover: np.ndarray, K: np.ndarray) -> np.ndarray:
    """12D LQR controller using Rodrigues parameter error for attitude (scale=2.0)."""
    q_setpoint = x_setpoint[3:7]
    q  = x[3:7]

    phi = qtorp(L(q_setpoint).T @ q, scale=2.0)

    dx = np.concatenate([
        x[0:3]  - x_setpoint[0:3],
        phi,
        x[7:10] - x_setpoint[7:10],
        x[10:13] - x_setpoint[10:13],
    ])

    return u_hover - K @ dx
