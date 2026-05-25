"""
thrust.py — Multi-engine thrust model (NumPy + JAX backends).

Engines are arranged in an equilateral triangle at the base of the rocket and
gimbal in spherical coordinates ``[θ, φ, T]``.  This module converts between
that gimbal representation and body-frame Cartesian thrust, and sums the per-
engine forces and moments about the centre of mass G into a 6-element body
wrench ``[Fx, Fy, Fz, Mx, My, Mz]``.

Geometry (body frame, X is the rocket long axis, +X points out of the nose):
    * Engine plane sits a distance ``l`` aft of G along −X.
    * Engines lie on a circle of radius ``a`` in the Y–Z plane.

Two numerical backends are exposed:
    * ``_np`` — pure NumPy, used in the hot SIL loop
                (see :mod:`rocket.scenarios.sil_hover` and the integrator's
                ``get_body_wrench`` callback).
    * ``_jax`` — JAX, used inside :func:`rocket.plant.dynamics.make_F` so that
                 the dynamics function is autodifferentiable.
"""

from __future__ import annotations

from math import cos, pi

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from rocket.math.quaternion import quat_to_rotmat_jax
from rocket.math.types import (
    BodyWrench,
    EnginesThrustArray,
    QuaternionVector,
)


# ---------------------------------------------------------------------------
# Single-engine spherical ↔ Cartesian conversions
# ---------------------------------------------------------------------------

def engine_spherical_to_body_force_np(spherical: np.ndarray) -> np.ndarray:
    """Convert one engine's thrust ``[θ, φ, T]`` to body-frame Cartesian (NumPy)."""
    theta, phi, r = spherical[0], spherical[1], spherical[2]
    return np.array([
         r * np.cos(phi),
        +r * np.sin(phi) * np.sin(theta),
        +r * np.cos(theta) * np.sin(phi),
    ], dtype=np.float64)


def engine_spherical_to_body_force_jax(spherical):
    """JAX version of :func:`engine_spherical_to_body_force_np`."""
    theta, phi, r = spherical[..., 0], spherical[..., 1], spherical[..., 2]
    return jnp.array([
         r * jnp.cos(phi),
        +r * jnp.sin(phi) * jnp.sin(theta),
        +r * jnp.cos(theta) * jnp.sin(phi),
    ], dtype=np.float64)


def body_force_to_spherical_np(force: np.ndarray) -> np.ndarray:
    """Inverse of :func:`engine_spherical_to_body_force_np`."""
    fx, fy, fz = force[0], force[1], force[2]
    r = np.sqrt(fx**2 + fy**2 + fz**2)
    phi = np.arctan2(np.sqrt(fy**2 + fz**2), fx)
    theta = np.arctan2(fy, fz)
    return np.array([theta, phi, r], dtype=np.float64)


def body_force_to_spherical_jax(force):
    """JAX version of :func:`body_force_to_spherical_np`."""
    fx, fy, fz = force[..., 0], force[..., 1], force[..., 2]
    r = jnp.sqrt(fx**2 + fy**2 + fz**2)
    phi = jnp.arctan2(jnp.sqrt(fy**2 + fz**2), fx)
    theta = jnp.arctan2(fy, fz)
    return jnp.stack([theta, phi, r], axis=-1)


# ---------------------------------------------------------------------------
# Multi-engine wrench (force + moment about G)
# ---------------------------------------------------------------------------

def _engine_attachment_points(a: float, l: float):
    """Body-frame positions of the three engines (equilateral triangle on the
    base plane, ``l`` aft of G along −X, radius ``a``)."""
    return [
        [-l,  a * cos(pi / 6),  -a * cos(pi / 3)],
        [-l,  0.0,               a              ],
        [-l, -a * cos(pi / 6),  -a * cos(pi / 3)],
    ]


def compute_thrust_forces_and_moments_cartesian_np(
    engine_thrust_cartesian: np.ndarray,
    a: float,
    l: float,
) -> np.ndarray:
    """Sum three Cartesian per-engine body-frame forces into the 6-element wrench
    ``[Fx, Fy, Fz, Mx, My, Mz]`` taken about G (NumPy)."""
    resultant    = np.sum(engine_thrust_cartesian, axis=0)
    GE           = np.array(_engine_attachment_points(a, l), dtype=np.float64)
    total_moment = np.sum(np.cross(GE, engine_thrust_cartesian), axis=0)
    return np.concatenate((resultant, total_moment))


def compute_thrust_forces_and_moments_cartesian_jax(
    engine_thrust_cartesian: EnginesThrustArray,
    a: float,
    l: float,
) -> BodyWrench:
    """JAX version of :func:`compute_thrust_forces_and_moments_cartesian_np`."""
    resultant = jnp.sum(engine_thrust_cartesian, axis=0)
    GE = jnp.array(_engine_attachment_points(a, l), dtype=np.float64)

    total_moment = sum(
        jnp.cross(GE[i], engine_thrust_cartesian[i])
        for i in range(3)
    )

    return jnp.concatenate((resultant, total_moment))


def compute_thrust_forces_and_moments_jax(
    engine_thrust: EnginesThrustArray,
    a: float,
    l: float,
) -> BodyWrench:
    """Sum three spherical-coordinate per-engine commands into the 6-element
    wrench (JAX).  Used by code that prefers the ``[θ, φ, T]`` representation
    end-to-end."""
    theta  = engine_thrust[:, 0]
    phi    = engine_thrust[:, 1]
    thrust = engine_thrust[:, 2]

    resultant = jnp.array([
        +jnp.sum(thrust * jnp.cos(phi)),
        +jnp.sum(thrust * jnp.sin(phi) * jnp.sin(theta)),
        +jnp.sum(thrust * jnp.sin(phi) * jnp.cos(theta)),
    ], dtype=np.float64)

    GE = jnp.array(_engine_attachment_points(a, l), dtype=np.float64)

    total_moment = sum(
        jnp.cross(GE[i], engine_spherical_to_body_force_jax(engine_thrust[i]))
        for i in range(3)
    )

    return jnp.concatenate((resultant, total_moment))


# ---------------------------------------------------------------------------
# Frame conversion helper
# ---------------------------------------------------------------------------

def body_frame_to_inertial_frame_force(
    body_force: npt.NDArray[np.float64],
    quaternion: QuaternionVector,
) -> npt.NDArray[np.float64]:
    """Rotate a body-frame force vector into the inertial frame using the
    rotation matrix obtained from ``quaternion`` (JAX backend)."""
    R = quat_to_rotmat_jax(quaternion)
    return R @ body_force
