"""
tvc.py — Thrust-Vector-Control coordinate conversions.

The flight-side controller works in Cartesian per-engine body forces
``[Fx, Fy, Fz]`` because that is the natural representation for the linearised
LQR design.  The hardware boundary, however, exposes the real TVC command:
``[α, β, T]`` per engine, where α is pitch about body Y, β is yaw about body Z,
and T is the engine throttle (Newtons).

Conversion (one engine):
    fx =  T · cos(α) · cos(β)
    fy =  T · cos(α) · sin(β)
    fz = −T · sin(α)

These ``cart9_to_gimbal9`` / ``gimbal9_to_cart9`` helpers stitch three engines
into a single length-9 vector matching the convention used by
:mod:`rocket.plant.dynamics`.
"""

from __future__ import annotations

import numpy as np


def engine_cart_to_euler(force: np.ndarray) -> np.ndarray:
    """Convert one engine's Cartesian thrust ``[Fx, Fy, Fz]`` → ``[α, β, T]``."""
    fx, fy, fz = force
    T = np.sqrt(fx * fx + fy * fy + fz * fz)
    if T < 1e-9:
        return np.zeros(3)
    return np.array([
        -np.arcsin(np.clip(fz / T, -1.0, 1.0)),
         np.arctan2(fy, fx),
         T,
    ])


def engine_euler_to_cart(gimbal: np.ndarray) -> np.ndarray:
    """Inverse of :func:`engine_cart_to_euler`."""
    alpha, beta, T = gimbal
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
         T * ca * np.cos(beta),
         T * ca * np.sin(beta),
        -T * sa,
    ])


def _engine_cart_to_euler_into(force: np.ndarray, out: np.ndarray) -> None:
    fx, fy, fz = force[0], force[1], force[2]
    T = np.sqrt(fx*fx + fy*fy + fz*fz)
    if T < 1e-9:
        out[0] = out[1] = out[2] = 0.0
    else:
        out[0] = -np.arcsin(np.clip(fz / T, -1.0, 1.0))
        out[1] = np.arctan2(fy, fx)
        out[2] = T


def _engine_euler_to_cart_into(gimbal: np.ndarray, out: np.ndarray) -> None:
    alpha, beta, T = gimbal[0], gimbal[1], gimbal[2]
    ca = np.cos(alpha);  sa = np.sin(alpha)
    out[0] =  T * ca * np.cos(beta)
    out[1] =  T * ca * np.sin(beta)
    out[2] = -T * sa


def cart9_to_gimbal9(u_cart: np.ndarray) -> np.ndarray:
    """Convert a 9-element Cartesian command (3 engines × ``[Fx, Fy, Fz]``)
    into a 9-element gimbal command (3 engines × ``[α, β, T]``)."""
    out = np.empty(9, dtype=np.float64)
    u   = u_cart.reshape(3, 3)
    _engine_cart_to_euler_into(u[0], out[0:3])
    _engine_cart_to_euler_into(u[1], out[3:6])
    _engine_cart_to_euler_into(u[2], out[6:9])
    return out


def gimbal9_to_cart9(u_gimbal: np.ndarray) -> np.ndarray:
    """Inverse of :func:`cart9_to_gimbal9`."""
    out = np.empty(9, dtype=np.float64)
    u   = u_gimbal.reshape(3, 3)
    _engine_euler_to_cart_into(u[0], out[0:3])
    _engine_euler_to_cart_into(u[1], out[3:6])
    _engine_euler_to_cart_into(u[2], out[6:9])
    return out
