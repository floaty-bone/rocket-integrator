"""
tangent_space.py — Quaternion tangent-space projection utilities.

The 13-element state has a 4-element quaternion that lives on the unit 3-sphere
S³ rather than in ℝ⁴.  To run an LQR design on this state we project the
Jacobians ``A`` (13×13) and ``B`` (13×9) down to the 12-element tangent
space at the current attitude using a 13×12 basis matrix ``E(q)``:

    A_12 = E_pinv(q) @ A_13 @ E(q)
    B_12 = E_pinv(q) @ B_13

Two scale conventions are supported:
    scale = 0.5 → matches the half-angle convention ``dq/dt = ½ Ω(ω) q``.
    scale = 1.0 → makes the columns of ``E(q)`` orthonormal (so ``E_pinv = Eᵀ``).

For attitude *error* the small-angle 3-vector representation
``φ = scale · q_v / q_w`` (Rodrigues/Gibbs parameters) is used.  The conjugate
pair (``E`` with scale 0.5) ↔ (``qtorp`` with scale 2.0) yields the standard
LQR feedback law in :mod:`rocket.control.attitude_law`.
"""

from __future__ import annotations

import numpy as np


def E(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """13×12 tangent-space basis at quaternion ``q``.

    Maps a 12-element tangent perturbation ``[δp, δφ, δv, δω]`` into the
    13-element state perturbation ``[δp, δq, δv, δω]``.
    """
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    G_T = scale * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ], dtype=np.float64)

    E_mat = np.zeros((13, 12), dtype=np.float64)
    E_mat[0:3, 0:3]    = np.eye(3)
    E_mat[3:7, 3:6]    = G_T
    E_mat[7:10, 6:9]   = np.eye(3)
    E_mat[10:13, 9:12] = np.eye(3)
    return E_mat


def E_pinv(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Left pseudoinverse of :func:`E`.  Satisfies ``E_pinv @ E = I_12``."""
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    G_pinv = (1.0 / scale) * np.array([
        [-qx,  qw,  qz, -qy],
        [-qy, -qz,  qw,  qx],
        [-qz,  qy, -qx,  qw],
    ], dtype=np.float64)

    Ep = np.zeros((12, 13), dtype=np.float64)
    Ep[0:3, 0:3]    = np.eye(3)
    Ep[3:6, 3:7]    = G_pinv
    Ep[6:9, 7:10]   = np.eye(3)
    Ep[9:12, 10:13] = np.eye(3)
    return Ep


def qtorp(q: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Convert a quaternion error ``q`` to Rodrigues parameters (Gibbs vector).

    Handles the quaternion double-cover (``q`` and ``-q`` map to the same φ)
    and guards against the 180°-singularity ``q_w → 0``.
    """
    q = np.asarray(q, dtype=np.float64)
    qw = q[0]
    qv = q[1:4]

    if qw < 0:
        qw = -qw
        qv = -qv

    if abs(qw) < 1e-8:
        return scale * qv / 1e-8

    return scale * qv / qw


def project_to_tangent_space(
    A_naive: np.ndarray,
    B_naive: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(A, B)`` from the 13-D state space into the 12-D tangent
    space at quaternion ``q`` using ``scale = 0.5`` and the pseudoinverse.

    Returns:
        ``(A_12, B_12)`` with shapes (12, 12) and (12, n_u) respectively.
    """
    E_mat = E(q, scale=0.5)
    Ep    = E_pinv(q, scale=0.5)
    A = Ep @ A_naive @ E_mat
    B = Ep @ B_naive
    return A, B
