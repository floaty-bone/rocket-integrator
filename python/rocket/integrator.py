"""
integrator.py — 4th-order Runge-Kutta integrator for 6-DOF rigid-body dynamics.

State vector layout (13 elements):
    [0:3]   x, y, z        — position in inertial frame (m)
    [3:7]   qw, qx, qy, qz — attitude quaternion (body → inertial)
    [7:10]  vx, vy, vz     — velocity in inertial frame (m/s)
    [10:13] wx, wy, wz     — angular velocity in body frame (rad/s)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rocket.helper import (
    ForceMomentVector,
    StateVector,
    construct_omega_matrix,
    quat_to_rotmat,
)

# RK4 blend weights  [k1, k2, k3, k4]
_RK4_WEIGHTS = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64)

# Inertial gravity direction (positive Z is "down" by convention here)
_GRAVITY_DIRECTION = np.array([0.0, 0.0, 1.0], dtype=np.float64)


class RK4Integrator:
    """6-DOF rigid-body dynamics integrator using the classical RK4 scheme.

    The integrator propagates a 13-element state vector one time-step forward
    by evaluating the equations of motion at four intermediate points and
    combining them with the standard 1/6 – 1/3 – 1/3 – 1/6 blend.

    Args:
        get_force_moment: Callable that returns a 6-element force/moment vector
                          [Fx, Fy, Fz, Mx, My, Mz] in the body frame.
        inertia_matrix:   3×3 inertia tensor in the body frame (kg·m²).
                          Defaults to the identity matrix.
        length:           Characteristic length of the rocket (m). Default 7.
        radius:           Engine cluster radius (m). Default 0.3.
        mass:             Total mass of the vehicle (kg). Default 85 000.
        gravity:          Gravitational acceleration magnitude (m/s²).
                          Applied along ``_GRAVITY_DIRECTION``. Default 9.8.
        step_size:        Integration time-step Δt (s). Default 0.001.

    Raises:
        np.linalg.LinAlgError: If the provided inertia matrix is singular.
    """

    def __init__(
        self,
        get_force_moment: Callable[[], ForceMomentVector],
        inertia_matrix: np.ndarray = np.eye(3, dtype=np.float64),
        length: float = 7.0,
        radius: float = 0.3,
        mass: float = 85_000.0,
        gravity: float = 9.8,
        step_size: float = 0.001,
    ) -> None:
        self.get_force_moment = get_force_moment
        self.inertia_matrix   = np.asarray(inertia_matrix, dtype=np.float64)
        self.length           = length
        self.radius           = radius
        self.mass             = mass
        self.step             = step_size

        # Pre-compute the inverse once — it never changes between steps.
        self._inertia_inv = np.linalg.inv(self.inertia_matrix)

        # Cache the gravity vector so we don't rebuild it every step.
        self._gravity = _GRAVITY_DIRECTION * gravity

    # ------------------------------------------------------------------
    # Equations of motion (derivatives)
    # ------------------------------------------------------------------

    def _d_position(self, velocity: np.ndarray) -> np.ndarray:
        """dx/dt = v  (trivial, but kept explicit for clarity)."""
        return velocity

    def _d_quaternion(
        self,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """dq/dt = ½ Ω(ω) q"""
        return 0.5 * construct_omega_matrix(angular_velocity) @ quaternion

    def _d_velocity(
        self,
        quaternion: np.ndarray,
        force_moment: ForceMomentVector,
    ) -> np.ndarray:
        """dv/dt = (1/m)(R @ F_body) + g_inertial"""
        R = quat_to_rotmat(quaternion)
        return (R @ force_moment[:3]) / self.mass + self._gravity

    def _d_angular_velocity(
        self,
        angular_velocity: np.ndarray,
        force_moment: ForceMomentVector,
    ) -> np.ndarray:
        """dω/dt = I⁻¹ (M − ω × (I ω))   (Euler's rotation equation)"""
        Iw = self.inertia_matrix @ angular_velocity
        return self._inertia_inv @ (force_moment[3:] - np.cross(angular_velocity, Iw))

    # ------------------------------------------------------------------
    # State derivative  (bundles all four equations into one vector)
    # ------------------------------------------------------------------

    def _state_derivative(
        self,
        state: StateVector,
        force_moment: ForceMomentVector,
    ) -> np.ndarray:
        """Evaluate ṡ = f(s) for the full 13-element state."""
        pos, quat, vel, omega = state[0:3], state[3:7], state[7:10], state[10:13]

        ds = np.empty(13, dtype=np.float64)
        ds[0:3]   = self._d_position(vel)
        ds[3:7]   = self._d_quaternion(quat, omega)
        ds[7:10]  = self._d_velocity(quat, force_moment)
        ds[10:13] = self._d_angular_velocity(omega, force_moment)
        return ds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step_forward(self, state: StateVector) -> StateVector:
        """Advance the state by one time-step using classical RK4.

        Args:
            state: Current 13-element state vector.

        Returns:
            Next 13-element state vector.
        """
        h  = self.step
        fm = self.get_force_moment()   # force/moment is constant within the step

        k1 = self._state_derivative(state,               fm)
        k2 = self._state_derivative(state + 0.5 * h * k1, fm)
        k3 = self._state_derivative(state + 0.5 * h * k2, fm)
        k4 = self._state_derivative(state +         h * k3, fm)

        # Weighted blend: (k1 + 2k2 + 2k3 + k4) / 6
        K = np.column_stack((k1, k2, k3, k4))           # shape (13, 4)
        return state + h * (K @ _RK4_WEIGHTS)
