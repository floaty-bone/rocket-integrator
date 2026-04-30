"""
integrator.py — 4th-order Runge-Kutta integrator for 6-DOF rigid-body dynamics.

State vector layout (13 elements):
    [0:3]   x, y, z        — position in inertial frame (m)
    [3:7]   qw, qx, qy, qz — attitude quaternion (body → inertial)
    [7:10]  vx, vy, vz     — velocity in inertial frame (m/s)
    [10:13] wx, wy, wz     — angular velocity in body frame (rad/s)

Force model — two orthogonal callbacks, each evaluated once per RK4 step:

    get_body_wrench() → [Fx, Fy, Fz, Mx, My, Mz]  (body frame)
        Forces and moments produced by actuators — thrust, aerodynamic
        control surfaces, reaction wheels, etc.  Forces are expressed in
        the body frame and rotated to the inertial frame internally via
        R @ F_body before entering dv/dt.  Moments are used directly in
        Euler's rotation equation (body frame).

    get_inertial_force() → [Fx, Fy, Fz]  (inertial frame, Newtons)
        Forces whose natural description is already in the inertial frame
        — gravity, electric/magnetic forces, drag models that decompose in
        wind axes aligned with inertial directions, etc.  No rotation is
        applied; the vector is added directly to dv/dt as F_inertial/m.
        Gravity is the canonical use-case:
            get_inertial_force = lambda: np.array([0, 0, mass * 9.8])

        If a force is more conveniently expressed in the body frame, put it
        in get_body_wrench instead — the two callbacks are additive in the
        translational equation of motion:
            dv/dt = (R @ F_body + F_inertial) / m
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rocket.helper import (
    BodyWrench,
    InertialForceVector,
    StateVector,
    construct_omega_matrix,
    quat_to_rotmat,
)

# RK4 blend weights  [k1, k2, k3, k4]
_RK4_WEIGHTS = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64)


class RK4Integrator:
    """6-DOF rigid-body dynamics integrator using the classical RK4 scheme.

    The integrator propagates a 13-element state vector one time-step forward
    by evaluating the equations of motion at four intermediate points and
    combining them with the standard 1/6 – 1/3 – 1/3 – 1/6 blend.

    Force model
    -----------
    Two separate callbacks cover the two natural frames for forces:

    ``get_body_wrench``
        Returns a 6-element wrench [Fx, Fy, Fz, Mx, My, Mz] where every
        component is expressed in the **body frame**.  Thrust, TVC moments,
        and aerodynamic body forces belong here.  The force part is rotated
        to the inertial frame (R @ F_body) before entering the translational
        equation; the moment part feeds directly into Euler's rotation
        equation unchanged.

    ``get_inertial_force``
        Returns a 3-element force [Fx, Fy, Fz] in **Newtons** expressed in
        the **inertial frame**.  No frame rotation is applied.  Use this for
        forces whose geometry is fixed in the world — gravity is the primary
        example.  The caller is responsible for scaling by mass when
        appropriate (i.e. return mass * g_vec, not g_vec).

    Translational equation of motion:
        dv/dt = (R @ F_body + F_inertial) / m

    Args:
        get_body_wrench:    Callable → 6-element [Fx, Fy, Fz, Mx, My, Mz]
                            in the body frame.  See Force model above.
        get_inertial_force: Callable → 3-element [Fx, Fy, Fz] in the
                            inertial frame (N).  See Force model above.
        inertia_matrix:     3×3 inertia tensor in the body frame (kg·m²).
                            Defaults to the identity matrix.
        length:             Characteristic length of the rocket (m). Default 7.
        radius:             Engine cluster radius (m). Default 0.3.
        mass:               Total mass of the vehicle (kg). Default 85 000.
        step_size:          Integration time-step Δt (s). Default 0.001.

    Raises:
        np.linalg.LinAlgError: If the provided inertia matrix is singular.
    """

    def __init__(
        self,
        get_body_wrench: Callable[[], BodyWrench],
        get_inertial_force: Callable[[], InertialForceVector],
        inertia_matrix: np.ndarray = np.eye(3, dtype=np.float64),
        length: float = 7.0,
        radius: float = 0.3,
        mass: float = 85_000.0,
        step_size: float = 0.001,
    ) -> None:
        self.get_body_wrench    = get_body_wrench
        self.get_inertial_force = get_inertial_force
        self.inertia_matrix     = np.asarray(inertia_matrix, dtype=np.float64)
        self.length             = length
        self.radius             = radius
        self.mass               = mass
        self.step               = step_size

        # Pre-compute the inverse once — it never changes between steps.
        self._inertia_inv = np.linalg.inv(self.inertia_matrix)

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
        body_wrench: BodyWrench,
        inertial_force: InertialForceVector,
    ) -> np.ndarray:
        """dv/dt = (R @ F_body + F_inertial) / m

        Both force contributions are summed in the inertial frame before
        dividing by mass.  R rotates the body-frame thrust into inertial
        coordinates; F_inertial (gravity, etc.) is already there.
        """
        R = quat_to_rotmat(quaternion)
        return (R @ body_wrench[:3] + inertial_force) / self.mass

    def _d_angular_velocity(
        self,
        angular_velocity: np.ndarray,
        body_wrench: BodyWrench,
    ) -> np.ndarray:
        """dω/dt = I⁻¹ (M − ω × (I ω))   (Euler's rotation equation)

        Only the moment part of the body wrench (indices [3:6]) is used here.
        Inertial-frame forces produce no torque about the body axes directly
        (they are accounted for in dv/dt).
        """
        Iw = self.inertia_matrix @ angular_velocity
        return self._inertia_inv @ (body_wrench[3:] - np.cross(angular_velocity, Iw))

    # ------------------------------------------------------------------
    # State derivative  (bundles all four equations into one vector)
    # ------------------------------------------------------------------

    def _state_derivative(
        self,
        state: StateVector,
        body_wrench: BodyWrench,
        inertial_force: InertialForceVector,
    ) -> np.ndarray:
        """Evaluate ṡ = f(s) for the full 13-element state."""
        pos, quat, vel, omega = state[0:3], state[3:7], state[7:10], state[10:13]

        ds = np.empty(13, dtype=np.float64)
        ds[0:3]   = self._d_position(vel)
        ds[3:7]   = self._d_quaternion(quat, omega)
        ds[7:10]  = self._d_velocity(quat, body_wrench, inertial_force)
        ds[10:13] = self._d_angular_velocity(omega, body_wrench)
        return ds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step_forward(self, state: StateVector) -> StateVector:
        """Advance the state by one time-step using classical RK4.

        Both force callbacks are evaluated **once** at the start of the step
        and held constant across all four RK4 stages.  This matches the
        assumption that actuator commands and gravitational field do not
        change appreciably over one time-step (typically 1 ms).

        Args:
            state: Current 13-element state vector.

        Returns:
            Next 13-element state vector.
        """
        h  = self.step
        bw = self.get_body_wrench()      # body-frame wrench — constant within the step
        fi = self.get_inertial_force()   # inertial-frame force — constant within the step

        k1 = self._state_derivative(state,                bw, fi)
        k2 = self._state_derivative(state + 0.5 * h * k1, bw, fi)
        k3 = self._state_derivative(state + 0.5 * h * k2, bw, fi)
        k4 = self._state_derivative(state +         h * k3, bw, fi)

        # Weighted blend: (k1 + 2k2 + 2k3 + k4) / 6
        K = np.column_stack((k1, k2, k3, k4))   # shape (13, 4)
        return state + h * (K @ _RK4_WEIGHTS)
