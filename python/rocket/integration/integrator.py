"""
integrator.py — Vehicle-agnostic 4th-order Runge-Kutta integrator for 6-DOF
rigid-body dynamics.

This module is intentionally generic: it propagates a 13-element rigid-body
state forward in time and knows **nothing** about engines, control surfaces,
LQR, or any specific vehicle.  Coupling to a particular vehicle happens only
through the two force callbacks (``get_body_wrench``, ``get_inertial_force``),
so the same integrator can drive a rocket, a CubeSat, a quadrotor, or any
other rigid body — only the callbacks change.

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

from rocket.math.types import BodyWrench, InertialForceVector, StateVector

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
    """

    def __init__(
        self,
        get_body_wrench: Callable[[], BodyWrench],
        get_inertial_force: Callable[[], InertialForceVector],
        inertia_matrix: np.ndarray = np.eye(3, dtype=np.float64),
        mass: float = 85_000.0,
        step_size: float = 0.001,
    ) -> None:
        self.get_body_wrench    = get_body_wrench
        self.get_inertial_force = get_inertial_force
        self.inertia_matrix     = np.asarray(inertia_matrix, dtype=np.float64)
        self.mass               = mass
        self.step               = step_size

        # Pre-compute the inverse and reciprocal mass once — they never change.
        self._inertia_inv = np.linalg.inv(self.inertia_matrix)
        self._inv_mass    = 1.0 / mass

        # Pre-allocated hot-loop buffers — eliminates ~18 k allocs/sec at 1 kHz.
        self._k1    = np.empty(13, dtype=np.float64)
        self._k2    = np.empty(13, dtype=np.float64)
        self._k3    = np.empty(13, dtype=np.float64)
        self._k4    = np.empty(13, dtype=np.float64)
        self._s_tmp = np.empty(13, dtype=np.float64)  # intermediate RK4 state
        self._K     = np.empty((13, 4), dtype=np.float64)  # [k1|k2|k3|k4]
        self._Iw    = np.empty(3, dtype=np.float64)   # I @ omega
        self._tau   = np.empty(3, dtype=np.float64)   # net body torque

    # ------------------------------------------------------------------
    # Equations of motion — inlined into _state_derivative_into for speed;
    # kept here as named helpers so callers/tests that reference them still work.
    # ------------------------------------------------------------------

    def _d_position(self, velocity: np.ndarray) -> np.ndarray:
        return velocity

    def _d_quaternion(
        self,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        from rocket.math.quaternion import construct_omega_matrix_np
        return 0.5 * construct_omega_matrix_np(angular_velocity) @ quaternion

    def _d_velocity(
        self,
        quaternion: np.ndarray,
        body_wrench: BodyWrench,
        inertial_force: InertialForceVector,
    ) -> np.ndarray:
        from rocket.math.quaternion import quat_to_rotmat_np
        R = quat_to_rotmat_np(quaternion)
        return (R @ body_wrench[:3] + inertial_force) / self.mass

    def _d_angular_velocity(
        self,
        angular_velocity: np.ndarray,
        body_wrench: BodyWrench,
    ) -> np.ndarray:
        Iw = self.inertia_matrix @ angular_velocity
        return self._inertia_inv @ (body_wrench[3:] - np.cross(angular_velocity, Iw))

    # ------------------------------------------------------------------
    # State derivative — two variants:
    #   _state_derivative       — allocates, kept for backward compat
    #   _state_derivative_into  — writes into a caller-supplied buffer (hot path)
    # ------------------------------------------------------------------

    def _state_derivative(
        self,
        state: StateVector,
        body_wrench: BodyWrench,
        inertial_force: InertialForceVector,
    ) -> np.ndarray:
        out = np.empty(13, dtype=np.float64)
        self._state_derivative_into(state, body_wrench, inertial_force, out)
        return out

    def _state_derivative_into(
        self,
        state: StateVector,
        body_wrench: BodyWrench,
        inertial_force: InertialForceVector,
        out: np.ndarray,
    ) -> None:
        """Compute the 13-element state derivative into `out` with zero heap allocation.

        Physics is identical to calling the four _d_* helpers in sequence;
        quaternion and rotation-matrix operations are inlined to avoid the
        temporary array allocations those helpers create.
        """
        # ── d_position = velocity ────────────────────────────────────────────
        out[0] = state[7]
        out[1] = state[8]
        out[2] = state[9]

        # ── d_quaternion = ½ Ω(ω) q  (Ω matrix inlined — no 4×4 allocation) ─
        #
        # Ω(ω) = [[ 0, -wx, -wy, -wz],
        #          [wx,   0,  wz, -wy],
        #          [wy, -wz,   0,  wx],
        #          [wz,  wy, -wx,   0]]
        ox = state[10];  oy = state[11];  oz = state[12]
        qw = state[3];   qx = state[4];   qy = state[5];   qz = state[6]
        out[3] = 0.5 * (-ox*qx - oy*qy - oz*qz)
        out[4] = 0.5 * ( ox*qw + oz*qy - oy*qz)
        out[5] = 0.5 * ( oy*qw - oz*qx + ox*qz)
        out[6] = 0.5 * ( oz*qw + oy*qx - ox*qy)

        # ── d_velocity = (R(q) @ F_body + F_inertial) / m  ──────────────────
        #
        # R is the standard quaternion → rotation-matrix formula.
        # q is normalised first (matches quat_to_rotmat_np behaviour).
        inv_n = 1.0 / np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        w = qw*inv_n;  x = qx*inv_n;  y = qy*inv_n;  z = qz*inv_n
        x2 = x*x;  y2 = y*y;  z2 = z*z
        xy = x*y;  xz = x*z;  yz = y*z
        wx_ = w*x;  wy_ = w*y;  wz_ = w*z   # w-cross products (avoid name clash with ω)

        fx = body_wrench[0];  fy = body_wrench[1];  fz = body_wrench[2]
        im = self._inv_mass
        out[7]  = ((1.0 - 2.0*(y2 + z2))*fx + 2.0*(xy - wz_)*fy + 2.0*(xz + wy_)*fz + inertial_force[0]) * im
        out[8]  = (2.0*(xy + wz_)*fx + (1.0 - 2.0*(x2 + z2))*fy + 2.0*(yz - wx_)*fz + inertial_force[1]) * im
        out[9]  = (2.0*(xz - wy_)*fx + 2.0*(yz + wx_)*fy + (1.0 - 2.0*(x2 + y2))*fz + inertial_force[2]) * im

        # ── d_angular_velocity = I⁻¹ (τ − ω × (I ω))  ──────────────────────
        np.dot(self.inertia_matrix, state[10:13], out=self._Iw)
        Iwx = self._Iw[0];  Iwy = self._Iw[1];  Iwz = self._Iw[2]
        self._tau[0] = body_wrench[3] - (oy*Iwz - oz*Iwy)
        self._tau[1] = body_wrench[4] - (oz*Iwx - ox*Iwz)
        self._tau[2] = body_wrench[5] - (ox*Iwy - oy*Iwx)
        np.dot(self._inertia_inv, self._tau, out=out[10:13])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step_forward(self, state: StateVector) -> StateVector:
        """Advance the state by one time-step using classical RK4.

        Both force callbacks are evaluated **once** at the start of the step
        and held constant across all four RK4 stages.
        """
        h  = self.step
        bw = self.get_body_wrench()
        fi = self.get_inertial_force()

        # k1 = f(state)
        self._state_derivative_into(state, bw, fi, self._k1)

        # k2 = f(state + ½h·k1)
        np.multiply(0.5 * h, self._k1, out=self._s_tmp)
        np.add(state, self._s_tmp, out=self._s_tmp)
        self._state_derivative_into(self._s_tmp, bw, fi, self._k2)

        # k3 = f(state + ½h·k2)
        np.multiply(0.5 * h, self._k2, out=self._s_tmp)
        np.add(state, self._s_tmp, out=self._s_tmp)
        self._state_derivative_into(self._s_tmp, bw, fi, self._k3)

        # k4 = f(state + h·k3)
        np.multiply(h, self._k3, out=self._s_tmp)
        np.add(state, self._s_tmp, out=self._s_tmp)
        self._state_derivative_into(self._s_tmp, bw, fi, self._k4)

        # result = state + h · (k1/6 + k2/3 + k3/3 + k4/6)
        self._K[:, 0] = self._k1
        self._K[:, 1] = self._k2
        self._K[:, 2] = self._k3
        self._K[:, 3] = self._k4
        result = np.empty(13, dtype=np.float64)
        np.dot(self._K, _RK4_WEIGHTS, out=result)
        result *= h
        result += state
        return result

    def normalize_state_quaternion(self, state: StateVector, inplace: bool = False) -> StateVector:
        """Normalize the quaternion portion of the state to combat numerical
        drift over many integration steps."""
        target = state if inplace else state.copy()

        qw, qx, qy, qz = target[3], target[4], target[5], target[6]
        norm_sq = qw*qw + qx*qx + qy*qy + qz*qz
        if norm_sq > 1e-12:
            target[3:7] *= 1.0 / np.sqrt(norm_sq)

        return target
