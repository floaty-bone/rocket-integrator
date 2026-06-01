"""Throwaway: compare attitude convergence of old hand-tuned vs new Bryson Q/R.

Reuses the real sil_hover plant/controller blocks but at reduced integration
rates (the ~0.5 rad/s closed-loop response is unaffected by integrator Hz).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rocket.config.vehicle import (
    COM_TO_ENGINE_PLANE, ENGINE_CLUSTER_RADIUS, GRAVITY_FORCE,
    INERTIA_MATRIX, VEHICLE_MASS, hover_thrust_per_engine,
)
from rocket.control.lqr_controller import LQRController
from rocket.plant.dynamics import make_F
from rocket.plant.thrust import compute_thrust_forces_and_moments_cartesian_np as wrench_np
from rocket.plant.tvc import cart9_to_gimbal9, gimbal9_to_cart9
from rocket.integration.integrator import RK4Integrator
from rocket.scenarios.sil_hover import _make_initial_conditions, controller_block, plant_block

SIM_TIME, SIM_FREQ, CONTROL_FREQ, LIN_RATE = 40.0, 4000, 2000, 80
STEP = 1.0 / SIM_FREQ
SAMPLE = int(SIM_FREQ / 50)

a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE
F = make_F(mass=VEHICLE_MASS, inertia_matrix=INERTIA_MATRIX,
           gravity_force=GRAVITY_FORCE, a=a, l=l)

Q_OLD = np.diag([2e7, 2e7, 1e7, 1e9, 1e9, 1e9, 2e8, 3e8, 3e8, 1, 1, 1])
R_OLD = np.diag([1.0] * 9)

RHO, THR, P, A, V, W = 1.0, 1.0e5, 5.0, 0.2, 5.0, 0.3
tol = np.array([P] * 3 + [A] * 3 + [V] * 3 + [W] * 3)
Q_NEW = RHO * np.diag((THR / tol) ** 2)
R_NEW = np.eye(9)


def att_err_deg(q_state, q_set):
    d = abs(float(np.dot(q_state, q_set)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def run(Q, R, label):
    controller = LQRController(F, Q=Q, R=R)
    x0, setpoint, u_nom_cart = _make_initial_conditions()
    u_nom_gimbal = cart9_to_gimbal9(u_nom_cart)
    controller.update_linearization(jnp.array(x0), jnp.array(u_nom_cart))

    wrench_buf = [wrench_np(gimbal9_to_cart9(u_nom_gimbal).reshape(3, 3), a, l)]
    integ = RK4Integrator(get_body_wrench=lambda: wrench_buf[0],
                          get_inertial_force=lambda: GRAVITY_FORCE,
                          inertia_matrix=INERTIA_MATRIX, mass=VEHICLE_MASS, step_size=STEP)

    n_steps = int(SIM_TIME / STEP)
    lin_steps = max(1, int(1.0 / (STEP * LIN_RATE)))
    state, u_gimbal = x0.copy(), u_nom_gimbal.copy()
    t_next, period = 0.0, 1.0 / CONTROL_FREQ

    times, errs, rates, lat = [], [], [], []
    for i in range(1, n_steps):
        t = i * STEP
        if t >= t_next:
            u_gimbal, u_cart = controller_block(state, setpoint, u_gimbal, i,
                                                controller, lin_steps, u_nom_cart)
            t_next += period
        state = plant_block(state, u_gimbal, wrench_buf, integ, a, l)
        if i % SAMPLE == 0:
            times.append(t)
            errs.append(att_err_deg(state[3:7], setpoint[3:7]))
            rates.append(float(np.linalg.norm(state[10:13])))
            uc = u_cart.reshape(3, 3)
            lat.append(float(np.max(np.hypot(uc[:, 1], uc[:, 2]))))

    times, errs = np.array(times), np.array(errs)
    thresh = 2.0  # degrees
    above = np.where(errs > thresh)[0]
    settle = times[above[-1]] if len(above) else 0.0
    print(f"\n=== {label} ===")
    print(f"  initial att err : {errs[0]:8.2f} deg")
    print(f"  final   att err : {errs[-1]:8.3f} deg")
    print(f"  peak    att err : {errs.max():8.2f} deg")
    print(f"  settle (<{thresh:g} deg): {settle:6.2f} s")
    print(f"  peak body rate  : {max(rates):8.3f} rad/s")
    print(f"  peak lateral F  : {max(lat) / 1e3:8.1f} kN/engine")
    return times, errs


if __name__ == "__main__":
    print("Running OLD hand-tuned weights...")
    run(Q_OLD, R_OLD, "OLD  Q=[2e7.. 1e9.. 1..]")
    print("\nRunning NEW Bryson weights...")
    run(Q_NEW, R_NEW, "NEW  Bryson (RHO=1)")
