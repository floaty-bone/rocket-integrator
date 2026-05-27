"""
lqr_sanity_check.py — One-shot LQR design sanity check.

Builds a :class:`LQRController` around the Starship reference vehicle, linearises
at a vertical-attitude setpoint with a hover thrust profile, evaluates the
control output for a small lateral-velocity perturbation, and prints the
resulting per-engine gimbal command in human-readable form.

Run with::

    python -m rocket.scenarios.lqr_sanity_check
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rocket.config.vehicle import GRAVITY_FORCE, INERTIA_MATRIX, VEHICLE_MASS
from rocket.control.lqr_controller import LQRController
from rocket.plant.dynamics import make_F
from rocket.plant.thrust import body_force_to_spherical_jax


def run_sanity_check() -> None:
    print("Initializing dynamics...")
    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
    )

    print("Initializing LQR Controller...")
    # Crank these scalars to see their effect on the control output.
    Q_scalar = 5.0
    R_scalar = 1.0

    # Q has 13 elements (pos, quat, vel, ang_vel).
    # Crank position penalty (first 3) to 1e6 to actually move the 120-t rocket.
    Q_diag = np.array([1e6, 1e6, 1e6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R_diag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    Q = np.diag(Q_diag) * Q_scalar
    R = np.diag(R_diag) * R_scalar

    controller = LQRController(F, Q=Q, R=R)

    # Setpoint: vertical rocket — rotation around body Y by −90°.
    theta = -math.pi / 2
    qw = math.cos(theta / 2)
    qy = math.sin(theta / 2)

    # State: upright rocket with a small lateral-velocity perturbation.
    state    = jnp.array([0, 0, 0,  qw, 0, qy, 0,  0, -20, 0,  0, 0, 0], dtype=jnp.float64)
    setpoint = jnp.array([0, 0, 0,  qw, 0, qy, 0,  0,   0, 0,  0, 0, 0], dtype=jnp.float64)

    # Nominal thrust: hover thrust against gravity (slightly above to feel margin).
    hover_thrust = (VEHICLE_MASS * 10) / 3.0
    u_nominal = jnp.array(
        [hover_thrust, 0, 0, hover_thrust, 0, 0, hover_thrust, 0, 0],
        dtype=jnp.float64,
    )

    print("Linearizing dynamics around the vertical setpoint (this may take a few seconds to compile)...")
    controller.update_linearization(setpoint, u_nominal)

    print("Computing control output U...")
    U = controller.update(state, setpoint=setpoint, u_nominal=u_nominal)

    np.set_printoptions(precision=4, suppress=True)
    print("\n" + "=" * 50)
    print("               SANITY CHECK RESULTS")
    print("=" * 50)
    print("Current State (Flat):      ", np.array(state))
    print("Setpoint (Vertical, -90Y): ", np.array(setpoint))
    print("-" * 50)
    print("Control Output U (Flattened 9x1):")
    print(np.array(U))
    print("\nControl Output U (Reshaped to 3x3 for the 3 engines):")
    U_reshaped = np.array(U).reshape(3, 3)
    print(U_reshaped)

    print("\nControl Output (Spherical/Gimbal Coordinates):")
    for i in range(3):
        spherical = body_force_to_spherical_jax(U_reshaped[i])
        theta_deg = math.degrees(spherical[0])
        phi_deg   = math.degrees(spherical[1])
        thrust    = spherical[2]
        print(
            f"  Engine {i + 1}: Gimbal Theta = {theta_deg:8.2f}°,  "
            f"Phi = {phi_deg:8.2f}°,  Thrust = {thrust:10.1f} N"
        )
    print("=" * 50)


if __name__ == "__main__":
    run_sanity_check()
