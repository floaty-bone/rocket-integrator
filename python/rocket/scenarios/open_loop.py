"""
open_loop.py — Open-loop dynamics simulation.

Drives the :class:`RK4Integrator` with a constant body wrench (no controller,
no feedback) and animates the resulting trajectory.  This is the original
``main.py`` scenario, ported onto the new package layout.

Run with::

    python -m rocket.scenarios.open_loop
"""

from __future__ import annotations

import numpy as np

from rocket.config.vehicle import (
    GRAVITY_FORCE,
    INERTIA_MATRIX,
    VEHICLE_MASS,
)
from rocket.integration.integrator import RK4Integrator
from rocket.plant.thrust import compute_thrust_forces_and_moments_jax
from rocket.viz.animation import animate


# =============================================================================
# CONFIG — edit these values to change the simulation scenario
# =============================================================================

# Body-frame wrench [Fx, Fy, Fz, Mx, My, Mz] from actuators (thrust, TVC, etc.)
# Max thrust sea-level Raptor engine: 2.8 MN, min thrust: 40 % of max.
BODY_WRENCH = compute_thrust_forces_and_moments_jax(  # 30 % of max thrust × 3 engines
    engine_thrust=np.array([
        [3.14, 0.4, 0.30 * 2.8e6],
        [3.14, 0.4, 0.30 * 2.8e6],
        [3.14, 0.4, 0.30 * 2.8e6],
    ], dtype=np.float64),
    a=1.5,
    l=18,
)

# Initial state  [x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
INITIAL_STATE = np.array(
    [0, 0, 0,  1, 0, 0, 0,  0, 0, -50,  0.0, 0.0, 0.0],
    dtype=np.float64,
)

SIM_TIME    = 5      # total duration (s)
STEP_SIZE   = 0.001  # RK4 time-step (s)
SAMPLE_RATE = 50     # save every Nth integration step (controls animation resolution)

# Output: set to a path (e.g. "rocket.mp4") to render at real-time speed
# instead of showing the live interactive window.
OUTPUT_FILE: str | None = None


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation() -> np.ndarray:
    """Run the RK4 integrator and return the sampled trajectory.

    Returns:
        Array of shape ``(n_frames, 13)`` containing the sampled state vectors.
    """
    integrator = RK4Integrator(
        get_body_wrench=lambda: BODY_WRENCH,
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        mass=VEHICLE_MASS,
        step_size=STEP_SIZE,
    )

    n_steps = int(SIM_TIME / STEP_SIZE)
    n_frames = n_steps // SAMPLE_RATE + 1

    trajectory = np.empty((n_frames, 13), dtype=np.float64)
    state = INITIAL_STATE.copy()
    trajectory[0] = state

    report_every = n_steps // 10
    frame_idx = 1
    for step_idx in range(1, n_steps):
        state = integrator.step_forward(state)
        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx] = state
            frame_idx += 1
        if report_every and step_idx % report_every == 0:
            print(f"  {step_idx * 100 // n_steps}%…", flush=True)

    return trajectory[:frame_idx]


def main() -> None:
    print("Running simulation…")
    trajectory = run_simulation()

    q_final = trajectory[-1, 3:7]
    print(f"  Frames generated      : {len(trajectory)}")
    print(f"  Final quaternion norm : {np.linalg.norm(q_final):.6f}  (should be ≈ 1)")
    print(f"  Final state           : {trajectory[-1]}")

    print("Launching animation…")
    animate(
        trajectory,
        sample_rate=SAMPLE_RATE,
        step_size=STEP_SIZE,
        sim_time=SIM_TIME,
        output_file=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
