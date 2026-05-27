"""
sil_hover.py — Software-in-the-Loop hover & translate scenario.

Closed-loop simulation:
  * Plant     :class:`RK4Integrator` driven by ``compute_wrench`` from the
    NumPy thrust model.
  * Controller :class:`LQRController` re-linearised periodically at
    ``LINEARIZATION_RATE`` Hz, evaluated at ``CONTROL_FREQ`` Hz.
  * SIL boundary uses the real TVC representation ``[α, β, T]`` per engine.

Run with::

    python -m rocket.scenarios.sil_hover
"""

from __future__ import annotations

import math
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rocket.config.vehicle import (
    COM_TO_ENGINE_PLANE,
    ENGINE_CLUSTER_RADIUS,
    GRAVITY_FORCE,
    INERTIA_MATRIX,
    VEHICLE_MASS,
    hover_thrust_per_engine,
)
from rocket.control.lqr_controller import LQRController
from rocket.plant.dynamics import make_F
from rocket.plant.thrust import compute_thrust_forces_and_moments_cartesian_np as compute_wrench_np
from rocket.plant.tvc import cart9_to_gimbal9, gimbal9_to_cart9
from rocket.integration.integrator import RK4Integrator
from rocket.viz.plots import plot_thrust, plot_trajectory_tracking


# =============================================================================
# CONFIG
# =============================================================================
SIM_TIME           = 40.0
SIM_FREQ           = 8000   # Hz
CONTROL_FREQ       = 5000   # Hz
LINEARIZATION_RATE = 30     # Hz
STEP_SIZE          = 1.0 / SIM_FREQ
SAMPLE_RATE        = int(SIM_FREQ / 50)  # record at 50 Hz


# =============================================================================
# CONTROLLER BLOCK
#   IN  : state [13], setpoint [13]
#   OUT : u_gimbal [9] — [alpha, beta, T] per engine  ← SIL boundary
# =============================================================================
def controller_block(
    state: np.ndarray,
    setpoint: np.ndarray,
    u_gimbal_prev: np.ndarray,
    step_idx: int,
    controller: LQRController,
    lin_steps: int,
    u_nominal_cart: np.ndarray,
) -> np.ndarray:
    if step_idx % lin_steps == 0:
        controller.update_linearization(
            jnp.array(state),
            jnp.array(gimbal9_to_cart9(u_gimbal_prev)),
        )
    u_cart = controller.update(state, setpoint=setpoint, u_nominal=u_nominal_cart)
    return cart9_to_gimbal9(u_cart), u_cart


# =============================================================================
# PLANT BLOCK
#   IN  : state [13], u_gimbal [9] — [alpha, beta, T] per engine  ← SIL boundary
#   OUT : state [13]
# =============================================================================
def plant_block(
    state: np.ndarray,
    u_gimbal: np.ndarray,
    wrench_buf: list,
    integrator: RK4Integrator,
    a: float,
    l: float,
) -> np.ndarray:
    wrench_buf[0] = compute_wrench_np(gimbal9_to_cart9(u_gimbal).reshape(3, 3), a, l)
    return integrator.step_forward(state)


# =============================================================================
# VEHICLE SETUP
# =============================================================================

def _make_controller():
    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
        a=ENGINE_CLUSTER_RADIUS,
        l=COM_TO_ENGINE_PLANE,
    )
    Q = np.diag([2.4e7, 2.4e7, 1e7, 0.1, 0.1, 0.1, 4e8, 4e8, 2.5e8, 0.1, 0.1, 0.1])
    R = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1])
    return LQRController(F, Q=Q, R=R)


def _make_initial_conditions():
    initial_pitch_deg = -90
    half = math.radians(initial_pitch_deg) / 2.0
    qwi, qxi, qyi, qzi = math.cos(half), 0.0, math.sin(half), 0.0

    setpoint_pitch_deg=-90
    setpoint_half = math.radians(setpoint_pitch_deg) / 2.0
    qw, qx, qy, qz =math.cos(setpoint_half),0.0,math.sin(setpoint_half), 0.0

    initial_state = np.array([70.0, 120.0, 600.0, qwi, qxi, qyi, qzi, 0.0, 0.0, -70.0, 0.0, 0.0, 0.0])
    setpoint      = np.array([0.0, 0.0, 49.0,   qw, qx, qy, qz, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0])
    hover         = hover_thrust_per_engine()
    u_nominal_cart = np.array([hover, 0, 0, hover, 0, 0, hover, 0, 0])
    return initial_state, setpoint, u_nominal_cart


# =============================================================================
# SIMULATION
# =============================================================================

def run_sil_simulation():
    a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE

    print("Initializing LQR Controller...", flush=True)
    controller = _make_controller()

    initial_state, setpoint, u_nominal_cart = _make_initial_conditions()
    u_nominal_gimbal = cart9_to_gimbal9(u_nominal_cart)

    print("Performing initial linearization...", flush=True)
    controller.update_linearization(jnp.array(initial_state), jnp.array(u_nominal_cart))

    # -- Integrator --
    wrench_buf = [compute_wrench_np(gimbal9_to_cart9(u_nominal_gimbal).reshape(3, 3), a, l)]

    integrator = RK4Integrator(
        get_body_wrench=lambda: wrench_buf[0],
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        mass=VEHICLE_MASS,
        step_size=STEP_SIZE,
    )

    # -- Buffers --
    n_steps   = int(SIM_TIME / STEP_SIZE)
    lin_steps = int(1.0 / (STEP_SIZE * LINEARIZATION_RATE))
    n_frames  = n_steps // SAMPLE_RATE + 1
    trajectory    = np.empty((n_frames, 13))
    u_history     = np.empty((n_frames, 9))
    ucart_history = np.empty((n_frames, 9))

    state    = initial_state.copy()
    u_gimbal = u_nominal_gimbal.copy()
    u_cart   = u_nominal_cart.copy()
    trajectory[0], u_history[0], ucart_history[0] = state, u_gimbal, u_cart

    # -- Loop --
    print(f"\nRunning SIL simulation for {SIM_TIME}s...", flush=True)
    start_time    = time.time()
    t_next_ctrl   = 0.0
    ctrl_period   = 1.0 / CONTROL_FREQ
    ctrl_count    = 0
    frame_idx     = 1
    report_every  = n_steps // 10
    ctrl_time     = 0.0
    plant_time    = 0.0

    for step_idx in range(1, n_steps):
        t = step_idx * STEP_SIZE

        # ── CONTROLLER BLOCK (CONTROL_FREQ) ──────────────────────────────────
        if t >= t_next_ctrl:
            _t0 = time.perf_counter()
            u_gimbal, u_cart = controller_block(
                state, setpoint, u_gimbal, step_idx, controller, lin_steps, u_nominal_cart,
            )
            ctrl_time  += time.perf_counter() - _t0
            t_next_ctrl += ctrl_period
            ctrl_count  += 1

        # ── SIL BOUNDARY: [α, β, T] → plant ──────────────────────────────────
        _t0 = time.perf_counter()
        state = plant_block(state, u_gimbal, wrench_buf, integrator, a, l)
        plant_time += time.perf_counter() - _t0

        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx], u_history[frame_idx], ucart_history[frame_idx] = state, u_gimbal, u_cart
            frame_idx += 1

        if step_idx % report_every == 0:
            print(f"  {step_idx * 100 // n_steps}% complete...", flush=True)

    elapsed      = time.time() - start_time
    effective_hz = n_steps / elapsed
    rt_ratio     = SIM_TIME / elapsed

    np.set_printoptions(precision=4, suppress=True)
    other_time = elapsed - ctrl_time - plant_time

    print(f"\n{'─'*44}", flush=True)
    print(f"  Sim time          : {SIM_TIME:.1f} s")
    print(f"  Wall time         : {elapsed:.3f} s")
    print(f"  Effective sim rate: {effective_hz:,.0f} Hz  (target {SIM_FREQ} Hz)")
    print(f"  Real-time ratio   : {rt_ratio:.2f}x  ({'faster' if rt_ratio > 1 else 'slower'} than real-time)")
    print(f"  Control updates   : {ctrl_count:,}  ({ctrl_count / SIM_TIME:.0f} Hz)")
    print(f"{'─'*44}")
    print(f"  Time breakdown:")
    print(f"    Plant (RK4)     : {plant_time:.3f} s  ({100*plant_time/elapsed:.1f}%)")
    print(f"    Controller      : {ctrl_time:.3f} s  ({100*ctrl_time/elapsed:.1f}%)")
    print(f"    Other           : {other_time:.3f} s  ({100*other_time/elapsed:.1f}%)")
    print(f"{'─'*44}")
    print(f"  Final position    : {state[0:3]}")
    print(f"  Target position   : {setpoint[0:3]}")
    print(f"  Quat norm         : {np.linalg.norm(state[3:7]):.8f}  (should be ≈ 1)")
    print(f"{'─'*44}", flush=True)

    return trajectory[:frame_idx], u_history[:frame_idx], ucart_history[:frame_idx], STEP_SIZE, SAMPLE_RATE, SIM_TIME, setpoint


def main() -> None:
    import asyncio
    from rocket.viz.ws_server import serve_trajectory

    trajectory, u_history, ucart_history, dt, sample, sim_time, setpoint = run_sil_simulation()

    print("Generating plots...", flush=True)
    plot_trajectory_tracking(trajectory, dt, sample, sim_time, setpoint)
    plot_thrust(u_history, dt, sample, sim_time)

    print("Starting WebSocket server...", flush=True)
    asyncio.run(serve_trajectory(trajectory, u_history, dt, sample, setpoint=setpoint, ucart_history=ucart_history))


if __name__ == "__main__":
    main()
