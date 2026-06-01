"""
Portfolio backend — FastAPI simulation API.

Deploy to Render:
  - Root dir  : portfolio/backend
  - Build     : pip install -r requirements.txt
  - Start     : uvicorn api:app --host 0.0.0.0 --port $PORT

The rocket package lives at <repo>/python/rocket/.
sys.path is patched below so it's importable from any working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# <repo>/python/ — works regardless of CWD (important for Render)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# ── Simulation constants ───────────────────────────────────────────────────────
SIM_TIME    = 20.0        # shorter than sil_hover to fit free-tier request limits
SIM_FREQ    = 5000        # Hz
STEP_SIZE   = 1.0 / SIM_FREQ
SAMPLE_RATE = int(SIM_FREQ / 50)   # 50 Hz output frames

# Attitude setpoint quaternion [w, x, y, z] — upright hover orientation
_Q_SETPOINT = np.array([
    0.364186915338164, -0.606108810937832,
   -0.364186915338164, -0.606108810937832,
])

# Initial attitude — tilted (same as sil_hover for a visible correction manoeuvre)
_Q_INITIAL = np.array([
    0.295413703592012, -0.702150346667812,
   -0.421894491949238, -0.491650965693363,
])

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Rocket Sim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SetpointRequest(BaseModel):
    x: float = Field(default=0.0, ge=-300.0, le=300.0, description="Target X position (m)")
    y: float = Field(default=0.0, ge=-300.0, le=300.0, description="Target Y position (m)")
    z: float = Field(default=50.0, ge=5.0,   le=500.0, description="Target altitude (m)")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(req: SetpointRequest):
    """
    Run a closed-loop LQR hover simulation and return the full trajectory.

    Initial position is [0, 0, 0] with a tilted attitude.
    The controller drives the vehicle to the requested [x, y, z] setpoint.
    """
    a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE

    # Controller
    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
        a=a, l=l,
    )
    Q = np.diag([
        1e8,  1e8,  0.9e8,
        1e12, 1e12, 1e12,
        1.7e8, 1.7e8, 3.6e9,
        1e13, 1e13, 1e13,
    ])
    R = np.diag([4, 10, 10, 4, 10, 10, 4, 10, 10])
    controller = LQRController(F, Q=Q, R=R)

    # Initial state: position [0,0,0], tilted attitude, zero velocities
    initial_state = np.array([0.0, 0.0, 0.0, *_Q_INITIAL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    setpoint      = np.array([req.x, req.y, req.z, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    hover = hover_thrust_per_engine()
    u_nominal_cart   = np.array([hover, 0, 0, hover, 0, 0, hover, 0, 0])
    u_nominal_gimbal = cart9_to_gimbal9(u_nominal_cart)

    controller.update_linearization(jnp.array(setpoint), jnp.array(u_nominal_cart))

    wrench_buf = [compute_wrench_np(gimbal9_to_cart9(u_nominal_gimbal).reshape(3, 3), a, l)]
    integrator = RK4Integrator(
        get_body_wrench=lambda: wrench_buf[0],
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        mass=VEHICLE_MASS,
        step_size=STEP_SIZE,
    )

    n_steps  = int(SIM_TIME / STEP_SIZE)
    n_frames = n_steps // SAMPLE_RATE + 1

    trajectory    = np.empty((n_frames, 13))
    u_history     = np.empty((n_frames, 9))
    ucart_history = np.empty((n_frames, 9))

    state    = initial_state.copy()
    u_gimbal = u_nominal_gimbal.copy()
    u_cart   = u_nominal_cart.copy()
    trajectory[0], u_history[0], ucart_history[0] = state, u_gimbal, u_cart

    t_next_ctrl = 0.0
    ctrl_period = STEP_SIZE   # CONTROL_FREQ == SIM_FREQ
    frame_idx   = 1

    for step_idx in range(1, n_steps):
        t = step_idx * STEP_SIZE

        if t >= t_next_ctrl:
            u_cart   = controller.update(state, setpoint=setpoint, u_nominal=u_nominal_cart)
            u_gimbal = cart9_to_gimbal9(u_cart)
            t_next_ctrl += ctrl_period

        wrench_buf[0] = compute_wrench_np(gimbal9_to_cart9(u_gimbal).reshape(3, 3), a, l)
        state = integrator.step_forward(state)

        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx], u_history[frame_idx], ucart_history[frame_idx] = state, u_gimbal, u_cart
            frame_idx += 1

    trajectory    = trajectory[:frame_idx]
    u_history     = u_history[:frame_idx]
    ucart_history = ucart_history[:frame_idx]

    dt         = STEP_SIZE * SAMPLE_RATE
    total_time = round(frame_idx * dt, 3)

    frames = []
    for i in range(frame_idx):
        s  = trajectory[i]
        u  = u_history[i]
        uc = ucart_history[i]
        frames.append({
            "type":    "frame",
            "i":       i,
            "t":       round(float(i * dt), 5),
            "pos":     [round(float(s[0]), 5), round(float(s[1]), 5), round(float(s[2]), 5)],
            "quat":    [round(float(s[3]), 7), round(float(s[4]), 7),
                        round(float(s[5]), 7), round(float(s[6]), 7)],
            "engines": [
                [round(float(u[0]), 6), round(float(u[1]), 6), round(float(u[2]), 1)],
                [round(float(u[3]), 6), round(float(u[4]), 6), round(float(u[5]), 1)],
                [round(float(u[6]), 6), round(float(u[7]), 6), round(float(u[8]), 1)],
            ],
            "omega":  [round(float(s[10]), 6), round(float(s[11]), 6), round(float(s[12]), 6)],
            "u_cart": [
                [round(float(uc[0]), 1), round(float(uc[1]), 1), round(float(uc[2]), 1)],
                [round(float(uc[3]), 1), round(float(uc[4]), 1), round(float(uc[5]), 1)],
                [round(float(uc[6]), 1), round(float(uc[7]), 1), round(float(uc[8]), 1)],
            ],
        })

    return {
        "meta": {
            "type":       "meta",
            "n_frames":   frame_idx,
            "dt":         round(float(dt), 6),
            "total_time": total_time,
            "setpoint":   [round(float(req.x), 4), round(float(req.y), 4), round(float(req.z), 4)],
        },
        "frames": frames,
    }
