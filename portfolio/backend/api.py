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

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# Import scenario constants and controller factory from sil_hover
from rocket.scenarios.sil_hover import (
    SIM_TIME,
    STEP_SIZE,
    SAMPLE_RATE,
    _make_controller,
    _make_initial_conditions,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Rocket Sim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Shared simulation logic ────────────────────────────────────────────────────

# Attitude setpoint quaternion [w, x, y, z] — upright hover orientation
_Q_SETPOINT = np.array([
    0.364186915338164, -0.606108810937832,
   -0.364186915338164, -0.606108810937832,
])


def _run_simulation(x: float, y: float, z: float) -> tuple[dict, list[dict]]:
    """
    Run the full closed-loop SIL simulation with the given position setpoint.

    Uses sil_hover initial conditions (rocket descending from altitude) so the
    animation shows the vehicle flying in, correcting attitude, and landing.

    Returns (meta_dict, frame_list) — ready to JSON-encode.
    """
    a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE

    controller = _make_controller()

    # Pull initial state from sil_hover; override setpoint position with user values.
    initial_state, _default_sp, u_nominal_cart = _make_initial_conditions()
    setpoint = np.array([x, y, z, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    ctrl_period = STEP_SIZE
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

    meta = {
        "type":       "meta",
        "n_frames":   frame_idx,
        "dt":         round(float(dt), 6),
        "total_time": total_time,
        "setpoint":   [round(float(x), 4), round(float(y), 4), round(float(z), 4)],
    }

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
            "omega": [round(float(s[10]), 6), round(float(s[11]), 6), round(float(s[12]), 6)],
            "u_cart": [
                [round(float(uc[0]), 1), round(float(uc[1]), 1), round(float(uc[2]), 1)],
                [round(float(uc[3]), 1), round(float(uc[4]), 1), round(float(uc[5]), 1)],
                [round(float(uc[6]), 1), round(float(uc[7]), 1), round(float(uc[8]), 1)],
            ],
        })

    return meta, frames


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok"}


class SetpointRequest(BaseModel):
    x: float = Field(default=0.0, ge=-300.0, le=300.0)
    y: float = Field(default=0.0, ge=-300.0, le=300.0)
    z: float = Field(default=50.0, ge=5.0,   le=500.0)


@app.post("/simulate")
def simulate(req: SetpointRequest):
    """Run simulation and return complete trajectory as JSON."""
    meta, frames = _run_simulation(req.x, req.y, req.z)
    return {"meta": meta, "frames": frames}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/simulate")
async def ws_simulate(
    websocket: WebSocket,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 50.0,
):
    """
    Stream simulation frames in real time over WebSocket.

    Connect to: ws://<host>/ws/simulate?x=0&y=0&z=50

    Protocol (identical to ws_server.py):
      1. Server sends meta JSON  {"type":"meta", "n_frames":..., "dt":..., ...}
      2. Server sends one frame JSON per sample, paced to wall-clock real time.
    """
    await websocket.accept()

    # Clamp to valid ranges
    x = max(-300.0, min(300.0, x))
    y = max(-300.0, min(300.0, y))
    z = max(5.0,    min(500.0, z))

    # Run CPU-bound simulation in a thread so the event loop stays responsive
    loop = asyncio.get_event_loop()
    try:
        meta, frames = await loop.run_in_executor(None, _run_simulation, x, y, z)
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc))
        return

    dt = meta["dt"]

    try:
        await websocket.send_json(meta)

        start = loop.time()
        for i, frame in enumerate(frames):
            await websocket.send_json(frame)

            # Pace delivery to match real time so the frontend animation is smooth
            next_deadline = start + (i + 1) * dt
            remaining = next_deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
