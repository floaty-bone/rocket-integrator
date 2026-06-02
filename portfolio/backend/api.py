"""
Portfolio backend — FastAPI simulation API.

Deploy to Render:
  - Root dir  : portfolio/backend
  - Build     : pip install -r requirements.txt
  - Start     : uvicorn api:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Generator

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
from rocket.plant.thrust import compute_thrust_forces_and_moments_cartesian_np as compute_wrench_np
from rocket.plant.tvc import cart9_to_gimbal9, gimbal9_to_cart9
from rocket.integration.integrator import RK4Integrator
from rocket.scenarios.sil_hover import (
    STEP_SIZE,
    SAMPLE_RATE,
    _make_controller,
    _make_initial_conditions,
)

app = FastAPI(title="Rocket Sim API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Upright hover quaternion [w, x, y, z] from sil_hover setpoint
_Q_SETPOINT = np.array([
    0.364186915338164, -0.606108810937832,
   -0.364186915338164, -0.606108810937832,
])

DT = STEP_SIZE * SAMPLE_RATE


def _make_frame(i: int, t: float, s: np.ndarray, u: np.ndarray, uc: np.ndarray) -> dict:
    return {
        "type": "frame", "i": i, "t": round(float(t), 5),
        "pos":  [round(float(s[0]), 5), round(float(s[1]), 5), round(float(s[2]), 5)],
        "quat": [round(float(s[3]), 7), round(float(s[4]), 7),
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
    }


def _sim_frames_infinite(
    setpoint_ref: list,        # [np.ndarray] — mutated from async task to hot-swap setpoint
    relinearize_flag: threading.Event,
    stop_event: threading.Event,
    initial_state: np.ndarray | None = None,
) -> Generator[tuple[str, dict], None, None]:
    """
    Infinite simulation generator. Runs until stop_event is set.

    Yields ('meta', dict) once at startup, then ('frame', dict) at SAMPLE_RATE cadence.
    Reads setpoint_ref[0] every control step — update it + set relinearize_flag to
    hot-swap the target without restarting.
    """
    a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE

    controller = _make_controller()
    _default_initial, _default_sp, u_nominal_cart = _make_initial_conditions()

    state    = initial_state.copy() if initial_state is not None else np.array([0.0, 0.0, 0.0, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    setpoint = setpoint_ref[0].copy()

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

    sp = setpoint_ref[0]
    yield 'meta', {
        "type": "meta", "n_frames": 0, "dt": round(float(DT), 6),
        "total_time": 86400.0,  # sentinel — sim runs indefinitely
        "setpoint": [round(float(sp[0]), 4), round(float(sp[1]), 4), round(float(sp[2]), 4)],
    }

    u_gimbal  = u_nominal_gimbal.copy()
    u_cart    = u_nominal_cart.copy()
    yield 'frame', _make_frame(0, 0.0, state, u_gimbal, u_cart)

    t_next_ctrl = 0.0
    ctrl_period = STEP_SIZE
    step_idx    = 0
    frame_idx   = 0

    while not stop_event.is_set():
        # Hot-swap setpoint when flagged by the async receiver
        if relinearize_flag.is_set():
            relinearize_flag.clear()
            setpoint = setpoint_ref[0].copy()
            controller.update_linearization(jnp.array(setpoint), jnp.array(u_nominal_cart))

        t = step_idx * STEP_SIZE
        if t >= t_next_ctrl:
            u_cart      = controller.update(state, setpoint=setpoint, u_nominal=u_nominal_cart)
            u_gimbal    = cart9_to_gimbal9(u_cart)
            t_next_ctrl += ctrl_period

        wrench_buf[0] = compute_wrench_np(gimbal9_to_cart9(u_gimbal).reshape(3, 3), a, l)
        state    = integrator.step_forward(state)
        step_idx += 1

        if step_idx % SAMPLE_RATE == 0:
            frame_idx += 1
            yield 'frame', _make_frame(frame_idx, frame_idx * DT, state, u_gimbal, u_cart)


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
    """Run a finite simulation (40 s) and return the full trajectory as JSON."""
    from rocket.scenarios.sil_hover import SIM_TIME
    stop  = threading.Event()
    relin = threading.Event()
    sp    = np.array([req.x, req.y, req.z, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sp_ref = [sp]
    meta, frames = None, []
    n_target = int(SIM_TIME / STEP_SIZE) // SAMPLE_RATE
    for kind, data in _sim_frames_infinite(sp_ref, relin, stop):
        if kind == 'meta':
            meta = data
        else:
            frames.append(data)
            if len(frames) >= n_target:
                stop.set()
                break
    return {"meta": meta, "frames": frames}


# ── WebSocket — infinite real-time stream with hot setpoint ────────────────────

@app.websocket("/ws/simulate")
async def ws_simulate(
    websocket: WebSocket,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 50.0,
    # Optional initial state (position + quaternion); velocities assumed zero
    ix: float | None = None,
    iy: float | None = None,
    iz: float | None = None,
    iqw: float | None = None,
    iqx: float | None = None,
    iqy: float | None = None,
    iqz: float | None = None,
):
    """
    Infinite real-time simulation stream.

    Outgoing (server → client): meta once, then frame at 50 Hz paced to wall clock.
    Incoming (client → server): {"type":"setpoint","x":...,"y":...,"z":...}
                                 hot-swaps the LQR target without restarting.

    Optional ix/iy/iz/iqw/iqx/iqy/iqz let the client resume from a saved state
    (e.g. after an idle disconnect) instead of starting from scratch.
    """
    await websocket.accept()

    x = max(-300.0, min(300.0, x))
    y = max(-300.0, min(300.0, y))
    z = max(5.0,    min(500.0, z))

    has_initial = all(v is not None for v in (ix, iy, iz, iqw, iqx, iqy, iqz))
    if has_initial:
        init_pos  = np.array([float(ix), float(iy), float(iz)])
        init_quat = np.array([float(iqw), float(iqx), float(iqy), float(iqz)])
        # Normalise in case of floating-point drift
        init_quat /= np.linalg.norm(init_quat)
        initial_state = np.array([*init_pos, *init_quat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        initial_state = None  # sim uses its own default

    loop             = asyncio.get_event_loop()
    frame_queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue(maxsize=8)
    stop_event       = threading.Event()
    relinearize_flag = threading.Event()
    setpoint_ref     = [np.array([x, y, z, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

    # ── Sim thread ──────────────────────────────────────────────────────────────
    def sim_worker():
        try:
            for kind, data in _sim_frames_infinite(setpoint_ref, relinearize_flag, stop_event, initial_state):
                if stop_event.is_set():
                    break
                future = asyncio.run_coroutine_threadsafe(
                    frame_queue.put((kind, data)), loop
                )
                try:
                    future.result(timeout=5)
                except Exception:
                    return  # consumer gone
        finally:
            try:
                asyncio.run_coroutine_threadsafe(frame_queue.put(None), loop).result(timeout=1)
            except Exception:
                pass

    thread = threading.Thread(target=sim_worker, daemon=True)
    thread.start()

    # ── Receive task: listen for setpoint updates from the frontend ─────────────
    async def receive_task():
        try:
            async for msg in websocket.iter_json():
                if not isinstance(msg, dict) or msg.get('type') != 'setpoint':
                    continue
                nx = max(-300.0, min(300.0, float(msg.get('x', 0))))
                ny = max(-300.0, min(300.0, float(msg.get('y', 0))))
                nz = max(5.0,    min(500.0, float(msg.get('z', 50))))
                setpoint_ref[0] = np.array([nx, ny, nz, *_Q_SETPOINT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                relinearize_flag.set()
                # Tell the frontend the setpoint was accepted (updates status + plots)
                await websocket.send_json({
                    "type": "meta", "n_frames": 0, "dt": round(float(DT), 6),
                    "total_time": 86400.0,
                    "setpoint": [round(nx, 4), round(ny, 4), round(nz, 4)],
                })
        except Exception:
            pass

    # ── Send task: pace frames to real time ─────────────────────────────────────
    async def send_task():
        start       = loop.time()
        frame_count = 0
        try:
            while True:
                item = await frame_queue.get()
                if item is None:
                    break
                kind, data = item
                await websocket.send_json(data)
                if kind == 'frame':
                    frame_count += 1
                    deadline  = start + frame_count * DT
                    remaining = deadline - loop.time()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
        except Exception:
            pass

    send_t = asyncio.create_task(send_task())
    recv_t = asyncio.create_task(receive_task())

    try:
        await send_t
    except Exception:
        pass
    finally:
        recv_t.cancel()
        stop_event.set()
        # Drain queue so the sim thread's put() unblocks and exits
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Exception:
                break
        thread.join(timeout=2)
        try:
            await websocket.close()
        except Exception:
            pass
