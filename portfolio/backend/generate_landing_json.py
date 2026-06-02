"""
Generate landing.json — pre-baked trajectory for the default setpoint (x=0, y=0, z=50).
Saves to portfolio/frontend/public/landing.json so the frontend can play it without a
WebSocket connection.

Run from the repo root:
    python portfolio/backend/generate_landing_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

import numpy as np

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
    SIM_TIME,
    _make_controller,
    _make_initial_conditions,
)
import jax.numpy as jnp

_Q_SETPOINT = np.array([
    0.364186915338164, -0.606108810937832,
   -0.364186915338164, -0.606108810937832,
])

DT = STEP_SIZE * SAMPLE_RATE


def generate() -> dict:
    a, l = ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE

    controller = _make_controller()
    initial_state, setpoint, u_nominal_cart = _make_initial_conditions()

    state = initial_state.copy()
    x, y, z = setpoint[0], setpoint[1], setpoint[2]

    controller.update_linearization(jnp.array(setpoint), jnp.array(u_nominal_cart))

    u_nominal_gimbal = cart9_to_gimbal9(u_nominal_cart)
    wrench_buf = [compute_wrench_np(gimbal9_to_cart9(u_nominal_gimbal).reshape(3, 3), a, l)]
    integrator = RK4Integrator(
        get_body_wrench=lambda: wrench_buf[0],
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        mass=VEHICLE_MASS,
        step_size=STEP_SIZE,
    )

    n_target = int(SIM_TIME / STEP_SIZE) // SAMPLE_RATE

    def make_frame(i, t, s, u_g, u_c):
        return {
            "type": "frame", "i": i, "t": round(float(t), 5),
            "pos":  [round(float(s[0]), 5), round(float(s[1]), 5), round(float(s[2]), 5)],
            "quat": [round(float(s[3]), 7), round(float(s[4]), 7),
                     round(float(s[5]), 7), round(float(s[6]), 7)],
            "engines": [
                [round(float(u_g[0]), 6), round(float(u_g[1]), 6), round(float(u_g[2]), 1)],
                [round(float(u_g[3]), 6), round(float(u_g[4]), 6), round(float(u_g[5]), 1)],
                [round(float(u_g[6]), 6), round(float(u_g[7]), 6), round(float(u_g[8]), 1)],
            ],
            "omega": [round(float(s[10]), 6), round(float(s[11]), 6), round(float(s[12]), 6)],
            "u_cart": [
                [round(float(u_c[0]), 1), round(float(u_c[1]), 1), round(float(u_c[2]), 1)],
                [round(float(u_c[3]), 1), round(float(u_c[4]), 1), round(float(u_c[5]), 1)],
                [round(float(u_c[6]), 1), round(float(u_c[7]), 1), round(float(u_c[8]), 1)],
            ],
        }

    u_gimbal = u_nominal_gimbal.copy()
    u_cart   = u_nominal_cart.copy()
    frames   = [make_frame(0, 0.0, state, u_gimbal, u_cart)]

    t_next_ctrl = 0.0
    step_idx    = 0
    frame_idx   = 0

    print(f"Generating landing.json  (setpoint x={x} y={y} z={z}, {SIM_TIME}s)…")
    while frame_idx < n_target:
        t = step_idx * STEP_SIZE
        if t >= t_next_ctrl:
            u_cart      = controller.update(state, setpoint=setpoint, u_nominal=u_nominal_cart)
            u_gimbal    = cart9_to_gimbal9(u_cart)
            t_next_ctrl += STEP_SIZE

        wrench_buf[0] = compute_wrench_np(gimbal9_to_cart9(u_gimbal).reshape(3, 3), a, l)
        state    = integrator.step_forward(state)
        step_idx += 1

        if step_idx % SAMPLE_RATE == 0:
            frame_idx += 1
            frames.append(make_frame(frame_idx, frame_idx * DT, state, u_gimbal, u_cart))
            if frame_idx % 500 == 0:
                print(f"  {frame_idx}/{n_target} frames  z={state[2]:.1f} m")

    meta = {
        "type": "meta",
        "n_frames": len(frames),
        "dt": round(float(DT), 6),
        "total_time": round(float(n_target * DT), 3),
        "setpoint": [round(x, 4), round(y, 4), round(z, 4)],
    }
    return {"meta": meta, "frames": frames}


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "frontend" / "public" / "landing.json"
    data = generate()
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024
    print(f"Written {len(data['frames'])} frames  {size_kb:.0f} KB  ->  {out_path}")
