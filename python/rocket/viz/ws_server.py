"""
ws_server.py — WebSocket server that replays a recorded trajectory in real time.

The server runs the full SIL simulation first, then streams frames to any
connected browser client at the correct wall-clock pace.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np


async def serve_trajectory(
    trajectory: np.ndarray,
    u_history: np.ndarray,
    step_size: float,
    sample_rate: int,
    setpoint: np.ndarray | None = None,
    ucart_history: np.ndarray | None = None,
    host: str = "localhost",
    port: int = 8765,
) -> None:
    """Stream a recorded trajectory to a WebSocket client at real-time speed.

    Args:
        trajectory:     ``(n_frames, 13)`` state vectors sampled from the sim.
        u_history:      ``(n_frames, 9)`` gimbal commands ``[α, β, T] × 3 engines``.
        step_size:      Integration step size (s).
        sample_rate:    Steps between recorded frames.
        ucart_history:  ``(n_frames, 9)`` raw Cartesian thrust ``[Fx, Fy, Fz] × 3``.
        host:           WebSocket bind address.
        port:           WebSocket port.
    """
    import websockets  # optional dep — imported here so the rest of the module loads without it

    dt = step_size * sample_rate
    n_frames = len(trajectory)
    total_time = round(n_frames * dt, 3)

    async def handler(websocket):
        addr = getattr(websocket, "remote_address", "?")
        print(f"  Client connected: {addr}", flush=True)

        sp = setpoint[:3].tolist() if setpoint is not None else [0.0, 0.0, 0.0]
        meta = {
            "type": "meta",
            "n_frames": n_frames,
            "dt": round(dt, 6),
            "total_time": total_time,
            "setpoint": [round(float(v), 4) for v in sp],
        }
        await websocket.send(json.dumps(meta))

        # Sanity-check first frame thrust so misconfigured sims surface quickly.
        u0 = u_history[0]
        print(f"  First-frame thrust values: {float(u0[2]):.1f}  {float(u0[5]):.1f}  {float(u0[8]):.1f} N", flush=True)

        loop = asyncio.get_event_loop()
        start = loop.time()

        for i in range(n_frames):
            state = trajectory[i]
            u = u_history[i]
            frame = {
                "type": "frame",
                "i": i,
                "t": round(i * dt, 5),
                "pos":  [round(float(state[0]), 5), round(float(state[1]), 5), round(float(state[2]), 5)],
                "quat": [round(float(state[3]), 7), round(float(state[4]), 7),
                         round(float(state[5]), 7), round(float(state[6]), 7)],
                # engines: [[α0, β0, T0], [α1, β1, T1], [α2, β2, T2]] — rad, rad, N
                "engines": [
                    [round(float(u[0]), 6), round(float(u[1]), 6), round(float(u[2]), 1)],
                    [round(float(u[3]), 6), round(float(u[4]), 6), round(float(u[5]), 1)],
                    [round(float(u[6]), 6), round(float(u[7]), 6), round(float(u[8]), 1)],
                ],
                "omega": [round(float(state[10]), 6), round(float(state[11]), 6), round(float(state[12]), 6)],
            }
            if ucart_history is not None:
                uc = ucart_history[i]
                frame["u_cart"] = [
                    [round(float(uc[0]), 1), round(float(uc[1]), 1), round(float(uc[2]), 1)],
                    [round(float(uc[3]), 1), round(float(uc[4]), 1), round(float(uc[5]), 1)],
                    [round(float(uc[6]), 1), round(float(uc[7]), 1), round(float(uc[8]), 1)],
                ]
            await websocket.send(json.dumps(frame))

            # Sleep only the remaining time until the next frame deadline so
            # JSON encoding + send latency don't accumulate into timing drift.
            next_deadline = start + (i + 1) * dt
            remaining = next_deadline - loop.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

        print("  Playback complete.", flush=True)

    async with websockets.serve(handler, host, port):
        print(f"\nWebSocket server ready — ws://{host}:{port}", flush=True)
        print(f"  {n_frames} frames  ·  dt = {dt:.4f} s  ·  total = {total_time:.1f} s", flush=True)
        print("Open the renderer and press P to start playback.\n", flush=True)
        await asyncio.Future()  # run until interrupted
