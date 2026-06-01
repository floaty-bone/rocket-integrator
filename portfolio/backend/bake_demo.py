"""
bake_demo.py — Pre-compute the default sil_hover trajectory and write demo.json.

Run once before building the frontend:
    python portfolio/backend/bake_demo.py

Output: portfolio/frontend/public/demo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

from rocket.scenarios.sil_hover import run_sil_simulation


def main() -> None:
    print("Running sil_hover simulation (this takes a few seconds)...")
    trajectory, u_history, ucart_history, step_size, sample_rate, sim_time, setpoint = (
        run_sil_simulation()
    )

    n_frames  = len(trajectory)
    frame_dt  = step_size * sample_rate
    total_time = round(n_frames * frame_dt, 3)

    frames = []
    for i in range(n_frames):
        s  = trajectory[i]
        u  = u_history[i]
        uc = ucart_history[i]
        frames.append({
            "type":    "frame",
            "i":       i,
            "t":       round(float(i * frame_dt), 5),
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

    out = {
        "meta": {
            "type":       "meta",
            "n_frames":   n_frames,
            "dt":         round(float(frame_dt), 6),
            "total_time": total_time,
            "setpoint": [
                round(float(setpoint[0]), 4),
                round(float(setpoint[1]), 4),
                round(float(setpoint[2]), 4),
            ],
        },
        "frames": frames,
    }

    out_path = Path(__file__).resolve().parent.parent / "frontend" / "public" / "demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f)

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {n_frames} frames ({size_kb:.0f} KB) → {out_path}")


if __name__ == "__main__":
    main()
