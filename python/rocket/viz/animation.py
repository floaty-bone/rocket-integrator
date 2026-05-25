"""
animation.py — 3-D interactive animation of a recorded trajectory.

Renders the rocket body (built by :mod:`rocket.viz.mesh`), the body-fixed axes,
and the trajectory trail at each sampled frame.  Can either run interactively
(``plt.show``) or write to a file (``mp4`` / ``gif``).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rocket.config.vehicle import (
    N_CIRCLE_SEGMENTS,
    ROCKET_BASE_BELOW_COM,
    ROCKET_DIAMETER,
    ROCKET_HEIGHT,
    ROCKET_NOSE_HEIGHT,
)
from rocket.math.quaternion import quat_to_rotmat_np
from rocket.viz.mesh import AXIS_COLORS, AXIS_LABELS, make_rocket_mesh


def animate(
    trajectory: np.ndarray,
    sample_rate: int,
    step_size: float,
    sim_time: float,
    output_file: str | None = None,
) -> None:
    """Launch the interactive 3-D animation window (or save to file).

    Args:
        trajectory:  ``(n_frames, 13)`` sampled state vectors.
        sample_rate: Every Nth integration step that was saved.
        step_size:   The integration time-step used.
        sim_time:    Total simulation duration for the title.
        output_file: If set (e.g. ``"rocket.mp4"``) render at real-time speed
                     to that file instead of showing the interactive window.
                     ``.mp4`` requires ffmpeg on PATH; ``.gif`` uses Pillow.
    """
    positions = trajectory[:, 0:3]
    rocket_vertices, rocket_faces, rocket_face_colors = make_rocket_mesh(
        ROCKET_DIAMETER, ROCKET_HEIGHT, ROCKET_NOSE_HEIGHT,
        ROCKET_BASE_BELOW_COM, N_CIRCLE_SEGMENTS,
    )
    axis_template = np.eye(3) * (ROCKET_HEIGHT * 0.4)

    # Dynamic axis limits — pad enough that the full rocket always fits.
    max_range = np.max(np.abs(positions)) + ROCKET_HEIGHT

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()

        state = trajectory[frame]
        pos   = state[0:3]
        quat  = state[3:7]
        R     = quat_to_rotmat_np(quat)

        rotated_verts = (R @ rocket_vertices.T).T + pos
        faces_3d      = [[rotated_verts[j] for j in face] for face in rocket_faces]

        poly = Poly3DCollection(faces_3d, alpha=0.95, edgecolor="black", linewidths=0.4)
        poly.set_facecolor(rocket_face_colors)
        ax.add_collection3d(poly)

        rotated_axes = (R @ axis_template.T).T
        for axis, color, label in zip(rotated_axes, AXIS_COLORS, AXIS_LABELS):
            ax.quiver(
                *pos, *axis,
                color=color, arrow_length_ratio=0.2,
                linewidth=2.5, label=label,
            )

        ax.plot(
            positions[:frame + 1, 0],
            positions[:frame + 1, 1],
            positions[:frame + 1, 2],
            color="gray", alpha=0.4, linewidth=1.5,
        )

        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        time_elapsed = frame * sample_rate * step_size
        fig.suptitle(
            f"Rigid Body Dynamics Simulation\nt = {time_elapsed:.2f} s / {sim_time:.1f} s",
            fontsize=14, y=0.95,
        )

        return (ax,)

    # Real-time playback (subtract a small buffer for Matplotlib overhead).
    real_time_fps = 1.0 / (sample_rate * step_size)
    interval_ms = max(1, int(1000.0 / real_time_fps) - 5)

    anim = FuncAnimation(  # noqa: F841  (kept alive via plt.show)
        fig, update,
        frames=len(trajectory),
        interval=interval_ms,
        blit=False,
    )

    plt.subplots_adjust(top=0.85)

    if output_file:
        print(f"Rendering {output_file} at {real_time_fps:.1f} fps (real time)…")
        anim.save(
            output_file,
            fps=real_time_fps,
            dpi=120,
            progress_callback=lambda i, n: print(f"  frame {i + 1}/{n}", flush=True)
                                            if (i + 1) % 20 == 0 else None,
        )
        print(f"Saved {output_file}")
    else:
        plt.show()
