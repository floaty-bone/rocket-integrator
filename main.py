"""
main.py — Simulation entry point and 3-D animation for the rocket RK4 integrator.

Usage:
    python main.py

All tunable parameters are grouped under the CONFIG section below.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from integrator import RK4Integrator
from helper import quat_to_rotmat


# =============================================================================
# CONFIG — edit these values to change the simulation scenario
# =============================================================================

# Applied force/moment vector in the body frame [Fx, Fy, Fz, Mx, My, Mz]
# Examples:
#   Pure roll  : [0, 0, 0, 650,   0,   0]
#   Pure pitch : [0, 0, 0,   0, 700,   0]
#   Pure yaw   : [0, 0, 0,   0,   0, 550]
#   Combined   : [0, 0, 0, 650, 700, 550]  ← default
FORCE_MOMENT = np.array([0.0, 0.0, 0.0, 650.0, 700.0, 550.0], dtype=np.float64)

# Initial state  [x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
INITIAL_STATE = np.array(
    [0, 0, 0,  1, 0, 0, 0,  0, 0, 0,  4, 0.01, 0.01],
    dtype=np.float64,
)

# Inertia tensor (kg·m²)
INERTIA_MATRIX = np.diag([300.0, 100.0, 30.0])

# Simulation timing
SIM_TIME    = 100.0   # total duration (s)
STEP_SIZE   = 0.001   # RK4 time-step (s)
SAMPLE_RATE = 50      # save every Nth integration step (controls animation resolution)

# Vehicle geometry (for visualisation only)
BODY_LENGTH = 4.0   # X extent (m)
BODY_WIDTH  = 2.0   # Y extent (m)
BODY_HEIGHT = 1.0   # Z extent (m)

# Animation
ANIMATION_INTERVAL_MS = 20   # milliseconds between frames


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation() -> np.ndarray:
    """Run the RK4 integrator and return the sampled trajectory.

    Returns:
        Array of shape (n_frames, 13) containing the sampled state vectors.
    """
    integrator = RK4Integrator(
        get_force_moment=lambda: FORCE_MOMENT,
        inertia_matrix=INERTIA_MATRIX,
        length=BODY_LENGTH,
        radius=2.0,
        mass=85_000,
        gravity=0.0,
        step_size=STEP_SIZE,
    )

    n_steps = int(SIM_TIME / STEP_SIZE)
    n_frames = n_steps // SAMPLE_RATE + 1

    trajectory = np.empty((n_frames, 13), dtype=np.float64)
    state = INITIAL_STATE.copy()
    trajectory[0] = state

    frame_idx = 1
    for step_idx in range(1, n_steps):
        state = integrator.step_forward(state)
        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx] = state
            frame_idx += 1

    return trajectory[:frame_idx]   # trim any unused pre-allocated rows


# =============================================================================
# ANIMATION
# =============================================================================

# Cuboid face vertex indices
_CUBOID_FACES = [
    [0, 1, 2, 3],   # bottom
    [4, 5, 6, 7],   # top
    [0, 1, 5, 4],   # front
    [2, 3, 7, 6],   # back
    [0, 3, 7, 4],   # left
    [1, 2, 6, 5],   # right
]
_FACE_COLORS = ["cyan", "cyan", "blue", "blue", "red", "red"]
_AXIS_COLORS  = ["r", "g", "b"]
_AXIS_LABELS  = ["X (Roll)", "Y (Pitch)", "Z (Yaw)"]


def _make_cuboid_vertices(length: float, width: float, height: float) -> np.ndarray:
    """Return the 8 corner vertices of a cuboid centred at the origin."""
    l, w, h = length / 2, width / 2, height / 2
    return np.array([
        [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],  # bottom ring
        [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h],  # top ring
    ], dtype=np.float64)


def animate(trajectory: np.ndarray) -> None:
    """Launch the interactive 3-D animation window.

    Args:
        trajectory: (n_frames, 13) array of sampled state vectors.
    """
    positions        = trajectory[:, 0:3]
    cuboid_vertices  = _make_cuboid_vertices(BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT)
    axis_template    = np.eye(3) * 2.0   # body-fixed axes, length 2 m

    # Dynamic axis limits — pad the extremes slightly
    max_range = np.max(np.abs(positions)) + 5.0

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()

        state = trajectory[frame]
        pos   = state[0:3]
        quat  = state[3:7]
        R     = quat_to_rotmat(quat)

        # Rotate and translate body vertices
        rotated_verts = (R @ cuboid_vertices.T).T + pos
        faces_3d      = [[rotated_verts[j] for j in face] for face in _CUBOID_FACES]

        poly = Poly3DCollection(faces_3d, alpha=0.7, edgecolor="black", linewidths=1.5)
        poly.set_facecolor(_FACE_COLORS)
        ax.add_collection3d(poly)

        # Body-fixed coordinate axes
        rotated_axes = (R @ axis_template.T).T
        for axis, color, label in zip(rotated_axes, _AXIS_COLORS, _AXIS_LABELS):
            ax.quiver(
                *pos, *axis,
                color=color, arrow_length_ratio=0.2,
                linewidth=2.5, label=label,
            )

        # Trajectory trail
        ax.plot(
            positions[:frame + 1, 0],
            positions[:frame + 1, 1],
            positions[:frame + 1, 2],
            color="gray", alpha=0.4, linewidth=1.5,
        )

        # Axes cosmetics
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Z", fontsize=12)
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        time_elapsed = frame * SAMPLE_RATE * STEP_SIZE
        ax.set_title(
            f"Rigid Body Dynamics  |  t = {time_elapsed:.2f} s / {SIM_TIME:.1f} s",
            fontsize=14,
        )

        return (ax,)

    anim = FuncAnimation(  # noqa: F841  (kept alive via plt.show)
        fig, update,
        frames=len(trajectory),
        interval=ANIMATION_INTERVAL_MS,
        blit=False,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Running simulation…")
    trajectory = run_simulation()

    # Quick sanity check: quaternion norm at the last saved frame
    q_final = trajectory[-1, 3:7]
    print(f"  Frames generated : {len(trajectory)}")
    print(f"  Final quaternion norm : {np.linalg.norm(q_final):.6f}  (should be ≈ 1)")
    print(f"  Final state : {trajectory[-1]}")

    print("Launching animation…")
    animate(trajectory)
