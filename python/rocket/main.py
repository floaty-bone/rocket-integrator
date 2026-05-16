"""
main.py — Simulation entry point and 3-D animation for the rocket RK4 integrator.

Usage:
    python main.py

All tunable parameters are grouped under the CONFIG section below.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rocket.integrator import RK4Integrator
from rocket.helper import quat_to_rotmat, compute_thrust_forces_and_moments


# =============================================================================
# CONFIG — edit these values to change the simulation scenario
# =============================================================================

# Body-frame wrench [Fx, Fy, Fz, Mx, My, Mz] from actuators (thrust, TVC, etc.)
# max thrust sea level raptor engine: 2.8 MN, min thrust: 40% of max
BODY_WRENCH = compute_thrust_forces_and_moments(  # 40% of single engine thrust
    engine_thrust=np.array([
        [3.14, 0.4, 0.30 * 2.8e6],
        [3.14, 0.4, 0.30 * 2.8e6],
        [3.14, 0.4, 0.30 * 2.8e6],
    ], dtype=np.float64),
    a=1.5,
    l=18,
)

# Vehicle mass used for both the integrator and the gravity force below.
VEHICLE_MASS = 120_000.0   # kg

# Gravitational acceleration (m/s²).  Positive Z is "up" in this simulation,
# so gravity points in the -Z direction.  Expressed in Newtons so it feeds
# directly into get_inertial_force (force = mass * g_vec).
_G = 9.8
GRAVITY_FORCE = np.array([0.0, 0.0, -VEHICLE_MASS * _G], dtype=np.float64)

# Initial state  [x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
INITIAL_STATE = np.array(
    [0, 0, 0,  1, 0, 0, 0,  0, 0, -50,  0.0, 0.0, 0.0],
    dtype=np.float64,
)

# Inertia tensor (kg·m²)
INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])  # [Ixx, Iyy, Izz] #empty estimates for starship (dry), should be updated with more accurate values when available

# Simulation timing
SIM_TIME    = 5   # total duration (s)
STEP_SIZE   = 0.001   # RK4 time-step (s)
SAMPLE_RATE = 50      # save every Nth integration step (controls animation resolution)

# Vehicle geometry (for visualisation only)
BODY_LENGTH = 4.0   # X extent (m)
BODY_WIDTH  = 2.0   # Y extent (m)
BODY_HEIGHT = 1.0   # Z extent (m)

# Rocket render geometry — cylinder + conical nose, long axis along body X.
# Body-frame origin = vehicle centre of mass; ROCKET_BASE_BELOW_COM places the
# base of the cylinder that many metres below the CoM along -X.
ROCKET_DIAMETER       = 9.0    # m  (cylinder + cone base diameter)
ROCKET_HEIGHT         = 50.0   # m  (base of cylinder → tip of nose, along body X)
ROCKET_NOSE_HEIGHT    = 10.0   # m  (length of conical nose section)
ROCKET_BASE_BELOW_COM = 20.0   # m  (CoM-to-base distance along -X)
N_CIRCLE_SEGMENTS     = 32     # azimuthal mesh resolution

# Animation
ANIMATION_INTERVAL_MS = 20   # milliseconds between frames

# Output: set OUTPUT_FILE to a path (e.g. "rocket.mp4") to render at real-time
# speed instead of showing the live (slow) interactive window. Requires ffmpeg
# on PATH for .mp4; .gif works out-of-the-box via Pillow.
OUTPUT_FILE = None


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation() -> np.ndarray:
    """Run the RK4 integrator and return the sampled trajectory.

    Returns:
        Array of shape (n_frames, 13) containing the sampled state vectors.
    """
    integrator = RK4Integrator(
        get_body_wrench=lambda: BODY_WRENCH,
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        length=BODY_LENGTH,
        radius=2.0,
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

    return trajectory[:frame_idx]   # trim any unused pre-allocated rows


# =============================================================================
# ANIMATION
# =============================================================================

_AXIS_COLORS  = ["r", "g", "b"]
_AXIS_LABELS  = ["X (Roll)", "Y (Pitch)", "Z (Yaw)"]

# Rocket render colours — two cylinder halves (so roll is visible), nose, base cap.
_CYL_COLOR_A = "white"
_CYL_COLOR_B = "lightgray"
_NOSE_COLOR  = "crimson"
_BASE_COLOR  = "dimgray"


def _make_rocket_mesh(
    diameter: float,
    total_height: float,
    nose_height: float,
    base_below_com: float,
    n_seg: int,
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """Build a rocket mesh (cylinder + conical nose) anchored to the CoM.

    The long axis is body X; the cone tip points in +X.  ``base_below_com`` is
    the distance from the body-frame origin (CoM) down to the base of the
    cylinder, so the base sits at x = -base_below_com and the tip at
    x = total_height - base_below_com.

    Returns the vertex array and a parallel list of polygon face index-lists
    plus their colours, suitable for ``Poly3DCollection``.
    """
    radius     = diameter / 2.0
    cyl_length = total_height - nose_height
    x_base     = -base_below_com
    x_shoulder = x_base + cyl_length
    x_tip      = x_base + total_height

    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cy = radius * np.cos(angles)
    cz = radius * np.sin(angles)

    base_ring     = np.column_stack([np.full(n_seg, x_base),     cy, cz])
    shoulder_ring = np.column_stack([np.full(n_seg, x_shoulder), cy, cz])
    tip           = np.array([[x_tip, 0.0, 0.0]])

    vertices = np.vstack([base_ring, shoulder_ring, tip])
    base_idx     = list(range(0, n_seg))
    shoulder_idx = list(range(n_seg, 2 * n_seg))
    tip_idx      = 2 * n_seg

    faces: list[list[int]] = []
    colors: list[str]      = []

    # Base cap — single n-gon, reverse winding so the outward normal points -X.
    faces.append(list(reversed(base_idx)))
    colors.append(_BASE_COLOR)

    # Cylinder side — quad strips, halved for roll visibility.
    half = n_seg // 2
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([base_idx[i], base_idx[j], shoulder_idx[j], shoulder_idx[i]])
        colors.append(_CYL_COLOR_A if i < half else _CYL_COLOR_B)

    # Conical nose — triangle fan from tip.
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([shoulder_idx[i], shoulder_idx[j], tip_idx])
        colors.append(_NOSE_COLOR)

    return vertices, faces, colors


def animate(trajectory: np.ndarray, sample_rate: int = SAMPLE_RATE, step_size: float = STEP_SIZE, sim_time: float = SIM_TIME) -> None:
    """Launch the interactive 3-D animation window.

    Args:
        trajectory: (n_frames, 13) array of sampled state vectors.
        sample_rate: Every Nth integration step that was saved.
        step_size: The integration time-step used.
        sim_time: Total simulation duration for the title.
    """
    positions = trajectory[:, 0:3]
    rocket_vertices, rocket_faces, rocket_face_colors = _make_rocket_mesh(
        ROCKET_DIAMETER, ROCKET_HEIGHT, ROCKET_NOSE_HEIGHT,
        ROCKET_BASE_BELOW_COM, N_CIRCLE_SEGMENTS,
    )
    axis_template = np.eye(3) * (ROCKET_HEIGHT * 0.4)   # body-fixed axes

    # Dynamic axis limits — pad enough that the 50 m rocket always fits.
    max_range = np.max(np.abs(positions)) + ROCKET_HEIGHT

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.clear()

        state = trajectory[frame]
        pos   = state[0:3]
        quat  = state[3:7]
        R     = quat_to_rotmat(quat)

        # Rotate and translate body vertices
        rotated_verts = (R @ rocket_vertices.T).T + pos
        faces_3d      = [[rotated_verts[j] for j in face] for face in rocket_faces]

        poly = Poly3DCollection(faces_3d, alpha=0.95, edgecolor="black", linewidths=0.4)
        poly.set_facecolor(rocket_face_colors)
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
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        time_elapsed = frame * sample_rate * step_size
        # Use suptitle to avoid cropping and provide more space
        fig.suptitle(
            f"Rigid Body Dynamics Simulation\nt = {time_elapsed:.2f} s / {sim_time:.1f} s",
            fontsize=14, y=0.95
        )

        return (ax,)

    # Calculate interval for real-time playback
    # We subtract a small 'buffer' (e.g. 5ms) to account for Python/Matplotlib overhead
    real_time_fps = 1.0 / (sample_rate * step_size)
    interval_ms = max(1, int(1000.0 / real_time_fps) - 5)

    anim = FuncAnimation(  # noqa: F841  (kept alive via plt.show)
        fig, update,
        frames=len(trajectory),
        interval=interval_ms,
        blit=False,
    )

    # Adjust layout to make room for the suptitle
    plt.subplots_adjust(top=0.85)

    if OUTPUT_FILE:
        print(f"Rendering {OUTPUT_FILE} at {real_time_fps:.1f} fps (real time)…")
        anim.save(
            OUTPUT_FILE,
            fps=real_time_fps,
            dpi=120,
            progress_callback=lambda i, n: print(f"  frame {i + 1}/{n}", flush=True)
                                            if (i + 1) % 20 == 0 else None,
        )
        print(f"Saved {OUTPUT_FILE}")
    else:
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
