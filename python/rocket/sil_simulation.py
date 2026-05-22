"""
sil_simulation.py — Software-in-the-Loop simulation for the rocket.
"""

import math
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import os

# Add both the root python directory and the rocket directory to path
# This matches the pattern used in other scripts like cas.py
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
rocket_dir = os.path.join(python_dir, 'rocket')
if python_dir not in sys.path: sys.path.append(python_dir)
if rocket_dir not in sys.path: sys.path.append(rocket_dir)

# =========================================================================
# CONFIG — edit these values to change the simulation scenario
# =========================================================================
SIM_TIME  = 30.0   # Total simulation duration (s)
SIM_FREQ  = 8000   # Integration frequency (Hz)
CONTROL_FREQ = 5000 # Control frequency (Hz)
STEP_SIZE = 1.0 / SIM_FREQ  # Integration time-step (s)
SAMPLE_RATE = int(SIM_FREQ / 5)  # Save at 5Hz for plotting/animation
LINEARIZATION_RATE = 30  # Hz

# Import components from the rocket package
from rocket.integrator import RK4Integrator
from rocket.controllers.base import LQRController
from rocket.dynamics_function import make_F
from rocket.helper import compute_thrust_forces_and_moments_cartesian as compute_wrench_jax
from rocket.helper_np import (
    compute_thrust_forces_and_moments_cartesian as compute_wrench_np,
)

# ---------------------------------------------------------------------------
# Controller ↔ Plant boundary converters
#
# The LQR controller operates internally in Cartesian body-frame forces
# [Fx, Fy, Fz] per engine.  At the interface boundary we expose the real
# TVC command: [alpha, beta, T] per engine — two orthogonal Euler-style
# gimbal angles plus thrust magnitude.  This matches the physical U-joint
# TVC mechanism (two orthogonal pivots).  The sub-TVC servo is assumed
# instantaneous (infinite bandwidth).
#
# Convention (rotation about body Y by α, then about body Z by β):
#     fx =  T · cos(α) · cos(β)
#     fy =  T · cos(α) · sin(β)
#     fz = -T · sin(α)
#
# Equilibrium: α=β=0, T along body +X. Bijective on α∈(-π/2, π/2), β∈(-π, π],
# T > 0. Singular only at α=±π/2 (thrust perpendicular to body X) — outside
# any realistic operating range.
# ---------------------------------------------------------------------------

def _engine_cart_to_euler(force: np.ndarray) -> np.ndarray:
    fx, fy, fz = force[0], force[1], force[2]
    T = np.sqrt(fx*fx + fy*fy + fz*fz)
    if T < 1e-9:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    alpha = -np.arcsin(np.clip(fz / T, -1.0, 1.0))
    beta  = np.arctan2(fy, fx)
    return np.array([alpha, beta, T], dtype=np.float64)

def _engine_euler_to_cart(gimbal: np.ndarray) -> np.ndarray:
    alpha, beta, T = gimbal[0], gimbal[1], gimbal[2]
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    return np.array([T*ca*cb, T*ca*sb, -T*sa], dtype=np.float64)

def cart9_to_gimbal9(u_cart: np.ndarray) -> np.ndarray:
    """Convert 9-elem Cartesian force vector to 9-elem [alpha, beta, T] per engine."""
    u_3x3 = u_cart.reshape(3, 3)
    return np.concatenate([_engine_cart_to_euler(u_3x3[i]) for i in range(3)])

def gimbal9_to_cart9(u_gimbal: np.ndarray) -> np.ndarray:
    """Convert 9-elem [alpha, beta, T] per engine to 9-elem Cartesian force vector."""
    u_3x3 = u_gimbal.reshape(3, 3)
    return np.concatenate([_engine_euler_to_cart(u_3x3[i]) for i in range(3)])


def run_sil_simulation():
    # =========================================================================
    # 1. Define Vehicle Parameters
    # =========================================================================
    VEHICLE_MASS   = 120_000.0  # kg
    INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])  # kg*m^2
    GRAVITY_FORCE  = np.array([0.0, 0.0, -VEHICLE_MASS * 9.8])  # N
    a = 1.5   # Engine cluster radius (m)
    l = 18.0  # Distance from CoM to engine plane (m)

    # =========================================================================
    # 2. Setup Dynamics Function
    # =========================================================================
    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
        a=a,
        l=l
    )

    # =========================================================================
    # 3. Setup LQR Controller
    # =========================================================================
    # Q weight tuning rationale:
    # - Original (50x vel/pos): overdamped — braked before reaching 40m, reversed, oscillated
    # - Equal (1x vel/pos):     underdamped — overshot by ~5m, large slow oscillation (ζ≈0.5)
    # - Target (10x vel/pos):   near-critical damping (ζ≈0.7–1.0) — smooth monotonic approach
    #
    # The velocity weight must be high enough to suppress overshoot, but not so high
    # that it brakes before the rocket reaches the target. 10x is the geometric midpoint
    # on the log scale between the two failure modes.
    Q_diag = np.array([
        1e7, 1e7, 1e7,      # Position (x, y, z)
        1e2, 1e2, 1e2, 1e2, # Orientation (qw, qx, qy, qz)
        1e5, 1e5, 1e5,      # Velocity (vx, vy, vz) — 10x position = near-critical damping
        1e1, 1e1, 1e1,      # Angular velocity (wx, wy, wz)
    ])
    R_diag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    
    print("Initializing LQR Controller...", flush=True)
    controller = LQRController(F, Q=Q, R=R)

    # =========================================================================
    # 4. Define Initial State and Setpoint
    # =========================================================================
    # The rocket vertical is a rotation around the y-axis by -90 degrees (-pi/2)
    # Body +X is aligned with Inertial +Z (Up)
    theta = -math.pi / 2
    qw = math.cos(theta / 2)
    qy = math.sin(theta / 2)

    # state: [px, py, pz,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
    initial_state = np.array([0.0, 0.0, 0.0,  qw, 0.0, qy, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0], dtype=np.float64)
    setpoint      = np.array([50.0, 100.0, 60.0, qw, 0.0, qy, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0], dtype=np.float64)

    # Nominal thrust: Hover thrust to counteract gravity
    # Total weight = 120,000kg * 9.8m/s^2 = 1,176,000 N
    # Per engine (3 engines) = 392,000 N along body +X axis
    hover_thrust = (VEHICLE_MASS * 9.8) / 3.0
    # Internal Cartesian format used by LQR dynamics model
    u_nominal_cart = np.array([hover_thrust, 0, 0, hover_thrust, 0, 0, hover_thrust, 0, 0], dtype=np.float64)
    # Gimbal format [alpha, beta, T] per engine — the real TVC command interface
    # At hover: zero gimbal angles, full thrust along body +X → alpha=0, beta=0, T=hover_thrust
    u_nominal_gimbal = cart9_to_gimbal9(u_nominal_cart)

    # Warmup / Initial linearization (always in Cartesian for the LQR dynamics model)
    print("Performing initial linearization (this may take a few seconds to compile JAX graph)...", flush=True)
    controller.update_linearization(jnp.array(initial_state), jnp.array(u_nominal_cart))

    # =========================================================================
    # 5. Setup Integrator and Simulation Loop
    # =========================================================================
    n_steps   = int(SIM_TIME / STEP_SIZE)
    lin_steps = int(1.0 / (STEP_SIZE * LINEARIZATION_RATE))
    n_frames = n_steps // SAMPLE_RATE + 1

    trajectory = np.empty((n_frames, 13), dtype=np.float64)
    u_history = np.empty((n_frames, 9), dtype=np.float64)
    
    # We need a mutable reference to the current body wrench for the RK4Integrator callback.
    # Initial wrench is the nominal hover thrust (gimbal nominal → Cartesian → wrench).
    current_body_wrench = compute_wrench_np(gimbal9_to_cart9(u_nominal_gimbal).reshape(3, 3), a, l)

    def get_body_wrench():
        return current_body_wrench

    integrator = RK4Integrator(
        get_body_wrench=get_body_wrench,
        get_inertial_force=lambda: GRAVITY_FORCE,
        inertia_matrix=INERTIA_MATRIX,
        length=l,
        radius=a,
        mass=VEHICLE_MASS,
        step_size=STEP_SIZE,
    )

    state = initial_state.copy()
    trajectory[0] = state
    u_gimbal = u_nominal_gimbal.copy()  # last command in gimbal format [alpha, beta, T] × 3
    u_history[0] = u_gimbal

    print(f"\nRunning SIL simulation for {SIM_TIME}s...", flush=True)
    start_time = time.time()
    
    frame_idx = 1
    report_every = n_steps // 10
    
    # =========================================================================
    # CONTROLLER BLOCK
    #   IN  : state [13]  — [pos(3), quat(4), vel(3), omega(3)]
    #   OUT : u_gimbal [9] — [alpha, beta, T] per engine  ← SIL boundary
    # =========================================================================
    def controller_block(state: np.ndarray, u_gimbal_prev: np.ndarray, step_idx: int) -> np.ndarray:
        """Run the LQR and return TVC commands [α, β, T] × 3 engines."""
        if step_idx % lin_steps == 0:
            controller.update_linearization(jnp.array(state), jnp.array(gimbal9_to_cart9(u_gimbal_prev)))
        u_cart = controller.update(state, setpoint=setpoint, u_nominal=u_nominal_cart)
        return cart9_to_gimbal9(u_cart)  # → [α, β, T] × 3  (SIL output)

    # =========================================================================
    # PLANT / SIMULATION BLOCK
    #   IN  : u_gimbal [9] — [alpha, beta, T] per engine  ← SIL boundary
    #   INTERNAL: converts gimbal → wrench → integrates dynamics
    # =========================================================================
    def plant_block(u_gimbal: np.ndarray) -> None:
        """Accept TVC commands [α, β, T] × 3 and update the shared body wrench."""
        nonlocal current_body_wrench
        current_body_wrench = compute_wrench_np(gimbal9_to_cart9(u_gimbal).reshape(3, 3), a, l)

    t_next_control = 0.0
    control_period = 1.0 / CONTROL_FREQ
    control_count = 0

    for step_idx in range(1, n_steps):
        t = step_idx * STEP_SIZE

        # ── CONTROLLER BLOCK (runs at CONTROL_FREQ) ──────────────────────────
        if t >= t_next_control:
            u_gimbal = controller_block(state, u_gimbal, step_idx)   # OUT: [α, β, T] × 3

            # ── SIL BOUNDARY: [α, β, T] per engine handed to plant ───────────
            plant_block(u_gimbal)                          # IN : [α, β, T] × 3
            # ─────────────────────────────────────────────────────────────────

            t_next_control += control_period
            control_count += 1

        # ── PLANT BLOCK: integration step (runs at SIM_FREQ) ─────────────────
        state = integrator.step_forward(state)
        
        # 4. Record state for animation
        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx] = state
            u_history[frame_idx] = u_gimbal
            frame_idx += 1
            
        # 5. Progress reporting
        if step_idx % report_every == 0:
            print(f"  {step_idx * 100 // n_steps}% complete...", flush=True)

    elapsed = time.time() - start_time
    print(f"\nSimulation finished in {elapsed:.2f}s (real time).", flush=True)
    print(f"Effective Control Frequency: {control_count / SIM_TIME:.1f} Hz")
    
    np.set_printoptions(precision=4, suppress=True)
    print(f"Final Position: {state[0:3]}")
    print(f"Target Position: {setpoint[0:3]}")
    
    # Measure the quaternion norm to check for numerical drift before we apply normalization
    q_final = state[3:7]
    print(f"Final Quaternion Norm: {np.linalg.norm(q_final):.8f} (should be ≈ 1)")
    
    return trajectory[:frame_idx], u_history[:frame_idx], STEP_SIZE, SAMPLE_RATE, SIM_TIME, setpoint

def plot_thrust(u_history, step_size, sample_rate, total_sim_time):
    """Plot gimbal angles (alpha, beta) and thrust magnitude T for each of the 3 engines."""
    times = np.arange(len(u_history)) * step_size * sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, ax in enumerate(axes):
        alpha = np.degrees(u_history[:, i*3])      # pitch about body Y (deg)
        beta  = np.degrees(u_history[:, i*3 + 1])  # yaw   about body Z (deg)
        T     = u_history[:, i*3 + 2] / 1e3        # thrust magnitude   (kN)

        print(f"Engine {i+1}: α range [{alpha.min():+.3f}, {alpha.max():+.3f}] deg,  "
              f"β range [{beta.min():+.3f}, {beta.max():+.3f}] deg,  "
              f"T range [{T.min():.1f}, {T.max():.1f}] kN")

        ax2 = ax.twinx()
        # Plot β first (underneath) and α on top with a dashed style + markers so it
        # stays visible even if α and β overlap or have very different magnitudes.
        ax.plot(times, beta,  label="β yaw   (deg)",  color='#3498db', linewidth=2, zorder=2)
        ax.plot(times, alpha, label="α pitch (deg)",  color='#e74c3c', linewidth=2.5, linestyle='--', zorder=3)
        ax2.plot(times, T,    label="T thrust (kN)",  color='#2ecc71', linewidth=2, linestyle=':',  zorder=1)

        ax.set_ylabel("Gimbal angle (deg)", fontsize=11)
        ax2.set_ylabel("Thrust (kN)", fontsize=11)
        ax.set_title(f"Engine {i+1} TVC Command", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc="upper right")

    fig.suptitle("Engine TVC Commands: Gimbal Angles (α=pitch, β=yaw) & Thrust", fontsize=14)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    
    plt.xlim(0, total_sim_time)
    plt.tight_layout()

def plot_trajectory_tracking(trajectory, step_size, sample_rate, total_sim_time, setpoint):
    """Plot X, Y, and Z positions of the center of mass against time."""
    times = np.arange(len(trajectory)) * step_size * sample_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    labels = ['X (Vertical)', 'Y (Lateral)', 'Z (Lateral)']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for i, ax in enumerate(axes):
        pos_data = trajectory[:, i]
        target = setpoint[i]
        
        ax.plot(times, pos_data, label=f"Current {labels[i]}", color=colors[i], linewidth=2)
        ax.axhline(y=target, color="#e74c3c", linestyle="--", label=f"Target ({target}m)", alpha=0.8)
        
        ax.set_ylabel(f"{labels[i]} (m)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, loc="lower right")
        
    axes[0].set_title(f"Rocket Trajectory Tracking: Position vs Time (Total: {total_sim_time}s)", fontsize=14, pad=15)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    
    plt.xlim(0, total_sim_time)
    plt.tight_layout()
    # Note: plt.show() is not called here; it will be called by animate() or at the end of main.
    # Note: plt.show() is not called here; it will be called by animate() or at the end of main.

if __name__ == "__main__":
    trajectory, u_history, dt, sample, sim_time, setpoint = run_sil_simulation()
    
    print("Generating plots...", flush=True)
    plot_trajectory_tracking(trajectory, dt, sample, sim_time, setpoint)
    plot_thrust(u_history, dt, sample, sim_time)
    
    print("Launching animation...", flush=True)
    from rocket.main import animate
    animate(trajectory, sample_rate=sample, step_size=dt, sim_time=sim_time)
