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
from rocket.helper_np import compute_thrust_forces_and_moments_cartesian as compute_wrench_np

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
        1e6, 1e6, 1e6,      # Position (x, y, z)
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
    initial_state = np.array([-200.0, 0.0, 200.0,  qw, 0.0, qy, 0.0,  0.0, 0.0, -25.0,  0.0, 0.0, 0.0], dtype=np.float64)
    setpoint      = np.array([0.0, 0.0, 0.0, qw, 0.0, qy, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0], dtype=np.float64)

    # Nominal thrust: Hover thrust to counteract gravity
    # Total weight = 120,000kg * 9.8m/s^2 = 1,176,000 N
    # Per engine (3 engines) = 392,000 N along body +X axis
    hover_thrust = (VEHICLE_MASS * 9.8) / 3.0
    u_nominal = np.array([hover_thrust, 0, 0, hover_thrust, 0, 0, hover_thrust, 0, 0], dtype=np.float64)

    # Warmup / Initial linearization
    print("Performing initial linearization (this may take a few seconds to compile JAX graph)...", flush=True)
    controller.update_linearization(jnp.array(initial_state), jnp.array(u_nominal))

    # =========================================================================
    # 5. Setup Integrator and Simulation Loop
    # =========================================================================
    n_steps   = int(SIM_TIME / STEP_SIZE)
    lin_steps = int(1.0 / (STEP_SIZE * LINEARIZATION_RATE))
    n_frames = n_steps // SAMPLE_RATE + 1

    trajectory = np.empty((n_frames, 13), dtype=np.float64)
    
    # We need a mutable reference to the current body wrench for the RK4Integrator callback
    # Initial wrench is just the nominal hover thrust.
    u_cart_initial = u_nominal.reshape(3, 3)
    current_body_wrench = compute_wrench_np(u_cart_initial, a, l)

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
    u = u_nominal.copy()  # last computed control; used for gain-scheduled linearization

    print(f"\nRunning SIL simulation for {SIM_TIME}s...", flush=True)
    start_time = time.time()
    
    frame_idx = 1
    report_every = n_steps // 10
    
    t_next_control = 0.0
    control_period = 1.0 / CONTROL_FREQ
    control_count = 0

    for step_idx in range(1, n_steps):
        t = step_idx * STEP_SIZE
        
        # 1. 10Hz Linearization update (TEMPORARILY DISABLED)
        # if step_idx % lin_steps == 0:
        #     # Gain scheduling: linearize around the actual current (state, u) operating point,
        #     # not u_nominal. The rocket's true input drifts from hover as it translates,
        #     # so using the last computed u gives a much more accurate local linear model.
        #     controller.update_linearization(jnp.array(state), jnp.array(u))
            
        # 2. Control update (runs at CONTROL_FREQ)
        if t >= t_next_control:
            u = controller.update(state, setpoint=setpoint, u_nominal=u_nominal)
            # Update the body wrench for the integrator (using fast NumPy version)
            u_cart = u.reshape(3, 3)
            current_body_wrench = compute_wrench_np(u_cart, a, l)
            t_next_control += control_period
            control_count += 1
        
        # 3. Integration step (runs at SIM_FREQ)
        state = integrator.step_forward(state)
        
        # 4. Record state for animation
        if step_idx % SAMPLE_RATE == 0 and frame_idx < n_frames:
            trajectory[frame_idx] = state
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
    
    return trajectory[:frame_idx], STEP_SIZE, SAMPLE_RATE, SIM_TIME, setpoint

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
    trajectory, dt, sample, sim_time, setpoint = run_sil_simulation()
    
    print("Generating plots...", flush=True)
    plot_trajectory_tracking(trajectory, dt, sample, sim_time, setpoint)
    
    print("Launching animation...", flush=True)
    from rocket.main import animate
    animate(trajectory, sample_rate=sample, step_size=dt, sim_time=sim_time)
