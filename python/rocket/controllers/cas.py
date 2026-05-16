import math
import numpy as np
import jax.numpy as jnp
import sys
import os

# Add both the root python directory and the rocket directory to path
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
rocket_dir = os.path.join(python_dir, 'rocket')
sys.path.extend([python_dir, rocket_dir])

from rocket.dynamics_function import make_F
from rocket.controllers.base import LQRController
from rocket.helper import body_force_to_spherical

def run_sanity_check():
    print("Initializing dynamics...")
    VEHICLE_MASS   = 120_000.0
    INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])
    GRAVITY_FORCE  = np.array([0.0, 0.0, -VEHICLE_MASS * 9.8])
    
    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
    )
    
    print("Initializing LQR Controller...")
    # Crank these scalars to see their effect on the control output
    Q_scalar = 5.0
    R_scalar = 1.0

    # Base diagonal values (ratios between states)
    # Q has 13 elements (pos, quat, vel, ang_vel)
    # Let's crank position penalty (first 3) to 1e6 to get this 120t rocket moving!
    Q_diag = np.array([1e6, 1e6, 1e6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R_diag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    Q = np.diag(Q_diag) * Q_scalar
    R = np.diag(R_diag) * R_scalar
    
    controller = LQRController(F, Q=Q, R=R)
    
    
    # Setpoint: vertical rocket
    # The rocket vertical is a rotation around the y-axis by -90 degrees (-pi/2)
    # Quaternion components for a rotation of theta around Y:
    # qw = cos(theta/2)
    # qy = sin(theta/2)
    theta = -math.pi / 2
    qw = math.cos(theta / 2)
    qy = math.sin(theta / 2)

    # State: upright rocket
    # State vector: [px, py, pz,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
    state = jnp.array([0, 0, 0,  qw, 0, qy, 0,  0, -20, 0,  0, 0, 0], dtype=jnp.float64)
    
    setpoint = jnp.array([0, 0, 0,  qw, 0, qy, 0,  0, 0, 0,  0, 0, 0], dtype=jnp.float64)
    
    # Nominal thrust: Hover thrust to counteract gravity
    # Total weight = 120,000kg * 9.8m/s^2 = 1,176,000 N
    # Per engine (3 engines) = 392,000 N
    hover_thrust = (VEHICLE_MASS * 10) / 3.0
    u_nominal = jnp.array([hover_thrust, 0, 0, hover_thrust, 0, 0, hover_thrust, 0, 0], dtype=jnp.float64)
    
    # Update the linearization around the vertical setpoint
    print("Linearizing dynamics around the vertical setpoint (this may take a few seconds to compile)...")
    controller.update_linearization(setpoint, u_nominal)
    
    # Compute the control output U
    print("Computing control output U...")
    U = controller.update(state, setpoint=setpoint, u_nominal=u_nominal)
    
    np.set_printoptions(precision=4, suppress=True)
    print("\n" + "="*50)
    print("               SANITY CHECK RESULTS")
    print("="*50)
    print("Current State (Flat):      ", np.array(state))
    print("Setpoint (Vertical, -90Y): ", np.array(setpoint))
    print("-" * 50)
    print("Control Output U (Flattened 9x1):")
    print(np.array(U))
    print("\nControl Output U (Reshaped to 3x3 for the 3 engines):")
    U_reshaped = np.array(U).reshape(3, 3)
    print(U_reshaped)
    
    print("\nControl Output (Spherical/Gimbal Coordinates):")
    for i in range(3):
        spherical = body_force_to_spherical(U_reshaped[i])
        # spherical is [theta, phi, thrust]
        theta_deg = math.degrees(spherical[0])
        phi_deg   = math.degrees(spherical[1])
        thrust    = spherical[2]
        print(f"  Engine {i+1}: Gimbal Theta = {theta_deg:8.2f}°,  Phi = {phi_deg:8.2f}°,  Thrust = {thrust:10.1f} N")
    print("="*50)

if __name__ == "__main__":
    run_sanity_check()
