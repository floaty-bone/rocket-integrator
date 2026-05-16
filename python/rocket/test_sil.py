import sys
import os
import math
import time
import numpy as np
import jax.numpy as jnp

# Add both the root python directory and the rocket directory to path
python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
rocket_dir = os.path.join(python_dir, 'rocket')
if python_dir not in sys.path: sys.path.append(python_dir)
if rocket_dir not in sys.path: sys.path.append(rocket_dir)

from rocket.integrator import RK4Integrator
from rocket.controllers.base import LQRController
from rocket.dynamics_function import make_F
from rocket.helper import compute_thrust_forces_and_moments_cartesian

def test():
    print("Test start")
    VEHICLE_MASS   = 120_000.0
    INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])
    GRAVITY_FORCE  = np.array([0.0, 0.0, -VEHICLE_MASS * 9.8])
    a, l = 1.5, 18.0
    F = make_F(VEHICLE_MASS, INERTIA_MATRIX, GRAVITY_FORCE, a, l)
    controller = LQRController(F)
    theta = -math.pi / 2
    qw = math.cos(theta / 2)
    qy = math.sin(theta / 2)
    state = np.array([0, 0, 0,  qw, 0, qy, 0,  0, 0, 0,  0, 0, 0], dtype=np.float64)
    u_nominal = np.array([(VEHICLE_MASS*9.8)/3, 0, 0, (VEHICLE_MASS*9.8)/3, 0, 0, (VEHICLE_MASS*9.8)/3, 0, 0])
    print("Linearizing...")
    controller.update_linearization(jnp.array(state), jnp.array(u_nominal))
    print("Done linearizing")
    u = controller.update(state)
    print(f"Control: {u[:3]}...")
    print("Test finished successfully")

if __name__ == "__main__":
    test()
