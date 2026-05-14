import numpy as np
import control as ctrl
import jax
import jax.numpy as jnp
import time
from dynamics_function import make_F

VEHICLE_MASS   = 120_000.0
INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])
GRAVITY_FORCE  = np.array([0.0, 0.0, -VEHICLE_MASS * 9.8])
Q=np.diag([1e3]*13)
R=np.diag([1e3]*9)    

F = make_F(
    mass=VEHICLE_MASS,
    inertia_matrix=INERTIA_MATRIX,
    gravity_force=GRAVITY_FORCE,
)

jac_x = jax.jit(jax.jacfwd(F, argnums=0))
jac_u = jax.jit(jax.jacfwd(F, argnums=1))
xl = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)
ul = jnp.array([0, 0, 20000, 0, 0, 20000, 0, 0, 20000], dtype=jnp.float64)

res_x = np.array(jac_x(xl, ul))
res_u = np.array(jac_u(xl, ul))

A_12 = np.delete(np.delete(res_x, 3, axis=0), 3, axis=1)
B_12 = np.delete(res_u, 3, axis=0)
Q_12 = np.delete(np.delete(Q, 3, axis=0), 3, axis=1)

print("A_12 shape:", A_12.shape)
print("B_12 shape:", B_12.shape)

try:
    K_12, S, E = ctrl.lqr(A_12, B_12, Q_12, R)
    print("LQR solved!")
except Exception as e:
    print("LQR failed:", e)
    
    # Check controllability
    Ctr = ctrl.ctrb(A_12, B_12)
    rank = np.linalg.matrix_rank(Ctr)
    print("Controllability matrix rank:", rank)
    
    # check eigenvalues
    evals, evecs = np.linalg.eig(A_12)
    print("Eigenvalues:", evals)
