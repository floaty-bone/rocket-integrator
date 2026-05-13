from helper import compute_thrust_forces_and_moments, quat_to_rotmat, construct_omega_matrix
import numpy as np
import time
import jax
import jax.numpy as jnp
import control as ctrl
# Enable 64-bit precision for JAX to avoid float32 truncation warnings
jax.config.update("jax_enable_x64", True)

def make_F(
    mass: float,
    inertia_matrix: np.ndarray,
    gravity_force: np.ndarray,
    a: float = 1.5,
    l: float = 18.0,
):
    """
    Factory that pre-computes constants and returns a fast F(X, U) callable.

    Args:
        mass           : vehicle mass (kg)
        inertia_matrix : (3,3) inertia tensor in body frame (kg·m²)
        gravity_force  : (3,)  gravitational force in inertial frame (N)
        a              : engine cluster radius (m)
        l              : CoM-to-engine-plane distance along X (m)

    Returns:
        F(X, U) -> X_dot  (13,)
    """
    _mass         = mass
    _I            = np.asarray(inertia_matrix, dtype=np.float64)
    _I_inv        = np.linalg.inv(_I)
    _gravity      = np.asarray(gravity_force,  dtype=np.float64)
    _a, _l        = a, l

    def F(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Ẋ = F(X, U)

        Args:
            X : (13,) — [px, py, pz,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
            U : (9,)  — [θ1, φ1, T1,  θ2, φ2, T2,  θ3, φ3, T3]
        """
        pos, quat, vel, omega = X[0:3], X[3:7], X[7:10], X[10:13]

        body_wrench = compute_thrust_forces_and_moments(U.reshape(3, 3), a=_a, l=_l)
        R           = quat_to_rotmat(quat)
        Iw          = _I @ omega

        dP = vel
        dq = 0.5 * construct_omega_matrix(omega) @ quat
        dv = (R @ body_wrench[:3] + _gravity) / _mass
        dw = _I_inv @ (body_wrench[3:] - jnp.cross(omega, Iw))

        return jnp.concatenate([dP, dq, dv, dw])

    return F


if __name__ == "__main__":
    VEHICLE_MASS   = 120_000.0
    INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])
    GRAVITY_FORCE  = np.array([0.0, 0.0, -VEHICLE_MASS * 9.8])
    # State weights for LQR controller 
    Q=np.diag([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])
    # Input weights for LQR controller
    R=np.diag([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])    

    F = make_F(
        mass=VEHICLE_MASS,
        inertia_matrix=INERTIA_MATRIX,
        gravity_force=GRAVITY_FORCE,
    )

    xl  = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    ul  = np.array([0, 3.14/2, 20000, 0, 3.14/2, 20000, 0, 3.14/2, 20000],  dtype=np.float64)

    jac_x = jax.jit(jax.jacfwd(F, argnums=0))
    jac_u = jax.jit(jax.jacfwd(F, argnums=1))
    xl = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    ul = jnp.array([0, 3.14/2, 20000, 0, 3.14/2, 20000, 0, 3.14/2, 20000], dtype=jnp.float32)

    # WARMUP: Trigger JIT compilation (takes a few hundred milliseconds)
    _ = jac_x(xl, ul).block_until_ready()
    _ = jac_u(xl, ul).block_until_ready()

    # Now measure actual execution time
    start_time = time.perf_counter()
    res_x = jac_x(xl, ul).block_until_ready()
    res_u = jac_u(xl, ul).block_until_ready()
    
    # Drop the redundant quaternion scalar part (qw at index 3) 
    # to make the system strictly stabilizable for LQR.
    A_12 = np.delete(np.delete(np.array(res_x), 3, axis=0), 3, axis=1)
    B_12 = np.delete(np.array(res_u), 3, axis=0)
    Q_12 = np.delete(np.delete(Q, 3, axis=0), 3, axis=1)
    
    K_12, S_12, E_12 = ctrl.lqr(A_12, B_12, Q_12, R)
    
    # Re-insert the zero column for qw so K is 9x13
    K = np.insert(K_12, 3, 0.0, axis=1)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    print(f"Execution time (after compilation): {total_time:.6f} seconds")

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    
    print("\n--- Jacobian w.r.t State (A matrix) [13 x 13] ---")
    print(np.array(res_x))
    
    print("\n--- Jacobian w.r.t Controls (B matrix) [13 x 9] ---")
    print(np.array(res_u))

    print("\n k matrix")
    print(np.array(K))
