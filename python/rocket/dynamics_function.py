from helper import compute_thrust_forces_and_moments, quat_to_rotmat, construct_omega_matrix, body_force_to_spherical, compute_thrust_forces_and_moments_cartesian
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
            U : (9,)  — [X1, Y1, Z1,  X2, Y2, Z2,  X3, Y3, Z3]
        """
        pos, quat, vel, omega = X[0:3], X[3:7], X[7:10], X[10:13]

        U_cart = U.reshape(3, 3)
        body_wrench = compute_thrust_forces_and_moments_cartesian(U_cart, a=_a, l=_l)
        R           = quat_to_rotmat(quat)
        Iw          = _I @ omega

        dP = vel
        dq = 0.5 * construct_omega_matrix(omega) @ quat
        dv = (R @ body_wrench[:3] + _gravity) / _mass
        dw = _I_inv @ (body_wrench[3:] - jnp.cross(omega, Iw))

        return jnp.concatenate([dP, dq, dv, dw])

    return F
def make_lqr_computer(F, Q, R):
    """
    Factory that returns a fast, pre-compiled callable to compute LQR gains.
    Ideal for embedding inside a control loop to avoid JAX recompilation overhead.
    
    Args:
        F: The dynamics function callable
        Q: The state weight matrix (13, 13)
        R: The input weight matrix (9, 9)
        
    Returns:
        compute_lqr(xl, ul) -> (K, A, B)
    """
    # 1. Pre-compile the jacobians exactly once (zero overhead on subsequent calls)
    jac_x_fn = jax.jit(jax.jacfwd(F, argnums=0))
    jac_u_fn = jax.jit(jax.jacfwd(F, argnums=1))
    
    # 2. Pre-process the Q and R matrices for the 12-state system once
    Q_12 = np.delete(np.delete(np.array(Q), 3, axis=0), 3, axis=1)
    R_np = np.array(R)
    
    def compute_lqr(xl, ul):
        # Evaluate pre-compiled JIT jacobians
        res_x = jac_x_fn(xl, ul).block_until_ready()
        res_u = jac_u_fn(xl, ul).block_until_ready()
        
        A = np.array(res_x)
        B = np.array(res_u)
        
        # Drop the redundant quaternion scalar part (qw at index 3) 
        # to make the system strictly stabilizable for LQR.
        A_12 = np.delete(np.delete(A, 3, axis=0), 3, axis=1)
        B_12 = np.delete(B, 3, axis=0)
        
        K_12, S_12, E_12 = ctrl.lqr(A_12, B_12, Q_12, R_np)
        
        # Re-insert the zero column for qw so K is 9x13
        K = np.insert(K_12, 3, 0.0, axis=1)
        
        return K, A, B

    return compute_lqr


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

    # Create the fast LQR computer factory (do this once outside the control loop)
    compute_lqr = make_lqr_computer(F, Q, R)
    
    xl = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)
    ul = jnp.array([200000, 0, 0, 200000, 0, 0, 200000, 0, 0], dtype=jnp.float64)

    # Warmup compilation so it isn't included in the timing
    _ = compute_lqr(xl, ul)

    start_time = time.perf_counter()
    K, A, B = compute_lqr(xl, ul)
    end_time = time.perf_counter()
    Xdot=F(xl,ul)
    total_time = end_time - start_time
    print(f"Execution time (after compilation): {total_time:.6f} seconds")

    np.set_printoptions(precision=6, suppress=False, linewidth=200)
    
    print("\n--- Jacobian w.r.t State (A matrix) [13 x 13] ---")
    print(np.array(A))
    
    print("\n--- Jacobian w.r.t Controls (B matrix) [13 x 9] ---")
    print(np.array(B))

    print("\n k matrix")
    print(np.array(K))
    print(Xdot)
