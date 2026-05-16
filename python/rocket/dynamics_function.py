from helper import compute_thrust_forces_and_moments, quat_to_rotmat, construct_omega_matrix, body_force_to_spherical, compute_thrust_forces_and_moments_cartesian, project_to_tangent_space, E, E_pinv
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
    jac_x_fn = jax.jit(jax.jacfwd(F, argnums=0))
    jac_u_fn = jax.jit(jax.jacfwd(F, argnums=1))
    Q_np = np.array(Q)
    R_np = np.array(R)

    def compute_lqr(xl, ul):
        res_x = jac_x_fn(xl, ul).block_until_ready()
        res_u = jac_u_fn(xl, ul).block_until_ready()

        A13 = np.array(res_x)
        B13 = np.array(res_u)
        
        # Project A13, B13 to 12D tangent space using the helper utility
        A12, B12 = project_to_tangent_space(A13, B13, xl[3:7])
        
        # Project Q13 to 12D tangent space if it is 13x13; otherwise use Q directly
        if Q_np.shape == (13, 13):
            E_mat = E(xl[3:7], scale=0.5)
            Q12 = E_mat.T @ Q_np @ E_mat
        else:
            Q12 = Q_np
        
        # --- Solve LQR on the well-posed 12-state system ---
        K12, S12, E12 = ctrl.lqr(A12, B12, Q12, R_np)
        
        return K12, A12, B12

    return compute_lqr

def rscale(A, B, C, D, K):
    """Feedforward gain N such that u = N*r - K*x tracks reference r at steady state."""
    Acl = A - B @ K
    Ccl = C - D @ K
    M = -Ccl @ np.linalg.inv(Acl) @ B + D
    N = np.linalg.inv(M)
    return N


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
    
    print("\n--- Jacobian w.r.t State (A matrix) [12 x 12] ---")
    print(np.array(A))
    
    print("\n--- Jacobian w.r.t Controls (B matrix) [12 x 9] ---")
    print(np.array(B))

    print("\n k matrix")
    print(np.array(K))
    print(Xdot)

    #sanity check:
    xset=jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float64)
