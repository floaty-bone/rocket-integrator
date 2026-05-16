import numpy as np
import math
import sys
import os

# Ensure the parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------------
# Implementations of requested functions and their dependencies
# ---------------------------------------------------------------------------

def L(q: np.ndarray) -> np.ndarray:
    """Left-quaternion multiplication matrix L(q) such that q1_multiply_q2 = L(q1) @ q2.
    
    Assumes quaternion is formatted as [qw, qx, qy, qz].
    """
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q
    return np.array([
        [qw, -qx, -qy, -qz],
        [qx,  qw, -qz,  qy],
        [qy,  qz,  qw, -qx],
        [qz, -qy,  qx,  qw]
    ], dtype=np.float64)

def E(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Tangent space projection matrix E(q) mapping 12D tangent perturbation to 13D state perturbation.
    
    scale: scale factor for quaternion block.
           - scale=0.5: matches G_T = 0.5 * [...] in dynamics_function.py
           - scale=1.0: makes the columns of E(q) orthonormal if q is a unit quaternion
    """
    q = np.asarray(q, dtype=np.float64)
    # Ensure unit quaternion
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    # Tangent space basis for quaternion block (4x3)
    # Maps 3D attitude perturbation to 4D quaternion perturbation
    G_T = scale * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw]
    ], dtype=np.float64)

    E_mat = np.zeros((13, 12), dtype=np.float64)
    E_mat[0:3, 0:3] = np.eye(3)    # position
    E_mat[3:7, 3:6] = G_T          # attitude tangent space
    E_mat[7:10, 6:9] = np.eye(3)   # velocity
    E_mat[10:13, 9:12] = np.eye(3) # angular velocity
    return E_mat

def E_pinv(q: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Left pseudoinverse of the tangent space projection matrix E(q).
    
    E_pinv @ E = I_12.
    """
    q = np.asarray(q, dtype=np.float64)
    q_norm = q / np.linalg.norm(q)
    qw, qx, qy, qz = q_norm

    # The left pseudoinverse of G_T is:
    # (G_T^T G_T)^-1 G_T^T = (scale^2 * I)^-1 * scale * [...] = (1/scale) * G_T^T (unit scale)
    # So it is:
    G_pinv = (1.0 / scale) * np.array([
        [-qx,  qw,  qz, -qy],
        [-qy, -qz,  qw,  qx],
        [-qz,  qy, -qx,  qw]
    ], dtype=np.float64)

    Ep = np.zeros((12, 13), dtype=np.float64)
    Ep[0:3, 0:3] = np.eye(3)
    Ep[3:6, 3:7] = G_pinv
    Ep[6:9, 7:10] = np.eye(3)
    Ep[9:12, 10:13] = np.eye(3)
    return Ep

def qtorp(q: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Convert a quaternion error q = [qw, qx, qy, qz] to Rodrigues parameters (Gibbs vector).
    
    phi = scale * q_v / q_w
    
    - scale=2.0: standard choice so that phi ~= rotation vector in radians for small angles
    - scale=1.0: pure Rodrigues parameters (Gibbs vector)
    """
    q = np.asarray(q, dtype=np.float64)
    qw = q[0]
    qv = q[1:4]
    
    # Handle quaternion double-cover (q and -q represent the same rotation)
    if qw < 0:
        qw = -qw
        qv = -qv
        
    # Avoid division by zero at 180 degrees singularity
    if abs(qw) < 1e-8:
        return scale * qv / 1e-8
        
    return scale * qv / qw

def project_to_tangent_space(A_naive: np.ndarray, B_naive: np.ndarray, q: np.ndarray, scale: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Returns A, B projected to tangent space using E(q)."""
    E_mat = E(q, scale=scale)
    A = E_mat.T @ A_naive @ E_mat
    B = E_mat.T @ B_naive 
    return A, B

def project_to_tangent_space_correct(A_naive: np.ndarray, B_naive: np.ndarray, q: np.ndarray, scale: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Returns A, B correctly projected to tangent space using the pseudoinverse of E(q)."""
    E_mat = E(q, scale=scale)
    Ep = E_pinv(q, scale=scale)
    A = Ep @ A_naive @ E_mat
    B = Ep @ B_naive 
    return A, B

def controller(x: np.ndarray, x_setpoint: np.ndarray, u_hover: np.ndarray, K: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """12D LQR controller using Rodrigues parameter error for attitude."""
    q_setpoint = x_setpoint[3:7]   # reference quaternion
    q  = x[3:7]    # current quaternion

    # Quaternion error in Rodrigues parameters
    # L(q_setpoint).T @ q computes the error quaternion (conjugate(q_setpoint) * q)
    phi = qtorp(L(q_setpoint).T @ q, scale=scale)

    # 12D error state (quaternion replaced by 3D rodrigues error)
    dx = np.concatenate([
        x[0:3]  - x_setpoint[0:3],     # position error    (3)
        phi,                           # attitude error    (3)
        x[7:10] - x_setpoint[7:10],    # velocity error    (3)
        x[10:13] - x_setpoint[10:13],  # ang. velocity err (3)
    ])

    return u_hover - K @ dx


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

def test_quaternion_algebra():
    print("--- Testing Quaternion Algebra ---")
    
    # 1. Identity quaternion
    q_eye = np.array([1.0, 0.0, 0.0, 0.0])
    L_eye = L(q_eye)
    assert np.allclose(L_eye, np.eye(4)), "L(identity) must be identity matrix"
    
    # 2. Multiplication test: q1 * q2
    theta1 = np.radians(30)
    q1 = np.array([np.cos(theta1/2), np.sin(theta1/2), 0.0, 0.0])
    theta2 = np.radians(45)
    q2 = np.array([np.cos(theta2/2), 0.0, np.sin(theta2/2), 0.0])
    
    qw = np.cos(theta1/2)*np.cos(theta2/2)
    qx = np.sin(theta1/2)*np.cos(theta2/2)
    qy = np.cos(theta1/2)*np.sin(theta2/2)
    qz = np.sin(theta1/2)*np.sin(theta2/2)
    q_product_expected = np.array([qw, qx, qy, qz])
    
    q_product_actual = L(q1) @ q2
    assert np.allclose(q_product_actual, q_product_expected), "L(q1) @ q2 must match manual quaternion multiplication"
    print("  [PASS] Quaternion multiplication L(q1) @ q2 matches analytical result.")

    # 3. Orthogonality: L(q).T @ L(q) = ||q||^2 * I
    q_rand = np.random.randn(4)
    q_rand_norm = q_rand / np.linalg.norm(q_rand)
    L_rand = L(q_rand_norm)
    assert np.allclose(L_rand.T @ L_rand, np.eye(4)), "L(unit_q) must be an orthogonal matrix"
    print("  [PASS] L(q) is orthogonal for unit quaternion.")


def test_projection_matrix():
    print("--- Testing Tangent Space Projection Matrix E(q) and Pseudoinverse ---")
    
    # Random unit quaternion
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    
    # E(q) with scale=0.5 and scale=1.0
    E_05 = E(q, scale=0.5)
    E_10 = E(q, scale=1.0)
    Ep_05 = E_pinv(q, scale=0.5)
    Ep_10 = E_pinv(q, scale=1.0)
    
    # 1. Dimension checks
    assert E_05.shape == (13, 12), "E(q) must be of shape 13x12"
    assert Ep_05.shape == (12, 13), "E_pinv(q) must be of shape 12x13"
    
    # 2. Orthonormality check for scale=1.0
    assert np.allclose(E_10.T @ E_10, np.eye(12)), "For scale=1.0, E(q).T @ E(q) must be identity"
    assert np.allclose(Ep_10, E_10.T), "For scale=1.0, Ep must be E.T"
    print("  [PASS] For scale=1.0, E(q) has orthonormal columns and E_pinv == E.T.")
    
    # 3. Pseudoinverse property
    assert np.allclose(Ep_05 @ E_05, np.eye(12)), "Ep @ E must be identity for scale=0.5"
    print("  [PASS] Pseudoinverse property Ep @ E = I_12 holds for scale=0.5.")
    
    # 4. Orthogonality to quaternion itself
    G_T_05 = E_05[3:7, 3:6]
    for i in range(3):
        dot_product = np.dot(q, G_T_05[:, i])
        assert abs(dot_product) < 1e-12, f"Column {i} of quaternion block must be orthogonal to q"
    print("  [PASS] Columns of tangent space block G^T are orthogonal to q.")


def test_project_to_tangent_space():
    print("--- Testing project_to_tangent_space ---")
    
    # Define dummy A_naive (13x13) and B_naive (13x9)
    A_naive = np.random.randn(13, 13)
    B_naive = np.random.randn(13, 9)
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    
    A, B = project_to_tangent_space(A_naive, B_naive, q, scale=0.5)
    
    assert A.shape == (12, 12), "Projected A must be 12x12"
    assert B.shape == (12, 9), "Projected B must be 12x9"
    print("  [PASS] Tangent space projection successfully mapped A (13x13) -> A_proj (12x12) and B (13x9) -> B_proj (12x9).")


def test_qtorp():
    print("--- Testing qtorp (Quaternion to Rodrigues Parameters) ---")
    
    # 1. Identity rotation
    q_eye = np.array([1.0, 0.0, 0.0, 0.0])
    phi_eye = qtorp(q_eye)
    assert np.allclose(phi_eye, np.zeros(3)), "qtorp(identity) must be zero vector"
    
    # 2. Small angle approximation
    theta = 0.02 # small angle (rad)
    u = np.array([1.0, 2.0, 3.0])
    u = u / np.linalg.norm(u)
    q = np.concatenate([[np.cos(theta/2)], np.sin(theta/2) * u])
    
    phi_20 = qtorp(q, scale=2.0)
    phi_10 = qtorp(q, scale=1.0)
    
    assert np.allclose(phi_20, theta * u, rtol=1e-3), "For small angles, qtorp with scale=2.0 must approximate axis-angle"
    assert np.allclose(phi_10, 0.5 * theta * u, rtol=1e-3), "For small angles, qtorp with scale=1.0 must approximate half axis-angle"
    print("  [PASS] Small angle approximations for qtorp hold (both scale=2.0 and scale=1.0).")
    
    # 3. Double-cover check
    phi_pos = qtorp(q)
    phi_neg = qtorp(-q)
    assert np.allclose(phi_pos, phi_neg), "qtorp must handle quaternion double-cover (q and -q must give same phi)"
    print("  [PASS] Quaternion double-cover handled successfully (qtorp(q) == qtorp(-q)).")


def test_controller_equivalence_and_consistency():
    print("--- Testing Controller Consistency and Scaling Invariance ---")
    
    # State vectors
    theta_setpoint = -math.pi / 2
    qw_set = math.cos(theta_setpoint / 2)
    qy_set = math.sin(theta_setpoint / 2)
    x_setpoint = np.array([30.0, 30.0, 30.0,  qw_set, 0.0, qy_set, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])
    
    # State with minor errors
    theta_state = theta_setpoint + 0.05
    qw_st = math.cos(theta_state / 2)
    qy_st = math.sin(theta_state / 2)
    x = np.array([29.8, 30.1, 30.05,  qw_st, 0.05, qy_st, 0.02,  0.1, -0.1, 0.05,  0.01, -0.02, 0.015])
    
    u_hover = np.array([392000.0, 0.0, 0.0, 392000.0, 0.0, 0.0, 392000.0, 0.0, 0.0])
    
    # Let's mock A_naive (13x13) and B_naive (13x9)
    np.random.seed(42)
    A_naive = np.random.randn(13, 13)
    B_naive = np.random.randn(13, 9)
    
    # Correctly project to tangent space under scale = 0.5
    A_05, B_05 = project_to_tangent_space_correct(A_naive, B_naive, x_setpoint[3:7], scale=0.5)
    # Correctly project to tangent space under scale = 1.0 (orthonormal)
    A_10, B_10 = project_to_tangent_space_correct(A_naive, B_naive, x_setpoint[3:7], scale=1.0)
    
    # Define Q and R for both spaces
    Q_naive = np.diag([1e6, 1e6, 1e6,  1e2, 1e2, 1e2, 1e2,  1e5, 1e5, 1e5,  1e1, 1e1, 1e1])
    R = np.diag([1.0] * 9)
    
    E_mat_05 = E(x_setpoint[3:7], scale=0.5)
    E_mat_10 = E(x_setpoint[3:7], scale=1.0)
    Ep_05 = E_pinv(x_setpoint[3:7], scale=0.5)
    Ep_10 = E_pinv(x_setpoint[3:7], scale=1.0)
    
    # Cost function projection: J = delta_X^T Q_naive delta_X = delta_x^T E^T Q_naive E delta_x
    Q_05 = E_mat_05.T @ Q_naive @ E_mat_05
    Q_10 = E_mat_10.T @ Q_naive @ E_mat_10
    
    try:
        import control as ctrl
        K_05, _, _ = ctrl.lqr(A_05, B_05, Q_05, R)
        K_10, _, _ = ctrl.lqr(A_10, B_10, Q_10, R)
        
        # Verify relationship between K_05 and K_10:
        # Since state scale_05 is 0.5 * state_10 (due to scale=0.5 in E(q) vs scale=1.0 in E(q)),
        # the gain K_05 for attitude error should be exactly 2.0 times K_10 for attitude error.
        # Other state gains (pos, vel, omega) should be identical.
        assert np.allclose(K_05[:, 0:3], K_10[:, 0:3]), "Position gains must be identical"
        assert np.allclose(K_05[:, 6:12], K_10[:, 6:12]), "Velocity and omega gains must be identical"
        assert np.allclose(K_05[:, 3:6], 0.5 * K_10[:, 3:6]), "Attitude gains must differ by a factor of 2"
        print("  [PASS] Relationship between K_05 and K_10 matches theoretical scaling when using correct projection.")
        
        # Evaluate controllers
        # For scale=0.5 in E, the corresponding coordinate error phi is scale=2.0 in qtorp (since phi ~= 2*qv)
        u_05 = controller(x, x_setpoint, u_hover, K_05, scale=2.0)
        # For scale=1.0 in E, the corresponding coordinate error phi is scale=1.0 in qtorp (since phi ~= 1*qv)
        u_10 = controller(x, x_setpoint, u_hover, K_10, scale=1.0)
        
        assert np.allclose(u_05, u_10), "Control outputs must be mathematically identical under coordinate scaling"
        print("  [PASS] Control outputs are identical under coordinate scaling (invariant to tangent representation).")
        
        # Now let's check what happens with the original Ritz-Galerkin projection (without pinv):
        A_rg_05 = E_mat_05.T @ A_naive @ E_mat_05
        B_rg_05 = E_mat_05.T @ B_naive
        K_rg_05, _, _ = ctrl.lqr(A_rg_05, B_rg_05, Q_05, R)
        
        print("\n--- Summary of Ritz-Galerkin vs Pseudoinverse Projection discrepancy ---")
        print(f"  Ritz-Galerkin A_rg_05[3:6, 9:12] (attitude to angular vel coupling):")
        print(f"    {A_rg_05[3:6, 9:12]}")
        print(f"  Pseudoinverse A_05[3:6, 9:12] (attitude to angular vel coupling):")
        print(f"    {A_05[3:6, 9:12]}")
        print(f"  Ratio (should be 4.0 due to scale^2 projection vs inverse):")
        print(f"    {A_05[3:6, 9:12] / (A_rg_05[3:6, 9:12] + 1e-15)}")
        
    except ImportError:
        print("  [SKIP] 'control' package not installed, skipping LQR gain scaling verification.")


if __name__ == "__main__":
    test_quaternion_algebra()
    test_projection_matrix()
    test_project_to_tangent_space()
    test_qtorp()
    test_controller_equivalence_and_consistency()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
