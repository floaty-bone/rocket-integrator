import numpy as np
import control as ctrl


def rscale(A, B, C, D, K):
    """Feedforward gain N such that u = N*r - K*x tracks reference r at steady state."""
    Acl = A - B @ K
    Ccl = C - D @ K
    M = -Ccl @ np.linalg.inv(Acl) @ B + D
    N = np.linalg.pinv(M)
    return N


def design_lqr(A, B, C, D, Q, R):
    K, S, E = ctrl.lqr(A, B, Q, R)
    N = rscale(A, B, C, D, K)
    return K, N, S, E
