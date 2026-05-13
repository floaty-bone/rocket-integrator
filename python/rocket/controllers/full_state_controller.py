from .base import FullStateFeedback
from .lqr import design_lqr
import numpy as np

INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7]).astype(np.float64)  # [Ixx, Iyy, Izz] #empty estimates for starship (dry), should be updated with more accurate values when available


A=np.zeros((3, 3), dtype=np.float64)  
B=np.linalg.inv(INERTIA_MATRIX)
C=np.eye(3, dtype=np.float64)
D=np.zeros((3, 3), dtype=np.float64)
Q=np.diag([1e6, 1e6, 1e6]).astype(np.float64)
R=np.diag([1e-3, 1e-3, 1e-3]).astype(np.float64)

if __name__ == "__main__":
    K, N, S, E = design_lqr(A, B, C, D, Q, R)
    print(f"K = np.array({K.tolist()})\nN = {N}")



class BodyRateController(FullStateFeedback):
    """
    Full-state feedback body rate controller.
    Setpoint: commanded angular velocity vector in body frame.
    Receives full state but only uses the angular velocity components.
    """
    def __init__(self, K, N):
        super().__init__(K, N)

    def update(self, state, setpoint, **kwargs):
        return self._compute_u(state, setpoint)
