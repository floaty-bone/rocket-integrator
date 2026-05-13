class FullStateFeedback:
    def __init__(self, K, N):
        self.K = K
        self.N = N

    def _compute_u(self, state, setpoint):
        return self.N * setpoint - self.K @ state

    def update(self, state, setpoint, **kwargs):
        """
        Override to extend the linear law with additional terms, e.g.:
        - feedforward: a known disturbance (gravity, thrust bias) added to cancel it before feedback acts
        - nonlinear compensation: terms the LQR misses because it was designed on a linearized model,
          such as gyroscopic coupling in body rates: I⁻¹(ω × Iω)
        """
        return self._compute_u(state, setpoint)
