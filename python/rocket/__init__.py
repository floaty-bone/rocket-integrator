"""
rocket — 6-DOF rigid-body dynamics simulator with LQR attitude/translation control.

Sub-packages
------------
math        Pure mathematical primitives (quaternion ops, tangent-space basis).
plant       Vehicle physics (thrust model, TVC conversions, continuous dynamics).
integration Vehicle-agnostic RK4 integrator for 6-DOF rigid-body dynamics.
control     LQR design + Rodrigues-error feedback law.
viz         3-D animation and 2-D diagnostic plots.
config      Centralised vehicle parameters.
scenarios   Executable end-to-end demos (open-loop, SIL hover, LQR sanity check).
tests       Self-contained unit / smoke tests.

Curated re-exports for short common imports follow.
"""

from rocket.control.lqr_controller import LQRController
from rocket.plant.dynamics import make_F
from rocket.integration.integrator import RK4Integrator

__all__ = ["LQRController", "make_F", "RK4Integrator"]
