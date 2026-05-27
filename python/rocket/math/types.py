"""
types.py — Type aliases for the 6-DOF state vector and force/moment quantities.

These aliases attach an expected NumPy shape to the underlying ndarray for
documentation and static-checker hints; they are not runtime-enforced.

State vector layout (13 elements):
    [0:3]   x, y, z        — position in inertial frame (m)
    [3:7]   qw, qx, qy, qz — attitude quaternion (body → inertial)
    [7:10]  vx, vy, vz     — velocity in inertial frame (m/s)
    [10:13] wx, wy, wz     — angular velocity in body frame (rad/s)
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import numpy.typing as npt

StateVector           = Annotated[npt.NDArray[np.float64], (13,)]
EnginesThrustArray    = Annotated[npt.NDArray[np.float64], (3, 3)]  # [theta, phi, thrust] per engine
BodyWrench            = Annotated[npt.NDArray[np.float64], (6,)]    # [Fx, Fy, Fz, Mx, My, Mz] — body frame
InertialForceVector   = Annotated[npt.NDArray[np.float64], (3,)]    # [Fx, Fy, Fz] — inertial frame (N)
AngularVelocityVector = Annotated[npt.NDArray[np.float64], (3,)]    # [wx, wy, wz]  (body frame)
QuaternionVector      = Annotated[npt.NDArray[np.float64], (4,)]    # [qw, qx, qy, qz]
