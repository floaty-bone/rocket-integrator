"""
vehicle.py — Centralised vehicle parameters (Starship-like reference vehicle).

Single source of truth for the constants shared between the open-loop sim, the
SIL scenario, and the LQR sanity check.  Inertia values are dry-mass estimates
and should be updated once mass-properties data is available.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------

VEHICLE_MASS   = 120_000.0                                # kg
INERTIA_MATRIX = np.diag([1.2e6, 3.5e7, 3.5e7])           # kg·m² — [Ixx, Iyy, Izz]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

GRAVITY_ACCEL = 9.8                                                # m/s² along −Z (inertial "up" is +Z)
GRAVITY_FORCE = np.array([0.0, 0.0, -VEHICLE_MASS * GRAVITY_ACCEL],
                          dtype=np.float64)


# ---------------------------------------------------------------------------
# Engine cluster geometry (equilateral triangle on the base plane)
# ---------------------------------------------------------------------------

ENGINE_CLUSTER_RADIUS = 1.5    # a — distance from rocket centreline to each engine (m)
COM_TO_ENGINE_PLANE   = 18.0   # l — CoM-to-engine-plane distance along body X (m)


# ---------------------------------------------------------------------------
# Body-frame geometry used by the open-loop sim's integrator constructor
# (these match the historical defaults from main.py).
# ---------------------------------------------------------------------------

BODY_LENGTH = 4.0   # m — X extent
BODY_WIDTH  = 2.0   # m — Y extent
BODY_HEIGHT = 1.0   # m — Z extent


# ---------------------------------------------------------------------------
# Render geometry (cylinder + conical nose, long axis along body X)
# ---------------------------------------------------------------------------

ROCKET_DIAMETER       = 9.0    # m — cylinder + cone base diameter
ROCKET_HEIGHT         = 50.0   # m — base of cylinder → tip of nose, along body X
ROCKET_NOSE_HEIGHT    = 10.0   # m — length of conical nose section
ROCKET_BASE_BELOW_COM = 20.0   # m — CoM-to-base distance along −X
N_CIRCLE_SEGMENTS     = 32     # azimuthal mesh resolution


def hover_thrust_per_engine(n_engines: int = 3) -> float:
    """Return the per-engine thrust (N) that exactly counteracts gravity."""
    return (VEHICLE_MASS * GRAVITY_ACCEL) / n_engines
