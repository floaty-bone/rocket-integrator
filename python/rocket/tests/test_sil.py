"""Smoke test for the LQR controller end-to-end at a hover-with-vertical
attitude operating point."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rocket.config.vehicle import (
    COM_TO_ENGINE_PLANE,
    ENGINE_CLUSTER_RADIUS,
    GRAVITY_FORCE,
    INERTIA_MATRIX,
    VEHICLE_MASS,
    hover_thrust_per_engine,
)
from rocket.control.lqr_controller import LQRController
from rocket.plant.dynamics import make_F


def test() -> None:
    print("Test start")
    F = make_F(
        VEHICLE_MASS, INERTIA_MATRIX, GRAVITY_FORCE,
        ENGINE_CLUSTER_RADIUS, COM_TO_ENGINE_PLANE,
    )
    controller = LQRController(F)
    theta = -math.pi / 2
    qw = math.cos(theta / 2)
    qy = math.sin(theta / 2)
    state = np.array(
        [0, 0, 0,  qw, 0, qy, 0,  0, 0, 0,  0, 0, 0],
        dtype=np.float64,
    )
    hover = hover_thrust_per_engine()
    u_nominal = np.array([hover, 0, 0, hover, 0, 0, hover, 0, 0])
    print("Linearizing...")
    controller.update_linearization(jnp.array(state), jnp.array(u_nominal))
    print("Done linearizing")
    u = controller.update(state)
    print(f"Control: {u[:3]}...")
    print("Test finished successfully")


if __name__ == "__main__":
    test()
