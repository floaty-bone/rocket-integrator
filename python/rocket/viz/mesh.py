"""
mesh.py — Rocket-body mesh construction for 3-D visualization.

Builds a cylinder + conical-nose mesh anchored to the centre of mass, with
the long axis aligned with body X (the cone tip points in +X).  Returned in
``Poly3DCollection``-friendly form: vertex array, face index list, face colours.

The cylinder side is split into two halves with distinct colours so the
rocket's roll is visually distinguishable.
"""

from __future__ import annotations

import numpy as np

# Body-fixed axis quivers (drawn separately, kept here for proximity).
AXIS_COLORS = ["r", "g", "b"]
AXIS_LABELS = ["X (Roll)", "Y (Pitch)", "Z (Yaw)"]

# Rocket render colours.
CYL_COLOR_A = "white"
CYL_COLOR_B = "lightgray"
NOSE_COLOR  = "crimson"
BASE_COLOR  = "dimgray"


def make_rocket_mesh(
    diameter: float,
    total_height: float,
    nose_height: float,
    base_below_com: float,
    n_seg: int,
) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """Build a rocket mesh (cylinder + conical nose) anchored at the CoM.

    Args:
        diameter:       Cylinder + cone base diameter (m).
        total_height:   Base of cylinder → tip of nose (m).
        nose_height:    Length of the conical nose section (m).
        base_below_com: Distance from the body-frame origin (CoM) to the base
                        of the cylinder along −X.
        n_seg:          Azimuthal mesh resolution.

    Returns:
        ``(vertices, faces, face_colors)`` ready to feed into
        ``matplotlib`` ``Poly3DCollection``.
    """
    radius     = diameter / 2.0
    cyl_length = total_height - nose_height
    x_base     = -base_below_com
    x_shoulder = x_base + cyl_length
    x_tip      = x_base + total_height

    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cy = radius * np.cos(angles)
    cz = radius * np.sin(angles)

    base_ring     = np.column_stack([np.full(n_seg, x_base),     cy, cz])
    shoulder_ring = np.column_stack([np.full(n_seg, x_shoulder), cy, cz])
    tip           = np.array([[x_tip, 0.0, 0.0]])

    vertices = np.vstack([base_ring, shoulder_ring, tip])
    base_idx     = list(range(0, n_seg))
    shoulder_idx = list(range(n_seg, 2 * n_seg))
    tip_idx      = 2 * n_seg

    faces:  list[list[int]] = []
    colors: list[str]       = []

    # Base cap — single n-gon, reversed winding so the outward normal points −X.
    faces.append(list(reversed(base_idx)))
    colors.append(BASE_COLOR)

    # Cylinder side — quad strips, halved for roll visibility.
    half = n_seg // 2
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([base_idx[i], base_idx[j], shoulder_idx[j], shoulder_idx[i]])
        colors.append(CYL_COLOR_A if i < half else CYL_COLOR_B)

    # Conical nose — triangle fan from the tip.
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([shoulder_idx[i], shoulder_idx[j], tip_idx])
        colors.append(NOSE_COLOR)

    return vertices, faces, colors
