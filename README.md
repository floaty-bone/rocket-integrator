# Rocket Integrator

A Python-based rigid-body dynamics simulator using a 4th-order Runge-Kutta (RK4) integration scheme. Originally developed to simulate the 6-DOF free-flight dynamics of a Starship-class launch vehicle, but designed to be general enough for any rigid body.

## Features

- **RK4 integration** of full 6-DOF equations of motion
- **Quaternion-based attitude kinematics** — no gimbal lock
- **Euler's rigid body rotation equations** for angular dynamics
- **Multi-engine thrust model** — 3 engines in a triangular arrangement, each with independently gimballed thrust expressed in spherical coordinates and converted to body-frame forces and moments
- **Real-time 3D animation** via `matplotlib` — shows the body rotating and translating through space with body-fixed axis indicators and trajectory trail

## State Vector

The simulator propagates a 13-element state vector:

```
[x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
```

| Elements | Description |
|---|---|
| `x, y, z` | Position in inertial frame (m) |
| `qw, qx, qy, qz` | Attitude quaternion (body → inertial) |
| `vx, vy, vz` | Velocity in inertial frame (m/s) |
| `wx, wy, wz` | Angular velocity in body frame (rad/s) |

## Project Structure

```
rocket-integrator/
├── main.py          # Simulation setup, run loop, and 3D animation
├── integrator.py    # Rk4_integrator class
├── helper.py        # Math utilities (quaternion, Ω matrix, thrust transforms)
├── matplottest.py   # Scratch / visualisation tests
└── pyproject.toml   # Project metadata and dependencies
```

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run the simulation
python main.py
```

## Configuration

All tunable parameters live at the top of `main.py`:

| Parameter | Description |
|---|---|
| `get_forceMoment()` | Returns `[Fx, Fy, Fz, Mx, My, Mz]` in the body frame |
| `sim_time` | Total simulation duration (s) |
| `step_size` | RK4 integration step size (s) |
| `sample_rate` | Save every Nth step for animation |
| `initial_state` | Initial 13-element state vector |
| `inertia_matrix` | 3×3 inertia tensor (kg·m²) |

## Physics Overview

### Angular Dynamics
Euler's equation is integrated at each step:

```
dω/dt = I⁻¹ (M − ω × (I ω))
```

### Attitude Kinematics
Quaternion propagation via the Ω matrix:

```
dq/dt = ½ Ω(ω) q
```

### Engine Model
Three engines are positioned at the base of the rocket in a triangular arrangement. Each engine's thrust is specified in spherical coordinates `(θ, φ, r)` and converted to body-frame Cartesian forces. Moments are computed as `Σ (GEᵢ × TEᵢ)`.

## License

MIT
