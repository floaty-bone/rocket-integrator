# Rocket Integrator

A Python-based rigid-body dynamics simulator using a 4th-order Runge-Kutta (RK4) integration scheme. Originally developed to simulate the 6-DOF free-flight dynamics of a Starship-class launch vehicle, but designed to be general enough for any rigid body.

## Features

- **RK4 integration** of full 6-DOF equations of motion
- **Quaternion-based attitude kinematics** — no gimbal lock
- **Euler's rigid body rotation equations** for angular dynamics
- **Multi-engine thrust model** — 3 engines in a triangular arrangement, each with independently gimballed thrust expressed in spherical coordinates and converted to body-frame forces and moments
- **Real-time 3D animation** via `matplotlib` — shows the body rotating and translating through space with body-fixed axis indicators and a trajectory trail

---

## Project Structure

```
rocket-integrator/
├── main.py          # Simulation config, run loop, and 3D animation
├── integrator.py    # RK4Integrator class
├── helper.py        # Math utilities (quaternion ops, Ω matrix, thrust model)
├── scratch/
│   └── matplottest.py   # Early visualisation prototype (not part of the sim)
└── pyproject.toml   # Project metadata and dependencies
```

---

## Quick Start

```bash
# Install dependencies (uv recommended)
uv pip install numpy matplotlib scipy

# Run the simulation
python main.py
```

---

## Configuration

All tunable parameters live in the `CONFIG` block at the top of `main.py`:

| Parameter        | Description                                              |
|------------------|----------------------------------------------------------|
| `FORCE_MOMENT`   | `[Fx, Fy, Fz, Mx, My, Mz]` in the body frame (N / N·m) |
| `INITIAL_STATE`  | 13-element state vector at t = 0                         |
| `INERTIA_MATRIX` | 3×3 inertia tensor (kg·m²)                               |
| `SIM_TIME`       | Total simulation duration (s)                            |
| `STEP_SIZE`      | RK4 integration time-step (s)                            |
| `SAMPLE_RATE`    | Save every Nth step for animation                        |
| `BODY_LENGTH/WIDTH/HEIGHT` | Cuboid dimensions for the visualiser (m)       |

---

## State Vector

The simulator propagates a 13-element state vector:

```
[x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz]
```

| Slice   | Elements           | Description                           |
|---------|--------------------|---------------------------------------|
| `[0:3]`  | `x, y, z`         | Position in inertial frame (m)        |
| `[3:7]`  | `qw, qx, qy, qz`  | Attitude quaternion (body → inertial) |
| `[7:10]` | `vx, vy, vz`      | Velocity in inertial frame (m/s)      |
| `[10:13]`| `wx, wy, wz`      | Angular velocity in body frame (rad/s)|

---

## Physics

### Angular Dynamics — Euler's Rotation Equation
```
dω/dt = I⁻¹ (M − ω × (I ω))
```

### Attitude Kinematics — Quaternion Propagation
```
dq/dt = ½ Ω(ω) q
```

### Engine Thrust Model
Three engines are mounted at the base in an equilateral triangle.  Each engine's
thrust is specified in spherical coordinates `(θ, φ, r)`, converted to body-frame
Cartesian forces, and its moment about G is computed as `GEᵢ × TEᵢ`.

---

## License

MIT
