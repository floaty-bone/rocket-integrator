# rocket-integrator — C++ implementation

A direct, optimised C++ translation of the Python 6-DOF RK4 integrator.
Produces a `trajectory.csv` instead of an animation window.

## Requirements

| Tool    | Version   | Notes                                      |
|---------|-----------|--------------------------------------------|
| CMake   | ≥ 3.20    | [cmake.org](https://cmake.org)             |
| C++ compiler | C++20 | GCC 11+, Clang 13+, or MSVC 2022       |
| Git     | any       | Eigen is fetched automatically at build time |
| Internet | first build only | Eigen 3.4.0 is downloaded via FetchContent |

No manual library installation required — Eigen is header-only and downloaded automatically.

---

## Build

```bash
# from the cpp/ directory
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The binary is placed at `build/rocket_sim` (Linux/macOS) or `build/Release/rocket_sim.exe` (Windows).

---

## Run

```bash
# default output: trajectory.csv in the current directory
./build/rocket_sim

# custom output path
./build/rocket_sim my_run.csv
```

---

## Output format

`trajectory.csv` — one row per sampled integration step.

| Column | Description | Unit |
|--------|-------------|------|
| `time` | Simulation time | s |
| `x`, `y`, `z` | Position in inertial frame | m |
| `qw`, `qx`, `qy`, `qz` | Attitude quaternion (body → inertial) | — |
| `vx`, `vy`, `vz` | Velocity in inertial frame | m/s |
| `wx`, `wy`, `wz` | Angular velocity in body frame | rad/s |

---

## Configuring the simulation

All parameters are in the `cfg` namespace at the top of `src/main.cpp`:

| Parameter      | Default | Description |
|----------------|---------|-------------|
| `FORCE_MOMENT` | `{0,0,0,650,700,550}` | Body-frame force/moment (N, N·m) |
| `INITIAL_STATE`| identity quaternion, ω = (4, 0.01, 0.01) | Initial 13-element state |
| `INERTIA_MATRIX` | diag(300, 100, 30) | Inertia tensor (kg·m²) |
| `SIM_TIME`     | 100 s  | Total simulation duration |
| `STEP_SIZE`    | 0.001 s | RK4 time-step Δt |
| `SAMPLE_RATE`  | 10     | Write every N steps to CSV |
| `MASS`         | 85 000 kg | Vehicle mass |
| `GRAVITY`      | 9.8 m/s² | Gravitational acceleration |
| `RENORM_INTERVAL` | 500 | Re-normalise quaternion every N steps |

---

## Project structure

```
cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── helper.hpp       # Math utilities (quaternion, omega matrix, engine model)
│   └── integrator.hpp   # RK4Integrator class
└── src/
    └── main.cpp         # Simulation loop + CSV output
```
