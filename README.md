# Rocket Integrator

A 6-DOF rigid-body flight simulator and LQR guidance/control stack for a
Starship-class booster, built around a hand-rolled 4th-order Runge–Kutta
integrator. The Python package `rocket` is the heart of the project: it models
the vehicle dynamics, synthesises an LQR attitude/translation controller via
JAX autodiff, runs a software-in-the-loop (SIL) closed-loop simulation, and
streams the resulting trajectory in real time to a Three.js renderer.

The same dynamics core has a standalone C++ port (`cpp/`), a browser renderer
(`rendering/`), and is embedded as an interactive demo in a personal portfolio
site (`portfolio/`).

---

## Highlights

- **Vehicle-agnostic RK4 integrator** — propagates a 13-element rigid-body state
  with zero heap allocation in the hot loop; couples to a vehicle only through
  two force callbacks, so the same integrator can drive a rocket, a CubeSat, or
  a quadrotor.
- **Quaternion attitude kinematics** — `dq/dt = ½ Ω(ω) q`, no gimbal lock, with
  periodic re-normalisation against numerical drift.
- **Multi-engine thrust model** — three engines in an equilateral triangle, each
  gimballed, with spherical `[θ, φ, T]` ↔ Cartesian ↔ TVC `[α, β, T]`
  conversions and summation into a body wrench.
- **LQR on a 12-D tangent space** — the dynamics are linearised with
  `jax.jacfwd`, projected from the 13-D state onto the 12-D quaternion tangent
  space, and solved via `python-control`. The runtime feedback law uses
  Rodrigues (Gibbs) attitude error so the controller never sees the quaternion
  discontinuity.
- **Real-time SIL loop** — plant and controller run as separate blocks across a
  realistic `[α, β, T]` gimbal hardware boundary at up to 5 kHz, then the
  recorded trajectory is streamed over WebSocket to the 3-D renderer.
- **Dual NumPy / JAX backends** — pure NumPy for the tight integration loop, JAX
  for everything that needs to be autodifferentiable.

---

## Repository layout

```
rocket-integrator/
├── python/          # ⭐ the rocket package — dynamics, control, SIL, viz
├── cpp/             # C++ port of the integrator (CSV output)
├── rendering/       # Three.js / Vite renderer + WebSocket trajectory player
├── portfolio/       # React front-end + FastAPI backend (embeds the demo)
└── README.md
```

---

## The `python/` package (`rocket`)

```
python/rocket/
├── math/            # Pure math primitives — no vehicle knowledge
│   ├── quaternion.py     # quat→rotmat, Ω(ω) matrix, L(q)  (NumPy + JAX)
│   ├── tangent_space.py  # E(q), E_pinv(q), qtorp, 13→12 projection
│   └── types.py          # shape-annotated state / wrench type aliases
│
├── plant/           # Vehicle physics
│   ├── dynamics.py       # make_F → autodiff-friendly ẋ = F(x, u)
│   ├── thrust.py         # multi-engine wrench model (NumPy + JAX)
│   └── tvc.py            # Cartesian ↔ [α, β, T] gimbal conversions
│
├── integration/
│   └── integrator.py     # RK4Integrator — vehicle-agnostic 6-DOF propagator
│
├── control/         # LQR design (slow) + feedback law (fast)
│   ├── lqr_design.py     # Jacobians, tangent projection, Riccati solve
│   ├── attitude_law.py   # u = u_hover − K·δx with Rodrigues attitude error
│   └── lqr_controller.py # stateful controller bundling design + runtime
│
├── viz/             # Visualisation / streaming
│   ├── animation.py      # interactive 3-D matplotlib animation
│   ├── mesh.py           # cylinder + conical-nose rocket mesh
│   ├── plots.py          # 2-D thrust & trajectory-tracking diagnostics
│   └── ws_server.py      # WebSocket server that replays a run in real time
│
├── config/
│   └── vehicle.py        # single source of truth for vehicle parameters
│
├── scenarios/       # Executable end-to-end demos (entry points)
│   ├── open_loop.py       # constant-wrench free flight + animation
│   ├── sil_hover.py       # closed-loop LQR hover & translate + WS streaming
│   └── lqr_sanity_check.py# one-shot gain / control-output check
│
├── tests/           # self-contained smoke / unit tests
└── docs/
    └── controller-twitching.md  # a real debugging write-up (see below)
```

### Architecture in one paragraph

The **integrator** knows nothing about rockets — it asks two callbacks for a
body wrench `[Fx, Fy, Fz, Mx, My, Mz]` and an inertial force `[Fx, Fy, Fz]`
(gravity) and propagates the state. The **plant** provides the rocket-specific
wrench by summing three gimballed engines. The **controller** is split in two:
a slow *design* side (`lqr_design`) that linearises `F` with JAX, projects the
13×13 / 13×9 Jacobians onto the 12-D quaternion tangent space, and solves the
LQR Riccati equation; and a fast *runtime* side (`attitude_law`) that applies
the precomputed gain `K` to a Rodrigues-parameter error state. The **scenarios**
wire these together and hand the recorded trajectory to **viz** for plotting,
animation, or real-time WebSocket streaming.

### State and control vectors

The 13-element state vector:

```
[ x, y, z,  qw, qx, qy, qz,  vx, vy, vz,  wx, wy, wz ]
```

| Slice     | Elements          | Description                            |
|-----------|-------------------|----------------------------------------|
| `[0:3]`   | `x, y, z`         | Position, inertial frame (m)           |
| `[3:7]`   | `qw, qx, qy, qz`  | Attitude quaternion (body → inertial)  |
| `[7:10]`  | `vx, vy, vz`      | Velocity, inertial frame (m/s)         |
| `[10:13]` | `wx, wy, wz`      | Angular velocity, body frame (rad/s)   |

The 9-element control vector is per-engine thrust. Internally the controller
works in Cartesian body force `[X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]`; the SIL
hardware boundary uses the real TVC representation `[α, β, T]` per engine
(α = pitch about body Y, β = yaw about body Z, T = throttle in N).

### Physics

```
Translation :  dv/dt = (R(q) · F_body + F_inertial) / m
Attitude    :  dq/dt = ½ Ω(ω) q
Rotation    :  dω/dt = I⁻¹ ( M − ω × (I ω) )          (Euler's equation)
```

Each engine's thrust is converted to a body-frame force and contributes a
moment `GEᵢ × Tᵢ` about the centre of mass G; the three are summed into the body
wrench.

---

## Quick start

The Python project uses [uv](https://docs.astral.sh/uv/) (an `uv.lock` is
checked in). Python 3.13 is pinned via `.python-version`; the package itself
supports 3.10+.

```bash
cd python

# create the environment and install dependencies from the lockfile
uv sync

# --- run a scenario ---

# 1. Open-loop free flight with an interactive 3-D matplotlib animation
uv run simulate                      # = python -m rocket.scenarios.open_loop

# 2. Closed-loop LQR hover & translate, then stream the run over WebSocket
uv run simulate-sil                  # = python -m rocket.scenarios.sil_hover

# 3. One-shot LQR design sanity check (prints per-engine gimbal commands)
uv run python -m rocket.scenarios.lqr_sanity_check
```

The two `simulate` / `simulate-sil` commands are declared as project scripts in
`pyproject.toml`.

### Watching the SIL run in 3-D

`simulate-sil` first runs the full closed-loop simulation, prints timing and
tracking diagnostics, pops up the 2-D thrust/trajectory plots, and then starts a
WebSocket server on `ws://localhost:8765`. To see the booster fly:

```bash
cd rendering
npm install
npm run dev          # Vite dev server
```

Open the renderer in a browser and press **P** to connect to the WebSocket and
start real-time playback (press **T** to play the built-in scripted trajectory
instead). The renderer animates the booster mesh, gimballing engines, grid fins,
and live telemetry plots.

---

## Tests

The tests are self-contained scripts (no pytest harness required):

```bash
cd python
uv run python -m rocket.tests.test_tangent_controller   # quaternion / tangent-space algebra
uv run python -m rocket.tests.test_sil                  # end-to-end controller smoke test
uv run python -m rocket.tests.test_lqr                  # LQR / controllability check
```

`test_tangent_controller` is the most thorough — it verifies the `L(q)`
quaternion algebra, the `E(q)` tangent-space basis and its pseudoinverse, the
`qtorp` Rodrigues conversion (including the quaternion double-cover), and that
the LQR feedback law is invariant under the tangent-space scale convention.

---

## Configuration

All vehicle parameters live in `python/rocket/config/vehicle.py` (mass, inertia,
gravity, engine cluster geometry, render dimensions). Scenario-specific knobs
(simulation time, frequency, Q/R weights, initial conditions, setpoints) live in
the `CONFIG` block at the top of each file in `scenarios/`.

The reference vehicle is Starship-like: 300 t, `I = diag(4.3e6, 1.9e8, 1.9e8)`
kg·m², three engines on a 1.5 m radius, 35 m aft of the CoM.

---

## Engineering notes

`python/docs/controller-twitching.md` is a worked debugging write-up of a real
control issue: the LQR re-linearisation at 80 Hz produced visible high-frequency
"twitching", especially in roll. It documents what was ruled out (the gimbal
`atan2` boundary), the live suspects (K-jumps between re-linearisations,
gyroscopic coupling, quaternion sign flips), the diagnostic ladder, and the
measured result — freezing `K` cut twitch energy ~40× while converging only ~2 s
slower. Worth reading as an example of how the control stack behaves in practice.

---

## Other components

### `cpp/` — C++ port

A direct, optimised C++20 translation of the RK4 integrator that writes a
`trajectory.csv` instead of animating. Eigen is fetched automatically by CMake.
See `cpp/README.md` for build/run details.

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/rocket_sim          # → trajectory.csv
```

### `rendering/` — Three.js renderer

A Vite + TypeScript renderer that loads GLB models of the booster, engines, and
grid fins and animates them either from a scripted trajectory (**T**) or from a
live WebSocket feed of a SIL run (**P**). `rendering/EXPORT_PIPELINE.md`
documents the Creo → Blender → glTF asset pipeline, including the exact part
names and pivot placements the renderer expects.

### `portfolio/` — personal site

A React/Vite front-end (`portfolio/frontend`) with a FastAPI backend
(`portfolio/backend`) that embeds the rocket demo as an interactive page, with
backend-free baked playback for static hosting.

---

## License

MIT
