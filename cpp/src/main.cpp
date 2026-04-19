/**
 * main.cpp — Simulation entry point and CSV output for the rocket RK4 integrator.
 *
 * Usage:
 *   ./rocket_sim [output_file.csv]
 *
 * Outputs a CSV with one row per sampled step:
 *   time, x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz
 *
 * All tunable parameters are grouped under the CONFIG section below.
 */

#include "integrator.hpp"
#include "helper.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// =============================================================================
// CONFIG — edit these values to change the simulation scenario
// =============================================================================

namespace cfg {

// Applied force/moment in body frame: [Fx, Fy, Fz, Mx, My, Mz]
//   Pure roll  : {0, 0, 0, 650,   0,   0}
//   Pure pitch : {0, 0, 0,   0, 700,   0}
//   Pure yaw   : {0, 0, 0,   0,   0, 550}
//   Combined   : {0, 0, 0, 650, 700, 550}  ← default
inline const rocket::ForceMoment FORCE_MOMENT =
    (rocket::ForceMoment() << 0.0, 0.0, 0.0, 650.0, 700.0, 550.0).finished();

// Initial state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
inline const rocket::State INITIAL_STATE =
    (rocket::State() << 0.0, 0.0, 0.0,   // position
                        1.0, 0.0, 0.0, 0.0,   // quaternion (identity)
                        0.0, 0.0, 0.0,   // velocity
                        4.0, 0.01, 0.01  // angular velocity
    ).finished();

// Inertia tensor (kg·m²) — diagonal for a symmetric vehicle
inline const Eigen::Matrix3d INERTIA_MATRIX =
    Eigen::DiagonalMatrix<double, 3>(300.0, 100.0, 30.0).toDenseMatrix();

constexpr double SIM_TIME    = 100.0;    // total simulation duration (s)
constexpr double STEP_SIZE   = 0.001;    // RK4 time-step Δt (s)
constexpr int    SAMPLE_RATE = 10;       // write one row every N integration steps
constexpr double MASS        = 85'000.0; // vehicle mass (kg)
constexpr double GRAVITY     = 9.8;      // gravitational acceleration (m/s²)

// Quaternion renormalisation period (steps).
// Drift from floating-point accumulation is tiny but non-zero over 100 s / 0.001 s = 100 000 steps.
// Re-normalise every RENORM_INTERVAL steps at essentially zero cost.
constexpr int RENORM_INTERVAL = 500;

} // namespace cfg

// =============================================================================
// CSV helpers
// =============================================================================

/// Write the CSV header row.
static void write_header(std::FILE* fp)
{
    std::fputs("time,x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz\n", fp);
}

/// Write one data row.  Using fprintf with a fixed format string is faster than
/// std::ofstream << for large numerical outputs.
static void write_row(std::FILE* fp, double t, const rocket::State& s)
{
    std::fprintf(fp,
        "%.6f,"
        "%.9f,%.9f,%.9f,"
        "%.9f,%.9f,%.9f,%.9f,"
        "%.9f,%.9f,%.9f,"
        "%.9f,%.9f,%.9f\n",
        t,
        s[0],  s[1],  s[2],
        s[3],  s[4],  s[5],  s[6],
        s[7],  s[8],  s[9],
        s[10], s[11], s[12]);
}

// =============================================================================
// Entry point
// =============================================================================

int main(int argc, char* argv[])
{
    const char* output_path = (argc > 1) ? argv[1] : "trajectory.csv";

    // ── Open output file ─────────────────────────────────────────────────────
    std::FILE* fp = std::fopen(output_path, "w");
    if (!fp) {
        std::fprintf(stderr, "Error: cannot open \"%s\" for writing.\n", output_path);
        return 1;
    }

    // ── Build integrator ─────────────────────────────────────────────────────
    rocket::RK4Integrator integrator(
        []() -> rocket::ForceMoment { return cfg::FORCE_MOMENT; },
        cfg::INERTIA_MATRIX,
        cfg::MASS,
        cfg::GRAVITY,
        cfg::STEP_SIZE
    );

    // ── Pre-flight summary ───────────────────────────────────────────────────
    const long total_steps  = static_cast<long>(cfg::SIM_TIME / cfg::STEP_SIZE);
    const long frames_out   = total_steps / cfg::SAMPLE_RATE + 1;

    std::printf("Rocket RK4 Integrator — 6-DOF rigid body\n");
    std::printf("  Simulation time  : %.1f s\n",   cfg::SIM_TIME);
    std::printf("  Step size        : %.4f s\n",   cfg::STEP_SIZE);
    std::printf("  Total steps      : %ld\n",       total_steps);
    std::printf("  Sample rate      : every %d steps\n", cfg::SAMPLE_RATE);
    std::printf("  Rows to write    : %ld\n",        frames_out);
    std::printf("  Output           : %s\n\n",      output_path);

    // ── Run simulation ───────────────────────────────────────────────────────
    write_header(fp);

    rocket::State state = cfg::INITIAL_STATE;
    write_row(fp, 0.0, state);   // write t = 0

    for (long step = 1; step <= total_steps; ++step) {
        state = integrator.step_forward(state);

        // Periodic quaternion renormalisation — keeps attitude numerically clean
        // over long runs without touching the physics equations.
        if (step % cfg::RENORM_INTERVAL == 0) {
            state.template segment<4>(3).normalize();
        }

        if (step % cfg::SAMPLE_RATE == 0) {
            write_row(fp, step * cfg::STEP_SIZE, state);
        }
    }

    std::fclose(fp);

    // ── Post-flight sanity check ─────────────────────────────────────────────
    const double q_norm = state.template segment<4>(3).norm();

    std::printf("Simulation complete.\n");
    std::printf("  Rows written         : %ld\n",   frames_out);
    std::printf("  Final quaternion norm: %.8f  (ideal = 1.0)\n", q_norm);
    std::printf("  Final position       : [%.3f, %.3f, %.3f] m\n",
                state[0], state[1], state[2]);
    std::printf("  Final velocity       : [%.3f, %.3f, %.3f] m/s\n",
                state[7], state[8], state[9]);

    return 0;
}
