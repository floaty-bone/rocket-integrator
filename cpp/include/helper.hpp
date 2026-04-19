#pragma once

/**
 * helper.hpp — Math utilities for the rocket RK4 integrator.
 *
 * Provides:
 *   - State / ForceMoment type aliases (fixed-size Eigen vectors, stack-allocated)
 *   - Quaternion → rotation matrix conversion
 *   - Omega matrix construction for quaternion kinematics
 *   - Engine spherical-coordinate → body-frame force conversion
 *   - Multi-engine thrust and moment resultant computation
 */

#include <Eigen/Dense>
#include <cmath>
#include <numbers>   // std::numbers::pi  (C++20)
#include <stdexcept>

namespace rocket {

// ─── Type aliases ────────────────────────────────────────────────────────────
//
//  All fixed-size: the compiler knows every dimension at compile time, so Eigen
//  allocates everything on the stack and can fully unroll/vectorise the math.

/// Full 13-element state vector: [x y z | qw qx qy qz | vx vy vz | wx wy wz]
using State = Eigen::Matrix<double, 13, 1>;

/// Body-frame force / moment: [Fx Fy Fz Mx My Mz]
using ForceMoment = Eigen::Matrix<double, 6, 1>;

/// Per-engine parameters stored row-major so row i = [theta_i, phi_i, thrust_i]
using EngineArray = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;


// ─── Quaternion utilities ─────────────────────────────────────────────────────

/**
 * Convert a unit quaternion q = [qw, qx, qy, qz] to a 3×3 rotation matrix.
 *
 * The returned matrix R satisfies:
 *     v_inertial = R * v_body
 *
 * The input is normalised internally; it need not arrive pre-normalised.
 */
[[nodiscard]] inline Eigen::Matrix3d
quat_to_rotmat(const Eigen::Ref<const Eigen::Vector4d>& q) noexcept
{
    // Normalise defensively (unit cost for the safety it buys)
    const Eigen::Vector4d qn = q * (1.0 / q.norm());
    const double w  = qn[0], x = qn[1], y = qn[2], z = qn[3];

    const double x2 = x*x, y2 = y*y, z2 = z*z;
    const double xy = x*y, xz = x*z, yz = y*z;
    const double wx = w*x, wy = w*y, wz = w*z;

    Eigen::Matrix3d R;
    R << 1.0 - 2.0*(y2+z2),       2.0*(xy - wz),       2.0*(xz + wy),
           2.0*(xy + wz),     1.0 - 2.0*(x2+z2),       2.0*(yz - wx),
           2.0*(xz - wy),         2.0*(yz + wx),   1.0 - 2.0*(x2+y2);
    return R;
}

/**
 * Build the 4×4 Omega matrix for the quaternion kinematic equation:
 *
 *     dq/dt = ½ Ω(ω) q
 *
 * @param w  Angular velocity vector [wx, wy, wz] in the body frame (rad/s).
 */
[[nodiscard]] inline Eigen::Matrix4d
construct_omega_matrix(const Eigen::Ref<const Eigen::Vector3d>& w) noexcept
{
    const double wx = w[0], wy = w[1], wz = w[2];

    Eigen::Matrix4d Om;
    Om <<  0.0, -wx,  -wy,  -wz,
            wx,  0.0,  wz,  -wy,
            wy,  -wz,  0.0,  wx,
            wz,   wy,  -wx,  0.0;
    return Om;
}


// ─── Engine thrust model ──────────────────────────────────────────────────────

/**
 * Convert a single engine's thrust from spherical to body-frame Cartesian.
 *
 * @param theta   Gimbal angle θ (rad)
 * @param phi     Gimbal angle φ (rad)
 * @param thrust  Thrust magnitude (N)
 *
 * @return Force vector [Fx, Fy, Fz] in the body frame.
 */
[[nodiscard]] inline Eigen::Vector3d
engine_spherical_to_body_force(double theta, double phi, double thrust) noexcept
{
    return {
         thrust * std::cos(phi),
        -thrust * std::sin(phi) * std::sin(theta),
        -thrust * std::cos(theta) * std::sin(phi),
    };
}

/**
 * Compute the total body-frame force and moment from three gimballed engines.
 *
 * Engines are arranged in an equilateral triangle at the base of the rocket.
 * Moments are taken about the centre of mass G.
 *
 * @param engines  (3×3) row-major matrix; row i = [theta_i, phi_i, thrust_i]
 * @param a        Engine cluster radius — centreline to each engine (m)
 * @param l        Distance from CoM to engine plane along the X axis (m)
 *
 * @return 6-element vector [Fx, Fy, Fz, Mx, My, Mz] in the body frame.
 */
[[nodiscard]] inline ForceMoment
compute_thrust_forces_and_moments(
    const EngineArray& engines,
    double a,
    double l) noexcept
{
    constexpr double pi = std::numbers::pi;

    // ── Resultant force ──────────────────────────────────────────────────────
    Eigen::Vector3d F = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        F += engine_spherical_to_body_force(
                engines(i, 0),   // theta
                engines(i, 1),   // phi
                engines(i, 2));  // thrust
    }

    // ── Engine attachment points (equilateral triangle in the base plane) ────
    //   GE[i] = vector from CoM G to engine attachment point E_i, body frame
    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> GE = (
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>() <<
            -l, -a * std::cos(pi / 6.0),  a * std::cos(pi / 3.0),
            -l,  a * std::cos(pi / 6.0),  a * std::cos(pi / 3.0),
            -l,  0.0,                     -a
    ).finished();

    // ── Total moment: Σ GE_i × F_i ──────────────────────────────────────────
    Eigen::Vector3d M = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        M += GE.row(i).transpose().cross(
                engine_spherical_to_body_force(
                    engines(i, 0),
                    engines(i, 1),
                    engines(i, 2)));
    }

    ForceMoment fm;
    fm << F, M;
    return fm;
}

} // namespace rocket
