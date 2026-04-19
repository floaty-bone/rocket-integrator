#pragma once

/**
 * integrator.hpp — 4th-order Runge-Kutta integrator for 6-DOF rigid-body dynamics.
 *
 * State vector layout (13 elements):
 *   [0:3]   x,  y,  z        — position in inertial frame (m)
 *   [3:7]   qw, qx, qy, qz   — attitude quaternion (body → inertial)
 *   [7:10]  vx, vy, vz       — velocity in inertial frame (m/s)
 *   [10:13] wx, wy, wz       — angular velocity in body frame (rad/s)
 */

#include "helper.hpp"

#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <cmath>

namespace rocket {

class RK4Integrator {
public:
    /// Callable signature: returns [Fx, Fy, Fz, Mx, My, Mz] in the body frame.
    using ForceMomentFn = std::function<ForceMoment()>;

    /**
     * Construct the integrator.
     *
     * The inertia inverse is pre-computed here once and reused every step —
     * the same optimisation as the Python implementation.
     *
     * @param get_force_moment  Callable returning the current force/moment vector.
     * @param inertia           3×3 inertia tensor in the body frame (kg·m²).
     * @param mass              Total vehicle mass (kg).          Default 85 000.
     * @param gravity           Gravitational acceleration (m/s²). Default 9.8.
     * @param step_size         Integration time-step Δt (s).     Default 0.001.
     *
     * @throws std::invalid_argument if the inertia matrix is singular.
     */
    explicit RK4Integrator(
        ForceMomentFn          get_force_moment,
        const Eigen::Matrix3d& inertia   = Eigen::Matrix3d::Identity(),
        double                 mass      = 85'000.0,
        double                 gravity   = 9.8,
        double                 step_size = 0.001)
        : get_force_moment_(std::move(get_force_moment))
        , inertia_(inertia)
        , inertia_inv_(inertia.inverse())
        , inv_mass_(1.0 / mass)            // reciprocal: multiply is cheaper than divide per step
        , step_(step_size)
        , gravity_vec_(0.0, 0.0, gravity)  // +Z is "down" by convention
    {
        if (std::abs(inertia.determinant()) < 1e-12)
            throw std::invalid_argument("Inertia matrix is singular or near-singular.");
    }

    /**
     * Advance the state by one time-step using classical RK4.
     *
     * The force/moment callable is evaluated once per step (constant within
     * the step), matching the Python implementation.
     *
     * @param state  Current 13-element state vector.
     * @return       Next 13-element state vector.
     */
    [[nodiscard]] State step_forward(const State& state) const
    {
        const double      h  = step_;
        const ForceMoment fm = get_force_moment_();

        const State k1 = state_derivative(state,               fm);
        const State k2 = state_derivative(state + 0.5*h * k1,  fm);
        const State k3 = state_derivative(state + 0.5*h * k2,  fm);
        const State k4 = state_derivative(state +     h * k3,  fm);

        // Weighted blend: (k1 + 2k2 + 2k3 + k4) / 6
        return state + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
    }

private:
    ForceMomentFn   get_force_moment_;
    Eigen::Matrix3d inertia_;
    Eigen::Matrix3d inertia_inv_;
    double          inv_mass_;    // 1/m — avoids a division inside the hot loop
    double          step_;
    Eigen::Vector3d gravity_vec_;

    // ── Equations of motion ──────────────────────────────────────────────────

    /// dx/dt = v
    [[nodiscard]] static inline Eigen::Vector3d
    d_position(const Eigen::Vector3d& vel) noexcept
    {
        return vel;
    }

    /// dq/dt = ½ Ω(ω) q
    [[nodiscard]] static inline Eigen::Vector4d
    d_quaternion(
        const Eigen::Vector4d& q,
        const Eigen::Vector3d& w) noexcept
    {
        return 0.5 * construct_omega_matrix(w) * q;
    }

    /// dv/dt = (1/m)(R @ F_body) + g_inertial
    [[nodiscard]] inline Eigen::Vector3d
    d_velocity(
        const Eigen::Vector4d& q,
        const ForceMoment&     fm) const noexcept
    {
        return quat_to_rotmat(q) * fm.template head<3>() * inv_mass_ + gravity_vec_;
    }

    /// dω/dt = I⁻¹ (M − ω × (Iω))   — Euler's rotation equation
    [[nodiscard]] inline Eigen::Vector3d
    d_angular_velocity(
        const Eigen::Vector3d& w,
        const ForceMoment&     fm) const noexcept
    {
        const Eigen::Vector3d Iw = inertia_ * w;
        return inertia_inv_ * (fm.template tail<3>() - w.cross(Iw));
    }

    // ── Full state derivative (bundles all four equations) ───────────────────

    [[nodiscard]] inline State
    state_derivative(const State& s, const ForceMoment& fm) const noexcept
    {
        // segment<N>(offset): compile-time size → no heap, no bounds check in release
        const Eigen::Vector3d pos  = s.template segment<3>(0);
        const Eigen::Vector4d quat = s.template segment<4>(3);
        const Eigen::Vector3d vel  = s.template segment<3>(7);
        const Eigen::Vector3d w    = s.template segment<3>(10);

        State ds;
        ds.template segment<3>(0)  = d_position(vel);
        ds.template segment<4>(3)  = d_quaternion(quat, w);
        ds.template segment<3>(7)  = d_velocity(quat, fm);
        ds.template segment<3>(10) = d_angular_velocity(w, fm);
        return ds;
    }
};

} // namespace rocket
