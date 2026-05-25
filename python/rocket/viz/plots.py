"""
plots.py — 2-D diagnostic plots for the SIL simulation.

Two figures are produced:
    * :func:`plot_thrust`             — per-engine α (pitch), β (yaw), T (thrust) vs time.
    * :func:`plot_trajectory_tracking` — position channels vs setpoint vs time.

Calling code is responsible for calling ``plt.show`` after both have been
created — these functions only build figures, they do not block.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_thrust(
    u_history: np.ndarray,
    step_size: float,
    sample_rate: int,
    total_sim_time: float,
) -> None:
    """Plot per-engine TVC commands (α, β in degrees; T in kN) vs time."""
    times = np.arange(len(u_history)) * step_size * sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, ax in enumerate(axes):
        alpha = np.degrees(u_history[:, i * 3])
        beta  = np.degrees(u_history[:, i * 3 + 1])
        T     = u_history[:, i * 3 + 2] / 1e3

        print(
            f"Engine {i + 1}: α [{alpha.min():+.3f}, {alpha.max():+.3f}] deg  "
            f"β [{beta.min():+.3f}, {beta.max():+.3f}] deg  "
            f"T [{T.min():.1f}, {T.max():.1f}] kN"
        )

        ax2 = ax.twinx()
        ax.plot(times, beta,  label="β yaw (deg)",   color='#3498db', linewidth=2)
        ax.plot(times, alpha, label="α pitch (deg)", color='#e74c3c', linewidth=2.5, linestyle='--')
        ax2.plot(times, T,    label="T thrust (kN)", color='#2ecc71', linewidth=2,   linestyle=':')

        ax.set_ylabel("Gimbal angle (deg)")
        ax2.set_ylabel("Thrust (kN)")
        ax.set_title(f"Engine {i + 1} TVC Command")
        ax.grid(True, linestyle=":", alpha=0.6)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("Engine TVC Commands: Gimbal Angles & Thrust")
    axes[2].set_xlabel("Time (s)")
    plt.xlim(0, total_sim_time)
    plt.tight_layout()


def plot_trajectory_tracking(
    trajectory: np.ndarray,
    step_size: float,
    sample_rate: int,
    total_sim_time: float,
    setpoint: np.ndarray,
) -> None:
    """Plot each position channel against its setpoint vs time."""
    times  = np.arange(len(trajectory)) * step_size * sample_rate
    labels = ['X (Vertical)', 'Y (Lateral)', 'Z (Lateral)']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(times, trajectory[:, i], label=f"Current {labels[i]}", color=colors[i], linewidth=2)
        ax.axhline(y=setpoint[i], color="#e74c3c", linestyle="--",
                   label=f"Target ({setpoint[i]}m)", alpha=0.8)
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

    axes[0].set_title(f"Trajectory Tracking: Position vs Time (Total: {total_sim_time}s)", pad=15)
    axes[2].set_xlabel("Time (s)")
    plt.xlim(0, total_sim_time)
    plt.tight_layout()
