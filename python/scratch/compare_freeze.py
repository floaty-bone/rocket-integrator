"""Throwaway: re-linearization ON vs frozen K, with the same Bryson weights.

Twitch metric: RMS and peak of |u_cart[i+1] - u_cart[i]| across the 9-vector,
sampled at the trajectory recording rate (50 Hz). Higher = jerkier commands.
"""
from __future__ import annotations

import math

import numpy as np

from rocket.scenarios import sil_hover


def att_err_deg(q_state, q_set):
    d = abs(float(np.dot(q_state, q_set)))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def run(freeze: bool, label: str):
    sil_hover.FREEZE_K = freeze
    print(f"\n>>> running ({label}) ...", flush=True)
    traj, _u_gim, ucart, dt, sample, sim_time, setpoint = sil_hover.run_sil_simulation()

    # Attitude error time series (degrees)
    errs = np.array([att_err_deg(traj[i, 3:7], setpoint[3:7]) for i in range(traj.shape[0])])
    frame_dt = dt * sample  # seconds between recorded frames

    # Twitch metric: per-frame Cartesian thrust delta magnitude
    du = np.linalg.norm(np.diff(ucart, axis=0), axis=1)  # (n_frames-1,)
    twitch_rms_kN_per_s = (np.sqrt(np.mean(du ** 2)) / frame_dt) / 1e3
    twitch_peak_kN_per_s = (du.max() / frame_dt) / 1e3

    thresh = 2.0  # deg
    times = np.arange(traj.shape[0]) * frame_dt
    above = np.where(errs > thresh)[0]
    settle = times[above[-1]] if len(above) else 0.0

    print(f"\n=== {label} ===")
    print(f"  final att err     : {errs[-1]:8.3f} deg")
    print(f"  peak  att err     : {errs.max():8.2f} deg")
    print(f"  settle (<{thresh:g} deg)   : {settle:6.2f} s")
    print(f"  twitch RMS        : {twitch_rms_kN_per_s:8.1f} kN/s   (sum over 9 cartesian ch.)")
    print(f"  twitch PEAK       : {twitch_peak_kN_per_s:8.1f} kN/s")
    return errs, du / frame_dt / 1e3


if __name__ == "__main__":
    run(False, "RE-LIN @ 80 Hz")
    run(True,  "FROZEN K")
