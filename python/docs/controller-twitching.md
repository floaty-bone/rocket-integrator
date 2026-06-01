# Controller twitching diagnosis

## Symptom

Booster struggles with roll control (rotation about body X). Coupled
pitch+yaw+roll maneuvers fail. The controller visibly twitches —
discontinuous, "panicky" commands, unsteady high-frequency oscillation.
**Not** a slow / sluggish convergence problem; a *jitter* problem.

## Ruled out

### SIL gimbal boundary parameterization

`cart9_to_gimbal9` (`plant/tvc.py:25`) uses `atan2(fy, fx)`, which wraps at
±π — a textbook discontinuity source. But the SIL loop's roundtrip is
exact: the controller emits `u_cart`, `controller_block` converts to
gimbal, and `plant_block` immediately converts back via `gimbal9_to_cart9`
before computing the wrench (`scenarios/sil_hover.py:88`). The atan2 wrap
is invisible to the plant.

Caveat: this **would** become a real source of chatter if gimbal-rate
limits or thrust saturation were added downstream. Currently neither is
modeled.

## Live suspects

### 1. Re-linearization jumps at 80 Hz  (primary)

LQR re-solves at `LINEARIZATION_RATE = 80 Hz` (`scenarios/sil_hover.py:47`)
→ new `K` every 12.5 ms. Controller fires `K @ δx` at 9 kHz → 112 cycles
use the same `K`, then `K` steps to a new value.

If the state moves fast between re-linearizations (coupled rotation,
especially roll with its low `Ixx = 1.2e6` → high angular accel), then
`A(x_op)` and `B(x_op)` change meaningfully between updates. The new `K`
applied to a similar state gives a different `u_cart` → step discontinuity
in commanded thrust at 80 Hz. To a human watching, that is an 80 Hz
"panicky twitch."

It hits roll hardest because roll dynamics evolve fastest (lowest inertia)
→ biggest state delta per 12.5 ms window → biggest K-jump.

### 2. Gyroscopic coupling not captured by point linearization

`ω × (Iω)` becomes large during coupled-axis spins, especially with the
asymmetric inertia (1.2e6 vs 3.5e7 × 3.5e7). The Jacobian captures this
locally at the operating point, but a 12.5 ms-old `K` does not track it.
Same fix surface as suspect 1.

### 3. Quaternion sign flip in the design path

`attitude_law.qtorp` explicitly handles `q_w < 0`
(`math/tangent_space.py:82`). But `make_lqr_computer` re-projects with
`E(xl[3:7], scale=0.5)` using the *raw* operating-point quaternion — **no**
sign normalization (`control/lqr_design.py:43`). If `xl[3:7]` lands in the
`q_w < 0` hemisphere between updates, the tangent-space basis flips sign →
`K` flips sign → instant discontinuity.

## Diagnostics (in order of effort)

1. **Freeze K.** Set `FREEZE_K = True` in `scenarios/sil_hover.py`. Skips
   `update_linearization` inside `controller_block`. If twitching
   disappears, re-linearization is the culprit. *Fastest test.*
2. **Raise `LINEARIZATION_RATE`** to 500–1000 Hz. Smaller K-jumps per
   update; twitching should soften proportionally.
3. **Log `||K - K_prev||`** per re-lin and look for spikes during coupled
   maneuvers. Direct evidence of where and when `K` jumps.

## Test result (2026-05-30)

40 s roll-recovery maneuver (160° about body X), Bryson-normalized Q/R,
re-linearization ON vs FROZEN. Twitch metric: RMS of
`|u_cart[i+1] - u_cart[i]|` per 50 Hz recording frame.

| | RE-LIN @ 80 Hz | FROZEN K |
|---|---:|---:|
| final attitude error | 0.001° | 0.006° |
| settling (<2°)       | 15.24 s | 17.70 s |
| twitch RMS           | 21,677 kN/s | **527 kN/s** |
| twitch PEAK          | 416,788 kN/s | **23,371 kN/s** |

Re-linearization accounts for ~40× the twitch energy. Frozen K converges
essentially as well (~2 s slower).

## Fixes if frozen K is not acceptable long-term

- **Sign-normalize the operating point** before `compute_lqr`:
  `if xl[3] < 0: xl = xl.at[3:7].set(-xl[3:7])`.
- **Low-pass filter K** between updates:
  `K_new = α·K_solved + (1 − α)·K_prev` with `α ≈ 0.02–0.05`.
- **Raise `LINEARIZATION_RATE`** so each individual K-jump is smaller.

For setpoint-tracking and hover, frozen K is usually the textbook answer
— re-linearization only pays off when the operating point traverses far
enough that one linearization isn't valid (landing, ascent burns).
