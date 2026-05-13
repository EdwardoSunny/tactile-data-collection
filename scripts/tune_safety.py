"""
Tactile gripper-safety threshold tuner.

Run this interactively to find a value for tactile_config.SAFETY_THRESHOLD.

You drive the gripper open/close from the terminal; the script prints the
live safety metrics (max_abs_z, max_norm, sum_abs_z) at every step so you
can pick a threshold that "trips" at the contact force you want as your
maximum squeeze.

CRITICAL DESIGN NOTE: by default this script disables the xArm safety
wrapper (Simple_Task is constructed with tactile=None for the controller),
so the gripper WILL close past any threshold you eventually want to set.
This is on purpose — it's the only way to observe the full metric range
while squeezing a real object. The tactile boards are still opened, just
for read-only display.

To verify the safety wrapper itself, pass --with-safety. In that mode the
controller is wired to the tactile sensors exactly as in collect_with_home.py;
the gripper will refuse to close further once the metric exceeds the
configured SAFETY_THRESHOLD. The script then prints both the REQUESTED
grasp and the ACTUAL grasp that the controller committed, so you can see
the clamp engage in real time.

Workflow:
  1. Place an object between the gripper fingers.
  2. Run this script: python scripts/tune_safety.py
  3. Press 'c' (or space) to close the gripper in small steps.
  4. Note the max_abs_z value at the point where the grip feels right.
  5. Edit tactile_config.py: SAFETY_THRESHOLD = <that value>
  6. Re-run collect_with_home.py; the gripper will now stop closing at
     that contact level.

Controls (single keypress, no Enter needed; works over SSH, no X):
  c / space  close gripper by 0.05
  o          open  gripper by 0.05
  C          close gripper by 0.20
  O          open  gripper by 0.20
  z          fully open  (grasp = 0.0)
  Z          fully close (grasp = 1.0)
  p          re-print readings without moving
  q          quit (also Ctrl+C)
"""
from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

# Make the parent package importable when run as `python scripts/tune_safety.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tactile_config as tc  # noqa: E402
from environment.tactile import (  # noqa: E402
    TactileConfig,
    TactileSensors,
    compute_safety_metric,
)
from tasks.simple_task import Simple_Task  # noqa: E402


# ---------------------------------------------------------------------------
# Single-key stdin reader (cbreak mode). Works over SSH; no X required.
# ---------------------------------------------------------------------------

def read_one_key(timeout=None):
    """Block (or wait up to ``timeout`` seconds) for one key from stdin.

    Returns the character or None if the timeout elapsed without input.
    Uses cbreak mode so single keypresses come through without waiting
    for Enter. Restores the terminal settings on exit even if the caller
    aborts.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if timeout is not None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _format_per_finger(states, stale_after_sec):
    """One-line per-finger summary: max|Bz| over connected cells."""
    now = time.time()
    out = []
    for label, st in zip(("L", "R"), states):
        host_ts = float(st.get("host_timestamp", 0.0))
        conn = np.asarray(st.get("connected", np.zeros(0)), dtype=np.int32)
        xyz = np.asarray(st.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        if conn.size == 0 or not np.any(conn) or host_ts <= 0 or (now - host_ts) > stale_after_sec:
            out.append(f"{label}=STALE/DISC")
            continue
        mask = conn > 0
        max_z = float(np.max(np.abs(xyz[mask, 2])))
        out.append(f"{label} max|Bz|={max_z:6.0f}")
    return "  ".join(out)


def report(tactile, cfg, requested_grasp, actual_grasp=None, safety_active=False):
    states = tactile.get_latest()
    # Pull baseline directly from config so the reported metrics match what
    # the live safety wrapper sees (delta-from-idle when baseline is set).
    baseline = cfg.baseline
    m_z, _ = compute_safety_metric(states, "max_abs_z", cfg.stale_after_sec, baseline=baseline)
    m_n, _ = compute_safety_metric(states, "max_norm",  cfg.stale_after_sec, baseline=baseline)
    m_s, _ = compute_safety_metric(states, "sum_abs_z", cfg.stale_after_sec, baseline=baseline)
    per = _format_per_finger(states, cfg.stale_after_sec)
    if actual_grasp is not None and abs(actual_grasp - requested_grasp) > 1e-6:
        # Safety wrapper clamped the command — make this visible.
        grasp_str = f"req={requested_grasp:.2f} act={actual_grasp:.2f} **CLAMPED**"
    else:
        grasp_str = f"grasp={requested_grasp:.2f}"
        if safety_active:
            grasp_str += " (safety active, but command not closing)"
    print(
        f"  {grasp_str:38s}"
        f"max_abs_z={m_z:7.1f}  max_norm={m_n:7.1f}  sum_abs_z={m_s:7.1f}   "
        f"({per})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-safety", action="store_true",
                    help="Construct the xArm WITH the tactile safety wrapper (like "
                         "collect_with_home.py). Use this to verify the wrapper actually "
                         "clamps the gripper at the configured SAFETY_THRESHOLD. Without "
                         "this flag (default), safety is OFF so you can drive past the "
                         "threshold to tune the value.")
    args = ap.parse_args()

    print("Opening tactile sensors...")
    cfg = TactileConfig(
        ports=[tc.LEFT_FINGER_PORT, tc.RIGHT_FINGER_PORT],
        baud=tc.TACTILE_BAUD,
        safety_metric=tc.SAFETY_METRIC,
        safety_threshold=tc.SAFETY_THRESHOLD,
        stale_after_sec=tc.SAFETY_STALE_AFTER_SEC,
        lag_warning_ms=tc.TACTILE_LAG_WARNING_MS,
    )
    tactile = TactileSensors(cfg, names=["L", "R"])
    tactile.__enter__()
    if not tactile.any_open:
        print("ERROR: neither tactile port opened. Check /dev/ttyACM* and try again.")
        tactile.__exit__(None, None, None)
        return 1
    if not tactile.all_open:
        print("WARN: only one tactile port opened; per-finger data will be partial.")

    if args.with_safety:
        print("Opening robot (xArm) WITH tactile safety wrapper enabled.")
        print("  -> gripper will refuse to close past SAFETY_THRESHOLD.")
        env = Simple_Task(tactile=tactile)
    else:
        print("Opening robot (xArm)... safety wrapper INTENTIONALLY DISABLED for tuning.")
        print("  -> use --with-safety to verify the wrapper actually clamps the gripper.")
        env = Simple_Task(tactile=None)
    env.reset(duration=3.0)
    time.sleep(3.0)

    # Capture baseline now (gripper open at home, no contact) so the printed
    # metrics + the live safety wrapper both run in delta-from-idle mode.
    print("Sampling tactile baseline (1.5 s, keep fingers untouched)...")
    samples = []
    t_end = time.time() + 1.5
    while time.time() < t_end:
        sts = tactile.get_latest()
        if all(s.get("host_timestamp", 0.0) > 0 for s in sts[:2]):
            samples.append(np.stack([sts[0]["xyz"], sts[1]["xyz"]]).astype(np.float32))
        time.sleep(0.05)
    if samples:
        cfg.baseline = np.mean(np.stack(samples), axis=0).astype(np.float32)
        print(f"  baseline captured from {len(samples)} samples; metrics now show DELTA from idle.")
    else:
        print("  [warn] no baseline samples; metrics will be raw values.")

    # Settle pose readback.
    for _ in range(10):
        env.get_obs()
        time.sleep(0.1)

    grasp = 0.0
    # Initial commanded pose — hold current.
    pose = list(env.get_obs()["pose"])

    def step_to(new_grasp):
        nonlocal grasp, pose
        grasp = float(np.clip(new_grasp, 0.0, 1.0))
        env.step(pose, grasp)
        time.sleep(0.3)  # let gripper move + tactile boards refresh

    def current_actual_grasp():
        """The grasp the controller actually committed (after safety clamping)."""
        return float(env.env.xarm.previous_grasp)

    def current_safety_active():
        return bool(getattr(env.env.xarm, "_safety_active", False))

    print()
    print("=" * 76)
    mode = "SAFETY WRAPPER ON" if args.with_safety else "safety wrapper OFF"
    print(f"  Tactile safety threshold tuner  [{mode}]")
    print("=" * 76)
    print("  c / space = close 0.05    o = open 0.05")
    print("  C         = close 0.20    O = open 0.20")
    print("  z         = fully open    Z = fully close")
    print("  p         = print without moving")
    print("  q         = quit")
    print(f"  current SAFETY_METRIC    in tactile_config.py = {tc.SAFETY_METRIC}")
    print(f"  current SAFETY_THRESHOLD in tactile_config.py = {tc.SAFETY_THRESHOLD}")
    if args.with_safety:
        print("  When the wrapper clamps the gripper, you'll see:")
        print("    req=0.55 act=0.30 **CLAMPED**  ...")
    print("=" * 76)
    print()
    report(tactile, cfg, grasp, current_actual_grasp(), current_safety_active())

    try:
        while True:
            k = read_one_key()
            if k is None:
                continue
            if k in ("c", " "):
                step_to(grasp + 0.05)
            elif k == "o":
                step_to(grasp - 0.05)
            elif k == "C":
                step_to(grasp + 0.20)
            elif k == "O":
                step_to(grasp - 0.20)
            elif k == "z":
                step_to(0.0)
            elif k == "Z":
                step_to(1.0)
            elif k == "p":
                pass  # fall through and reprint
            elif k in ("q", "\x03"):   # 'q' or Ctrl+C in cbreak
                break
            else:
                continue
            report(tactile, cfg, grasp, current_actual_grasp(), current_safety_active())
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print("Re-opening gripper...")
        try:
            step_to(0.0)
        except Exception:
            pass
        tactile.__exit__(None, None, None)

    print()
    print("When you've picked a value:")
    print("  1. Edit tactile_config.py and set:")
    print(f"       SAFETY_THRESHOLD = <chosen value>   # currently {tc.SAFETY_THRESHOLD}")
    print("  2. Re-run collect_with_home.py — the new threshold will be in effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
