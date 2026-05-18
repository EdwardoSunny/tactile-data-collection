"""
Configuration for the tactile hardware + gripper-safety wrapper.

Overlay drawing now lives entirely in environment/sensordrawing/ — its camera
intrinsics, robot->camera extrinsics, finger kinematics, per-board calibration
(offset + scale), arrow style and alpha are all bundled there. This file is
just the runtime config the rest of the data-collection harness needs:
serial-port assignments, baud, safety thresholds, and the camera serial ->
role mapping consumed by environment/threads.RecordingThread.
"""

# ---------------------------------------------------------------------------
# Hardware: two ESP32 boards, one per finger, each streaming a 3x3 magnetometer
# grid (9 cells, idx 0..8) per the receive_a31301_stream_no_ros.py protocol.
# Replace with the actual /dev/tty* paths after plugging the boards in.
# CLI flags on collect_with_home.py override these.
# ---------------------------------------------------------------------------
LEFT_FINGER_PORT  = "/dev/ttyACM0"
RIGHT_FINGER_PORT = "/dev/ttyACM1"
TACTILE_BAUD      = 115200

# Warn if a sensor's last-completed frame is older than this at recording-tick
# time (proxy for "is the ESP32 still streaming?"). One-shot warnings per finger.
TACTILE_LAG_WARNING_MS = 200

# ---------------------------------------------------------------------------
# Gripper-safety defaults (forwarded to TactileConfig). Units match what the
# A31301 boards stream (raw counts unless reconfigured).
#
# Metric choice — verified empirically with scripts/tune_safety.py:
#   max_abs_z does NOT work for this sensor mounting. The cells with the
#   highest static |Bz| are not the same cells whose magnets move under
#   contact, so max_abs_z stays pinned to the resting cell and never
#   registers a squeeze.
#
#   sum_abs_z works well. Measured progression while closing on a rigid
#   object: idle ~30575; first contact ~31103 (+528 counts); firm hold
#   ~31527. Idle noise band ~53 counts; contact step ~500 counts.
#
# A baseline is captured at startup and installed on TactileConfig.baseline so
# the safety metric runs in "delta from idle" units (much smaller than raw).
# ---------------------------------------------------------------------------
SAFETY_METRIC          = "sum_abs_z"   # works for this hardware; max_abs_z does NOT
SAFETY_THRESHOLD       = 1500.0        # delta-from-idle counts; tune via tune_safety.py
SAFETY_STALE_AFTER_SEC = 0.2           # readings older than this are treated as unsafe

# Legacy single-axis reference (only consulted by the startup baseline-print
# in collect_with_home.py to label which axis it's showing).
FORCE_AXIS = "x"

# ---------------------------------------------------------------------------
# Camera serial -> role mapping. sensordrawing's per-camera K + T_rc are keyed
# on these serials; both threads.RecordingThread and the calibration loader
# in environment/sensordrawing/transforms/transforms.npy use them.
# ---------------------------------------------------------------------------
AGENT_CAMERA_SERIAL = "327122079374"    # side / third-person — has trc in transforms.npy
WRIST_CAMERA_SERIAL = "332322072612"    # wrist — T_rc computed from joint angles via T_link7_to_cam
