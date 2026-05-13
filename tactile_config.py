"""
Tunable constants for the tactile overlay + recording pipeline.
Edit values here to retune sensor placement, arrow style, port assignments, etc.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Hardware: two ESP32 boards, one per finger, each streaming a 3x3 magnetometer
# grid (9 cells, idx 0..8) per the receive_a31301_stream_no_ros.py protocol.
# These paths are placeholders — replace with the actual /dev/tty* paths after
# plugging the boards in. CLI flags on collect_with_home.py override these.
# ---------------------------------------------------------------------------
LEFT_FINGER_PORT  = "/dev/ttyACM0"
RIGHT_FINGER_PORT = "/dev/ttyACM1"
TACTILE_BAUD      = 115200

# Warn if a sensor's last-completed frame is older than this at recording-tick
# time (proxy for "is the ESP32 still streaming?"). One-shot warnings per finger.
TACTILE_LAG_WARNING_MS = 200

# Gripper-safety defaults (forwarded to TactileConfig). Units match what the
# A31301 boards stream (raw counts unless reconfigured).
#
# Metric choice — verified empirically 2026-05-13 with scripts/tune_safety.py:
#   max_abs_z does NOT work for this sensor mounting. The cells with the
#   highest static |Bz| (e.g. ACM0 idx 6 ≈ 2241 at rest) are not the same
#   cells whose magnets move under contact, so max_abs_z stays pinned to
#   the resting cell and never registers a squeeze.
#
#   sum_abs_z works well. Measured progression while closing on a rigid
#   object:
#     grasp=0.20  sum_abs_z ≈ 30575   (no contact yet)
#     grasp=0.25  sum_abs_z ≈ 31103   (+528 — first contact)
#     grasp=0.95  sum_abs_z ≈ 31527   (firm hold)
#   Idle noise band is ~53 counts; contact step is ~500 counts.
#
# Threshold below sits ~120 counts above the idle worst case (30617), well
# clear of the noise band but well below the +500 first-contact step.
# Trips on the lightest touch. Use scripts/tune_safety.py to widen if you
# want the gripper to apply more force before locking.
SAFETY_METRIC = "sum_abs_z"   # "sum_abs_z" works for this hardware; max_abs_z does NOT
# Delta-from-idle units (per-cell baseline subtracted at script startup).
# At idle the sum is ~0-100 (noise); firm grip observed at ~1500-2000.
# 1500 = "grip pretty hard before locking". Tune via scripts/tune_safety.py.
SAFETY_THRESHOLD = 1500.0     # delta-from-idle; bigger = harder grip allowed
SAFETY_STALE_AFTER_SEC = 0.2  # readings older than this are treated as unsafe

# ---------------------------------------------------------------------------
# PCB cell geometry on each finger (3x3 grid).
# Cells are in the FINGER inner-face plane: spaced in y (along finger width)
# and z (along finger length). All values in millimeters.
#
# Bumped from 5 mm to 12 mm so the 3x3 grid (~24 mm span) projects to more
# than a 1-px cluster after the 640->224 resize. Tune to the actual physical
# PCB by measuring or by watching the agent-camera overlay.
# ---------------------------------------------------------------------------
CELL_SPACING_MM = 20.0     # 3x3 spans 40 mm — projects to ~15 px in the agent cam (224 res)
CELL_GRID_SHAPE = (3, 3)   # (rows along finger length, cols across finger width)

# Map ESP32 idx (0..8) -> (row, col) on the 3x3 grid.
# Placeholder: row-major. You'll likely permute this once you observe which
# physical cell rises when you press it.
IDX_TO_ROWCOL = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
]

# ---------------------------------------------------------------------------
# Finger placement in the END-EFFECTOR frame (mm). Convention used here:
#   +x = gripper-closing direction toward the LEFT  finger inner face.
#   -x = gripper-closing direction toward the RIGHT finger inner face.
#    y = along finger width.
#    z = along finger length (more negative = farther from gripper tip).
# These are placeholders — tune by watching the agent-camera overlay.
# ---------------------------------------------------------------------------
FINGER_INNER_X_MM = 35.0    # |closing-axis| half-gap (bigger = grids further apart in agent image; allows bigger arrows)
FINGER_Z_TIP_MM   = 10.0    # z of the PCB row CLOSEST to the gripper tip (more positive = grids lower in image)

# Which EE-frame axis is perpendicular to the inner face of the finger (i.e. the
# gripper-closing direction). For most xArm gripper mountings this is "y" (the
# jaws open along EE +/- y). If your overlay appears rotated 90 deg from the
# actual gripper in the agent camera, flip this between "y" and "x".
FINGER_CLOSING_AXIS_EE = "y"

# ---------------------------------------------------------------------------
# Force interpretation
# ---------------------------------------------------------------------------
# User says: "when the gripper grabs something, the magnets get pushed INTO
# the finger". So per cell, the squeeze signal is the B-component along the
# axis perpendicular to the finger's inner face. Per the convention above,
# that axis is x, signed +1 for left and -1 for right so both fingers
# produce "positive when squeezed".
FORCE_AXIS = "x"        # which of "x" / "y" / "z" maps to gripping pressure
LEFT_FORCE_SIGN  = -1   # +1 or -1, flip if left-finger readings go negative when squeezed
RIGHT_FORCE_SIGN = +1   # +1 or -1, flip if right-finger readings go negative when squeezed

# ---------------------------------------------------------------------------
# Wrist camera anchors (pixels in the NATIVE 640x480 wrist frame).
# Overlay is drawn at native resolution then the image is resized to 224x224
# for storage, so these coordinates are in the pre-resize space.
# Tune by inspecting an actual wrist-camera snapshot.
#
# WRIST_GRID_TRANSPOSED: set to True if the fingers appear ROTATED in the
# wrist view (e.g. extending horizontally instead of vertically). With it
# True, the 3x3 grid's row axis runs along u (image x) and the col axis
# runs along v (image y) — i.e. the grid is rotated 90 deg from the
# default. With it False, rows run along v (down) and cols along u (right).
# Defaults to True because in the current wrist mounting the fingers are
# horizontal in the image. Flip if your mount changes.
# ---------------------------------------------------------------------------
WRIST_GRID_TRANSPOSED   = True
# Anchors placed where the fingers actually appear in the wrist image —
# lower-middle (the gripper occupies roughly y=300..470 in native res).
# Each anchor is the (u, v) of the cell at (row=0, col=0) of its finger.
WRIST_LEFT_TOP_LEFT_UV  = (231, 349)   # left-finger anchor cell  (tuned 2026-05-13)
WRIST_RIGHT_TOP_LEFT_UV = (470, 347)   # right-finger anchor cell (tuned 2026-05-13)
WRIST_CELL_PIX          = 35           # pixel spacing between adjacent cells

# ---------------------------------------------------------------------------
# Arrow rendering
# ---------------------------------------------------------------------------
# With baseline-subtraction active (see tactile_overlay.set_baseline) the
# signal range is DELTA from idle. ARROW_SCALE controls visual sensitivity
# only — it does NOT change the safety threshold or the recorded values.
# Bumped so light contact produces visibly large arrows.
ARROW_SCALE_PX_PER_UNIT = 0.6     # pixels per raw-count of delta-from-idle (per-cell, grid mode)
ARROW_MAX_LENGTH_PX     = 130     # arrow cap at native res (~45 px after 224x224 resize)
ARROW_MIN_LENGTH_PX     = 4       # below this, draw a small dot instead of a line

# For "arrow" and "point" modes the aggregation reduces the 9 cells to one
# scalar per finger (see AGGREGATE_METHOD below). With AGGREGATE_METHOD="max"
# the aggregate is the same kind of "per-cell delta" magnitude grid uses,
# so we keep matching scale + cap. With "sum" drop the scale by ~10x.
AGGREGATE_SCALE_PX_PER_UNIT = 0.6     # per-finger aggregate -> pixels
AGGREGATE_MAX_LENGTH_PX     = 130     # cap at native res (~45 px after 224x224 resize)

# Cap that clamps an arrow's length to FRAC * (pixel distance to the
# target it points at). 0.5 = arrow head reaches at most the midpoint
# between the two fingers, so opposing arrows never cross.
#
# In the WRIST view this rarely binds (inter-finger distance ~240 px, so
# 0.5 * 240 = 120 px which is already above AGGREGATE_MAX_LENGTH_PX). In
# the AGENT view it DOES bind: projected inter-finger distance is ~14 px
# at typical camera placements, so 0.5 * 14 = 7 px max arrow. Tiny, but
# pointing in an unambiguous direction. If you want bigger agent markers,
# use --viz-mode point (circles, no direction) instead of arrow.
AGGREGATE_ARROW_FRAC_CAP = 0.6

# How the 9 per-cell deltas collapse to ONE per-finger scalar for arrow/point/bar.
#   "max"  : peak |delta| across the 9 cells.  Robust to noise (default).
#   "sum"  : sum of |delta|.  Amplifies multi-cell contact; also amplifies noise.
#   "mean" : average |delta|.  Diluted by quiet cells; rarely useful.
# Affects: arrow/point lengths/radii AND the bar's trip comparison.
AGGREGATE_METHOD = "max"

# Hide-when-no-meaningful-reading thresholds. Any per-cell or per-finger
# magnitude below the relevant threshold won't draw anything in the
# corresponding mode. Raised from 30 -> 60 to keep noise from triggering
# a tiny "ghost" arrow at rest. Set to 0 to draw everything.
GRID_MIN_MAGNITUDE_VISIBLE       = 60.0   # per-cell threshold for grid mode
AGGREGATE_MIN_MAGNITUDE_VISIBLE  = 60.0   # per-finger aggregate threshold for arrow + point modes
# Per-side arrow colors. LEFT = ACM0, RIGHT = ACM1. BGR tuples.
LEFT_ARROW_COLOR_BGR    = (0, 255, 0)     # green for LEFT (ACM0)
RIGHT_ARROW_COLOR_BGR   = (0, 0, 255)     # red   for RIGHT (ACM1)
ARROW_COLOR_BGR         = (0, 0, 255)     # legacy single-color fallback
ARROW_THICKNESS         = 5
ARROW_TIP_LENGTH        = 0.35    # fraction of the line, for cv2.arrowedLine
# Alpha for blending the drawn overlay onto the camera image. 1.0 = fully
# opaque (no see-through); 0.0 = invisible. ~0.75 = slightly transparent.
ARROW_ALPHA             = 0.75

DISCONNECTED_COLOR_BGR  = (128, 128, 128) # gray dot for cells reporting connected=0
DISCONNECTED_RADIUS_PX  = 2

# ---------------------------------------------------------------------------
# Visualization mode (controls live --viz windows + the burned-in overlay
# saved into /data/img_*). Raw camera frames are always also saved to
# /data/img_*_raw so the mode can be changed post hoc when re-rendering.
#
#   "arrow"  one inward-pointing arrow per finger, length = per-finger
#            aggregate squeeze magnitude. (Default — least noisy.)
#   "grid"   9 arrows per finger (one per cell), each at its projected
#            position with length proportional to that cell's squeeze.
#            Use when you want to see WHICH cells are being pressed.
#   "point"  one solid circle per finger at the grid center, radius =
#            per-finger aggregate magnitude.
#   "bar"    two horizontal bars at the bottom of the image; each bar
#            lights up GREEN when the per-finger aggregate magnitude
#            exceeds BAR_TRIP_THRESHOLD, gray otherwise. Most compact.
# ---------------------------------------------------------------------------
VISUALIZATION_MODE = "arrow"

# Per-finger AGGREGATE (in whatever units AGGREGATE_METHOD produces) above
# which the "bar" mode shows its bar green. With AGGREGATE_METHOD="max" the
# aggregate is per-cell-scale; 80 lights up only on real contact, ignores noise.
# A bar that's NOT above threshold is NOT drawn at all (no gray placeholder).
BAR_TRIP_THRESHOLD = 80.0

# ---------------------------------------------------------------------------
# Camera serial -> role mapping. transforms.npy keys by serial; we need to
# know which one is the agent (third-person) and which is the wrist.
# ---------------------------------------------------------------------------
AGENT_CAMERA_SERIAL = "327122079374"    # has trc/tcr in transforms.npz
WRIST_CAMERA_SERIAL = "332322072612"    # no transform; wrist overlay uses pixel anchors

# Path (relative to phone_data_collection root) of the converted .npz that
# replaces the numpy-2.x-only transforms.npy in /home/u-ril/edward/robot_calib/.
TRANSFORMS_NPZ_PATH = "transforms_agent.npz"
