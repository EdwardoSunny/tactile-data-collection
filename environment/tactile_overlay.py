"""
Draw the 2-finger x 9-cell tactile overlay on the agent and wrist camera images.

The overlay is "shear-aware":
  - per-cell SHEAR (Bx, By in the sensor's local frame) drives the arrow
    DIRECTION in the image plane
  - per-cell NORMAL |Bz - baseline_z| drives the arrow LENGTH (also circle
    radius in `point` mode and bar trip in `bar` mode)
both after baseline subtraction (idle field captured at startup, see
collect_with_home.py / set_baseline below).

Agent (third-person) camera: cell positions are 3D (EE frame, mm), projected
through the current robot pose and the camera's extrinsics+intrinsics to
pixels. Per-cell shear is converted to a 3D vector in EE frame (sensor.x ->
finger-width axis, sensor.y -> finger-length axis) and projected by computing
the pixel offset of (cell + small_step * unit_shear) - cell, so the arrows
rotate correctly with the EE.

Wrist camera: 18 fixed pixel anchors (9 per finger) from tactile_config; no
projection. Per-cell shear is mapped to image-plane direction through a per-
finger 2x2 matrix (WRIST_SHEAR_*_UV_FROM_SENSOR) so signs/swaps can be tuned
to whatever orientation the wrist camera + sensor PCB happen to be in.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import tactile_config as tc


# -----------------------------------------------------------------------------
# Per-cell baseline used to "normalize" the sensor field away from the static
# magnetic field each cell sees at rest. Without this, every reading is
# dominated by the magnet sitting near each magnetometer at idle, so deltas
# (the actual contact signal) are buried under a large constant offset.
#
# Set by the recording script after a short idle capture (gripper open, no
# contact). Shape (2, 9, 3) — finger x cell x (Bx,By,Bz). None disables
# baseline subtraction (raw values used).
# -----------------------------------------------------------------------------
_BASELINE = None  # type: ignore


def set_baseline(baseline_2x9x3):
    """Install per-cell idle baseline used by shear_and_normal()."""
    global _BASELINE
    arr = np.asarray(baseline_2x9x3, dtype=np.float32)
    if arr.shape != (2, 9, 3):
        raise ValueError(f"baseline must be (2,9,3); got {arr.shape}")
    _BASELINE = arr.copy()


def clear_baseline():
    global _BASELINE
    _BASELINE = None


def get_baseline():
    """Returns the current baseline (a copy) or None if not set."""
    return None if _BASELINE is None else _BASELINE.copy()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cell_positions_ee_frame(side: str) -> np.ndarray:
    """Return (9, 3) positions of the 3x3 cells on one finger's inner face, in
    end-effector frame (mm). Cells indexed by raw idx via tc.IDX_TO_ROWCOL.

    The gripper-closing axis is given by tc.FINGER_CLOSING_AXIS_EE ("x" or "y").
    The width axis is whichever of (x,y) is NOT the closing axis; the length
    axis is always z. Flip FINGER_CLOSING_AXIS_EE if the projected overlay
    appears rotated 90 deg from the actual gripper.
    """
    rows, cols = tc.CELL_GRID_SHAPE
    if side == "left":
        sgn = +1.0
    elif side == "right":
        sgn = -1.0
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    closing_axis = getattr(tc, "FINGER_CLOSING_AXIS_EE", "y").lower()
    total_width = (cols - 1) * tc.CELL_SPACING_MM
    positions = np.zeros((9, 3), dtype=np.float32)

    for idx, (row, col) in enumerate(tc.IDX_TO_ROWCOL):
        w = -total_width / 2.0 + col * tc.CELL_SPACING_MM    # along finger width
        h = tc.FINGER_Z_TIP_MM - row * tc.CELL_SPACING_MM    # along finger length (-z)
        if closing_axis == "x":
            # +x = left finger inner face, -x = right; width axis is y.
            positions[idx] = (sgn * tc.FINGER_INNER_X_MM, w, h)
        elif closing_axis == "y":
            # +y = left finger inner face, -y = right; width axis is x.
            positions[idx] = (w, sgn * tc.FINGER_INNER_X_MM, h)
        else:
            raise ValueError(
                f"FINGER_CLOSING_AXIS_EE must be 'x' or 'y', got {closing_axis!r}"
            )
    return positions


def ee_pose_to_matrix(pose_6d) -> np.ndarray:
    """xArm 6D pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] -> 4x4 EE -> robot."""
    pos = np.asarray(pose_6d[:3], dtype=np.float64)
    euler = np.asarray(pose_6d[3:6], dtype=np.float64)
    rot = R.from_euler("xyz", euler, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def project_points_to_image(points_ee_mm, ee_pose, trc, intrinsics):
    """
    Project 3D points (EE frame, mm) to pixel coordinates in the agent camera.

    Args:
        points_ee_mm : (N, 3) mm, EE frame
        ee_pose      : (6,) [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        trc          : (3, 4) Robot -> Camera (translation in meters)
        intrinsics   : librealsense rs.intrinsics OR any object with
                       .fx, .fy, .ppx, .ppy attributes

    Returns:
        pixels : (N, 2) int32 pixel coordinates
        valid  : (N,) bool — True if the point is in front of the camera
    """
    pts = np.asarray(points_ee_mm, dtype=np.float64)
    T_ee_robot = ee_pose_to_matrix(ee_pose)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])           # (N, 4)
    pts_robot_mm = (T_ee_robot @ pts_h.T)[:3].T                # (N, 3) mm
    pts_robot_m = pts_robot_mm / 1000.0                        # mm -> m

    # trc is (3, 4): rotation 3x3 in :, :3, translation in :, 3
    pts_cam = (trc[:, :3] @ pts_robot_m.T).T + trc[:, 3]       # (N, 3) m

    valid = pts_cam[:, 2] > 1e-3
    safe_z = np.where(valid, pts_cam[:, 2], 1.0)
    u = intrinsics.fx * pts_cam[:, 0] / safe_z + intrinsics.ppx
    v = intrinsics.fy * pts_cam[:, 1] / safe_z + intrinsics.ppy
    pixels = np.stack([u, v], axis=1).astype(np.int32)
    return pixels, valid


# ---------------------------------------------------------------------------
# Per-cell signal extraction
# ---------------------------------------------------------------------------

def squeeze_magnitudes(tactile_9x3: np.ndarray, side: str) -> np.ndarray:
    """LEGACY single-axis squeeze: per-cell sign * (B[FORCE_AXIS] - baseline).

    The current shear-aware overlay uses shear_and_normal() instead, which
    keeps direction (Bx, By) and magnitude (|Bz|) separate. This function is
    kept for any external callers that already depend on it.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[tc.FORCE_AXIS]
    sign = tc.LEFT_FORCE_SIGN if side == "left" else tc.RIGHT_FORCE_SIGN
    val = tactile_9x3[:, axis_idx].astype(np.float32)
    if _BASELINE is not None:
        finger_idx = 0 if side == "left" else 1
        val = val - _BASELINE[finger_idx, :, axis_idx]
    return sign * val


def shear_and_normal(tactile_9x3: np.ndarray, side: str):
    """Per-cell (shear_xy_sensor, normal_signed) after baseline subtraction.

    Returns:
      shear_xy_sensor : (9, 2) float32 — (Bx-baseline_x, By-baseline_y) in
                        the sensor's local PCB frame. Drives ARROW DIRECTION
                        (mapped to image plane separately per view).
      normal_signed   : (9,)  float32 — sign * (Bz-baseline_z), per-finger
                        sign chosen so positive = "squeezed". |normal| drives
                        ARROW LENGTH / circle radius / bar trip.
    """
    side_idx = 0 if side == "left" else 1
    sign_n = tc.LEFT_FORCE_SIGN if side == "left" else tc.RIGHT_FORCE_SIGN
    arr = np.asarray(tactile_9x3, dtype=np.float32)
    if _BASELINE is not None:
        arr = arr - _BASELINE[side_idx]
    shear = arr[:, :2].copy()           # (9, 2) sensor-frame shear delta
    normal = sign_n * arr[:, 2]         # (9,)

    # Soft deadband on per-cell |shear|. Direction is normalize(shear), which
    # is discontinuous near zero — without this, a few counts of sensor
    # noise produce a full unit-direction arrow that flips frame to frame.
    # Cells below the deadband become exactly zero (so they contribute no
    # direction downstream); cells above lose `deadband` from their
    # magnitude (linear above the threshold, with no jump at the threshold).
    deadband = float(getattr(tc, "SHEAR_DEADBAND", 0.0))
    if deadband > 0:
        norms = np.linalg.norm(shear, axis=1)
        scale = np.maximum(0.0, norms - deadband) / np.where(norms > 1e-3, norms, 1.0)
        shear = shear * scale[:, None]

    return shear, normal


# ---------------------------------------------------------------------------
# Aggregation across the 9 cells of a finger
# ---------------------------------------------------------------------------

def _aggregate_per_finger(tact_L, tact_R):
    """LEGACY single-axis aggregate (|squeeze_magnitudes| reduced over cells)."""
    sq_L = np.abs(squeeze_magnitudes(tact_L, "left"))
    sq_R = np.abs(squeeze_magnitudes(tact_R, "right"))
    method = getattr(tc, "AGGREGATE_METHOD", "max").lower()
    if method == "max":
        return float(np.max(sq_L)), float(np.max(sq_R))
    if method == "sum":
        return float(np.sum(sq_L)), float(np.sum(sq_R))
    if method == "mean":
        return float(np.mean(sq_L)), float(np.mean(sq_R))
    raise ValueError(f"AGGREGATE_METHOD must be 'max' | 'sum' | 'mean', got {method!r}")


def _aggregate_normal_per_finger(tact_L, tact_R):
    """Per-finger aggregate of |normal| using AGGREGATE_METHOD.

    Used by the point / bar modes (which have no direction) and by the
    arrow modes for the per-finger arrow LENGTH.
    """
    _, n_L = shear_and_normal(tact_L, "left")
    _, n_R = shear_and_normal(tact_R, "right")
    method = getattr(tc, "AGGREGATE_METHOD", "max").lower()
    if method == "max":
        return float(np.max(np.abs(n_L))), float(np.max(np.abs(n_R)))
    if method == "sum":
        return float(np.sum(np.abs(n_L))), float(np.sum(np.abs(n_R)))
    if method == "mean":
        return float(np.mean(np.abs(n_L))), float(np.mean(np.abs(n_R)))
    raise ValueError(f"AGGREGATE_METHOD must be 'max' | 'sum' | 'mean', got {method!r}")


def _aggregate_shear_per_finger(image_dirs, normal):
    """Per-finger aggregate (sum_dir_2vec, mag_scalar).

    sum_dir : sum_i image_dirs[i] * |normal[i]|. Uses RAW image_dirs (not
              per-cell unit vectors) so cells with weak shear contribute
              proportionally less to the aggregate direction; combined with
              the SHEAR_DEADBAND in shear_and_normal, this keeps the arrow
              direction stable when only noise-level shear is present.
              Per-cell weighting by |normal| then gives heavily-pressed
              cells more say.
    mag_scalar : aggregate(|normal|) per AGGREGATE_METHOD; sets arrow length.
    """
    abs_n = np.abs(normal).astype(np.float32)
    sum_dir = np.sum(image_dirs.astype(np.float32) * abs_n[:, None], axis=0)
    method = getattr(tc, "AGGREGATE_METHOD", "max").lower()
    if method == "max":
        agg = float(np.max(abs_n))
    elif method == "sum":
        agg = float(np.sum(abs_n))
    elif method == "mean":
        agg = float(np.mean(abs_n))
    else:
        raise ValueError(f"AGGREGATE_METHOD must be 'max' | 'sum' | 'mean', got {method!r}")
    return sum_dir, agg


# ---------------------------------------------------------------------------
# Sensor xy -> image-plane direction (per view)
# ---------------------------------------------------------------------------

def _wrist_shear_to_image_dirs(shear_sensor_xy: np.ndarray, side: str) -> np.ndarray:
    """Map (9, 2) sensor-frame shear to (9, 2) wrist-image-plane vectors.

    Per-finger 2x2 matrix from tactile_config:
      WRIST_SHEAR_LEFT_UV_FROM_SENSOR
      WRIST_SHEAR_RIGHT_UV_FROM_SENSOR
    Defaults: identity (left), -x flip (right) so a magnet shearing toward
    the user's left in the world produces a leftward arrow on both fingers.
    Tune to taste by inducing a known shear and flipping signs.
    """
    if side == "left":
        M = np.asarray(getattr(tc, "WRIST_SHEAR_LEFT_UV_FROM_SENSOR",
                               [[1.0, 0.0], [0.0, 1.0]]), dtype=np.float32)
    else:
        M = np.asarray(getattr(tc, "WRIST_SHEAR_RIGHT_UV_FROM_SENSOR",
                               [[-1.0, 0.0], [0.0, 1.0]]), dtype=np.float32)
    return shear_sensor_xy.astype(np.float32) @ M.T   # (9, 2)


def _agent_shear_to_image_dirs(cell_pos_ee, shear_sensor_xy, side,
                               ee_pose, trc, intrinsics):
    """Project per-cell sensor-frame shear into agent-image direction vectors.

    1. Apply per-finger 2x2 (AGENT_SHEAR_*_FROM_SENSOR) to sensor (sx, sy).
    2. Embed the rotated (sx', sy') as a 3D vector in EE frame:
         FINGER_CLOSING_AXIS_EE="y"  ->  sensor.x->EE.x (width), sensor.y->EE.z (length)
         FINGER_CLOSING_AXIS_EE="x"  ->  sensor.x->EE.y (width), sensor.y->EE.z (length)
       The closing axis (perpendicular to the inner face) gets 0 shear by
       construction.
    3. For each cell, project (cell_pos) and (cell_pos + STEP * unit_shear_ee)
       and take the pixel-space difference. That difference is the image-
       plane direction the arrow should point along; rotates correctly with
       the EE.
    Returns (9, 2) image-plane vectors (NOT normalized — magnitude is "pixel
    offset of a STEP-mm shear" and downstream normalizes).
    """
    if side == "left":
        M = np.asarray(getattr(tc, "AGENT_SHEAR_LEFT_FROM_SENSOR",
                               [[1.0, 0.0], [0.0, 1.0]]), dtype=np.float32)
    else:
        M = np.asarray(getattr(tc, "AGENT_SHEAR_RIGHT_FROM_SENSOR",
                               [[-1.0, 0.0], [0.0, 1.0]]), dtype=np.float32)
    sxy = shear_sensor_xy.astype(np.float32) @ M.T   # (9, 2) sensor frame, post-rotate

    closing_axis = getattr(tc, "FINGER_CLOSING_AXIS_EE", "y").lower()
    shear_ee = np.zeros((len(sxy), 3), dtype=np.float32)
    if closing_axis == "y":
        shear_ee[:, 0] = sxy[:, 0]   # width
        shear_ee[:, 2] = sxy[:, 1]   # length
    elif closing_axis == "x":
        shear_ee[:, 1] = sxy[:, 0]
        shear_ee[:, 2] = sxy[:, 1]

    # Project (cell) and (cell + scaled_shear_ee). Scale sensor counts to
    # mm by AGENT_SHEAR_MM_PER_COUNT so the projected pixel offset
    # PRESERVES per-cell shear magnitude — strong-shear cells produce
    # longer image vectors and dominate the per-finger aggregate, weak
    # ones contribute proportionally less. Cells with shear ~ 0 (e.g.
    # killed by SHEAR_DEADBAND) project to ~ 0, falling through to the
    # dot-fallback in the drawing primitives.
    mm_per_count = float(getattr(tc, "AGENT_SHEAR_MM_PER_COUNT", 0.02))
    step_ee = shear_ee * mm_per_count

    pix1, _ = project_points_to_image(cell_pos_ee, ee_pose, trc, intrinsics)
    pix2, _ = project_points_to_image(cell_pos_ee + step_ee, ee_pose, trc, intrinsics)
    return (pix2 - pix1).astype(np.float32)            # (9, 2)


def _select_mode(mode):
    return (mode or getattr(tc, "VISUALIZATION_MODE", "arrow")).lower()


def _blend_overlay(drawn, original):
    """Alpha-blend the drawn overlay onto the unmodified camera image.

    Pixels that weren't touched by drawing are identical in both arrays so
    blending leaves them unchanged; only the arrow/point/bar pixels get the
    see-through effect.
    """
    alpha = float(getattr(tc, "ARROW_ALPHA", 1.0))
    if alpha >= 0.999:
        return drawn
    return cv2.addWeighted(drawn, alpha, original, 1.0 - alpha, 0.0)


# ---------------------------------------------------------------------------
# Drawing primitives (shear-aware)
# ---------------------------------------------------------------------------

def _draw_shear_arrows(img, anchors, image_dirs, normal, connected,
                       valid=None, color=None):
    """Per-cell arrow draw (used by `grid` mode).

    Direction = normalize(image_dirs[i])  (image_dirs are NOT pre-normalized;
                                            their raw magnitude is whatever
                                            the per-view mapper produced).
    Length    = |normal[i]| * tc.ARROW_SCALE_PX_PER_UNIT, capped at
                tc.ARROW_MAX_LENGTH_PX. Cells with |normal|<min_visible draw
                nothing; cells with shear ~ 0 but normal > min_visible draw
                a small dot (force present, no direction).
    Disconnected cells draw a small gray dot regardless.
    """
    h, w = img.shape[:2]
    n = len(anchors)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    if color is None:
        color = tc.ARROW_COLOR_BGR
    min_visible = float(getattr(tc, "GRID_MIN_MAGNITUDE_VISIBLE", 0.0))

    for i in range(n):
        if not valid[i]:
            continue
        u, v = int(anchors[i, 0]), int(anchors[i, 1])
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        if connected[i] == 0:
            cv2.circle(img, (u, v), tc.DISCONNECTED_RADIUS_PX,
                       tc.DISCONNECTED_COLOR_BGR, -1)
            continue
        mag = abs(float(normal[i]))
        if mag < min_visible:
            continue
        length = float(np.clip(mag * tc.ARROW_SCALE_PX_PER_UNIT,
                               0.0, tc.ARROW_MAX_LENGTH_PX))
        dir_vec = image_dirs[i].astype(np.float32)
        dn = float(np.linalg.norm(dir_vec))
        if dn < 1e-3 or length < tc.ARROW_MIN_LENGTH_PX:
            # Either no shear direction, or arrow would be too short. Draw a
            # filled circle whose radius scales modestly with normal so the
            # cell still indicates "force present" without a misleading dir.
            r = max(2, min(8, int(round(length / 2))))
            cv2.circle(img, (u, v), r, color, -1)
            continue
        unit = dir_vec / dn
        end = (int(round(u + unit[0] * length)),
               int(round(v + unit[1] * length)))
        cv2.arrowedLine(img, (u, v), end, color,
                        tc.ARROW_THICKNESS, tipLength=tc.ARROW_TIP_LENGTH)


def _shear_arrow_at(img, center, dir_vec, mag, color=None):
    """Single per-finger arrow at `center` (used by `arrow` mode).

    Direction = normalize(dir_vec), length = mag * AGGREGATE_SCALE_PX_PER_UNIT
    (capped). When dir_vec is ~0 (no shear), draws a filled circle whose
    radius scales with mag so normal-only contact is still visible.
    """
    if color is None:
        color = tc.ARROW_COLOR_BGR
    min_visible = float(getattr(tc, "AGGREGATE_MIN_MAGNITUDE_VISIBLE", 0.0))
    if mag < min_visible:
        return
    norm = float(np.linalg.norm(dir_vec))
    scale = float(getattr(tc, "AGGREGATE_SCALE_PX_PER_UNIT", 1.5))
    max_len = float(getattr(tc, "AGGREGATE_MAX_LENGTH_PX", 350))
    length = float(np.clip(mag * scale, 0.0, max_len))
    cx, cy = int(center[0]), int(center[1])
    if norm < 1e-3 or length < tc.ARROW_MIN_LENGTH_PX:
        r = max(3, min(int(np.clip(length, 3, 24)), 24))
        cv2.circle(img, (cx, cy), r, color, -1)
        return
    unit = np.asarray(dir_vec, dtype=np.float32) / norm
    end = (int(round(cx + unit[0] * length)),
           int(round(cy + unit[1] * length)))
    cv2.arrowedLine(img, (cx, cy), end, color,
                    tc.ARROW_THICKNESS, tipLength=tc.ARROW_TIP_LENGTH)


def _single_point_at(img, center, mag, color=None):
    """Draw one solid circle at `center` with radius scaled by mag.

    Returns without drawing if mag is below tc.AGGREGATE_MIN_MAGNITUDE_VISIBLE.
    """
    if color is None:
        color = tc.ARROW_COLOR_BGR
    min_visible = float(getattr(tc, "AGGREGATE_MIN_MAGNITUDE_VISIBLE", 0.0))
    if mag < min_visible:
        return
    scale = float(getattr(tc, "AGGREGATE_SCALE_PX_PER_UNIT", 1.0))
    max_len = float(getattr(tc, "AGGREGATE_MAX_LENGTH_PX", 250))
    radius = int(np.clip(mag * scale, 3, max_len))
    cv2.circle(img, (int(center[0]), int(center[1])),
               radius, color, -1)


# ---------------------------------------------------------------------------
# Mode implementations — agent view
# ---------------------------------------------------------------------------

def _agent_grid(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """9 arrows per finger at projected cell positions.
    Direction = projected sensor xy shear; length = |sensor dz - baseline|.
    """
    pos_L = cell_positions_ee_frame("left")
    pos_R = cell_positions_ee_frame("right")
    pix_L, valid_L = project_points_to_image(pos_L, ee_pose, trc, intr)
    pix_R, valid_R = project_points_to_image(pos_R, ee_pose, trc, intr)
    shear_L, normal_L = shear_and_normal(tact_L, "left")
    shear_R, normal_R = shear_and_normal(tact_R, "right")
    img_dirs_L = _agent_shear_to_image_dirs(pos_L, shear_L, "left",
                                             ee_pose, trc, intr)
    img_dirs_R = _agent_shear_to_image_dirs(pos_R, shear_R, "right",
                                             ee_pose, trc, intr)
    _draw_shear_arrows(img_out, pix_L, img_dirs_L, normal_L, conn_L,
                       valid=valid_L, color=tc.LEFT_ARROW_COLOR_BGR)
    _draw_shear_arrows(img_out, pix_R, img_dirs_R, normal_R, conn_R,
                       valid=valid_R, color=tc.RIGHT_ARROW_COLOR_BGR)


def _agent_arrow(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """One arrow per finger at the grid center.
    Direction = normal-weighted sum of per-cell projected shear.
    Length    = aggregate |normal|.
    """
    pos_L = cell_positions_ee_frame("left")
    pos_R = cell_positions_ee_frame("right")
    pix_L, valid_L = project_points_to_image(pos_L, ee_pose, trc, intr)
    pix_R, valid_R = project_points_to_image(pos_R, ee_pose, trc, intr)
    if not valid_L.any() or not valid_R.any():
        return
    cen_L = np.mean(pix_L[valid_L].astype(np.float32), axis=0)
    cen_R = np.mean(pix_R[valid_R].astype(np.float32), axis=0)
    shear_L, normal_L = shear_and_normal(tact_L, "left")
    shear_R, normal_R = shear_and_normal(tact_R, "right")
    img_dirs_L = _agent_shear_to_image_dirs(pos_L, shear_L, "left",
                                             ee_pose, trc, intr)
    img_dirs_R = _agent_shear_to_image_dirs(pos_R, shear_R, "right",
                                             ee_pose, trc, intr)
    sum_L, mag_L = _aggregate_shear_per_finger(img_dirs_L, normal_L)
    sum_R, mag_R = _aggregate_shear_per_finger(img_dirs_R, normal_R)
    _shear_arrow_at(img_out, cen_L, sum_L, mag_L, color=tc.LEFT_ARROW_COLOR_BGR)
    _shear_arrow_at(img_out, cen_R, sum_R, mag_R, color=tc.RIGHT_ARROW_COLOR_BGR)


def _agent_point(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """Single solid circle per finger at the grid center, radius = |normal| aggregate."""
    pos_L = cell_positions_ee_frame("left")
    pos_R = cell_positions_ee_frame("right")
    pix_L, valid_L = project_points_to_image(pos_L, ee_pose, trc, intr)
    pix_R, valid_R = project_points_to_image(pos_R, ee_pose, trc, intr)
    mag_L, mag_R = _aggregate_normal_per_finger(tact_L, tact_R)
    if valid_L.any():
        _single_point_at(img_out,
                         np.mean(pix_L[valid_L].astype(np.float32), axis=0),
                         mag_L, color=tc.LEFT_ARROW_COLOR_BGR)
    if valid_R.any():
        _single_point_at(img_out,
                         np.mean(pix_R[valid_R].astype(np.float32), axis=0),
                         mag_R, color=tc.RIGHT_ARROW_COLOR_BGR)


def _bottom_bar(img_out, tact_L, tact_R):
    """Two horizontal binary bars at the bottom of the image.

    Each bar lights up in its per-side color ONLY if that finger's |normal|
    aggregate is above tc.BAR_TRIP_THRESHOLD; below threshold, nothing is
    drawn for that finger. (No gray placeholder — keeps the overlay clean
    when there's no contact, per the "hide-when-no-reading" rule.)
    """
    h, w = img_out.shape[:2]
    mag_L, mag_R = _aggregate_normal_per_finger(tact_L, tact_R)
    trip = float(getattr(tc, "BAR_TRIP_THRESHOLD", 50.0))
    bar_h = max(20, h // 12)
    bar_w_each = (w // 2) - 20
    y0 = h - bar_h - 10

    if mag_L >= trip:
        x0 = 10
        cv2.rectangle(img_out, (x0, y0), (x0 + bar_w_each, y0 + bar_h),
                      tc.LEFT_ARROW_COLOR_BGR, -1)
        cv2.putText(img_out, f"L  {int(mag_L)}",
                    (x0 + 8, y0 + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if mag_R >= trip:
        x0 = (w // 2) + 10
        cv2.rectangle(img_out, (x0, y0), (x0 + bar_w_each, y0 + bar_h),
                      tc.RIGHT_ARROW_COLOR_BGR, -1)
        cv2.putText(img_out, f"R  {int(mag_R)}",
                    (x0 + 8, y0 + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_agent_overlay(img, ee_pose, tactile_left_9x3, tactile_right_9x3,
                       connected_left, connected_right, trc, intrinsics,
                       mode=None):
    """Dispatcher. See tactile_config.VISUALIZATION_MODE for options."""
    mode = _select_mode(mode)
    img_out = img.copy()
    if mode == "grid":
        _agent_grid(img_out, ee_pose, tactile_left_9x3, tactile_right_9x3,
                    connected_left, connected_right, trc, intrinsics)
    elif mode == "arrow":
        _agent_arrow(img_out, ee_pose, tactile_left_9x3, tactile_right_9x3,
                     connected_left, connected_right, trc, intrinsics)
    elif mode == "point":
        _agent_point(img_out, ee_pose, tactile_left_9x3, tactile_right_9x3,
                     connected_left, connected_right, trc, intrinsics)
    elif mode == "bar":
        _bottom_bar(img_out, tactile_left_9x3, tactile_right_9x3)
    else:
        raise ValueError(f"unknown VISUALIZATION_MODE: {mode!r}")
    return _blend_overlay(img_out, img)


# ---------------------------------------------------------------------------
# Mode implementations — wrist view
# ---------------------------------------------------------------------------

def _wrist_anchors_for_finger(top_left_uv) -> np.ndarray:
    """3x3 grid of pixel anchors expanding from top_left_uv.

    With tc.WRIST_GRID_TRANSPOSED=False:  row axis = down,  col axis = right
    With tc.WRIST_GRID_TRANSPOSED=True :  row axis = right, col axis = down
    (i.e. the grid is rotated 90 deg, used when the finger appears horizontal
    in the wrist image instead of vertical).
    """
    anchors = np.zeros((9, 2), dtype=np.int32)
    u0, v0 = top_left_uv
    transposed = getattr(tc, "WRIST_GRID_TRANSPOSED", False)
    for idx, (row, col) in enumerate(tc.IDX_TO_ROWCOL):
        if transposed:
            du = row * tc.WRIST_CELL_PIX
            dv = col * tc.WRIST_CELL_PIX
        else:
            du = col * tc.WRIST_CELL_PIX
            dv = row * tc.WRIST_CELL_PIX
        anchors[idx] = (u0 + du, v0 + dv)
    return anchors


def _wrist_grid(img_out, tact_L, tact_R, conn_L, conn_R):
    anchors_L = _wrist_anchors_for_finger(tc.WRIST_LEFT_TOP_LEFT_UV)
    anchors_R = _wrist_anchors_for_finger(tc.WRIST_RIGHT_TOP_LEFT_UV)
    shear_L, normal_L = shear_and_normal(tact_L, "left")
    shear_R, normal_R = shear_and_normal(tact_R, "right")
    img_dirs_L = _wrist_shear_to_image_dirs(shear_L, "left")
    img_dirs_R = _wrist_shear_to_image_dirs(shear_R, "right")
    _draw_shear_arrows(img_out, anchors_L, img_dirs_L, normal_L, conn_L,
                       color=tc.LEFT_ARROW_COLOR_BGR)
    _draw_shear_arrows(img_out, anchors_R, img_dirs_R, normal_R, conn_R,
                       color=tc.RIGHT_ARROW_COLOR_BGR)


def _wrist_arrow(img_out, tact_L, tact_R, conn_L, conn_R):
    anchors_L = _wrist_anchors_for_finger(tc.WRIST_LEFT_TOP_LEFT_UV)
    anchors_R = _wrist_anchors_for_finger(tc.WRIST_RIGHT_TOP_LEFT_UV)
    cen_L = np.mean(anchors_L, axis=0).astype(np.float32)
    cen_R = np.mean(anchors_R, axis=0).astype(np.float32)
    shear_L, normal_L = shear_and_normal(tact_L, "left")
    shear_R, normal_R = shear_and_normal(tact_R, "right")
    img_dirs_L = _wrist_shear_to_image_dirs(shear_L, "left")
    img_dirs_R = _wrist_shear_to_image_dirs(shear_R, "right")
    sum_L, mag_L = _aggregate_shear_per_finger(img_dirs_L, normal_L)
    sum_R, mag_R = _aggregate_shear_per_finger(img_dirs_R, normal_R)
    _shear_arrow_at(img_out, cen_L, sum_L, mag_L, color=tc.LEFT_ARROW_COLOR_BGR)
    _shear_arrow_at(img_out, cen_R, sum_R, mag_R, color=tc.RIGHT_ARROW_COLOR_BGR)


def _wrist_point(img_out, tact_L, tact_R, conn_L, conn_R):
    anchors_L = _wrist_anchors_for_finger(tc.WRIST_LEFT_TOP_LEFT_UV)
    anchors_R = _wrist_anchors_for_finger(tc.WRIST_RIGHT_TOP_LEFT_UV)
    center_L = np.mean(anchors_L, axis=0).astype(np.float32)
    center_R = np.mean(anchors_R, axis=0).astype(np.float32)
    mag_L, mag_R = _aggregate_normal_per_finger(tact_L, tact_R)
    _single_point_at(img_out, center_L, mag_L, color=tc.LEFT_ARROW_COLOR_BGR)
    _single_point_at(img_out, center_R, mag_R, color=tc.RIGHT_ARROW_COLOR_BGR)


def draw_wrist_overlay(img, tactile_left_9x3, tactile_right_9x3,
                       connected_left, connected_right,
                       mode=None):
    """Dispatcher. See tactile_config.VISUALIZATION_MODE for options."""
    mode = _select_mode(mode)
    img_out = img.copy()
    if mode == "grid":
        _wrist_grid(img_out, tactile_left_9x3, tactile_right_9x3,
                    connected_left, connected_right)
    elif mode == "arrow":
        _wrist_arrow(img_out, tactile_left_9x3, tactile_right_9x3,
                     connected_left, connected_right)
    elif mode == "point":
        _wrist_point(img_out, tactile_left_9x3, tactile_right_9x3,
                     connected_left, connected_right)
    elif mode == "bar":
        _bottom_bar(img_out, tactile_left_9x3, tactile_right_9x3)
    else:
        raise ValueError(f"unknown VISUALIZATION_MODE: {mode!r}")
    return _blend_overlay(img_out, img)
