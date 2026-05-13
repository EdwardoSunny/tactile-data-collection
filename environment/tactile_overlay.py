"""
Draw the 2-finger x 9-cell tactile overlay on the agent and wrist camera images.

Agent (third-person) camera: cell positions are 3D (EE frame, mm), projected
through the current robot pose and the camera's extrinsics+intrinsics to pixels.

Wrist camera: 18 fixed pixel anchors (9 per finger) from tactile_config; no
projection. Arrow direction for each cell points toward the OPPOSITE finger's
projected centroid in the image, so the arrows visually represent "pushed
inward toward the gripped object".
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import tactile_config as tc


# -----------------------------------------------------------------------------
# Per-cell baseline used to "normalize" the squeeze magnitude away from the
# static magnetic field each cell sees at rest. Without this, every arrow has
# a length proportional to the cell's idle |B| (which is dominated by the
# magnet sitting near each magnetometer), so all arrows look basically the
# same size regardless of contact. With it, arrow length is proportional to
# the change from idle — which is the actual contact signal.
#
# Set by the recording script after a short idle capture (gripper open, no
# contact). Shape (2, 9, 3) — finger x cell x (Bx,By,Bz). None disables.
# -----------------------------------------------------------------------------
_BASELINE = None  # type: ignore


def set_baseline(baseline_2x9x3):
    """Install per-cell idle baseline used by squeeze_magnitudes()."""
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


def squeeze_magnitudes(tactile_9x3: np.ndarray, side: str) -> np.ndarray:
    """Per-cell scalar squeeze. Positive = squeezed (magnet pushed into finger).

    If a baseline has been installed via set_baseline(), the per-cell idle
    field is subtracted first so the result is "change from idle", not the
    raw absolute value — this is what actually carries the contact signal.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[tc.FORCE_AXIS]
    sign = tc.LEFT_FORCE_SIGN if side == "left" else tc.RIGHT_FORCE_SIGN
    val = tactile_9x3[:, axis_idx].astype(np.float32)
    if _BASELINE is not None:
        finger_idx = 0 if side == "left" else 1
        val = val - _BASELINE[finger_idx, :, axis_idx]
    return sign * val


def _aggregate_per_finger(tact_L, tact_R):
    """Per-finger aggregate squeeze magnitude.

    Reduces the 9 per-cell |delta-from-baseline| values to a single scalar
    per finger. The method is set by ``tactile_config.AGGREGATE_METHOD``:

      "max"   : peak |delta| across the 9 cells (noise-robust; default).
      "sum"   : sum of |delta|       (amplifies multi-cell contact AND noise).
      "mean"  : average |delta|      (rarely useful; included for completeness).

    Used by the "arrow", "point", and "bar" visualization modes.
    """
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


def _select_mode(mode):
    return (mode or getattr(tc, "VISUALIZATION_MODE", "arrow")).lower()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_arrows(img, anchors, magnitudes, connected, dirs_unit, valid=None,
                 target_xy=None):
    """In-place arrow drawing. img is a BGR ndarray (modified).

    anchors      : (N, 2) int pixel coords
    magnitudes   : (N,) scalar magnitudes (positive = squeezed)
    connected    : (N,) 0/1
    dirs_unit    : (N, 2) image-plane unit vectors (where each arrow points)
    valid        : optional (N,) bool — points that don't project well are
                   treated as off-screen and skipped (no marker).
    target_xy    : optional (2,) — the point each arrow is conceptually
                   pointing TOWARD. When given, each arrow's length is also
                   capped at tc.AGGREGATE_ARROW_FRAC_CAP * (distance from
                   anchor to target_xy). Keeps grid arrows scaled to the
                   actual finger-pair geometry instead of overshooting.

    A cell whose |magnitude| is below tc.GRID_MIN_MAGNITUDE_VISIBLE draws
    NOTHING. Disconnected cells still show a small gray marker so hardware
    faults stay visible.
    """
    h, w = img.shape[:2]
    n = len(anchors)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    min_visible = float(getattr(tc, "GRID_MIN_MAGNITUDE_VISIBLE", 0.0))
    frac_cap = float(getattr(tc, "AGGREGATE_ARROW_FRAC_CAP", 2.0))

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
        mag = float(magnitudes[i])
        abs_mag = abs(mag)
        if abs_mag < min_visible:
            continue  # no meaningful signal — draw nothing
        length = abs_mag * tc.ARROW_SCALE_PX_PER_UNIT
        length = float(np.clip(length, 0.0, tc.ARROW_MAX_LENGTH_PX))
        if target_xy is not None and frac_cap > 0:
            dist = float(np.linalg.norm(target_xy - anchors[i].astype(np.float32)))
            length = min(length, dist * frac_cap)
        if length < tc.ARROW_MIN_LENGTH_PX:
            cv2.circle(img, (u, v), 2, tc.ARROW_COLOR_BGR, -1)
            continue
        # If the signed magnitude is negative, flip the arrow direction.
        sign = 1.0 if mag >= 0 else -1.0
        dx = float(dirs_unit[i, 0]) * length * sign
        dy = float(dirs_unit[i, 1]) * length * sign
        end = (int(round(u + dx)), int(round(v + dy)))
        cv2.arrowedLine(img, (u, v), end, tc.ARROW_COLOR_BGR,
                        tc.ARROW_THICKNESS, tipLength=tc.ARROW_TIP_LENGTH)


def _agent_project_pixels(ee_pose, trc, intrinsics):
    """Project both fingers' cells and return (pix_L, valid_L, pix_R, valid_R)."""
    pos_L = cell_positions_ee_frame("left")
    pos_R = cell_positions_ee_frame("right")
    pix_L, valid_L = project_points_to_image(pos_L, ee_pose, trc, intrinsics)
    pix_R, valid_R = project_points_to_image(pos_R, ee_pose, trc, intrinsics)
    return pix_L, valid_L, pix_R, valid_R


def _unit_dirs(anchors_int, target_xy):
    deltas = target_xy[None, :] - anchors_int.astype(np.float32)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms = np.where(norms > 1e-3, norms, 1.0)
    return deltas / norms


def _agent_grid(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """9 arrows per finger at projected cell positions (the original mode)."""
    pix_L, valid_L, pix_R, valid_R = _agent_project_pixels(ee_pose, trc, intr)
    h, w = img_out.shape[:2]
    center_R = (np.mean(pix_R[valid_R].astype(np.float32), axis=0)
                if valid_R.any() else np.array([w / 2.0, h / 2.0], dtype=np.float32))
    center_L = (np.mean(pix_L[valid_L].astype(np.float32), axis=0)
                if valid_L.any() else np.array([w / 2.0, h / 2.0], dtype=np.float32))
    dirs_L = _unit_dirs(pix_L, center_R)
    dirs_R = _unit_dirs(pix_R, center_L)
    mag_L = squeeze_magnitudes(tact_L, "left")
    mag_R = squeeze_magnitudes(tact_R, "right")
    _draw_arrows(img_out, pix_L, mag_L, conn_L, dirs_L, valid=valid_L, target_xy=center_R)
    _draw_arrows(img_out, pix_R, mag_R, conn_R, dirs_R, valid=valid_R, target_xy=center_L)


def _agent_arrow(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """Single arrow per finger at the grid center, length = aggregate squeeze."""
    pix_L, valid_L, pix_R, valid_R = _agent_project_pixels(ee_pose, trc, intr)
    if not valid_L.any() or not valid_R.any():
        return
    cen_L = np.mean(pix_L[valid_L].astype(np.float32), axis=0)
    cen_R = np.mean(pix_R[valid_R].astype(np.float32), axis=0)
    mag_L, mag_R = _aggregate_per_finger(tact_L, tact_R)
    _single_arrow_at(img_out, cen_L, cen_R, mag_L)
    _single_arrow_at(img_out, cen_R, cen_L, mag_R)


def _agent_point(img_out, ee_pose, tact_L, tact_R, conn_L, conn_R, trc, intr):
    """Single solid circle per finger at the grid center, radius = aggregate."""
    pix_L, valid_L, pix_R, valid_R = _agent_project_pixels(ee_pose, trc, intr)
    mag_L, mag_R = _aggregate_per_finger(tact_L, tact_R)
    if valid_L.any():
        _single_point_at(img_out, np.mean(pix_L[valid_L].astype(np.float32), axis=0), mag_L)
    if valid_R.any():
        _single_point_at(img_out, np.mean(pix_R[valid_R].astype(np.float32), axis=0), mag_R)


def _single_arrow_at(img, center, target, mag):
    """Draw one arrow from `center` toward `target`, length scaled by mag.

    Returns without drawing if mag is below tc.AGGREGATE_MIN_MAGNITUDE_VISIBLE
    (no meaningful contact signal). Otherwise length is mag * scale, capped
    at AGGREGATE_MAX_LENGTH_PX AND at half the distance to `target` so
    opposing arrows from the two fingers don't overlap.
    """
    min_visible = float(getattr(tc, "AGGREGATE_MIN_MAGNITUDE_VISIBLE", 0.0))
    if mag < min_visible:
        return
    dir_vec = target - center
    norm = float(np.linalg.norm(dir_vec))
    if norm < 1e-3:
        return
    unit = dir_vec / norm
    scale = float(getattr(tc, "AGGREGATE_SCALE_PX_PER_UNIT", 1.5))
    max_len = float(getattr(tc, "AGGREGATE_MAX_LENGTH_PX", 350))
    frac_cap = float(getattr(tc, "AGGREGATE_ARROW_FRAC_CAP", 2.0))
    length = float(np.clip(mag * scale, 0.0, max_len))
    if frac_cap > 0:
        length = min(length, norm * frac_cap)
    if length < tc.ARROW_MIN_LENGTH_PX:
        return  # too small to draw an arrow AND below the visibility floor
    end = (int(round(center[0] + unit[0] * length)),
           int(round(center[1] + unit[1] * length)))
    cv2.arrowedLine(img, (int(center[0]), int(center[1])), end,
                    tc.ARROW_COLOR_BGR, tc.ARROW_THICKNESS,
                    tipLength=tc.ARROW_TIP_LENGTH)


def _single_point_at(img, center, mag):
    """Draw one solid circle at `center` with radius scaled by mag.

    Returns without drawing if mag is below tc.AGGREGATE_MIN_MAGNITUDE_VISIBLE.
    """
    min_visible = float(getattr(tc, "AGGREGATE_MIN_MAGNITUDE_VISIBLE", 0.0))
    if mag < min_visible:
        return
    scale = float(getattr(tc, "AGGREGATE_SCALE_PX_PER_UNIT", 1.0))
    max_len = float(getattr(tc, "AGGREGATE_MAX_LENGTH_PX", 250))
    radius = int(np.clip(mag * scale, 3, max_len))
    cv2.circle(img, (int(center[0]), int(center[1])),
               radius, tc.ARROW_COLOR_BGR, -1)


def _bottom_bar(img_out, tact_L, tact_R):
    """Two horizontal binary bars at the bottom of the image.

    Each bar is drawn green ONLY if that finger's aggregate is above
    tc.BAR_TRIP_THRESHOLD; below threshold, nothing is drawn for that
    finger. (No gray placeholder — keeps the overlay clean when there's
    no contact, per the "hide-when-no-reading" rule.)
    """
    h, w = img_out.shape[:2]
    mag_L, mag_R = _aggregate_per_finger(tact_L, tact_R)
    trip = float(getattr(tc, "BAR_TRIP_THRESHOLD", 50.0))
    bar_h = max(20, h // 12)
    bar_w_each = (w // 2) - 20
    y0 = h - bar_h - 10
    lit = (0, 220, 0)        # green

    if mag_L >= trip:
        x0 = 10
        cv2.rectangle(img_out, (x0, y0), (x0 + bar_w_each, y0 + bar_h),
                      lit, -1)
        cv2.putText(img_out, f"L  {int(mag_L)}",
                    (x0 + 8, y0 + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if mag_R >= trip:
        x0 = (w // 2) + 10
        cv2.rectangle(img_out, (x0, y0), (x0 + bar_w_each, y0 + bar_h),
                      lit, -1)
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
    return img_out


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
    center_L = np.mean(anchors_L, axis=0).astype(np.float32)
    center_R = np.mean(anchors_R, axis=0).astype(np.float32)
    dirs_L = _unit_dirs(anchors_L, center_R)
    dirs_R = _unit_dirs(anchors_R, center_L)
    mag_L = squeeze_magnitudes(tact_L, "left")
    mag_R = squeeze_magnitudes(tact_R, "right")
    _draw_arrows(img_out, anchors_L, mag_L, conn_L, dirs_L, target_xy=center_R)
    _draw_arrows(img_out, anchors_R, mag_R, conn_R, dirs_R, target_xy=center_L)


def _wrist_arrow(img_out, tact_L, tact_R, conn_L, conn_R):
    anchors_L = _wrist_anchors_for_finger(tc.WRIST_LEFT_TOP_LEFT_UV)
    anchors_R = _wrist_anchors_for_finger(tc.WRIST_RIGHT_TOP_LEFT_UV)
    center_L = np.mean(anchors_L, axis=0).astype(np.float32)
    center_R = np.mean(anchors_R, axis=0).astype(np.float32)
    mag_L, mag_R = _aggregate_per_finger(tact_L, tact_R)
    _single_arrow_at(img_out, center_L, center_R, mag_L)
    _single_arrow_at(img_out, center_R, center_L, mag_R)


def _wrist_point(img_out, tact_L, tact_R, conn_L, conn_R):
    anchors_L = _wrist_anchors_for_finger(tc.WRIST_LEFT_TOP_LEFT_UV)
    anchors_R = _wrist_anchors_for_finger(tc.WRIST_RIGHT_TOP_LEFT_UV)
    center_L = np.mean(anchors_L, axis=0).astype(np.float32)
    center_R = np.mean(anchors_R, axis=0).astype(np.float32)
    mag_L, mag_R = _aggregate_per_finger(tact_L, tact_R)
    _single_point_at(img_out, center_L, mag_L)
    _single_point_at(img_out, center_R, mag_R)


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
    return img_out
