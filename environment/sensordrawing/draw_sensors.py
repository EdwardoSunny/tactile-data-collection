import numpy as np
from pathlib import Path
import cv2

from .transforms.load_transform import load_transform
from .xram_kinematics import (
    get_finger_transforms,
    left_finger_T_left_sensor,
    right_finger_T_right_sensor,
    get_xarm7_forward_kinematics,
)


def project_points(K, T_rc, points_3d):
    """
    Project points from robot base frame to camera frame and then to 2D image coordinates.
    points_3d should be a list or array of shape (N, 3)
    """
    if len(points_3d) == 0:
        return []

    pts_3d = np.float32(points_3d)

    R_mat = T_rc[:3, :3]
    t_vec = T_rc[:3, 3]
    pts_cam = (R_mat @ pts_3d.T).T + t_vec

    if np.any(pts_cam[:, 2] <= 0):
        return None

    x_norm = pts_cam[:, 0] / pts_cam[:, 2]
    y_norm = pts_cam[:, 1] / pts_cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    imgpts = np.vstack((u, v)).T
    return np.int32(imgpts)

class SensorDrawer:
    def __init__(self, camera_select='side'):
        self.camera_select = camera_select
        self.T_link7_to_cam = None  # Only used for wrist camera

        if camera_select == 'side':
            self.K = np.array([
                [609.36084,   0.     , 319.85806],
                [  0.     , 609.16156, 247.67696],
                [  0.     ,   0.     ,   1.     ]
            ], dtype=np.float32)
            serial_arg = '327122079374'
        elif camera_select == 'wrist':
            serial_arg = '332322072612'
            self.K = np.array([
                [606.7692,    0.     , 332.23438],
                [  0.     , 606.62616, 247.47606],
                [  0.     ,   0.     ,   1.     ]
            ], dtype=np.float32)
            # Load wrist camera calibration
            calib_path = Path(__file__).resolve().parent / "transforms" / "T_link7_to_cam.npy"
            self.T_link7_to_cam = np.load(str(calib_path))
            print(f"Loaded T_link7_to_cam from {calib_path}")
        else:
            raise ValueError(f"Unsupported camera_select '{camera_select}'")

        if camera_select == 'side':
            try:
                T_rc, serial_used, info = load_transform(serial=serial_arg)
                print(f"Loaded transform for camera {serial_used}:\n{info}")

                # Compensate constant offset in robot base frame:
                T_offset = np.eye(4)
                T_offset[0, 3] = -0.020
                T_offset[2, 3] = -0.040
                self.T_rc = T_rc @ T_offset
            except Exception as e:
                print(f"Error loading transform: {e}")
                self.T_rc = np.eye(4)
                serial_used = None
        else:
            self.T_rc = np.eye(4)  # Will be computed dynamically for wrist
            serial_used = serial_arg

        self.serial_used = serial_used

        local_pts_mm = [
            [0, 0, 0],
            [-12, -12, 0],
            [12, -12, 0],
            [-12, 12, 0],
            [12, 12, 0],
            [-12, 0, 0],
            [12, 0, 0],
            [0, -12, 0],
            [0, 12, 0]
        ]
        self.base_local_pts = np.array(local_pts_mm, dtype=float) / 1000.0

        # Sensor dot positions indexed by sensor idx (0-8)
        sensor_dot_positions_mm = np.array([
            [-12, 12, 0],   # idx 0
            [0, 12, 0],     # idx 1
            [12, 12, 0],    # idx 2
            [-12, 0, 0],    # idx 3
            [0, 0, 0],      # idx 4
            [12, 0, 0],     # idx 5
            [-12, -12, 0],  # idx 6
            [0, -12, 0],    # idx 7
            [12, -12, 0],   # idx 8
        ], dtype=float)
        self.sensor_dot_positions = sensor_dot_positions_mm / 1000.0

        # Apply constant z offset for wrist camera calibration
        if camera_select == 'wrist':
            z_offset = -7.6 / 1000.0  # 7.6mm in meters
            self.base_local_pts[:, 2] += z_offset
            self.sensor_dot_positions[:, 2] += z_offset

    def draw_on_image(self, image, angles, grip_pos, dot_size=10,
                      left_color=(0, 0, 255, 255), right_color=(255, 0, 0, 255),
                      scale=1.0, is_spatial=True,
                      normalized_left_sensor=None, normalized_right_sensor=None,
                      arrow_thickness=2, arrow_length_scale=0.12, mode="points9_arrow",
                      color_scale=1.0):
        """
        Draw sensor points and force arrows on the given image.

        Args:
            image: BGR/RGB numpy array to draw on (will not be modified in-place, currently ONLY work for 640x480).
            angles: List of 7 joint angles (degrees) from xArm servo_angle.
            grip_pos: Gripper position value from xArm get_gripper_position.
            dot_size: Diameter in pixels for each sensor dot.
            left_color: RGBA tuple for left sensor dots and arrows.
            right_color: RGBA tuple for right sensor dots and arrows.
            scale: Multiplier applied to sensor local point positions.
            is_spatial: If True, project 3D points onto the image via camera model.
                If False, draw a simple 2D grid overlay in screen corners.
            normalized_left_sensor: (9, 3) array of normalized force vectors for the
                left sensor, or None to skip. Each row is (fx, fy, fz) in sensor frame.
            normalized_right_sensor: Same as above for the right sensor.
            arrow_thickness: Line thickness in pixels for force arrows.
            arrow_length_scale: Multiplier converting normalized force magnitude to
                arrow length in meters (sensor local frame) before projection.
            mode: Drawing mode, one of:
                "points9_arrow" - Draw 9 dots + 9 force arrows per side (requires is_spatial=True).
                "points1_arrow" - Draw 1 center dot + 1 averaged force arrow per side (requires is_spatial=True).
                "points1_contact" - Draw 1 center dot per side only when any sensor norm >= 0.05
                    (supports both is_spatial=True and is_spatial=False).
                "points9_color" - Draw 9 dots per side, no arrows. Each dot color is derived
                    from its sensor xyz reading mapped linearly to RGB. Left maps to 0-127,
                    right maps to 128-255. Fully opaque. Supports both is_spatial=True and False.
            color_scale: Multiplier applied to sensor xyz values before color mapping
                in points9_color mode. Values are clamped to [-1, 1] after scaling.
        """
        valid_modes = ("points9_arrow", "points1_arrow", "points1_contact", "points9_color")
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode '{mode}', must be: {', '.join(valid_modes)}")

        if mode in ("points9_arrow", "points1_arrow") and not is_spatial:
            raise ValueError(f"mode='{mode}' requires is_spatial=True")

        img_out = image.copy()

        T_left, T_right = get_finger_transforms(angles, grip_pos)
        T_left_sensor = T_left @ left_finger_T_left_sensor
        T_right_sensor = T_right @ right_finger_T_right_sensor

        # For wrist camera, compute T_rc dynamically from current joint angles
        if self.camera_select == 'wrist' and self.T_link7_to_cam is not None:
            T_base_link7 = get_xarm7_forward_kinematics(angles)
            self.T_rc = self.T_link7_to_cam @ np.linalg.inv(T_base_link7)

        # Helper: determine draw order so the closer side is drawn on top
        def _left_first():
            left_cam = (self.T_rc[:3, :3] @ T_left_sensor[:3, 3]) + self.T_rc[:3, 3]
            right_cam = (self.T_rc[:3, :3] @ T_right_sensor[:3, 3]) + self.T_rc[:3, 3]
            return np.linalg.norm(left_cam) > np.linalg.norm(right_cam)

        if mode == "points9_arrow":
            local_pts = self.base_local_pts * scale
            ones = np.ones((len(local_pts), 1))
            local_pts_homo = np.hstack([local_pts, ones])
            n_pts = len(local_pts)

            left_pts_base = (T_left_sensor @ local_pts_homo.T).T[:, :3]
            right_pts_base = (T_right_sensor @ local_pts_homo.T).T[:, :3]
            all_pts_base = np.vstack([left_pts_base, right_pts_base])

            imgpts = project_points(self.K, self.T_rc, all_pts_base)
            if imgpts is not None:
                overlay = img_out.copy()
                imgpts_left = imgpts[:n_pts]
                imgpts_right = imgpts[n_pts:]
                left_first = _left_first()

                if left_first:
                    draw_order = [(imgpts_left, left_color), (imgpts_right, right_color)]
                else:
                    draw_order = [(imgpts_right, right_color), (imgpts_left, left_color)]

                for pts, color in draw_order:
                    for pt in pts:
                        cv2.circle(overlay, (pt[0], pt[1]), dot_size // 2, color[:3], -1)

                alpha_d = left_color[3] / 255.0 if len(left_color) > 3 else 1.0
                cv2.addWeighted(overlay, alpha_d, img_out, 1 - alpha_d, 0, img_out)

                # Draw arrows for sensor readings
                if normalized_left_sensor is not None or normalized_right_sensor is not None:
                    sensor_pts = self.sensor_dot_positions * scale
                    ones_s = np.ones((9, 1))
                    sensor_pts_homo = np.hstack([sensor_pts, ones_s])

                    if left_first:
                        arrow_order = [
                            (normalized_left_sensor, T_left_sensor, left_color),
                            (normalized_right_sensor, T_right_sensor, right_color),
                        ]
                    else:
                        arrow_order = [
                            (normalized_right_sensor, T_right_sensor, right_color),
                            (normalized_left_sensor, T_left_sensor, left_color),
                        ]

                    for sensor_data, T_sensor, color in arrow_order:
                        if sensor_data is None:
                            continue

                        # Arrow tip positions in sensor frame (full 3D offset)
                        arrow_tips = sensor_pts.copy()
                        arrow_tips[:, 0] += sensor_data[:, 0] * arrow_length_scale
                        arrow_tips[:, 1] += sensor_data[:, 1] * arrow_length_scale
                        arrow_tips[:, 2] += sensor_data[:, 2] * arrow_length_scale
                        arrow_tips_homo = np.hstack([arrow_tips, ones_s])

                        start_base = (T_sensor @ sensor_pts_homo.T).T[:, :3]
                        end_base = (T_sensor @ arrow_tips_homo.T).T[:, :3]

                        start_img = project_points(self.K, self.T_rc, start_base)
                        end_img = project_points(self.K, self.T_rc, end_base)

                        if start_img is not None and end_img is not None:
                            arrow_overlay = img_out.copy()
                            for i in range(9):
                                cv2.arrowedLine(arrow_overlay,
                                    tuple(start_img[i]), tuple(end_img[i]),
                                    color[:3], arrow_thickness, tipLength=0.3)
                            alpha_a = color[3] / 255.0 if len(color) > 3 else 1.0
                            cv2.addWeighted(arrow_overlay, alpha_a, img_out, 1 - alpha_a, 0, img_out)

        elif mode == "points1_arrow":
            # Single center dot + arrow with maxpooled force per side
            center_pt = self.base_local_pts[0:1] * scale  # (1,3) — the (0,0,0) point
            ones_c = np.ones((1, 1))
            center_homo = np.hstack([center_pt, ones_c])

            left_center_base = (T_left_sensor @ center_homo.T).T[:, :3]
            right_center_base = (T_right_sensor @ center_homo.T).T[:, :3]
            all_centers = np.vstack([left_center_base, right_center_base])

            imgpts = project_points(self.K, self.T_rc, all_centers)
            if imgpts is not None:
                overlay = img_out.copy()
                pt_left, pt_right = imgpts[0], imgpts[1]
                left_first = _left_first()

                if left_first:
                    draw_order = [(pt_left, left_color), (pt_right, right_color)]
                else:
                    draw_order = [(pt_right, right_color), (pt_left, left_color)]

                for pt, color in draw_order:
                    cv2.circle(overlay, (pt[0], pt[1]), dot_size // 2, color[:3], -1)

                alpha_d = left_color[3] / 255.0 if len(left_color) > 3 else 1.0
                cv2.addWeighted(overlay, alpha_d, img_out, 1 - alpha_d, 0, img_out)

                # Draw maxpooled arrows
                if normalized_left_sensor is not None or normalized_right_sensor is not None:
                    if left_first:
                        arrow_order = [
                            (normalized_left_sensor, T_left_sensor, left_color),
                            (normalized_right_sensor, T_right_sensor, right_color),
                        ]
                    else:
                        arrow_order = [
                            (normalized_right_sensor, T_right_sensor, right_color),
                            (normalized_left_sensor, T_left_sensor, left_color),
                        ]

                    for sensor_data, T_sensor, color in arrow_order:
                        if sensor_data is None:
                            continue
                        # Average across 9 sensors, then scale by 9
                        max_force = sensor_data.mean(axis=0) * 9

                        arrow_tip = center_pt.copy()
                        arrow_tip[0] += max_force * arrow_length_scale
                        arrow_tip_homo = np.hstack([arrow_tip, ones_c])

                        start_base = (T_sensor @ center_homo.T).T[:, :3]
                        end_base = (T_sensor @ arrow_tip_homo.T).T[:, :3]

                        start_img = project_points(self.K, self.T_rc, start_base)
                        end_img = project_points(self.K, self.T_rc, end_base)

                        if start_img is not None and end_img is not None:
                            arrow_overlay = img_out.copy()
                            cv2.arrowedLine(arrow_overlay,
                                tuple(start_img[0]), tuple(end_img[0]),
                                color[:3], arrow_thickness, tipLength=0.3)
                            alpha_a = color[3] / 255.0 if len(color) > 3 else 1.0
                            cv2.addWeighted(arrow_overlay, alpha_a, img_out, 1 - alpha_a, 0, img_out)

        elif mode == "points1_contact":
            # Single dot at center when any sensor norm >= 0.05
            contact_threshold = 0.05
            left_contact = (normalized_left_sensor is not None and
                            np.any(np.linalg.norm(normalized_left_sensor, axis=1) >= contact_threshold))
            right_contact = (normalized_right_sensor is not None and
                             np.any(np.linalg.norm(normalized_right_sensor, axis=1) >= contact_threshold))

            if is_spatial:
                center_pt = self.base_local_pts[0:1] * scale
                ones_c = np.ones((1, 1))
                center_homo = np.hstack([center_pt, ones_c])

                left_center_base = (T_left_sensor @ center_homo.T).T[:, :3]
                right_center_base = (T_right_sensor @ center_homo.T).T[:, :3]
                all_centers = np.vstack([left_center_base, right_center_base])

                imgpts = project_points(self.K, self.T_rc, all_centers)
                if imgpts is not None:
                    overlay = img_out.copy()
                    pt_left, pt_right = imgpts[0], imgpts[1]
                    left_first = _left_first()

                    if left_first:
                        draw_order = [(left_contact, pt_left, left_color),
                                      (right_contact, pt_right, right_color)]
                    else:
                        draw_order = [(right_contact, pt_right, right_color),
                                      (left_contact, pt_left, left_color)]

                    for contact, pt, color in draw_order:
                        if contact:
                            cv2.circle(overlay, (pt[0], pt[1]), dot_size // 2, color[:3], -1)

                    alpha_d = left_color[3] / 255.0 if len(left_color) > 3 else 1.0
                    cv2.addWeighted(overlay, alpha_d, img_out, 1 - alpha_d, 0, img_out)
            else:
                # Draw contact dots in screen corners
                h, w = img_out.shape[:2]
                overlay = img_out.copy()
                if left_contact:
                    cv2.circle(overlay, (50, h - 50), dot_size // 2, left_color[:3], -1)
                if right_contact:
                    cv2.circle(overlay, (w - 50, h - 50), dot_size // 2, right_color[:3], -1)
                if left_contact or right_contact:
                    alpha_d = left_color[3] / 255.0 if len(left_color) > 3 else 1.0
                    cv2.addWeighted(overlay, alpha_d, img_out, 1 - alpha_d, 0, img_out)

        elif mode == "points9_color":
            # 9 dots per side, color from sensor xyz mapped to RGB, no arrows
            def _sensor_to_colors(sensor_data, lo, hi):
                """Map normalized sensor xyz in [-1,1] to RGB in [lo, hi]."""
                if sensor_data is None:
                    mid = (lo + hi) // 2
                    return [(mid, mid, mid)] * 9
                colors = []
                for i in range(9):
                    rgb = []
                    for j in range(3):
                        v = np.clip(sensor_data[i, j] * color_scale, -1.0, 1.0)
                        rgb.append(int(lo + (v + 1.0) / 2.0 * (hi - lo)))
                    colors.append(tuple(rgb))
                return colors

            left_colors = _sensor_to_colors(normalized_left_sensor, 0, 127)
            right_colors = _sensor_to_colors(normalized_right_sensor, 128, 255)

            if is_spatial:
                sensor_pts = self.sensor_dot_positions * scale
                ones_s = np.ones((9, 1))
                sensor_pts_homo = np.hstack([sensor_pts, ones_s])

                left_pts_base = (T_left_sensor @ sensor_pts_homo.T).T[:, :3]
                right_pts_base = (T_right_sensor @ sensor_pts_homo.T).T[:, :3]

                left_img = project_points(self.K, self.T_rc, left_pts_base)
                right_img = project_points(self.K, self.T_rc, right_pts_base)

                if left_img is not None or right_img is not None:
                    # Build per-dot list: (cam_distance, pixel_pt, color)
                    dots = []
                    if left_img is not None:
                        for i in range(9):
                            pt_cam = (self.T_rc[:3, :3] @ left_pts_base[i]) + self.T_rc[:3, 3]
                            dots.append((np.linalg.norm(pt_cam), left_img[i], left_colors[i]))
                    if right_img is not None:
                        for i in range(9):
                            pt_cam = (self.T_rc[:3, :3] @ right_pts_base[i]) + self.T_rc[:3, 3]
                            dots.append((np.linalg.norm(pt_cam), right_img[i], right_colors[i]))

                    # Sort by distance descending (further dots drawn first, closer on top)
                    dots.sort(key=lambda d: d[0], reverse=True)

                    for _, pt, color in dots:
                        cv2.circle(img_out, (pt[0], pt[1]), dot_size // 2, color, -1)
            else:
                # Draw 3x3 color grids in screen corners
                h, w = img_out.shape[:2]
                x_offset_left = 50
                x_offset_right = w - 100
                y_offset = h - 50 - 2 * (dot_size + 5)

                for i in range(9):
                    gx = i % 3
                    gy = i // 3
                    px_l = x_offset_left + gx * (dot_size + 5)
                    px_r = x_offset_right + gx * (dot_size + 5)
                    py = y_offset + gy * (dot_size + 5)
                    cv2.circle(img_out, (int(px_l), int(py)), dot_size // 2, left_colors[i], -1)
                    cv2.circle(img_out, (int(px_r), int(py)), dot_size // 2, right_colors[i], -1)

        return img_out
