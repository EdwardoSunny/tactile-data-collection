"""
Kinematics-aware tactile overlay, delegated to the vendored sensordrawing package.

This module replaces the legacy shear-arrow renderer. The actual geometry —
camera intrinsics, finger-link kinematics from the xArm URDF, sensor-on-PCB
positions, the wrist-camera dynamic T_rc, alpha blending, draw order — all
lives in environment/sensordrawing/. We expose:

  - MODES / MODE_KEYS : the 6 (mode, is_spatial) variants that match
                        sensordrawing/example_draw.py.
  - mode_key(mode, is_spatial) : the canonical string label used as zarr
                                  array suffix and CLI flag value.
  - SensorOverlay     : holds one SensorDrawer per camera role ('side',
                        'wrist') + one SensorNormalizer per finger; both
                        threads.py and scripts/render_overlays.py use this.

Inputs to draw():
  - role           : 'side' (third-person) or 'wrist'
  - image          : 640x480 BGR uint8 (sensordrawing's K is calibrated at 640x480)
  - angles         : list/array of 7 joint angles in degrees (from arm.get_servo_angle)
  - grip_pos       : raw gripper position 0-850 (from arm.get_gripper_position)
  - normalized_L/R : (9, 3) normalized tactile from SensorNormalizer.normalize()
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .sensordrawing import SensorDrawer, SensorNormalizer


# (mode, is_spatial, arrow_length_scale) — verbatim from sensordrawing/example_draw.py.
# Order is also the canonical ordering for render_overlays.py output arrays.
MODES = [
    ("points9_arrow",   True,  0.06),
    ("points1_arrow",   True,  0.006),
    ("points1_contact", True,  0.12),
    ("points9_color",   True,  0.12),
    ("points1_contact", False, 0.12),
    ("points9_color",   False, 0.12),
]


def mode_key(mode: str, is_spatial: bool) -> str:
    """Canonical string label for a (mode, is_spatial) pair.

    Used both as the suffix on img_{i}_{key} zarr arrays and as the choice
    value for the --viz-mode CLI flag.
    """
    if mode in ("points9_arrow", "points1_arrow"):
        return mode  # spatial-only modes have no flat variant
    return f"{mode}_{'spatial' if is_spatial else 'flat'}"


MODE_KEYS = [mode_key(m, s) for (m, s, _) in MODES]
DEFAULT_MODE_KEY = "points9_arrow"


def _key_to_spec(key: str) -> Tuple[str, bool, float]:
    for m, s, scale in MODES:
        if mode_key(m, s) == key:
            return m, s, scale
    raise ValueError(f"Unknown mode_key {key!r}; expected one of {MODE_KEYS}")


def _calib_path(side: str) -> str:
    return str(Path(__file__).parent / "sensordrawing" / f"calibration_{side}.npz")


class SensorOverlay:
    """Owns SensorDrawer instances (one per camera role) + SensorNormalizer
    instances (one per finger).

    Construct once at session start. Per-tick / per-frame:
        nL, nR = ov.normalize(raw_L_9x3, raw_R_9x3)
        out = ov.draw('side', img_bgr_640x480, angles, grip_pos, nL, nR, mode_key='points9_arrow')

    Both `image` and the returned drawing are 640x480 uint8 BGR.
    """

    # RGBA colors (alpha=77/255 ~= 0.3 transparency), mirroring example_draw.py.
    LEFT_COLOR = (255, 0, 0, 77)
    RIGHT_COLOR = (0, 0, 255, 77)

    def __init__(self):
        self.drawers = {
            "side":  SensorDrawer(camera_select="side"),
            "wrist": SensorDrawer(camera_select="wrist"),
        }
        self.norm_L = SensorNormalizer(_calib_path("left"))
        self.norm_R = SensorNormalizer(_calib_path("right"))

    def normalize(self, raw_L: Optional[np.ndarray], raw_R: Optional[np.ndarray]):
        """Apply per-finger SensorNormalizer to raw (9, 3) tactile arrays.

        Returns (normalized_L, normalized_R); either may be None if the
        corresponding input was None.
        """
        nL = None if raw_L is None else self.norm_L.normalize(np.asarray(raw_L, dtype=float))
        nR = None if raw_R is None else self.norm_R.normalize(np.asarray(raw_R, dtype=float))
        return nL, nR

    def draw(self, role: str, image: np.ndarray, angles, grip_pos: float,
             normalized_L: Optional[np.ndarray], normalized_R: Optional[np.ndarray],
             mode_key: Optional[str] = None,
             mode: Optional[str] = None,
             is_spatial: Optional[bool] = None,
             arrow_length_scale: Optional[float] = None) -> np.ndarray:
        """Render one overlay variant onto a 640x480 BGR image.

        Either pass `mode_key` (canonical string from MODE_KEYS), or pass
        `mode` + `is_spatial` directly. `arrow_length_scale` falls back to
        the per-mode default from MODES.
        """
        if role not in self.drawers:
            raise ValueError(f"role must be 'side' or 'wrist', got {role!r}")

        if mode_key is not None:
            mode, is_spatial, default_scale = _key_to_spec(mode_key)
            if arrow_length_scale is None:
                arrow_length_scale = default_scale
        else:
            if mode is None or is_spatial is None:
                raise ValueError("Pass either mode_key, or both mode and is_spatial")
            if arrow_length_scale is None:
                arrow_length_scale = next(
                    (s for m, sp, s in MODES if m == mode and sp == is_spatial),
                    0.12,
                )

        return self.drawers[role].draw_on_image(
            image, angles, grip_pos,
            normalized_left_sensor=normalized_L,
            normalized_right_sensor=normalized_R,
            mode=mode,
            is_spatial=is_spatial,
            arrow_length_scale=arrow_length_scale,
            left_color=self.LEFT_COLOR,
            right_color=self.RIGHT_COLOR,
        )
