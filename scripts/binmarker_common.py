"""Shared logic for the binmarker overlay variant.

binmarker (ported from LIBERO_contact_overlay's `binbars` style, presented
spatially): one arrow per finger, anchored at the finger center pad
(points1_arrow anchor), with FIXED length AND FIXED direction (sensor-frame
pad normal), drawn iff that finger's binary contact state is ON. Like
LIBERO's binbars, the overlay carries NO force magnitude and NO force
direction — the only signal is the per-finger contact bit.

Normalization: unlike render_overlays.py (global median offset), the offset
here is the median over KNOWN-IDLE frames (first IDLE_FRAMES frames of each
episode — robot just homed, gripper open, no contact). The global median is
contact-polluted on tasks that spend most frames in contact (charger: 60%
closed -> idle read as ~1.2 fake magnitude; idle-window offset fixes it to
~0.24). Scales are p99 of |raw - offset| as usual.
"""
import numpy as np

FIXED_ARROW_LEN = 0.12    # arrow_length_scale when contact is on (matches the
                          # points9_arrow default scale -> "strong contact" size)
IDLE_FRAMES = 10          # frames after each episode start treated as idle
DATASET_PERCENTILE = 99.0

# Contact detection: metric is the max over the 9 cells of each cell's
# normalized-vector L2 norm ("max-cell"). Chosen over ||sum of cells|| after
# tuning on all 400 episodes: sum cancels opposing xy components and is
# dominated by task-specific spikes, giving idle p99.5 up to 0.45 (charger)
# vs real light holds at 0.5 (tube/dishwasher weak finger). Max-cell gives
# idle p99.5 <= 0.24 on every task with real contacts >= ~0.3.
# Hysteresis + debounce (Schmitt trigger) so light-but-real holds don't
# flicker: ON after N_ON consecutive frames >= T_ON, OFF after N_OFF
# consecutive frames < T_OFF. Validated: 0 idle false positives in 400
# post-homing windows; cube weak-finger holds (p05 dip 0.23) stay latched.
T_ON = 0.25
T_OFF = 0.18
N_ON = 2
N_OFF = 3


def idle_window_normalization(raw, episode_ends):
    """offset from idle windows, per-finger p99 scales from whole dataset.

    raw: (N, 2, 9, 3) float64. Returns (offset (2,9,3), scale_xy (2,), scale_z (2,)).
    """
    ends = np.asarray(episode_ends)
    starts = np.r_[0, ends[:-1]]
    idle_idx = np.concatenate(
        [np.arange(s, min(s + IDLE_FRAMES, e)) for s, e in zip(starts, ends)])
    offset = np.median(raw[idle_idx], axis=0)
    delta = raw - offset
    scale_xy = np.empty(2)
    scale_z = np.empty(2)
    for fi in range(2):
        scale_xy[fi] = max(np.percentile(np.abs(delta[:, fi, :, :2]), DATASET_PERCENTILE), 1.0)
        scale_z[fi] = max(np.percentile(np.abs(delta[:, fi, :, 2]), DATASET_PERCENTILE), 1.0)
    return offset, scale_xy, scale_z


def normalize_all(raw, offset, scale_xy, scale_z):
    """(N,2,9,3) raw -> normalized, per-finger scales."""
    n = np.empty_like(raw)
    for fi in range(2):
        n[:, fi, :, :2] = (raw[:, fi, :, :2] - offset[fi, :, :2]) / scale_xy[fi]
        n[:, fi, :, 2] = (raw[:, fi, :, 2] - offset[fi, :, 2]) / scale_z[fi]
    return n


def contact_metric(norm):
    """(N,2,9,3) normalized -> (N,2) max-cell magnitude per finger."""
    return np.linalg.norm(norm, axis=-1).max(axis=2)


def contact_states(m_episode):
    """(T,) per-frame metric for ONE finger of ONE episode -> (T,) bool ON/OFF.

    Hysteresis + debounce; state resets at episode boundaries (call per
    episode). Never carries state across episodes — the robot re-homes.
    """
    on = np.zeros(len(m_episode), dtype=bool)
    state = False
    c_on = c_off = 0
    for i, v in enumerate(m_episode):
        if not state:
            c_on = c_on + 1 if v >= T_ON else 0
            if c_on >= N_ON:
                state, c_off = True, 0
        else:
            c_off = c_off + 1 if v < T_OFF else 0
            if c_off >= N_OFF:
                state, c_on = False, 0
        on[i] = state
    return on


# Fixed marker direction in the SENSOR-LOCAL frame: +z = the pad normal
# (pressure axis). The drawn arrow is therefore a rigid "flag" attached to the
# finger — identical geometry every frame contact is on. Its 2D appearance
# varies only with gripper pose / camera view (as any spatial overlay does),
# NEVER with the force reading.
FIXED_DIRECTION = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def binmarker_feed(norm_frame_finger, is_on):
    """Precomputed ON/OFF -> (9,3) SensorDrawer feed. PURE BINARY.

    points1_arrow draws the arrow as mean(cells)*9 * arrow_length_scale, so
    feeding tile(FIXED_DIRECTION/9) yields an arrow of length exactly
    arrow_length_scale along the constant sensor-frame direction; zeros yield
    no visible arrow. BOTH force magnitude AND force direction are omitted —
    the only information in the overlay is the per-finger contact bit from
    contact_states (max-cell metric + hysteresis). norm_frame_finger is
    accepted for signature compatibility but intentionally unused.
    """
    if not is_on:
        return np.zeros((9, 3), dtype=np.float32)
    return np.tile(FIXED_DIRECTION / 9.0, (9, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# binmarker9: the 9-arrow counterpart of binmarker.
# ---------------------------------------------------------------------------
# binmarker (1 arrow)  and binmarker9 (9 arrows) form a matched pair that
# isolates ARROW COUNT while holding the information content fixed: both carry
# ONLY the per-finger binary contact bit — no force direction, no magnitude.
#
# Why a separate feed builder instead of reusing binmarker_feed():
#   points1_arrow draws ONE arrow from mean(cells)*9, so a feed of
#   tile(FIXED_DIRECTION/9) integrates to exactly FIXED_DIRECTION.
#   points9_arrow instead draws a SEPARATE arrow per cell i, with tip at
#   sensor_pts[i] + sensor_data[i]*arrow_length_scale. To give each of the 9
#   arrows the same length/direction as binmarker's single arrow, each cell
#   must carry the FULL unit vector — NOT the /9-scaled one. Feeding the /9
#   version here would render 9 arrows at 1/9 the intended length.
def binmarker9_feed(norm_frame_finger, is_on):
    """Precomputed ON/OFF -> (9,3) SensorDrawer feed for mode='points9_arrow'.

    PURE BINARY, exactly like binmarker_feed: every visible arrow has fixed
    length (arrow_length_scale=FIXED_ARROW_LEN) and fixed direction
    (FIXED_DIRECTION, the sensor-frame pad normal). Both force magnitude and
    force direction are omitted; the only signal is the per-finger contact bit.
    `norm_frame_finger` is accepted for signature compatibility and is
    intentionally unused (the output cannot depend on the force reading).
    """
    if not is_on:
        return np.zeros((9, 3), dtype=np.float32)
    return np.tile(FIXED_DIRECTION, (9, 1)).astype(np.float32)
