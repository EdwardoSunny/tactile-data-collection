"""
Post-hoc tactile-overlay renderer (sensordrawing edition).

Reads the raw recording at SRC (default teleop_data.zarr) and writes a fresh
DST zarr (default teleop_data_overlay.zarr) containing the SAME per-frame
data PLUS six overlay variants per camera, drawn by the vendored
sensordrawing pipeline. Variant suffixes mirror tactile_overlay.MODE_KEYS:

    /data/img_{i}_points9_arrow
    /data/img_{i}_points1_arrow
    /data/img_{i}_points1_contact_spatial
    /data/img_{i}_points9_color_spatial
    /data/img_{i}_points1_contact_flat
    /data/img_{i}_points9_color_flat

DST is fully regenerated on every run (any existing directory is removed
first), so re-running picks up any newly-recorded episodes appended to SRC.

sensordrawing's camera intrinsics (K) and robot->camera extrinsic (T_rc)
are calibrated for native 640x480. The recorded zarr stores 224x224 frames,
so each frame is upscaled to 640x480 before drawing and downsized back
afterwards. Inputs sensordrawing needs each frame:
  - /data/joint_angles : (N, 7) servo angles in degrees
  - /data/grip_pos     : (N, 1) raw xArm gripper position (0..850)
  - /data/tactile      : (N, 2, 9, 3) raw per-finger tactile xyz
  Plus per-finger calibration_{left,right}.npz under environment/sensordrawing/.

Tactile NORMALIZATION is computed from the dataset itself, NOT from the
1.5s idle baseline captured at session start or from the shipped per-cell
hardware calibration. Specifically:
  - per-cell, per-axis OFFSET = median across all frames (robust idle estimate)
  - per-finger XY scale       = 95th percentile of |xy - offset_xy|
  - per-finger Z  scale       = 95th percentile of |z  - offset_z|
After this, the strongest 5% of contacts saturate above 1.0 and lighter
contacts get the rest of the dynamic range. The arrows you see are therefore
scaled to "what this task's contacts actually look like" — a heavy press in
cube and the same heavy press in tube will both produce arrows near full
length, even if the absolute raw counts differ.

Arrows + dots are drawn bolder than the sensordrawing defaults (5 px lines,
16 px dots) so they read cleanly when viewed at the recorded 224x224 scale.
"""
import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np
import zarr

# Allow running as `python scripts/render_overlays.py ...` from repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.tactile_overlay import MODES, MODE_KEYS, SensorOverlay, mode_key


# sensordrawing was calibrated at 640x480; upscale recorded frames to this
# before drawing so K and T_rc map onto the right pixel coordinates.
NATIVE_W, NATIVE_H = 640, 480


# Visual style: arrows + dots are drawn bolder than the sensordrawing defaults
# (2 px lines, 10 px dots) so they read clearly at 224x224 viewing scale.
BOLD_ARROW_THICKNESS = 8
BOLD_DOT_SIZE = 22


# Dataset-wide normalization knobs. Per cell+axis offset is the median across
# the WHOLE dataset (robust idle estimate); per-finger scale is the Nth
# percentile of |raw - offset| across all frames+cells for that finger.
# 99 instead of 95 keeps the saturation point well past the noise band — at
# p95 in datasets where most frames are idle, the scale ends up dominated by
# the upper end of the noise distribution and lighter contacts amplify noise
# visually.
DATASET_PERCENTILE = 99.0

# After normalization, suppress arrows for cells whose normalized force-vector
# magnitude is below this threshold. Acts as a hard noise gate: idle wobble
# normalizes to a small but non-zero vector (because the noise floor is real),
# and without a gate it draws visible jitter even when nothing's touching the
# fingers. 0.12 = "ignore the bottom 12% of arrow length", which empirically
# kills idle jitter on cube/tube while preserving any real contact arrow.
NOISE_DEADBAND = 0.12


def _apply_deadband(norm_arr, threshold):
    """Per-cell hard noise gate: zero out cells whose normalized (x, y, z)
    vector has L2 magnitude below `threshold`. Preserves vector direction
    for cells above the gate.

    norm_arr shape: (9, 3) or None. Returns same shape.
    """
    if norm_arr is None or threshold <= 0:
        return norm_arr
    arr = np.asarray(norm_arr, dtype=np.float32).copy()
    mag = np.linalg.norm(arr, axis=-1, keepdims=True)  # (9, 1)
    return np.where(mag < threshold, 0.0, arr)


def _dataset_normalization(tactile_zarr):
    """Compute offset (idle per cell+axis) and per-finger global xy/z scales
    from the WHOLE dataset, replacing the SensorNormalizer's shipped per-cell
    calibration with stats derived from this task's actual data.

    Args:
        tactile_zarr: zarr array of shape (N, 2, 9, 3)

    Returns:
        offset    : (2, 9, 3) float32   per-cell median over all frames
        scale_xy  : (2,) float32        per-finger 95th pct of |xy - offset_xy|
        scale_z   : (2,) float32        per-finger 95th pct of |z  - offset_z|
    """
    raw = np.asarray(tactile_zarr[:], dtype=np.float64)  # (N, 2, 9, 3)
    if raw.size == 0:
        return (np.zeros((2, 9, 3), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32))

    # Per-cell, per-axis median is the right "what does this cell read at
    # rest" estimate: it ignores contact events, which are a small fraction
    # of frames in most teleop datasets.
    offset = np.median(raw, axis=0)                       # (2, 9, 3)
    delta = raw - offset                                   # (N, 2, 9, 3)

    scale_xy = np.empty(2, dtype=np.float64)
    scale_z = np.empty(2, dtype=np.float64)
    for fi in range(2):
        xy = np.abs(delta[:, fi, :, :2]).ravel()
        z = np.abs(delta[:, fi, :, 2]).ravel()
        scale_xy[fi] = np.percentile(xy, DATASET_PERCENTILE) if xy.size else 1.0
        scale_z[fi] = np.percentile(z, DATASET_PERCENTILE) if z.size else 1.0
    # Floor so we never divide by ~0 on a dataset where one axis hardly moves.
    scale_xy = np.maximum(scale_xy, 1.0)
    scale_z = np.maximum(scale_z, 1.0)
    return (offset.astype(np.float32),
            scale_xy.astype(np.float32),
            scale_z.astype(np.float32))


def _copy_per_frame_arrays(src_data, dst_data, extra_keys=()):
    """Copy state / action / n_contacts / joint_angles / grip_pos / tactile_*
    (+ any `extra_keys` such as the raw img_{i} arrays) from src to dst verbatim.
    Streams chunk-by-chunk so peak memory stays bounded."""
    pass_through_keys = list(extra_keys) + [
        "state", "action", "n_contacts",
        "joint_angles", "grip_pos",
        "tactile", "tactile_connected", "tactile_ts_ms", "tactile_lag_ms",
    ]
    for key in pass_through_keys:
        if key not in src_data:
            continue
        src_arr = src_data[key]
        dst = dst_data.create_dataset(
            key,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            compressor=src_arr.compressor,
        )
        chunk = src_arr.chunks[0]
        for start in range(0, src_arr.shape[0], chunk):
            end = min(start + chunk, src_arr.shape[0])
            dst[start:end] = src_arr[start:end]


def _create_overlay_arrays(dst_data, num_cams, img_shape, total_n, dtype):
    """One (N, H, W, 3) array per (camera, mode_key) variant."""
    arrays = {}
    img_compressor = zarr.Blosc(cname="zstd", clevel=3,
                                shuffle=zarr.Blosc.BITSHUFFLE)
    for i in range(num_cams):
        for key in MODE_KEYS:
            name = f"img_{i}_{key}"
            arrays[(i, key)] = dst_data.create_dataset(
                name, shape=(total_n, *img_shape),
                chunks=(32, *img_shape), dtype=dtype,
                compressor=img_compressor,
            )
    return arrays


def _to_uint8_bgr(img_float01):
    """float32 [0,1] BGR -> uint8 [0,255] BGR for cv2.draw* primitives."""
    return np.clip(img_float01 * 255.0, 0.0, 255.0).astype(np.uint8)


def _to_float32_01(img_uint8):
    return img_uint8.astype(np.float32) / 255.0


def _img_keys_in_order(src_data):
    """Return ['img_0', 'img_1', ...] in numeric order, skipping any
    suffixed overlay variants. Pure-digit suffix = bare raw frame."""
    keys = []
    for k in src_data.keys():
        if not k.startswith("img_"):
            continue
        rest = k[len("img_"):]
        if rest.isdigit():
            keys.append((int(rest), k))
    keys.sort()
    return [k for _, k in keys]


# By project convention image index 0 is the agent (side) camera and index 1
# is the wrist camera (see collect_with_home._camera_role_wiring + the
# enumeration order in environment/utils.py / cameras.py).
def _role_for_index(cam_idx):
    if cam_idx == 0:
        return "side"
    if cam_idx == 1:
        return "wrist"
    return None  # any extra cameras get a pass-through


def render(src_path, dst_path):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)

    print(f"  Source: {src_path}")
    print(f"  Dest  : {dst_path}  (will be wiped + regenerated)")
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)

    src_root = zarr.open(src_path, mode="r")
    if "data" not in src_root or "state" not in src_root["data"]:
        print("  [error] source zarr has no /data/state — nothing to render.")
        sys.exit(1)
    src_data = src_root["data"]
    total_n = int(src_data["state"].shape[0])

    if total_n == 0:
        print("  [warn] source has zero frames; writing an empty overlay zarr.")
    if "tactile" not in src_data:
        print("  [error] source has no /data/tactile (was --no-tactile passed "
              "at collection time?). Nothing to draw; aborting.")
        sys.exit(1)
    if "joint_angles" not in src_data or "grip_pos" not in src_data:
        print("  [error] source is missing /data/joint_angles or /data/grip_pos.")
        print("          These were added when the sensordrawing overlay landed; ")
        print("          older recordings need to be re-collected to render overlays.")
        sys.exit(1)

    img_keys = _img_keys_in_order(src_data)
    num_cams = len(img_keys)
    if num_cams == 0:
        print("  [error] no /data/img_* arrays in source.")
        sys.exit(1)
    img_shape = tuple(src_data[img_keys[0]].shape[1:])
    img_dtype = src_data[img_keys[0]].dtype
    recorded_wh = (img_shape[1], img_shape[0])  # (W, H)

    print(f"  Frames: {total_n}   cameras: {num_cams}   recorded img_shape: {img_shape}")

    # Build sensordrawing once — eagerly constructs both SensorDrawers and
    # both SensorNormalizers. Cheap; identical state for every frame.
    #
    # We OVERRIDE the SensorNormalizer's per-cell offset + per-axis scale
    # with statistics derived from the entire input dataset (median for
    # offset, 95th percentile of |raw - offset| for scale). This makes the
    # arrows scale meaningfully relative to the contacts THIS task actually
    # contains, instead of relying on (a) a 1.5s-of-idle baseline captured
    # at session start, or (b) shipped hardware calibration that doesn't
    # match an individual sensor unit. Pass offset_override=None at
    # construction time so the shipped values are used as a placeholder,
    # then patch the live attributes below.
    overlay = SensorOverlay(baseline=None)

    src_meta = src_root["meta"]
    print(f"  Computing dataset-wide normalization across {total_n} frames...")
    offset_2x9x3, scale_xy_2, scale_z_2 = _dataset_normalization(src_data["tactile"])
    overlay.norm_L.offset = offset_2x9x3[0]
    overlay.norm_L.global_scale_xy = float(scale_xy_2[0])
    overlay.norm_L.global_scale_z = float(scale_z_2[0])
    overlay.norm_R.offset = offset_2x9x3[1]
    overlay.norm_R.global_scale_xy = float(scale_xy_2[1])
    overlay.norm_R.global_scale_z = float(scale_z_2[1])
    print(f"    LEFT  finger:  scale_xy={scale_xy_2[0]:8.1f}  scale_z={scale_z_2[0]:8.1f}"
          f"  (p{DATASET_PERCENTILE:.0f} of |raw - median|)")
    print(f"    RIGHT finger:  scale_xy={scale_xy_2[1]:8.1f}  scale_z={scale_z_2[1]:8.1f}")

    # ---- Create destination zarr -------------------------------------
    dst_root = zarr.open(dst_path, mode="a")
    dst_data = dst_root.require_group("data")
    dst_meta = dst_root.require_group("meta")

    # Pass-through raw img_{i} arrays so the overlay zarr is a standalone,
    # training-ready dataset: img_{i} = raw frame, img_{i}_{key} = overlaid.
    _copy_per_frame_arrays(src_data, dst_data, extra_keys=img_keys)

    # Pass-through /meta -> /meta verbatim.
    for key in src_root["meta"].keys():
        src_arr = src_root["meta"][key]
        dst = dst_meta.create_dataset(
            key, shape=src_arr.shape, dtype=src_arr.dtype,
        )
        dst[...] = src_arr[...]

    overlay_arrs = _create_overlay_arrays(dst_data, num_cams, img_shape, total_n, img_dtype)

    if total_n == 0:
        print("  Done (0 frames).")
        return

    # ---- Per-frame render --------------------------------------------
    CHUNK = 64
    joint_full = src_data["joint_angles"]    # (N, 7) deg
    grip_full = src_data["grip_pos"]         # (N, 1) raw
    tactile_xyz_full = src_data["tactile"]   # (N, 2, 9, 3)

    t_start = time.time()
    last_log = t_start

    for start in range(0, total_n, CHUNK):
        end = min(start + CHUNK, total_n)
        out_buf = {(i, k): np.empty((end - start, *img_shape), dtype=img_dtype)
                   for i in range(num_cams) for k in MODE_KEYS}

        tac_chunk = tactile_xyz_full[start:end]
        joint_chunk = joint_full[start:end]
        grip_chunk = grip_full[start:end]

        # Normalize tactile once per frame (shared across cameras + modes),
        # then gate per-cell normalized vectors below the noise floor so idle
        # frames don't draw jittery arrows.
        norm_pairs = []
        for j in range(end - start):
            nL, nR = overlay.normalize(tac_chunk[j, 0], tac_chunk[j, 1])
            nL = _apply_deadband(nL, NOISE_DEADBAND)
            nR = _apply_deadband(nR, NOISE_DEADBAND)
            norm_pairs.append((nL, nR))

        for cam_idx in range(num_cams):
            src_imgs = src_data[f"img_{cam_idx}"][start:end]
            role = _role_for_index(cam_idx)

            for j in range(end - start):
                base_224 = _to_uint8_bgr(src_imgs[j])
                base_native = cv2.resize(base_224, (NATIVE_W, NATIVE_H))

                nL, nR = norm_pairs[j]
                angles = joint_chunk[j].tolist()
                grip = float(grip_chunk[j, 0]) if grip_chunk.ndim == 2 else float(grip_chunk[j])

                for mode, is_spatial, scale in MODES:
                    key = mode_key(mode, is_spatial)
                    if role is None:
                        # No role mapped (extra camera) — pass through raw.
                        drawn_native = base_native
                    else:
                        drawn_native = overlay.draw(
                            role, base_native, angles, grip, nL, nR,
                            mode=mode, is_spatial=is_spatial,
                            arrow_length_scale=scale,
                            arrow_thickness=BOLD_ARROW_THICKNESS,
                            dot_size=BOLD_DOT_SIZE,
                        )
                    drawn_recorded = cv2.resize(drawn_native, recorded_wh)
                    out_buf[(cam_idx, key)][j] = _to_float32_01(drawn_recorded)

        for k, arr in out_buf.items():
            overlay_arrs[k][start:end] = arr

        now = time.time()
        if now - last_log >= 1.0:
            elapsed = now - t_start
            fps = end / max(elapsed, 1e-6)
            eta = (total_n - end) / max(fps, 1e-6)
            print(f"  ...{end:>6d}/{total_n}   ({fps:5.1f} fps, eta {eta:5.1f}s)")
            last_log = now

    elapsed = time.time() - t_start
    print(f"  Done. {total_n} frames x {num_cams} cams x {len(MODE_KEYS)} modes "
          f"in {elapsed:.1f}s ({total_n / max(elapsed, 1e-6):.1f} fps).")
    print(f"  Wrote: {dst_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?", default="teleop_data.zarr",
                    help="Source raw zarr (default: teleop_data.zarr)")
    ap.add_argument("dst", nargs="?", default="teleop_data_overlay.zarr",
                    help="Destination overlay zarr (default: teleop_data_overlay.zarr). "
                         "Always wiped + regenerated.")
    args = ap.parse_args()
    render(args.src, args.dst)


if __name__ == "__main__":
    main()
