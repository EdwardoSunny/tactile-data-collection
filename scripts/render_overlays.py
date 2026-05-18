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
    # Plumb the captured per-cell baseline (if present) into the normalizers
    # so post-hoc rendering subtracts the SAME live idle field that the live
    # --viz overlay used at collection time. Otherwise SensorNormalizer falls
    # back to the shipped offsets in calibration_{left,right}.npz, which
    # generally do NOT match an individual hardware unit and bias the arrows.
    src_meta = src_root["meta"]
    baseline = None
    if "tactile_baseline" in src_meta:
        baseline = np.asarray(src_meta["tactile_baseline"][:], dtype=np.float32)
        if baseline.shape != (2, 9, 3):
            print(f"  [warn] /meta/tactile_baseline has unexpected shape {baseline.shape}; "
                  f"ignoring and using shipped offsets")
            baseline = None
        else:
            print(f"  Baseline: using /meta/tactile_baseline (shape {baseline.shape})")
    else:
        print(f"  [warn] no /meta/tactile_baseline in source; "
              f"using shipped offsets (overlay may be biased)")
    overlay = SensorOverlay(baseline=baseline)

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

        # Normalize tactile once per frame (shared across cameras + modes).
        norm_pairs = []
        for j in range(end - start):
            nL, nR = overlay.normalize(tac_chunk[j, 0], tac_chunk[j, 1])
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
