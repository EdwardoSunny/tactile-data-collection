"""
Render the points9_arrow_len0 overlay variant: the standard points9_arrow
rendering (9 projected pad dots per finger) but with arrow_length_scale=0.0,
so every force arrow has zero length. The result carries the same spatial
contact-position cue as points9_arrow while carrying NO force information —
the ablation control for "is it the force vectors or just the dot positions
that help".

Reads any zarr that has the raw per-frame fields (a raw teleop_data zarr OR
an overlay zarr — both contain them):

    /data/img_{i}        (N, 224, 224, 3) float32 [0,1]   un-overlaid frames
    /data/joint_angles   (N, 7)  deg
    /data/grip_pos       (N, 1)  raw 0..850
    /data/tactile        (N, 2, 9, 3) raw counts

Writes /data/img_{i}_points9_arrow_len0 arrays. Two output modes:

    # fresh standalone dst (copies raw fields + meta through, like render_overlays.py)
    python scripts/render_arrowlen0.py SRC.zarr DST.zarr

    # append into an existing zarr (e.g. straight into the overlay zarr)
    python scripts/render_arrowlen0.py SRC.zarr --in-place

Normalization + deadband mirror scripts/render_overlays.py exactly, though
with zero-length arrows they only affect the (invisible) arrow geometry —
dots are drawn at FK-projected pad positions regardless of force.
"""
import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np
import zarr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.tactile_overlay import SensorOverlay

MODE_KEY = "points9_arrow_len0"
MODE = "points9_arrow"
IS_SPATIAL = True
ARROW_LENGTH_SCALE = 0.0  # <- the whole point of this variant

NATIVE_W, NATIVE_H = 640, 480
BOLD_ARROW_THICKNESS = 8
BOLD_DOT_SIZE = 22
DATASET_PERCENTILE = 99.0
NOISE_DEADBAND = 0.12


def _apply_deadband(norm_arr, threshold):
    if norm_arr is None or threshold <= 0:
        return norm_arr
    arr = np.asarray(norm_arr, dtype=np.float32).copy()
    mag = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.where(mag < threshold, 0.0, arr)


def _dataset_normalization(tactile_zarr):
    raw = np.asarray(tactile_zarr[:], dtype=np.float64)  # (N, 2, 9, 3)
    if raw.size == 0:
        return (np.zeros((2, 9, 3), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32))
    offset = np.median(raw, axis=0)
    delta = raw - offset
    scale_xy = np.empty(2, dtype=np.float64)
    scale_z = np.empty(2, dtype=np.float64)
    for fi in range(2):
        xy = np.abs(delta[:, fi, :, :2]).ravel()
        z = np.abs(delta[:, fi, :, 2]).ravel()
        scale_xy[fi] = np.percentile(xy, DATASET_PERCENTILE) if xy.size else 1.0
        scale_z[fi] = np.percentile(z, DATASET_PERCENTILE) if z.size else 1.0
    scale_xy = np.maximum(scale_xy, 1.0)
    scale_z = np.maximum(scale_z, 1.0)
    return (offset.astype(np.float32),
            scale_xy.astype(np.float32),
            scale_z.astype(np.float32))


def _copy_per_frame_arrays(src_data, dst_data, extra_keys=()):
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


def _copy_meta(src, dst):
    for key in src.keys():
        item = src[key]
        if hasattr(item, "shape"):
            d = dst.create_dataset(key, shape=item.shape, dtype=item.dtype)
            d[...] = item[...]
        else:
            _copy_meta(item, dst.require_group(key))


def _to_uint8_bgr(img_float01):
    return np.clip(img_float01 * 255.0, 0.0, 255.0).astype(np.uint8)


def _to_float32_01(img_uint8):
    return img_uint8.astype(np.float32) / 255.0


def _img_keys_in_order(src_data):
    keys = []
    for k in src_data.keys():
        if not k.startswith("img_"):
            continue
        rest = k[len("img_"):]
        if rest.isdigit():
            keys.append((int(rest), k))
    keys.sort()
    return [k for _, k in keys]


def _role_for_index(cam_idx):
    if cam_idx == 0:
        return "side"
    if cam_idx == 1:
        return "wrist"
    return None


def render(src_path, dst_path, in_place):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)

    src_root = zarr.open(src_path, mode="r+" if in_place else "r")
    if "data" not in src_root or "state" not in src_root["data"]:
        print("  [error] source zarr has no /data/state.")
        sys.exit(1)
    src_data = src_root["data"]
    for req in ("tactile", "joint_angles", "grip_pos"):
        if req not in src_data:
            print(f"  [error] source is missing /data/{req} — cannot render.")
            sys.exit(1)

    total_n = int(src_data["state"].shape[0])
    img_keys = _img_keys_in_order(src_data)
    if not img_keys:
        print("  [error] no /data/img_* arrays in source.")
        sys.exit(1)
    num_cams = len(img_keys)
    img_shape = tuple(src_data[img_keys[0]].shape[1:])
    img_dtype = src_data[img_keys[0]].dtype
    recorded_wh = (img_shape[1], img_shape[0])

    print(f"  Source: {src_path}")
    print(f"  Frames: {total_n}   cameras: {num_cams}   img_shape: {img_shape}")
    print(f"  Mode  : {MODE_KEY} (mode={MODE}, arrow_length_scale={ARROW_LENGTH_SCALE})")

    overlay = SensorOverlay(baseline=None)
    print(f"  Computing dataset-wide normalization across {total_n} frames...")
    offset_2x9x3, scale_xy_2, scale_z_2 = _dataset_normalization(src_data["tactile"])
    overlay.norm_L.offset = offset_2x9x3[0]
    overlay.norm_L.global_scale_xy = float(scale_xy_2[0])
    overlay.norm_L.global_scale_z = float(scale_z_2[0])
    overlay.norm_R.offset = offset_2x9x3[1]
    overlay.norm_R.global_scale_xy = float(scale_xy_2[1])
    overlay.norm_R.global_scale_z = float(scale_z_2[1])

    if in_place:
        dst_data = zarr.open(src_path, mode="r+")["data"]
        print(f"  Dest  : {src_path} (in-place append)")
    else:
        print(f"  Dest  : {dst_path} (wiped + regenerated)")
        if os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        dst_root = zarr.open(dst_path, mode="a")
        dst_data = dst_root.require_group("data")
        _copy_per_frame_arrays(src_data, dst_data, extra_keys=img_keys)
        if "meta" in src_root:
            _copy_meta(src_root["meta"], dst_root.require_group("meta"))

    img_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    out_arrs = {}
    for i in range(num_cams):
        name = f"img_{i}_{MODE_KEY}"
        if name in dst_data:
            del dst_data[name]
        out_arrs[i] = dst_data.create_dataset(
            name, shape=(total_n, *img_shape),
            chunks=(32, *img_shape), dtype=img_dtype,
            compressor=img_compressor,
        )

    if total_n == 0:
        print("  Done (0 frames).")
        return

    CHUNK = 64
    joint_full = src_data["joint_angles"]
    grip_full = src_data["grip_pos"]
    tactile_full = src_data["tactile"]

    t_start = time.time()
    last_log = t_start
    for start in range(0, total_n, CHUNK):
        end = min(start + CHUNK, total_n)
        tac_chunk = tactile_full[start:end]
        joint_chunk = joint_full[start:end]
        grip_chunk = grip_full[start:end]

        norm_pairs = []
        for j in range(end - start):
            nL, nR = overlay.normalize(tac_chunk[j, 0], tac_chunk[j, 1])
            nL = _apply_deadband(nL, NOISE_DEADBAND)
            nR = _apply_deadband(nR, NOISE_DEADBAND)
            norm_pairs.append((nL, nR))

        for cam_idx in range(num_cams):
            src_imgs = src_data[f"img_{cam_idx}"][start:end]
            role = _role_for_index(cam_idx)
            buf = np.empty((end - start, *img_shape), dtype=img_dtype)
            for j in range(end - start):
                base_224 = _to_uint8_bgr(src_imgs[j])
                base_native = cv2.resize(base_224, (NATIVE_W, NATIVE_H))
                if role is None:
                    drawn_native = base_native
                else:
                    nL, nR = norm_pairs[j]
                    angles = joint_chunk[j].tolist()
                    grip = float(grip_chunk[j, 0]) if grip_chunk.ndim == 2 else float(grip_chunk[j])
                    drawn_native = overlay.draw(
                        role, base_native, angles, grip, nL, nR,
                        mode=MODE, is_spatial=IS_SPATIAL,
                        arrow_length_scale=ARROW_LENGTH_SCALE,
                        arrow_thickness=BOLD_ARROW_THICKNESS,
                        dot_size=BOLD_DOT_SIZE,
                    )
                buf[j] = _to_float32_01(cv2.resize(drawn_native, recorded_wh))
            out_arrs[cam_idx][start:end] = buf

        now = time.time()
        if now - last_log >= 1.0:
            fps = end / max(now - t_start, 1e-6)
            eta = (total_n - end) / max(fps, 1e-6)
            print(f"  ...{end:>6d}/{total_n}   ({fps:5.1f} fps, eta {eta:5.1f}s)")
            last_log = now

    elapsed = time.time() - t_start
    print(f"  Done. {total_n} frames x {num_cams} cams in {elapsed:.1f}s.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Source zarr (raw teleop_data or overlay zarr)")
    ap.add_argument("dst", nargs="?", default=None,
                    help="Destination zarr (wiped + regenerated). Omit with --in-place.")
    ap.add_argument("--in-place", action="store_true",
                    help="Append img_{i}_points9_arrow_len0 into SRC instead of writing DST.")
    args = ap.parse_args()
    if not args.in_place and args.dst is None:
        ap.error("provide DST or pass --in-place")
    render(args.src, args.dst, args.in_place)


if __name__ == "__main__":
    main()
