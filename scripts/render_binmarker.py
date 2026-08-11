"""Render the binmarker overlay variant.

binmarker = binary contact indicator drawn as a FIXED-LENGTH arrow, ported
from LIBERO_contact_overlay's `binbars` (binary per-finger contact gating)
+ `arrow` (spatial per-finger force-direction arrow) styles:

  - one arrow per finger, anchored at the finger center pad (points1_arrow
    anchor), pointing in the aggregate force direction
  - length FIXED at binmarker_common.FIXED_ARROW_LEN whenever visible
  - visible iff the finger's contact state is ON per the tuned hysteresis
    detector in binmarker_common (max-cell metric, T_ON/T_OFF + debounce,
    validated visually on 5 episodes per task — see tune_binmarker.py)
  - center dots always drawn (consistent with the arrow-variant family)

Usage mirrors render_arrowlen0.py:
    python scripts/render_binmarker.py SRC.zarr DST.zarr
    python scripts/render_binmarker.py SRC.zarr --in-place
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.tactile_overlay import SensorOverlay
from binmarker_common import (FIXED_ARROW_LEN, idle_window_normalization,
                              normalize_all, binmarker_feed, contact_metric,
                              contact_states)

MODE_KEY = "binmarker"
NATIVE_W, NATIVE_H = 640, 480
BOLD_ARROW_THICKNESS = 8
BOLD_DOT_SIZE = 22


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
            key, shape=src_arr.shape, chunks=src_arr.chunks,
            dtype=src_arr.dtype, compressor=src_arr.compressor)
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


def _img_keys_in_order(src_data):
    keys = []
    for k in src_data.keys():
        if k.startswith("img_") and k[len("img_"):].isdigit():
            keys.append((int(k[len("img_"):]), k))
    keys.sort()
    return [k for _, k in keys]


def _role_for_index(cam_idx):
    return {0: "side", 1: "wrist"}.get(cam_idx)


def render(src_path, dst_path, in_place):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)
    src_root = zarr.open(src_path, mode="r")
    src_data = src_root["data"]
    for req in ("tactile", "joint_angles", "grip_pos", "state"):
        if req not in src_data:
            print(f"  [error] source missing /data/{req}")
            sys.exit(1)

    total_n = int(src_data["state"].shape[0])
    img_keys = _img_keys_in_order(src_data)
    num_cams = len(img_keys)
    img_shape = tuple(src_data[img_keys[0]].shape[1:])
    img_dtype = src_data[img_keys[0]].dtype
    recorded_wh = (img_shape[1], img_shape[0])

    print(f"  Source: {src_path}")
    print(f"  Frames: {total_n}   cameras: {num_cams}   mode: {MODE_KEY}")

    # ---- contact detection over the whole dataset ----
    raw = np.asarray(src_data["tactile"][:], dtype=np.float64)
    ends = np.asarray(src_root["meta/episode_ends"][:])
    starts = np.r_[0, ends[:-1]]
    offset, sxy, sz = idle_window_normalization(raw, ends)
    norm = normalize_all(raw, offset, sxy, sz).astype(np.float32)
    metric = contact_metric(norm)
    on = np.zeros_like(metric, dtype=bool)
    for s, e in zip(starts, ends):
        for f in range(2):
            on[s:e, f] = contact_states(metric[s:e, f])
    print(f"  Contact ON: left {100*on[:,0].mean():.1f}%  right {100*on[:,1].mean():.1f}% of frames")

    overlay = SensorOverlay(baseline=None)
    for n_obj, fi in ((overlay.norm_L, 0), (overlay.norm_R, 1)):
        n_obj.offset = offset[fi].astype(np.float32)
        n_obj.global_scale_xy = float(sxy[fi])
        n_obj.global_scale_z = float(sz[fi])

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
            name, shape=(total_n, *img_shape), chunks=(32, *img_shape),
            dtype=img_dtype, compressor=img_compressor)

    CHUNK = 64
    joint_full = src_data["joint_angles"]
    grip_full = src_data["grip_pos"]
    t_start = time.time()
    last_log = t_start
    for start in range(0, total_n, CHUNK):
        end = min(start + CHUNK, total_n)
        joint_chunk = joint_full[start:end]
        grip_chunk = grip_full[start:end]
        feeds = [(binmarker_feed(norm[g, 0], on[g, 0]),
                  binmarker_feed(norm[g, 1], on[g, 1])) for g in range(start, end)]

        for cam_idx in range(num_cams):
            src_imgs = src_data[f"img_{cam_idx}"][start:end]
            role = _role_for_index(cam_idx)
            buf = np.empty((end - start, *img_shape), dtype=img_dtype)
            for j in range(end - start):
                base_native = cv2.resize(_to_uint8_bgr(src_imgs[j]), (NATIVE_W, NATIVE_H))
                if role is None:
                    drawn = base_native
                else:
                    fL, fR = feeds[j]
                    grip = float(np.ravel(grip_chunk[j])[0])
                    drawn = overlay.draw(
                        role, base_native, joint_chunk[j].tolist(), grip,
                        fL, fR, mode="points1_arrow", is_spatial=True,
                        arrow_length_scale=FIXED_ARROW_LEN,
                        arrow_thickness=BOLD_ARROW_THICKNESS, dot_size=BOLD_DOT_SIZE)
                buf[j] = cv2.resize(drawn, recorded_wh).astype(np.float32) / 255.0
            out_arrs[cam_idx][start:end] = buf

        now = time.time()
        if now - last_log >= 5.0:
            fps = end / max(now - t_start, 1e-6)
            print(f"  ...{end:>6d}/{total_n}  ({fps:5.1f} fps, eta {(total_n-end)/max(fps,1e-6):5.1f}s)", flush=True)
            last_log = now

    print(f"  Done. {total_n} frames x {num_cams} cams in {time.time()-t_start:.1f}s.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()
    if not args.in_place and args.dst is None:
        ap.error("provide DST or pass --in-place")
    render(args.src, args.dst, args.in_place)


if __name__ == "__main__":
    main()
