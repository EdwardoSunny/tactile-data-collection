"""
Render one full episode to an MP4 for arrow-tuning before doing a full
overlay render of a dataset.

Uses the SAME normalization pipeline as scripts/render_overlays.py:
per-episode offset (mean of first N_BASELINE_FRAMES frames) + per-finger
global percentile scale across the whole input zarr. So the previewed
arrows match what the full render will produce, modulo any (mode_key,
arrow_length_scale, arrow_thickness, dot_size, noise_deadband) overrides
you pass on the command line.

The MP4 stacks img_0 (side / agent cam) and img_1 (wrist cam) horizontally;
both have the chosen overlay variant drawn on them.

    python scripts/render_overlay_preview.py teleop_data_cube.zarr
    python scripts/render_overlay_preview.py teleop_data_dishwasher.zarr \
        --episode 50 --mode points1_arrow --arrow-length-scale 0.04 \
        --arrow-thickness 10 --dot-size 26 --out preview_dish.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np
import zarr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.tactile_overlay import (
    BOLD_ARROW_THICKNESS, BOLD_DOT_SIZE, MODES, MODE_KEYS,
    NOISE_DEADBAND, SensorOverlay, apply_deadband, mode_key,
)
from scripts.render_overlays import (
    NATIVE_W, NATIVE_H, N_BASELINE_FRAMES, DATASET_PERCENTILE,
    _per_episode_offsets, _global_scales_post_offset,
    _frame_to_episode_index, _load_or_compute_normalization,
)


def _to_uint8_bgr(img_float01):
    return np.clip(img_float01 * 255.0, 0.0, 255.0).astype(np.uint8)


def render_preview(src_path, episode_idx, mode_str, is_spatial,
                   arrow_length_scale, arrow_thickness, dot_size,
                   noise_deadband, fps, out_path):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)

    src = zarr.open(src_path, mode="r")
    data = src["data"]
    meta = src["meta"]
    for required in ("state", "tactile", "joint_angles", "grip_pos", "img_0", "img_1"):
        if required not in data:
            print(f"  [error] {src_path} missing /data/{required}")
            sys.exit(1)
    if "episode_ends" not in meta:
        print(f"  [error] {src_path} missing /meta/episode_ends")
        sys.exit(1)

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    E = len(episode_ends)
    if episode_idx < 0 or episode_idx >= E:
        print(f"  [error] episode {episode_idx} out of range (have {E} episodes)")
        sys.exit(1)

    ep_start = int(episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
    ep_end = int(episode_ends[episode_idx])
    n = ep_end - ep_start
    if n == 0:
        print(f"  [error] episode {episode_idx} is empty")
        sys.exit(1)

    print(f"  Source: {src_path}")
    print(f"  Episode {episode_idx}: frames [{ep_start}, {ep_end}) = {n} frames")
    print(f"  Mode: {mode_str} (is_spatial={is_spatial})")
    print(f"  arrow_length_scale={arrow_length_scale}  arrow_thickness={arrow_thickness}"
          f"  dot_size={dot_size}  noise_deadband={noise_deadband}")
    print()

    # Normalization setup — identical to render_overlays.py. Will read from
    # /meta/normalization if present, otherwise fall back to per-task derive.
    total_n = data["state"].shape[0]
    raw_clip_low, raw_clip_high, ep_offsets, scale_xy_2, scale_z_2, deadband = \
        _load_or_compute_normalization(src, data, episode_ends, total_n)
    frame_to_ep = _frame_to_episode_index(episode_ends)

    overlay = SensorOverlay(baseline=None)
    overlay.norm_L.offset = np.zeros((9, 3), dtype=np.float32)
    overlay.norm_L.global_scale_xy = float(scale_xy_2[0])
    overlay.norm_L.global_scale_z = float(scale_z_2[0])
    overlay.norm_R.offset = np.zeros((9, 3), dtype=np.float32)
    overlay.norm_R.global_scale_xy = float(scale_xy_2[1])
    overlay.norm_R.global_scale_z = float(scale_z_2[1])
    print(f"    LEFT  finger:  scale_xy={scale_xy_2[0]:8.1f}  scale_z={scale_z_2[0]:8.1f}")
    print(f"    RIGHT finger:  scale_xy={scale_xy_2[1]:8.1f}  scale_z={scale_z_2[1]:8.1f}")
    # Per-CLI override of the deadband (the --noise-deadband flag) wins over
    # whatever was loaded; otherwise we use the loaded/legacy value.
    if noise_deadband is None:
        noise_deadband = deadband
    print(f"    deadband:      {noise_deadband:.4f}")
    print()

    img0 = data["img_0"][ep_start:ep_end]
    img1 = data["img_1"][ep_start:ep_end]
    joint = data["joint_angles"][ep_start:ep_end]
    grip = data["grip_pos"][ep_start:ep_end]
    tac = data["tactile"][ep_start:ep_end]

    # MP4 writer dimensions: side-by-side at native 640x480.
    H_out, W_out = NATIVE_H, NATIVE_W * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_out, H_out))
    if not writer.isOpened():
        print(f"  [error] cv2.VideoWriter could not open {out_path}")
        sys.exit(1)

    print(f"  Writing {n} frames to {out_path} ...")
    for j in range(n):
        ep_i = int(frame_to_ep[ep_start + j])
        raw_L = np.asarray(tac[j, 0], dtype=np.float64)
        raw_R = np.asarray(tac[j, 1], dtype=np.float64)
        if raw_clip_low is not None:
            raw_L = np.clip(raw_L, raw_clip_low[0], raw_clip_high[0])
            raw_R = np.clip(raw_R, raw_clip_low[1], raw_clip_high[1])
        centered_L = raw_L - ep_offsets[ep_i, 0]
        centered_R = raw_R - ep_offsets[ep_i, 1]
        nL, nR = overlay.normalize(centered_L, centered_R)
        nL = apply_deadband(nL, noise_deadband)
        nR = apply_deadband(nR, noise_deadband)

        angles = joint[j].tolist()
        gp = float(grip[j, 0]) if grip.ndim == 2 else float(grip[j])

        side_img = cv2.resize(_to_uint8_bgr(img0[j]), (NATIVE_W, NATIVE_H))
        wrist_img = cv2.resize(_to_uint8_bgr(img1[j]), (NATIVE_W, NATIVE_H))

        side_drawn = overlay.draw(
            "side", side_img, angles, gp, nL, nR,
            mode=mode_str, is_spatial=is_spatial,
            arrow_length_scale=arrow_length_scale,
            arrow_thickness=arrow_thickness,
            dot_size=dot_size,
        )
        wrist_drawn = overlay.draw(
            "wrist", wrist_img, angles, gp, nL, nR,
            mode=mode_str, is_spatial=is_spatial,
            arrow_length_scale=arrow_length_scale,
            arrow_thickness=arrow_thickness,
            dot_size=dot_size,
        )

        combined = np.concatenate([side_drawn, wrist_drawn], axis=1)
        writer.write(combined)

    writer.release()
    print(f"  Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Source raw zarr (e.g. teleop_data_cube.zarr)")
    ap.add_argument("--episode", type=int, default=None,
                    help="Episode index to preview (default: middle of dataset)")
    ap.add_argument("--mode", choices=MODE_KEYS, default="points9_arrow",
                    help="Overlay variant (default: points9_arrow)")
    ap.add_argument("--arrow-length-scale", type=float, default=None,
                    help="Override per-mode default arrow length")
    ap.add_argument("--arrow-thickness", type=int, default=BOLD_ARROW_THICKNESS,
                    help=f"Arrow line thickness in px (default: {BOLD_ARROW_THICKNESS})")
    ap.add_argument("--dot-size", type=int, default=BOLD_DOT_SIZE,
                    help=f"Sensor dot size in px (default: {BOLD_DOT_SIZE})")
    ap.add_argument("--noise-deadband", type=float, default=None,
                    help=f"Per-cell L2 noise gate (default: loaded from "
                         f"/meta/normalization, else {NOISE_DEADBAND})")
    ap.add_argument("--fps", type=int, default=10,
                    help="MP4 framerate (default: 10, matches recording rate)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output MP4 path (default: preview_<basename>_<mode>_ep<N>.mp4)")
    args = ap.parse_args()

    # Resolve mode -> (mode_str, is_spatial, default_scale)
    for m, s, scale in MODES:
        if mode_key(m, s) == args.mode:
            mode_str, is_spatial, default_scale = m, s, scale
            break
    arrow_length_scale = args.arrow_length_scale if args.arrow_length_scale is not None else default_scale

    # Default episode: middle of the dataset.
    src = zarr.open(args.src, mode="r")
    E = len(src["meta/episode_ends"][:])
    episode_idx = args.episode if args.episode is not None else E // 2

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.src.rstrip("/")))[0]
        out_path = f"preview_{base}_{args.mode}_ep{episode_idx}.mp4"
    else:
        out_path = args.out

    render_preview(args.src, episode_idx, mode_str, is_spatial,
                   arrow_length_scale, args.arrow_thickness, args.dot_size,
                   args.noise_deadband, args.fps, out_path)


if __name__ == "__main__":
    main()
