"""
Throwaway preview: dump one episode of overlay-rendered frames from a
teleop_data_*_overlay.zarr to an MP4 so you can eyeball whether the arrows
look reasonable on real training data.

Frames are already overlay-burned by render_overlays.py — this script just
slices on /meta/episode_ends, converts float32 [0,1] BGR to uint8 BGR, and
writes them out at the original recording rate. No re-normalization, no
re-rendering; what you see is exactly what training will see.

Default behavior: episode 0 of teleop_data_cube_overlay.zarr, points9_arrow
mode, both cameras (agent + wrist) side-by-side, 10 fps -> preview.mp4

Usage:
    python scripts/episode_to_mp4.py
    python scripts/episode_to_mp4.py /data/edward/teleop_data_tube_overlay.zarr --episode 3
    python scripts/episode_to_mp4.py ... --mode points1_arrow --out tube_ep3.mp4
    python scripts/episode_to_mp4.py ... --camera 0     # agent only
    python scripts/episode_to_mp4.py ... --camera 1     # wrist only
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np
import zarr


_VALID_MODES = [
    "raw",
    "points9_arrow", "points1_arrow",
    "points1_contact_spatial", "points9_color_spatial",
    "points1_contact_flat", "points9_color_flat",
]


def _img_key(cam_idx: int, mode: str) -> str:
    return f"img_{cam_idx}" if mode == "raw" else f"img_{cam_idx}_{mode}"


def _bare_cams(group):
    pat = re.compile(r"^img_(\d+)$")
    return sorted(int(pat.match(k).group(1)) for k in group.keys() if pat.match(k))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?", default="/data/edward/teleop_data_cube_overlay.zarr",
                    help="Overlay zarr (default: cube)")
    ap.add_argument("--episode", type=int, default=0,
                    help="0-indexed episode number (default 0)")
    ap.add_argument("--mode", default="points9_arrow", choices=_VALID_MODES,
                    help="Overlay variant to read (default points9_arrow)")
    ap.add_argument("--camera", type=int, default=None,
                    help="Camera index. Omit to show all cams side-by-side.")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Playback fps (default 10, matches recording rate)")
    ap.add_argument("--out", default="preview.mp4",
                    help="Output mp4 path (default preview.mp4)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"  [error] not a zarr directory: {args.src}")
        sys.exit(1)

    root = zarr.open(args.src, mode="r")
    data = root["data"]
    meta = root["meta"]

    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if args.episode < 0 or args.episode >= len(ends):
        print(f"  [error] episode {args.episode} out of range; have {len(ends)} episodes (0..{len(ends)-1}).")
        sys.exit(1)
    start = int(ends[args.episode - 1]) if args.episode > 0 else 0
    end = int(ends[args.episode])
    n = end - start

    cam_idxs = _bare_cams(data) if args.camera is None else [args.camera]
    # Sanity-check that the requested mode exists for each cam.
    for c in cam_idxs:
        key = _img_key(c, args.mode)
        if key not in data:
            print(f"  [error] {key} not in source zarr.")
            print(f"          Available image keys: {sorted(k for k in data.keys() if k.startswith('img_'))}")
            sys.exit(1)

    print(f"  Source : {args.src}")
    print(f"  Episode: {args.episode}  ({start}..{end-1}, {n} frames)")
    print(f"  Mode   : {args.mode}")
    print(f"  Cams   : {cam_idxs}")
    print(f"  Output : {args.out}  @ {args.fps:.1f} fps")

    # Probe the frame shape from the first camera's array.
    first_key = _img_key(cam_idxs[0], args.mode)
    h, w = data[first_key].shape[1:3]
    out_w = w * len(cam_idxs)  # side-by-side
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"  [error] cv2.VideoWriter failed to open {args.out}")
        sys.exit(1)

    # Stream frame-by-frame so we don't load the whole episode into RAM.
    CHUNK = 32
    for s in range(start, end, CHUNK):
        e = min(s + CHUNK, end)
        cam_chunks = [np.asarray(data[_img_key(c, args.mode)][s:e]) for c in cam_idxs]
        for j in range(e - s):
            tiles = [np.clip(cc[j] * 255.0, 0, 255).astype(np.uint8) for cc in cam_chunks]
            frame = tiles[0] if len(tiles) == 1 else np.concatenate(tiles, axis=1)
            writer.write(frame)

    writer.release()
    print(f"  Done. Wrote {n} frames to {args.out}.")


if __name__ == "__main__":
    main()
