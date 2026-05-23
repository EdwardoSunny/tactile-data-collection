"""
Strip a zarr down to ONLY the canonical raw fields that scripts/render_overlays.py
needs, and write the result to a new path. Handy when a source zarr has
deprecated columns (e.g. old img_*_raw mirror arrays, stale /meta camera
metadata from before the sensordrawing rewrite) and you want a clean copy
that the current renderer will load without complaints.

Required source fields (the renderer hard-errors if any of these are missing):
    /data/state          (N, 7)       float32
    /data/joint_angles   (N, 7)       float32   sensordrawing FK input
    /data/grip_pos       (N, 1)       float32   sensordrawing FK input
    /data/tactile        (N, 2, 9, 3) float32
    /data/img_0..K       (N, H, W, 3) float32   one or more bare img_{i}

Optional source fields (copied through if present, ignored if absent):
    /data/action                                  (only when use_actions=True at collection time)
    /data/n_contacts
    /data/tactile_connected, tactile_ts_ms, tactile_lag_ms
    /meta/episode_ends                            (so renderer can slice episodes; technically
                                                   optional but you almost certainly want it)
    /meta/tactile_baseline                        (preferred — otherwise renderer falls back to
                                                   the shipped per-cell offsets in
                                                   environment/sensordrawing/calibration_*.npz)

Anything NOT in those allowlists is dropped — no img_*_raw, no img_*_<mode>
variants, no legacy /meta/camera_serials / camera_intrinsics_native / trc_agent
(the sensordrawing pipeline bundles its own calibration).

Usage:
    python scripts/extract_raw.py SRC DST

Example (copy cube + tube from ~ into /data/edward as canonical raw zarrs):
    python scripts/extract_raw.py ~/teleop_data_cube.zarr /data/edward/teleop_data_cube.zarr
    python scripts/extract_raw.py ~/teleop_data_tube.zarr /data/edward/teleop_data_tube.zarr

Then render overlays:
    python scripts/render_overlays.py /data/edward/teleop_data_cube.zarr /data/edward/teleop_data_cube_overlay.zarr
"""
import argparse
import os
import re
import shutil
import sys
import time

import zarr


# Fields render_overlays.py needs present (hard-fails if any are missing).
_REQUIRED_DATA_KEYS = ("state", "joint_angles", "grip_pos", "tactile")

# Fields render_overlays.py reads and uses if present, but doesn't require.
_OPTIONAL_DATA_KEYS = (
    "action",
    "n_contacts",
    "tactile_connected", "tactile_ts_ms", "tactile_lag_ms",
)

# /meta keys to copy through. episode_ends is "optional" in the renderer's
# strict sense but you basically always want it. tactile_baseline is also
# optional but strongly preferred — without it the renderer uses generic
# shipped offsets that don't match an individual hardware unit.
_META_KEYS = ("episode_ends", "tactile_baseline")

# Bare-numeric img keys (img_0, img_1, ...). Overlay variants like
# img_0_points9_arrow are NOT in this set and won't be copied.
_BARE_IMG_RE = re.compile(r"^img_(\d+)$")


def _bare_img_keys(group):
    keys = [(int(m.group(1)), k) for k in group.keys()
            if (m := _BARE_IMG_RE.match(k))]
    keys.sort()
    return [k for _, k in keys]


def _stream_copy(src_arr, dst_arr):
    chunk = src_arr.chunks[0] if src_arr.chunks else max(1, src_arr.shape[0])
    n = src_arr.shape[0]
    if n == 0:
        return
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dst_arr[s:e] = src_arr[s:e]


def _create_like(group, name, src_arr):
    """Create dst array mirroring src's shape/chunks/dtype/compressor."""
    return group.create_dataset(
        name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=src_arr.compressor,
    )


def _copy_scalar_like(group, name, src_arr):
    """Same as _create_like but for /meta entries that don't have meaningful
    chunks (small scalars / fixed-size arrays). Avoids a chunk-larger-than-shape
    warning from zarr v2."""
    return group.create_dataset(
        name,
        shape=src_arr.shape,
        dtype=src_arr.dtype,
    )


def extract(src_path, dst_path):
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"  [error] src and dst are the same path; refusing to overwrite in place.")
        sys.exit(1)

    src = zarr.open(src_path, mode="r")
    if "data" not in src:
        print(f"  [error] source has no /data group.")
        sys.exit(1)
    src_data = src["data"]
    src_meta = src["meta"] if "meta" in src else None

    img_keys = _bare_img_keys(src_data)
    if not img_keys:
        print(f"  [error] source has no bare img_{{i}} arrays.")
        sys.exit(1)

    missing = [k for k in _REQUIRED_DATA_KEYS if k not in src_data]
    if missing:
        print(f"  [error] source is missing required field(s): {missing}")
        print(f"          (renderer needs {list(_REQUIRED_DATA_KEYS)} + at least one img_{{i}})")
        sys.exit(1)

    total_n = int(src_data["state"].shape[0])
    print(f"  Source: {src_path}")
    print(f"  Dest  : {dst_path}  (will be wiped + regenerated)")
    print(f"  Frames: {total_n}   cameras: {len(img_keys)} ({img_keys})")
    print()

    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)

    dst = zarr.open(dst_path, mode="a")
    dst_data = dst.require_group("data")
    dst_meta = dst.require_group("meta")

    t0 = time.time()

    # /data — required fields (we already validated they exist).
    for key in _REQUIRED_DATA_KEYS:
        src_arr = src_data[key]
        print(f"  [data] copy  {key:20s}  shape={src_arr.shape}  dtype={src_arr.dtype}")
        dst_arr = _create_like(dst_data, key, src_arr)
        _stream_copy(src_arr, dst_arr)

    # /data — optional fields, copied if present.
    for key in _OPTIONAL_DATA_KEYS:
        if key not in src_data:
            continue
        src_arr = src_data[key]
        print(f"  [data] copy  {key:20s}  shape={src_arr.shape}  dtype={src_arr.dtype}")
        dst_arr = _create_like(dst_data, key, src_arr)
        _stream_copy(src_arr, dst_arr)

    # /data — bare image arrays only.
    for key in img_keys:
        src_arr = src_data[key]
        print(f"  [data] copy  {key:20s}  shape={src_arr.shape}  dtype={src_arr.dtype}")
        dst_arr = _create_like(dst_data, key, src_arr)
        _stream_copy(src_arr, dst_arr)

    # /meta — allowlisted keys only.
    if src_meta is not None:
        for key in _META_KEYS:
            if key not in src_meta:
                continue
            src_arr = src_meta[key]
            print(f"  [meta] copy  {key:20s}  shape={src_arr.shape}  dtype={src_arr.dtype}")
            dst_arr = _copy_scalar_like(dst_meta, key, src_arr)
            dst_arr[...] = src_arr[...]
    else:
        print(f"  [warn] source has no /meta group — episode_ends + tactile_baseline both missing.")

    # What did we drop?
    dropped_data = [k for k in src_data.keys()
                    if k not in _REQUIRED_DATA_KEYS
                    and k not in _OPTIONAL_DATA_KEYS
                    and not _BARE_IMG_RE.match(k)]
    dropped_meta = (
        [k for k in src_meta.keys() if k not in _META_KEYS]
        if src_meta is not None else []
    )
    if dropped_data or dropped_meta:
        print()
        if dropped_data:
            print(f"  [note] dropped /data fields: {sorted(dropped_data)}")
        if dropped_meta:
            print(f"  [note] dropped /meta fields: {sorted(dropped_meta)}")

    print()
    print(f"  Done in {time.time() - t0:.1f}s.  Wrote {dst_path}.")
    print(f"  Next:  python scripts/render_overlays.py {dst_path} <dst>_overlay.zarr")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Source zarr (possibly with extra/legacy fields)")
    ap.add_argument("dst", help="Destination canonical-raw zarr. Wiped if exists.")
    args = ap.parse_args()
    extract(args.src, args.dst)


if __name__ == "__main__":
    main()
