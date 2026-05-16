"""
One-shot schema converter: old-format teleop zarr -> new raw-only schema.

Old schema (pre-2026-05-15 refactor):
  /data/img_{i}      = camera frame WITH overlay burned in (224x224, float32)
  /data/img_{i}_raw  = camera frame WITHOUT overlay (224x224, float32)
  /data/state, n_contacts, tactile_*, action(optional)
  /meta/episode_ends, tactile_baseline

New schema (this repo):
  /data/img_{i}      = raw 224x224 float32 frame (no overlay)
  /data/state, n_contacts, tactile_*, action(optional)   <- unchanged
  /meta/episode_ends, tactile_baseline                   <- unchanged
  /meta/camera_serials, camera_intrinsics_native, camera_native_size,
        agent_camera_serial, wrist_camera_serial,
        trc_agent, recorded_img_size                     <- NEW

Conversion:
  1. img_{i}_raw -> img_{i}   (the old _raw frames become the new raw)
  2. drop old img_{i} (overlay-burned; not needed in new schema)
  3. copy state, n_contacts, tactile_*, action, episode_ends, tactile_baseline verbatim
  4. invent the new /meta camera fields from tactile_config.py + transforms_agent.npz
     and (since pyrealsense2 isn't available offline) reasonable D435 color-stream
     intrinsics defaults. Wrist overlay uses fixed pixel anchors so it does not
     depend on intrinsics. Agent overlay DOES depend on intrinsics; the defaults
     here will project arrows in roughly the right place but the exact pixels
     may differ slightly from the original recording.

Usage:
    python scripts/convert_old_to_new_zarr.py SRC DST

After conversion, render overlays:
    python scripts/render_overlays.py DST DST_overlay.zarr
"""
import argparse
import os
import shutil
import sys

import numpy as np
import zarr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tactile_config as tc


# RealSense D435 color-stream native (640x480 BGR8) intrinsics, typical
# factory-calibrated values. Used as a placeholder because pyrealsense2 isn't
# importable in this offline environment. Per-unit calibration varies by a few
# pixels; if you want exact agent-overlay reproduction, re-run conversion on a
# machine with the live cameras attached so the actual intrinsics get saved.
_D435_DEFAULT_INTRINSICS_640x480 = {
    "fx":  617.0,
    "fy":  617.0,
    "ppx": 320.0,
    "ppy": 240.0,
}
_DEFAULT_NATIVE_W = 640
_DEFAULT_NATIVE_H = 480


def _load_trc_agent(npz_path):
    """Return (agent_serial, trc 3x4) from transforms_agent.npz.
    Falls back to (None, None) on any error so the conversion proceeds and
    the agent overlay just passes raw at render time."""
    if not os.path.isfile(npz_path):
        print(f"  [warn] transforms file not found: {npz_path}")
        return None, None
    try:
        d = np.load(npz_path, allow_pickle=False)
        return str(d["serial"]), np.asarray(d["trc"], dtype=np.float64)
    except Exception as e:
        print(f"  [warn] couldn't read {npz_path}: {e}")
        return None, None


def _stream_copy(src_arr, dst_arr, chunk=None):
    """Copy a zarr array start-to-end in chunks so peak RAM stays bounded."""
    if chunk is None:
        chunk = src_arr.chunks[0]
    n = src_arr.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dst_arr[start:end] = src_arr[start:end]


def _create_like(group, name, src_arr):
    return group.create_dataset(
        name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=src_arr.compressor,
    )


def convert(src_path, dst_path):
    if not os.path.isdir(src_path):
        print(f"  [error] source not found: {src_path}")
        sys.exit(1)
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print(f"  [error] src and dst are the same path; refusing to overwrite in place.")
        sys.exit(1)

    src = zarr.open(src_path, mode="r")
    if "data" not in src or "state" not in src["data"]:
        print(f"  [error] source has no /data/state.")
        sys.exit(1)

    # Detect old schema: presence of /data/img_{i}_raw means img_{i} is the
    # overlay-burned variant and img_{i}_raw is what we want as the new raw.
    src_data = src["data"]
    src_meta = src["meta"]
    raw_img_keys = sorted(k for k in src_data.keys() if k.endswith("_raw")
                          and k[:-len("_raw")].startswith("img_"))
    if not raw_img_keys:
        print(f"  [info] source has no img_*_raw arrays — assuming img_* IS already "
              f"raw (already in new schema). Will copy through unchanged and only "
              f"add the new /meta camera fields if missing.")
    # Decide num_cameras and the mapping (old-name -> new-name):
    img_remap = {}  # source_key -> dest_key
    if raw_img_keys:
        for k in raw_img_keys:
            new_key = k[:-len("_raw")]
            img_remap[k] = new_key  # img_0_raw -> img_0
    else:
        for k in src_data.keys():
            if k.startswith("img_") and k[len("img_"):].isdigit():
                img_remap[k] = k

    num_cameras = len(img_remap)
    print(f"  Source: {src_path}")
    print(f"  Dest  : {dst_path}")
    print(f"  Frames in source: {int(src_data['state'].shape[0])}")
    print(f"  Cameras: {num_cameras}")
    for s, d in sorted(img_remap.items()):
        print(f"    {s}  ->  {d}")

    if os.path.isdir(dst_path):
        print(f"  Wiping existing {dst_path} ...")
        shutil.rmtree(dst_path)

    dst = zarr.open(dst_path, mode="a")
    dst_data = dst.require_group("data")
    dst_meta = dst.require_group("meta")

    # ---------------- /data ----------------
    pass_through_keys = [
        "state", "action", "n_contacts",
        "tactile", "tactile_connected", "tactile_ts_ms", "tactile_lag_ms",
    ]
    for key in pass_through_keys:
        if key not in src_data:
            continue
        print(f"  [data] copying {key:20s}  shape={src_data[key].shape}")
        dst_arr = _create_like(dst_data, key, src_data[key])
        _stream_copy(src_data[key], dst_arr)

    for src_key, dst_key in sorted(img_remap.items()):
        print(f"  [data] copying {src_key:20s}  ->  {dst_key:8s}  shape={src_data[src_key].shape}")
        dst_arr = _create_like(dst_data, dst_key, src_data[src_key])
        _stream_copy(src_data[src_key], dst_arr)

    # ---------------- /meta (verbatim copies) ----------------
    verbatim_meta = ["episode_ends", "tactile_baseline"]
    for key in verbatim_meta:
        if key not in src_meta:
            continue
        print(f"  [meta] copying {key}")
        src_arr = src_meta[key]
        dst_arr = dst_meta.create_dataset(
            key, shape=src_arr.shape, dtype=src_arr.dtype,
        )
        dst_arr[...] = src_arr[...]

    # ---------------- /meta (new camera fields) ----------------
    # Order matters: index 0 should be the agent, index 1 the wrist, to match
    # the original capture-time wiring in collect_with_home.py (which mapped
    # serial_to_index by RealSense enumeration order). We use the constants
    # from tactile_config.py for serial -> role; the renderer's role
    # resolution will then map back to the right /data/img_{i}.
    agent_serial = tc.AGENT_CAMERA_SERIAL
    wrist_serial = tc.WRIST_CAMERA_SERIAL
    serials_in_index_order = [agent_serial, wrist_serial][:num_cameras]
    # Pad with placeholder if needed (shouldn't happen for this data).
    while len(serials_in_index_order) < num_cameras:
        serials_in_index_order.append(f"unknown_cam_{len(serials_in_index_order)}")
    print(f"  [meta] camera_serials = {serials_in_index_order}  (from tactile_config.py)")

    dst_meta.create_dataset("camera_serials",
                            shape=(num_cameras,), dtype="S64")
    dst_meta["camera_serials"][...] = np.array(
        [s.encode("utf-8") for s in serials_in_index_order], dtype="S64"
    )

    intr = np.array(
        [[_D435_DEFAULT_INTRINSICS_640x480["fx"],
          _D435_DEFAULT_INTRINSICS_640x480["fy"],
          _D435_DEFAULT_INTRINSICS_640x480["ppx"],
          _D435_DEFAULT_INTRINSICS_640x480["ppy"]]
         for _ in range(num_cameras)],
        dtype=np.float32,
    )
    dst_meta.create_dataset("camera_intrinsics_native",
                            shape=intr.shape, dtype=intr.dtype)
    dst_meta["camera_intrinsics_native"][...] = intr
    print(f"  [meta] camera_intrinsics_native: D435 640x480 defaults "
          f"(fx={intr[0,0]:.0f}, fy={intr[0,1]:.0f}, ppx={intr[0,2]:.0f}, ppy={intr[0,3]:.0f}) — "
          f"agent overlay may be off by a few pixels from the original capture; "
          f"see the script docstring.")

    native = np.array(
        [[_DEFAULT_NATIVE_W, _DEFAULT_NATIVE_H] for _ in range(num_cameras)],
        dtype=np.int32,
    )
    dst_meta.create_dataset("camera_native_size",
                            shape=native.shape, dtype=native.dtype)
    dst_meta["camera_native_size"][...] = native

    dst_meta.create_dataset("agent_camera_serial", shape=(), dtype="S64")
    dst_meta["agent_camera_serial"][...] = agent_serial.encode("utf-8")
    dst_meta.create_dataset("wrist_camera_serial", shape=(), dtype="S64")
    dst_meta["wrist_camera_serial"][...] = wrist_serial.encode("utf-8")

    # trc_agent
    npz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            tc.TRANSFORMS_NPZ_PATH)
    agent_serial_from_npz, trc_agent = _load_trc_agent(npz_path)
    if trc_agent is not None:
        if agent_serial_from_npz and agent_serial_from_npz != agent_serial:
            print(f"  [warn] tactile_config.AGENT_CAMERA_SERIAL = {agent_serial} "
                  f"but transforms_agent.npz has {agent_serial_from_npz}. "
                  f"Using the latter for trc_agent; keeping the former as the "
                  f"camera-serial label.")
        dst_meta.create_dataset("trc_agent", shape=trc_agent.shape, dtype=trc_agent.dtype)
        dst_meta["trc_agent"][...] = trc_agent
        print(f"  [meta] trc_agent installed from {npz_path}")
    else:
        print(f"  [warn] no trc_agent saved; agent overlay will be skipped at render time.")

    dst_meta.create_dataset("recorded_img_size", shape=(2,), dtype=np.int32)
    dst_meta["recorded_img_size"][...] = np.array([224, 224], dtype=np.int32)

    print()
    print(f"  Done. Wrote {dst_path}.")
    print(f"  Now run:  python scripts/render_overlays.py {dst_path} <dst>_overlay.zarr")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Source zarr (old schema with img_*_raw arrays)")
    ap.add_argument("dst", help="Destination zarr (new schema). Wiped if exists.")
    args = ap.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
