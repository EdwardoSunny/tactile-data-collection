"""
Post-hoc tactile-overlay renderer.

Reads the raw recording at SRC (default teleop_data.zarr) — which contains
un-overlaid 224x224 images, per-frame robot state, raw tactile, episode_ends,
and camera/extrinsics metadata under /meta — and writes a fresh DST zarr
(default teleop_data_overlay.zarr) containing the SAME per-frame data plus
four overlay-rendered image arrays per camera:

    /data/img_{i}_arrow   single per-finger arrow
    /data/img_{i}_grid    nine arrows per finger
    /data/img_{i}_point   single solid circle per finger
    /data/img_{i}_bar     binary bottom-bar per finger

DST is fully regenerated on every run (any existing directory is removed
first), so re-running picks up any newly-recorded episodes appended to SRC.

The overlay constants in tactile_config.py (wrist anchors, arrow-length cap,
arrow thickness) are calibrated for native-resolution drawing (the original
recorder drew at 640x480 and THEN downsized). To keep the rendered overlay
visually identical to what the live --viz windows show, this script upscales
each recorded 224x224 frame back to the camera's native size, draws the
overlay there, and downsizes the result back to 224x224. Native dims and the
agent-camera robot->camera extrinsic are read from /meta, written there once
at collection time by recorder.py.
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

from environment import tactile_overlay
import tactile_config as tc


# All four modes get a separate per-camera array.
_MODES = ("arrow", "grid", "point", "bar")


class _Intrinsics:
    """Duck-typed stand-in for rs.intrinsics; only .fx/.fy/.ppx/.ppy are
    read by tactile_overlay.project_points_to_image."""

    __slots__ = ("fx", "fy", "ppx", "ppy")

    def __init__(self, fx, fy, ppx, ppy):
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)


def _decode_serial(b):
    """zarr stores fixed-width bytes for strings; lift back to str."""
    if isinstance(b, np.ndarray):
        b = b.item() if b.shape == () else b[0]
    if isinstance(b, bytes):
        return b.decode("utf-8").rstrip("\x00")
    return str(b)


def _load_meta(src_root):
    """Pull everything render_overlays needs out of /meta. Fields are None
    when missing so we can degrade gracefully (wrist-only rendering when the
    agent transform isn't available, etc.)."""
    meta = src_root["meta"]
    out = {
        "episode_ends": meta["episode_ends"][:] if "episode_ends" in meta else None,
        "tactile_baseline": (
            meta["tactile_baseline"][:] if "tactile_baseline" in meta else None
        ),
        "serials": None,
        "intrinsics_native": None,
        "native_size": None,
        "agent_serial": None,
        "wrist_serial": None,
        "trc_agent": None,
        "recorded_img_size": None,
    }
    if "camera_serials" in meta:
        out["serials"] = [_decode_serial(s) for s in meta["camera_serials"][:]]
    if "camera_intrinsics_native" in meta:
        out["intrinsics_native"] = meta["camera_intrinsics_native"][:]
    if "camera_native_size" in meta:
        out["native_size"] = meta["camera_native_size"][:]
    if "agent_camera_serial" in meta:
        out["agent_serial"] = _decode_serial(meta["agent_camera_serial"][...])
    if "wrist_camera_serial" in meta:
        out["wrist_serial"] = _decode_serial(meta["wrist_camera_serial"][...])
    if "trc_agent" in meta:
        out["trc_agent"] = meta["trc_agent"][:]
    if "recorded_img_size" in meta:
        out["recorded_img_size"] = meta["recorded_img_size"][:]
    return out


def _resolve_camera_roles(meta_info, num_cams):
    """Map agent/wrist serials -> image index in /data/img_{i}.

    Falls back to: cam 0 = agent, cam 1 = wrist when serials weren't recorded
    (matches the original capture-time wiring in collect_with_home.py).
    Returns (agent_idx, wrist_idx); either may be None if not resolvable.
    """
    serials = meta_info["serials"]
    if serials is None or len(serials) == 0:
        agent_idx = 0 if num_cams >= 1 else None
        wrist_idx = 1 if num_cams >= 2 else (0 if num_cams >= 1 else None)
        return agent_idx, wrist_idx

    serial_to_idx = {s: i for i, s in enumerate(serials)}
    agent_idx = serial_to_idx.get(meta_info["agent_serial"]) if meta_info["agent_serial"] else None
    wrist_idx = serial_to_idx.get(meta_info["wrist_serial"]) if meta_info["wrist_serial"] else None
    if agent_idx is None and num_cams >= 1:
        agent_idx = 0
    if wrist_idx is None:
        wrist_idx = 1 if num_cams >= 2 else (0 if num_cams >= 1 else None)
    return agent_idx, wrist_idx


def _copy_per_frame_arrays(src_data, dst_data, extra_keys=()):
    """Copy state / action / n_contacts / tactile_* (+ any `extra_keys` such
    as the raw img_{i} arrays) from src to dst verbatim. Streams chunk-by-
    chunk so peak memory stays bounded."""
    pass_through_keys = list(extra_keys) + [
        "state", "action", "n_contacts",
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
    """One (N, H, W, 3) array per (camera, mode), matching the source img dtype
    (float32 in [0,1])."""
    arrays = {}
    img_compressor = zarr.Blosc(cname="zstd", clevel=3,
                                shuffle=zarr.Blosc.BITSHUFFLE)
    for i in range(num_cams):
        for mode in _MODES:
            name = f"img_{i}_{mode}"
            arrays[(i, mode)] = dst_data.create_dataset(
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
    """Return ['img_0', 'img_1', ...] in numeric order, skipping any future
    suffixed variants like img_0_arrow (which shouldn't appear in a raw zarr
    anyway). Pure-digit suffix = bare raw frame."""
    keys = []
    for k in src_data.keys():
        if not k.startswith("img_"):
            continue
        rest = k[len("img_"):]
        if rest.isdigit():
            keys.append((int(rest), k))
    keys.sort()
    return [k for _, k in keys]


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

    meta_info = _load_meta(src_root)
    if meta_info["tactile_baseline"] is not None:
        tactile_overlay.set_baseline(meta_info["tactile_baseline"])
        print(f"  Baseline: installed from /meta/tactile_baseline "
              f"(shape={meta_info['tactile_baseline'].shape})")
    else:
        tactile_overlay.clear_baseline()
        print(f"  Baseline: NONE in source /meta — overlay will use raw tactile values "
              f"(noisier; see CLAUDE.md > Tactile pipeline).")

    img_keys = _img_keys_in_order(src_data)
    num_cams = len(img_keys)
    if num_cams == 0:
        print("  [error] no /data/img_* arrays in source.")
        sys.exit(1)
    img_shape = tuple(src_data[img_keys[0]].shape[1:])
    img_dtype = src_data[img_keys[0]].dtype
    recorded_wh = (img_shape[1], img_shape[0])  # (W, H)

    print(f"  Frames: {total_n}   cameras: {num_cams}   recorded img_shape: {img_shape}")

    agent_idx, wrist_idx = _resolve_camera_roles(meta_info, num_cams)

    # Per-camera native size + agent intrinsics. We RENDER at native resolution
    # so the tactile_config constants (wrist anchors, arrow-length caps) match
    # what they were tuned for, then downsize back to recorded size.
    native_wh_per_cam = []
    for i in range(num_cams):
        if meta_info["native_size"] is not None and i < len(meta_info["native_size"]):
            w, h = int(meta_info["native_size"][i][0]), int(meta_info["native_size"][i][1])
        else:
            # Fallback: assume 640x480 (the only resolution the camera config opens).
            w, h = 640, 480
        native_wh_per_cam.append((w, h))

    agent_intr_native = None
    if (agent_idx is not None
            and meta_info["intrinsics_native"] is not None
            and agent_idx < len(meta_info["intrinsics_native"])):
        intr = meta_info["intrinsics_native"][agent_idx]
        agent_intr_native = _Intrinsics(intr[0], intr[1], intr[2], intr[3])
        print(f"  Agent cam (idx {agent_idx}, native {native_wh_per_cam[agent_idx]}): "
              f"fx={agent_intr_native.fx:.1f} fy={agent_intr_native.fy:.1f} "
              f"ppx={agent_intr_native.ppx:.1f} ppy={agent_intr_native.ppy:.1f}")
    else:
        print(f"  [warn] agent-camera intrinsics missing; "
              f"agent overlay will be skipped (wrist still rendered).")

    trc_agent = meta_info["trc_agent"]
    if trc_agent is None and agent_intr_native is not None:
        print(f"  [warn] /meta/trc_agent missing; agent overlay will be skipped.")

    # ---- Create destination zarr -------------------------------------
    dst_root = zarr.open(dst_path, mode="a")
    dst_data = dst_root.require_group("data")
    dst_meta = dst_root.require_group("meta")

    # Also pass through the raw img_{i} arrays so the overlay zarr is a
    # standalone, training-ready dataset: img_{i} = raw frame, img_{i}_{mode}
    # = same frame with the named overlay drawn on it.
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
    # Stream per chunk: float32 224x224x3 * 64 frames * 4 modes * 2 cams ~= 9 MB
    # in flight, well under memory.
    CHUNK = 64
    poses_full = src_data["state"][:, :6]            # (N, 6)
    tactile_xyz_full = src_data["tactile"]           # (N, 2, 9, 3)
    tactile_conn_full = src_data["tactile_connected"]  # (N, 2, 9)

    t_start = time.time()
    last_log = t_start

    for start in range(0, total_n, CHUNK):
        end = min(start + CHUNK, total_n)
        out_buf = {(i, m): np.empty((end - start, *img_shape), dtype=img_dtype)
                   for i in range(num_cams) for m in _MODES}

        tac_chunk = tactile_xyz_full[start:end]
        conn_chunk = tactile_conn_full[start:end]
        poses_chunk = poses_full[start:end]

        for cam_idx in range(num_cams):
            src_imgs = src_data[f"img_{cam_idx}"][start:end]
            native_wh = native_wh_per_cam[cam_idx]
            is_agent = (cam_idx == agent_idx)
            is_wrist = (cam_idx == wrist_idx)
            can_render_agent = (is_agent
                                and agent_intr_native is not None
                                and trc_agent is not None)

            for j in range(end - start):
                base_224 = _to_uint8_bgr(src_imgs[j])
                vals_L = tac_chunk[j, 0]
                vals_R = tac_chunk[j, 1]
                conn_L = conn_chunk[j, 0]
                conn_R = conn_chunk[j, 1]
                pose = poses_chunk[j]

                # Upscale once per frame; reuse the native canvas across modes.
                base_native = cv2.resize(base_224, native_wh)

                for mode in _MODES:
                    if can_render_agent:
                        drawn_native = tactile_overlay.draw_agent_overlay(
                            base_native, pose, vals_L, vals_R, conn_L, conn_R,
                            trc_agent, agent_intr_native, mode=mode,
                        )
                    elif is_wrist:
                        drawn_native = tactile_overlay.draw_wrist_overlay(
                            base_native, vals_L, vals_R, conn_L, conn_R, mode=mode,
                        )
                    else:
                        # No overlay defined for this camera; just pass through.
                        drawn_native = base_native

                    # Downsize back to recorded shape and convert to float [0,1].
                    drawn_recorded = cv2.resize(drawn_native, recorded_wh)
                    out_buf[(cam_idx, mode)][j] = _to_float32_01(drawn_recorded)

        for key, arr in out_buf.items():
            overlay_arrs[key][start:end] = arr

        now = time.time()
        if now - last_log >= 1.0:
            elapsed = now - t_start
            fps = end / max(elapsed, 1e-6)
            eta = (total_n - end) / max(fps, 1e-6)
            print(f"  ...{end:>6d}/{total_n}   ({fps:5.1f} fps, eta {eta:5.1f}s)")
            last_log = now

    elapsed = time.time() - t_start
    print(f"  Done. {total_n} frames x {num_cams} cams x {len(_MODES)} modes "
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
