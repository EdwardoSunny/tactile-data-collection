"""
Split the overlay-augmented zarr into N single-mode "training-ready" zarrs.

Reads `teleop_data_overlay.zarr` (the output of scripts/render_overlays.py)
and writes one zarr per overlay mode under an output directory, each containing:

  /data/img_0, img_1, ...   <- ONLY the chosen mode's overlay (renamed to the
                               canonical img_{i} key; for mode=raw these come
                               from the source's bare img_{i} arrays)
  /data/state               <- 10-dim (xyz + 6D rotation + grasp), via
                               xarm_state_to_10d. Replaces the on-disk 7-dim
                               state because diffusion policies don't want
                               raw Euler angles (wraparound discontinuity).
  /data/action              <- 10-dim. Default: action[t] = state_10d[t+1] within
                               the episode, last frame repeated. With
                               --delta-actions: per-step delta on the first 9
                               dims (xyz + 6D rot), grasp stays absolute.
  /data/n_contacts, tactile, tactile_connected, tactile_ts_ms, tactile_lag_ms
                            <- copied verbatim from the overlay zarr
  /meta                     <- copied verbatim (episode_ends, tactile_baseline,
                               camera_*, trc_agent, recorded_img_size, ...)

So you get e.g.:
    <out>/raw.zarr     <- img_0 = bare un-overlaid frame
    <out>/arrow.zarr   <- img_0 = the arrow-overlay frame
    <out>/grid.zarr
    <out>/point.zarr
    <out>/bar.zarr

…all otherwise identical, so you can run ablations like
    train policy_A on <out>/arrow.zarr
    train policy_B on <out>/grid.zarr
and compare which overlay (if any) makes the policy better.

Each output zarr is wiped + regenerated on every run, so re-running picks up
any new episodes that have been added to the overlay source.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def xarm_state_to_10d(state):
    """7-dim on-disk state [x,y,z,rx,ry,rz,grasp] -> 10-dim training state
    [x, y, z, r00, r01, r02, r10, r11, r12, grasp].

    Equivalent to environment.utils.xarm_state_to_10d but written in pure
    numpy / scipy so this script can run in envs that don't have torch or
    pyrealsense2 installed (environment/utils.py drags both in at import).
    The 6D rotation representation is the first two rows of the rotation
    matrix flattened, per Zhou et al. 2019 — same convention as the original.
    """
    state = np.asarray(state, dtype=np.float64)
    pos = state[:, :3]
    euler = state[:, 3:6]
    rot_mat = R.from_euler("xyz", euler, degrees=True).as_matrix()  # (N,3,3)
    rot_6d = rot_mat[:, :2, :].reshape(-1, 6)                       # (N,6)
    out = np.zeros((state.shape[0], 10), dtype=np.float64)
    out[:, :3] = pos
    out[:, 3:9] = rot_6d
    out[:, 9] = state[:, 6]
    return out


_ALL_MODES = ("raw", "arrow", "grid", "point", "bar")


def _bare_camera_indices(group):
    """img_0, img_1, ... -> [0, 1, ...]"""
    import re
    pat = re.compile(r"^img_(\d+)$")
    out = []
    for k in group.keys():
        m = pat.match(k)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _mode_img_key(cam_idx, mode):
    """Source key for camera `cam_idx`'s frames in `mode`."""
    return f"img_{cam_idx}" if mode == "raw" else f"img_{cam_idx}_{mode}"


def _compute_actions(state_10d, episode_ends, delta_actions):
    """Per-episode action derivation, matching XArmDataset.

    Default: action[t] = state_10d[t+1] within episode; action[last] = state_10d[last].
    Delta: action[t, :9] = state_10d[t+1, :9] - state_10d[t, :9]; action[t, 9] = grasp[t]
           (absolute); action[last, :9] = 0.
    """
    actions = np.empty_like(state_10d)
    start = 0
    for end in episode_ends:
        ep = state_10d[start:end]
        if delta_actions:
            d = np.zeros_like(ep)
            d[:-1, :9] = ep[1:, :9] - ep[:-1, :9]
            d[:, 9] = ep[:, 9]
            d[-1, :9] = 0.0
            actions[start:end] = d
        else:
            actions[start:end] = np.concatenate((ep[1:], ep[[-1]]), axis=0)
        start = int(end)
    return actions


def _stream_copy_array(src_arr, dst_group, name, dtype=None):
    """Create dst_group/name with src_arr's shape/chunks/compressor and stream-copy."""
    dt = dtype if dtype is not None else src_arr.dtype
    dst = dst_group.create_dataset(
        name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,
        dtype=dt,
        compressor=src_arr.compressor,
    )
    chunk = src_arr.chunks[0]
    for s in range(0, src_arr.shape[0], chunk):
        e = min(s + chunk, src_arr.shape[0])
        if dtype is None:
            dst[s:e] = src_arr[s:e]
        else:
            dst[s:e] = np.asarray(src_arr[s:e], dtype=dt)
    return dst


def _copy_meta(src_meta, dst_meta):
    for k in src_meta.keys():
        src = src_meta[k]
        dst = dst_meta.create_dataset(k, shape=src.shape, dtype=src.dtype)
        dst[...] = src[...]


def build_one_mode(src_root, out_path, mode, delta_actions):
    print(f"  [{mode}] writing {out_path}  (delta_actions={delta_actions})")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    src_data = src_root["data"]
    src_meta = src_root["meta"]
    cam_idxs = _bare_camera_indices(src_data)
    if not cam_idxs:
        raise RuntimeError("source zarr has no img_{i} arrays")

    # Make sure every camera HAS this mode (or it's the raw mode which always
    # exists). For non-raw modes, missing img_{i}_{mode} is a hard error so the
    # user notices the bad input rather than silently falling back to raw.
    if mode != "raw":
        for i in cam_idxs:
            key = _mode_img_key(i, mode)
            if key not in src_data:
                raise RuntimeError(
                    f"source has img_{i} but not {key} — re-run "
                    f"scripts/render_overlays.py and try again."
                )

    dst_root = zarr.open(out_path, mode="a")
    dst_data = dst_root.require_group("data")
    dst_meta = dst_root.require_group("meta")

    # --- per-frame arrays copied verbatim ---
    verbatim_keys = ["n_contacts", "tactile", "tactile_connected",
                     "tactile_ts_ms", "tactile_lag_ms"]
    for k in verbatim_keys:
        if k in src_data:
            _stream_copy_array(src_data[k], dst_data, k)

    # --- images: copy chosen mode, rename to bare img_{i} ---
    for i in cam_idxs:
        src_key = _mode_img_key(i, mode)
        src = src_data[src_key]
        dst = dst_data.create_dataset(
            f"img_{i}",
            shape=src.shape,
            chunks=src.chunks,
            dtype=src.dtype,
            compressor=src.compressor,
        )
        chunk = src.chunks[0]
        for s in range(0, src.shape[0], chunk):
            e = min(s + chunk, src.shape[0])
            dst[s:e] = src[s:e]

    # --- /data/state: 10-dim ---
    state_7d = np.asarray(src_data["state"][:], dtype=np.float32)
    state_10d = xarm_state_to_10d(state_7d).astype(np.float32)
    dst_state = dst_data.create_dataset(
        "state",
        shape=state_10d.shape,
        chunks=(min(1024, state_10d.shape[0]), state_10d.shape[1]),
        dtype=np.float32,
        compressor=src_data["state"].compressor,
    )
    dst_state[...] = state_10d

    # --- /data/action: 10-dim ---
    episode_ends = np.asarray(src_meta["episode_ends"][:], dtype=np.int64)
    actions = _compute_actions(state_10d, episode_ends, delta_actions)
    dst_act = dst_data.create_dataset(
        "action",
        shape=actions.shape,
        chunks=(min(1024, actions.shape[0]), actions.shape[1]),
        dtype=np.float32,
        compressor=src_data["state"].compressor,
    )
    dst_act[...] = actions

    # --- /meta: verbatim, plus a couple of derived fields ---
    _copy_meta(src_meta, dst_meta)
    # Note which mode this zarr was built for so downstream code can introspect.
    dst_meta.create_dataset("image_mode", shape=(), dtype="S16")
    dst_meta["image_mode"][...] = mode.encode("utf-8")
    dst_meta.create_dataset("action_kind", shape=(), dtype="S16")
    dst_meta["action_kind"][...] = (b"delta" if delta_actions else b"next_state")
    # Original 7-dim state is dropped from /data (state in /data is now 10-dim),
    # but we keep the kind tag so it's clear.
    dst_meta.create_dataset("state_kind", shape=(), dtype="S16")
    dst_meta["state_kind"][...] = b"10d_xyz_rot6d_grasp"

    print(f"  [{mode}] done. frames={state_10d.shape[0]}  episodes={len(episode_ends)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?", default="teleop_data_overlay.zarr",
                    help="Source overlay zarr (default: teleop_data_overlay.zarr)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: <src-dir>/training_datasets)")
    ap.add_argument("--modes", nargs="+", default=list(_ALL_MODES),
                    choices=list(_ALL_MODES),
                    help="Which modes to build (default: all 5)")
    ap.add_argument("--delta-actions", action="store_true",
                    help="Compute action as per-step delta on the first 9 dims "
                         "(grasp stays absolute). Default: action[t] = state_10d[t+1].")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"  [error] source zarr not found: {args.src}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.src)), "training_datasets"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Source : {args.src}")
    print(f"  Out dir: {out_dir}")
    print(f"  Modes  : {args.modes}")
    print(f"  Action : {'delta' if args.delta_actions else 'next_state'}")
    print()

    src_root = zarr.open(args.src, mode="r")
    t0 = time.time()
    for mode in args.modes:
        out_path = os.path.join(out_dir, f"{mode}.zarr")
        build_one_mode(src_root, out_path, mode, args.delta_actions)
    print()
    print(f"  All done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
