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

Tactile NORMALIZATION is preferably loaded from /meta/normalization (written
by scripts/compute_overlay_normalization.py — robust and pool-able across
multiple task zarrs). When that group is present, this script reads:
  - raw_clip_low / raw_clip_high  (per cell, per axis): Hampel-derived spike
                                  bounds applied BEFORE any aggregation
  - episode_offsets               (per episode, per cell, per axis):
                                  mean of first N_BASELINE_FRAMES with
                                  consensus-fallback for non-idle starts
  - scale_xy / scale_z            (per finger): median across episodes of
                                  per-episode p95 of |centered|, optionally
                                  pooled across multiple task zarrs so
                                  arrows mean the same thing across tasks
  - deadband                      (scalar): 2x the p95 of normalized idle
                                  magnitudes; auto-derived

Fallback (no /meta/normalization): per-task per-episode offsets only, with
DATASET_PERCENTILE scale and the constant NOISE_DEADBAND from
environment/tactile_overlay.py. Strongly suggest running
compute_overlay_normalization.py against your task zarrs once before relying
on the rendered overlays for downstream training.

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

from environment.tactile_overlay import (
    BOLD_ARROW_THICKNESS,
    BOLD_DOT_SIZE,
    MODES,
    MODE_KEYS,
    NOISE_DEADBAND,
    SensorOverlay,
    apply_deadband,
    mode_key,
)


# sensordrawing was calibrated at 640x480; upscale recorded frames to this
# before drawing so K and T_rc map onto the right pixel coordinates.
NATIVE_W, NATIVE_H = 640, 480


# Dataset-wide normalization knobs.
# - Offset is per EPISODE (mean of the first N_BASELINE_FRAMES of that
#   episode), not per dataset. Cancels session-to-session DC drift; each
#   episode "starts at zero" after the first ~1.5s of idle frames.
# - Scale is per FINGER, computed across the whole dataset from the
#   already-offset-cancelled values, at this percentile. 99 (not 95) keeps
#   the saturation point past the noise band — at p95 in datasets where
#   most frames are idle, the scale ends up dominated by the upper end of
#   the noise distribution and lighter contacts amplify noise visually.
DATASET_PERCENTILE = 99.0

# Episode-start baseline window. At 10Hz this is 1.5s of frames assumed to
# be contact-free (gripper just homed, episode just begun). If your episodes
# start with the fingers already in contact this number is wrong for you —
# bump it down to whatever the genuine idle window is.
N_BASELINE_FRAMES = 15

# BOLD_ARROW_THICKNESS, BOLD_DOT_SIZE, NOISE_DEADBAND, apply_deadband all live
# in environment/tactile_overlay.py so the live --viz path (threads.py) reads
# the exact same constants and helpers we use here. Do not redefine.


def _per_episode_offsets(tactile_zarr, episode_ends,
                         n_baseline_frames=N_BASELINE_FRAMES):
    """Per-episode offset: mean of the first n_baseline_frames frames of
    each episode, per (finger, cell, axis).

    Args:
        tactile_zarr      : zarr array (N, 2, 9, 3)
        episode_ends      : (E,) cumulative end indices
        n_baseline_frames : how many leading frames to average. Capped at
                            the actual episode length if the episode is
                            shorter than that.

    Returns:
        offsets : (E, 2, 9, 3) float32, one per episode.
    """
    raw = np.asarray(tactile_zarr[:], dtype=np.float64)  # (N, 2, 9, 3)
    E = len(episode_ends)
    offsets = np.zeros((E, 2, 9, 3), dtype=np.float64)
    if raw.size == 0 or E == 0:
        return offsets.astype(np.float32)

    starts = np.concatenate([[0], np.asarray(episode_ends[:-1], dtype=np.int64)])
    for ep_i, (s, e) in enumerate(zip(starts, episode_ends)):
        n = int(min(n_baseline_frames, e - s))
        if n > 0:
            offsets[ep_i] = np.mean(raw[s:s + n], axis=0)
    return offsets.astype(np.float32)


def _global_scales_post_offset(tactile_zarr, episode_ends, offsets,
                               percentile=DATASET_PERCENTILE):
    """Per-finger XY/Z scale = `percentile` of |raw - per_episode_offset|
    across the WHOLE dataset (every frame contributes; per-episode offset
    is subtracted first).

    Args:
        tactile_zarr  : zarr array (N, 2, 9, 3)
        episode_ends  : (E,) cumulative end indices
        offsets       : (E, 2, 9, 3) from _per_episode_offsets

    Returns:
        scale_xy : (2,) float32
        scale_z  : (2,) float32
    """
    raw = np.asarray(tactile_zarr[:], dtype=np.float64)
    if raw.size == 0:
        return np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)

    # Per-frame centered = raw - offset_of_its_episode. Done in place to
    # avoid doubling memory.
    starts = np.concatenate([[0], np.asarray(episode_ends[:-1], dtype=np.int64)])
    centered = raw.copy()
    for ep_i, (s, e) in enumerate(zip(starts, episode_ends)):
        centered[s:e] -= offsets[ep_i]

    scale_xy = np.empty(2, dtype=np.float64)
    scale_z = np.empty(2, dtype=np.float64)
    for fi in range(2):
        xy = np.abs(centered[:, fi, :, :2]).ravel()
        z = np.abs(centered[:, fi, :, 2]).ravel()
        scale_xy[fi] = np.percentile(xy, percentile) if xy.size else 1.0
        scale_z[fi] = np.percentile(z, percentile) if z.size else 1.0
    # Floor so we never divide by ~0 on a dataset where one axis hardly moves.
    scale_xy = np.maximum(scale_xy, 1.0)
    scale_z = np.maximum(scale_z, 1.0)
    return scale_xy.astype(np.float32), scale_z.astype(np.float32)


def _frame_to_episode_index(episode_ends):
    """Build a per-frame lookup: out[n] = which episode frame n belongs to.

    Returns (N,) int32 where N = episode_ends[-1].
    """
    if len(episode_ends) == 0:
        return np.empty(0, dtype=np.int32)
    N = int(episode_ends[-1])
    out = np.empty(N, dtype=np.int32)
    prev = 0
    for ep_i, end in enumerate(episode_ends):
        out[prev:int(end)] = ep_i
        prev = int(end)
    return out


def _copy_per_frame_arrays(src_data, dst_data, extra_keys=(),
                            img_dtype_override=None):
    """Copy state / action / n_contacts / joint_angles / grip_pos / tactile_*
    (+ any `extra_keys` such as the raw img_{i} arrays) from src to dst verbatim.
    Streams chunk-by-chunk so peak memory stays bounded.

    img_dtype_override: if set (np.uint8 or np.float32), the raw img_{i}
        arrays in `extra_keys` are converted to that dtype during the copy
        so the overlay zarr's raw passthrough matches the overlay variants'
        storage format. Other keys are copied verbatim."""
    pass_through_keys = list(extra_keys) + [
        "state", "action", "n_contacts",
        "joint_angles", "grip_pos",
        "tactile", "tactile_connected", "tactile_ts_ms", "tactile_lag_ms",
    ]
    img_extras = set(extra_keys)
    for key in pass_through_keys:
        if key not in src_data:
            continue
        src_arr = src_data[key]
        out_dtype = src_arr.dtype
        convert = False
        if img_dtype_override is not None and key in img_extras and \
                np.dtype(img_dtype_override) != src_arr.dtype:
            out_dtype = np.dtype(img_dtype_override)
            convert = True
        dst = dst_data.create_dataset(
            key,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=out_dtype,
            compressor=src_arr.compressor,
        )
        chunk = src_arr.chunks[0]
        for start in range(0, src_arr.shape[0], chunk):
            end = min(start + chunk, src_arr.shape[0])
            data = src_arr[start:end]
            if convert:
                if out_dtype == np.uint8:
                    data = np.clip(data * 255.0, 0.0, 255.0).astype(np.uint8)
                elif out_dtype == np.float32:
                    data = data.astype(np.float32) / 255.0
            dst[start:end] = data


def _create_overlay_arrays(dst_data, num_cams, img_shape, total_n, dtype,
                            variant_keys):
    """One (N, H, W, 3) array per (camera, mode_key) variant.

    variant_keys: subset of MODE_KEYS to actually create arrays for.
    """
    arrays = {}
    img_compressor = zarr.Blosc(cname="zstd", clevel=3,
                                shuffle=zarr.Blosc.BITSHUFFLE)
    for i in range(num_cams):
        for key in variant_keys:
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


def _load_or_compute_normalization(src_root, src_data, episode_ends, total_n):
    """Return (raw_clip_low, raw_clip_high, ep_offsets, scale_xy, scale_z, deadband).

    Preference order:
      1. Read from /meta/normalization if present (written by
         scripts/compute_overlay_normalization.py). This is the robust,
         cross-task-poolable path.
      2. Otherwise fall back to the legacy in-process computation: per-episode
         offset (no Hampel clip, no consensus fallback), per-task scale at
         DATASET_PERCENTILE, raw clip disabled, deadband = NOISE_DEADBAND
         from environment/tactile_overlay.py.

    raw_clip_low / raw_clip_high may both be None when raw clipping is disabled.
    """
    meta = src_root["meta"]
    if "normalization" in meta:
        norm = meta["normalization"]
        print("  Loaded normalization stats from /meta/normalization "
              "(robust + pooled).")
        try:
            sources = list(norm.attrs["source_zarrs"])
            print(f"    pooled across: {sources}")
        except KeyError:
            pass
        raw_clip_low = np.asarray(norm["raw_clip_low"][:], dtype=np.float64)
        raw_clip_high = np.asarray(norm["raw_clip_high"][:], dtype=np.float64)
        ep_offsets = np.asarray(norm["episode_offsets"][:], dtype=np.float32)
        scale_xy = np.asarray(norm["scale_xy"][:], dtype=np.float32)
        scale_z = np.asarray(norm["scale_z"][:], dtype=np.float32)
        deadband = float(np.asarray(norm["deadband"]))
        if ep_offsets.shape[0] != len(episode_ends):
            print(f"  [warn] /meta/normalization/episode_offsets has "
                  f"{ep_offsets.shape[0]} entries but /meta/episode_ends has "
                  f"{len(episode_ends)}. Re-run compute_overlay_normalization.py "
                  f"after appending data.")
        return raw_clip_low, raw_clip_high, ep_offsets, scale_xy, scale_z, deadband

    print("  [warn] /meta/normalization not present. Falling back to legacy")
    print("         per-task computation (no Hampel clip, no consensus fallback,")
    print("         no cross-task pooling). Run:")
    print(f"           python scripts/compute_overlay_normalization.py <zarrs>")
    print("         to install robust + pooled stats for this dataset.")
    print(f"  Computing per-episode offsets across {len(episode_ends)} episodes "
          f"(first {N_BASELINE_FRAMES} frames each)...")
    ep_offsets = _per_episode_offsets(src_data["tactile"], episode_ends)
    print(f"  Computing per-task global scales (p{DATASET_PERCENTILE:.0f}) "
          f"across {total_n} frames...")
    scale_xy, scale_z = _global_scales_post_offset(
        src_data["tactile"], episode_ends, ep_offsets,
    )
    return None, None, ep_offsets, scale_xy, scale_z, float(NOISE_DEADBAND)


def render(src_path, dst_path, variants=None, storage_dtype="uint8"):
    """Render overlay zarr from a raw zarr.

    variants: list of mode_key strings to render. None -> all of MODE_KEYS.
    storage_dtype: "uint8" (default, 4x smaller, [0,255]) or "float32" ([0,1]).
                   Applies to BOTH the raw img_{i} passthrough and the
                   overlay img_{i}_{key} variants in the destination zarr.
    """
    if not os.path.isdir(src_path):
        print(f"  [error] source zarr not found: {src_path}")
        sys.exit(1)

    if variants is None:
        variants = list(MODE_KEYS)
    invalid = [v for v in variants if v not in MODE_KEYS]
    if invalid:
        print(f"  [error] unknown variants: {invalid}. Valid: {MODE_KEYS}")
        sys.exit(1)

    if storage_dtype == "uint8":
        out_dtype = np.uint8
    elif storage_dtype == "float32":
        out_dtype = np.float32
    else:
        print(f"  [error] storage_dtype must be uint8 or float32, got {storage_dtype!r}")
        sys.exit(1)

    print(f"  Source: {src_path}")
    print(f"  Dest  : {dst_path}  (will be wiped + regenerated)")
    print(f"  Variants: {variants}")
    print(f"  Storage dtype: {storage_dtype}")
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
    recorded_wh = (img_shape[1], img_shape[0])  # (W, H)

    print(f"  Frames: {total_n}   cameras: {num_cams}   recorded img_shape: {img_shape}")

    # Build sensordrawing once — eagerly constructs both SensorDrawers and
    # both SensorNormalizers. Cheap; identical state for every frame.
    overlay = SensorOverlay(baseline=None)

    src_meta = src_root["meta"]
    if "episode_ends" not in src_meta:
        print("  [error] source has no /meta/episode_ends — can't compute "
              "per-episode offsets.")
        sys.exit(1)
    episode_ends = np.asarray(src_meta["episode_ends"][:], dtype=np.int64)
    n_episodes = len(episode_ends)

    # Load pooled / robust normalization stats from /meta/normalization if
    # available (written by scripts/compute_overlay_normalization.py). These
    # are: Hampel raw clip bounds (per cell), per-episode offsets with
    # consensus fallback, cross-task pooled scales, and adaptive deadband.
    # If not present, fall back to the old self-derived path: per-episode
    # offset (no Hampel, no consensus fallback) + per-task DATASET_PERCENTILE
    # scale + the constant NOISE_DEADBAND from environment/tactile_overlay.py.
    raw_clip_low, raw_clip_high, ep_offsets, scale_xy_2, scale_z_2, deadband = \
        _load_or_compute_normalization(src_root, src_data, episode_ends, total_n)

    # Per-frame episode index for the loop below.
    frame_to_ep = _frame_to_episode_index(episode_ends)

    # Zero out the normalizer's offset (we subtract per-episode upstream).
    overlay.norm_L.offset = np.zeros((9, 3), dtype=np.float32)
    overlay.norm_L.global_scale_xy = float(scale_xy_2[0])
    overlay.norm_L.global_scale_z = float(scale_z_2[0])
    overlay.norm_R.offset = np.zeros((9, 3), dtype=np.float32)
    overlay.norm_R.global_scale_xy = float(scale_xy_2[1])
    overlay.norm_R.global_scale_z = float(scale_z_2[1])
    print(f"    LEFT  finger:  scale_xy={scale_xy_2[0]:8.1f}  scale_z={scale_z_2[0]:8.1f}")
    print(f"    RIGHT finger:  scale_xy={scale_xy_2[1]:8.1f}  scale_z={scale_z_2[1]:8.1f}")
    print(f"    deadband:      {deadband:.4f}")

    # ---- Create destination zarr -------------------------------------
    dst_root = zarr.open(dst_path, mode="a")
    dst_data = dst_root.require_group("data")
    dst_meta = dst_root.require_group("meta")

    # Pass-through raw img_{i} arrays so the overlay zarr is a standalone,
    # training-ready dataset: img_{i} = raw frame, img_{i}_{key} = overlaid.
    # Raw img passthrough is converted to the same storage_dtype as the
    # overlay variants so the whole overlay zarr is dtype-consistent.
    _copy_per_frame_arrays(src_data, dst_data, extra_keys=img_keys,
                            img_dtype_override=out_dtype)

    # Pass-through /meta -> /meta verbatim (recursive — meta/normalization is a subgroup).
    def _copy_meta(src, dst):
        for key in src.keys():
            item = src[key]
            if hasattr(item, "shape"):  # zarr Array
                d = dst.create_dataset(key, shape=item.shape, dtype=item.dtype)
                d[...] = item[...]
            else:  # zarr Group
                _copy_meta(item, dst.require_group(key))
    _copy_meta(src_root["meta"], dst_meta)

    overlay_arrs = _create_overlay_arrays(dst_data, num_cams, img_shape,
                                          total_n, out_dtype, variants)

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

    # Restrict the per-frame iteration to only the requested variants.
    active_modes = [(m, s, scale) for (m, s, scale) in MODES
                    if mode_key(m, s) in variants]

    for start in range(0, total_n, CHUNK):
        end = min(start + CHUNK, total_n)
        out_buf = {(i, k): np.empty((end - start, *img_shape), dtype=out_dtype)
                   for i in range(num_cams) for k in variants}

        tac_chunk = tactile_xyz_full[start:end]
        joint_chunk = joint_full[start:end]
        grip_chunk = grip_full[start:end]

        # Stage 1 raw Hampel clip (if loaded), then subtract per-episode
        # offset, then run through the normalizer (which now only applies the
        # global per-finger scale, since we zeroed its offset above). Then
        # gate per-cell normalized vectors below the loaded deadband so idle
        # frames don't draw jittery arrows.
        norm_pairs = []
        for j in range(end - start):
            ep_i = int(frame_to_ep[start + j])
            raw_L = np.asarray(tac_chunk[j, 0], dtype=np.float64)
            raw_R = np.asarray(tac_chunk[j, 1], dtype=np.float64)
            if raw_clip_low is not None:
                raw_L = np.clip(raw_L, raw_clip_low[0], raw_clip_high[0])
                raw_R = np.clip(raw_R, raw_clip_low[1], raw_clip_high[1])
            centered_L = raw_L - ep_offsets[ep_i, 0]
            centered_R = raw_R - ep_offsets[ep_i, 1]
            nL, nR = overlay.normalize(centered_L, centered_R)
            nL = apply_deadband(nL, deadband)
            nR = apply_deadband(nR, deadband)
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

                for mode, is_spatial, scale in active_modes:
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
                    if out_dtype == np.uint8:
                        out_buf[(cam_idx, key)][j] = drawn_recorded
                    else:
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
    print(f"  Done. {total_n} frames x {num_cams} cams x {len(variants)} variants "
          f"in {elapsed:.1f}s ({total_n / max(elapsed, 1e-6):.1f} fps).")
    print(f"  Wrote: {dst_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="?", default="teleop_data.zarr",
                    help="Source raw zarr (default: teleop_data.zarr)")
    ap.add_argument("dst", nargs="?", default="teleop_data_overlay.zarr",
                    help="Destination overlay zarr (default: teleop_data_overlay.zarr). "
                         "Always wiped + regenerated.")
    ap.add_argument("--variants", type=str, default=None,
                    help=f"Comma-separated subset of overlay variants to render. "
                         f"Default: all 6. Valid: {','.join(MODE_KEYS)}")
    ap.add_argument("--dtype", choices=["uint8", "float32"], default="uint8",
                    help="Storage dtype for both raw img passthrough and overlay "
                         "variants in the dest zarr. uint8 is ~4x smaller and the "
                         "natural choice for integer images (training pipelines "
                         "typically normalize on load anyway). float32 [0,1] "
                         "preserves the legacy on-disk format.")
    args = ap.parse_args()
    variants = (None if args.variants is None
                else [v.strip() for v in args.variants.split(",") if v.strip()])
    render(args.src, args.dst, variants=variants, storage_dtype=args.dtype)


if __name__ == "__main__":
    main()
