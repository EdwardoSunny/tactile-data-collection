"""
Compute overlay-rendering normalization stats and save them to each task zarr.

Pipeline (per the design discussion in CLAUDE.md context):

  Stage 1: PER-CELL Hampel clip on raw tactile.
    For each (finger, taxel, axis): median + MAD across all frames in the
    task. Clip raw values to median +/- HAMPEL_K * MAD. Nukes sensor spikes
    so they don't pollute downstream stats.

  Stage 2: PER-EPISODE offset (mean of first N_BASELINE_FRAMES frames),
    with a consensus sanity-check fallback.
    For each episode: candidate = mean(first N frames of post-clip raw).
    consensus = median over episodes of those candidates (per cell+axis).
    Per cell+axis: if |candidate - consensus| > 5 * MAD(candidate),
    that episode didn't start idle for that cell — fall back to consensus.

  Stage 3: CROSS-TASK POOLED scale (median of per-episode p95 of |centered|).
    For each finger, gather p95 of |centered xy| and |centered z| from
    every episode in every input zarr, then take the median per finger.
    Insensitive to single bad episodes or single bad tasks; gives one
    shared scale so arrows mean the same thing across tasks.

  Stage 5: ADAPTIVE deadband from idle-frame noise.
    First N_BASELINE_FRAMES of each episode is assumed idle. Normalize
    those frames with the computed scales, take p95 of per-cell L2
    magnitude, double it. Caps at DEADBAND_FLOOR. The deadband is also
    shared across all input zarrs.

  Stage 6: persist to /meta/normalization/ in each input zarr.
    raw_clip_low, raw_clip_high           (2, 9, 3)  - task-specific
    episode_offsets                       (E, 2, 9, 3) - task-specific
    scale_xy, scale_z                     (2,)       - shared (pooled)
    deadband                              scalar     - shared (pooled)
    + scalar knobs (n_baseline_frames, mad_k, percentile)
    + attrs.source_zarrs : which zarrs were pooled together

Usage
-----
    # Pool the 4 task datasets together:
    python scripts/compute_overlay_normalization.py \
        teleop_data_cube.zarr teleop_data_charger.zarr \
        teleop_data_dishwasher.zarr teleop_data_tube.zarr

    # Inspect the numbers but don't write anything:
    python scripts/compute_overlay_normalization.py teleop_data_cube.zarr --no-save
"""
import argparse
import os
import sys

import numpy as np
import zarr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- knobs (saved alongside the stats so the renderer can verify) -----------
N_BASELINE_FRAMES = 15
HAMPEL_K = 8.0
EPISODE_PERCENTILE = 95.0
DEADBAND_FLOOR = 0.03
DEADBAND_NOISE_MULTIPLIER = 2.0
CONSENSUS_FALLBACK_K = 5.0


def _stage1_hampel_clip_bounds(raw, mad_k=HAMPEL_K):
    """Per-cell Hampel bounds. raw: (N, 2, 9, 3) -> two (2, 9, 3) bound arrays."""
    med = np.median(raw, axis=0)                           # (2, 9, 3)
    mad = np.median(np.abs(raw - med), axis=0)             # (2, 9, 3)
    # Floor MAD so a perfectly-constant cell doesn't collapse the bounds.
    mad = np.maximum(mad, 1.0)
    return (med - mad_k * mad).astype(np.float64), (med + mad_k * mad).astype(np.float64)


def _stage2_episode_offsets(raw_clipped, episode_ends,
                            n_baseline_frames=N_BASELINE_FRAMES,
                            consensus_k=CONSENSUS_FALLBACK_K):
    """Per-episode offset with consensus fallback for episodes that weren't idle.

    Returns:
        offsets : (E, 2, 9, 3) float64
    """
    starts = np.concatenate([[0], np.asarray(episode_ends[:-1], dtype=np.int64)])
    E = len(episode_ends)
    candidates = np.zeros((E, 2, 9, 3), dtype=np.float64)
    for ep_i, (s, e) in enumerate(zip(starts, episode_ends)):
        n_b = int(min(n_baseline_frames, e - s))
        if n_b > 0:
            candidates[ep_i] = np.mean(raw_clipped[s:s + n_b], axis=0)
    if E == 0:
        return candidates
    # Consensus = median across episodes per cell; MAD of candidate around that.
    consensus = np.median(candidates, axis=0)                   # (2, 9, 3)
    consensus_mad = np.median(np.abs(candidates - consensus), axis=0)  # (2, 9, 3)
    consensus_mad = np.maximum(consensus_mad, 1.0)
    # Cell-wise: where candidate disagrees with consensus, prefer consensus.
    deviation = np.abs(candidates - consensus)                  # (E, 2, 9, 3)
    threshold = consensus_k * consensus_mad                     # (2, 9, 3)
    fall_back = deviation > threshold                           # (E, 2, 9, 3)
    return np.where(fall_back, consensus, candidates)


def _apply_per_episode_offsets(raw_clipped, episode_ends, offsets):
    """In-place subtract per-episode offsets from raw_clipped, return centered."""
    starts = np.concatenate([[0], np.asarray(episode_ends[:-1], dtype=np.int64)])
    centered = raw_clipped.copy()
    for ep_i, (s, e) in enumerate(zip(starts, episode_ends)):
        centered[s:e] -= offsets[ep_i]
    return centered


def _stage3_pooled_scale(task_centered, percentile=EPISODE_PERCENTILE):
    """Cross-task pooled scale via median of per-episode percentiles per finger.

    task_centered: list of (centered, episode_ends) tuples.
    """
    xy_p = [[], []]
    z_p = [[], []]
    for centered, eps in task_centered:
        starts = np.concatenate([[0], np.asarray(eps[:-1], dtype=np.int64)])
        for s, e in zip(starts, eps):
            for fi in range(2):
                xy = np.abs(centered[s:e, fi, :, :2]).ravel()
                z = np.abs(centered[s:e, fi, :, 2]).ravel()
                if xy.size:
                    xy_p[fi].append(np.percentile(xy, percentile))
                if z.size:
                    z_p[fi].append(np.percentile(z, percentile))
    scale_xy = np.array([np.median(xy_p[fi]) if xy_p[fi] else 1.0 for fi in range(2)],
                        dtype=np.float64)
    scale_z = np.array([np.median(z_p[fi]) if z_p[fi] else 1.0 for fi in range(2)],
                       dtype=np.float64)
    scale_xy = np.maximum(scale_xy, 1.0)
    scale_z = np.maximum(scale_z, 1.0)
    return scale_xy, scale_z


def _stage5_adaptive_deadband(task_centered, scale_xy, scale_z,
                              n_baseline_frames=N_BASELINE_FRAMES,
                              floor=DEADBAND_FLOOR,
                              multiplier=DEADBAND_NOISE_MULTIPLIER):
    """Sample idle-frame (post-centering) per-cell L2 magnitudes from all
    tasks, take p95, multiply, floor."""
    mags = []
    for centered, eps in task_centered:
        starts = np.concatenate([[0], np.asarray(eps[:-1], dtype=np.int64)])
        for s, e in zip(starts, eps):
            n_b = int(min(n_baseline_frames, e - s))
            if n_b == 0:
                continue
            idle = centered[s:s + n_b]  # (n_b, 2, 9, 3)
            for fi in range(2):
                xy_n = idle[:, fi, :, :2] / scale_xy[fi]
                z_n = np.abs(idle[:, fi, :, 2:]) / scale_z[fi]
                full = np.concatenate([xy_n, z_n], axis=-1)
                mags.extend(np.linalg.norm(full, axis=-1).ravel().tolist())
    if not mags:
        return float(floor)
    noise_floor = float(np.percentile(np.array(mags), 95))
    return float(max(multiplier * noise_floor, floor))


def _save_to_meta(path, raw_clip_low, raw_clip_high, episode_offsets,
                  scale_xy, scale_z, deadband, source_paths):
    z = zarr.open(path, mode="a")
    meta = z["meta"]
    if "normalization" in meta:
        del meta["normalization"]
    norm = meta.create_group("normalization")
    norm.create_dataset("raw_clip_low",     data=raw_clip_low.astype(np.float32))
    norm.create_dataset("raw_clip_high",    data=raw_clip_high.astype(np.float32))
    norm.create_dataset("episode_offsets",  data=episode_offsets.astype(np.float32))
    norm.create_dataset("scale_xy",         data=np.asarray(scale_xy, dtype=np.float32))
    norm.create_dataset("scale_z",          data=np.asarray(scale_z, dtype=np.float32))
    norm.create_dataset("deadband",         data=np.float32(deadband))
    norm.create_dataset("n_baseline_frames", data=np.int32(N_BASELINE_FRAMES))
    norm.create_dataset("mad_k",             data=np.float32(HAMPEL_K))
    norm.create_dataset("percentile",        data=np.float32(EPISODE_PERCENTILE))
    norm.attrs["source_zarrs"] = list(source_paths)


def _process_one(path):
    """Run Stage 1+2 for one task. Returns dict + (centered, eps) for pooling."""
    z = zarr.open(path, mode="r")
    if "tactile" not in z["data"] or "episode_ends" not in z["meta"]:
        raise SystemExit(f"  [error] {path} missing /data/tactile or /meta/episode_ends")
    raw = np.asarray(z["data/tactile"][:], dtype=np.float64)
    eps = np.asarray(z["meta/episode_ends"][:], dtype=np.int64)

    clip_low, clip_high = _stage1_hampel_clip_bounds(raw)
    raw_clipped = np.clip(raw, clip_low, clip_high)
    offsets = _stage2_episode_offsets(raw_clipped, eps)
    centered = _apply_per_episode_offsets(raw_clipped, eps, offsets)
    return {
        "path": path,
        "raw_clip_low": clip_low,
        "raw_clip_high": clip_high,
        "episode_offsets": offsets,
        "episode_ends": eps,
        "centered": centered,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("zarrs", nargs="+", help="Task zarrs to pool together")
    ap.add_argument("--no-save", action="store_true",
                    help="Compute + print only; don't write to /meta")
    args = ap.parse_args()

    print(f"Computing pooled normalization across {len(args.zarrs)} zarrs:")
    for p in args.zarrs:
        print(f"  - {p}")
    print()

    # Stages 1+2 per task.
    results = []
    for path in args.zarrs:
        print(f"  {path}")
        print(f"    Stage 1: per-cell Hampel clip (k={HAMPEL_K})...")
        r = _process_one(path)
        print(f"    Stage 2: per-episode offsets (first {N_BASELINE_FRAMES} frames, "
              f"consensus-fallback k={CONSENSUS_FALLBACK_K})...")
        print(f"      raw_clip avg span:    "
              f"[{r['raw_clip_low'].mean():7.0f}, {r['raw_clip_high'].mean():7.0f}]")
        n_eps = len(r['episode_ends'])
        print(f"      episode_offsets:      ({n_eps}, 2, 9, 3) "
              f"mean(|offset|)={np.abs(r['episode_offsets']).mean():.1f}")
        results.append(r)

    # Stage 3 (shared).
    print(f"\n  Stage 3: cross-task pooled scale "
          f"(median of per-episode p{EPISODE_PERCENTILE:.0f} of |centered|)...")
    task_centered = [(r["centered"], r["episode_ends"]) for r in results]
    scale_xy, scale_z = _stage3_pooled_scale(task_centered)
    print(f"    LEFT  finger:  scale_xy={scale_xy[0]:8.1f}  scale_z={scale_z[0]:8.1f}")
    print(f"    RIGHT finger:  scale_xy={scale_xy[1]:8.1f}  scale_z={scale_z[1]:8.1f}")

    # Stage 5 (shared).
    print(f"\n  Stage 5: adaptive deadband from pooled idle-frame noise...")
    deadband = _stage5_adaptive_deadband(task_centered, scale_xy, scale_z)
    print(f"    deadband: {deadband:.4f}")

    # Stage 6.
    if args.no_save:
        print("\n  --no-save set: skipping persistence.")
        return
    print(f"\n  Stage 6: writing to /meta/normalization/ in each task zarr...")
    for r in results:
        _save_to_meta(r["path"], r["raw_clip_low"], r["raw_clip_high"],
                      r["episode_offsets"], scale_xy, scale_z, deadband,
                      args.zarrs)
        print(f"    saved -> {r['path']}/meta/normalization/")

    print("\nDone. To render overlays, the per-task rendering scripts will now "
          "pick up these stats automatically.")


if __name__ == "__main__":
    main()
