"""Visual tuning aids for the binmarker contact threshold.

For each task and each of the requested episodes, produces:
  <out>/<task>_timeline.png  — per-finger aggregate magnitude m(t) for the 5
                               episodes, grasp state shaded, candidate
                               thresholds as horizontal lines
  <out>/<task>_montage_T<thr>.png — for each episode, sample frames rendered
                               with the binmarker overlay at threshold <thr>:
                               approach (pre-contact), first contact, mid-grasp,
                               post-release. Agent view, 2x upscaled, with the
                               per-finger m values printed above each frame.

Run: python scripts/tune_binmarker.py --tasks cube tube charger dishwasher \
        --episodes 0 20 45 70 95 --thresholds 0.5 0.6 0.8 --render-thr 0.6
"""
import argparse
import os
import sys

import cv2
import numpy as np
import zarr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.tactile_overlay import SensorOverlay
from binmarker_common import (FIXED_ARROW_LEN, T_ON, T_OFF, idle_window_normalization,
                              normalize_all, binmarker_feed, contact_metric, contact_states)

NATIVE_W, NATIVE_H = 640, 480
DATA = "/workspace-vast/edwardosunny/tactile_data"


def to_u8(img01):
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def pick_moments(m_ep, grasp_ep, thr, on_ep=None):
    """Indices (within episode) of approach / first-contact / mid-grasp / release."""
    above = on_ep.any(axis=1) if on_ep is not None else (m_ep.max(axis=1) >= thr)
    first_c = int(np.argmax(above)) if above.any() else len(m_ep) // 2
    approach = max(first_c - 8, 0)
    closed = np.where(grasp_ep >= 0.5)[0]
    mid = int(closed[len(closed) // 2]) if len(closed) else len(m_ep) // 2
    if len(closed):
        release = min(int(closed[-1]) + 6, len(m_ep) - 1)
    else:
        release = len(m_ep) - 1
    return [("approach", approach), ("first_contact", first_c),
            ("mid_grasp", mid), ("post_release", release)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["cube", "tube", "charger", "dishwasher"])
    ap.add_argument("--episodes", nargs="+", type=int, default=[0, 20, 45, 70, 95])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.8])
    ap.add_argument("--render-thr", type=float, default=0.6)
    ap.add_argument("--out", default="/workspace-vast/edwardosunny/tmp/binmarker_tuning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for task in args.tasks:
        z = zarr.open(f"{DATA}/teleop_data_{task}.zarr", mode="r")
        raw = np.asarray(z["data/tactile"][:], dtype=np.float64)
        grasp = np.asarray(z["data/state"][:, 6])
        ends = np.asarray(z["meta/episode_ends"][:])
        starts = np.r_[0, ends[:-1]]
        eps = [e for e in args.episodes if e < len(ends)]

        offset, sxy, sz = idle_window_normalization(raw, ends)
        norm = normalize_all(raw, offset, sxy, sz)
        m = contact_metric(norm)  # (N, 2) max-cell
        on = np.zeros_like(m, dtype=bool)
        for s_, e_ in zip(starts, ends):
            for f in range(2):
                on[s_:e_, f] = contact_states(m[s_:e_, f])

        # ---- timeline figure ----
        fig, axes = plt.subplots(len(eps), 1, figsize=(14, 2.2 * len(eps)), sharey=True)
        for ax, ep in zip(np.atleast_1d(axes), eps):
            s, e = starts[ep], ends[ep]
            t = np.arange(e - s)
            ax.plot(t, m[s:e, 0], lw=0.9, label="left finger m")
            ax.plot(t, m[s:e, 1], lw=0.9, label="right finger m")
            ax.fill_between(t, 0, m[s:e].max(), where=grasp[s:e] >= 0.5,
                            alpha=0.12, color="k", label="grasp closed")
            for thr in args.thresholds:
                ax.axhline(thr, ls="--", lw=0.8, color="r" if thr == args.render_thr else "gray")
            ax.set_ylabel(f"ep{ep}")
            ax.set_ylim(0, min(max(3.5, np.percentile(m[s:e], 99.5) * 1.2), 8))
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7)
        np.atleast_1d(axes)[0].set_title(
            f"{task}: per-finger aggregate contact magnitude; dashed=thresholds {args.thresholds} (red={args.render_thr})")
        fig.tight_layout()
        fig.savefig(f"{args.out}/{task}_timeline.png", dpi=110)
        plt.close(fig)

        # ---- montage at render threshold ----
        overlay = SensorOverlay(baseline=None)
        for n_obj, off_f in ((overlay.norm_L, 0), (overlay.norm_R, 1)):
            n_obj.offset = offset[off_f].astype(np.float32)
            n_obj.global_scale_xy = float(sxy[off_f])
            n_obj.global_scale_z = float(sz[off_f])

        joint = z["data/joint_angles"]
        grip = z["data/grip_pos"]
        img0 = z["data/img_0"]
        rows = []
        for ep in eps:
            s, e = starts[ep], ends[ep]
            cells = []
            for lbl, idx in pick_moments(m[s:e], grasp[s:e], args.render_thr, on[s:e]):
                gi = s + idx
                base = cv2.resize(to_u8(img0[gi]), (NATIVE_W, NATIVE_H))
                feedL = binmarker_feed(norm[gi, 0], on[gi, 0])
                feedR = binmarker_feed(norm[gi, 1], on[gi, 1])
                drawn = overlay.draw("side", base, joint[gi].tolist(), float(np.ravel(grip[gi])[0]),
                                     feedL, feedR, mode="points1_arrow", is_spatial=True,
                                     arrow_length_scale=FIXED_ARROW_LEN,
                                     arrow_thickness=8, dot_size=22)
                cell = cv2.resize(drawn, (448, 336))
                bar = np.zeros((26, 448, 3), np.uint8)
                onstr = "L" * int(on[gi, 0]) + "R" * int(on[gi, 1])
                cv2.putText(bar, f"ep{ep} {lbl} mL={m[gi,0]:.2f} mR={m[gi,1]:.2f} on={onstr or '-'}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cells.append(np.vstack([bar, cell]))
            rows.append(np.hstack(cells))
        cv2.imwrite(f"{args.out}/{task}_montage_T{args.render_thr}.png", np.vstack(rows))
        print(f"{task}: wrote timeline + montage (T={args.render_thr})", flush=True)


if __name__ == "__main__":
    main()
