"""Quick visual check: did the camera rig move since data collection?

Reads a frame from a training zarr and blends it with the live RealSense feed.
If the rig is still aligned, static scene elements (table edges, mounts, robot
home pose) line up between training and live frames; misalignment means the
camera moved.

Two windows pop up — one per camera (agent / wrist). Each window shows three
panels side-by-side: training frame | live frame | 50/50 blend.

Usage
-----
    python scripts/check_camera_drift.py                          # uses teleop_data.zarr, frame 0
    python scripts/check_camera_drift.py --zarr teleop_data_cube.zarr --frame 5
    python scripts/check_camera_drift.py --episode 3 --alpha 0.6

Keys (focus a window first):
    q  quit
    s  save current blended frames to drift_snapshot_<task>_<ts>.png
    r  reload the training frame from disk (in case --zarr was overwritten)
    +/-  bump blend alpha (training weight) by 0.05
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import zarr

# Read the canonical camera-serial-to-role mapping from the same module the
# collection pipeline uses, so this script can't drift away from how img_0 /
# img_1 were assigned at recording time.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tactile_config import AGENT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL  # noqa: E402

import pyrealsense2 as rs  # noqa: E402  (imported after sys.path tweak)

# Recorded frames live at 224x224 (recorder.py resize); blend at 640x480 for
# better visibility, matching what cameras.py captures live.
DISPLAY_W, DISPLAY_H = 640, 480


def load_training_frames(zarr_path: str, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (agent_bgr_uint8, wrist_bgr_uint8), each (H, W, 3) at 640x480.

    Training imgs are stored BGR (recorder.py just divides by 255 without color
    conversion; raw cv2/RealSense capture order is BGR). The overlay zarr
    stores either uint8 [0,255] or float32 [0,1]; we handle both.
    """
    root = zarr.open(zarr_path, mode="r")
    if "data" not in root or "img_0" not in root["data"] or "img_1" not in root["data"]:
        raise SystemExit(f"[error] {zarr_path} missing /data/img_0 or /data/img_1")
    n = int(root["data/img_0"].shape[0])
    if frame_idx < 0 or frame_idx >= n:
        raise SystemExit(f"[error] frame {frame_idx} out of range (zarr has {n} frames)")

    def _read(key):
        arr = root[f"data/{key}"][frame_idx]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.resize(arr, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_CUBIC)

    return _read("img_0"), _read("img_1")


def open_realsense(target_serial: str) -> rs.pipeline:
    """Open a single RealSense color stream at 640x480 BGR @ 30 fps to match
    environment/cameras.py — no manual exposure / WB overrides so this script
    behaves like the actual collection pipeline."""
    ctx = rs.context()
    attached = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    if target_serial not in attached:
        raise SystemExit(
            f"[error] RealSense serial {target_serial} not attached. "
            f"Found: {attached}. Are the cameras plugged in / not held by another process?"
        )
    cfg = rs.config()
    cfg.enable_device(target_serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe = rs.pipeline()
    pipe.start(cfg)
    # Brief warm-up so AE/AWB settle.
    for _ in range(15):
        pipe.wait_for_frames()
    return pipe


def grab_live(pipe: rs.pipeline) -> np.ndarray | None:
    frames = pipe.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        return None
    img = np.asanyarray(color.get_data())
    if img.shape[:2] != (DISPLAY_H, DISPLAY_W):
        img = cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
    return img


def make_panel(training: np.ndarray, live: np.ndarray, alpha: float, label: str) -> np.ndarray:
    blend = cv2.addWeighted(training, alpha, live, 1.0 - alpha, 0.0)

    def _annot(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (DISPLAY_W, 28), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    panel = np.concatenate([
        _annot(training, f"{label} training"),
        _annot(live, f"{label} live"),
        _annot(blend, f"{label} blend a={alpha:.2f}"),
    ], axis=1)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr", type=str, default="teleop_data.zarr",
                    help="Training zarr to compare against (default: teleop_data.zarr).")
    ap.add_argument("--frame", type=int, default=0,
                    help="Index into /data/img_{0,1}. Default 0 (first recorded frame, "
                         "usually robot at home pose).")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Initial blend weight for the training frame (1.0 = only "
                         "training, 0.0 = only live). Adjust live with +/-.")
    args = ap.parse_args()

    print(f"Loading training frame {args.frame} from {args.zarr}...")
    train_agent, train_wrist = load_training_frames(args.zarr, args.frame)

    print(f"Opening RealSense  agent={AGENT_CAMERA_SERIAL}  wrist={WRIST_CAMERA_SERIAL} ...")
    agent_pipe = open_realsense(AGENT_CAMERA_SERIAL)
    wrist_pipe = open_realsense(WRIST_CAMERA_SERIAL)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    print("Controls:  q quit   s save snapshot   r reload training frame   +/- alpha")

    try:
        while True:
            live_agent = grab_live(agent_pipe)
            live_wrist = grab_live(wrist_pipe)
            if live_agent is None or live_wrist is None:
                continue
            agent_panel = make_panel(train_agent, live_agent, alpha, "AGENT")
            wrist_panel = make_panel(train_wrist, live_wrist, alpha, "WRIST")
            stacked = np.concatenate([agent_panel, wrist_panel], axis=0)
            cv2.imshow("camera-drift check (agent above, wrist below)", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = f"drift_snapshot_{stamp}.png"
                cv2.imwrite(path, stacked)
                print(f"  saved {path}")
            if key == ord("r"):
                train_agent, train_wrist = load_training_frames(args.zarr, args.frame)
                print(f"  reloaded training frame {args.frame}")
            if key in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
                print(f"  alpha={alpha:.2f}")
            if key in (ord("-"), ord("_")):
                alpha = max(0.0, alpha - 0.05)
                print(f"  alpha={alpha:.2f}")
    finally:
        cv2.destroyAllWindows()
        for pipe in (agent_pipe, wrist_pipe):
            try:
                pipe.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
