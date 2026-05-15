"""
Re-render the shear-aware tactile overlay on already-recorded zarr data.

Pulls /data/img_*_raw (un-overlaid camera frames), /data/tactile (raw 9-cell
xyz per finger), /data/state (xArm pose) and /meta/tactile_baseline (per-cell
idle field) out of a recording, then redraws the overlay using the current
environment/tactile_overlay.py logic — sensor (Bx, By) drives the arrow
DIRECTION, sensor (|Bz|) drives the arrow MAGNITUDE.

No hardware required. Needs an X display (uses cv2.imshow) — for headless
SSH you'll need `ssh -X`, VNC, or run it locally.

Usage:
  python scripts/visualize_shear_overlay.py --data teleop_data.zarr
  python scripts/visualize_shear_overlay.py --data teleop_data.zarr --view both
  python scripts/visualize_shear_overlay.py --data teleop_data.zarr --mode grid

Controls (cv2 window must be focused):
  SPACE        play/pause
  Right / 'l'  +1 frame
  Left  / 'h'  -1 frame
  Down  / 'j'  next episode
  Up    / 'k'  prev episode
  m            cycle mode  (arrow -> grid -> point -> bar -> arrow ...)
  v            cycle view  (wrist -> agent -> both -> wrist ...)  if both stored
  +/-          playback speed up / down
  q / Esc      quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import zarr

# Make the parent package importable when run as `python scripts/visualize_shear_overlay.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tactile_config as tc                       # noqa: E402
from environment import tactile_overlay           # noqa: E402


# Stand-in for librealsense rs.intrinsics. The recorder doesn't save real
# intrinsics, so for the agent view we use typical Intel RealSense D435 color
# values at 640x480. Override via --intrinsics fx,fy,ppx,ppy if your camera
# differs and you care about exact projection.
class _Intrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

    def __repr__(self):
        return f"_Intrinsics(fx={self.fx}, fy={self.fy}, ppx={self.ppx}, ppy={self.ppy})"


# Recorded images are 224x224 (resized from native 640x480). The overlay
# anchors and pixel scales in tactile_config are tuned for native resolution,
# so we render at native res and resize at the end — same as what the live
# recorder used to produce burned-in overlays.
NATIVE_W, NATIVE_H = 640, 480
DISPLAY_W, DISPLAY_H = 640, 480   # display at native res for readability

MODES = ["arrow", "grid", "point", "bar"]
VIEWS = ["wrist", "agent", "both"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path):
    """Open zarr, return everything we need or raise with a useful message."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"zarr path not found: {path}")
    root = zarr.open(path, mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError(f"{path} doesn't look like a recording (no /data, /meta)")
    data = root["data"]
    meta = root["meta"]

    if "tactile" not in data:
        raise ValueError(
            f"{path}/data has no 'tactile' array — this script needs a recording "
            "made with --record AND tactile sensors enabled."
        )

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0:
        raise ValueError(f"{path} has no completed episodes")

    # Prefer the un-overlaid frames; fall back to the overlay-burned-in frames
    # with a warning (older recordings might not have *_raw).
    cam_keys = []
    using_raw = True
    for i in range(8):
        if f"img_{i}_raw" in data:
            cam_keys.append(f"img_{i}_raw")
        elif f"img_{i}" in data:
            cam_keys.append(f"img_{i}")
            using_raw = False
        else:
            break
    if not cam_keys:
        raise ValueError(f"{path}/data has no img_* arrays")
    if not using_raw:
        print(f"  [warn] {path} has no img_*_raw — using img_* (old overlay is "
              f"already burned in). Pass --record --no-save-raw-images=False next time.")

    state = data["state"]
    if state.shape[1] < 7:
        raise ValueError(f"state dim {state.shape[1]} < 7 — wrong dataset?")

    if "tactile_baseline" in meta:
        baseline = np.asarray(meta["tactile_baseline"][:], dtype=np.float32)
        baseline_source = "/meta/tactile_baseline"
    else:
        baseline = None
        baseline_source = None

    return {
        "root": root, "data": data, "meta": meta,
        "episode_ends": episode_ends,
        "cam_keys": cam_keys,
        "using_raw": using_raw,
        "state_arr": state,
        "tactile_arr": data["tactile"],
        "connected_arr": data["tactile_connected"] if "tactile_connected" in data else None,
        "baseline": baseline,
        "baseline_source": baseline_source,
    }


def _compute_baseline_from_first_frames(tactile_arr, n_samples=30):
    """Fallback baseline: average of the first n_samples frames.

    The robot has just homed and the gripper is open at script start, so
    these frames should be contact-free (matching what the live capture does).
    """
    n = min(int(n_samples), tactile_arr.shape[0])
    if n <= 0:
        return None
    sample = np.asarray(tactile_arr[:n], dtype=np.float32)   # (n, 2, 9, 3)
    return sample.mean(axis=0).astype(np.float32)            # (2, 9, 3)


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _to_native(img):
    """Upscale a stored 224x224 (float32 in [0,1] or uint8) frame to native res."""
    arr = np.asarray(img)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[:2] != (NATIVE_H, NATIVE_W):
        arr = cv2.resize(arr, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)
    return arr


def _render_wrist_overlay(wrist_img_native, tact_L, tact_R, conn_L, conn_R, mode):
    return tactile_overlay.draw_wrist_overlay(
        wrist_img_native, tact_L, tact_R, conn_L, conn_R, mode=mode,
    )


def _render_agent_overlay(agent_img_native, ee_pose, tact_L, tact_R, conn_L, conn_R,
                          trc, intrinsics, mode):
    if trc is None or intrinsics is None:
        # No extrinsics/intrinsics available — degrade gracefully to bar mode
        # (the only mode that doesn't need projection).
        if mode != "bar":
            return None
    return tactile_overlay.draw_agent_overlay(
        agent_img_native, ee_pose, tact_L, tact_R, conn_L, conn_R,
        trc, intrinsics, mode=mode,
    )


def _hud(img, lines, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Stamp a few HUD lines into the top-left of img (in place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20
    for line in lines:
        (tw, th), bl = cv2.getTextSize(line, font, 0.5, 1)
        cv2.rectangle(img, (4, y - th - 2), (4 + tw + 4, y + bl + 2), bg_color, -1)
        cv2.putText(img, line, (6, y), font, 0.5, text_color, 1, cv2.LINE_AA)
        y += th + bl + 6


def _composite(wrist_img, agent_img, view):
    """Build the displayed image given the requested view (and what's available)."""
    if view == "wrist":
        return wrist_img if wrist_img is not None else _placeholder("no wrist data")
    if view == "agent":
        return agent_img if agent_img is not None else _placeholder("no agent overlay\n(missing trc/intrinsics)")
    # both
    panels = []
    panels.append(wrist_img if wrist_img is not None else _placeholder("no wrist"))
    panels.append(agent_img if agent_img is not None else _placeholder("no agent"))
    return np.hstack(panels)


def _placeholder(text):
    img = np.full((DISPLAY_H, DISPLAY_W, 3), 32, dtype=np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (12, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Interactive playback
# ---------------------------------------------------------------------------

def play(ds, agent_idx, wrist_idx, trc, intrinsics, mode_init, view_init, fps_init):
    episode_ends = ds["episode_ends"]
    starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)

    cam_keys = ds["cam_keys"]
    state_arr = ds["state_arr"]
    tactile_arr = ds["tactile_arr"]
    connected_arr = ds["connected_arr"]

    # Resolve which img_*_raw column is which camera. Default: img_0=agent,
    # img_1=wrist (matches the recorder's RecordingThread._record_one_tick).
    if agent_idx >= len(cam_keys) and wrist_idx >= len(cam_keys):
        raise ValueError(f"--agent-idx / --wrist-idx out of range; only {len(cam_keys)} img_* arrays")
    agent_arr = ds["data"][cam_keys[agent_idx]] if agent_idx < len(cam_keys) else None
    wrist_arr = ds["data"][cam_keys[wrist_idx]] if wrist_idx < len(cam_keys) else None

    print()
    print(f"  Dataset           : {ds['root'].store.path if hasattr(ds['root'].store, 'path') else '?'}")
    print(f"  Episodes          : {len(episode_ends)}  ({state_arr.shape[0]} total frames)")
    print(f"  State dim         : {state_arr.shape[1]}")
    print(f"  Image keys        : {cam_keys}  (using_raw={ds['using_raw']})")
    print(f"  Agent column      : {cam_keys[agent_idx] if agent_idx < len(cam_keys) else 'none'}")
    print(f"  Wrist column      : {cam_keys[wrist_idx] if wrist_idx < len(cam_keys) else 'none'}")
    print(f"  Tactile shape     : {tactile_arr.shape}")
    print(f"  Baseline source   : {ds['baseline_source'] or 'computed from first 30 frames'}")
    print(f"  Intrinsics (agent): {intrinsics}")
    print(f"  Initial mode/view : {mode_init} / {view_init}")
    print()
    print("  Controls:  SPACE play/pause  Left/Right frame  Up/Down episode")
    print("             m=mode  v=view  +/-=speed  q/Esc=quit")
    print()

    ep_idx = 0
    frame_offset = 0          # within episode
    playing = False
    mode = mode_init
    view = view_init
    fps = float(fps_init)
    last_step_time = time.monotonic()

    win = "shear-aware tactile overlay (recorded)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        ep_start = int(starts[ep_idx])
        ep_end   = int(episode_ends[ep_idx])
        ep_len   = ep_end - ep_start
        if ep_len <= 0:
            print(f"  [warn] episode {ep_idx} is empty; skipping")
            ep_idx = (ep_idx + 1) % len(episode_ends)
            frame_offset = 0
            continue
        frame_offset = max(0, min(frame_offset, ep_len - 1))
        global_idx = ep_start + frame_offset

        # ---- pull the frame's payload from zarr ----
        tact = np.asarray(tactile_arr[global_idx], dtype=np.float32)   # (2, 9, 3)
        tact_L, tact_R = tact[0], tact[1]
        if connected_arr is not None:
            conn = np.asarray(connected_arr[global_idx], dtype=np.uint8)
        else:
            conn = np.ones((2, 9), dtype=np.uint8)
        conn_L, conn_R = conn[0], conn[1]
        ee_pose = np.asarray(state_arr[global_idx, :6], dtype=np.float64)

        # ---- render wrist + (optional) agent ----
        wrist_native = _to_native(wrist_arr[global_idx]) if wrist_arr is not None else None
        agent_native = _to_native(agent_arr[global_idx]) if agent_arr is not None else None

        wrist_overlay = (
            _render_wrist_overlay(wrist_native, tact_L, tact_R, conn_L, conn_R, mode)
            if wrist_native is not None else None
        )
        agent_overlay = (
            _render_agent_overlay(agent_native, ee_pose,
                                  tact_L, tact_R, conn_L, conn_R,
                                  trc, intrinsics, mode)
            if agent_native is not None else None
        )

        # ---- HUD: per-finger stats so the user can sanity-check ----
        shear_L, normal_L = tactile_overlay.shear_and_normal(tact_L, "left")
        shear_R, normal_R = tactile_overlay.shear_and_normal(tact_R, "right")
        max_shear_L = float(np.linalg.norm(shear_L, axis=1).max())
        max_shear_R = float(np.linalg.norm(shear_R, axis=1).max())
        max_norm_L = float(np.abs(normal_L).max())
        max_norm_R = float(np.abs(normal_R).max())

        hud_lines = [
            f"ep {ep_idx + 1}/{len(episode_ends)}  frame {frame_offset + 1}/{ep_len}  "
            f"global {global_idx}",
            f"mode={mode}  view={view}  fps={fps:.1f}  {'PLAY' if playing else 'PAUSE'}",
            f"L  max|shear|={max_shear_L:6.0f}  max|normal|={max_norm_L:6.0f}",
            f"R  max|shear|={max_shear_R:6.0f}  max|normal|={max_norm_R:6.0f}",
        ]
        for img in (wrist_overlay, agent_overlay):
            if img is not None:
                _hud(img, hud_lines)

        composite = _composite(wrist_overlay, agent_overlay, view)
        cv2.imshow(win, composite)

        # Key handling. waitKey returns 0xFF if no key pressed within the wait
        # window; we always poll briefly even when paused so the UI stays live.
        wait_ms = int(max(1, 1000.0 / fps)) if playing else 30
        k = cv2.waitKey(wait_ms) & 0xFF
        if k == 0xFF:
            pass
        elif k in (ord("q"), 27):
            break
        elif k == ord(" "):
            playing = not playing
        elif k in (ord("l"), 83, 3):       # right
            playing = False
            frame_offset = min(ep_len - 1, frame_offset + 1)
        elif k in (ord("h"), 81, 2):       # left
            playing = False
            frame_offset = max(0, frame_offset - 1)
        elif k in (ord("j"), 84, 1):       # down -> next episode
            playing = False
            ep_idx = (ep_idx + 1) % len(episode_ends)
            frame_offset = 0
        elif k in (ord("k"), 82, 0):       # up   -> prev episode
            playing = False
            ep_idx = (ep_idx - 1) % len(episode_ends)
            frame_offset = 0
        elif k == ord("m"):
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            print(f"  mode -> {mode}")
        elif k == ord("v"):
            view = VIEWS[(VIEWS.index(view) + 1) % len(VIEWS)]
            print(f"  view -> {view}")
        elif k in (ord("+"), ord("=")):
            fps = min(120.0, fps * 1.25)
        elif k in (ord("-"), ord("_")):
            fps = max(1.0, fps / 1.25)

        if playing:
            now = time.monotonic()
            if now - last_step_time >= 1.0 / fps:
                frame_offset += 1
                if frame_offset >= ep_len:
                    frame_offset = 0
                    ep_idx = (ep_idx + 1) % len(episode_ends)
                last_step_time = now

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default="teleop_data.zarr",
                    help="Path to the zarr recording (default: teleop_data.zarr)")
    ap.add_argument("--mode", choices=MODES, default="arrow",
                    help="Initial visualization mode")
    ap.add_argument("--view", choices=VIEWS, default="both",
                    help="Initial view (wrist / agent / both side-by-side)")
    ap.add_argument("--fps", type=float, default=10.0, help="Initial playback fps")
    ap.add_argument("--agent-idx", type=int, default=0,
                    help="Index into img_* arrays for the AGENT camera (default 0)")
    ap.add_argument("--wrist-idx", type=int, default=1,
                    help="Index into img_* arrays for the WRIST camera (default 1)")
    ap.add_argument("--transforms", default=None,
                    help="Path to transforms_agent.npz (default: <repo>/transforms_agent.npz). "
                         "If missing, agent-view modes that need projection are disabled.")
    ap.add_argument("--intrinsics", default=None,
                    help="Override agent-camera intrinsics as 'fx,fy,ppx,ppy' "
                         "(default: D435 typical at 640x480: 606,606,320,240)")
    ap.add_argument("--shear-deadband", type=float, default=None,
                    help="Override tactile_config.SHEAR_DEADBAND for this run. "
                         "Bigger = steadier arrow direction (more shear "
                         "required to register). 0 disables.")
    args = ap.parse_args()

    if args.shear_deadband is not None:
        tc.SHEAR_DEADBAND = float(args.shear_deadband)
        print(f"  [info] SHEAR_DEADBAND override: {tc.SHEAR_DEADBAND}")

    ds = _load_dataset(args.data)

    # Install baseline (preferred: from /meta; fallback: averaged from first frames).
    baseline = ds["baseline"]
    if baseline is None:
        print("  [info] no /meta/tactile_baseline; computing from first 30 frames "
              "(assumes the gripper is open and untouched at start)")
        baseline = _compute_baseline_from_first_frames(ds["tactile_arr"], n_samples=30)
        ds["baseline_source"] = "computed (first 30 frames)"
    if baseline is not None:
        tactile_overlay.set_baseline(baseline)
    else:
        print("  [warn] couldn't establish baseline — overlay will use raw values")

    # Agent-camera transform + intrinsics (only used by 'agent'/'both' views in
    # arrow/grid/point modes; bar mode never needs them).
    trc = None
    transforms_path = args.transforms or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "transforms_agent.npz",
    )
    if os.path.isfile(transforms_path):
        try:
            d = np.load(transforms_path, allow_pickle=False)
            trc = np.asarray(d["trc"], dtype=np.float64)
            print(f"  [info] loaded agent extrinsics from {transforms_path}")
        except Exception as e:
            print(f"  [warn] couldn't load {transforms_path}: {e}; agent overlay disabled")
    else:
        print(f"  [warn] {transforms_path} not found; agent overlay disabled")

    if args.intrinsics:
        try:
            fx, fy, ppx, ppy = [float(x) for x in args.intrinsics.split(",")]
            intrinsics = _Intrinsics(fx, fy, ppx, ppy)
        except Exception as e:
            print(f"  [warn] couldn't parse --intrinsics: {e}; using D435 defaults")
            intrinsics = _Intrinsics(606.0, 606.0, 320.0, 240.0)
    else:
        intrinsics = _Intrinsics(606.0, 606.0, 320.0, 240.0)

    play(ds, args.agent_idx, args.wrist_idx, trc, intrinsics,
         mode_init=args.mode, view_init=args.view, fps_init=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
