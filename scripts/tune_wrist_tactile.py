"""
Interactive tuner for the wrist tactile overlay anchor positions.

Opens the wrist camera and shows the live feed with the configured 3x3 grids
drawn on top. Click anywhere in the window to set the CENTER of the active
finger's grid to that pixel. Toggle which finger you're placing with the 'l'
or 'r' key; rotate the grid 90 deg with 't'; adjust cell spacing with '+'/'-'.
When the dots sit on the actual sensor PCBs, press 's' to print the
tactile_config.py values to paste in.

Controls:
  click       set CENTER of the active finger's 3x3 grid to that pixel
  l           switch active finger to LEFT  (cells drawn in green)
  r           switch active finger to RIGHT (cells drawn in blue)
  t           toggle WRIST_GRID_TRANSPOSED (rotate the 3x3 grid 90 deg)
  + or =      increase WRIST_CELL_PIX (spread cells apart)
  - or _      decrease WRIST_CELL_PIX
  s           print current values for pasting into tactile_config.py
  q or Esc    quit (also prints values on exit)

Notes:
  - The values printed under 'WRIST_*_TOP_LEFT_UV' are the TOP-LEFT-cell
    pixel positions, which is the format tactile_config.py uses. The script
    converts from your clicked CENTER to the top-left internally.
  - This script needs an X display (uses cv2.imshow). Won't work over plain
    SSH; use VNC or `ssh -X` if you must be remote.
  - Close any running collect_with_home.py / tune_safety.py first: only one
    process can hold the RealSense pipeline at a time.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyrealsense2 as rs  # noqa: E402

import tactile_config as tc  # noqa: E402


# Mutable state, edited interactively.
state = {
    "left_uv":    list(tc.WRIST_LEFT_TOP_LEFT_UV),   # top-left cell (matches config format)
    "right_uv":   list(tc.WRIST_RIGHT_TOP_LEFT_UV),
    "cell_pix":   tc.WRIST_CELL_PIX,
    "transposed": tc.WRIST_GRID_TRANSPOSED,
    "active":     "left",                            # which side gets the next click
}


def _grid_anchors(top_left_uv, cell_pix, transposed):
    """Return 9 (u, v) pixel anchors for a 3x3 grid given its top-left cell."""
    rows, cols = tc.CELL_GRID_SHAPE
    u0, v0 = top_left_uv
    anchors = np.zeros((9, 2), dtype=np.int32)
    for idx, (row, col) in enumerate(tc.IDX_TO_ROWCOL):
        if transposed:
            du, dv = row * cell_pix, col * cell_pix
        else:
            du, dv = col * cell_pix, row * cell_pix
        anchors[idx] = (u0 + du, v0 + dv)
    return anchors


def _click_center_to_top_left(click_uv, cell_pix, transposed):
    """Convert a CENTER click to the top-left (row=0, col=0) cell position."""
    rows, cols = tc.CELL_GRID_SHAPE
    cu, cv = click_uv
    if transposed:
        # rows -> u, cols -> v
        offset_u = (rows - 1) / 2.0 * cell_pix
        offset_v = (cols - 1) / 2.0 * cell_pix
    else:
        # cols -> u, rows -> v
        offset_u = (cols - 1) / 2.0 * cell_pix
        offset_v = (rows - 1) / 2.0 * cell_pix
    return (int(round(cu - offset_u)), int(round(cv - offset_v)))


def _print_save():
    print()
    print("=" * 64)
    print("Paste into tactile_config.py:")
    print("=" * 64)
    print(f"WRIST_LEFT_TOP_LEFT_UV  = {tuple(state['left_uv'])}")
    print(f"WRIST_RIGHT_TOP_LEFT_UV = {tuple(state['right_uv'])}")
    print(f"WRIST_CELL_PIX          = {state['cell_pix']}")
    print(f"WRIST_GRID_TRANSPOSED   = {state['transposed']}")
    print("=" * 64)


def _on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    top_left = _click_center_to_top_left(
        (x, y), state["cell_pix"], state["transposed"]
    )
    if state["active"] == "left":
        state["left_uv"] = list(top_left)
        print(f"  LEFT  center=({x},{y}) -> top-left={top_left}")
    elif state["active"] == "right":
        state["right_uv"] = list(top_left)
        print(f"  RIGHT center=({x},{y}) -> top-left={top_left}")


def main() -> int:
    print(f"Opening wrist camera (serial={tc.WRIST_CAMERA_SERIAL})...")
    cfg = rs.config()
    cfg.enable_device(tc.WRIST_CAMERA_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    try:
        pipeline.start(cfg)
    except Exception as e:
        print(f"ERROR: failed to open wrist camera: {e}")
        print(
            "Make sure the wrist camera is connected and no other script "
            "(collect_with_home.py / tune_safety.py) is currently holding it."
        )
        return 1

    print(__doc__)
    print(
        f"Initial values: L={tuple(state['left_uv'])}  R={tuple(state['right_uv'])}  "
        f"cell_pix={state['cell_pix']}  transposed={state['transposed']}"
    )

    window = "wrist tactile anchor tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            # Draw the two grids.
            L = _grid_anchors(state["left_uv"], state["cell_pix"], state["transposed"])
            R = _grid_anchors(state["right_uv"], state["cell_pix"], state["transposed"])
            for u, v in L:
                cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0), -1)  # green
            for u, v in R:
                cv2.circle(img, (int(u), int(v)), 4, (255, 0, 0), -1)  # blue

            # Mark the CENTER of each grid (where you actually clicked) with a cross.
            for side, color_bgr in (("left", (0, 255, 0)), ("right", (255, 0, 0))):
                anchors = L if side == "left" else R
                cx = int(round(np.mean(anchors[:, 0])))
                cy = int(round(np.mean(anchors[:, 1])))
                cv2.drawMarker(
                    img, (cx, cy), color_bgr,
                    markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1,
                )

            # HUD
            cv2.putText(img, f"active: {state['active'].upper()}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if state["active"] == "left" else (255, 0, 0), 2)
            cv2.putText(img, f"L top-left={tuple(state['left_uv'])}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, f"R top-left={tuple(state['right_uv'])}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img, f"cell_pix={state['cell_pix']}  transposed={state['transposed']}",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img,
                        "click=center  l/r=side  t=rotate  +/-=spacing  s=save  q=quit",
                        (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(window, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 0xFF:
                continue
            if k in (ord("q"), 27):
                break
            elif k == ord("l"):
                state["active"] = "left"
                print("  active: LEFT")
            elif k == ord("r"):
                state["active"] = "right"
                print("  active: RIGHT")
            elif k == ord("t"):
                state["transposed"] = not state["transposed"]
                print(f"  transposed: {state['transposed']}")
            elif k in (ord("+"), ord("=")):
                state["cell_pix"] = state["cell_pix"] + 2
                print(f"  cell_pix: {state['cell_pix']}")
            elif k in (ord("-"), ord("_")):
                state["cell_pix"] = max(2, state["cell_pix"] - 2)
                print(f"  cell_pix: {state['cell_pix']}")
            elif k == ord("s"):
                _print_save()
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        _print_save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
