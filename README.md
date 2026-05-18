# tactile-data-collection

iPhone-teleoperated xArm data collection with two-finger A31301 tactile sensing.
Recording writes **raw frames only**; overlay-augmented images are rendered
post-hoc from the raw zarr (six sensordrawing variants per camera).

## Run

```bash
./collect_and_render.sh     # teleop + record + live viz, then render overlays on exit
```

In-session: **phone button** = start/stop episode (homes first), **Backspace** =
discard last episode, **Ctrl+C** = flush + quit.

Separately:
```bash
python collect_with_home.py --record --viz             # collect only
python scripts/render_overlays.py [src] [dst]          # render only (dst is wiped + rebuilt)
```

Flags: `--no-tactile`, `--no-viz-overlay`, `--safety-threshold N`, `--safety-metric {sum_abs_z,max_abs_z,max_norm}`, `--viz-mode {points9_arrow,points1_arrow,points1_contact_{spatial,flat},points9_color_{spatial,flat}}`.

## Output

**`teleop_data.zarr`** (raw, appended): `state (N,7)`, `joint_angles (N,7)`, `grip_pos (N,1)`, `img_0..K (N,224,224,3)`, `tactile (N,2,9,3)` + connected/ts/lag, `/meta/episode_ends`, `/meta/tactile_baseline`.

**`teleop_data_overlay.zarr`** (regenerated): raw fields verbatim + `img_{i}_{mode_key}` for each camera × each of the 6 sensordrawing variants.

## Tactile sensor setup (per board, from `customsensor/`)

1. Plug the board's USB-C into the workstation.
2. Find the port:
   ```bash
   ls /dev/tty*
   ```
   Usually `/dev/ttyACM0` (first board) and `/dev/ttyACM1` (second).
3. Grant access:
   ```bash
   sudo chmod 777 /dev/ttyACM0
   ```
4. On the board's onboard screen: select **sensor**, right-push the button; then select **stream**, right-push the button. The board now streams `S,ts,idx,addr,conn,x,y,z` rows over USB serial.
5. Repeat for the second board. By convention LEFT finger = `/dev/ttyACM0`, RIGHT = `/dev/ttyACM1` — match this to `tactile_config.LEFT_FINGER_PORT` / `RIGHT_FINGER_PORT` (or override with `--left-port` / `--right-port`).

To sanity-check streaming before running the full harness, use the standalone
`receive_a31301_stream.py` script in `customsensor/`.

## Dependencies

`numpy`, `scipy`, `opencv-python`, `zarr<3`, `pyserial`, `pyrealsense2`, `pynput`, `torch`, plus `teledex` (iPhone AR client) and `xarm` (UFactory SDK) — install both from their upstream repos.

## Where to look

- `collect_with_home.py` — main entry point
- `threads.py` — `PhoneReadThread` + `RecordingThread`
- `recorder.py` — `DatasetRecorder` (memory buffer + background flush)
- `environment/` — `env.py`, `xarm_controller.py` (safety wrapper), `tactile.py`, `tactile_overlay.py`, `sensordrawing/` (vendored geometry: K, T_rc, FK, calibrations)
- `tactile_config.py` — ports / baud / safety thresholds / camera serials (the only user-tunable file)
- `scripts/render_overlays.py` — raw → overlay zarr

See `CLAUDE.md` for deeper internals (threading model, safety semantics, full schema, gotchas).
