# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teleoperation + data-collection harness for a UFactory xArm robot, controlled by an iPhone over `teledex` (ARKit pose streamed to the workstation). Optional two-finger tactile sensing (A31301 boards over USB serial) provides (a) a gripper-safety wrapper that prevents the gripper from closing further when contact exceeds threshold, and (b) a shear-aware force overlay (arrow/grid/point/bar modes) — the overlay is drawn on-screen during teleop for operator feedback, and (separately) rendered post-hoc onto recorded frames by `scripts/render_overlays.py`. **Recorded frames are always RAW** (no overlay burned in); the overlay-augmented dataset lives in a separate zarr that is regenerated on demand from the raw one. Episodes get logged shaped for behavior-cloning / diffusion-policy training (see `XArmDataset` in `dataset.py`). There is no test suite, no linter, no build step — this is a script-driven Python project. Hardware required to actually run anything: an xArm at the configured IP, Intel RealSense cameras, an iPhone running the teledex client, and (optionally) two A31301 ESP32 boards.

## Running things

`collect_and_render.sh` is the convenience wrapper for a full session: it runs `collect_with_home.py --record --viz` (forwarding any extra args), and then **always** runs `scripts/render_overlays.py` to regenerate `teleop_data_overlay.zarr` from the (newly-grown) `teleop_data.zarr`. The render step runs even if collect exited non-zero (e.g. Ctrl+C), because Ctrl+C still produces good data. Use this as the normal entry point:

```bash
./collect_and_render.sh                           # standard session
./collect_and_render.sh --safety-threshold 1800   # override + forward flags
RAW_ZARR=foo.zarr OVERLAY_ZARR=foo_overlay.zarr ./collect_and_render.sh
```

`collect_with_home.py` is the data-collection entry point on its own:

```bash
# Teleop only, no recording, no tactile.
python collect_with_home.py --no-tactile

# Teleop + tactile safety + recording (writes teleop_data.zarr in the cwd).
python collect_with_home.py --record

# Recording + live --viz overlay windows (recorded zarr stays raw).
python collect_with_home.py --record --viz

# Disable the live overlay computation (raw cameras in the viz windows).
python collect_with_home.py --record --viz --no-viz-overlay
```

`scripts/render_overlays.py` is the overlay-augmentation entry point. It reads the raw zarr, renders **all four overlay modes** (`arrow`/`grid`/`point`/`bar`) per camera into a fresh `teleop_data_overlay.zarr`, and copies state/action/tactile/episode_ends/baseline through verbatim. The destination is wiped + regenerated on every run, so re-running picks up any newly-recorded episodes appended to the raw zarr:

```bash
python scripts/render_overlays.py                     # teleop_data.zarr -> teleop_data_overlay.zarr
python scripts/render_overlays.py raw.zarr aug.zarr   # explicit paths
```

Recording control during a session:
- Press the **phone button** to start/stop an episode (3-second cooldown enforced). Starting an episode first homes the robot and re-anchors the phone's AR frame to the freshly homed pose, then begins recording. There's a `grasp_open_latch` that forces grasp=0 after each home until the phone toggle drops below 0.5, so a held-on toggle doesn't immediately re-close the gripper.
- Press **Backspace** in the terminal (DEL `\x7f` or BS `\x08`) to discard the most recently completed episode — calls `recorder.discard_last_episode()`, which flushes pending writes, truncates every `/data/*` array back to the previous episode's end, and pops the trailing `/meta/episode_ends` entry. Refuses if recording is still in progress. Single-key input goes through termios cbreak mode, so it works over SSH.
- Press **Ctrl+C** in the terminal to quit (flushes outstanding zarr writes via `recorder.close()`). `q`-key quitting is intentionally disabled — `pynput` needs an X display and breaks over SSH.

Useful flags (`collect_with_home.py`):
- `--record` — enable zarr recording (off by default). Recorded images are always RAW; the overlay is rendered post-hoc.
- `--reset-duration SECS` — length of the smooth home motion (default 3.0).
- `--no-tactile` — skip tactile entirely (no safety wrapper, no live overlay, no tactile zarr columns). Without tactile, `render_overlays.py` has nothing to draw and exits with an error.
- `--left-port`, `--right-port`, `--tactile-baud` — override the A31301 serial defaults from `tactile_config.py`.
- `--safety-threshold FLOAT`, `--safety-metric {max_abs_z,max_norm,sum_abs_z}` — tune the gripper-safety trip point. Threshold units depend on whether a baseline was captured (delta-from-idle vs raw counts); see Tactile pipeline below.
- `--viz` — show live agent + wrist overlay windows at ~15 Hz (needs a DISPLAY; auto-disables on first cv2 error so the teleop loop keeps running). Works without `--record`. Does not affect what gets written to the zarr.
- `--viz-mode {arrow,grid,point,bar}` — what the LIVE windows draw (default `arrow`). Does not affect the recorded zarr (always raw) and does not affect `render_overlays.py` (which always renders all four modes).
- `--no-viz-overlay` — turn off the live overlay computation entirely. The `--viz` windows (if on) then just show the raw camera feeds. Tactile safety still applies.

Dataset utilities (run from repo root; most have a `DATA_PATH` constant hardcoded at the top of the file — edit it or pass `--data`):

```bash
python scripts/count_data.py --data teleop_data.zarr
python scripts/dataset_viewer.py          # browse / delete episodes
python scripts/create_2d_dataset.py       # strip state to xy only
python scripts/combine_zarrs.py           # merge two zarrs (paths hardcoded)
python scripts/repack_zarr.py <in.zarr>   # re-chunk + zstd compress
python scripts/render_overlays.py [src] [dst]   # raw -> overlay-augmented, all 4 modes (run by collect_and_render.sh)
```

`scripts/model_visualizer.py` imports a `policies.FlowMatchingPolicy` module that is **not in this repo** — policy training/inference lives elsewhere; this repo is the data-collection half of that workflow.

## Architecture

### Runtime topology (two threads + tactile threads + main loop)

`collect_with_home.py` wires together concurrent producers/consumers with the main thread as the action dispatcher:

1. **`PhoneReadThread`** (`threads.py`) — tight loop polling `Phone.get_target_pose()` / `get_grasp_state()` / `get_button_state()` (plus `get_button_secondary_state()` — TeleDex's second on-screen button, currently captured but not consumed by `collect_with_home.py`); exposes the latest values behind a lock. Reads are at ~1kHz; the main loop samples them.
2. **`RecordingThread`** (`threads.py`) — fixed-frequency (default 10 Hz) sampler. Per tick it pulls a fresh `env.get_obs()` (including camera frames), optionally pulls per-board tactile state via `TactileSensors.get_latest()`, and — on a **separate copy** of each camera frame — draws the live tactile overlay for the `--viz` windows. The recorded frames are NEVER overlaid: the recorder receives the resized 224×224 RAW frame as `imgs=...`. When NOT recording, `_compute_tick()` still runs each tick to keep the cached `get_latest_viz()` frames fresh. State written each tick is `[pose(6), grasp(1)]` → 7-dim. Obs is sampled **independently** of the main control loop — pose written to disk is the pose at recording-sample time, not at action-issue time. There's a post-`_compute_tick` race guard that drops the frame if `set_recording(False)` fired while the tick was in flight, so frames never land after `end_episode()`.
3. **`TactileSensor`** threads (`environment/tactile.py`) — one daemon thread per serial port, owns the serial connection, parses `S,ts,idx,addr,conn,x,y,z` rows, accumulates a 9-taxel frame keyed on shared `ts_ms`, and publishes the latest complete frame under a lock. Handles ESP32 reboots and serial-open failures gracefully. Bundled into a single `TactileSensors` context manager which is the only thing the rest of the code sees.
4. **Main loop** — reads latest phone pose, on button rising-edge toggles recording (with the homing dance), calls `env.step(target_pose, grasp_state)`.

The teleop math (mapping phone delta-pose → robot delta-pose, with a 1200× position gain and ±500mm clip) lives in `Phone.get_target_pose()` in `environment/phone.py`. `Phone.reset(initial_robot_pose)` must be called once after the robot reaches its home pose — it captures the phone's frame as the origin for subsequent deltas, **and it must be re-called every time the robot is re-homed** otherwise the next dispatched pose will snap the arm back to its pre-home target.

### Task layer

`tasks/{simple,pushT,soccer}_task.py` are thin wrappers around `XArmEnvironment` that fix a reset position and (for pushT/soccer) lock Z to a constant so the user only commands XY. All three accept an optional `tactile=TactileSensors` and forward it to the env. `collect_with_home.py` uses `Simple_Task`; to swap tasks, edit that line — there is no CLI flag.

### Environment layer

`environment/env.py` (`XArmEnvironment`) is the user-facing API: `step(grasp, target_pose=...)`, `get_obs()`, `reset(duration)`, `go_to_position(...)`. It owns a list of `Camera` objects (`environment/cameras.py`, RealSense via `pyrealsense2`) and a `XArm` instance. The `tactile=` kwarg is forwarded to `XArm` so the gripper-safety wrapper sees the live sensor state.

`environment/xarm_controller.py` is a single synchronous `XArm` wrapper around `XArmAPI`. Adapted from `tactile-ril-env/ril_env/xarm_controller.py` (the legacy class), with:
- Continuous grasp mapping in `[0, 1]` via `_apply_grasp` (epsilon-gated to avoid spamming the gripper API). `grasp=0.0` → `gripper_open_pos` (850), `grasp=1.0` → `gripper_close_pos` (0).
- Optional `tactile: TactileSensors` hook. Every `step_abs()` calls `tactile.safety() -> (metric, is_safe)`; when `is_safe=False` and the requested grasp would close further (`grasp > previous_grasp`), the command is clamped to `previous_grasp`. Opening is always allowed. Stale tactile readings count as unsafe (fail-safe).
- `step_abs(new_position, new_orientation, grasp)` is the only stepping method. Delta-based motion is the env's job (see `go_to_position`'s cosine interpolation).

There is no multiprocessing controller in this repo — the codebase is single-process by design. If you need shared-memory IPC, see the sibling `tactile-ril-env/` repo.

Workspace bounds are hardcoded in `XArmEnvironment.__init__`: `[[250,800],[-600,600],[10,380]]` mm (z-min progression per the inline comments in `env.py`: originally 50, then 30, now 10). Robot IP is hardcoded in `XArmConfig` (`192.168.1.223`). xArm is run in servo mode (`set_mode(1)`); homing temporarily flips to mode 0 then back.

### Tactile pipeline (`environment/tactile.py`)

A single module covers:
- **`TactileConfig`** — dataclass with ports, baud, taxel count, `safety_metric` (`max_abs_z` / `max_norm` / `sum_abs_z`), `safety_threshold`, `stale_after_sec`, `lag_warning_ms`, optional device-side `CMD,SET,UNITS,...` / `CMD,SET,RATE_HZ,...` overrides, and an optional `baseline` field of shape `(n_sensors, n_taxels, 3)`. When `baseline` is set, the safety metric reduces over `xyz - baseline` instead of raw `xyz` — i.e. it becomes a "delta from idle" quantity, and `safety_threshold` is interpreted in the same units.
- **`TactileSensor(threading.Thread)`** — one per board. Owns the `pyserial.Serial`, parses frames, handles ESP32 reboots and re-syncs on `BEGIN_STREAM`. Publishes the latest frame as a dict `{xyz: (9,3), connected: (9,), device_ts_ms, host_timestamp}` under a lock. `open_failed` is set when the port can't be opened so callers can degrade gracefully.
- **`TactileSensors`** — context-managed bundle of N `TactileSensor`s. Exposes `.get_latest()` → list of dicts, `.safety()` → `(metric, is_safe_to_close)`, plus `.all_open` / `.any_open` so the caller can gracefully degrade if a port fails.
- **`evaluate_safety` / `compute_safety_metric`** — pure-Python helpers used by `XArm._apply_tactile_safety`. Both treat stale or zero-connected-taxel data as unsafe.

**Startup baseline capture.** `collect_with_home.py` samples ~1.5 s of frames while the gripper is open at home (`_capture_tactile_baseline`), averages them per cell, and installs the resulting `(2, 9, 3)` baseline in three places: `tactile_overlay.set_baseline(...)` (arrows), `tactile.config.baseline = ...` (safety wrapper), and once into `/meta/tactile_baseline` (so downstream code can compute deltas off the raw `/data/tactile` later). Captured unconditionally when tactile is on — even when `--no-overlay` is passed — because the safety wrapper depends on it. Default `SAFETY_METRIC` is `sum_abs_z` and default `SAFETY_THRESHOLD` is `1500.0` in delta-from-idle units; `max_abs_z` does NOT work for this sensor mounting (see comment in `tactile_config.py:24-40`).

Defaults for ports, baud, lag-warning threshold, safety metric, and safety threshold live in `tactile_config.py` and are exposed as CLI flags on `collect_with_home.py`. Overlay-specific constants (cell geometry, per-finger force sign, sensor→image-plane 2×2 mappings, wrist anchor pixels, deadband, arrow style, camera-serial → role mapping, path to `transforms_agent.npz`) also live in `tactile_config.py`.

`environment/tactile_overlay.py` is **shear-aware**: per-cell SHEAR `(Bx, By) - baseline` drives the arrow DIRECTION in the image plane, and per-cell NORMAL `sign · (Bz - baseline)` drives the arrow LENGTH. The per-finger sign is `LEFT_FORCE_SIGN` / `RIGHT_FORCE_SIGN` so both fingers produce "positive when squeezed". A `SHEAR_DEADBAND` (default 50 counts) kills the noise-driven direction flips that result from normalizing near-zero shear.

Four visualization modes (`--viz-mode`, default `arrow`):
- **`arrow`** — one arrow per finger at the projected grid center. Direction = normal-weighted sum of per-cell projected shear; length = aggregate `|normal|` reduced by `AGGREGATE_METHOD` (`max` / `sum` / `mean`).
- **`grid`** — nine arrows per finger, one per cell at its projected pixel position.
- **`point`** — solid circle per finger at the grid center, radius = aggregate `|normal|`.
- **`bar`** — two horizontal bars at the bottom of the image; each lights up in its per-side color only when its finger's aggregate exceeds `BAR_TRIP_THRESHOLD`.

Agent (third-person) camera: 9 cell positions per finger are computed in EE frame (`cell_positions_ee_frame`), projected through the current robot pose × `trc` agent-camera extrinsics × intrinsics; for direction, the per-cell sensor-frame shear `(sx, sy)` is rotated by `AGENT_SHEAR_{LEFT,RIGHT}_FROM_SENSOR`, embedded as a 3D vector in EE frame (sensor.x → finger-width axis, sensor.y → finger-length axis, closing axis = 0), and the image-plane direction comes from `project(cell + step) - project(cell)` so arrows rotate correctly with the EE. Wrist camera: 18 fixed pixel anchors (9 per finger) from `WRIST_{LEFT,RIGHT}_TOP_LEFT_UV` + `WRIST_CELL_PIX`, optionally transposed via `WRIST_GRID_TRANSPOSED`; sensor-frame shear maps to image-plane via `WRIST_SHEAR_{LEFT,RIGHT}_UV_FROM_SENSOR`. Disconnected cells render as gray dots. The overlay is alpha-blended onto the raw frame at `ARROW_ALPHA` (default 0.75). The wrist anchor positions and arrow-length / thickness constants in `tactile_config.py` are calibrated for **native (640×480) pixels**; both the live `--viz` path and the post-hoc renderer draw on native-size frames and downsize to 224×224 afterwards. A legacy single-axis `squeeze_magnitudes()` helper is kept for external callers but is not used by the current overlay.

### Dataset layer

There are two distinct on-disk shapes:

**Raw zarr** (`teleop_data.zarr`, written by `recorder.py` during teleop, appended to across sessions):
```
/data
  state              (N, 7)       float32   [x,y,z,rx,ry,rz,grasp] — xArm euler degrees
  n_contacts         (N, 1)       float32   placeholder (always 0 in current recorder)
  img_0..K-1         (N,H,W,3)    float32   per-camera frames, 224×224, [0,1] — RAW (no overlay)
  action             (N, A)       float32   only present when recorder.use_actions=True (default False)
  tactile            (N, 2, 9, 3) float32   only when use_tactile=True — per-finger (Bx,By,Bz) per taxel, RAW (no baseline subtraction)
  tactile_connected  (N, 2, 9)    uint8     0/1 per taxel as reported by firmware
  tactile_ts_ms      (N, 2)       int64     device-side ms timestamp per board
  tactile_lag_ms     (N, 2)       float32   recording-time staleness per board
/meta
  episode_ends             (E,)               int64     cumulative end indices into /data
  tactile_baseline         (2, 9, 3)          float32   one-time idle field captured at startup
  camera_serials           (K,)               |S64      RealSense serial number per camera index
  camera_intrinsics_native (K, 4)             float32   per-camera [fx, fy, ppx, ppy] in NATIVE pixel coords
  camera_native_size       (K, 2)             int32     per-camera [width, height] in native pixels
  agent_camera_serial      ()                 |S64      which serial is the agent camera
  wrist_camera_serial      ()                 |S64      which serial is the wrist camera
  trc_agent                (3, 4)             float64   robot -> agent-camera extrinsic, from transforms_agent.npz
  recorded_img_size        (2,)               int32     [width, height] images were resized to (currently [224, 224])
```

**Overlay-augmented zarr** (`teleop_data_overlay.zarr`, regenerated by `scripts/render_overlays.py` from the raw zarr — wiped + rebuilt every run):
```
/data
  state, n_contacts, action, tactile*, img_0..K-1   ← copied verbatim from the raw zarr
  img_{i}_arrow      (N,H,W,3)    float32   one arrow per finger
  img_{i}_grid       (N,H,W,3)    float32   nine arrows per finger
  img_{i}_point      (N,H,W,3)    float32   single circle per finger
  img_{i}_bar        (N,H,W,3)    float32   binary bottom-bars per finger
/meta                                       ← copied verbatim from the raw zarr
```

Renderer pipeline: each recorded 224×224 frame is upscaled to the camera's native size (read from `/meta/camera_native_size`), the overlay is drawn with the existing native-pixel constants in `tactile_config.py` (wrist anchors, arrow-length cap, thickness), and the result is downsized back to 224×224. This matches the visual result of the original "draw-then-resize" recorder pipeline exactly. The agent overlay needs `/meta/trc_agent` and the agent-camera intrinsics; if either is missing, the agent column passes through raw and a warning is printed (the wrist column still renders).

`DatasetRecorder` (`recorder.py`) buffers in memory (`deque`s) and flushes to zarr every `flush_interval` seconds (default 2.0 s in `collect_with_home.py`) via a background thread; it also force-flushes when the buffer hits 80% capacity. **It divides camera images by 255 inside `append()`** — do not normalize again upstream. State dim and number of cameras are inferred on the first `append()` call from the data shapes. When `use_tactile=True`, the recorder requires all four `tactile_*` kwargs on every `append()` and fails loudly if any are missing (so that a forgotten kwarg can't silently misalign rows). Camera metadata (serials, intrinsics, native sizes, `trc_agent`, recorded image size) is written once to `/meta` from the optional `camera_metadata={...}` constructor kwarg so `render_overlays.py` doesn't depend on the live RealSense or `transforms_agent.npz` being present at render time. `discard_last_episode()` flushes pending writes, truncates every `/data/*` array back to the previous episode's end, pops the trailing `/meta/episode_ends` entry, and rewinds `zarr_n` — wired to the Backspace key in `collect_with_home.py`. There are no `img_{i}_raw` arrays anymore; `img_{i}` IS the raw frame.

`XArmDataset` (`dataset.py`) is the consumer:
- Converts the on-disk 7-dim xArm state (xyz + euler + grasp) into a **10-dim representation** (xyz + 6D rotation + grasp) via `xarm_state_to_10d`. The inverse `from_10d_to_xarm_state` exists for inference.
- Skips the first `skip_first_n` frames of each episode (default 5 in the dataset, 10 in `create_dataloader`) — these are the post-reset settling frames.
- Derives `action` from state: by default, action = next state (with last frame repeated); with `use_delta_actions=True`, action = per-step delta on the 6D-rot part plus absolute grasp.
- Returns chunked sequences shaped by `pred_horizon` / `obs_horizon` / `action_horizon`, with `obs_horizon` past observations and `pred_horizon` future actions.
- Does **not** currently consume the tactile columns — the recorder writes them but they're not yet plumbed into the training tensors.

If you change the state representation, image preprocessing in the recorder, or add new tactile columns, `XArmDataset` and any downstream policy code must change in lockstep — there is no schema versioning.

### External dependencies of note

- `teledex` — iPhone AR session client. Not pip-published; assumed already installed in the environment.
- `xarm` (`xArm-Python-SDK`) — UFactory robot SDK.
- `pyrealsense2`, `pynput`, `pyserial`, `zarr<3`, `scipy`, `opencv-python`, `torch` (for the 6D-rotation conversion in `dataset.py`).

### Sibling repos

- `tactile-ril-env/` (a separate git checkout sitting alongside this repo) is the source the synchronous `XArm` and tactile-safety code in `environment/` was adapted from. It also contains a multiprocessing-based stack (`XArmController(mp.Process)`, `MultiRealsense`, `RealEnv`, `SharedMemoryRingBuffer`) for higher-frequency control + recording. Consult it as reference; do not import from it.
