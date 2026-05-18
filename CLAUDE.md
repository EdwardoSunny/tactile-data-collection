# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teleoperation + data-collection harness for a UFactory xArm robot, controlled by an iPhone over `teledex` (ARKit pose streamed to the workstation). Optional two-finger tactile sensing (A31301 boards over USB serial) provides (a) a gripper-safety wrapper that prevents the gripper from closing further when contact exceeds threshold, and (b) a kinematics-aware force overlay rendered by the vendored `environment/sensordrawing/` package (six variants: `points9_arrow`, `points1_arrow`, `points1_contact_{spatial,flat}`, `points9_color_{spatial,flat}`) — the overlay is drawn on-screen during teleop for operator feedback, and (separately) rendered post-hoc onto recorded frames by `scripts/render_overlays.py`. **Recorded frames are always RAW** (no overlay burned in); the overlay-augmented dataset lives in a separate zarr that is regenerated on demand from the raw one. Episodes get logged shaped for behavior-cloning / diffusion-policy training (see `XArmDataset` in `dataset.py`). There is no test suite, no linter, no build step — this is a script-driven Python project. Hardware required to actually run anything: an xArm at the configured IP, Intel RealSense cameras, an iPhone running the teledex client, and (optionally) two A31301 ESP32 boards.

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

# Pick a different live overlay variant.
python collect_with_home.py --record --viz --viz-mode points9_color_spatial

# Disable the live overlay computation (raw cameras in the viz windows).
python collect_with_home.py --record --viz --no-viz-overlay
```

`scripts/render_overlays.py` is the overlay-augmentation entry point. It reads the raw zarr, calls the vendored `sensordrawing` pipeline to render **all six overlay variants** per camera into a fresh `teleop_data_overlay.zarr`, and copies state/joint_angles/grip_pos/tactile/n_contacts/episode_ends/baseline through verbatim. The destination is wiped + regenerated on every run, so re-running picks up any newly-recorded episodes appended to the raw zarr:

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
- `--viz-mode {points9_arrow, points1_arrow, points1_contact_spatial, points9_color_spatial, points1_contact_flat, points9_color_flat}` — which sensordrawing variant the LIVE windows draw (default `points9_arrow`). Does not affect the recorded zarr (always raw) and does not affect `render_overlays.py` (which always renders all six).
- `--no-viz-overlay` — turn off the live overlay computation entirely. The `--viz` windows (if on) then just show the raw camera feeds. Tactile safety still applies.

Dataset utilities (run from repo root; most have a `DATA_PATH` constant hardcoded at the top of the file — edit it or pass `--data`):

```bash
python scripts/count_data.py --data teleop_data.zarr
python scripts/dataset_viewer.py          # browse / delete episodes
python scripts/create_2d_dataset.py       # strip state to xy only
python scripts/combine_zarrs.py           # merge two zarrs (paths hardcoded)
python scripts/repack_zarr.py <in.zarr>   # re-chunk + zstd compress
python scripts/render_overlays.py [src] [dst]   # raw -> overlay-augmented, all 6 variants (run by collect_and_render.sh)
```

`scripts/model_visualizer.py` imports a `policies.FlowMatchingPolicy` module that is **not in this repo** — policy training/inference lives elsewhere; this repo is the data-collection half of that workflow.

## Architecture

### Runtime topology (two threads + tactile threads + main loop)

`collect_with_home.py` wires together concurrent producers/consumers with the main thread as the action dispatcher:

1. **`PhoneReadThread`** (`threads.py`) — tight loop polling `Phone.get_target_pose()` / `get_grasp_state()` / `get_button_state()` (plus `get_button_secondary_state()` — TeleDex's second on-screen button, currently captured but not consumed by `collect_with_home.py`); exposes the latest values behind a lock. Reads are at ~1kHz; the main loop samples them.
2. **`RecordingThread`** (`threads.py`) — fixed-frequency (default 10 Hz) sampler. Per tick it pulls a fresh `env.get_obs()` (pose, joint_angles, grip_pos, plus camera frames), optionally pulls per-board tactile state via `TactileSensors.get_latest()`, normalizes tactile through the bundled `SensorNormalizer`, and — on a **separate copy** of each camera frame — calls the `SensorOverlay` (sensordrawing) to draw the chosen live-viz variant. The recorded frames are NEVER overlaid: the recorder receives the resized 224×224 RAW frame as `imgs=...`. When NOT recording, `_compute_tick()` still runs each tick to keep the cached `get_latest_viz()` frames fresh. State written each tick is `[pose(6), grasp(1)]` → 7-dim. There's a post-`_compute_tick` race guard that drops the frame if `set_recording(False)` fired while the tick was in flight, so frames never land after `end_episode()`.
3. **`TactileSensor`** threads (`environment/tactile.py`) — one daemon thread per serial port, owns the serial connection, parses `S,ts,idx,addr,conn,x,y,z` rows, accumulates a 9-taxel frame keyed on shared `ts_ms`, and publishes the latest complete frame under a lock. Handles ESP32 reboots and serial-open failures gracefully. Bundled into a single `TactileSensors` context manager which is the only thing the rest of the code sees.
4. **Main loop** — reads latest phone pose, on button rising-edge toggles recording (with the homing dance), calls `env.step(target_pose, grasp_state)`.

The teleop math (mapping phone delta-pose → robot delta-pose, with a 1200× position gain and ±500mm clip) lives in `Phone.get_target_pose()` in `environment/phone.py`. `Phone.reset(initial_robot_pose)` must be called once after the robot reaches its home pose — it captures the phone's frame as the origin for subsequent deltas, **and it must be re-called every time the robot is re-homed** otherwise the next dispatched pose will snap the arm back to its pre-home target.

### Task layer

`tasks/{simple,pushT,soccer}_task.py` are thin wrappers around `XArmEnvironment` that fix a reset position and (for pushT/soccer) lock Z to a constant so the user only commands XY. All three accept an optional `tactile=TactileSensors` and forward it to the env. `collect_with_home.py` uses `Simple_Task`; to swap tasks, edit that line — there is no CLI flag.

### Environment layer

`environment/env.py` (`XArmEnvironment`) is the user-facing API: `step(grasp, target_pose=...)`, `get_obs()`, `reset(duration)`, `go_to_position(...)`. `get_obs()` returns `pose` (6-dim xyz+euler degrees, from `arm.get_position`), `joint_angles` (7 servo angles in degrees, from `arm.get_servo_angle`), `grip_pos` (raw 0..850, from `arm.get_gripper_position`), plus one `camera_{i}` dict per camera. `joint_angles` + `grip_pos` exist specifically so sensordrawing's FK has its required inputs every tick. It owns a list of `Camera` objects (`environment/cameras.py`, RealSense via `pyrealsense2`) and a `XArm` instance. The `tactile=` kwarg is forwarded to `XArm` so the gripper-safety wrapper sees the live sensor state.

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

**Startup baseline capture** (`collect_with_home._capture_tactile_baseline`). Samples ~1.5 s of frames while the gripper is open at home, averages per-cell, installs the resulting `(2, 9, 3)` baseline on `tactile.config.baseline` and saves a copy to `/meta/tactile_baseline`. The baseline is consumed by the **gripper-safety wrapper only**: it lets the safety threshold operate in delta-from-idle units (~1500) instead of raw counts (~30000). The sensordrawing overlay does NOT consume this baseline — it has its own per-board calibration (offset + scale) shipped under `environment/sensordrawing/calibration_{left,right}.npz`.

Default `SAFETY_METRIC` is `sum_abs_z` and default `SAFETY_THRESHOLD` is `1500.0` in delta-from-idle units; `max_abs_z` does NOT work for this sensor mounting (see comment in `tactile_config.py`).

Defaults for ports, baud, lag-warning threshold, safety metric, safety threshold, and the camera serial → role mapping (which determines whether `SensorOverlay` projects through the side or the wrist calibration) live in `tactile_config.py`. Everything else about overlay rendering (camera intrinsics K, robot→camera extrinsics, finger-link kinematics, per-taxel positions, draw style, alpha, calibration tables) lives in `environment/sensordrawing/` and is not user-tunable from `tactile_config.py`.

### Overlay pipeline (`environment/tactile_overlay.py` + `environment/sensordrawing/`)

`environment/sensordrawing/` is a **vendored copy** of the standalone `sensordrawing` repo. It owns all the geometry: forward kinematics of the xArm7 (URDF DH parameters in `xram_kinematics.py`), the four-bar finger linkage, per-board sensor-on-PCB layouts, the side-camera intrinsics and `T_rc` from `transforms/transforms.npy`, and the wrist-camera dynamic `T_rc` derived per-frame from `T_link7_to_cam.npy` × inv(FK(joint_angles)). Two helper classes:

- **`SensorDrawer(camera_select={'side','wrist'})`** — `.draw_on_image(image_640x480_bgr, angles, grip_pos, normalized_L, normalized_R, mode=..., is_spatial=..., arrow_length_scale=..., left_color=..., right_color=...)`. Six valid `(mode, is_spatial)` combinations.
- **`SensorNormalizer(calibration_*.npz)`** — `.normalize(raw_9x3)` → normalized `(9, 3)` in `[-1, 1]` for xy and `[0, 1]` for z, using the bundled per-board offset+scale tables.

`environment/tactile_overlay.py` is a thin wrapper that owns one `SensorDrawer` per role and one `SensorNormalizer` per finger:
- `SensorOverlay.normalize(raw_L, raw_R)` → `(normalized_L, normalized_R)`.
- `SensorOverlay.draw(role, image_640x480_bgr, joint_angles, grip_pos, nL, nR, mode_key='points9_arrow')` → drawn image.
- `MODES` / `MODE_KEYS`: the canonical list of six variants, mirroring `sensordrawing/example_draw.py`:

| `mode_key`                 | sensordrawing `mode`  | `is_spatial` | description |
|---                         | ---                   | ---          | --- |
| `points9_arrow`            | `points9_arrow`       | `True`       | 9 dots + 9 force arrows per finger |
| `points1_arrow`            | `points1_arrow`       | `True`       | 1 center dot + 1 averaged arrow per finger |
| `points1_contact_spatial`  | `points1_contact`     | `True`       | 1 center dot per finger, only when any sensor norm ≥ 0.05 |
| `points9_color_spatial`    | `points9_color`       | `True`       | 9 dots per finger, color from sensor xyz mapped to RGB |
| `points1_contact_flat`     | `points1_contact`     | `False`      | same as above but anchored in screen corners (no projection) |
| `points9_color_flat`       | `points9_color`       | `False`      | same as above but anchored in screen corners |

Both threads.py and scripts/render_overlays.py go through `SensorOverlay`; threads.py picks the single live-viz variant from `--viz-mode`, render_overlays.py renders all six per camera into the overlay zarr. By project convention `img_0` = side / agent camera (`role='side'`), `img_1` = wrist camera (`role='wrist'`).

Native size: sensordrawing's K and T_rc are calibrated at 640×480, so the live viz path and `render_overlays.py` both upscale recorded 224×224 frames back to 640×480, draw the overlay, and downsize back to 224×224 afterwards.

### Dataset layer

There are two distinct on-disk shapes:

**Raw zarr** (`teleop_data.zarr`, written by `recorder.py` during teleop, appended to across sessions):
```
/data
  state              (N, 7)       float32   [x,y,z,rx,ry,rz,grasp_01] — xArm euler degrees + normalized grasp
  joint_angles       (N, 7)       float32   7 servo angles in degrees — sensordrawing FK input
  grip_pos           (N, 1)       float32   raw xArm gripper position 0..850 — sensordrawing FK input
  n_contacts         (N, 1)       float32   placeholder (always 0 in current recorder)
  img_0..K-1         (N,H,W,3)    float32   per-camera frames, 224×224, [0,1] — RAW (no overlay)
  action             (N, A)       float32   only present when recorder.use_actions=True (default False)
  tactile            (N, 2, 9, 3) float32   only when use_tactile=True — per-finger (Bx,By,Bz) per taxel, RAW (no baseline subtraction)
  tactile_connected  (N, 2, 9)    uint8     0/1 per taxel as reported by firmware
  tactile_ts_ms      (N, 2)       int64     device-side ms timestamp per board
  tactile_lag_ms     (N, 2)       float32   recording-time staleness per board
/meta
  episode_ends             (E,)               int64     cumulative end indices into /data
  tactile_baseline         (2, 9, 3)          float32   one-time idle field captured at startup (safety wrapper input)
```

**Overlay-augmented zarr** (`teleop_data_overlay.zarr`, regenerated by `scripts/render_overlays.py` from the raw zarr — wiped + rebuilt every run):
```
/data
  state, joint_angles, grip_pos, n_contacts, action, tactile*, img_0..K-1   ← copied verbatim from the raw zarr
  img_{i}_points9_arrow             (N,H,W,3)  float32
  img_{i}_points1_arrow             (N,H,W,3)  float32
  img_{i}_points1_contact_spatial   (N,H,W,3)  float32
  img_{i}_points9_color_spatial     (N,H,W,3)  float32
  img_{i}_points1_contact_flat      (N,H,W,3)  float32
  img_{i}_points9_color_flat        (N,H,W,3)  float32
/meta                                       ← copied verbatim from the raw zarr
```

Renderer pipeline: per frame, normalize tactile via the bundled `SensorNormalizer`, upscale the recorded 224×224 to 640×480, call `SensorDrawer.draw_on_image` for each of the six (mode, is_spatial, arrow_length_scale) tuples from `MODES`, downsize back to 224×224, store as float32 [0,1].

`DatasetRecorder` (`recorder.py`) buffers in memory (`deque`s) and flushes to zarr every `flush_interval` seconds (default 2.0 s in `collect_with_home.py`) via a background thread; it also force-flushes when the buffer hits 80% capacity. **It divides camera images by 255 inside `append()`** — do not normalize again upstream. State dim and number of cameras are inferred on the first `append()` call from the data shapes. When `use_tactile=True`, the recorder requires all four `tactile_*` kwargs on every `append()` and fails loudly if any are missing (so that a forgotten kwarg can't silently misalign rows). `joint_angles` and `grip_pos` are also REQUIRED on every `append()` (sensordrawing needs them per frame). `discard_last_episode()` flushes pending writes, truncates every `/data/*` array back to the previous episode's end, pops the trailing `/meta/episode_ends` entry, and rewinds `zarr_n` — wired to the Backspace key in `collect_with_home.py`. When resuming an existing dataset that predates the new schema, missing `joint_angles` / `grip_pos` columns get backfilled with zeros for the historical rows (overlay rendering of those rows will look wrong, but new data is unaffected).

`XArmDataset` (`dataset.py`) is the consumer:
- Converts the on-disk 7-dim xArm state (xyz + euler + grasp_01) into a **10-dim representation** (xyz + 6D rotation + grasp) via `xarm_state_to_10d`. The inverse `from_10d_to_xarm_state` exists for inference.
- Skips the first `skip_first_n` frames of each episode (default 5 in the dataset, 10 in `create_dataloader`) — these are the post-reset settling frames.
- Derives `action` from state: by default, action = next state (with last frame repeated); with `use_delta_actions=True`, action = per-step delta on the 6D-rot part plus absolute grasp.
- Returns chunked sequences shaped by `pred_horizon` / `obs_horizon` / `action_horizon`, with `obs_horizon` past observations and `pred_horizon` future actions.
- Does **not** currently consume the tactile, joint_angles, or grip_pos columns — the recorder writes them but they're not yet plumbed into the training tensors.

If you change the state representation, image preprocessing in the recorder, or add new tactile columns, `XArmDataset` and any downstream policy code must change in lockstep — there is no schema versioning.

### External dependencies of note

- `teledex` — iPhone AR session client. Not pip-published; assumed already installed in the environment.
- `xarm` (`xArm-Python-SDK`) — UFactory robot SDK.
- `pyrealsense2`, `pynput`, `pyserial`, `zarr<3`, `scipy`, `opencv-python`, `torch` (for the 6D-rotation conversion in `dataset.py`).

### Sibling repos

- `tactile-ril-env/` (a separate git checkout sitting alongside this repo) is the source the synchronous `XArm` and tactile-safety code in `environment/` was adapted from. It also contains a multiprocessing-based stack (`XArmController(mp.Process)`, `MultiRealsense`, `RealEnv`, `SharedMemoryRingBuffer`) for higher-frequency control + recording. Consult it as reference; do not import from it.
- `sensordrawing/` (separate checkout) is the upstream for `environment/sensordrawing/`. Treat the vendored copy as the source of truth; if you re-pull from upstream, preserve the relative-import edits in `draw_sensors.py` (and any other tweaks).
