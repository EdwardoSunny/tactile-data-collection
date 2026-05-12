# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teleoperation + data-collection harness for a UFactory xArm robot, controlled by an iPhone over `teledex` (ARKit pose streamed to the workstation). Episodes get logged to a zarr dataset shaped for behavior-cloning / diffusion-policy training (see `XArmDataset` in `dataset.py`). There is no test suite, no linter, no build step — this is a script-driven Python project. Hardware required to actually run anything: an xArm at the configured IP, Intel RealSense cameras, and an iPhone running the teledex client.

## Running things

```bash
# Teleop only (no recording)
python run.py
# Teleop + record episodes into teleop_data.zarr
python run.py --record
# Variant entry point (identical except for a longer pose-settle loop before phone.reset)
python collect.py --record
```

Recording control during a session:
- Press the **phone button** to start/stop an episode (3-second cooldown enforced).
- Press **q** in the terminal to quit (flushes outstanding zarr writes via `recorder.close()`).

Dataset utilities (run from repo root; most have a `DATA_PATH` constant hardcoded at the top of the file — edit it or pass `--data`):

```bash
python scripts/count_data.py --data teleop_data.zarr
python scripts/dataset_viewer.py          # browse / delete episodes
python scripts/create_2d_dataset.py       # strip state to xy only
python scripts/combine_zarrs.py           # merge two zarrs (paths hardcoded)
python scripts/repack_zarr.py <in.zarr>   # re-chunk + zstd compress
```

`scripts/model_visualizer.py` imports a `policies.FlowMatchingPolicy` module that is **not in this repo** — policy training/inference lives elsewhere; this repo is the data-collection half of that workflow.

## Architecture

### Runtime topology (three threads + main loop)

`run.py` / `collect.py` wire together three concurrent producers/consumers with the main thread as the action dispatcher:

1. **`PhoneReadThread`** (`threads.py`) — tight loop polling `Phone.get_target_pose()` / `get_grasp_state()` / `get_button_state()`; exposes the latest values behind a lock. Reads are at ~1kHz; the main loop samples them.
2. **`RecordingThread`** (`threads.py`) — fixed-frequency (default 10 Hz) sampler. When `recording` is enabled, it pulls a fresh `env.get_obs()` (including camera frames), resizes camera images to 224×224, concatenates `[pose(6), grasp(1)]` into a 7-dim state, and calls `recorder.append(...)`. Important: it samples obs **independently** of the main control loop — pose written to disk is the pose at recording-sample time, not at action-issue time.
3. **`KeystrokeCounter`** (`environment/keystroke_counter.py`) — pynput listener for the `q` quit key.
4. **Main loop** — reads latest phone pose, on button rising-edge toggles recording, calls `env.step(target_pose, grasp_state)`.

The teleop math (mapping phone delta-pose → robot delta-pose, with a 1200× position gain and ±500mm clip) lives in `Phone.get_target_pose()` in `environment/phone.py`. `Phone.reset(initial_robot_pose)` must be called once after the robot reaches its home pose — it captures the phone's frame as the origin for subsequent deltas.

### Task layer

`tasks/{simple,pushT,soccer}_task.py` are thin wrappers around `XArmEnvironment` that fix a reset position and (for pushT/soccer) lock Z to a constant so the user only commands XY. To swap tasks, edit the `env = Simple_Task()` line in `run.py` / `collect.py` — there is no CLI flag.

### Environment layer

`environment/env.py` (`XArmEnvironment`) is the user-facing API: `step(grasp, target_pose=...)`, `get_obs()`, `reset(duration)`, `go_to_position(...)`. It owns a list of `Camera` objects (`environment/cameras.py`, RealSense via `pyrealsense2`) and an `XArm` instance.

`environment/xarm_controller.py` defines two things:
- **`XArm`** — direct synchronous `XArmAPI` wrapper, used by `XArmEnvironment`. The docstring marks it as "legacy" but it's the one currently wired up.
- **`XArmController`** — a multiprocessing-based controller using `SharedMemoryQueue`/`SharedMemoryRingBuffer` (in the same directory). Intended for higher-frequency control; **not** used by `run.py`. If you wire it in, also bring in `shared_memory_queue.py` / `shared_memory_ring_buffer.py` / `shared_ndarray.py`.

Workspace bounds are hardcoded in `XArmEnvironment.__init__`: `[[250,800],[-600,600],[50,400]]` mm. Robot IP is hardcoded in `XArmConfig` (`192.168.1.223`).

### Dataset layer

**On-disk schema** (zarr group, e.g. `teleop_data.zarr`):
```
/data
  state       (N, 7)    float32   [x,y,z,rx,ry,rz,grasp] — xArm euler degrees
  n_contacts  (N, 1)    float32   placeholder (always 0 in current recorder)
  img_0..K-1  (N,H,W,3) float32   per-camera frames, 224×224, normalized to [0,1] at append time
  action      (N, A)    float32   only present when recorder.use_actions=True (default False)
/meta
  episode_ends (E,)     int64     cumulative end indices into /data
```

`DatasetRecorder` (`recorder.py`) buffers in memory (`deque`s) and flushes to zarr every `flush_interval` seconds via a background thread; it also force-flushes when the buffer hits 80% capacity. **It divides camera images by 255 inside `append()`** — do not normalize again upstream. State dim and number of cameras are inferred on the first `append()` call from the data shapes.

`XArmDataset` (`dataset.py`) is the consumer:
- Converts the on-disk 7-dim xArm state (xyz + euler + grasp) into a **10-dim representation** (xyz + 6D rotation + grasp) via `xarm_state_to_10d`. The inverse `from_10d_to_xarm_state` exists for inference.
- Skips the first `skip_first_n` frames of each episode (default 5 in the dataset, 10 in `create_dataloader`) — these are the post-reset settling frames.
- Derives `action` from state: by default, action = next state (with last frame repeated); with `use_delta_actions=True`, action = per-step delta on the 6D-rot part plus absolute grasp.
- Returns chunked sequences shaped by `pred_horizon` / `obs_horizon` / `action_horizon`, with `obs_horizon` past observations and `pred_horizon` future actions.

If you change the state representation or the image preprocessing in the recorder, `XArmDataset` and any downstream policy code must change in lockstep — there is no schema versioning.

### External dependencies of note

- `teledex` — iPhone AR session client. Not pip-published; assumed already installed in the environment.
- `xarm` (`xArm-Python-SDK`) — UFactory robot SDK.
- `pyrealsense2`, `pynput`, `zarr<3`, `scipy`, `opencv-python`, `torch` (for the 6D-rotation conversion in `dataset.py`).
