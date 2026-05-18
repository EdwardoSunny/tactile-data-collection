"""
Background threads for the teleop loop:
  PhoneReadThread   — ~1 kHz poll of phone pose / grasp / button under a lock.
  RecordingThread   — fixed-Hz sampler that pulls env.get_obs() + tactile state,
                      downsizes images to 224x224, and hands the RAW frame
                      to the DatasetRecorder. Overlay drawing (sensordrawing)
                      runs on a separate copy of each native-res frame for the
                      live --viz windows ONLY; the recorded zarr never contains
                      overlay-burned pixels. Post-hoc overlay rendering is
                      scripts/render_overlays.py.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from environment.tactile import TactileSensors
from environment.tactile_overlay import SensorOverlay, DEFAULT_MODE_KEY


class PhoneReadThread(threading.Thread):
    def __init__(self, phone):
        super().__init__(daemon=True)
        self.phone = phone
        self.stop_thread = False
        self._lock = threading.Lock()
        self.latest_target_pose = None
        self.latest_grasp_state = 0.0
        self.latest_button_state = False
        self.latest_button_secondary_state = False

    def get_data(self):
        with self._lock:
            return self.latest_target_pose, self.latest_grasp_state, self.latest_button_state

    def get_button_secondary(self) -> bool:
        with self._lock:
            return self.latest_button_secondary_state

    def stop(self):
        self.stop_thread = True

    def run(self):
        while not self.stop_thread:
            try:
                target_pose = self.phone.get_target_pose()
                grasp_state = self.phone.get_grasp_state()
                button_state = self.phone.get_button_state()
                # TeleDex's second on-screen button. Captured but not consumed
                # by collect_with_home.py at the moment.
                button_secondary_state = self.phone.get_button_secondary_state()
                with self._lock:
                    self.latest_target_pose = target_pose
                    self.latest_grasp_state = grasp_state
                    self.latest_button_state = button_state
                    self.latest_button_secondary_state = button_secondary_state
            except Exception as e:
                print(f"Error reading phone data: {e}")
            time.sleep(0.001)


class RecordingThread(threading.Thread):
    """Fixed-frequency sampler. Per tick:
      1. env.get_obs() -> pose + joint_angles + grip_pos + camera frames
      2. tactile.get_latest() -> per-sensor xyz/connected/ts (if wired)
      3. Optionally normalize tactile + draw the sensordrawing overlay on a
         separate copy of each camera frame (live --viz only)
      4. Resize raw frames to 224x224
      5. recorder.append(...)
    """

    def __init__(
        self,
        recorder,
        env,
        frequency: float,
        tactile: Optional[TactileSensors] = None,
        sensor_labels: Optional[List[str]] = None,
        agent_serial: Optional[str] = None,
        wrist_serial: Optional[str] = None,
        serial_to_index: Optional[Dict[str, int]] = None,
        overlay: Optional[SensorOverlay] = None,
        draw_overlay: bool = True,
        viz_mode_key: str = DEFAULT_MODE_KEY,
    ):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.env = env
        self.frequency = frequency
        self.record_interval = 1.0 / frequency
        self.recording = False
        self.episode_started = False
        self.stop_thread = False
        self._lock = threading.Lock()
        self.latest_target_pose = None
        self.latest_grasp_state = None
        self.last_record_time = 0.0

        # Tactile wiring
        self.tactile = tactile
        self.use_tactile = tactile is not None
        if sensor_labels is None and self.use_tactile:
            sensor_labels = ["L", "R"][: len(tactile.sensors)]
        self.sensor_labels = sensor_labels or []
        self.lag_warning_ms = tactile.config.lag_warning_ms if self.use_tactile else 200

        # Overlay wiring — sensordrawing owns all geometry; we only need the
        # camera-role mapping to know which obs key feeds which SensorDrawer.
        self.agent_serial = agent_serial
        self.wrist_serial = wrist_serial
        self.serial_to_index = serial_to_index or {}
        self.overlay = overlay
        self.draw_overlay = draw_overlay and (overlay is not None) and self.use_tactile
        self.viz_mode_key = viz_mode_key

        # One-shot per-sensor lag warning, re-armed once data is fresh again.
        self._lag_warned: List[bool] = [False] * len(self.sensor_labels)

        # Cached native-res overlaid frames for live viz. Populated every tick
        # (even when not recording) so the main thread's cv2.imshow loop has
        # fresh frames; consumed by get_latest_viz().
        self._viz_lock = threading.Lock()
        self._latest_agent_viz: Optional[np.ndarray] = None
        self._latest_wrist_viz: Optional[np.ndarray] = None
        self._latest_viz_time: float = 0.0

    def set_recording(self, recording: bool):
        with self._lock:
            self.recording = recording
            self.episode_started = recording

    def is_recording(self) -> bool:
        with self._lock:
            return self.recording

    def update_data(self, target_pose, grasp_state):
        with self._lock:
            self.latest_target_pose = target_pose
            self.latest_grasp_state = grasp_state

    def stop(self):
        self.stop_thread = True

    def set_viz_mode_key(self, key: str):
        """Hot-swap the live --viz mode (next tick picks it up)."""
        self.viz_mode_key = key

    # -----------------------------------------------------------------
    # Per-tick work
    # -----------------------------------------------------------------

    def _read_tactile(self):
        """Returns (values (n_s,9,3), connected (n_s,9), ts (n_s,), lag (n_s,))
        or None if tactile isn't wired.
        """
        if not self.use_tactile:
            return None

        states = self.tactile.get_latest()
        now = time.time()

        values_list, connected_list, ts_list, lag_list = [], [], [], []
        for i, state in enumerate(states):
            values_list.append(state["xyz"])
            connected_list.append(state["connected"].astype(np.uint8))
            ts_list.append(int(state["device_ts_ms"]))
            host_ts = float(state["host_timestamp"])
            lag_ms = (now - host_ts) * 1000.0 if host_ts > 0 else float("inf")
            lag_list.append(lag_ms)

            # One-shot staleness warning per sensor.
            label = self.sensor_labels[i] if i < len(self.sensor_labels) else f"s{i}"
            if not self._lag_warned[i] and lag_ms > self.lag_warning_ms:
                print(f"  [tactile-{label}] stream stale: lag={lag_ms:.0f}ms")
                self._lag_warned[i] = True
            elif self._lag_warned[i] and lag_ms < 50:
                self._lag_warned[i] = False

        values = np.stack(values_list).astype(np.float32)
        connected = np.stack(connected_list).astype(np.uint8)
        ts_arr = np.asarray(ts_list, dtype=np.int64)
        # Clamp +inf so it survives the float32 cast cleanly downstream.
        lag_arr = np.asarray([min(l, 1e9) for l in lag_list], dtype=np.float32)
        return values, connected, ts_arr, lag_arr

    def _pick_cameras(self, obs):
        """Return (agent_img, wrist_img). Either may be None."""
        camera_keys = sorted(k for k in obs.keys() if k.startswith("camera_"))

        def img_at(idx):
            key = f"camera_{idx}"
            if key in obs and obs[key] is not None and "color_image" in obs[key]:
                return obs[key]["color_image"]
            return None

        agent_idx = self.serial_to_index.get(self.agent_serial)
        wrist_idx = self.serial_to_index.get(self.wrist_serial)
        agent_img = img_at(agent_idx) if agent_idx is not None else None
        wrist_img = img_at(wrist_idx) if wrist_idx is not None else None

        # Fallback to ordinal positions for whichever role didn't resolve.
        if agent_img is None or wrist_img is None:
            available = [
                obs[k]["color_image"]
                for k in camera_keys
                if obs[k] is not None and "color_image" in obs[k]
            ]
            if agent_img is None and len(available) >= 1:
                agent_img = available[0]
            if wrist_img is None:
                if len(available) >= 2:
                    wrist_img = available[1]
                elif len(available) >= 1:
                    wrist_img = available[0]

        return agent_img, wrist_img

    def _compute_tick(self):
        """Per-tick observation + overlay computation.

        Always runs at tick rate (even when not recording) so the live viz
        windows can read fresh frames. Returns everything _record_one_tick
        needs to flush a frame, or None if no observation was available.
        """
        obs = self.env.get_obs()
        if obs is None:
            return None

        with self._lock:
            grasp_state = self.latest_grasp_state
        if grasp_state is None:
            grasp_state = 0.0
        state = np.concatenate([np.array(obs["pose"]), [grasp_state]])
        joint_angles = np.asarray(obs.get("joint_angles", [0.0] * 7), dtype=np.float32)
        grip_pos = float(obs.get("grip_pos", 0.0))

        tactile = self._read_tactile()
        if tactile is not None:
            tactile_values, tactile_connected, ts_arr, lag_arr = tactile
            vals_L = tactile_values[0] if tactile_values.shape[0] > 0 else None
            vals_R = tactile_values[1] if tactile_values.shape[0] > 1 else None
        else:
            vals_L = vals_R = None
            tactile_values = tactile_connected = ts_arr = lag_arr = None

        agent_img_raw, wrist_img_raw = self._pick_cameras(obs)
        # Defensive copies — _pick_cameras may return numpy views over rs
        # frame buffers that get recycled. The recorder only ever sees these
        # raw versions; the overlay below is purely for live --viz.
        if agent_img_raw is not None:
            agent_img_raw = agent_img_raw.copy()
        if wrist_img_raw is not None:
            wrist_img_raw = wrist_img_raw.copy()

        # Live viz overlay (sensordrawing). Drawn on separate copies so the
        # recorder's raw frames stay untouched. Skipped when overlay isn't
        # wired or the operator passed --no-viz-overlay.
        agent_viz = None if agent_img_raw is None else agent_img_raw.copy()
        wrist_viz = None if wrist_img_raw is None else wrist_img_raw.copy()
        if self.draw_overlay and self.overlay is not None:
            try:
                nL, nR = self.overlay.normalize(vals_L, vals_R)
                if agent_viz is not None:
                    agent_viz = self.overlay.draw(
                        "side", agent_viz, joint_angles, grip_pos, nL, nR,
                        mode_key=self.viz_mode_key,
                    )
                if wrist_viz is not None:
                    wrist_viz = self.overlay.draw(
                        "wrist", wrist_viz, joint_angles, grip_pos, nL, nR,
                        mode_key=self.viz_mode_key,
                    )
            except Exception as e:
                # Don't take down the recording thread if the overlay hits a bad
                # frame — log once and keep going with raw pixels.
                print(f"  [overlay] draw failed: {e}")

        # Cache native-res overlaid frames for live viz consumers.
        with self._viz_lock:
            self._latest_agent_viz = None if agent_viz is None else agent_viz.copy()
            self._latest_wrist_viz = None if wrist_viz is None else wrist_viz.copy()
            self._latest_viz_time = time.monotonic()

        return (state, joint_angles, grip_pos,
                agent_img_raw, wrist_img_raw,
                tactile_values, tactile_connected, ts_arr, lag_arr)

    def _record_one_tick(self):
        """Compute a tick + downsize images + hand to the recorder.

        No-op when recorder is None (viz-only mode). Resize-to-224 happens here
        so the recorder only ever sees 224x224 frames regardless of camera res.
        Recorder always receives RAW (un-overlaid) images; any overlay
        rendering happens via scripts/render_overlays.py.
        """
        result = self._compute_tick()
        if result is None or self.recorder is None:
            return
        (state, joint_angles, grip_pos,
         agent_img_raw, wrist_img_raw,
         tactile_values, tactile_connected, ts_arr, lag_arr) = result

        # Race guard: the main thread may have called set_recording(False)
        # while we were inside _compute_tick (env.get_obs + overlay together
        # can take 40-80ms). If recording just turned off, drop this frame —
        # otherwise it'd land AFTER end_episode() ran and orphan into the
        # "in-progress" slot, breaking discard_last_episode()'s assertion.
        with self._lock:
            still_recording = self.recording and self.episode_started
        if not still_recording:
            return

        def _to_224(img):
            if img is None:
                return np.zeros((224, 224, 3), dtype=np.uint8)
            return cv2.resize(img, (224, 224))

        camera_imgs = [_to_224(agent_img_raw), _to_224(wrist_img_raw)]

        if tactile_values is not None:
            self.recorder.append(
                state=state,
                n_contacts=np.array([0]),
                imgs=camera_imgs,
                joint_angles=joint_angles,
                grip_pos=grip_pos,
                tactile=tactile_values,
                tactile_connected=tactile_connected,
                tactile_ts_ms=ts_arr,
                tactile_lag_ms=lag_arr,
            )
        else:
            self.recorder.append(
                state=state,
                n_contacts=np.array([0]),
                imgs=camera_imgs,
                joint_angles=joint_angles,
                grip_pos=grip_pos,
            )

    def get_latest_viz(self):
        """Return (agent_native_overlaid, wrist_native_overlaid, monotonic_ts).

        Either image may be None if it isn't available yet. Returned arrays
        are copies — caller may modify freely.
        """
        with self._viz_lock:
            agent = None if self._latest_agent_viz is None else self._latest_agent_viz.copy()
            wrist = None if self._latest_wrist_viz is None else self._latest_wrist_viz.copy()
            ts = self._latest_viz_time
        return agent, wrist, ts

    def run(self):
        self.last_record_time = time.monotonic()
        while not self.stop_thread:
            current_time = time.monotonic()

            if current_time - self.last_record_time >= self.record_interval:
                with self._lock:
                    should_record = self.recording and self.episode_started

                try:
                    if should_record and self.recorder is not None:
                        self._record_one_tick()
                    else:
                        # Compute + cache overlay even when not recording — feeds live viz.
                        self._compute_tick()
                except Exception as e:
                    import traceback
                    print(f"Error in recording thread: {e}")
                    traceback.print_exc()

                self.last_record_time = current_time

            time.sleep(0.01)
