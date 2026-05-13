"""
Background threads for the teleop loop:
  PhoneReadThread   — ~1 kHz poll of phone pose / grasp / button under a lock.
  RecordingThread   — fixed-Hz sampler that pulls env.get_obs() + tactile state,
                      optionally draws force overlays, downsizes images to
                      224x224, and hands the frame to the DatasetRecorder.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from environment import tactile_overlay
from environment.tactile import TactileSensors


class PhoneReadThread(threading.Thread):
    def __init__(self, phone):
        super().__init__(daemon=True)
        self.phone = phone
        self.stop_thread = False
        self._lock = threading.Lock()
        self.latest_target_pose = None
        self.latest_grasp_state = 0.0
        self.latest_button_state = False

    def get_data(self):
        with self._lock:
            return self.latest_target_pose, self.latest_grasp_state, self.latest_button_state

    def stop(self):
        self.stop_thread = True

    def run(self):
        while not self.stop_thread:
            try:
                target_pose = self.phone.get_target_pose()
                grasp_state = self.phone.get_grasp_state()
                button_state = self.phone.get_button_state()
                with self._lock:
                    self.latest_target_pose = target_pose
                    self.latest_grasp_state = grasp_state
                    self.latest_button_state = button_state
            except Exception as e:
                print(f"Error reading phone data: {e}")
            time.sleep(0.001)


class RecordingThread(threading.Thread):
    """Fixed-frequency sampler. Per tick:
      1. env.get_obs() -> pose + camera frames
      2. tactile.get_latest() -> per-sensor xyz/connected/ts (if wired)
      3. Optionally draw force overlays on the agent + wrist images at native res
      4. Resize images to 224x224
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
        trc_agent: Optional[np.ndarray] = None,
        intrinsics_agent=None,
        draw_overlay: bool = True,
        viz_mode: Optional[str] = None,
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
        # Labels are only for log messages; default to "L"/"R" if not given.
        if sensor_labels is None and self.use_tactile:
            sensor_labels = ["L", "R"][: len(tactile.sensors)]
        self.sensor_labels = sensor_labels or []
        self.lag_warning_ms = tactile.config.lag_warning_ms if self.use_tactile else 200

        # Overlay wiring
        self.agent_serial = agent_serial
        self.wrist_serial = wrist_serial
        self.serial_to_index = serial_to_index or {}
        self.trc_agent = trc_agent
        self.intrinsics_agent = intrinsics_agent
        self.draw_overlay = draw_overlay
        self.viz_mode = viz_mode  # None -> use tactile_config.VISUALIZATION_MODE default

        # One-shot per-sensor lag warning, re-armed once data is fresh again.
        self._lag_warned: List[bool] = [False] * len(self.sensor_labels)

        # Cached native-resolution overlaid images for live viz. Populated on
        # every tick whether recording or not; consumed by get_latest_viz().
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
        windows can read fresh frames via get_latest_viz(). Returns a tuple
        of everything _record_one_tick needs to flush a frame, or None if
        no observation could be acquired.
        """
        obs = self.env.get_obs()
        if obs is None:
            return None

        with self._lock:
            grasp_state = self.latest_grasp_state
        if grasp_state is None:
            grasp_state = 0.0
        state = np.concatenate([np.array(obs["pose"]), [grasp_state]])

        tactile = self._read_tactile()
        if tactile is not None:
            tactile_values, tactile_connected, ts_arr, lag_arr = tactile
            # Pull out per-finger arrays for the overlay; expects left=idx 0, right=idx 1.
            vals_L = tactile_values[0] if tactile_values.shape[0] > 0 else None
            vals_R = tactile_values[1] if tactile_values.shape[0] > 1 else None
            conn_L = tactile_connected[0] if tactile_connected.shape[0] > 0 else None
            conn_R = tactile_connected[1] if tactile_connected.shape[0] > 1 else None
        else:
            vals_L = vals_R = conn_L = conn_R = None
            tactile_values = tactile_connected = ts_arr = lag_arr = None

        agent_img, wrist_img = self._pick_cameras(obs)

        # Overlay is drawn on the native-resolution image, BEFORE the 224x224 resize.
        if self.use_tactile and self.draw_overlay:
            if agent_img is not None and self.trc_agent is not None and self.intrinsics_agent is not None:
                agent_img = tactile_overlay.draw_agent_overlay(
                    agent_img, obs["pose"],
                    vals_L, vals_R, conn_L, conn_R,
                    self.trc_agent, self.intrinsics_agent,
                    mode=self.viz_mode,
                )
            if wrist_img is not None:
                wrist_img = tactile_overlay.draw_wrist_overlay(
                    wrist_img, vals_L, vals_R, conn_L, conn_R,
                    mode=self.viz_mode,
                )

        # Cache native-res overlaid frames for live viz consumers.
        with self._viz_lock:
            self._latest_agent_viz = None if agent_img is None else agent_img.copy()
            self._latest_wrist_viz = None if wrist_img is None else wrist_img.copy()
            self._latest_viz_time = time.monotonic()

        return (state, agent_img, wrist_img,
                tactile_values, tactile_connected, ts_arr, lag_arr)

    def _record_one_tick(self):
        """Compute a tick + downsize images + hand to the recorder.

        No-op when recorder is None (viz-only mode). Resize-to-224 happens here
        so the recorder only ever sees 224x224 frames regardless of camera res.
        """
        result = self._compute_tick()
        if result is None or self.recorder is None:
            return
        (state, agent_img, wrist_img,
         tactile_values, tactile_connected, ts_arr, lag_arr) = result

        camera_imgs = []
        for img in (agent_img, wrist_img):
            if img is None:
                camera_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                camera_imgs.append(cv2.resize(img, (224, 224)))

        if tactile_values is not None:
            self.recorder.append(
                state=state,
                n_contacts=np.array([0]),
                imgs=camera_imgs,
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
