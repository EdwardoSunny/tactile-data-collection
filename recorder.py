import os
import numpy as np
import zarr
import threading
import time
from collections import deque

import cv2


# Compression settings. zstd is fast + ratios good for both images and small
# numeric arrays; BITSHUFFLE helps a lot for repeated low-entropy bytes
# (which floats-in-[0,1] images are, since most byte positions repeat),
# SHUFFLE for general numeric arrays. clevel=3 is a good speed/ratio tradeoff
# for online recording; if you want max compression at the cost of write
# time, scripts/repack_zarr.py re-compresses with clevel=5.
_img_compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
_num_compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)


class DatasetRecorder:
    def __init__(self, path, memory_buffer_size=5000, flush_interval=1.0,
                 use_actions=True, use_tactile=False, tactile_baseline=None,
                 record_videos=True, video_fps=10.0):
        self.path = path
        self.memory_buffer_size = memory_buffer_size
        self.flush_interval = flush_interval
        self.use_actions = use_actions
        self.use_tactile = use_tactile
        # Optional per-cell idle baseline (n_sensors, n_taxels, 3). Saved
        # once to /meta/tactile_baseline so downstream code can subtract it
        # to get a delta-from-idle view of /data/tactile (which stays raw).
        self.tactile_baseline = tactile_baseline
        self.num_cameras = None
        self.initialized = False

        # Per-episode MP4 recording (sibling directory to the zarr). One mp4
        # per camera per episode, opened on the first frame of an episode and
        # closed on end_episode (or recorder shutdown).
        self.record_videos = bool(record_videos)
        self.video_fps = float(video_fps)
        # videos live alongside the zarr: e.g.  teleop_data.zarr_videos/
        self._video_dir = (path.rstrip("/") + "_videos") if self.record_videos else None
        self._video_writers = {}        # camera_index -> cv2.VideoWriter (open during an episode)
        self._video_episode_num = 0     # incremented on each new episode (1-based)

        self.memory_buffer = {
            'state': deque(maxlen=memory_buffer_size),
            'n_contacts': deque(maxlen=memory_buffer_size),
            'imgs': [],
        }
        if self.use_actions:
            self.memory_buffer['action'] = deque(maxlen=memory_buffer_size)
        if self.use_tactile:
            # Each entry: per-tick ndarray of fixed shape.
            self.memory_buffer['tactile']           = deque(maxlen=memory_buffer_size)
            self.memory_buffer['tactile_connected'] = deque(maxlen=memory_buffer_size)
            self.memory_buffer['tactile_ts_ms']     = deque(maxlen=memory_buffer_size)
            self.memory_buffer['tactile_lag_ms']    = deque(maxlen=memory_buffer_size)
        self.episode_ends_buffer = deque()
        
        self._ep_step_counter = 0
        self._total_steps = 0
        
        self._flush_thread = None
        self._stop_flushing = False
        
        self.store = None
        self.zarr_n = 0
        
    def _init_zarr_store(self, state_dim, act_dim, img_shapes):
        self.store = zarr.open(self.path, mode="a")
        data = self.store.require_group("data")
        meta = self.store.require_group("meta")
        
        self.num_cameras = len(img_shapes)
        self.memory_buffer['imgs'] = [deque(maxlen=self.memory_buffer_size) for _ in range(self.num_cameras)]
        
        if "state" in data:
            self.zarr_n = data["state"].shape[0]
            print(f"Resuming existing dataset at step {self.zarr_n}")
        else:
            self.zarr_n = 0
            data.create_dataset(
                "state", shape=(0, state_dim),
                chunks=(1024, state_dim), dtype=np.float32,
                compressor=_num_compressor,
            )
            if self.use_actions:
                data.create_dataset(
                    "action", shape=(0, act_dim),
                    chunks=(1024, act_dim), dtype=np.float32,
                    compressor=_num_compressor,
                )
            for i, img_shape in enumerate(img_shapes):
                data.create_dataset(
                    f"img_{i}", shape=(0, *img_shape),
                    chunks=(32, *img_shape), dtype=np.float32,
                    compressor=_img_compressor,
                )
            data.create_dataset(
                "n_contacts", shape=(0, 1),
                chunks=(1024, 1), dtype=np.float32,
                compressor=_num_compressor,
            )
            if self.use_tactile:
                # tactile values: (N, 2, 9, 3) — fingers x cells x (Bx,By,Bz)
                data.create_dataset(
                    "tactile", shape=(0, 2, 9, 3),
                    chunks=(1024, 2, 9, 3), dtype=np.float32,
                    compressor=_num_compressor,
                )
                data.create_dataset(
                    "tactile_connected", shape=(0, 2, 9),
                    chunks=(1024, 2, 9), dtype=np.uint8,
                    compressor=_num_compressor,
                )
                data.create_dataset(
                    "tactile_ts_ms", shape=(0, 2),
                    chunks=(1024, 2), dtype=np.int64,
                    compressor=_num_compressor,
                )
                data.create_dataset(
                    "tactile_lag_ms", shape=(0, 2),
                    chunks=(1024, 2), dtype=np.float32,
                    compressor=_num_compressor,
                )

        # Save the tactile baseline once (idempotent — only writes if absent or
        # different from what's already there).
        if self.use_tactile and self.tactile_baseline is not None:
            baseline_arr = np.asarray(self.tactile_baseline, dtype=np.float32)
            if "tactile_baseline" not in meta:
                meta.create_dataset(
                    "tactile_baseline", shape=baseline_arr.shape,
                    chunks=baseline_arr.shape, dtype=np.float32,
                )
                meta["tactile_baseline"][:] = baseline_arr
            else:
                # If resuming an existing dataset, overwrite the baseline only
                # if it's a different shape; otherwise leave the original alone
                # so downstream code can use a single consistent baseline.
                existing = meta["tactile_baseline"]
                if existing.shape != baseline_arr.shape:
                    print(
                        f"  [recorder] tactile_baseline shape changed "
                        f"({existing.shape} -> {baseline_arr.shape}); overwriting"
                    )
                    del meta["tactile_baseline"]
                    meta.create_dataset(
                        "tactile_baseline", shape=baseline_arr.shape,
                        chunks=baseline_arr.shape, dtype=np.float32,
                    )
                    meta["tactile_baseline"][:] = baseline_arr

        if "episode_ends" not in meta:
            meta.create_dataset(
                "episode_ends", shape=(0,),
                chunks=(1024,), dtype=np.int64,
                compressor=_num_compressor,
            )
        
        self.initialized = True
        self._start_flush_thread()

        # Sync next episode number from existing zarr meta (so resume keeps counting).
        if self.record_videos and self._video_dir:
            try:
                os.makedirs(self._video_dir, exist_ok=True)
            except Exception as e:
                print(f"  [warn] couldn't create video dir {self._video_dir}: {e}")
                self.record_videos = False
        if self.record_videos and "episode_ends" in meta:
            try:
                self._video_episode_num = int(meta["episode_ends"].shape[0])
            except Exception:
                self._video_episode_num = 0

    def _open_video_writers(self, frame_hw):
        """Open one VideoWriter per camera for the upcoming episode."""
        if not self.record_videos or self.num_cameras is None:
            return
        self._video_episode_num += 1
        ep = self._video_episode_num
        h, w = int(frame_hw[0]), int(frame_hw[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for i in range(self.num_cameras):
            video_path = os.path.join(
                self._video_dir, f"ep_{ep:03d}_cam_{i}.mp4"
            )
            writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (w, h))
            if writer.isOpened():
                self._video_writers[i] = writer
            else:
                print(f"  [warn] couldn't open video writer at {video_path}; videos for this episode disabled")
                writer.release()

    def _close_video_writers(self):
        for w in list(self._video_writers.values()):
            try:
                w.release()
            except Exception:
                pass
        self._video_writers = {}
            
    def _start_flush_thread(self):
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
    def _flush_loop(self):
        while not self._stop_flushing:
            time.sleep(self.flush_interval)
            self._flush_to_zarr()
            
    def _flush_to_zarr(self):
        if not self.memory_buffer['state']:
            return
            
        states = list(self.memory_buffer['state'])
        n_contacts = list(self.memory_buffer['n_contacts'])
        if self.use_actions:
            actions = list(self.memory_buffer['action'])
        imgs = [list(self.memory_buffer['imgs'][i]) for i in range(self.num_cameras)]
        if self.use_tactile:
            tactile           = list(self.memory_buffer['tactile'])
            tactile_connected = list(self.memory_buffer['tactile_connected'])
            tactile_ts_ms     = list(self.memory_buffer['tactile_ts_ms'])
            tactile_lag_ms    = list(self.memory_buffer['tactile_lag_ms'])
        episode_ends = list(self.episode_ends_buffer)

        self.memory_buffer['state'].clear()
        self.memory_buffer['n_contacts'].clear()
        if self.use_actions:
            self.memory_buffer['action'].clear()
        for i in range(self.num_cameras):
            self.memory_buffer['imgs'][i].clear()
        if self.use_tactile:
            self.memory_buffer['tactile'].clear()
            self.memory_buffer['tactile_connected'].clear()
            self.memory_buffer['tactile_ts_ms'].clear()
            self.memory_buffer['tactile_lag_ms'].clear()
        self.episode_ends_buffer.clear()

        if not states:
            return

        min_length = len(states)
        if self.use_actions:
            min_length = min(min_length, len(actions))
        min_length = min(min_length, len(n_contacts))
        if self.num_cameras > 0:
            min_length = min(min_length, min(len(imgs[i]) for i in range(self.num_cameras)))
        if self.use_tactile:
            min_length = min(min_length, len(tactile), len(tactile_connected),
                             len(tactile_ts_ms), len(tactile_lag_ms))
            
        if min_length == 0:
            return
        
        # Per-flush "Flushing N steps" print disabled — internal disk I/O, not relevant during recording.
        # if self.use_actions:
        #     print(f"Flushing {min_length} steps (states: {len(states)}, actions: {len(actions)}, imgs: {[len(imgs[i]) for i in range(self.num_cameras)]})")
        # else:
        #     print(f"Flushing {min_length} steps (states: {len(states)}, imgs: {[len(imgs[i]) for i in range(self.num_cameras)]})")
        
        states = states[:min_length]
        n_contacts = n_contacts[:min_length]
        if self.use_actions:
            actions = actions[:min_length]
        imgs = [img_list[:min_length] for img_list in imgs]
        if self.use_tactile:
            tactile           = tactile[:min_length]
            tactile_connected = tactile_connected[:min_length]
            tactile_ts_ms     = tactile_ts_ms[:min_length]
            tactile_lag_ms    = tactile_lag_ms[:min_length]

        states_array = np.array(states)
        n_contacts_array = np.array(n_contacts)
        if self.use_actions:
            actions_array = np.array(actions)
        imgs_arrays = [np.array(img_list) for img_list in imgs]
        if self.use_tactile:
            tactile_arr           = np.asarray(tactile,           dtype=np.float32)
            tactile_connected_arr = np.asarray(tactile_connected, dtype=np.uint8)
            tactile_ts_ms_arr     = np.asarray(tactile_ts_ms,     dtype=np.int64)
            tactile_lag_ms_arr    = np.asarray(tactile_lag_ms,    dtype=np.float32)
        
        data = self.store["data"]
        meta = self.store["meta"]
        
        new_size = self.zarr_n + min_length
        data["state"].resize((new_size, data["state"].shape[1]))
        if self.use_actions:
            data["action"].resize((new_size, data["action"].shape[1]))
        data["n_contacts"].resize((new_size, data["n_contacts"].shape[1]))
        for i in range(self.num_cameras):
            data[f"img_{i}"].resize((new_size, *data[f"img_{i}"].shape[1:]))
        if self.use_tactile:
            data["tactile"].resize((new_size, *data["tactile"].shape[1:]))
            data["tactile_connected"].resize((new_size, *data["tactile_connected"].shape[1:]))
            data["tactile_ts_ms"].resize((new_size, *data["tactile_ts_ms"].shape[1:]))
            data["tactile_lag_ms"].resize((new_size, *data["tactile_lag_ms"].shape[1:]))

        data["state"][self.zarr_n:new_size] = states_array
        if self.use_actions:
            data["action"][self.zarr_n:new_size] = actions_array
        data["n_contacts"][self.zarr_n:new_size] = n_contacts_array
        for i in range(self.num_cameras):
            data[f"img_{i}"][self.zarr_n:new_size] = imgs_arrays[i]
        if self.use_tactile:
            data["tactile"][self.zarr_n:new_size]           = tactile_arr
            data["tactile_connected"][self.zarr_n:new_size] = tactile_connected_arr
            data["tactile_ts_ms"][self.zarr_n:new_size]     = tactile_ts_ms_arr
            data["tactile_lag_ms"][self.zarr_n:new_size]    = tactile_lag_ms_arr
            
        if episode_ends:
            old_ep_size = meta["episode_ends"].shape[0]
            new_ep_size = old_ep_size + len(episode_ends)
            meta["episode_ends"].resize((new_ep_size,))
            adjusted_ends = [ep + self.zarr_n for ep in episode_ends]
            meta["episode_ends"][old_ep_size:new_ep_size] = adjusted_ends
            
        self.zarr_n = new_size

        # print(f"Flushed {min_length} steps to zarr (total: {self.zarr_n})")  # disabled — see flush print above
        
    def append(self, state, n_contacts, imgs, action=None,
               tactile=None, tactile_connected=None,
               tactile_ts_ms=None, tactile_lag_ms=None):
        if not self.initialized:
            state_dim = len(state)
            act_dim = len(action) if action is not None else 1
            img_shapes = [img.shape for img in imgs]
            print(f"Initializing recorder with state_dim={state_dim}, act_dim={act_dim}, img_shapes={img_shapes}, use_tactile={self.use_tactile}")
            self._init_zarr_store(state_dim, act_dim, img_shapes)

        self.memory_buffer['state'].append(state)
        if self.use_actions and action is not None:
            self.memory_buffer['action'].append(action)
        self.memory_buffer['n_contacts'].append(n_contacts)

        # Open video writers lazily on the first frame of each new episode.
        # An "episode" here = the run between end_episode() calls. We detect
        # the start by checking that no writers are currently open AND there
        # are camera frames to write.
        if self.record_videos and not self._video_writers and len(imgs) > 0 and imgs[0] is not None:
            self._open_video_writers(frame_hw=imgs[0].shape[:2])

        for i, img in enumerate(imgs):
            # Write the (already-overlaid, uint8) frame to MP4 BEFORE normalizing for zarr.
            if i in self._video_writers and img is not None:
                try:
                    self._video_writers[i].write(img)
                except Exception as e:
                    print(f"  [warn] video write failed for cam {i}: {e}")
            self.memory_buffer['imgs'][i].append(img / 255.0)

        if self.use_tactile:
            # Caller is required to pass all four tactile fields when use_tactile=True.
            # We don't silently substitute zeros: if you forget one, the next flush
            # would silently misalign rows. Fail loudly instead.
            if tactile is None or tactile_connected is None \
                    or tactile_ts_ms is None or tactile_lag_ms is None:
                raise ValueError(
                    "use_tactile=True but a tactile_* kwarg was None. "
                    "Pass all four: tactile, tactile_connected, tactile_ts_ms, tactile_lag_ms."
                )
            self.memory_buffer['tactile'].append(np.asarray(tactile, dtype=np.float32))
            self.memory_buffer['tactile_connected'].append(np.asarray(tactile_connected, dtype=np.uint8))
            self.memory_buffer['tactile_ts_ms'].append(np.asarray(tactile_ts_ms, dtype=np.int64))
            self.memory_buffer['tactile_lag_ms'].append(np.asarray(tactile_lag_ms, dtype=np.float32))

        self._ep_step_counter += 1
        self._total_steps += 1
        
        if len(self.memory_buffer['state']) >= self.memory_buffer_size * 0.8:
            print(f"Memory buffer getting full ({len(self.memory_buffer['state'])}/{self.memory_buffer_size}), forcing flush...")
            self._flush_to_zarr()
        
    def end_episode(self):
        relative_end = len(self.memory_buffer['state'])
        self.episode_ends_buffer.append(relative_end)
        self._ep_step_counter = 0
        # Finalize this episode's MP4s; next append starts a new one.
        self._close_video_writers()
        # print(f"Episode ended at step {relative_end} (memory buffer)")  # disabled — collect_with_home.py prints its own per-episode summary
        
    def close(self):
        self._stop_flushing = True
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)

        # Ensure any in-progress episode's MP4s are properly finalized.
        self._close_video_writers()

        self._flush_to_zarr()

        # print(f"Saved {self.zarr_n} total steps to {self.path}")  # disabled — collect_with_home.py prints a richer end-of-session summary
