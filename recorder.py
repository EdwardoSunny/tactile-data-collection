import numpy as np
import zarr
import threading
import time
from collections import deque

class DatasetRecorder:
    def __init__(self, path, memory_buffer_size=5000, flush_interval=1.0, use_actions=True):
        self.path = path
        self.memory_buffer_size = memory_buffer_size
        self.flush_interval = flush_interval
        self.use_actions = use_actions
        self.num_cameras = None
        self.initialized = False
        
        self.memory_buffer = {
            'state': deque(maxlen=memory_buffer_size),
            'n_contacts': deque(maxlen=memory_buffer_size),
            'imgs': []
        }
        if self.use_actions:
            self.memory_buffer['action'] = deque(maxlen=memory_buffer_size)
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
                "state", shape=(0, state_dim), maxshape=(None, state_dim),
                chunks=(1024, state_dim), dtype=np.float32
            )
            if self.use_actions:
                data.create_dataset(
                    "action", shape=(0, act_dim), maxshape=(None, act_dim),
                    chunks=(1024, act_dim), dtype=np.float32
                )
            for i, img_shape in enumerate(img_shapes):
                data.create_dataset(
                    f"img_{i}", shape=(0, *img_shape), maxshape=(None, *img_shape),
                    chunks=(64, *img_shape), dtype=np.float32
                )
            data.create_dataset(
                "n_contacts", shape=(0, 1), maxshape=(None, 1),
                chunks=(1024, 1), dtype=np.float32
            )
            
        if "episode_ends" not in meta:
            meta.create_dataset(
                "episode_ends", shape=(0,), maxshape=(None,),
                chunks=(1024,), dtype=np.int64
            )
        
        self.initialized = True
        self._start_flush_thread()
            
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
        episode_ends = list(self.episode_ends_buffer)
        
        self.memory_buffer['state'].clear()
        self.memory_buffer['n_contacts'].clear()
        if self.use_actions:
            self.memory_buffer['action'].clear()
        for i in range(self.num_cameras):
            self.memory_buffer['imgs'][i].clear()
        self.episode_ends_buffer.clear()
        
        if not states:
            return
            
        min_length = len(states)
        if self.use_actions:
            min_length = min(min_length, len(actions))
        min_length = min(min_length, len(n_contacts))
        if self.num_cameras > 0:
            min_length = min(min_length, min(len(imgs[i]) for i in range(self.num_cameras)))
            
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
        
        states_array = np.array(states)
        n_contacts_array = np.array(n_contacts)
        if self.use_actions:
            actions_array = np.array(actions)
        imgs_arrays = [np.array(img_list) for img_list in imgs]
        
        data = self.store["data"]
        meta = self.store["meta"]
        
        new_size = self.zarr_n + min_length
        data["state"].resize((new_size, data["state"].shape[1]))
        if self.use_actions:
            data["action"].resize((new_size, data["action"].shape[1]))
        data["n_contacts"].resize((new_size, data["n_contacts"].shape[1]))
        for i in range(self.num_cameras):
            data[f"img_{i}"].resize((new_size, *data[f"img_{i}"].shape[1:]))
            
        data["state"][self.zarr_n:new_size] = states_array
        if self.use_actions:
            data["action"][self.zarr_n:new_size] = actions_array
        data["n_contacts"][self.zarr_n:new_size] = n_contacts_array
        for i in range(self.num_cameras):
            data[f"img_{i}"][self.zarr_n:new_size] = imgs_arrays[i]
            
        if episode_ends:
            old_ep_size = meta["episode_ends"].shape[0]
            new_ep_size = old_ep_size + len(episode_ends)
            meta["episode_ends"].resize((new_ep_size,))
            adjusted_ends = [ep + self.zarr_n for ep in episode_ends]
            meta["episode_ends"][old_ep_size:new_ep_size] = adjusted_ends
            
        self.zarr_n = new_size

        # print(f"Flushed {min_length} steps to zarr (total: {self.zarr_n})")  # disabled — see flush print above
        
    def append(self, state, n_contacts, imgs, action=None):
        if not self.initialized:
            state_dim = len(state)
            act_dim = len(action) if action is not None else 1
            img_shapes = [img.shape for img in imgs]
            print(f"Initializing recorder with state_dim={state_dim}, act_dim={act_dim}, img_shapes={img_shapes}")
            self._init_zarr_store(state_dim, act_dim, img_shapes)
        
        self.memory_buffer['state'].append(state)
        if self.use_actions and action is not None:
            self.memory_buffer['action'].append(action)
        self.memory_buffer['n_contacts'].append(n_contacts)
        
        for i, img in enumerate(imgs):
            self.memory_buffer['imgs'][i].append(img / 255.0)
            
        self._ep_step_counter += 1
        self._total_steps += 1
        
        if len(self.memory_buffer['state']) >= self.memory_buffer_size * 0.8:
            print(f"Memory buffer getting full ({len(self.memory_buffer['state'])}/{self.memory_buffer_size}), forcing flush...")
            self._flush_to_zarr()
        
    def end_episode(self):
        relative_end = len(self.memory_buffer['state'])
        self.episode_ends_buffer.append(relative_end)
        self._ep_step_counter = 0
        # print(f"Episode ended at step {relative_end} (memory buffer)")  # disabled — collect_with_home.py prints its own per-episode summary
        
    def close(self):
        self._stop_flushing = True
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            
        self._flush_to_zarr()

        # print(f"Saved {self.zarr_n} total steps to {self.path}")  # disabled — collect_with_home.py prints a richer end-of-session summary
