from scipy.spatial.transform import Rotation as R
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr

from environment.utils import xarm_state_to_10d

def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(
        train_data: Dict[str, np.ndarray],
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int
    ) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)

            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]

            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]

            data[sample_start_idx:sample_end_idx] = sample

        result[key] = data
    return result
def get_data_stats(data: np.ndarray):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data




class XArmDataset(Dataset):
    def __init__(
            self,
            zarr_path: str,
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
            skip_first_n: int = 5,
            use_delta_actions: bool = False
        ):
        
        self.dataset_root: zarr.Group = zarr.open(zarr_path, 'r')
        self.use_delta_actions = use_delta_actions


        
        state_data = self.dataset_root['data']['state']
        print(f"Original state data shape: {state_data.shape}")
        state_data = xarm_state_to_10d(state_data)
        print(f"State data after converting to 9D pose: {state_data.shape}")

        episode_ends = self.dataset_root['meta']['episode_ends'][:]
        
        assert episode_ends[-1] == state_data.shape[0]
        
        valid_indices = []
        for i in range(len(episode_ends)):
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            valid_indices.extend(range(start_idx + skip_first_n, end_idx))
        
        state_data = state_data[valid_indices]
        
        episode_ends_adjusted = []
        cumulative = 0
        for i in range(len(episode_ends)):
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx - skip_first_n
            cumulative += episode_length
            episode_ends_adjusted.append(cumulative)
        
        episode_ends = np.array(episode_ends_adjusted)
        
        actions = []
        start_idx = 0
        for end_idx in episode_ends:
            state_ep = state_data[start_idx:end_idx]
            
            if use_delta_actions:
                deltas_ep = np.zeros_like(state_ep)
                deltas_ep[:-1, :9] = state_ep[1:, :9] - state_ep[:-1, :9]
                deltas_ep[:, 9] = state_ep[:, 9]
                deltas_ep[-1, :9] = 0.0
                actions.append(deltas_ep)
            else:
                actions_ep = np.concatenate((
                    state_ep[1:],
                    state_ep[[-1]],
                ))
                actions.append(actions_ep)
            
            start_idx = end_idx
        
        actions = np.concatenate(actions, axis=0)
        
        img_keys = [key for key in self.dataset_root['data'].keys() if key.startswith('img_')]
        img_keys.sort()
        self.img_keys = img_keys
        self.num_cameras = len(img_keys)
        
        train_data = {
            'robot_state': state_data,
            'action': actions
        }
        
        for img_key in img_keys:
            img_data_full = self.dataset_root['data'][img_key][:]
            train_data[img_key] = img_data_full[valid_indices]
        
        if img_keys:
            sample_img = train_data[img_keys[0]][0]
            self.img_height, self.img_width = sample_img.shape[:2]

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        stats: Dict[str, Dict[str, np.ndarray]] = dict()
        
        robot_state_flat = train_data['robot_state'].reshape(-1, train_data['robot_state'].shape[-1])
        stats['robot_state'] = {
            'min': np.min(robot_state_flat, axis=0),
            'max': np.max(robot_state_flat, axis=0)
        }
        
        action_flat = train_data['action'].reshape(-1, train_data['action'].shape[-1])
        stats['action'] = {
            'min': np.min(action_flat, axis=0),
            'max': np.max(action_flat, axis=0)
        }
        
        normalized_train_data = {}
        normalized_train_data['robot_state'] = normalize_data(train_data['robot_state'], stats['robot_state'])
        normalized_train_data['action'] = normalize_data(train_data['action'], stats['action'])
        
        for img_key in img_keys:
            normalized_train_data[img_key] = train_data[img_key]
        
        self.normalized_train_data = normalized_train_data
        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        
        obs_imgs = {}
        for img_key in self.img_keys:
            obs_imgs[img_key] = nsample[img_key][:self.obs_horizon]
        
        robot_state = nsample['robot_state'][:self.obs_horizon, :]
        action = nsample['action']
        
        return {
            'robot_state': torch.from_numpy(robot_state).float(),
            'images': obs_imgs,
            'action': torch.from_numpy(action).float()
        }
def create_dataloader(zarr_path, pred_horizon, obs_horizon, action_horizon, batch_size=64, shuffle=True, num_workers=4, skip_first_n=10, use_delta_actions=False):
    dataset = XArmDataset(zarr_path, pred_horizon, obs_horizon, action_horizon, skip_first_n=skip_first_n, use_delta_actions=use_delta_actions)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,  
        pin_memory=True,  
        persistent_workers=True if num_workers > 0 else False,  
        prefetch_factor=2 if num_workers > 0 else None,  
        drop_last=True  
    )
    return loader


def get_observation_dim(zarr_path):
    dataset_root = zarr.open(zarr_path, 'r')
    robot_state_dim = dataset_root['data']['state'].shape[1]
    img_keys = [key for key in dataset_root['data'].keys() if key.startswith('img_')]
    num_cameras = len(img_keys)
    
    return {
        'robot_state_dim': 10,
        'num_cameras': num_cameras,
        'action_dim': 10
    }

