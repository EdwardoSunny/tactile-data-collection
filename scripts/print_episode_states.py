import zarr
import numpy as np

DATA_PATH = "teleop_data.zarr"


def print_episode_states():
    store = zarr.open(DATA_PATH, mode="r")
    data = store["data"]
    meta = store["meta"]
    
    episode_ends = np.array(meta["episode_ends"])
    states = np.array(data["state"])
    
    num_episodes = len(episode_ends)
    print(f"\nDataset: {DATA_PATH}")
    print(f"Total episodes: {num_episodes}")
    print(f"Total frames: {states.shape[0]}")
    print(f"State dimension: {states.shape[1]}")
    print("=" * 80)
    
    starts = np.concatenate([[0], episode_ends[:-1]])
    
    for ep_idx in range(num_episodes):
        start_idx = starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        length = end_idx - start_idx
        
        first_state = states[start_idx]
        last_state = states[end_idx - 1]
        
        print(f"\nEpisode {ep_idx}:")
        print(f"  First state: {first_state[0:2]}")
        print(f"  Last state:  {last_state[0:2]}")
        print("-" * 80)
    
    print()


if __name__ == "__main__":
    print_episode_states()
