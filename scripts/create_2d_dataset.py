import zarr
import numpy as np

INPUT_PATH = "teleop_data.zarr"
OUTPUT_PATH = "teleop_data_2d.zarr"


def create_2d_dataset():
    print(f"Opening source dataset: {INPUT_PATH}")
    src_store = zarr.open(INPUT_PATH, mode="r")
    src_data = src_store["data"]
    src_meta = src_store["meta"]
    
    total_frames = src_data["state"].shape[0]
    state_dim = src_data["state"].shape[1]
    episode_ends = np.array(src_meta["episode_ends"])
    
    print(f"Source state shape: ({total_frames}, {state_dim})")
    print(f"Total frames: {total_frames}")
    print(f"Total episodes: {len(episode_ends)}")
    
    img_keys = [key for key in src_data.keys() if key.startswith('img')]
    img_shapes = {key: src_data[key].shape[1:] for key in img_keys}
    print(f"Found image keys: {img_keys}")
    
    print(f"\nCreating new dataset: {OUTPUT_PATH}")
    dst_store = zarr.open(OUTPUT_PATH, mode="w")
    dst_data = dst_store.create_group("data")
    dst_meta = dst_store.create_group("meta")
    
    print("Creating state dataset (2D)...")
    dst_data.create_dataset("state", shape=(total_frames, 2), chunks=(1024, 2), dtype=np.float32)
    
    for img_key in img_keys:
        print(f"Creating {img_key} dataset...")
        dst_data.create_dataset(img_key, shape=(total_frames, *img_shapes[img_key]), chunks=(64, *img_shapes[img_key]), dtype=np.float32)
    
    print("Copying episode_ends...")
    dst_meta.create_dataset("episode_ends", data=episode_ends, chunks=(1024,), dtype=np.int64)
    
    chunk_size = 1000
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    
    print(f"\nCopying data in {num_chunks} chunks of {chunk_size} frames...")
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        
        print(f"  Chunk {i+1}/{num_chunks}: frames {start_idx}-{end_idx}")
        
        state_chunk = src_data["state"][start_idx:end_idx, :2]
        dst_data["state"][start_idx:end_idx] = state_chunk
        
        for img_key in img_keys:
            img_chunk = src_data[img_key][start_idx:end_idx]
            dst_data[img_key][start_idx:end_idx] = img_chunk
    
    print(f"\n✓ Successfully created 2D dataset at {OUTPUT_PATH}")
    print(f"  Original state dim: {state_dim}")
    print(f"  New state dim: 2")
    print(f"  Total frames: {total_frames}")
    print(f"  Total episodes: {len(episode_ends)}")


if __name__ == "__main__":
    create_2d_dataset()
