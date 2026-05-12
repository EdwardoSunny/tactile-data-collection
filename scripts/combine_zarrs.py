import zarr
import numpy as np

INPUT_PATH_1 = "pushT_data.zarr"
INPUT_PATH_2 = "teleop_data_2d.zarr"
OUTPUT_PATH = "combined_data.zarr"


def combine_zarrs():
    print(f"Opening first dataset: {INPUT_PATH_1}")
    src_store_1 = zarr.open(INPUT_PATH_1, mode="r")
    src_data_1 = src_store_1["data"]
    src_meta_1 = src_store_1["meta"]
    
    print(f"Opening second dataset: {INPUT_PATH_2}")
    src_store_2 = zarr.open(INPUT_PATH_2, mode="r")
    src_data_2 = src_store_2["data"]
    src_meta_2 = src_store_2["meta"]
    
    # Get dimensions from first dataset
    total_frames_1 = src_data_1["state"].shape[0]
    total_frames_2 = src_data_2["state"].shape[0]
    total_frames = total_frames_1 + total_frames_2
    
    state_dim = src_data_1["state"].shape[1]
    
    episode_ends_1 = np.array(src_meta_1["episode_ends"])
    episode_ends_2 = np.array(src_meta_2["episode_ends"])
    
    # Offset second dataset's episode ends by the total frames from first dataset
    episode_ends_2_offset = episode_ends_2 + total_frames_1
    episode_ends_combined = np.concatenate([episode_ends_1, episode_ends_2_offset])
    
    print(f"\nDataset 1:")
    print(f"  State shape: ({total_frames_1}, {state_dim})")
    print(f"  Episodes: {len(episode_ends_1)}")
    
    print(f"\nDataset 2:")
    print(f"  State shape: ({total_frames_2}, {state_dim})")
    print(f"  Episodes: {len(episode_ends_2)}")
    
    print(f"\nCombined dataset:")
    print(f"  Total frames: {total_frames}")
    print(f"  Total episodes: {len(episode_ends_combined)}")
    
    # Find all image keys (img_0, img_1, etc.)
    img_keys = [key for key in src_data_1.keys() if key.startswith('img')]
    img_shapes = {key: src_data_1[key].shape[1:] for key in img_keys}
    print(f"  Image keys: {img_keys}")
    
    # Create output dataset
    print(f"\nCreating combined dataset: {OUTPUT_PATH}")
    dst_store = zarr.open(OUTPUT_PATH, mode="w")
    dst_data = dst_store.create_group("data")
    dst_meta = dst_store.create_group("meta")
    
    # Create state dataset
    print("Creating state dataset...")
    dst_data.create_dataset("state", shape=(total_frames, state_dim), chunks=(1024, state_dim), dtype=np.float32)
    
    # Create image datasets
    for img_key in img_keys:
        print(f"Creating {img_key} dataset...")
        dst_data.create_dataset(img_key, shape=(total_frames, *img_shapes[img_key]), chunks=(64, *img_shapes[img_key]), dtype=np.float32)
    
    # Copy episode_ends
    print("Copying episode_ends...")
    dst_meta.create_dataset("episode_ends", data=episode_ends_combined, chunks=(1024,), dtype=np.int64)
    
    # Copy data from first dataset
    chunk_size = 1000
    print(f"\nCopying data from dataset 1...")
    num_chunks_1 = (total_frames_1 + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks_1):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames_1)
        
        print(f"  Chunk {i+1}/{num_chunks_1}: frames {start_idx}-{end_idx}")
        
        # Copy state
        state_chunk = src_data_1["state"][start_idx:end_idx]
        dst_data["state"][start_idx:end_idx] = state_chunk
        
        # Copy images
        for img_key in img_keys:
            img_chunk = src_data_1[img_key][start_idx:end_idx]
            dst_data[img_key][start_idx:end_idx] = img_chunk
    
    # Copy data from second dataset
    print(f"\nCopying data from dataset 2...")
    num_chunks_2 = (total_frames_2 + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks_2):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames_2)
        dst_start_idx = total_frames_1 + start_idx
        dst_end_idx = total_frames_1 + end_idx
        
        print(f"  Chunk {i+1}/{num_chunks_2}: frames {start_idx}-{end_idx} -> {dst_start_idx}-{dst_end_idx}")
        
        # Copy state
        state_chunk = src_data_2["state"][start_idx:end_idx]
        dst_data["state"][dst_start_idx:dst_end_idx] = state_chunk
        
        # Copy images
        for img_key in img_keys:
            img_chunk = src_data_2[img_key][start_idx:end_idx]
            dst_data[img_key][dst_start_idx:dst_end_idx] = img_chunk
    
    print(f"\n✓ Successfully created combined dataset at {OUTPUT_PATH}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total episodes: {len(episode_ends_combined)}")
    print(f"  State dimension: {state_dim}")
    print(f"  Image keys: {img_keys}")


if __name__ == "__main__":
    combine_zarrs()
