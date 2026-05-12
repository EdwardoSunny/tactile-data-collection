import zarr
import numpy as np
import sys

def repack_zarr(input_path, output_path=None, chunk_size=1000):
    if output_path is None:
        output_path = input_path.replace(".zarr", "_repacked.zarr")
    
    print(f"Opening source dataset: {input_path}")
    src_store = zarr.open(input_path, mode="r")
    src_data = src_store["data"]
    src_meta = src_store["meta"]
    
    total_frames = src_data["state"].shape[0]
    state_dim = src_data["state"].shape[1]
    episode_ends = np.array(src_meta["episode_ends"])
    
    print(f"Total frames: {total_frames}")
    print(f"Total episodes: {len(episode_ends)}")
    print(f"State dimension: {state_dim}")
    
    all_keys = list(src_data.keys())
    print(f"Found data keys: {all_keys}")
    
    print(f"\nCreating optimized dataset: {output_path}")
    dst_store = zarr.open(output_path, mode="w")
    dst_data = dst_store.create_group("data")
    dst_meta = dst_store.create_group("meta")
    
    print("Creating datasets with optimized chunks and compression...")
    
    for key in all_keys:
        if key == 'n_contacts':
            print(f"  Skipping {key} (removed)")
            continue
            
        shape = src_data[key].shape
        dtype = src_data[key].dtype
        
        if key.startswith('img'):
            chunks = (32, *shape[1:])
            print(f"  {key}: shape={shape}, chunks={chunks}, compression=blosc")
            dst_data.create_dataset(
                key, 
                shape=shape, 
                chunks=chunks, 
                dtype=dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
            )
        elif key == 'state':
            chunks = (2048, state_dim)
            print(f"  {key}: shape={shape}, chunks={chunks}, compression=blosc")
            dst_data.create_dataset(
                key,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
            )
        elif key == 'action':
            chunks = (2048, shape[1])
            print(f"  {key}: shape={shape}, chunks={chunks}, compression=blosc")
            dst_data.create_dataset(
                key,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
            )
        else:
            print(f"  {key}: shape={shape}, default chunks")
            dst_data.create_dataset(
                key,
                shape=shape,
                dtype=dtype
            )
    
    print("Copying episode_ends...")
    dst_meta.create_dataset("episode_ends", data=episode_ends, chunks=(1024,), dtype=np.int64)
    
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    print(f"\nCopying data in {num_chunks} chunks of {chunk_size} frames...")
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        
        print(f"  Chunk {i+1}/{num_chunks}: frames {start_idx}-{end_idx}")
        
        for key in all_keys:
            data_chunk = src_data[key][start_idx:end_idx]
            dst_data[key][start_idx:end_idx] = data_chunk
    
    print(f"\n✓ Successfully repacked dataset to {output_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total episodes: {len(episode_ends)}")
    
    import os
    if os.path.exists(input_path):
        src_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, _, filenames in os.walk(input_path)
                      for filename in filenames) / (1024 * 1024)
        print(f"  Original size: {src_size:.2f} MB")
    
    if os.path.exists(output_path):
        dst_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, _, filenames in os.walk(output_path)
                      for filename in filenames) / (1024 * 1024)
        print(f"  Repacked size: {dst_size:.2f} MB")
        if src_size > 0:
            print(f"  Compression: {(1 - dst_size/src_size) * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repack_zarr.py <input.zarr> [output.zarr]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    repack_zarr(input_path, output_path)
