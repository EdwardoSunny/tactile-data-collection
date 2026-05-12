"""
Count statistics about the dataset.
Shows total frames, episodes, and frames per episode.
"""

import zarr
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "pushT_data.zarr"


def count_dataset_stats(zarr_path):
    """Count and display dataset statistics."""
    
    print(f"Analyzing dataset: {zarr_path}")
    print("=" * 60)
    
    dataset_root = zarr.open(zarr_path, 'r')
    
    # Get episode information
    episode_ends = dataset_root['meta']['episode_ends'][:]
    num_episodes = len(episode_ends)
    
    # Calculate episode starts and lengths
    episode_starts = [0] + list(episode_ends[:-1])
    episode_lengths = [episode_ends[i] - episode_starts[i] for i in range(num_episodes)]
    
    # Total data points
    total_frames = episode_ends[-1]
    
    # State dimension
    state_shape = dataset_root['data']['state'].shape
    state_dim = state_shape[1] if len(state_shape) > 1 else 1

    print(f"Dataset keys: {list(dataset_root['data'].keys())}")
    
    # Image information
    img_keys = [key for key in dataset_root['data'].keys() if key.startswith('img_')]
    img_keys.sort()
    num_cameras = len(img_keys)
    
    if num_cameras > 0:
        sample_img = dataset_root['data'][img_keys[0]][0]
        img_shape = sample_img.shape
    else:
        img_shape = None
    
    # Print statistics
    print(f"\n📊 DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Data Points (Frames): {total_frames:,}")
    print(f"Total Episodes: {num_episodes}")
    print(f"\nState Dimension: {state_dim}")
    print(f"Number of Cameras: {num_cameras}")
    if img_shape is not None:
        print(f"Image Shape: {img_shape}")
    
    print(f"\n📈 EPISODE STATISTICS")
    print("=" * 60)
    print(f"Average frames per episode: {np.mean(episode_lengths):.1f}")
    print(f"Median frames per episode: {np.median(episode_lengths):.1f}")
    print(f"Min frames per episode: {np.min(episode_lengths)}")
    print(f"Max frames per episode: {np.max(episode_lengths)}")
    print(f"Std dev frames per episode: {np.std(episode_lengths):.1f}")
    
    # Distribution
    print(f"\n📉 EPISODE LENGTH DISTRIBUTION")
    print("=" * 60)
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    
    for i in range(len(bins) - 1):
        count = sum(1 for length in episode_lengths if bins[i] < length <= bins[i+1])
        if count > 0:
            percentage = (count / num_episodes) * 100
            print(f"  {labels[i]:>10} frames: {count:4d} episodes ({percentage:5.1f}%)")
    
    # Show first and last few episodes
    print(f"\n📋 EPISODE DETAILS (First 10)")
    print("=" * 60)
    print("Episode | Start Frame | End Frame | Length")
    print("-" * 60)
    for i in range(min(10, num_episodes)):
        print(f"{i:7d} | {episode_starts[i]:11d} | {episode_ends[i]:9d} | {episode_lengths[i]:6d}")
    
    if num_episodes > 10:
        print("...")
        print(f"\n📋 EPISODE DETAILS (Last 10)")
        print("=" * 60)
        print("Episode | Start Frame | End Frame | Length")
        print("-" * 60)
        for i in range(max(0, num_episodes - 10), num_episodes):
            print(f"{i:7d} | {episode_starts[i]:11d} | {episode_ends[i]:9d} | {episode_lengths[i]:6d}")
    
    print("\n" + "=" * 60)
    
    return {
        'total_frames': total_frames,
        'num_episodes': num_episodes,
        'state_dim': state_dim,
        'num_cameras': num_cameras,
        'img_shape': img_shape,
        'episode_lengths': episode_lengths
    }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Count data points in dataset')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to zarr dataset')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset not found: {args.data}")
        return 1
    
    stats = count_dataset_stats(args.data)
    
    return 0


if __name__ == "__main__":
    exit(main())
