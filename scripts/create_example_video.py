"""
Create example video from 5 different episodes in the dataset.
Shows cameras side-by-side with no overlay text.
"""

import zarr
import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "teleop_data.zarr"
OUTPUT_VIDEO = "example.mp4"
NUM_EPISODES = 1
FPS = 30  # Frames per second for output video
MAX_FRAMES_PER_EPISODE = 10009  # Limit frames per episode for faster processing


def get_episode_boundaries(zarr_path):
    """Get episode start and end indices."""
    dataset_root = zarr.open(zarr_path, 'r')
    episode_ends = dataset_root['meta']['episode_ends'][:]
    
    episode_starts = [0] + list(episode_ends[:-1])
    episode_lengths = [episode_ends[i] - episode_starts[i] for i in range(len(episode_ends))]
    
    return episode_starts, episode_ends, episode_lengths


def get_episode_frames(zarr_path, episode_idx, max_frames=None):
    """Get all frames from an episode."""
    dataset_root = zarr.open(zarr_path, 'r')
    episode_ends = dataset_root['meta']['episode_ends'][:]
    
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    
    # Limit frames if requested
    if max_frames is not None:
        episode_length = end_idx - start_idx
        if episode_length > max_frames:
            end_idx = start_idx + max_frames
    
    # Get image keys
    img_keys = [key for key in dataset_root['data'].keys() if key.startswith('img_')]
    img_keys.sort()
    
    print(f"  Loading frames {start_idx} to {end_idx}...")
    
    # Load all frames at once (much faster than one-by-one)
    frames = []
    camera_data = []
    for img_key in img_keys:
        imgs = dataset_root['data'][img_key][start_idx:end_idx]
        camera_data.append(imgs)
    
    # Process frames
    num_frames = end_idx - start_idx
    for frame_idx in range(num_frames):
        camera_imgs = []
        for cam_idx, imgs in enumerate(camera_data):
            img = imgs[frame_idx]
            
            # Convert to uint8 if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # Ensure 3 channels BGR for OpenCV
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            # If already 3 channels, keep as is (assume already BGR from cameras)
            
            camera_imgs.append(img)
        
        # Concatenate cameras horizontally
        combined_frame = np.hstack(camera_imgs)
        frames.append(combined_frame)
    
    return frames


def create_example_video(zarr_path, output_path, num_episodes=5, fps=10, max_frames_per_episode=None):
    """Create video from multiple episodes."""
    
    print(f"Opening dataset: {zarr_path}")
    episode_starts, episode_ends, episode_lengths = get_episode_boundaries(zarr_path)
    total_episodes = len(episode_lengths)
    
    print(f"Found {total_episodes} episodes")
    
    if total_episodes < num_episodes:
        print(f"Warning: Only {total_episodes} episodes available, using all of them")
        num_episodes = total_episodes
    
    # Select episodes - evenly spaced throughout dataset
    if num_episodes == 1:
        selected_episodes = [0]
    else:
        step = (total_episodes - 1) / (num_episodes - 1)
        selected_episodes = [int(i * step) for i in range(num_episodes)]
    
    print(f"Selected episodes: {selected_episodes}")
    if max_frames_per_episode:
        print(f"Limiting to {max_frames_per_episode} frames per episode")
    
    # Collect all frames from selected episodes
    all_frames = []
    for ep_idx in selected_episodes:
        print(f"Processing episode {ep_idx} ({episode_lengths[ep_idx]} frames)...")
        frames = get_episode_frames(zarr_path, ep_idx, max_frames_per_episode)
        all_frames.extend(frames)
    
    if not all_frames:
        print("Error: No frames collected!")
        return False
    
    # Get video dimensions from first frame
    height, width = all_frames[0].shape[:2]
    
    print(f"Creating video: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(all_frames)/fps:.1f}s")
    
    # Try different codecs for better compatibility
    # First try MJPEG (most reliable)
    output_path_avi = output_path.replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(
        output_path_avi,
        fourcc,
        float(fps),
        (width, height),
        isColor=True
    )
    
    if not video_writer.isOpened():
        print("Error: Failed to create video writer!")
        return False
    
    print(f"Using MJPEG codec, saving as: {output_path_avi}")
    
    # Write all frames
    print("Writing frames...")
    for i, frame in enumerate(all_frames):
        # Ensure frame is contiguous and correct format
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        video_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  Written {i + 1}/{len(all_frames)} frames")
    
    video_writer.release()
    print(f"✅ Video saved successfully: {output_path_avi}")
    print(f"💡 Note: Saved as AVI with MJPEG codec for reliability")
    print(f"   To convert to MP4: ffmpeg -i {output_path_avi} -c:v libx264 -preset fast -crf 23 {output_path}")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create example video from dataset episodes')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to zarr dataset')
    parser.add_argument('--output', type=str, default=OUTPUT_VIDEO, help='Output video path')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Number of episodes to include')
    parser.add_argument('--fps', type=int, default=FPS, help='Output video FPS')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset not found: {args.data}")
        return 1
    
    success = create_example_video(args.data, args.output, args.episodes, args.fps)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
