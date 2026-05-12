import torch
import numpy as np
import cv2
import zarr
import sys
import os

# Add parent directory to path to import policy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies import FlowMatchingPolicy
from dataset import unnormalize_data, normalize_data
from environment.utils import from_10d_to_xarm_state, xarm_state_to_10d

CHECKPOINT_PATH = "xarm_teleop_flow_epoch_300.pth"
DATA_PATH = "teleop_data.zarr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INFERENCE_STEPS = 50  # Increased for better quality predictions


def predict_action(policy, robot_state_sequence, image_history):
    """
    Predict action using the policy.
    
    Args:
        policy: FlowMatchingPolicy instance
        robot_state_sequence: List of robot states [obs_horizon, 7]
        image_history: List of image lists [[cam0, cam1], [cam0, cam1], ...]
    
    Returns:
        Predicted actions [pred_horizon, 7]
    """
    return policy.predict_action(
        state_history=robot_state_sequence,
        image_history=image_history,
        num_inference_steps=NUM_INFERENCE_STEPS
    )

def get_episode_boundaries(zarr_path):
    dataset_root = zarr.open(zarr_path, 'r')
    episode_ends = dataset_root['meta']['episode_ends'][:]
    
    episode_starts = [0] + list(episode_ends[:-1])
    episode_lengths = [episode_ends[i] - episode_starts[i] for i in range(len(episode_ends))]
    
    return episode_starts, episode_ends, episode_lengths

def get_episode_data(zarr_path, episode_idx):
    dataset_root = zarr.open(zarr_path, 'r')
    episode_ends = dataset_root['meta']['episode_ends'][:]
    
    start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    
    img_keys = [key for key in dataset_root['data'].keys() if key.startswith('img_')]
    img_keys.sort()
    
    episode_data = {
        'state': dataset_root['data']['state'][start_idx:end_idx],
        'images': {}
    }
    
    for img_key in img_keys:
        episode_data['images'][img_key] = dataset_root['data'][img_key][start_idx:end_idx]
    
    return episode_data

def draw_text_with_background(img, text, position, font_scale=0.6, thickness=2, 
                               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    cv2.rectangle(img, (x, y - text_height - baseline), 
                  (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def visualize_interactive(policy, zarr_path):
    """Interactive visualization of model predictions."""
    obs_horizon = policy.obs_horizon
    pred_horizon = policy.pred_horizon
    stats = policy.stats
    
    if stats is None:
        raise ValueError("Policy does not contain stats. Please retrain or use a checkpoint with stats.")
    
    episode_starts, episode_ends, episode_lengths = get_episode_boundaries(zarr_path)
    num_episodes = len(episode_lengths)
    
    current_episode = 0
    current_frame = 0
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZER")
    print("="*60)
    print("Controls:")
    print("  LEFT/RIGHT arrows: Navigate frames")
    print("  UP/DOWN arrows: Navigate episodes")
    print("  SPACE: Play/Pause")
    print("  Q/ESC: Quit")
    print("="*60 + "\n")
    
    playing = False
    
    while True:
        episode_data = get_episode_data(zarr_path, current_episode)
        episode_length = episode_lengths[current_episode]
        
        if current_frame >= episode_length:
            current_frame = episode_length - 1
        
        # Get obs_horizon frames of state history (keep in original 7D format)
        start_frame = max(0, current_frame - obs_horizon + 1)
        state_frames = []
        for i in range(start_frame, current_frame + 1):
            state_frames.append(episode_data['state'][i])
        
        # Pad if we don't have enough history (at the start of episode)
        while len(state_frames) < obs_horizon:
            state_frames.insert(0, state_frames[0])
        
        # Get obs_horizon frames of images
        image_history = []
        for i in range(start_frame, current_frame + 1):
            images_at_t = []
            for img_key in sorted(episode_data['images'].keys()):
                images_at_t.append(episode_data['images'][img_key][i])
            image_history.append(images_at_t)
        
        # Pad if needed
        while len(image_history) < obs_horizon:
            image_history.insert(0, image_history[0])
        
        # Predict using policy (handles all conversion internally)
        pred_np = predict_action(policy, state_frames, image_history)

        
        gt_actions = []
        for i in range(current_frame, min(current_frame + pred_horizon, episode_length)):
            if i + 1 < episode_length:
                gt_actions.append(episode_data['state'][i + 1])
            else:
                gt_actions.append(episode_data['state'][i])
        
        while len(gt_actions) < pred_horizon:
            gt_actions.append(gt_actions[-1])
        
        gt_np = np.array(gt_actions)
        
        imgs_to_display = []
        for img_key in sorted(episode_data['images'].keys()):
            img = episode_data['images'][img_key][current_frame].copy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            img = cv2.resize(img, (480, 360))
            imgs_to_display.append(img)
        
        if len(imgs_to_display) == 2:
            combined_img = np.hstack(imgs_to_display)
        elif len(imgs_to_display) == 1:
            combined_img = imgs_to_display[0]
        else:
            combined_img = np.hstack(imgs_to_display)
        
        info_height = 400
        info_panel = np.zeros((info_height, combined_img.shape[1], 3), dtype=np.uint8)
        
        y_offset = 30
        draw_text_with_background(info_panel, f"Episode: {current_episode + 1}/{num_episodes}", (10, y_offset))
        draw_text_with_background(info_panel, f"Frame: {current_frame + 1}/{episode_length}", (10, y_offset + 30))
        draw_text_with_background(info_panel, f"{'PLAYING' if playing else 'PAUSED'}", (10, y_offset + 60), 
                                 text_color=(0, 255, 0) if playing else (255, 255, 255))
        
        y_offset = 120
        x_offset = 10
        
        draw_text_with_background(info_panel, "Predicted Action Chunk (16 steps):", (x_offset, y_offset), 
                                 text_color=(0, 150, 255))
        y_offset += 30
        
        show_steps = min(8, pred_horizon)
        for t in range(show_steps):
            pred_str = f"t+{t}: "
            pred_str += f"P[{pred_np[t, 0]:.3f},{pred_np[t, 1]:.3f},{pred_np[t, 2]:.3f}] "
            pred_str += f"R[{pred_np[t, 3]:.3f},{pred_np[t, 4]:.3f},{pred_np[t, 5]:.3f}] "
            pred_str += f"G:{pred_np[t, 6]:.2f}"
            draw_text_with_background(info_panel, pred_str, (x_offset, y_offset), 
                                     font_scale=0.45, thickness=1, text_color=(100, 200, 255))
            y_offset += 18
        
        y_offset = 120
        x_offset = combined_img.shape[1] // 2 + 10
        
        draw_text_with_background(info_panel, "Ground Truth Action Chunk (16 steps):", (x_offset, y_offset), 
                                 text_color=(0, 255, 0))
        y_offset += 30
        
        for t in range(show_steps):
            gt_str = f"t+{t}: "
            gt_str += f"P[{gt_np[t, 0]:.3f},{gt_np[t, 1]:.3f},{gt_np[t, 2]:.3f}] "
            gt_str += f"R[{gt_np[t, 3]:.3f},{gt_np[t, 4]:.3f},{gt_np[t, 5]:.3f}] "
            gt_str += f"G:{gt_np[t, 6]:.2f}"
            draw_text_with_background(info_panel, gt_str, (x_offset, y_offset), 
                                     font_scale=0.45, thickness=1, text_color=(100, 255, 100))
            y_offset += 18
        
        y_offset += 20
        x_offset = 10
        
        current_gt = gt_np[0]
        current_pred = pred_np[0]
        
        draw_text_with_background(info_panel, "Current Step Comparison:", (x_offset, y_offset), 
                                 text_color=(255, 255, 0))
        y_offset += 25
        
        for i in range(6):
            diff = abs(current_pred[i] - current_gt[i])
            color = (0, 255, 0) if diff < 0.01 else (0, 255, 255) if diff < 0.05 else (0, 100, 255)
            draw_text_with_background(info_panel, 
                                     f"Pose[{i}]: Pred={current_pred[i]:.4f} GT={current_gt[i]:.4f} Diff={diff:.4f}", 
                                     (x_offset, y_offset), font_scale=0.45, thickness=1, text_color=color)
            y_offset += 18
        
        diff = abs(current_pred[6] - current_gt[6])
        color = (0, 255, 0) if diff < 0.01 else (0, 255, 255) if diff < 0.05 else (0, 100, 255)
        draw_text_with_background(info_panel, 
                                 f"Grasp: Pred={current_pred[6]:.4f} GT={current_gt[6]:.4f} Diff={diff:.4f}", 
                                 (x_offset, y_offset), font_scale=0.45, thickness=1, text_color=color)
        
        final_display = np.vstack([combined_img, info_panel])
        
        cv2.imshow('Interactive Prediction Viewer', final_display)
        
        wait_time = 33 if playing else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
        elif key == 81 or key == 2:
            current_frame = max(0, current_frame - 1)
        elif key == 83 or key == 3:
            current_frame = min(episode_length - 1, current_frame + 1)
        elif key == 82 or key == 0:
            current_episode = max(0, current_episode - 1)
            current_frame = 0
        elif key == 84 or key == 1:
            current_episode = min(num_episodes - 1, current_episode + 1)
            current_frame = 0
        
        if playing:
            current_frame += 1
            if current_frame >= episode_length:
                current_frame = 0
                current_episode = (current_episode + 1) % num_episodes
    
    cv2.destroyAllWindows()

def main():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    print(f"Using device: {DEVICE}")
    
    # Load policy from checkpoint
    policy = FlowMatchingPolicy.load_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    policy.eval()
    
    print(f"Policy loaded successfully!")
    print(f"Robot state dim: {policy.robot_state_dim}")
    print(f"Action dim: {policy.action_dim}")
    print(f"Number of cameras: {policy.num_cameras}")
    print(f"Obs horizon: {policy.obs_horizon}")
    print(f"Pred horizon: {policy.pred_horizon}")
    
    if policy.stats is not None:
        print("✓ Stats found in policy")
    else:
        print("✗ Warning: No stats found in policy. This may cause errors.")
    
    visualize_interactive(policy, DATA_PATH)

if __name__ == "__main__":
    main()
