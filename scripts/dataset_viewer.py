import zarr
import numpy as np
import cv2
import time

DATA_PATH = "teleop_data.zarr"


def delete_episode(episode_idx):
    import shutil
    import os
    
    try:
        print(f"Opening dataset for deletion...")
        store = zarr.open(DATA_PATH, mode="r")
        data = store["data"] 
        meta = store["meta"]
        
        episode_ends = np.array(meta["episode_ends"])
        print(f"Found {len(episode_ends)} episodes")
        
        if episode_idx >= len(episode_ends):
            print(f"Episode index {episode_idx} out of range")
            return False
            
        starts = np.concatenate([[0], episode_ends[:-1]])
        start_idx = starts[episode_idx]
        end_idx = episode_ends[episode_idx]
        episode_length = end_idx - start_idx
        
        print(f"Deleting episode {episode_idx}: frames {start_idx}-{end_idx} ({episode_length} frames)")
        
        total_frames = data["state"].shape[0]
        state_shape = data["state"].shape[1:]
        
        print(f"Total frames: {total_frames}")
        print(f"State shape: {state_shape}")
        
        img_keys = [key for key in data.keys() if key.startswith('img')]
        print(f"Found image keys: {img_keys}")
        
        img_shapes = {}
        for img_key in img_keys:
            img_shapes[img_key] = data[img_key].shape[1:]
            print(f"{img_key} shape: {data[img_key].shape}")
        
        del store, data, meta
        
        backup_path = DATA_PATH + ".backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(DATA_PATH, backup_path)
        print(f"Created backup at {backup_path}")
        
        try:
            shutil.rmtree(DATA_PATH)
            
            new_store = zarr.open(DATA_PATH, mode="w")
            new_data = new_store.create_group("data")
            new_meta = new_store.create_group("meta")
            
            print("Creating new zarr arrays...")
            
            new_total_frames = total_frames - episode_length
            print(f"New total frames: {new_total_frames}")
            
            try:
                new_state = new_data.create_dataset("state", shape=(new_total_frames, *state_shape), chunks=True)
                print(f"Created state array: {new_state.shape}")
            except Exception as e:
                print(f"Error creating state array: {e}")
                raise
            
            try:
                backup_store = zarr.open(backup_path, mode="r")
                backup_data = backup_store["data"]
                
                if "n_contacts" in backup_data:
                    n_contacts_shape = backup_data["n_contacts"].shape[1:]
                    new_n_contacts = new_data.create_dataset("n_contacts", shape=(new_total_frames, *n_contacts_shape), chunks=True)
                else:
                    new_n_contacts = new_data.create_dataset("n_contacts", shape=(new_total_frames, 1), chunks=True)
                print(f"Created n_contacts array: {new_n_contacts.shape}")
            except Exception as e:
                print(f"Error creating n_contacts array: {e}")
                raise
            
            new_img_arrays = {}
            for img_key in img_keys:
                try:
                    img_shape = img_shapes[img_key]
                    print(f"Creating {img_key} with shape {(new_total_frames, *img_shape)}")
                    new_img_arrays[img_key] = new_data.create_dataset(img_key, shape=(new_total_frames, *img_shape), chunks=True)
                    print(f"Created {img_key} array with shape {new_img_arrays[img_key].shape}")
                except Exception as e:
                    print(f"Error creating {img_key} array: {e}")
                    print(f"Image shape was: {img_shape}")
                    raise
            
            print("Copying data before deleted episode...")
            
            if start_idx > 0:
                chunk_size = min(1000, start_idx)
                for i in range(0, start_idx, chunk_size):
                    end_chunk = min(i + chunk_size, start_idx)
                    
                    new_state[i:end_chunk] = backup_data["state"][i:end_chunk]
                    
                    if "n_contacts" in backup_data:
                        new_n_contacts[i:end_chunk] = backup_data["n_contacts"][i:end_chunk]
                    else:
                        new_n_contacts[i:end_chunk] = 0
                    
                    for img_key in img_keys:
                        new_img_arrays[img_key][i:end_chunk] = backup_data[img_key][i:end_chunk]
                    
                    print(f"  Copied chunk {i}:{end_chunk}")
            
            print("Copying data after deleted episode...")
            
            if end_idx < total_frames:
                dest_start = start_idx
                remaining_frames = total_frames - end_idx
                
                chunk_size = min(1000, remaining_frames)
                for i in range(0, remaining_frames, chunk_size):
                    src_start = end_idx + i
                    src_end = min(src_start + chunk_size, total_frames)
                    dest_end = dest_start + (src_end - src_start)
                    
                    new_state[dest_start:dest_end] = backup_data["state"][src_start:src_end]
                    
                    if "n_contacts" in backup_data:
                        new_n_contacts[dest_start:dest_end] = backup_data["n_contacts"][src_start:src_end]
                    else:
                        new_n_contacts[dest_start:dest_end] = 0
                    
                    for img_key in img_keys:
                        new_img_arrays[img_key][dest_start:dest_end] = backup_data[img_key][src_start:src_end]
                    
                    dest_start = dest_end
                    print(f"  Copied chunk {src_start}:{src_end} -> {dest_start-(src_end-src_start)}:{dest_start}")
            
            new_episode_ends = []
            for i, ep_end in enumerate(episode_ends):
                if i == episode_idx:
                    continue
                elif ep_end > episode_ends[episode_idx]:
                    new_episode_ends.append(ep_end - episode_length)
                else:
                    new_episode_ends.append(ep_end)
            
            new_episode_ends = np.array(new_episode_ends)
            new_meta.create_dataset("episode_ends", data=new_episode_ends, chunks=True)
            print(f"Updated episode ends: {new_episode_ends}")
            
            shutil.rmtree(backup_path)
            print("Successfully deleted episode and cleaned up backup")
            return True
            
        except Exception as e:
            print(f"Error recreating dataset: {e}")
            if os.path.exists(backup_path):
                if os.path.exists(DATA_PATH):
                    shutil.rmtree(DATA_PATH)
                shutil.move(backup_path, DATA_PATH)
                print("Restored backup due to error")
            return False
            
    except Exception as e:
        print(f"Error deleting episode: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_dataset_lazy(path):
    store = zarr.open(path, mode="r")
    data = store["data"]
    meta = store["meta"]
    return store, data, meta


def get_episode_data(data, episode_idx, episode_ends):
    starts = np.concatenate([[0], episode_ends[:-1]])
    start_idx = starts[episode_idx]
    end_idx = episode_ends[episode_idx]
    print(list(data.keys()))
    episode_states = np.array(data["state"][start_idx:end_idx])
    
    img_keys = [key for key in data.keys() if key.startswith('img')]
    if 'img_0' in img_keys:
        imgs_list = []
        for i in range(len([k for k in img_keys if k.startswith('img_')])):
            imgs_list.append(np.array(data[f"img_{i}"][start_idx:end_idx]))
        episode_imgs = np.concatenate(imgs_list, axis=2)
    else:
        episode_imgs = np.array(data["img"][start_idx:end_idx])
    
    return episode_imgs, episode_states


def play_dataset_lazy(store, data, meta):
    episode_ends = np.array(meta["episode_ends"])
    
    if len(episode_ends) == 0:
        print("❌ No episodes found in dataset!")
        return False

    ep_idx = 0
    frame_idx = 0
    playing = True
    
    current_episode_data = None
    current_ep_idx = -1

    print(f"▶️ Loaded {len(episode_ends)} episodes total")
    print("Controls:")
    print("  SPACE - pause/play")
    print("  LEFT ARROW - previous episode")
    print("  RIGHT ARROW - next episode") 
    print("  'd' - delete current episode")
    print("  'q' - quit")

    while True:
        if ep_idx != current_ep_idx:
            print(f"Loading episode {ep_idx + 1}...")
            current_episode_data = get_episode_data(data, ep_idx, episode_ends)
            current_ep_idx = ep_idx
            frame_idx = 0
            
        if current_episode_data is None:
            break
            
        imgs, states = current_episode_data
        episode_length = len(imgs)
        
        if frame_idx >= episode_length:
            frame_idx = 0
            
        state = states[frame_idx]
        pose = state[:]
        # grasp = state[-1]
        
        frame = np.clip(imgs[frame_idx] * 255, 0, 255).astype(np.uint8)
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        text_lines = [
            f"Episode {ep_idx + 1}/{len(episode_ends)} | Frame {frame_idx + 1}/{episode_length}",
            f"Pose: [{', '.join([f'{p:.1f}' for p in pose])}]",
            # f"Grasp: {'CLOSED' if grasp > 0.5 else 'OPEN'} ({grasp:.2f})"
        ]
        
        y_offset = 25
        for i, text in enumerate(text_lines):
            cv2.putText(display_frame, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Dataset Playback", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
            print(f"{'▶️ Playing' if playing else '⏸️ Paused'}")
        elif key == 83:
            ep_idx = (ep_idx + 1) % len(episode_ends)
            print(f"➡️ Episode {ep_idx + 1}/{len(episode_ends)}")
        elif key == 81:
            ep_idx = (ep_idx - 1) % len(episode_ends)
            print(f"⬅️ Episode {ep_idx + 1}/{len(episode_ends)}")
        elif key == ord('d'):
            print(f"⚠️ Delete episode {ep_idx + 1}? Press 'y' to confirm, any other key to cancel")
            confirm_key = cv2.waitKey(0) & 0xFF
            if confirm_key == ord('y'):
                print(f"🗑️ Deleting episode {ep_idx + 1}...")
                if delete_episode(ep_idx):
                    print("✅ Episode deleted! Reloading dataset...")
                    return True
                else:
                    print("❌ Failed to delete episode")
            else:
                print("❌ Delete cancelled")
        
        if playing:
            frame_idx += 1
            if frame_idx >= episode_length:
                frame_idx = 0

    cv2.destroyAllWindows()
    return False


def main():
    try:
        store, data, meta = load_dataset_lazy(DATA_PATH)
        
        episode_ends = np.array(meta["episode_ends"])
        
        print(f"📊 Dataset Statistics:")
        print(f"  Total frames: {data['state'].shape[0] if 'state' in data else 0}")
        print(f"  Episodes: {len(episode_ends)}")
        
        if 'state' in data:
            print(f"  Image shape: {data['img_0'].shape[1:] if 'img_0' in data else data['img'].shape[1:] if 'img' in data else 'Unknown'}")
            print(f"  State dim: {data['state'].shape[1]}")
        
        img_keys = [key for key in data.keys() if key.startswith('img')]
        if 'img_0' in img_keys:
            print(f"  Number of cameras: {len([k for k in img_keys if k.startswith('img_')])}")
        
        if len(episode_ends) > 0:
            episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
            print(f"  Episode lengths: {episode_lengths}")
            print(f"  Avg episode length: {np.mean(episode_lengths):.1f} frames")
        else:
            print("  No complete episodes recorded yet")
        
        if len(episode_ends) > 0:
            should_reload = play_dataset_lazy(store, data, meta)
            if should_reload:
                print("🔄 Reloading dataset...")
                main()
        else:
            print("❌ No episodes found in dataset")
        
    except FileNotFoundError:
        print(f"❌ Dataset not found: {DATA_PATH}")
        print("Make sure you've recorded some data first using collect.py")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
