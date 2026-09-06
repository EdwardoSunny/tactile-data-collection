import argparse
import re
import zarr
import numpy as np
import cv2
import time

DATA_PATH = "/home/u-ril/edward/phone_data_collection/teleop_data.zarr"

# A bare-camera image key is exactly "img_<digit(s)>" (e.g. img_0, img_1).
# Overlay-augmented arrays are "img_<digit(s)>_<mode>" (e.g. img_0_arrow).
_BARE_IMG_RE = re.compile(r"^img_(\d+)$")
_VARIANT_IMG_RE = re.compile(r"^img_(\d+)_(\w+)$")


def _bare_camera_indices(data):
    """Return sorted list of camera indices that have a bare img_{i} array."""
    idxs = []
    for k in data.keys():
        m = _BARE_IMG_RE.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def _available_modes(data):
    """Return ['raw'] + sorted list of overlay suffixes found in data.
    'raw' always comes first; the others come from the img_{i}_<mode> arrays
    (the same mode set is assumed for every camera that has variants)."""
    modes = set()
    for k in data.keys():
        m = _VARIANT_IMG_RE.match(k)
        if m:
            modes.add(m.group(2))
    return ["raw"] + sorted(modes)


def _img_key_for(cam_idx, mode):
    return f"img_{cam_idx}" if mode == "raw" else f"img_{cam_idx}_{mode}"


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
        start_idx = int(starts[episode_idx])
        end_idx = int(episode_ends[episode_idx])
        episode_length = end_idx - start_idx

        print(f"Deleting episode {episode_idx}: frames {start_idx}-{end_idx} ({episode_length} frames)")

        total_frames = int(data["state"].shape[0])
        new_total_frames = total_frames - episode_length
        print(f"Total frames: {total_frames} -> {new_total_frames}")

        # Snapshot per-array specs (shape/dtype/chunks) for every /data/* array.
        data_keys = list(data.keys())
        data_specs = {}
        for key in data_keys:
            arr = data[key]
            if arr.shape[0] != total_frames:
                print(f"  [skip] /data/{key} has leading dim {arr.shape[0]} != {total_frames}; not a per-frame array")
                continue
            data_specs[key] = {
                "shape": arr.shape,
                "dtype": arr.dtype,
                "chunks": arr.chunks,
            }
        print(f"Per-frame /data arrays to rewrite: {sorted(data_specs.keys())}")

        # Snapshot non-episode_ends meta arrays so we can copy them verbatim.
        meta_keys = [k for k in meta.keys() if k != "episode_ends"]
        print(f"/meta arrays to copy verbatim: {meta_keys}")

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

            backup_store = zarr.open(backup_path, mode="r")
            backup_data = backup_store["data"]
            backup_meta = backup_store["meta"]

            print("Creating new /data arrays...")
            new_arrays = {}
            for key, spec in data_specs.items():
                new_shape = (new_total_frames, *spec["shape"][1:])
                new_arrays[key] = new_data.create_dataset(
                    key,
                    shape=new_shape,
                    dtype=spec["dtype"],
                    chunks=spec["chunks"],
                )
                print(f"  Created /data/{key}: shape={new_shape} dtype={spec['dtype']}")

            def copy_range(src_lo, src_hi, dst_lo):
                """Copy backup_data[k][src_lo:src_hi] -> new_arrays[k][dst_lo:dst_lo+(src_hi-src_lo)] for every key."""
                length = src_hi - src_lo
                if length <= 0:
                    return
                chunk = 1000
                for off in range(0, length, chunk):
                    s0 = src_lo + off
                    s1 = min(s0 + chunk, src_hi)
                    d0 = dst_lo + off
                    d1 = d0 + (s1 - s0)
                    for key in data_specs:
                        new_arrays[key][d0:d1] = backup_data[key][s0:s1]
                    print(f"  Copied src[{s0}:{s1}] -> dst[{d0}:{d1}]")

            print("Copying frames before deleted episode...")
            copy_range(0, start_idx, 0)

            print("Copying frames after deleted episode...")
            copy_range(end_idx, total_frames, start_idx)

            # Rebuild episode_ends without the deleted entry, shifting later ones back.
            new_episode_ends = []
            for i, ep_end in enumerate(episode_ends):
                if i == episode_idx:
                    continue
                elif ep_end > episode_ends[episode_idx]:
                    new_episode_ends.append(int(ep_end) - episode_length)
                else:
                    new_episode_ends.append(int(ep_end))
            new_episode_ends = np.array(new_episode_ends, dtype=episode_ends.dtype)
            new_meta.create_dataset("episode_ends", data=new_episode_ends, chunks=True)
            print(f"Updated episode_ends: {new_episode_ends}")

            # Copy every other /meta/* array verbatim (e.g. tactile_baseline).
            for key in meta_keys:
                src = backup_meta[key]
                new_meta.create_dataset(
                    key,
                    data=np.array(src),
                    dtype=src.dtype,
                    chunks=src.chunks,
                )
                print(f"  Copied /meta/{key} verbatim: shape={src.shape}")

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


def get_episode_data(data, episode_idx, episode_ends, mode="raw"):
    """Return (imgs, states) for one episode.

    `imgs` is the per-frame N-camera frames concatenated horizontally. `mode`
    selects which image array to read for each camera:
        "raw"   -> data["img_{i}"]
        "arrow" -> data["img_{i}_arrow"]   (and same for grid / point / bar)
    Falls back to the legacy single-camera 'img' key only if no img_{i} exists.
    """
    starts = np.concatenate([[0], episode_ends[:-1]])
    start_idx = starts[episode_idx]
    end_idx = episode_ends[episode_idx]
    episode_states = np.array(data["state"][start_idx:end_idx])

    cam_idxs = _bare_camera_indices(data)
    if cam_idxs:
        imgs_list = []
        for i in cam_idxs:
            key = _img_key_for(i, mode)
            if key not in data:
                # Mode unavailable for this camera; fall back to raw.
                key = f"img_{i}"
            imgs_list.append(np.array(data[key][start_idx:end_idx]))
        episode_imgs = np.concatenate(imgs_list, axis=2)
    elif "img" in data:
        episode_imgs = np.array(data["img"][start_idx:end_idx])
    else:
        raise KeyError("dataset has no img_* arrays")

    return episode_imgs, episode_states


def play_dataset_lazy(store, data, meta, mode="raw"):
    episode_ends = np.array(meta["episode_ends"])

    if len(episode_ends) == 0:
        print("❌ No episodes found in dataset!")
        return False

    modes_available = _available_modes(data)
    # Mode keys we listen for. Always include all modes the dataset has so the
    # user can flip between raw + overlays without re-running.
    _MODE_KEYS = {"r": "raw", "a": "arrow", "g": "grid", "p": "point", "b": "bar"}

    ep_idx = 0
    frame_idx = 0
    playing = True

    current_episode_data = None
    current_ep_idx = -1
    current_mode = mode if mode in modes_available else "raw"

    print(f"▶️ Loaded {len(episode_ends)} episodes total")
    print(f"Available image modes: {modes_available}  (currently: {current_mode})")
    print("Controls:")
    print("  SPACE        - pause/play")
    print("  LEFT ARROW   - previous episode")
    print("  RIGHT ARROW  - next episode")
    print("  r / a / g / p / b - switch view: raw / arrow / grid / point / bar")
    print("  'd' - delete current episode")
    print("  'q' - quit")

    while True:
        # Reload when episode OR mode changes.
        if ep_idx != current_ep_idx or current_episode_data is None:
            print(f"Loading episode {ep_idx + 1} (mode={current_mode})...")
            current_episode_data = get_episode_data(
                data, ep_idx, episode_ends, mode=current_mode
            )
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
            f"Episode {ep_idx + 1}/{len(episode_ends)} | Frame {frame_idx + 1}/{episode_length} | mode={current_mode}",
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
        elif key < 128 and chr(key) in _MODE_KEYS:
            requested = _MODE_KEYS[chr(key)]
            if requested in modes_available:
                if requested != current_mode:
                    current_mode = requested
                    current_episode_data = None  # force reload
                    print(f"🎨 Switched view to: {current_mode}")
            else:
                print(f"  [skip] '{requested}' not available in this dataset "
                      f"(have: {modes_available})")
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


def main(path=None, mode="raw"):
    global DATA_PATH
    if path is not None:
        DATA_PATH = path

    try:
        store, data, meta = load_dataset_lazy(DATA_PATH)

        episode_ends = np.array(meta["episode_ends"])
        cam_idxs = _bare_camera_indices(data)
        modes_available = _available_modes(data)

        print(f"📊 Dataset Statistics:")
        print(f"  Path: {DATA_PATH}")
        print(f"  Total frames: {data['state'].shape[0] if 'state' in data else 0}")
        print(f"  Episodes: {len(episode_ends)}")

        if 'state' in data:
            sample_key = f"img_{cam_idxs[0]}" if cam_idxs else ("img" if "img" in data else None)
            shape_str = data[sample_key].shape[1:] if sample_key else "Unknown"
            print(f"  Image shape: {shape_str}")
            print(f"  State dim: {data['state'].shape[1]}")

        if cam_idxs:
            print(f"  Cameras: {cam_idxs}  ({len(cam_idxs)} bare img_* arrays)")
        print(f"  Available view modes: {modes_available}")

        if len(episode_ends) > 0:
            episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
            print(f"  Episode lengths: {episode_lengths}")
            print(f"  Avg episode length: {np.mean(episode_lengths):.1f} frames")
        else:
            print("  No complete episodes recorded yet")

        if len(episode_ends) > 0:
            should_reload = play_dataset_lazy(store, data, meta, mode=mode)
            if should_reload:
                print("🔄 Reloading dataset...")
                main(path=DATA_PATH, mode=mode)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH,
                    help=f"Path to zarr dataset (default: {DATA_PATH})")
    ap.add_argument("--mode", default="raw",
                    choices=["raw", "arrow", "grid", "point", "bar"],
                    help="Initial image mode to display (toggle in-app with r/a/g/p/b)")
    args = ap.parse_args()
    main(path=args.data, mode=args.mode)
