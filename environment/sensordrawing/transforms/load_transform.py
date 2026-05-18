#!/usr/bin/env python3
"""
Load a transform from transforms/transforms.npy and return a 4x4 matrix.

Behavior:
- If the saved entry contains 'trc' (robot -> camera), return that (augmented to 4x4).
- Else if it contains 'tcr' (camera -> robot), return inverse(augment(tcr)).
- Else if entry['mode'] == 'eye_in_hand', prefer 'tce' (ee -> camera) and return that
  but note this is end-effector->camera (not base->camera) and must be composed
  with a robot base->ee pose to obtain base->camera.

Usage:
    python load_transform.py [SERIAL]
If no SERIAL is given, the first key in the saved dict is used.
"""

from pathlib import Path
import numpy as np
import sys


def _augment_3x4_to_4x4(T34):
    T = np.eye(4, dtype=float)
    T[:3, :4] = T34
    return T


def load_transform(serial=None, transform_dir=None):
    """Load transforms/transforms.npy and return a (T, serial, info) tuple.

    Returns:
        T: np.ndarray shape (4,4) — the transform matrix returned according to
           the rules in the module docstring. Use the returned `info` string to
           understand what frame mapping T actually performs.
        serial: the serial used (string)
        info: short description of which stored key was used and what T maps.
    """
    base_dir = Path(transform_dir) if transform_dir is not None else Path(__file__).resolve().parent
    tf_file = base_dir / "transforms.npy"

    if not tf_file.exists():
        raise FileNotFoundError(f"Transform file not found: {tf_file}")

    data = np.load(tf_file, allow_pickle=True).item()
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError(f"Transforms file does not contain a dict or is empty: {tf_file}")

    if serial is None:
        # pick the first key
        serial = next(iter(data.keys()))

    if serial not in data:
        raise KeyError(f"Serial {serial} not found in {tf_file}. Available: {list(data.keys())}")

    entry = data[serial]

    # Prefer trc (robot -> camera) if present
    if isinstance(entry, dict) and 'trc' in entry:
        T = _augment_3x4_to_4x4(np.asarray(entry['trc'], dtype=float))
        info = "trc (robot->camera) — augmented 3x4 -> 4x4"
        return T, serial, info

    # If not, look for tcr (camera -> robot) and invert
    if isinstance(entry, dict) and 'tcr' in entry:
        T_cam_to_robot = _augment_3x4_to_4x4(np.asarray(entry['tcr'], dtype=float))
        T = np.linalg.inv(T_cam_to_robot)
        info = "tcr (camera->robot) inverted to robot->camera"
        return T, serial, info

    # Eye-in-hand mode: saved maps are between camera and end-effector
    if isinstance(entry, dict) and entry.get('mode') == 'eye_in_hand':
        # prefer tce (end-effector -> camera) if present
        if 'tce' in entry:
            T_ee_to_cam = _augment_3x4_to_4x4(np.asarray(entry['tce'], dtype=float))
            info = "tce (end-effector->camera). NOTE: this is NOT base->camera; compose with T_base_to_ee"
            return T_ee_to_cam, serial, info
        # else maybe tec is present (camera->ee)
        if 'tec' in entry:
            T_cam_to_ee = _augment_3x4_to_4x4(np.asarray(entry['tec'], dtype=float))
            T_ee_to_cam = np.linalg.inv(T_cam_to_ee)
            info = "tec (camera->end-effector) inverted to end-effector->camera. NOTE: NOT base->camera"
            return T_ee_to_cam, serial, info

    raise RuntimeError(f"Could not determine a usable transform for serial {serial} from {tf_file}")


if __name__ == '__main__':
    serial_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        T, serial_used, info = load_transform(serial=serial_arg)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print(f"Serial: {serial_used}")
    print(f"Info: {info}")
    # pretty-print the matrix
    np.set_printoptions(precision=6, suppress=True)
    print("Transform (4x4):\n", T)
