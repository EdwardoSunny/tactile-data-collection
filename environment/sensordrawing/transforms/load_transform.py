#!/usr/bin/env python3
"""
Load a transform from transforms/transforms.npz (preferred) or transforms.npy
(legacy) and return a 4x4 matrix.

The .npz format is pickle-free and works under both numpy 1.x and 2.x. The
legacy .npy format pickles a Python dict, which breaks cross-numpy-version
loading (numpy 2.x writes a `numpy._core.*` reference that numpy 1.x cannot
resolve). New installs ship transforms.npz only.

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


def _decode_str(val):
    """Lift `np.bytes_` / `np.ndarray('S...')` / bytes back to a plain str."""
    if isinstance(val, np.ndarray):
        val = val.item() if val.shape == () else val[0]
    if isinstance(val, bytes):
        return val.decode("utf-8").rstrip("\x00")
    return str(val)


def _load_from_npz(npz_file, serial):
    """transforms.npz layout (flat, pickle-free):
        _serials              : list of available serial strings (S64)
        <serial>__trc         : (3, 4) float64
        <serial>__tcr         : (3, 4) float64
        <serial>__tce         : (3, 4) float64    (eye-in-hand only)
        <serial>__tec         : (3, 4) float64    (eye-in-hand only)
        <serial>__mode        : S64 string        (e.g. 'eye_in_hand')
    """
    data = np.load(npz_file)  # no allow_pickle needed
    serials_arr = data.get("_serials", None)
    if serials_arr is not None:
        serials = [_decode_str(s) for s in serials_arr]
    else:
        # Fallback: infer serials from key prefixes.
        serials = sorted({k.split("__", 1)[0] for k in data.files if "__" in k})

    if not serials:
        raise RuntimeError(f"Transforms file has no recognizable serials: {npz_file}")

    if serial is None:
        serial = serials[0]
    if serial not in serials:
        raise KeyError(f"Serial {serial} not found in {npz_file}. Available: {serials}")

    def has(key):
        return f"{serial}__{key}" in data.files

    def get(key):
        return np.asarray(data[f"{serial}__{key}"], dtype=float)

    if has("trc"):
        return _augment_3x4_to_4x4(get("trc")), serial, "trc (robot->camera) — augmented 3x4 -> 4x4"

    if has("tcr"):
        T_cam_to_robot = _augment_3x4_to_4x4(get("tcr"))
        return np.linalg.inv(T_cam_to_robot), serial, "tcr (camera->robot) inverted to robot->camera"

    mode = _decode_str(data[f"{serial}__mode"]) if has("mode") else None
    if mode == "eye_in_hand":
        if has("tce"):
            return (_augment_3x4_to_4x4(get("tce")), serial,
                    "tce (end-effector->camera). NOTE: not base->camera; compose with T_base_to_ee")
        if has("tec"):
            T_cam_to_ee = _augment_3x4_to_4x4(get("tec"))
            return (np.linalg.inv(T_cam_to_ee), serial,
                    "tec (camera->end-effector) inverted to end-effector->camera. NOTE: not base->camera")

    raise RuntimeError(f"Could not determine a usable transform for serial {serial} from {npz_file}")


def _load_from_npy(npy_file, serial):
    """Legacy path: numpy-pickled dict. Cross-version-fragile; kept only as a
    fallback for already-deployed datasets. Will fail with 'No module named
    numpy._core' when a numpy-2.x-saved file is loaded under numpy 1.x."""
    data = np.load(npy_file, allow_pickle=True).item()
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError(f"Transforms file does not contain a dict or is empty: {npy_file}")

    if serial is None:
        serial = next(iter(data.keys()))
    if serial not in data:
        raise KeyError(f"Serial {serial} not found in {npy_file}. Available: {list(data.keys())}")
    entry = data[serial]

    if isinstance(entry, dict) and 'trc' in entry:
        T = _augment_3x4_to_4x4(np.asarray(entry['trc'], dtype=float))
        return T, serial, "trc (robot->camera) — augmented 3x4 -> 4x4"
    if isinstance(entry, dict) and 'tcr' in entry:
        T_cam_to_robot = _augment_3x4_to_4x4(np.asarray(entry['tcr'], dtype=float))
        return np.linalg.inv(T_cam_to_robot), serial, "tcr (camera->robot) inverted to robot->camera"
    if isinstance(entry, dict) and entry.get('mode') == 'eye_in_hand':
        if 'tce' in entry:
            T_ee_to_cam = _augment_3x4_to_4x4(np.asarray(entry['tce'], dtype=float))
            return T_ee_to_cam, serial, "tce (end-effector->camera). NOTE: not base->camera"
        if 'tec' in entry:
            T_cam_to_ee = _augment_3x4_to_4x4(np.asarray(entry['tec'], dtype=float))
            return np.linalg.inv(T_cam_to_ee), serial, "tec (camera->end-effector) inverted"
    raise RuntimeError(f"Could not determine a usable transform for serial {serial} from {npy_file}")


def load_transform(serial=None, transform_dir=None):
    """Load and return (T, serial, info). Prefers transforms.npz (pickle-free,
    cross-numpy-version safe); falls back to legacy transforms.npy."""
    base_dir = Path(transform_dir) if transform_dir is not None else Path(__file__).resolve().parent
    npz_file = base_dir / "transforms.npz"
    npy_file = base_dir / "transforms.npy"

    if npz_file.exists():
        return _load_from_npz(npz_file, serial)
    if npy_file.exists():
        return _load_from_npy(npy_file, serial)
    raise FileNotFoundError(f"Transform file not found: {npz_file} or {npy_file}")


if __name__ == '__main__':
    serial_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        T, serial_used, info = load_transform(serial=serial_arg)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print(f"Serial: {serial_used}")
    print(f"Info: {info}")
    np.set_printoptions(precision=6, suppress=True)
    print("Transform (4x4):\n", T)
