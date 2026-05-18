import numpy as np

class SensorNormalizer:
    def __init__(self, npz_path: str, global_scale: float = 0.8, offset_override: np.ndarray = None):
        print(f"Loading calibration from {npz_path}...")
        calib = np.load(npz_path)
        self.offset = calib['offset']       # shape: (9, 3)
        self.scale = calib['scale']         # shape: (9, 3)
        self.global_scale = global_scale

        if offset_override is not None:
            self.offset = offset_override

        self.global_scale_xy = np.max(self.scale[:, :2]) * self.global_scale
        self.global_scale_z = np.max(self.scale[:, 2]) * self.global_scale
        print("Calibration loaded successfully.")
        print(f"Global XY scale: {self.global_scale_xy:.2f}, Global Z scale: {self.global_scale_z:.2f}")

    def normalize(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Normalize a 9x3 array of raw sensor data.
        raw_data: numpy array of shape (9, 3) [x, y, z] for 9 sensors
        Returns normalized numpy array of same shape.
        """
        norm_data = np.zeros_like(raw_data, dtype=float)

        if self.global_scale_xy > 0:
            norm_data[:, 0] = (raw_data[:, 0] - self.offset[:, 0]) / self.global_scale_xy
            norm_data[:, 1] = (raw_data[:, 1] - self.offset[:, 1]) / self.global_scale_xy
            norm_data[:, :2] = np.clip(norm_data[:, :2], -1.0, 1.0)

        if self.global_scale_z > 0:
            nz = np.abs(raw_data[:, 2] - self.offset[:, 2]) / self.global_scale_z
            norm_data[:, 2] = np.clip(nz, 0.0, 1.0)

        return norm_data
