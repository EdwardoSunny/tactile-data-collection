import pyrealsense2 as rs
import numpy as np
import cv2
from environment.cameras import Camera
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def xarm_state_to_10d(state):
    state_pos = state[:, :3]
    state_euler = state[:, 3:6]
    state_rot_mat = R.from_euler('xyz', state_euler, degrees=True).as_matrix()
    state_rot_mat_torch = torch.from_numpy(state_rot_mat).float()
    state_rot_6d = matrix_to_rotation_6d(state_rot_mat_torch)
    state_pose_new_6d = np.zeros((state.shape[0], 10))
    state_pose_new_6d[:, :3] = state_pos
    state_pose_new_6d[:, 3:9] = state_rot_6d.numpy()
    state_pose_new_6d[:, 9] = state[:, 6]
    return state_pose_new_6d


def from_10d_to_xarm_state(state_pose_10d):
    state_pos = state_pose_10d[:, :3]
    state_rot_6d = state_pose_10d[:, 3:9]
    state_rot_mat_torch = rotation_6d_to_matrix(torch.from_numpy(state_rot_6d).float())
    state_rot_mat = state_rot_mat_torch.numpy()
    state_euler = R.from_matrix(state_rot_mat).as_euler('xyz', degrees=True)
    state_xarm_pose = np.concatenate([state_pos, state_euler,  state_pose_10d[:, 9:]], axis=1)
    return state_xarm_pose


def list_devices():
    context = rs.context()
    devices = context.query_devices()[:]  # Modified to skip the first device
    if len(devices) == 0:
        print("No RealSense devices found!")
    else:
        print(f"{len(devices)} RealSense devices found:")
        for i, device in enumerate(devices):
            print(f"{i}: {device.get_info(rs.camera_info.name)}")
    return devices

def initialize_cameras():
    devices = list_devices()
    cameras = []

    for i, device in enumerate(devices):
        try:
            camera = Camera(device, i)
            cameras.append(camera)
            print(f"Initialized camera: {camera}")
        except Exception as e:
            print(f"Failed to initialize camera {device.get_info(rs.camera_info.serial_number)}: {e}")
    
    return cameras

def get_cameras():
    return initialize_cameras()

if __name__ == "__main__":
    cameras = initialize_cameras()

    try:
        while True:
            frames_list = []
            
            for camera in cameras:
                latest_data = camera.get_latest()
                if latest_data:
                    frames_list.append(latest_data['color_image'])

            if frames_list:
                combined_frame = np.vstack(frames_list)
                cv2.imshow("Multiple RealSense Cameras", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()

