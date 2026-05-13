import time
from typing import Optional

import cv2
import numpy as np

from environment.tactile import TactileSensors
from environment.utils import get_cameras
from environment.xarm_controller import XArm, XArmConfig


class XArmEnvironment:
    def __init__(
        self,
        reset_position=None,
        reset_orientation=None,
        use_gripper: bool = False,
        tactile: Optional[TactileSensors] = None,
    ):
        self.cameras = get_cameras()
        xarm_config = XArmConfig(use_gripper=use_gripper)
        self.xarm_config = xarm_config
        self.xarm = XArm(xarm_config, tactile=tactile)
        # z lower bound shifted down 20 mm from the original [50, 400].
        self.bounds = np.array([[250.0, 800.0], [-600.0, 600.0], [10.0, 380.0]])  # z_min lowered another 20 mm (was 30)
        self.reset_position = (
            reset_position if reset_position is not None else [461.148376, 0.0, 85.0]
        )
        # None = hold whatever rotation the arm currently has.
        self.reset_orientation = reset_orientation
        self.xarm.initialize()

    def step(self, grasp, target_pose=None, delta_target_pose=None, timesteps=None):
        if target_pose is not None:
            position = np.clip(target_pose[:3], self.bounds[:, 0], self.bounds[:, 1])
            rotation = target_pose[3:]
            self.xarm.step_abs(new_position=position, new_orientation=rotation, grasp=grasp)
        elif delta_target_pose is not None:
            current_pose = self.get_pose_6d()
            new_position = np.array(current_pose[:3]) + delta_target_pose[:3, 3]
            self.xarm.step_abs(new_position=new_position, grasp=grasp)

    def get_pose_6d(self):
        _, pose_6d = self.xarm.arm.get_position(is_radian=False)
        return pose_6d

    def get_obs(self):
        obs = {"pose": self.get_pose_6d()}
        for camera in self.cameras:
            obs[f"camera_{camera.index}"] = camera.get_latest()
        return obs

    def go_to_position(self, target_position, duration=2.0, frequency=30, grasp=0.0, target_orientation=None):
        current_pose_6d = self.get_pose_6d()
        current_position = np.array(current_pose_6d[:3], dtype=np.float64)
        current_orientation = np.array(current_pose_6d[3:6], dtype=np.float64)

        target_position = np.array(target_position, dtype=np.float64)
        target_position = np.clip(target_position, self.bounds[:, 0], self.bounds[:, 1])

        # Wrap each axis's delta into [-180, 180] so we always take the shortest
        # angular path, avoiding a 350-degree spin near the +/-180 wraparound.
        if target_orientation is not None:
            target_orientation = np.array(target_orientation, dtype=np.float64)
            delta_orientation = ((target_orientation - current_orientation + 180.0) % 360.0) - 180.0
        else:
            delta_orientation = np.zeros(3)

        num_steps = int(duration * frequency)
        dt = 1.0 / frequency

        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interpolated_position = current_position + smooth_alpha * (target_position - current_position)
            interpolated_position = np.clip(interpolated_position, self.bounds[:, 0], self.bounds[:, 1])
            interpolated_orientation = current_orientation + smooth_alpha * delta_orientation

            self.xarm.step_abs(
                new_position=interpolated_position,
                new_orientation=interpolated_orientation,
                grasp=grasp,
            )

            if step < num_steps:
                time.sleep(dt)

        return self.get_obs()

    def reset(self, duration=3.0):
        return self.go_to_position(
            self.reset_position, duration=duration, target_orientation=self.reset_orientation
        )

    def render(self):
        frames_list = []
        for camera in self.cameras:
            latest_data = camera.get_latest()
            if latest_data:
                frames_list.append(latest_data["color_image"])
        if frames_list:
            combined_frame = np.hstack(frames_list)
            cv2.imshow("XArm Environment Cameras", combined_frame)
            cv2.waitKey(1)
