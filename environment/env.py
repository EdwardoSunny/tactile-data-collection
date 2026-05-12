import cv2
import numpy as np
import time

from environment.xarm_controller import XArm, XArmConfig
from environment.utils import get_cameras

class XArmEnvironment:
    def __init__(self, reset_position=None, use_gripper=False):
        self.cameras = get_cameras()
        xarm_config = XArmConfig(use_gripper=use_gripper)
        self.xarm_config = xarm_config
        self.xarm = XArm(xarm_config)
        self.bounds = np.array([[250.0, 800.0],[-600.0, 600.0],[50.0, 400.0]])
        self.reset_position = reset_position if reset_position is not None else [461.148376, 0.0, 85.0]
        self.xarm.initialize()

    def step(self, grasp, target_pose=None, delta_target_pose=None, timesteps=None):
        if target_pose is not None:
            position = target_pose[:3]
            position = np.clip(position, self.bounds[:,0], self.bounds[:,1])
            rotation = target_pose[3:]
            self.xarm.step_abs(new_position=position, new_orientation=rotation, grasp=grasp)
        elif delta_target_pose is not None:
            current_pose = self.get_pose()
            current_pose[:3,3] += delta_target_pose[:3,3]
            self.xarm.step_abs(new_position=current_pose[:3,3], grasp=grasp)
    
    def get_pose_6d(self):
        _, pose_6d = self.xarm.arm.get_position(is_radian=False)
        return pose_6d
        
    def get_obs(self):
        obs = {"pose": self.get_pose_6d()}
        for camera in self.cameras:
            obs[f"camera_{camera.index}"] = camera.get_latest()
        return obs
    
    def go_to_position(self, target_position, duration=2.0, frequency=30, grasp=0.0):
        current_pose_6d = self.get_pose_6d()
        current_position = np.array(current_pose_6d[:3], dtype=np.float64)
        current_orientation = np.array(current_pose_6d[3:6], dtype=np.float64)
        
        target_position = np.array(target_position, dtype=np.float64)
        target_position = np.clip(target_position, self.bounds[:,0], self.bounds[:,1])
        
        num_steps = int(duration * frequency)
        dt = 1.0 / frequency
        
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            interpolated_position = current_position + smooth_alpha * (target_position - current_position)
            interpolated_position = np.clip(interpolated_position, self.bounds[:,0], self.bounds[:,1])
            
            self.xarm.step_abs(
                new_position=interpolated_position,
                new_orientation=current_orientation,
                grasp=grasp
            )
            
            if step < num_steps:
                time.sleep(dt)
        
        return self.get_obs()
    
    def reset(self, duration=3.0):
        return self.go_to_position(self.reset_position, duration=duration)
    
    def render(self):
        frames_list = []
        for camera in self.cameras:
            latest_data = camera.get_latest()
            if latest_data:
                frames_list.append(latest_data['color_image'])
        if frames_list:
            combined_frame = np.hstack(frames_list)
            cv2.imshow("XArm Environment Cameras", combined_frame)
            cv2.waitKey(1)