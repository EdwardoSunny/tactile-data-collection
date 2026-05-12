import numpy as np
from environment.env import XArmEnvironment


class Simple_Task:
    def __init__(self, reset_position=None):
        if reset_position is None:
            reset_position = [400, 0.0, 290.0]
        
        self.env = XArmEnvironment(reset_position=reset_position, use_gripper=True)
    
    def reset(self, duration=3.0):
        return self.env.reset(duration=duration)
    
    def step(self, target, grasp_state):
        target_pose = np.zeros(6)
        target_pose[0] = target[0]
        target_pose[1] = target[1]
        target_pose[2] = target[2]
        target_pose[3] = target[3]
        target_pose[4] = target[4]
        target_pose[5] = target[5]
        self.env.step(grasp=grasp_state, target_pose=target_pose, delta_target_pose=None, timesteps=None)
    
    def get_obs(self):
        return self.env.get_obs()
