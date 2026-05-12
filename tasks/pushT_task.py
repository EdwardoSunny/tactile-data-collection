import numpy as np
from environment.env import XArmEnvironment


class PushT_Task:
    def __init__(self, reset_position=None, fixed_z=85.0):
        self.fixed_z = fixed_z
        if reset_position is None:
            reset_position = [290, -200.0, self.fixed_z]
        else:
            reset_position = [reset_position[0], reset_position[1], self.fixed_z]
        
        self.env = XArmEnvironment(reset_position=reset_position)
    
    def reset(self, duration=3.0):
        return self.env.reset(duration=duration)
    
    def step(self, target_xy, grasp_state):
        target_pose = np.zeros(6)
        target_pose[0] = target_xy[0]
        target_pose[1] = target_xy[1]
        target_pose[2] = self.fixed_z
        self.env.step(grasp=grasp_state, target_pose=target_pose, delta_target_pose=None, timesteps=None)
    
    def get_obs(self):
        return self.env.get_obs()
