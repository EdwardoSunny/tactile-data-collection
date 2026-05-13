from typing import Optional

import numpy as np

from environment.env import XArmEnvironment
from environment.tactile import TactileSensors


class Simple_Task:
    def __init__(
        self,
        reset_position=None,
        reset_orientation=None,
        tactile: Optional[TactileSensors] = None,
    ):
        if reset_position is None:
            reset_position = [400, 0.0, 290.0]
        if reset_orientation is None:
            # End-effector pointing straight down (xArm euler xyz, degrees).
            reset_orientation = [180.0, 0.0, 0.0]

        self.env = XArmEnvironment(
            reset_position=reset_position,
            reset_orientation=reset_orientation,
            use_gripper=True,
            tactile=tactile,
        )

    def reset(self, duration=3.0):
        return self.env.reset(duration=duration)

    def step(self, target, grasp_state):
        target_pose = np.asarray(target, dtype=np.float64)
        self.env.step(grasp=grasp_state, target_pose=target_pose)

    def get_obs(self):
        return self.env.get_obs()
