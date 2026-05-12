import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import teledex

class Phone:
    def __init__(self):
        self.session = teledex.Session()
        self.session.start()
        print("Waiting for phone connection...")
        while self.session.get_latest_data()["position"] is None:
            time.sleep(0.1)
        print("Phone connection established.")
        self.initial_robot_pose = None
        self.initial_ar_pose = None
        self.hand = 0
        if self.hand:
            self.position_name = "position_hand"
            self.rotation_name = "rotation_hand"
        else:
            self.position_name = "position"
            self.rotation_name = "rotation"

    def get_position_rotation(self):
        data = self.session.get_latest_data()
        position = data[self.position_name]
        rotation = data[self.rotation_name].reshape(3, 3)

        return position, rotation
    
    def reset(self, initial_pose):
        self.initial_robot_pose = initial_pose.copy()
        self.initial_ar_pose = np.eye(4)
        position, rotation = self.get_position_rotation()
        self.initial_ar_pose[:3, 3] = position.reshape(3).copy()
        self.initial_ar_pose[:3, :3] = rotation.reshape(3, 3).copy()
            
    def get_target_pose(self):
        if self.initial_robot_pose is None or self.initial_ar_pose is None:
            raise RuntimeError("Must call reset() with initial pose before getting target pose")
        
        phone_position, phone_rotation = self.get_position_rotation()
        delta_phone_position = (
            self.initial_ar_pose[:3, :3].T
            @ (phone_position - self.initial_ar_pose[:3, 3])
        )
        delta_phone_position = np.clip(delta_phone_position * 1200.0, -500, 500)

        # delta_phone_rotation = phone_rotation @ self.initial_ar_pose[:3, :3].T
        delta_phone_rotation = (
            self.initial_ar_pose[:3, :3].T
            @ phone_rotation
        )
        delta_phone_euler = R.from_matrix(delta_phone_rotation).as_euler('xyz', degrees=True)
        
        new_position = self.initial_robot_pose[:3].copy()
        new_position[0] += delta_phone_position[0] 
        new_position[1] += delta_phone_position[1] 
        new_position[2] += delta_phone_position[2] 
        
        new_rotation = self.initial_robot_pose[3:].copy()
        new_rotation[0] += delta_phone_euler[0]
        new_rotation[1] += delta_phone_euler[1]
        new_rotation[2] += delta_phone_euler[2]
        
        target_pose = np.concatenate([new_position, new_rotation])
        return target_pose
    
    def get_grasp_state(self):
        if self.hand:
            data = self.session.get_latest_data()
            if "landmarks" not in data:
                return 0.0
            landmarks = data["landmarks"]
            if landmarks is None or len(landmarks) < 21:
                return None
            
            fingers = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [0, 9, 10, 11, 12],
                [0, 13, 14, 15, 16],
                [0, 17, 18, 19, 20]
            ]
            
            closed = 0
            palm = landmarks[0]
            
            for indices in fingers:
                tip_to_base_vec = landmarks[indices[-1]] - palm
                tip_to_base = np.sqrt(np.sum(tip_to_base_vec * tip_to_base_vec))
                
                segments = landmarks[indices[1:]] - landmarks[indices[:-1]]
                segment_lengths = np.sqrt(np.sum(segments * segments, axis=1))
                extended_length = np.sum(segment_lengths)
                
                if extended_length > 0:
                    curl = 1 - (tip_to_base / extended_length)
                    if curl > 0.3:
                        closed += 1
            
            return closed > 2
        else:
            return self.session.get_latest_data()["toggle"] * 1.0
    
    def get_button_state(self):
        return self.session.get_latest_data()["button"]