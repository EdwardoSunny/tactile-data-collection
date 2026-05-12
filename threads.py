import time
import numpy as np
import cv2
import threading


class PhoneReadThread(threading.Thread):
    def __init__(self, phone):
        super().__init__(daemon=True)
        self.phone = phone
        self.stop_thread = False
        self._lock = threading.Lock()
        self.latest_target_pose = None
        self.latest_grasp_state = 0.0
        self.latest_button_state = False
        
    def get_data(self):
        with self._lock:
            return self.latest_target_pose, self.latest_grasp_state, self.latest_button_state
    
    def stop(self):
        self.stop_thread = True
        
    def run(self):
        while not self.stop_thread:
            try:
                target_pose = self.phone.get_target_pose()
                grasp_state = self.phone.get_grasp_state()
                button_state = self.phone.get_button_state()
                
                with self._lock:
                    self.latest_target_pose = target_pose
                    self.latest_grasp_state = grasp_state
                    self.latest_button_state = button_state
                    
            except Exception as e:
                print(f"Error reading phone data: {e}")
            
            time.sleep(0.001)


class RecordingThread(threading.Thread):
    def __init__(self, recorder, env, frequency):
        super().__init__(daemon=True)
        self.recorder = recorder
        self.env = env
        self.frequency = frequency
        self.record_interval = 1.0 / frequency
        self.recording = False
        self.episode_started = False
        self.stop_thread = False
        self._lock = threading.Lock()
        self.latest_target_pose = None
        self.latest_grasp_state = None
        self.last_record_time = 0
        
    def set_recording(self, recording):
        with self._lock:
            self.recording = recording
            self.episode_started = recording
    
    def is_recording(self):
        with self._lock:
            return self.recording
    
    def update_data(self, target_pose, grasp_state):
        with self._lock:
            self.latest_target_pose = target_pose
            self.latest_grasp_state = grasp_state
    
    def stop(self):
        self.stop_thread = True
        
    def run(self):
        self.last_record_time = time.monotonic()
        while not self.stop_thread:
            current_time = time.monotonic()
            
            if current_time - self.last_record_time >= self.record_interval:
                with self._lock:
                    should_record = self.recording and self.episode_started
                    target_pose = self.latest_target_pose
                    grasp_state = self.latest_grasp_state
                
                if should_record and target_pose is not None:
                    obs = self.env.get_obs()
                    
                    if obs is not None:
                        try:
                            state = np.concatenate([np.array(obs["pose"]), [grasp_state]])
                            
                            camera_keys = [key for key in obs.keys() if key.startswith('camera_')]
                            if camera_keys:
                                camera_imgs = []
                                for camera_key in camera_keys:
                                    img = obs[camera_key]['color_image']
                                    resized_img = cv2.resize(img, (224, 224))
                                    camera_imgs.append(resized_img)
                            else:
                                camera_imgs = [np.zeros((224, 224, 3), dtype=np.uint8)]
                            
                            self.recorder.append(
                                state=state,
                                n_contacts=np.array([0]),
                                imgs=camera_imgs
                            )
                            
                            actual_freq = 1.0 / (current_time - self.last_record_time)
                            print(f"Real frequency: {actual_freq:.2f} Hz")
                            
                        except Exception as e:
                            print(f"Error in recording thread: {e}")
                
                self.last_record_time = current_time
            
            time.sleep(0.01)
