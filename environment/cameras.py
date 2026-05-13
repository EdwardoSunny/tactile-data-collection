import pyrealsense2 as rs
import numpy as np
import cv2

class Camera:
    def __init__(self, device, index):
        self.device = device
        self.serial_number = device.get_info(rs.camera_info.serial_number)
        self.name = device.get_info(rs.camera_info.name)
        self.index = index

        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(self.config)

        # Cache color-stream intrinsics so the tactile overlay can project
        # 3D sensor positions to pixels without re-querying every frame.
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()

        self.latest_color_frame = None
        self.latest_color_image = None
    
    def get_latest(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return None
            
            self.latest_color_frame = color_frame
            self.latest_color_image = np.asanyarray(color_frame.get_data())
            
            return {
                'color_frame': self.latest_color_frame,
                'color_image': self.latest_color_image
            }
        except Exception as e:
            print(f"Error getting latest frame from camera {self.serial_number}: {e}")
            return None
    
    def stop(self):
        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"Error stopping camera {self.serial_number}: {e}")
    
    def __str__(self):
        return f"Camera({self.name}, {self.serial_number})"
    
    def __repr__(self):
        return self.__str__()