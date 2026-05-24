import os
import cv2, time
from typing import Tuple
from .abs.robot_wrapper import RobotWrapper

class FrameReader:
    def __init__(self, cap):
        # Initialize the video capture
        self.cap = cap
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    @property
    def frame(self):
        # Read a frame from the video capture
        ret, frame = self.cap.read()
        if not ret:
            # raise ValueError("Could not read frame")
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class VirtualRobotWrapper(RobotWrapper):

    # ***
    def __init__(self, enable_video: bool = False):
        self.stream_on = False
        self.enable_video = enable_video
        # Keep simulation responsive by default while allowing callers to tune
        # pacing to mimic real-world drone latency when needed.
        self.command_delay_s = max(0.0, float(os.getenv("TYPEFLY_VIRTUAL_COMMAND_DELAY_S", "0.05")))
        self.movement_x_accumulator = 0.0 
        self.movement_y_accumulator = 0.0 
        self.movement_z_accumulator = 0.0 
        self.rotation_accumulator = 0.0 
        
        if self.enable_video:  
            self.start_stream()  
        
        print(f"[INFO] VirtualRobotWrapper: enable_video = {self.enable_video}")

    def _sleep_after_command(self):
        if self.command_delay_s > 0:
            time.sleep(self.command_delay_s)

    def keep_active(self):
        pass

    def connect(self):
        pass

    def takeoff(self) -> bool:
        return True

    def land(self):
        pass

    # ***
    def start_stream(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera for virtual robot.")
            self.stream_on = True
        print("[VR] Video stream started.")
        
    # ***
    def stop_stream(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            self.stream_on = False
            print("[VR] Video stream stopped.")


    def get_frame_reader(self):
        if not self.stream_on:
            return None
        return FrameReader(self.cap)
        
    # ***
    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving forward {distance} m")
        self.movement_y_accumulator += distance
        self._sleep_after_command()
        return True, False
    # ***
    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving backward {distance} m")
        self.movement_y_accumulator -= distance
        self._sleep_after_command()
        return True, False
    # ***
    def move_left(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving left {distance} m")
        self.movement_x_accumulator -= distance
        self._sleep_after_command()
        return True, False
    # ***
    def move_right(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving right {distance} m")
        self.movement_x_accumulator += distance
        self._sleep_after_command()
        return True, False
    # ***
    def move_up(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving up {distance} m")
        self.movement_z_accumulator += distance  # ***
        self._sleep_after_command()
        return True, False
    # ***
    def move_down(self, distance: float) -> Tuple[bool, bool]:
        print(f"-> Moving down {distance} m")
        self.movement_z_accumulator -= distance  # ***
        self._sleep_after_command()
        return True, False

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CCW {degree} degrees")
        self.rotation_accumulator += degree
        if degree >= 90:
            print("-> Turning CCW over 90 degrees")
            return True, False
        self._sleep_after_command()
        return True, False

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        print(f"-> Turning CW {degree} degrees")
        self.rotation_accumulator -= degree
        if degree >= 90:
            print("-> Turning CW over 90 degrees")
            return True, False
        self._sleep_after_command()
        return True, False
   
    # ***    
    def get_drone_position(self) -> Tuple[float, float, float]:
        # return current drone_position
        return (
            round(self.movement_x_accumulator, 2),
            round(self.movement_y_accumulator, 2),
            round(self.movement_z_accumulator, 2)
        )
