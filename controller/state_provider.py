import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from .abs.robot_wrapper import RobotWrapper
from .uwb_wrapper import UWBWrapper


class StateProvider(ABC):
    def __init__(self):
        self._callback: Optional[Callable[[Tuple[float, float, float, float]], None]] = None

    def register_callback(self, callback: Callable[[Tuple[float, float, float, float]], None]):
        self._callback = callback

    @abstractmethod
    def get_user_position(self) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def get_drone_position(self) -> Tuple[float, float, float]:
        pass

    def get_drone_velocity(self) -> Tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def get_drone_yaw(self) -> float:
        return 0.0

    def get_ground_truth_drone_position(self) -> Tuple[float, float, float]:
        return self.get_drone_position()

    def get_estimated_drone_position(self) -> Tuple[float, float, float]:
        return self.get_drone_position()

    def get_ground_truth_user_position(self) -> Tuple[float, float, float]:
        return self.get_user_position()

    def get_estimated_user_position(self) -> Tuple[float, float, float]:
        return self.get_user_position()

    def get_anchor_positions(self):
        return []

    def get_latest_state_packet(self):
        return None

    def get_latest_received_state_packet(self):
        return None

    def get_latest_drone_state_packet(self):
        return self.get_latest_state_packet()

    def get_latest_user_state_packet(self):
        return None

    def get_latest_received_drone_packet(self):
        return self.get_latest_received_state_packet()

    def get_latest_received_user_packet(self):
        return None

    def get_latest_packet_generation_timestamp(self) -> Optional[float]:
        return None

    def flush_due_packets(self, now: Optional[float] = None):
        return []

    def get_latest_gcs_safety_state(self, now: Optional[float] = None):
        return None

    def get_latest_safety_context(self, now: Optional[float] = None):
        return None

    @abstractmethod
    def has_valid_position(self) -> bool:
        pass

    @abstractmethod
    def get_user_yaw(self) -> float:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class UwbStateProvider(StateProvider):
    def __init__(self, uwb: Optional[UWBWrapper] = None, robot: Optional[RobotWrapper] = None):
        super().__init__()
        self.uwb = uwb or UWBWrapper()
        self.robot = robot

    def register_callback(self, callback: Callable[[Tuple[float, float, float, float]], None]):
        super().register_callback(callback)
        self.uwb.register_callback(callback)

    def get_user_position(self) -> Tuple[float, float, float]:
        return self.uwb.get_user_position()

    def get_drone_position(self) -> Tuple[float, float, float]:
        if self.robot is None:
            return 0.0, 0.0, 0.0
        return self.robot.get_drone_position()

    def has_valid_position(self) -> bool:
        return self.uwb.latest_position != (0.00, 0.00, 0.00)

    def get_user_yaw(self) -> float:
        return 0.0

    def start(self):
        self.uwb.start_with_retry()

    def stop(self):
        self.uwb.stop()


class NullUserProvider(StateProvider):
    def __init__(self, robot: RobotWrapper, user_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        super().__init__()
        self.robot = robot
        self.user_position = user_position
        self._active = False

    def get_user_position(self) -> Tuple[float, float, float]:
        return self.user_position

    def get_drone_position(self) -> Tuple[float, float, float]:
        return self.robot.get_drone_position()

    def has_valid_position(self) -> bool:
        return True

    def get_user_yaw(self) -> float:
        get_yaw = getattr(self.robot, "get_drone_yaw", None)
        if callable(get_yaw):
            return float(get_yaw())
        return 0.0

    def start(self):
        self._active = True

        def loop():
            while self._active:
                if self._callback:
                    x, y, z = self.get_user_position()
                    self._callback((time.time(), x, y, z))
                time.sleep(0.1)

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._active = False


class SimStateProvider(NullUserProvider):
    """Backward-compatible alias for previous simulation provider name."""

    pass
