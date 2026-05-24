import math
import os
import threading
import time
from typing import Optional, Tuple

from .virtual_robot_wrapper import VirtualRobotWrapper
from .utils import print_debug, print_t


class Px4SimRobotWrapper(VirtualRobotWrapper):
    """PX4 simulator robot wrapper backed by SimStateProvider cache.

    Minimal MVP:
    - get_drone_position() comes from SimStateProvider
    - takeoff() performs offboard arm + position-setpoint climb
    - move/turn skills update active setpoint targets in local frame
    - a background loop continuously publishes active offboard setpoints for stable hold

    TODO:
    - Replace ad-hoc publishing with dedicated mission_executor adapter/service API.
    - Add robust frame conversion (NED/ENU) based on simulator config.
    """

    def __init__(self, enable_video: bool = False):
        super().__init__(enable_video=enable_video)
        self._nav_state_offboard = 14
        self._arming_state_armed = 2
        self._state_provider = None
        self._rclpy = None
        self._node = None
        self._pub_offboard_mode = None
        self._pub_traj_sp = None
        self._pub_vehicle_cmd = None

        self._offboard_counter = 0

        self._target_lock = threading.Lock()
        self._active_setpoint: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._setpoint_stream_active = False
        self._setpoint_thread: Optional[threading.Thread] = None
        self._setpoint_thread_running = False
        self._publish_log_interval_s = 1.0
        self._control_log_interval_s = 0.5
        self._last_setpoint_log_ts = 0.0
        self._last_control_log_ts = 0.0
        self._last_logged_setpoint: Optional[Tuple[float, float, float, Optional[float]]] = None
        self._active_command_name: Optional[str] = None
        self._active_command_value: Optional[float] = None
        self._active_command_start_time: Optional[float] = None
        # Offboard warmup dominates per-command latency in px4_sim mode.
        # Keep it tunable and skip warmup entirely once already OFFBOARD+ARMED.
        self._offboard_warmup_s = max(0.0, float(os.getenv("TYPEFLY_PX4_OFFBOARD_WARMUP_S", "0.25")))
        self._offboard_confirm_timeout_s = max(0.1, float(os.getenv("TYPEFLY_PX4_OFFBOARD_CONFIRM_TIMEOUT_S", "1.0")))
        self._offboard_max_attempts = max(1, int(os.getenv("TYPEFLY_PX4_OFFBOARD_MAX_ATTEMPTS", "2")))

    def set_state_provider(self, state_provider):
        self._state_provider = state_provider

    def _ensure_ros_publishers(self) -> bool:
        if self._node is not None:
            return True
        try:
            import rclpy
            from rclpy.node import Node
            from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
        except ImportError as exc:
            print(f"[WARN] Px4SimRobotWrapper ROS2/PX4 unavailable: {exc}")
            return False

        self._rclpy = rclpy
        if not self._rclpy.ok():
            self._rclpy.init(args=None)

        self._node = Node("px4_sim_robot_wrapper")
        self._msg_OffboardControlMode = OffboardControlMode
        self._msg_TrajectorySetpoint = TrajectorySetpoint
        self._msg_VehicleCommand = VehicleCommand

        self._pub_offboard_mode = self._node.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )
        self._pub_traj_sp = self._node.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )
        self._pub_vehicle_cmd = self._node.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        self._start_setpoint_loop()
        return True

    def _start_setpoint_loop(self):
        if self._setpoint_thread is not None and self._setpoint_thread.is_alive():
            return

        self._setpoint_thread_running = True

        def _loop():
            while self._setpoint_thread_running:
                self._spin_once()
                with self._target_lock:
                    target = self._active_setpoint if self._setpoint_stream_active else None
                if target is not None:
                    tx, ty, tz, tyaw = target
                    self._publish_offboard_setpoint(tx, ty, tz, yaw=tyaw)
                time.sleep(0.05)  # 20 Hz offboard stream

        self._setpoint_thread = threading.Thread(target=_loop, daemon=True)
        self._setpoint_thread.start()

    def _stop_setpoint_loop(self):
        self._setpoint_thread_running = False
        if self._setpoint_thread and self._setpoint_thread.is_alive():
            self._setpoint_thread.join(timeout=1.0)
        self._setpoint_thread = None

    def _set_active_target(self, x: float, y: float, z: float, yaw: Optional[float]):
        with self._target_lock:
            self._active_setpoint = (float(x), float(y), float(z), None if yaw is None else float(yaw))
            self._setpoint_stream_active = True

    def _clear_active_target(self):
        with self._target_lock:
            self._setpoint_stream_active = False

    def _now_us(self) -> int:
        return int(time.time() * 1_000_000)

    def _spin_once(self):
        if self._rclpy is not None and self._node is not None and self._rclpy.ok():
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0, param7: float = 0.0):
        msg = self._msg_VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = int(command)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param7 = float(param7)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._pub_vehicle_cmd.publish(msg)
        print_debug(
            f"[PX4-CMD] vehicle_command={int(command)} param1={param1:.2f} "
            f"param2={param2:.2f} param7={param7:.2f} timestamp_us={msg.timestamp}"
        )

    def _is_offboard_ready(self) -> bool:
        return (
            int(self.get_navigation_state()) == int(self._nav_state_offboard)
            and int(self.get_arming_state()) == int(self._arming_state_armed)
        )

    def _wait_for_offboard_ready(self, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._is_offboard_ready():
                return True
            time.sleep(0.05)
        return self._is_offboard_ready()

    def _ensure_offboard_control(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        warmup_s: Optional[float] = None,
        confirm_timeout_s: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> bool:
        """Warm the position-setpoint stream, then request OFFBOARD + arm.

        PX4 simulator motion commands rely on the offboard stream being active before
        mode switching. Re-issuing the commands here keeps movement/rotation skills from
        depending on takeoff() having been the only code path that armed and entered
        offboard mode.
        """
        cmd_mode = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176)
        cmd_arm = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400)
        if warmup_s is None:
            warmup_s = self._offboard_warmup_s
        if confirm_timeout_s is None:
            confirm_timeout_s = self._offboard_confirm_timeout_s
        if max_attempts is None:
            max_attempts = self._offboard_max_attempts

        # Fast path: during normal px4_sim operation we are already OFFBOARD+ARMED.
        self._set_active_target(x, y, z, yaw)
        if self._is_offboard_ready():
            return True

        for attempt in range(1, max_attempts + 1):
            time.sleep(warmup_s)
            self._publish_vehicle_command(cmd_mode, param1=1.0, param2=6.0)
            self._publish_vehicle_command(cmd_arm, param1=1.0)
            if self._wait_for_offboard_ready(timeout_s=confirm_timeout_s):
                print_debug(
                    f"[PX4-OFFBOARD] ready attempt={attempt} "
                    f"nav_state={self.get_navigation_state()} arming_state={self.get_arming_state()}"
                )
                return True
            print_debug(
                f"[PX4-OFFBOARD] retry attempt={attempt} "
                f"nav_state={self.get_navigation_state()} arming_state={self.get_arming_state()}"
            )
        return False

    def _publish_offboard_setpoint(self, x: float, y: float, z: float, yaw: Optional[float] = None):
        mode = self._msg_OffboardControlMode()
        mode.timestamp = self._now_us()
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self._pub_offboard_mode.publish(mode)

        sp = self._msg_TrajectorySetpoint()
        sp.timestamp = self._now_us()
        sp.position = [float(x), float(y), float(z)]
        if yaw is not None:
            sp.yaw = float(yaw)
        self._pub_traj_sp.publish(sp)
        publish_ts = time.time()
        current_setpoint = (float(x), float(y), float(z), None if yaw is None else float(yaw))
        if (
            self._last_logged_setpoint != current_setpoint
            or (publish_ts - self._last_setpoint_log_ts) >= self._publish_log_interval_s
        ):
            self._last_logged_setpoint = current_setpoint
            self._last_setpoint_log_ts = publish_ts
            print_debug(
                f"[PX4-SP] command={self._active_command_name or 'hold'} "
                f"target=({x:.2f}, {y:.2f}, {z:.2f}) yaw="
                f"{'None' if yaw is None else f'{yaw:.3f}'} publish_ts={publish_ts:.3f}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def _get_state(self) -> Tuple[Tuple[float, float, float], float]:
        pos = self.get_drone_position()
        yaw = self.get_drone_yaw()
        return pos, yaw

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _format_position(self, position: Tuple[float, float, float]) -> str:
        return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"

    def _begin_motion_debug(self, skill_name: str, command_value: float):
        self._active_command_name = skill_name
        self._active_command_value = float(command_value)
        self._active_command_start_time = time.time()
        print_debug(
            f"[PX4-MOVE] skill={skill_name} command_value={float(command_value):.2f}m "
            f"start_time={self._active_command_start_time:.3f}"
        )

    def _begin_rotation_debug(self, skill_name: str, command_value_deg: float):
        self._active_command_name = skill_name
        self._active_command_value = float(command_value_deg)
        self._active_command_start_time = time.time()
        print_debug(
            f"[PX4-MOVE] skill={skill_name} command_value={float(command_value_deg):.2f}deg "
            f"start_time={self._active_command_start_time:.3f}"
        )

    def _log_tracking_state(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        scalar_error: float,
        target_yaw: Optional[float] = None,
        yaw_error: Optional[float] = None,
        force: bool = False,
    ):
        now = time.time()
        if not force and (now - self._last_control_log_ts) < self._control_log_interval_s:
            return
        self._last_control_log_ts = now
        gt_position = self.get_ground_truth_drone_position()
        nav_state = self.get_navigation_state()
        arming_state = self.get_arming_state()
        message = (
            f"[PX4-STATE] command={self._active_command_name or 'hold'} "
            f"gt_position={self._format_position(gt_position)} "
            f"target={self._format_position((target_x, target_y, target_z))} "
            f"position_error={scalar_error:.3f}m nav_state={nav_state} arming_state={arming_state}"
        )
        if target_yaw is not None:
            message += f" target_yaw={target_yaw:.3f}"
        if yaw_error is not None:
            message += f" yaw_error={yaw_error:.3f}rad"
        print_debug(message)

    def _move_to_local_target(self, target_x: float, target_y: float, target_z: float, yaw: float,
                              timeout_s: float = 5.0, pos_tol: float = 0.25) -> Tuple[bool, bool]:
        self._set_active_target(target_x, target_y, target_z, yaw)
        deadline = time.time() + timeout_s
        stable_since = None
        settle_s = 0.3
        best_error = float("inf")
        last_progress_ts = time.time()
        print_debug(
            f"[PX4-MOVE] target_setpoint={self._format_position((target_x, target_y, target_z))} "
            f"yaw={yaw:.3f} completion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s timeout={timeout_s:.2f}s"
        )
        while time.time() < deadline:
            cx, cy, cz = self.get_drone_position()
            err = math.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2 + (target_z - cz) ** 2)
            self._log_tracking_state(target_x, target_y, target_z, err)
            if err + 0.05 < best_error:
                best_error = err
                last_progress_ts = time.time()
            elif err > pos_tol and (time.time() - last_progress_ts) < 1.0:
                deadline = max(deadline, time.time() + 0.75)
            if err < pos_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= settle_s:
                    self._log_tracking_state(target_x, target_y, target_z, err, force=True)
                    print_debug(
                        f"[PX4-MOVE] completed command={self._active_command_name or 'move'} "
                        f"criterion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s"
                    )
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        final_pos = self.get_ground_truth_drone_position()
        self._log_tracking_state(target_x, target_y, target_z, err, force=True)
        print_debug(
            f"[PX4-MOVE] timeout command={self._active_command_name or 'move'} "
            f"final_gt_position={self._format_position(final_pos)} target={self._format_position((target_x, target_y, target_z))} "
            f"criterion=position_error<{pos_tol:.2f}m for {settle_s:.2f}s"
        )
        return False, False


    def _ensure_airborne_for_translation(self) -> bool:
        x, y, z = self.get_ground_truth_drone_position()
        if z <= -0.35:
            return True
        print_debug(
            f"[PX4-MOVE] auto_takeoff_before_translation current_gt={self._format_position((x, y, z))}"
        )
        return bool(self.takeoff())

    def _move_by_body_offset(self, skill_name: str, command_distance: float, forward_m: float = 0.0, right_m: float = 0.0, up_m: float = 0.0,
                             timeout_scale: float = 6.0) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                return False, False
        self._begin_motion_debug(skill_name, command_distance)

        if (abs(forward_m) > 1e-6 or abs(right_m) > 1e-6) and not self._ensure_airborne_for_translation():
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=auto_takeoff_failed")
            return False, False

        (x, y, z), yaw = self._get_state()
        if not self._ensure_offboard_control(x, y, z, yaw):
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=offboard_not_ready")
            return False, False

        # Body-to-world mapping aligned to UI XY convention:
        # +forward moves toward heading; +right moves to drone starboard side.
        dx = forward_m * math.cos(yaw) + right_m * math.sin(yaw)
        dy = forward_m * math.sin(yaw) - right_m * math.cos(yaw)
        dz = -up_m  # up in NED means z decreases

        target_x = x + dx
        target_y = y + dy
        target_z = z + dz
        timeout_s = max(3.0, (abs(forward_m) + abs(right_m) + abs(up_m)) * timeout_scale)
        return self._move_to_local_target(target_x, target_y, target_z, yaw=yaw, timeout_s=timeout_s)

    def _rotate_by(self, skill_name: str, delta_yaw_rad: float, command_degrees: float, timeout_s: float = 4.0, yaw_tol: float = 0.12) -> Tuple[bool, bool]:
        if not self._ensure_ros_publishers():
            return False, False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=2.0):
                return False, False
        self._begin_rotation_debug(skill_name, command_degrees)

        (x, y, z), yaw = self._get_state()
        if not self._ensure_offboard_control(x, y, z, yaw):
            print_debug(f"[PX4-MOVE] abort command={skill_name} reason=offboard_not_ready")
            return False, False
        target_yaw = self._normalize_angle(yaw + delta_yaw_rad)
        self._set_active_target(x, y, z, target_yaw)
        print_debug(
            f"[PX4-MOVE] target_setpoint={self._format_position((x, y, z))} "
            f"yaw={target_yaw:.3f} completion=yaw_error<{yaw_tol:.2f}rad for 0.30s timeout={timeout_s:.2f}s"
        )

        stable_since = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            current_yaw = self.get_drone_yaw()
            err = abs(self._normalize_angle(target_yaw - current_yaw))
            self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err)
            if err <= yaw_tol:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= 0.3:
                    self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err, force=True)
                    print_debug(
                        f"[PX4-MOVE] completed command={self._active_command_name or 'rotate'} "
                        f"criterion=yaw_error<{yaw_tol:.2f}rad for 0.30s"
                    )
                    return True, False
            else:
                stable_since = None
            time.sleep(0.05)
        self._log_tracking_state(x, y, z, 0.0, target_yaw=target_yaw, yaw_error=err, force=True)
        print_debug(
            f"[PX4-MOVE] timeout command={self._active_command_name or 'rotate'} "
            f"target_yaw={target_yaw:.3f} criterion=yaw_error<{yaw_tol:.2f}rad for 0.30s"
        )
        return False, False

    def get_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_position"):
            return self._state_provider.get_drone_position()
        return super().get_drone_position()

    def get_ground_truth_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_ground_truth_drone_position"):
            return self._state_provider.get_ground_truth_drone_position()
        return self.get_drone_position()

    def get_estimated_drone_position(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_estimated_drone_position"):
            return self._state_provider.get_estimated_drone_position()
        return self.get_drone_position()

    def get_latest_localization_packet(self):
        if self._state_provider is not None and hasattr(self._state_provider, "get_latest_received_state_packet"):
            return self._state_provider.get_latest_received_state_packet()
        return None

    def get_latest_user_localization_packet(self):
        if self._state_provider is not None and hasattr(self._state_provider, "get_latest_received_user_packet"):
            return self._state_provider.get_latest_received_user_packet()
        return None

    def get_drone_velocity(self) -> Tuple[float, float, float]:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_velocity"):
            return self._state_provider.get_drone_velocity()
        return (0.0, 0.0, 0.0)

    def get_drone_yaw(self) -> float:
        if self._state_provider is not None and hasattr(self._state_provider, "get_drone_yaw"):
            return self._state_provider.get_drone_yaw()
        return 0.0

    def get_navigation_state(self) -> int:
        if self._state_provider is not None and hasattr(self._state_provider, "get_navigation_state"):
            return self._state_provider.get_navigation_state()
        return 0

    def get_arming_state(self) -> int:
        if self._state_provider is not None and hasattr(self._state_provider, "get_arming_state"):
            return self._state_provider.get_arming_state()
        return 0

    def connect(self):
        self._ensure_ros_publishers()

    def takeoff(self) -> bool:
        """Pure offboard takeoff using local position setpoints.

        Local frame assumption (PX4 local NED):
        - +X: forward, +Y: right, +Z: down
        - Ascend (go up) means Z becomes more negative.
        """
        if not self._ensure_ros_publishers():
            return False
        self._begin_motion_debug("takeoff", 1.0)

        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=3.0):
                print("[PX4_SIM] takeoff aborted: no valid local position")
                return False

        (x, y, z), yaw = self._get_state()

        # 1) Warm-up offboard stream, switch to offboard mode, and arm.
        if not self._ensure_offboard_control(x, y, z, yaw):
            print("[PX4_SIM] takeoff aborted: failed to enter offboard+armed state")
            return False

        # 2) Climb by setting higher target in local NED (z more negative = up).
        takeoff_height_m = 1.0
        z_tolerance_m = 0.15
        settle_time_s = 1.0
        target_z = z - takeoff_height_m
        self._set_active_target(x, y, target_z, yaw)
        print(f"[PX4_SIM] takeoff start_z={z:.2f}, target_z={target_z:.2f} (NED)")

        stable_since = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            _, _, cz = self.get_drone_position()

            if abs(cz - target_z) <= z_tolerance_m:
                if stable_since is None:
                    stable_since = time.time()
                elif (time.time() - stable_since) >= settle_time_s:
                    print(
                        f"[PX4_SIM] takeoff stabilized at target_z={target_z:.2f}, current_z={cz:.2f}; "
                        f"keep holding active setpoint"
                    )
                    return True
            else:
                stable_since = None

            time.sleep(0.05)

        _, _, final_z = self.get_drone_position()
        print(
            f"[PX4_SIM] takeoff timeout: target_z={target_z:.2f}, current_z={final_z:.2f}, "
            f"tolerance={z_tolerance_m:.2f}"
        )
        return False

    def move_forward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_forward", float(distance), forward_m=float(distance))

    def move_backward(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_backward", float(distance), forward_m=-float(distance))

    def move_left(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_left", float(distance), right_m=-float(distance))

    def move_right(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_right", float(distance), right_m=float(distance))

    def move_up(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_up", float(distance), up_m=float(distance))

    def move_down(self, distance: float) -> Tuple[bool, bool]:
        return self._move_by_body_offset("move_down", float(distance), up_m=-float(distance))

    def turn_ccw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by("turn_ccw", math.radians(float(degree)), float(degree))

    def turn_cw(self, degree: int) -> Tuple[bool, bool]:
        return self._rotate_by("turn_cw", -math.radians(float(degree)), float(degree))

    def land(self):
        if not self._ensure_ros_publishers():
            return
        # Stop offboard setpoint stream before LAND command to avoid fighting auto-land.
        self._clear_active_target()
        cmd_land = getattr(self._msg_VehicleCommand, "VEHICLE_CMD_NAV_LAND", 21)
        self._publish_vehicle_command(cmd_land)

    def reposition_for_scenario(self, scenario) -> bool:
        if not self._ensure_ros_publishers():
            return False
        if self._state_provider is not None and hasattr(self._state_provider, "wait_for_position"):
            if not self._state_provider.wait_for_position(timeout_s=3.0):
                return False

        # Support both ScenarioConfig fields (`drone_position_3d`, `drone_yaw_rad`)
        # and BaselineScene fields (`drone_initial_pose`, `drone_initial_yaw_rad`).
        target_pose = getattr(scenario, "drone_position_3d", None)
        if target_pose is None:
            target_pose = getattr(scenario, "drone_initial_pose", None)
        if target_pose is None:
            raise AttributeError(
                "scenario must define drone_position_3d or drone_initial_pose"
            )

        target_yaw = getattr(scenario, "drone_yaw_rad", None)
        if target_yaw is None:
            target_yaw = getattr(scenario, "drone_initial_yaw_rad", 0.0)

        tx, ty, tz = [float(v) for v in target_pose]
        target_yaw = float(target_yaw)
        (x, y, z), yaw = self._get_state()

        # If scenario wants airborne state but current vehicle is on/near ground, lift first.
        if tz < -0.2 and z > -0.3:
            if not self.takeoff():
                return False
            (x, y, z), yaw = self._get_state()

        if not self._ensure_offboard_control(x, y, z, yaw):
            return False

        self._begin_motion_debug("scenario_reposition", math.dist((x, y, z), (tx, ty, tz)))
        ok, _ = self._move_to_local_target(tx, ty, tz, yaw=target_yaw, timeout_s=15.0, pos_tol=0.35)
        if not ok:
            # Retry once with looser tolerance to handle simulator transient convergence issues.
            ok, _ = self._move_to_local_target(tx, ty, tz, yaw=target_yaw, timeout_s=8.0, pos_tol=0.50)
        if not ok:
            print_debug(
                f"[PX4-SCENARIO] reposition timeout target={self._format_position((tx, ty, tz))}"
            )
            return False
        self._set_active_target(tx, ty, tz, target_yaw)
        print_t(
            f"[PX4-SCENARIO] repositioned to target={self._format_position((tx, ty, tz))} yaw={target_yaw:.2f}"
        )
        return True

    def stop_stream(self):
        super().stop_stream()

    def keep_active(self):
        # Streaming is handled by the internal setpoint thread.
        pass
