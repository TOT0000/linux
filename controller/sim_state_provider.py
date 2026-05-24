import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import numpy as np

from .anchor_provider import AnchorGeometryProvider
from .gcs_safety_assessment import GcsSafetyAssessmentService
from .gcs_safety_state import GcsSafetyStateService
from .localization_error_model import LocalizationErrorModel
from .localization_estimator import IterativeLeastSquaresEstimator3D
from .state_packet import LocalizedStatePacket
from .state_provider import StateProvider
from .utils import print_debug


@dataclass
class _SimStateCache:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    nav_state: int = 0
    arming_state: int = 0


class _SharedRos2Context:
    """Process-wide ROS2 context guard with reference counting."""

    _lock = threading.Lock()
    _ref_count = 0
    _rclpy = None

    @classmethod
    def acquire(cls):
        try:
            import rclpy
        except ImportError:
            return None

        with cls._lock:
            if not rclpy.ok():
                rclpy.init(args=None)
            cls._ref_count += 1
            cls._rclpy = rclpy
            return cls._rclpy

    @classmethod
    def release(cls, rclpy_module):
        if rclpy_module is None:
            return

        with cls._lock:
            if cls._ref_count > 0:
                cls._ref_count -= 1

            if cls._ref_count == 0 and rclpy_module.ok():
                rclpy_module.shutdown()
                cls._rclpy = None


class SimStateProvider(StateProvider):
    """State provider for PX4 SITL via ROS2 topics.

    Subscribes to:
    - /fmu/out/vehicle_local_position_v1
    - /fmu/out/vehicle_status_v2
    """

    def __init__(self, fixed_user_position: Optional[Tuple[float, float, float]] = None):
        super().__init__()
        self._lock = threading.Lock()
        self._cache = _SimStateCache()
        self._active = False
        self._spin_thread: Optional[threading.Thread] = None
        self._node = None
        self._rclpy = None
        self._context = None
        self._executor = None

        # TODO(px4-sim): user position is currently fixed/env-driven, not dynamic from simulator topics.
        self._fixed_user_position = fixed_user_position or self._load_user_position_from_env()
        self._user_position: Tuple[float, float, float] = self._fixed_user_position
        self._user_position_locked: bool = False
        self._user_position_lock_reason: str = ""
        self._last_position_ts: float = 0.0
        self._last_user_position_ts: float = 0.0
        self._ros_ready: bool = False
        self._user_position_topic: str = os.getenv("SIM_USER_POSITION_TOPIC", "/sim/user_position")
        self._user_position_msg_type_name: str = os.getenv("SIM_USER_POSITION_MSG_TYPE", "PointStamped")
        # Support both versioned and unversioned PX4 ROS2 bridge topic names.
        self._local_position_topics = ("/fmu/out/vehicle_local_position_v1", "/fmu/out/vehicle_local_position")
        self._vehicle_status_topics = ("/fmu/out/vehicle_status_v2", "/fmu/out/vehicle_status")
        self._anchor_provider = AnchorGeometryProvider()
        self._localization_error_model = LocalizationErrorModel()
        self._localization_estimator = IterativeLeastSquaresEstimator3D()
        seed = int(os.getenv("SIM_LOCALIZATION_RNG_SEED", "12345"))
        self._rng = np.random.default_rng(seed)
        self._confidence_alpha = float(os.getenv("SIM_LOCALIZATION_CONFIDENCE_ALPHA", "0.95"))
        self._safety_assessment_service = GcsSafetyAssessmentService()
        self._sequence_numbers = {"drone": 0, "user": 0}
        self._latest_generated_packets: dict[str, Optional[LocalizedStatePacket]] = {"drone": None, "user": None}
        self._latest_received_packets: dict[str, Optional[LocalizedStatePacket]] = {"drone": None, "user": None}
        self._latest_packet_generation_timestamps: dict[str, Optional[float]] = {"drone": None, "user": None}
        self._latest_gcs_safety_state = None
        self._latest_safety_context = None
        self._last_safety_update_timestamp: Optional[float] = None
        self._localization_detail_log_interval_s = float(os.getenv("SIM_LOCALIZATION_DEBUG_INTERVAL_S", "1.0"))
        self._localization_summary_log_interval_s = float(os.getenv("SIM_LOCALIZATION_SUMMARY_INTERVAL_S", "2.0"))
        self._last_generation_detail_log_ts = {"drone": 0.0, "user": 0.0}
        self._last_generation_summary_log_ts = {"drone": 0.0, "user": 0.0}
        self._last_receive_detail_log_ts = {"drone": 0.0, "user": 0.0}
        self._last_receive_summary_log_ts = {"drone": 0.0, "user": 0.0}

    def _load_user_position_from_env(self) -> Tuple[float, float, float]:
        raw = os.getenv("SIM_USER_POSITION", "8,8,0")
        try:
            x, y, z = [float(v.strip()) for v in raw.split(",")]
            return (x, y, z)
        except Exception:
            return (0.0, 0.0, 0.0)

    def _fmt_vec(self, vec) -> str:
        arr = np.asarray(vec, dtype=float).reshape(-1)
        return "(" + ", ".join(f"{float(v):.2f}" for v in arr) + ")"

    def _fmt_arr(self, arr) -> str:
        values = np.asarray(arr, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"

    def _localization_error_norm(self, packet: LocalizedStatePacket) -> float:
        return float(np.linalg.norm(packet.localization_error_vector_3d))

    def _log_localization_generation(self, packet: LocalizedStatePacket):
        now = time.time()
        entity_type = packet.entity_type
        if (now - self._last_generation_detail_log_ts[entity_type]) >= self._localization_detail_log_interval_s:
            self._last_generation_detail_log_ts[entity_type] = now
            print_debug(
                f"[LOC-TX-{entity_type.upper()}]\n"
                f"  seq={packet.sequence_number}\n"
                f"  state_generation_timestamp={packet.state_generation_timestamp:.3f}\n"
                f"  gt_position_3d={self._fmt_vec(packet.gt_position_3d)}\n"
                f"  est_position_3d={self._fmt_vec(packet.estimated_position_3d)}\n"
                f"  localization_error_vector={self._fmt_vec(packet.localization_error_vector_3d)}\n"
                f"  localization_error_norm={self._localization_error_norm(packet):.3f}m\n"
                f"  true_ranges={self._fmt_arr(packet.true_ranges)}\n"
                f"  measured_ranges={self._fmt_arr(packet.measured_ranges)}\n"
                f"  bias_values={self._fmt_arr(packet.bias_values)}\n"
                f"  random_noise_values={self._fmt_arr(packet.random_noise_values)}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        if (now - self._last_generation_summary_log_ts[entity_type]) >= self._localization_summary_log_interval_s:
            self._last_generation_summary_log_ts[entity_type] = now
            print_debug(
                f"[LOC-{entity_type.upper()}] gt={self._fmt_vec(packet.gt_position_3d)} "
                f"est={self._fmt_vec(packet.estimated_position_3d)} "
                f"err={self._localization_error_norm(packet):.3f}m "
                f"seq={packet.sequence_number}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def _log_localization_receive(self, packet: LocalizedStatePacket):
        now = time.time()
        entity_type = packet.entity_type
        if (now - self._last_receive_detail_log_ts[entity_type]) >= self._localization_detail_log_interval_s:
            self._last_receive_detail_log_ts[entity_type] = now
            print_debug(
                f"[LOC-RX-{entity_type.upper()}]\n"
                f"  seq={packet.sequence_number}\n"
                f"  state_generation_timestamp={packet.state_generation_timestamp:.3f}\n"
                f"  delivery=immediate",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        if (now - self._last_receive_summary_log_ts[entity_type]) >= self._localization_summary_log_interval_s:
            self._last_receive_summary_log_ts[entity_type] = now
            print_debug(
                f"[LOC-{entity_type.upper()}] gt={self._fmt_vec(packet.gt_position_3d)} "
                f"est={self._fmt_vec(packet.estimated_position_3d)} "
                f"err={self._localization_error_norm(packet):.3f}m "
                f"seq={packet.sequence_number}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def debug_log_latest_localization_snapshot(self, reason: str = "snapshot", now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        self.flush_due_packets(now=now)
        with self._lock:
            packets = {
                "drone": None if self._latest_received_packets["drone"] is None else self._latest_received_packets["drone"].copy(),
                "user": None if self._latest_received_packets["user"] is None else self._latest_received_packets["user"].copy(),
            }
        for entity_type, packet in packets.items():
            if packet is None:
                print_debug(
                    f"[LOC-{entity_type.upper()}] reason={reason} no received packet yet",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
                continue
            print_debug(
                f"[LOC-{entity_type.upper()}] reason={reason} "
                f"gt={self._fmt_vec(packet.gt_position_3d)} "
                f"est={self._fmt_vec(packet.estimated_position_3d)} "
                f"err={self._localization_error_norm(packet):.3f}m "
                f"seq={packet.sequence_number}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def _on_vehicle_local_position(self, msg):
        position = (float(msg.x), float(msg.y), float(msg.z))
        velocity = (float(msg.vx), float(msg.vy), float(msg.vz))
        yaw = float(getattr(msg, "heading", 0.0))
        timestamp_now = time.time()

        with self._lock:
            self._cache.position = position
            self._cache.velocity = velocity
            self._cache.yaw = yaw
            self._last_position_ts = timestamp_now
            current_user_position = np.asarray(self._user_position, dtype=float)

        self._generate_and_queue_entity_state_packet("drone", np.asarray(position, dtype=float), timestamp_now)
        self._generate_and_queue_entity_state_packet("user", current_user_position, timestamp_now)
        self.flush_due_packets(now=timestamp_now)

        if self._callback:
            self._callback((timestamp_now, position[0], position[1], position[2]))

    def _on_vehicle_status(self, msg):
        with self._lock:
            self._cache.nav_state = int(getattr(msg, "nav_state", 0))
            self._cache.arming_state = int(getattr(msg, "arming_state", 0))

    def _resolve_user_position_msg_type(self, point_cls, point_stamped_cls) -> Optional[Type]:
        """Resolve a single ROS2 message type for /sim/user_position subscription."""
        choice = (self._user_position_msg_type_name or "PointStamped").strip().lower()
        if choice in {"pointstamped", "point_stamped", "stamped"}:
            return point_stamped_cls
        if choice in {"point", "geometry_msgs/point"}:
            return point_cls
        return point_stamped_cls

    def _on_user_position(self, msg):
        with self._lock:
            if self._user_position_locked:
                return
        # Support the configured single subscription type (Point or PointStamped).
        point = getattr(msg, "point", msg)
        x = float(getattr(point, "x", 0.0))
        y = float(getattr(point, "y", 0.0))
        z = float(getattr(point, "z", 0.0))
        with self._lock:
            self._user_position = (x, y, z)
            timestamp_now = time.time()
            self._last_user_position_ts = timestamp_now
        self._generate_and_queue_entity_state_packet("user", np.asarray((x, y, z), dtype=float), timestamp_now)
        self.flush_due_packets(now=timestamp_now)

    def get_user_position(self) -> Tuple[float, float, float]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._user_position

    def set_user_position(self, x: float, y: float, z: float, source: str = "manual"):
        timestamp_now = time.time()
        with self._lock:
            current_drone_position = np.asarray(self._cache.position, dtype=float)
        with self._lock:
            self._user_position = (float(x), float(y), float(z))
            self._last_user_position_ts = timestamp_now
        # Refresh both entities at same timestamp so safety context uses synchronized packets.
        self._generate_and_queue_entity_state_packet(
            "drone",
            current_drone_position,
            timestamp_now,
        )
        self._generate_and_queue_entity_state_packet(
            "user",
            np.asarray((float(x), float(y), float(z)), dtype=float),
            timestamp_now,
        )
        self.flush_due_packets(now=timestamp_now)
        if self._callback:
            self._callback((timestamp_now, float(x), float(y), float(z)))
        print_debug(
            f"[SIM-USER-POS] source={source} position=({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
        )

    def lock_user_position(self, locked: bool, reason: str = ""):
        with self._lock:
            self._user_position_locked = bool(locked)
            self._user_position_lock_reason = str(reason)
        print_debug(
            f"[SIM-USER-POS] lock={self._user_position_locked} reason={self._user_position_lock_reason}"
        )

    def get_user_yaw(self) -> float:
        """Fallback user yaw for simulation.

        PX4_SIM currently has no subscribed user orientation source,
        so we intentionally return a fixed placeholder yaw (0.0 rad).
        """
        return 0.0

    def has_valid_position(self) -> bool:
        with self._lock:
            # PX4 local position (0,0,0) 可能是合法原點，故以時間戳判定是否曾收過資料
            return (time.time() - self._last_position_ts) < 1.0 and self._last_position_ts > 0.0

    def get_drone_position(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._cache.position

    def get_ground_truth_drone_position(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._cache.position

    def get_estimated_drone_position(self) -> Tuple[float, float, float]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_received_packets["drone"]
            if packet is None:
                packet = self._latest_generated_packets["drone"]
            if packet is not None:
                return tuple(float(v) for v in packet.estimated_position_3d)
            return self._cache.position

    def get_ground_truth_user_position(self) -> Tuple[float, float, float]:
        return self.get_user_position()

    def get_estimated_user_position(self) -> Tuple[float, float, float]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_received_packets["user"]
            if packet is None:
                packet = self._latest_generated_packets["user"]
            if packet is not None:
                return tuple(float(v) for v in packet.estimated_position_3d)
            return self._user_position

    def get_anchor_positions(self):
        return self._anchor_provider.get_anchor_positions().copy()

    def get_workspace_bounds(self):
        return self._anchor_provider.get_workspace_bounds()

    def get_latest_state_packet(self) -> Optional[LocalizedStatePacket]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_generated_packets["drone"]
            return None if packet is None else packet.copy()

    def get_latest_received_state_packet(self) -> Optional[LocalizedStatePacket]:
        return self.get_latest_received_drone_packet()

    def get_latest_drone_state_packet(self) -> Optional[LocalizedStatePacket]:
        return self.get_latest_state_packet()

    def get_latest_user_state_packet(self) -> Optional[LocalizedStatePacket]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_generated_packets["user"]
            return None if packet is None else packet.copy()

    def get_latest_received_drone_packet(self) -> Optional[LocalizedStatePacket]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_received_packets["drone"]
            return None if packet is None else packet.copy()

    def get_latest_received_user_packet(self) -> Optional[LocalizedStatePacket]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            packet = self._latest_received_packets["user"]
            return None if packet is None else packet.copy()

    def get_latest_packet_generation_timestamp(self) -> Optional[float]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._latest_packet_generation_timestamps["drone"]

    def flush_due_packets(self, now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        self._refresh_cached_safety_state(now=now)
        return []

    def _refresh_cached_safety_state(self, now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        with self._lock:
            drone_packet = None if self._latest_received_packets["drone"] is None else self._latest_received_packets["drone"].copy()
            user_packet = None if self._latest_received_packets["user"] is None else self._latest_received_packets["user"].copy()
        if drone_packet is None or user_packet is None:
            with self._lock:
                self._latest_gcs_safety_state = None
                self._latest_safety_context = None
                self._last_safety_update_timestamp = now
            return None
        safety_state = GcsSafetyStateService.build(drone_packet, user_packet)
        safety_context = self._safety_assessment_service.build_from_safety_state(safety_state, now=now)
        with self._lock:
            self._latest_gcs_safety_state = safety_state
            self._latest_safety_context = safety_context
            self._last_safety_update_timestamp = now
        return safety_state

    def _generate_and_queue_entity_state_packet(self, entity_type: str, gt_position: np.ndarray, generation_timestamp: float):
        with self._lock:
            gt_user_position = np.asarray(self._user_position, dtype=float)
            latest_received_user = self._latest_received_packets["user"]

        anchors = self._anchor_provider.get_anchor_positions()
        true_ranges = np.linalg.norm(gt_position[None, :] - anchors, axis=1)
        range_result = self._localization_error_model.perturb_ranges(
            true_ranges,
            self._rng,
            entity_key=entity_type,
            timestamp=generation_timestamp,
        )
        initial_guess = self._anchor_provider.get_workspace_center()

        try:
            estimate = self._localization_estimator.estimate(
                anchors=anchors,
                measured_ranges=range_result.measured_ranges,
                sigma_values=range_result.sigma_values,
                bias_values=range_result.bias_values,
                initial_guess=initial_guess,
                true_ranges=true_ranges,
            )
            est_position = estimate.est_position_3d
            jacobian_h_3d = estimate.jacobian_h_3d
            P_3d = estimate.P_3d
            b_3d = estimate.b_3d
            M_3d = estimate.M_3d
            P_xy = estimate.P_xy
            b_xy = estimate.b_xy
            M_xy = estimate.M_xy
            range_residuals = estimate.range_residuals
            range_residual_rms_m = estimate.range_residual_rms_m
            normalized_range_residual_rms = estimate.normalized_range_residual_rms
        except Exception:
            est_position = gt_position.copy()
            jacobian_h_3d = np.zeros((anchors.shape[0], 3), dtype=float)
            P_3d = np.eye(3, dtype=float)
            b_3d = np.zeros(3, dtype=float)
            M_3d = P_3d.copy()
            P_xy = P_3d[0:2, 0:2].copy()
            b_xy = b_3d[0:2].copy()
            M_xy = P_xy.copy()
            range_residuals = np.zeros(anchors.shape[0], dtype=float)
            range_residual_rms_m = 0.0
            normalized_range_residual_rms = 0.0

        localization_error_vector = est_position - gt_position
        packet = LocalizedStatePacket(
            entity_type=entity_type,
            sequence_number=self._sequence_numbers[entity_type],
            state_generation_timestamp=float(generation_timestamp),
            gt_position_3d=gt_position.copy(),
            estimated_position_3d=np.asarray(est_position, dtype=float).copy(),
            localization_error_vector_3d=np.asarray(localization_error_vector, dtype=float).copy(),
            range_residuals=np.asarray(range_residuals, dtype=float).copy(),
            range_residual_rms_m=float(range_residual_rms_m),
            normalized_range_residual_rms=float(normalized_range_residual_rms),
            gt_user_position_3d=gt_user_position.copy(),
            est_user_position_3d=(
                np.asarray(est_position, dtype=float).copy()
                if entity_type == "user"
                else (
                    None
                    if latest_received_user is None
                    else latest_received_user.estimated_position_3d.copy()
                )
            ),
            anchor_positions_3d=anchors.copy(),
            true_ranges=range_result.true_ranges.copy(),
            measured_ranges=range_result.measured_ranges.copy(),
            bias_values=range_result.bias_values.copy(),
            sigma_values=range_result.sigma_values.copy(),
            random_noise_values=range_result.random_noise_values.copy(),
            jacobian_h_3d=np.asarray(jacobian_h_3d, dtype=float).copy(),
            P_3d=np.asarray(P_3d, dtype=float).copy(),
            b_3d=np.asarray(b_3d, dtype=float).copy(),
            M_3d=np.asarray(M_3d, dtype=float).copy(),
            P_xy=np.asarray(P_xy, dtype=float).copy(),
            b_xy=np.asarray(b_xy, dtype=float).copy(),
            M_xy=np.asarray(M_xy, dtype=float).copy(),
            confidence_alpha=self._confidence_alpha,
            est_position_timestamp=float(generation_timestamp),
        )
        with self._lock:
            self._latest_generated_packets[entity_type] = packet.copy()
            self._latest_received_packets[entity_type] = packet.copy()
            self._latest_packet_generation_timestamps[entity_type] = generation_timestamp
            self._last_safety_update_timestamp = generation_timestamp
        self._sequence_numbers[entity_type] += 1
        self._log_localization_generation(packet)
        self._log_localization_receive(packet)
        return generation_timestamp

    def get_latest_gcs_safety_state(self, now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        self.flush_due_packets(now=now)
        with self._lock:
            safety_state = self._latest_gcs_safety_state
        if safety_state is not None:
            return safety_state
        return self._refresh_cached_safety_state(now=now)

    def get_latest_safety_context(self, now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        self.flush_due_packets(now=now)
        with self._lock:
            safety_context = self._latest_safety_context
        if safety_context is not None:
            return safety_context
        self._refresh_cached_safety_state(now=now)
        with self._lock:
            return self._latest_safety_context

    def get_drone_velocity(self) -> Tuple[float, float, float]:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._cache.velocity

    def get_drone_yaw(self) -> float:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._cache.yaw

    def get_navigation_state(self) -> int:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._cache.nav_state

    def get_arming_state(self) -> int:
        self.flush_due_packets(now=time.time())
        with self._lock:
            return self._cache.arming_state

    def start(self):
        if self._active:
            return

        rclpy = None
        Node = None
        SingleThreadedExecutor = None
        VehicleLocalPosition = None
        VehicleStatus = None
        Point = None
        PointStamped = None
        qos_profile_sensor_data = None
        try:
            import rclpy as _rclpy
            from rclpy.node import Node as _Node
            from rclpy.executors import SingleThreadedExecutor as _SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data as _qos_profile_sensor_data
            from px4_msgs.msg import VehicleLocalPosition as _VehicleLocalPosition, VehicleStatus as _VehicleStatus
            try:
                from geometry_msgs.msg import Point as _Point, PointStamped as _PointStamped
            except ImportError:
                _Point = None
                _PointStamped = None
            rclpy = _rclpy
            Node = _Node
            SingleThreadedExecutor = _SingleThreadedExecutor
            qos_profile_sensor_data = _qos_profile_sensor_data
            VehicleLocalPosition = _VehicleLocalPosition
            VehicleStatus = _VehicleStatus
            Point = _Point
            PointStamped = _PointStamped
        except ImportError as exc:
            print(f"[WARN] SimStateProvider disabled (ROS2/PX4 messages unavailable): {exc}")
            self._active = True
            self._ros_ready = False
            return

        if rclpy is None or Node is None or SingleThreadedExecutor is None or VehicleLocalPosition is None or VehicleStatus is None:
            print("[WARN] SimStateProvider disabled: ROS2 symbols not initialized")
            self._active = True
            self._ros_ready = False
            return

        self._rclpy = rclpy
        # Use an isolated context so provider startup is robust even when
        # other components initialize/shutdown the default global context.
        self._context = self._rclpy.context.Context()
        self._context.init(args=None)
        self._node = Node("sim_state_provider", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._ros_ready = True

        sensor_qos = qos_profile_sensor_data if qos_profile_sensor_data is not None else 10

        for topic in self._local_position_topics:
            self._node.create_subscription(
                VehicleLocalPosition,
                topic,
                self._on_vehicle_local_position,
                sensor_qos,
            )

        for topic in self._vehicle_status_topics:
            self._node.create_subscription(
                VehicleStatus,
                topic,
                self._on_vehicle_status,
                sensor_qos,
            )

        # User position topic uses exactly one configured message type.
        user_position_msg_type = self._resolve_user_position_msg_type(Point, PointStamped)
        if user_position_msg_type is not None:
            self._node.create_subscription(
                user_position_msg_type,
                self._user_position_topic,
                self._on_user_position,
                sensor_qos,
            )
        else:
            print(
                f"[WARN] SimStateProvider user-position subscription disabled: "
                f"message type '{self._user_position_msg_type_name}' unavailable"
            )

        self._active = True

        def _spin():
            while self._active and self._context is not None and self._context.ok():
                self._executor.spin_once(timeout_sec=0.1)

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()

    def is_ros_ready(self) -> bool:
        return self._ros_ready

    def wait_for_position(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.has_valid_position():
                return True
            time.sleep(0.05)
        return self.has_valid_position()

    def stop(self):
        self._active = False

        if self._rclpy is None:
            return

        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            self._executor = None

        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if self._context is not None and self._context.ok():
            self._context.shutdown()
        self._context = None
        self._executor = None
        self._ros_ready = False
