from PIL import Image
from dataclasses import replace
import math
import queue, time, os, sys, subprocess
import re
from collections import deque
from typing import Optional, Tuple
import asyncio
import uuid
import threading
import numpy as np

from .shared_frame import SharedFrame, Frame
from .gcs_safety_assessment import GcsSafetyAssessmentService
from .tello_wrapper import TelloWrapper
from .virtual_robot_wrapper import VirtualRobotWrapper
from .px4_sim_robot_wrapper import Px4SimRobotWrapper
from .abs.robot_wrapper import RobotWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .llm_planner import LLMPlanner
from .skillset import SkillSet, LowLevelSkillItem, SkillArg
from .utils import print_debug, print_t
from .minispec_interpreter import MiniSpecInterpreter, Statement
from .abs.robot_wrapper import RobotType
from .uwb_wrapper import UWBWrapper
from .state_provider import StateProvider, UwbStateProvider
from .sim_state_provider import SimStateProvider
from .scenario_manager import ScenarioManager
from .safety_context import SafetyContext
from .task_run_logger import TaskRunLogger
from .pipeline_registry import get_pipeline_config, normalize_pipeline_id
from .baseline_scenes import (
    BASELINE_SCENES,
    BaselineScene,
    ObstacleEnvelopeState,
    build_all_scene_expectations,
    build_scene_expectations,
    compute_obstacle_envelope_states,
    evaluate_path_clear,
    get_task_point,
    normalize_baseline_scene_id,
    worker_mode_summary_log,
)
from .safety_envelope import build_safety_envelope
from .benchmark_layout import (
    CHECKPOINT_DWELL_SECONDS,
    FULL_REPLAN_LIMIT,
    AGENT_HEARTBEAT_INTERVAL_SECONDS,
    TYPEFLY_REPLAN_THRESHOLD,
    UAV_GC_NOMINAL_SPEED_MPS,
    UAV_BODY_NOMINAL_SPEED_MPS,
    BENCHMARK_CHECKPOINT_ORDER,
    BENCHMARK_CHECKPOINTS_BY_ID,
    BENCHMARK_CHECKPOINTS,
    BENCHMARK_ZONES,
    UAV_RADIUS_M,
    WORKER_RADIUS_M,
)
from .langgraph_agent import LangGraphOrchestrationRunner

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTED_COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD = TYPEFLY_REPLAN_THRESHOLD
PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD = TYPEFLY_REPLAN_THRESHOLD
PREDICTED_COLLISION_PROBABILITY_REARM_THRESHOLD = TYPEFLY_REPLAN_THRESHOLD
AUTO_REPLAN_PROTECTION_STATEMENTS = 2
COLLISION_RISK_WORKER_IDS = ("worker_1", "worker_2", "worker_3")
NEAR_MISS_ENTER_DISTANCE_M = 1.0
NEAR_MISS_EXIT_DISTANCE_M = 1.05

MODE_TYPEFLY_ONESHOT = "typefly-oneshot"
MODE_TYPEFLY_THRESHOLD_REPLAN = "typefly-threshold-replan"
MODE_AGENT_HEARTBEAT_SOFT = "agent-heartbeat-soft"
MODE_AGENT_HEARTBEAT_HARDGATE = "agent-heartbeat-hardgate"
SUPPORTED_FRAMEWORK_MODES = {
    MODE_TYPEFLY_ONESHOT,
    MODE_TYPEFLY_THRESHOLD_REPLAN,
    MODE_AGENT_HEARTBEAT_SOFT,
    MODE_AGENT_HEARTBEAT_HARDGATE,
}

class LLMController():
    def __init__(self, robot_type, virtual_queue, use_http=False, message_queue: Optional[queue.Queue]=None, enable_video=False, state_provider: Optional[StateProvider]=None):
        self.virtual_queue = virtual_queue
        self.robot_type = robot_type
        self.enable_video = enable_video
        self.shared_frame = SharedFrame()
        self.yolo_client = None
        if self.enable_video:
            if use_http:
                from .yolo_client import YoloClient
                self.yolo_client = YoloClient(shared_frame=self.shared_frame)
            else:
                from .yolo_grpc_client import YoloGRPCClient
                self.yolo_client = YoloGRPCClient(shared_frame=self.shared_frame)
        self.vision = VisionSkillWrapper(self.shared_frame, enabled=enable_video)
        self.latest_frame = None
        self.controller_active = True
        self.controller_wait_takeoff = True
        self.message_queue = message_queue
        if message_queue is None:
            self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        else:
            try:
                self.cache_folder = message_queue.get(timeout=1.0)
            except queue.Empty:
                self.cache_folder = "cache/default"

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        
        match robot_type:
            case RobotType.TELLO:
                print_t("[C] Start Tello drone...")
                self.drone: RobotWrapper = TelloWrapper()
            case RobotType.GEAR:
                print_t("[C] Start Gear robot car...")
                from .gear_wrapper import GearWrapper
                self.drone: RobotWrapper = GearWrapper()
            case RobotType.PX4_SIM:
                print_t("[C] Start PX4 sim drone...")
                self.drone: RobotWrapper = Px4SimRobotWrapper(enable_video=self.enable_video)
            case _:
                print_t("[C] Start virtual drone...")
                self.drone: RobotWrapper = VirtualRobotWrapper(enable_video=self.enable_video)
        
        self.planner = LLMPlanner(robot_type)
        self.planner.controller = self
        self.safety_assessor = GcsSafetyAssessmentService()

        # state provider
        self.uwb = UWBWrapper()
        if robot_type == RobotType.PX4_SIM:
            self.state_provider = SimStateProvider()
        elif state_provider is not None:
            self.state_provider = state_provider
        else:
            self.state_provider = UwbStateProvider(self.uwb, self.drone)


        # inject provider into PX4 sim wrapper
        if robot_type == RobotType.PX4_SIM and hasattr(self.drone, "set_state_provider"):
            self.drone.set_state_provider(self.state_provider)

        self.position_update_callback = None
        self.state_provider.register_callback(self.notify_user_position_updated)

        # load low-level skills
        self.low_level_skillset = SkillSet(level="low")
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_forward", self.drone.move_forward, "Move forward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_backward", self.drone.move_backward, "Move backward by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_left", self.drone.move_left, "Move left by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_right", self.drone.move_right, "Move right by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_up", self.drone.move_up, "Move up by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("move_down", self.drone.move_down, "Move down by a distance", args=[SkillArg("distance", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("go_checkpoint", self.skill_go_checkpoint, "Navigate toward a benchmark checkpoint by ID", args=[SkillArg("checkpoint_id", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("takeoff", self.skill_takeoff, "Take off and climb to a safe hover height"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("land", self.skill_land, "Land safely at current location"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_cw", self.drone.turn_cw, "Rotate clockwise/right by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("turn_ccw", self.drone.turn_ccw, "Rotate counterclockwise/left by certain degrees", args=[SkillArg("degrees", int)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("delay", self.skill_delay, "Wait for specified seconds", args=[SkillArg("seconds", float)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("is_visible", self.vision.is_visible, "Check the visibility of target object", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_x", self.vision.object_x, "Get object's X-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_y", self.vision.object_y, "Get object's Y-coordinate in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_width", self.vision.object_width, "Get object's width in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_height", self.vision.object_height, "Get object's height in (0,1)", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("object_dis", self.vision.object_distance, "Get object's distance in cm", args=[SkillArg("object_name", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("probe", self.planner.probe, "Probe the LLM for reasoning", args=[SkillArg("question", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("log", self.skill_log, "Output text to console", args=[SkillArg("text", str)]))
        self.low_level_skillset.add_skill(LowLevelSkillItem("take_picture", self.skill_take_picture, "Take a picture"))
        self.low_level_skillset.add_skill(LowLevelSkillItem("re_plan", self.skill_re_plan, "Replanning"))

        self.low_level_skillset.add_skill(LowLevelSkillItem("time", self.skill_time, "Get current execution time", args=[]))
      

        Statement.low_level_skillset = self.low_level_skillset
        Statement.high_level_skillset = SkillSet(level="high", lower_level_skillset=self.low_level_skillset)
        self.planner.init(low_level_skillset=self.low_level_skillset, vision_skill=self.vision)

        self.current_plan = None
        self.execution_history = None
        self.execution_time = time.time()
        self.latest_safety_context = None
        self.scenario_manager = ScenarioManager(default_name=os.getenv("TYPEFLY_SCENARIO", "SAFE"))
        self.task_run_logger = TaskRunLogger(excel_path=os.getenv("TYPEFLY_TASK_LOG_XLSX"))
        self._task_id_counter = 0
        self.latest_scenario_report = None
        self.initial_scenario_state = None
        self.baseline_scene_id = normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_BENCHMARK_DEMO"))
        self.baseline_scene_state = None
        self.latest_baseline_decision = None
        self.execution_mode = "Waiting"
        self.framework_mode = MODE_TYPEFLY_ONESHOT
        self.selected_pipeline_id = normalize_pipeline_id(os.getenv("TYPEFLY_BASELINE_ID", "baseline1"))
        self.archive_enabled = True
        self.near_miss_count = 0
        self.collision_count = 0
        self.min_uav_worker_distance_m: Optional[float] = None
        self._near_miss_active_by_worker = {wid: False for wid in COLLISION_RISK_WORKER_IDS}
        self._collision_active_by_worker = {wid: False for wid in COLLISION_RISK_WORKER_IDS}
        self._latest_near_miss_events = []
        self.predicted_collision_replan_threshold = float(PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD)
        self.predicted_collision_rearm_threshold = float(PREDICTED_COLLISION_PROBABILITY_REARM_THRESHOLD)
        self.predicted_collision_replan_strictly_greater = False
        self.heartbeat_interval_seconds = float(AGENT_HEARTBEAT_INTERVAL_SECONDS)
        self.active_objective_set = self._default_active_objective_set()
        self.latest_benchmark_progress = {
            "completed": [],
            "current_target": None,
        }
        self._benchmark_completed: set[str] = set()
        self._benchmark_active_enter_ts: Optional[float] = None
        self._benchmark_active_enter_ts_by_checkpoint: dict[str, float] = {}
        self._benchmark_last_update_ts: Optional[float] = None
        self._benchmark_last_distance_m: Optional[float] = None
        self._benchmark_prev_target: Optional[str] = None
        self._benchmark_prev_in_radius: bool = False
        self._benchmark_prev_dwell_satisfied: bool = False
        self._benchmark_prev_dwell_bucket: int = 0
        self._benchmark_prev_completed_ids: set[str] = set()
        self._benchmark_focus_checkpoint_id: Optional[str] = None
        self._benchmark_plan_checkpoint_sequence: list[str] = []
        self._benchmark_executed_gc_sequence: list[str] = []
        self._progress_event_cv = threading.Condition()
        self._progress_event_seq = 0
        self._progress_event_queue = deque(maxlen=256)
        self._progress_event_cursor_by_checkpoint: dict[str, int] = {}
        self.latest_ui_collision_probability = None
        self.latest_ui_collision_timestamp = 0.0
        self.planner_mode = str(os.getenv("TYPEFLY_PLANNER_MODE", "llm_baseline")).strip().lower()
        if self.planner_mode not in {"llm_baseline", "rule_baseline"}:
            self.planner_mode = "llm_baseline"
        self.user_heading_yaw_rad = 0.0
        self.manual_worker_selection_id = "worker_1"
        self.manual_worker_poses: dict[str, dict] = {}
        self.manual_worker_localization_state: dict[str, dict] = {}
        self.auto_replan_armed = True
        self.auto_replan_protection_remaining = 0
        self.replan_limit = int(FULL_REPLAN_LIMIT)
        self.current_task_description = ""
        self.last_heartbeat_ts = 0.0
        self._pending_heartbeat_replan_plan: Optional[str] = None
        self._pending_heartbeat_reason: str = ""
        self._runtime_replan_event = threading.Event()
        self._runtime_replan_reason: str = ""
        self._replan_response_history: list[dict] = []
        self._mission_original_plan: Optional[str] = None
        self._current_active_plan: Optional[str] = None
        self._latest_full_replan_response: Optional[str] = None
        self._agent_heartbeat_index = 0
        self._agent_pending_replan_records: list[dict] = []
        self._agent_feedback_memory_packets: list[dict] = []
        self._agent_ready_for_eval_records: list[dict] = []
        self._agent_eval_results_pending_commit: list[dict] = []
        self._agent_eval_lock = threading.Lock()
        self._agent_eval_thread: Optional[threading.Thread] = None
        self._agent_eval_inflight = False
        self._agent_eval_generation = 0
        self.mission_start_ts: Optional[float] = None
        self.mission_end_ts: Optional[float] = None
        self.final_mission_summary: Optional[dict] = None
        self.interrupted_for_replan = False
        self.entered_awaiting_replan_response = False
        self.execution_resumed_from_new_plan = False
        self.next_statement_executed_after_interrupt = False
        self._interrupt_pending_until_new_plan = False
        self.langgraph_runner = LangGraphOrchestrationRunner(self)
        self.set_selected_pipeline(self.selected_pipeline_id)

        # PX4_SIM optional managed user-position publisher lifecycle
        self._sim_user_publisher_proc: Optional[subprocess.Popen] = None
        self._owns_sim_user_publisher = False

    def _default_active_objective_set(self) -> dict:
        return {
            "active_zone_ids": [zone.id for zone in BENCHMARK_ZONES],
            "active_checkpoint_ids": list(BENCHMARK_CHECKPOINT_ORDER),
            "source": "default_all",
        }

    def _reset_benchmark_progress_tracking(self):
        self._benchmark_completed = set()
        self._benchmark_active_enter_ts = None
        self._benchmark_active_enter_ts_by_checkpoint = {}
        self._benchmark_last_update_ts = None
        self._benchmark_last_distance_m = None
        self._benchmark_prev_target = None
        self._benchmark_prev_in_radius = False
        self._benchmark_prev_dwell_satisfied = False
        self._benchmark_prev_dwell_bucket = 0
        self._benchmark_prev_completed_ids = set()
        self._benchmark_focus_checkpoint_id = None
        self._benchmark_plan_checkpoint_sequence = []
        self._benchmark_executed_gc_sequence = []
        with self._progress_event_cv:
            self._progress_event_queue.clear()
            self._progress_event_seq = 0
            self._progress_event_cursor_by_checkpoint.clear()
        self.update_benchmark_progress(
            completed_checkpoint_ids=[],
            current_target_checkpoint=None,
            in_radius=False,
            dwell_seconds=0.0,
            required_dwell_seconds=float(CHECKPOINT_DWELL_SECONDS),
            dwell_satisfied=False,
            active_enter_ts=None,
            completed=False,
            distance_to_target_m=None,
            drone_true_position=None,
            checkpoint_center=None,
            tick_ts=None,
        )

    def set_benchmark_progress_focus_checkpoint(self, checkpoint_id: Optional[str]):
        self._benchmark_focus_checkpoint_id = (None if checkpoint_id is None else str(checkpoint_id).upper())

    def _record_replan_response(
        self,
        *,
        source: str,
        reason: str,
        plan_text: str,
        raw_response: Optional[str] = None,
    ):
        plan = str(plan_text or "").strip()
        if not plan:
            return
        self._replan_response_history.append(
            {
                "ts": float(time.time()),
                "source": str(source),
                "reason": str(reason or ""),
                "plan": plan,
                "raw_response": str(raw_response or ""),
            }
        )
        if len(self._replan_response_history) > 20:
            self._replan_response_history = self._replan_response_history[-20:]
        self._latest_full_replan_response = plan

    def _build_execution_history_for_llm(self):
        history = self.execution_history
        if history is None:
            base_text = "(n/a)"
        elif isinstance(history, list):
            base_text = ";".join(str(v) for v in history)
        else:
            base_text = str(history)
        if not self._replan_response_history:
            return base_text
        replan_lines = [
            "replan_response_history:"
        ]
        for idx, item in enumerate(self._replan_response_history, start=1):
            replan_lines.append(
                f"- #{idx} source={item.get('source')} reason={item.get('reason')} plan={item.get('plan')}"
            )
        return f"{base_text}\n" + "\n".join(replan_lines)

    def _extract_checkpoint_sequence_from_plan(self, minispec: str) -> list[str]:
        text = str(minispec or "")
        found = re.findall(r"(?:gc|go_checkpoint)\(\s*['\"]?\s*([A-Za-z]\d+)\s*['\"]?\s*\)", text)
        seq = []
        for cid in found:
            cp = str(cid).upper()
            if cp in BENCHMARK_CHECKPOINTS_BY_ID:
                seq.append(cp)
        return seq

    def _resolve_progress_target(self, active_ids: set[str]) -> Optional[str]:
        focus_checkpoint = (None if self._benchmark_focus_checkpoint_id is None else str(self._benchmark_focus_checkpoint_id).upper())
        if focus_checkpoint is not None and focus_checkpoint in active_ids and focus_checkpoint not in self._benchmark_completed:
            return focus_checkpoint
        for cid in self._benchmark_plan_checkpoint_sequence:
            if cid in active_ids and cid not in self._benchmark_completed:
                return cid
        for cid in self._benchmark_executed_gc_sequence:
            if cid in active_ids and cid not in self._benchmark_completed:
                return cid
        return next((cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid in active_ids and cid not in self._benchmark_completed), None)

    def _emit_progress_event(
        self,
        *,
        event_type: str,
        checkpoint_id: Optional[str],
        in_radius: bool,
        dwell_seconds: float,
        required_dwell_seconds: float,
        dwell_satisfied: bool,
        completed: bool,
        timestamp: float,
        reason: Optional[str] = None,
        risk: Optional[float] = None,
    ):
        event = {
            "event_type": str(event_type),
            "checkpoint_id": (None if checkpoint_id is None else str(checkpoint_id).upper()),
            "in_radius": bool(in_radius),
            "dwell_seconds": float(dwell_seconds),
            "required_dwell_seconds": float(required_dwell_seconds),
            "dwell_satisfied": bool(dwell_satisfied),
            "completed": bool(completed),
            "timestamp": float(timestamp),
            "reason": reason,
            "risk": (None if risk is None else float(risk)),
        }
        with self._progress_event_cv:
            self._progress_event_seq += 1
            event["event_id"] = int(self._progress_event_seq)
            self._progress_event_queue.append(event)
            self._progress_event_cv.notify_all()
        print_debug(
            "[BENCHMARK-PROGRESS-EVENT] "
            f"id={event['event_id']} type={event_type} checkpoint={event.get('checkpoint_id')} "
            f"in_radius={in_radius} dwell={dwell_seconds:.3f}/{required_dwell_seconds:.3f} "
            f"dwell_satisfied={dwell_satisfied} completed={completed} reason={reason} risk={risk}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )

    def wait_for_checkpoint_progress_event(
        self,
        checkpoint_id: str,
        *,
        timeout_seconds: float = 8.0,
        risk_abort_threshold: float | None = PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD,
    ) -> dict:
        checkpoint_key = str(checkpoint_id or "").upper()
        deadline = time.time() + max(0.2, float(timeout_seconds))
        with self._progress_event_cv:
            last_seen_event_id = int(self._progress_event_cursor_by_checkpoint.get(checkpoint_key, 0))
        print_debug(
            "[BENCHMARK-WAIT-START] "
            f"checkpoint={checkpoint_key} cursor={last_seen_event_id} "
            f"queue_len={len(self._progress_event_queue)}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        self.set_benchmark_progress_focus_checkpoint(checkpoint_key)
        try:
            while True:
                now = time.time()
                if now >= deadline:
                    print_debug(
                        "[BENCHMARK-WAIT-TIMEOUT] "
                        f"checkpoint={checkpoint_key} "
                        f"queue_len={len(self._progress_event_queue)} "
                        f"last_seen_event_id={last_seen_event_id}",
                        env_var="TYPEFLY_VERBOSE_DEBUG",
                    )
                    return {
                        "event_type": "waiting_timeout",
                        "checkpoint_id": checkpoint_key,
                        "in_radius": False,
                        "dwell_seconds": 0.0,
                        "required_dwell_seconds": 0.0,
                        "dwell_satisfied": False,
                        "completed": False,
                        "timestamp": now,
                        "reason": f"timeout_{timeout_seconds:.1f}s",
                        "risk": None,
                    }

                snapshot = self.get_live_ui_snapshot()
                progress = snapshot.get("benchmark_progress") if isinstance(snapshot, dict) else {}
                safety = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
                risk = None if safety is None else float(getattr(safety, "predicted_collision_probability", 0.0))
                if (risk_abort_threshold is not None) and risk is not None and risk >= float(risk_abort_threshold):
                    return {
                        "event_type": "risk_abort",
                        "checkpoint_id": checkpoint_key,
                        "in_radius": bool(progress.get("in_radius", False)),
                        "dwell_seconds": float(progress.get("dwell_seconds", 0.0) or 0.0),
                        "required_dwell_seconds": float(progress.get("required_dwell_seconds", 0.0) or 0.0),
                        "dwell_satisfied": bool(progress.get("dwell_satisfied", False)),
                        "completed": checkpoint_key in set(str(v).upper() for v in progress.get("completed", [])),
                        "timestamp": time.time(),
                        "reason": f"risk={risk:.3f}>=threshold({risk_abort_threshold:.3f})",
                        "risk": float(risk),
                    }

                with self._progress_event_cv:
                    queue_list = list(self._progress_event_queue)
                    skipped_old = [
                        e for e in queue_list
                        if str(e.get("checkpoint_id", "")).upper() == checkpoint_key
                        and int(e.get("event_id", 0)) <= last_seen_event_id
                    ]
                    matching_events = [
                        e for e in queue_list
                        if str(e.get("checkpoint_id", "")).upper() == checkpoint_key
                        and int(e.get("event_id", 0)) > last_seen_event_id
                    ]
                    if matching_events:
                        event = dict(matching_events[-1])
                        last_seen_event_id = int(event.get("event_id", 0))
                        self._progress_event_cursor_by_checkpoint[checkpoint_key] = last_seen_event_id
                        print_debug(
                            "[BENCHMARK-WAIT-MATCH] "
                            f"checkpoint={checkpoint_key} event_id={event.get('event_id')} "
                            f"type={event.get('event_type')} queue_len={len(queue_list)} "
                            f"cursor={last_seen_event_id} skipped_old={len(skipped_old)}",
                            env_var="TYPEFLY_VERBOSE_DEBUG",
                        )
                        return event

                    print_debug(
                        "[BENCHMARK-WAIT-POLL] "
                        f"checkpoint={checkpoint_key} waiting_for={checkpoint_key} "
                        f"cursor={last_seen_event_id} "
                        f"progress_target={progress.get('current_target')} "
                        f"in_radius={progress.get('in_radius')} "
                        f"active_enter_ts={progress.get('active_enter_ts')} "
                        f"dwell_seconds={progress.get('dwell_seconds')} "
                        f"dwell_satisfied={progress.get('dwell_satisfied')} "
                        f"completed={progress.get('completed')} "
                        f"distance_m={progress.get('distance_to_target_m')} "
                        f"drone_pos={progress.get('drone_true_position')} "
                        f"cp_center={progress.get('checkpoint_center')} "
                        f"queue_len={len(queue_list)}",
                        env_var="TYPEFLY_VERBOSE_DEBUG",
                    )
                    wait_s = min(0.25, max(0.01, deadline - time.time()))
                    self._progress_event_cv.wait(timeout=wait_s)
        finally:
            self.set_benchmark_progress_focus_checkpoint(None)

    def _update_benchmark_progress_from_snapshot(self, snapshot: dict):
        if not isinstance(snapshot, dict):
            return
        drone_gt = snapshot.get("drone_gt")
        if drone_gt is None:
            return

        active_ids = set(str(v).upper() for v in self.active_objective_set.get("active_checkpoint_ids", []))
        self._benchmark_completed = set(cid for cid in self._benchmark_completed if cid in active_ids)
        current_target = self._resolve_progress_target(active_ids)
        now = time.time()
        self._benchmark_last_update_ts = now

        if current_target is None:
            self._benchmark_active_enter_ts = None
            self._benchmark_active_enter_ts_by_checkpoint = {}
            self._benchmark_last_distance_m = None
            self._benchmark_prev_target = None
            self._benchmark_prev_in_radius = False
            self._benchmark_prev_dwell_satisfied = False
            self._benchmark_prev_dwell_bucket = 0
            self._benchmark_prev_completed_ids = set(str(v).upper() for v in self._benchmark_completed)
            self.update_benchmark_progress(
                completed_checkpoint_ids=sorted(self._benchmark_completed),
                current_target_checkpoint=None,
                in_radius=False,
                dwell_seconds=0.0,
                required_dwell_seconds=float(CHECKPOINT_DWELL_SECONDS),
                dwell_satisfied=False,
                active_enter_ts=None,
                completed=bool(active_ids) and len(self._benchmark_completed) == len(active_ids),
                distance_to_target_m=None,
                drone_true_position=drone_gt,
                checkpoint_center=None,
                tick_ts=now,
            )
            return

        cp = BENCHMARK_CHECKPOINTS_BY_ID[current_target]
        distance_m = math.hypot(float(drone_gt[0]) - float(cp.x), float(drone_gt[1]) - float(cp.y))
        self._benchmark_last_distance_m = float(distance_m)
        in_radius = bool(distance_m <= float(cp.radius_m))
        dwell_seconds = 0.0
        target_enter_ts = self._benchmark_active_enter_ts_by_checkpoint.get(current_target)
        if in_radius:
            if target_enter_ts is None:
                target_enter_ts = now
                self._benchmark_active_enter_ts_by_checkpoint[current_target] = target_enter_ts
                print_debug(
                    "[BENCHMARK-DWELL] "
                    f"checkpoint={current_target} event=dwell_start ts={target_enter_ts:.6f}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            dwell_seconds = max(0.0, now - float(target_enter_ts))
            if dwell_seconds >= float(CHECKPOINT_DWELL_SECONDS):
                self._benchmark_completed.add(str(current_target).upper())
                self._benchmark_active_enter_ts_by_checkpoint.pop(current_target, None)
                print_debug(
                    "[BENCHMARK-DWELL] "
                    f"checkpoint={current_target} event=dwell_complete dwell={dwell_seconds:.3f}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
        else:
            if target_enter_ts is not None:
                print_debug(
                    "[BENCHMARK-DWELL] "
                    f"checkpoint={current_target} event=dwell_reset dwell={max(0.0, now - float(target_enter_ts)):.3f}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            self._benchmark_active_enter_ts_by_checkpoint.pop(current_target, None)
        self._benchmark_active_enter_ts = self._benchmark_active_enter_ts_by_checkpoint.get(current_target)

        dwell_satisfied = bool(dwell_seconds >= float(CHECKPOINT_DWELL_SECONDS))
        completed = bool(active_ids) and all(cid in self._benchmark_completed for cid in active_ids)
        checkpoint_center = (float(cp.x), float(cp.y), float(cp.radius_m))
        self.update_benchmark_progress(
            completed_checkpoint_ids=sorted(self._benchmark_completed),
            current_target_checkpoint=current_target,
            in_radius=in_radius,
            dwell_seconds=float(dwell_seconds),
            required_dwell_seconds=float(CHECKPOINT_DWELL_SECONDS),
            dwell_satisfied=dwell_satisfied,
            active_enter_ts=self._benchmark_active_enter_ts,
            completed=completed,
            distance_to_target_m=float(distance_m),
            drone_true_position=drone_gt,
            checkpoint_center=checkpoint_center,
            tick_ts=now,
        )
        print_debug(
            "[BENCHMARK-PROGRESS-TARGET] "
            f"focus={self._benchmark_focus_checkpoint_id} "
            f"plan_sequence={self._benchmark_plan_checkpoint_sequence} "
            f"executed_gc={self._benchmark_executed_gc_sequence} "
            f"current_target={current_target}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        current_completed_ids = set(str(v).upper() for v in self.latest_benchmark_progress.get("completed", []))
        required_dwell = float(CHECKPOINT_DWELL_SECONDS)
        dwell_bucket = int(dwell_seconds / 0.5)
        if self._benchmark_prev_target != current_target:
            self._benchmark_prev_in_radius = False
            self._benchmark_prev_dwell_satisfied = False
            self._benchmark_prev_dwell_bucket = 0
            self._benchmark_prev_completed_ids = set()
        entered_transition = bool(in_radius and (not self._benchmark_prev_in_radius))
        if in_radius and (not self._benchmark_prev_in_radius):
            self._emit_progress_event(
                event_type="entered_checkpoint_area",
                checkpoint_id=current_target,
                in_radius=in_radius,
                dwell_seconds=dwell_seconds,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=dwell_satisfied,
                completed=(str(current_target).upper() in current_completed_ids),
                timestamp=now,
                reason="entered_radius",
            )
        if entered_transition:
            self._emit_progress_event(
                event_type="dwell_started",
                checkpoint_id=current_target,
                in_radius=in_radius,
                dwell_seconds=dwell_seconds,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=dwell_satisfied,
                completed=(str(current_target).upper() in current_completed_ids),
                timestamp=now,
                reason="active_enter_ts_set",
            )
        if in_radius and dwell_bucket > self._benchmark_prev_dwell_bucket and (not dwell_satisfied):
            self._emit_progress_event(
                event_type="dwell_progress",
                checkpoint_id=current_target,
                in_radius=in_radius,
                dwell_seconds=dwell_seconds,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=dwell_satisfied,
                completed=(str(current_target).upper() in current_completed_ids),
                timestamp=now,
                reason="dwell_bucket_advanced",
            )
        if dwell_satisfied and (not self._benchmark_prev_dwell_satisfied):
            self._emit_progress_event(
                event_type="dwell_satisfied",
                checkpoint_id=current_target,
                in_radius=in_radius,
                dwell_seconds=dwell_seconds,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=dwell_satisfied,
                completed=(str(current_target).upper() in current_completed_ids),
                timestamp=now,
                reason="dwell_threshold_met",
            )
        if (not in_radius) and self._benchmark_prev_in_radius and (str(current_target).upper() not in current_completed_ids):
            self._emit_progress_event(
                event_type="left_checkpoint_area",
                checkpoint_id=current_target,
                in_radius=in_radius,
                dwell_seconds=dwell_seconds,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=dwell_satisfied,
                completed=False,
                timestamp=now,
                reason="left_before_completion",
            )
        newly_completed = sorted(current_completed_ids - self._benchmark_prev_completed_ids)
        for cid in newly_completed:
            self._emit_progress_event(
                event_type="checkpoint_completed",
                checkpoint_id=cid,
                in_radius=in_radius if cid == str(current_target).upper() else False,
                dwell_seconds=dwell_seconds if cid == str(current_target).upper() else 0.0,
                required_dwell_seconds=required_dwell,
                dwell_satisfied=(dwell_satisfied if cid == str(current_target).upper() else True),
                completed=True,
                timestamp=now,
                reason="added_to_completed_set",
            )
            print_debug(
                "[BENCHMARK-COMPLETION] "
                f"checkpoint={cid} completed=True completed_set={sorted(current_completed_ids)}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        self._benchmark_prev_target = str(current_target).upper()
        self._benchmark_prev_in_radius = bool(in_radius)
        self._benchmark_prev_dwell_satisfied = bool(dwell_satisfied)
        self._benchmark_prev_dwell_bucket = int(dwell_bucket)
        self._benchmark_prev_completed_ids = set(current_completed_ids)
        print_debug(
            "[BENCHMARK-PROGRESS-TICK] "
            f"current_target={current_target} "
            f"in_radius={in_radius} "
            f"active_enter_ts={self._benchmark_active_enter_ts} "
            f"dwell_seconds={dwell_seconds:.3f} "
            f"required_dwell_seconds={float(CHECKPOINT_DWELL_SECONDS):.3f} "
            f"dwell_satisfied={dwell_satisfied} "
            f"completed={completed} "
            f"checkpoint_center={checkpoint_center} "
            f"uav_true_position={tuple(float(v) for v in drone_gt)} "
            f"distance_m={distance_m:.3f} "
            f"tick_ts={now:.6f}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )

    def _resolve_active_objective_set(self, task_text: str) -> dict:
        text = str(task_text or "")
        normalized = text.upper()
        all_zone_ids = [zone.id for zone in BENCHMARK_ZONES]
        zone_to_checkpoints = {
            "zone_A": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("A")],
            "zone_B": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("B")],
            "zone_C": [cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid.startswith("C")],
        }

        all_keywords = (
            "ALL ZONES",
            "ALL CHECKPOINT",
            "COMPLETE ALL",
            "全部區域",
            "全部检查点",
            "全部檢查點",
            "全部巡檢點",
        )
        if any(key in normalized for key in all_keywords) or any(key in text for key in all_keywords):
            resolved = {
                "active_zone_ids": all_zone_ids,
                "active_checkpoint_ids": list(BENCHMARK_CHECKPOINT_ORDER),
                "source": "task_parse_all",
            }
            print_debug(
                "[OBJECTIVE-RESOLVE] "
                f"task={text!r} zones={resolved.get('active_zone_ids')} "
                f"checkpoints={resolved.get('active_checkpoint_ids')} source={resolved.get('source')}"
            )
            return resolved

        zone_tokens = set()
        context_hits = any(word in normalized for word in ("ZONE", "ZONES", "AREA", "CHECKPOINT", "INSPECT", "SEARCH"))
        context_hits = context_hits or any(word in text for word in ("區域", "巡檢", "搜尋", "搜索", "檢查點", "检查点"))
        for token, zone_id in (("A", "zone_A"), ("B", "zone_B"), ("C", "zone_C")):
            if re.search(rf"\b(?:ZONE|AREA)[\s_-]*{token}\b", normalized):
                zone_tokens.add(zone_id)
            if re.search(rf"\b{token}[\s_-]*(?:ZONE|AREA)\b", normalized):
                zone_tokens.add(zone_id)
            if re.search(rf"\b(?:ZONE|AREA){token}\b", normalized):
                zone_tokens.add(zone_id)
            if re.search(rf"\b{token}(?:ZONE|AREA)\b", normalized):
                zone_tokens.add(zone_id)
            if f"{token}區域" in text or f"{token} 區域" in text or f"區域{token}" in text:
                zone_tokens.add(zone_id)
            if context_hits and re.search(rf"\b{token}\b", normalized):
                zone_tokens.add(zone_id)

        if not zone_tokens:
            resolved = self._default_active_objective_set()
            print_debug(
                "[OBJECTIVE-RESOLVE] "
                f"task={text!r} zones={resolved.get('active_zone_ids')} "
                f"checkpoints={resolved.get('active_checkpoint_ids')} source={resolved.get('source')}"
            )
            return resolved

        active_zone_ids = sorted(zone_tokens)
        active_checkpoint_ids = []
        for zid in active_zone_ids:
            active_checkpoint_ids.extend(zone_to_checkpoints.get(zid, []))
        resolved = {
            "active_zone_ids": active_zone_ids,
            "active_checkpoint_ids": active_checkpoint_ids,
            "source": "task_parse_zone",
        }
        print_debug(
            "[OBJECTIVE-RESOLVE] "
            f"task={text!r} zones={resolved.get('active_zone_ids')} "
            f"checkpoints={resolved.get('active_checkpoint_ids')} source={resolved.get('source')}"
        )
        return resolved
        
    def register_position_callback(self, callback):
        self.position_update_callback = callback
        
    def notify_user_position_updated(self, position: Tuple[float, float, float, float]):
        timestamp, x, y, z = position
        position_str = f"User position updated: x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        if hasattr(self, 'position_update_callback') and self.position_update_callback:
            source = "user"
            self.position_update_callback(x, y, z, source)
    
    def _get_planner_position_context(self):
        drone_pos = (0.00, 0.00, 0.00)
        user_pos = (0.00, 0.00, 0.00)
        drone_source = "fallback_zero"
        user_source = "fallback_zero"

        try:
            provider = getattr(self, "state_provider", None)
            drone = getattr(self, "drone", None)

            if provider is not None:
                get_est_drone = getattr(provider, "get_estimated_drone_position", None)
                if callable(get_est_drone):
                    value = get_est_drone()
                    if value is not None:
                        drone_pos = tuple(float(v) for v in value)
                        drone_source = "provider_estimated"

                get_est_user = getattr(provider, "get_estimated_user_position", None)
                if callable(get_est_user):
                    value = get_est_user()
                    if value is not None:
                        user_pos = tuple(float(v) for v in value)
                        user_source = "provider_estimated"

                if drone_source == "fallback_zero":
                    packet = None
                    get_packet = getattr(provider, "get_latest_received_drone_packet", None)
                    if callable(get_packet):
                        packet = get_packet()
                    if packet is not None:
                        drone_pos = tuple(float(v) for v in packet.estimated_position_3d)
                        drone_source = "received_packet_estimated"

                if user_source == "fallback_zero":
                    packet = None
                    get_packet = getattr(provider, "get_latest_received_user_packet", None)
                    if callable(get_packet):
                        packet = get_packet()
                    if packet is not None:
                        user_pos = tuple(float(v) for v in packet.estimated_position_3d)
                        user_source = "received_packet_estimated"

                if drone_source == "fallback_zero":
                    get_gt_drone = getattr(provider, "get_drone_position", None)
                    if callable(get_gt_drone):
                        value = get_gt_drone()
                        if value is not None:
                            drone_pos = tuple(float(v) for v in value)
                            drone_source = "provider_position_fallback"

                if user_source == "fallback_zero":
                    get_user = getattr(provider, "get_user_position", None)
                    if callable(get_user):
                        value = get_user()
                        if value is not None:
                            user_pos = tuple(float(v) for v in value)
                            user_source = "provider_position_fallback"

            if drone_source == "fallback_zero" and drone is not None:
                get_drone_position = getattr(drone, "get_drone_position", None)
                if callable(get_drone_position):
                    value = get_drone_position()
                    if value is not None:
                        drone_pos = tuple(float(v) for v in value)
                        drone_source = "drone_position_fallback"
        except Exception:
            pass

        return {
            "drone_pos": drone_pos,
            "user_pos": user_pos,
            "drone_source": drone_source,
            "user_source": user_source,
        }

    def _format_planner_location_info(self) -> str:
        context = self._get_planner_position_context()
        drone_pos = context["drone_pos"]
        print_debug(
            "[P-LOCATION-CONTEXT] "
            f"drone_source={context['drone_source']} "
            f"drone_est={drone_pos}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}"

    def skill_get_drone_position(self) -> Tuple[str, bool]:
        context = self._get_planner_position_context()
        x, y, z = context["drone_pos"]
        position_str = f"Drone estimated position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def skill_get_user_position(self) -> Tuple[str, bool]:
        context = self._get_planner_position_context()
        x, y, z = context["user_pos"]
        position_str = f"User estimated position is x={x:.2f}, y={y:.2f}, z={z:.2f}"
       #self.append_message(f"[LOG] {position_str}")
        return position_str, False
        
    def start_uwb(self):
        print_t("[C] Starting UWB tracking...")
        self.state_provider.start()
        self.uwb_active = True
        
    def stop_uwb(self):
        print_t("[C] Stopping UWB tracking...")
        self.state_provider.stop()
        self.uwb_active = False
        
    def start_virtual_position_loop(self):
        self.virtual_position_active = True
        def loop():
            while self.controller_active and self.virtual_position_active:
                x, y, z = self.drone.get_drone_position()
                try:
                    self.virtual_queue.put_nowait((time.time(), x, y, z))
                except queue.Full:
                    self.virtual_queue.get()
                    self.virtual_queue.put_nowait((time.time(), x, y, z))
                if self.position_update_callback:
                    self.position_update_callback(x, y, z, "drone")
                time.sleep(0.1)
        threading.Thread(target=loop, daemon=True).start()
        
    def stop_virtual_position_loop(self):
        self.virtual_position_active = False


    def skill_time(self) -> Tuple[float, bool]:
        return time.time() - self.execution_time, False

    def skill_take_picture(self) -> Tuple[None, bool]:
        img_path = os.path.join(self.cache_folder, f"{uuid.uuid4()}.jpg")
        Image.fromarray(self.latest_frame).save(img_path)
        print_t(f"[C] Picture saved to {img_path}")
        self.append_message((img_path,))
        return None, False

    def skill_log(self, text: str) -> Tuple[None, bool]:
        self.append_message(f"[LOG] {text}")
        print_t(f"[LOG] {text}")
        return None, False
    
    def skill_re_plan(self) -> Tuple[None, bool]:
        print_debug("[REPLAN_DEBUG] source=rp_skill", env_var="TYPEFLY_VERBOSE_DEBUG")
        return None, True

    def skill_takeoff(self) -> Tuple[None, bool]:
        ok = self.drone.takeoff()
        if isinstance(ok, tuple):
            return ok
        return (None, not bool(ok))

    def skill_land(self) -> Tuple[None, bool]:
        self.drone.land()
        return None, False

    def skill_delay(self, s: float) -> Tuple[None, bool]:
        time.sleep(s)
        return None, False

    def skill_go_checkpoint(self, checkpoint_id: str) -> Tuple[str, bool]:
        checkpoint_key = str(checkpoint_id or "").strip().strip("'\"").upper()
        checkpoint = BENCHMARK_CHECKPOINTS_BY_ID.get(checkpoint_key)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint_id `{checkpoint_id}`")
        # Ensure dwell/completion tracking follows the actual commanded checkpoint,
        # instead of falling back to lexical benchmark order.
        self.set_benchmark_progress_focus_checkpoint(checkpoint_key)
        self._benchmark_executed_gc_sequence.append(checkpoint_key)
        print_debug(
            "[BENCHMARK-EXECUTOR] "
            f"event=gc_begin checkpoint={checkpoint_key} executed_gc={self._benchmark_executed_gc_sequence}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        # Keep an explicit robot-mode flag in scope for gc() diagnostics and to
        # avoid NameError regressions when instrumenting this block.
        is_px4_sim = isinstance(self.drone, Px4SimRobotWrapper)

        max_step_m = 1.0
        # gc policy: align to checkpoint heading first, then move straight forward.
        # Use tighter alignment thresholds to avoid diagonal-looking trajectories.
        heading_align_far_deg = 8.0
        heading_align_near_deg = 5.0
        max_turn_step_deg = 28

        initial_snapshot = self.get_live_ui_snapshot()
        initial_control = (
            initial_snapshot.get("drone_est_bias_corrected")
            or initial_snapshot.get("drone_est")
            or initial_snapshot.get("drone_gt")
            or (0.0, 0.0, 0.0)
        )
        initial_dist = math.hypot(float(checkpoint.x) - float(initial_control[0]), float(checkpoint.y) - float(initial_control[1]))
        max_iterations = max(12, min(80, int(math.ceil(initial_dist / 0.35)) + 6))

        reached = False
        final_dist = None
        best_dist = float("inf")
        stop_reason = "max_iterations"
        preempted_for_replan = False

        def _should_preempt_for_replan(current_p: float) -> bool:
            nonlocal reached, stop_reason, preempted_for_replan
            event_triggered, event_reason = self._consume_runtime_replan_event()
            if event_triggered:
                stop_reason = str(event_reason or "runtime_replan_event")
                print_t(f"[QUEUE] clearing remaining statements due to replan ({stop_reason})")
                preempted_for_replan = True
                return True
            objective_completed_now = self._is_active_objective_completed()
            if objective_completed_now:
                reached = True
                stop_reason = "objective_completed"
                return True
            if self._maybe_run_agent_heartbeat():
                stop_reason = f"agent_heartbeat_replan({self._pending_heartbeat_reason or 'llm'})"
                print_t(f"[QUEUE] clearing remaining statements due to replan")
                preempted_for_replan = True
                return True
            if self._should_trigger_auto_replan(current_p, source="go_checkpoint_loop"):
                print_t(
                    "[TYPEFLY-INTERRUPT] "
                    f"predicted_collision_probability={current_p:.6f} > {self.predicted_collision_replan_threshold:.2f}, "
                    "aborting current execution"
                )
                stop_reason = f"collision_probability_high({current_p:.3f})"
                preempted_for_replan = True
                return True
            return False

        for idx in range(max_iterations):
            snapshot = self.get_live_ui_snapshot()
            true_pos = snapshot.get("drone_gt")
            est_raw = snapshot.get("drone_est")
            est_bias = snapshot.get("drone_est_bias_corrected") or est_raw
            control_pos = est_bias or true_pos or (0.0, 0.0, 0.0)
            yaw = float(snapshot.get("drone_yaw_rad") or 0.0)
            progress = dict(snapshot.get("benchmark_progress") or {})
            completed_set = set(str(v).upper() for v in list(progress.get("completed") or []))
            completion_state = checkpoint.id in completed_set
            runtime_target = None if progress.get("current_target") is None else str(progress.get("current_target")).upper()
            progress_in_radius = bool(progress.get("in_radius", False))
            progress_target_match = bool(runtime_target == checkpoint.id)
            safety_context = snapshot.get("safety_context")
            current_p = 0.0 if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))
            if _should_preempt_for_replan(current_p):
                break

            dx_w = float(checkpoint.x) - float(control_pos[0])
            dy_w = float(checkpoint.y) - float(control_pos[1])
            dist_control = math.hypot(dx_w, dy_w)
            dist_true = None
            if true_pos is not None:
                dist_true = math.hypot(float(checkpoint.x) - float(true_pos[0]), float(checkpoint.y) - float(true_pos[1]))
            final_dist = dist_control

            in_radius_by_control = dist_control <= float(checkpoint.radius_m)
            in_radius_by_true = (dist_true is not None and dist_true <= float(checkpoint.radius_m))
            in_progress_radius = bool(progress_target_match and progress_in_radius)
            if true_pos is None:
                stop_condition = bool(completion_state or in_progress_radius or in_radius_by_control)
            else:
                # Do not require progress-target synchronization to stop at the commanded checkpoint.
                # If true position is already in checkpoint radius, gc should be considered reached.
                stop_condition = bool(completion_state or in_radius_by_true or in_progress_radius)

            body_forward = math.cos(yaw) * dx_w + math.sin(yaw) * dy_w
            body_right = -math.sin(yaw) * dx_w + math.cos(yaw) * dy_w

            chosen_action = "none"
            dist_trend = "improving" if dist_control + 0.02 < best_dist else "flat_or_worse"
            if stop_condition:
                reached = True
                if completion_state:
                    stop_reason = "completion_state"
                elif in_radius_by_true and in_progress_radius:
                    stop_reason = "true_radius+progress_in_radius"
                elif in_progress_radius:
                    stop_reason = "progress_in_radius"
                elif in_radius_by_true:
                    stop_reason = "true_radius"
                else:
                    stop_reason = "estimated_radius"
            else:
                if dist_control + 0.02 < best_dist:
                    best_dist = dist_control

                if is_px4_sim:
                    local_step_cap = 0.60 if dist_control < 0.80 else max_step_m
                else:
                    local_step_cap = 0.25 if dist_control < 0.35 else max_step_m

                desired_yaw = math.atan2(dy_w, dx_w)
                yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
                yaw_error_deg = math.degrees(yaw_error)
                align_threshold = heading_align_near_deg if dist_control < 0.8 else heading_align_far_deg
                if abs(yaw_error_deg) > align_threshold:
                    if _should_preempt_for_replan(current_p):
                        break
                    turn_deg = int(max(5.0, min(float(max_turn_step_deg), abs(yaw_error_deg))))
                    turn_cmd = "turn_ccw" if yaw_error_deg > 0 else "turn_cw"
                    if turn_cmd == "turn_ccw":
                        self.drone.turn_ccw(turn_deg)
                    else:
                        self.drone.turn_cw(turn_deg)
                    chosen_action = f"{turn_cmd}({turn_deg})"
                    if _should_preempt_for_replan(current_p):
                        break
                else:
                    if _should_preempt_for_replan(current_p):
                        break
                    if is_px4_sim:
                        forward_step = min(local_step_cap, max(0.35, dist_control * 0.90))
                    else:
                        forward_step = min(local_step_cap, max(0.15, dist_control * 0.65))
                    self.drone.move_forward(forward_step)
                    chosen_action = f"move_forward({forward_step:.2f})"
                    if _should_preempt_for_replan(current_p):
                        break

            print_debug(
                "[GC_DEBUG] "
                f"cp={checkpoint.id} "
                f"target=({checkpoint.x:.2f},{checkpoint.y:.2f}) "
                f"zone={checkpoint.zone_id} "
                f"iter={idx + 1}/{max_iterations} "
                f"true_pos={true_pos} est_raw={est_raw} est_bias={est_bias} "
                f"control_pos={control_pos} "
                f"dx={dx_w:.3f} dy={dy_w:.3f} "
                f"dist_control={dist_control:.3f} dist_true={'n/a' if dist_true is None else f'{dist_true:.3f}'} "
                f"yaw={yaw:.3f} "
                f"body_forward={body_forward:.3f} body_right={body_right:.3f} "
                f"runtime_target={runtime_target} "
                f"is_px4_sim={is_px4_sim} "
                f"action={chosen_action} "
                f"stop_condition={stop_condition} completion_state={completion_state} "
                f"dist_trend={dist_trend} "
                f"reason={stop_reason if stop_condition else 'continue'}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

            if stop_condition:
                break

        status = "reached" if reached else "approached"
        summary = (
            f"go_checkpoint({checkpoint.id}) {status}: "
            f"zone={checkpoint.zone_id}, target=({checkpoint.x:.2f},{checkpoint.y:.2f}), "
            f"remaining_dist={0.0 if final_dist is None else float(final_dist):.2f}m, "
            f"stop_reason={stop_reason}"
        )
        print_t(f"[C] {summary}")
        print_debug(
            "[BENCHMARK-EXECUTOR] "
            f"event=gc_end checkpoint={checkpoint.id} reached={reached} stop_reason={stop_reason}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        should_request_replan = bool(
            preempted_for_replan
            or str(stop_reason).startswith("collision_probability_high")
            or str(stop_reason).startswith("agent_heartbeat_replan")
            or str(stop_reason).startswith("runtime_replan_event")
        )
        if should_request_replan:
            self.interrupted_for_replan = True
            self.entered_awaiting_replan_response = True
            self._interrupt_pending_until_new_plan = True
        # Keep gc non-replan on pure kinematic convergence limits (e.g. max_iterations).
        return summary, should_request_replan

    def append_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def update_benchmark_progress(
        self,
        completed_checkpoint_ids,
        current_target_checkpoint,
        in_radius: Optional[bool] = None,
        dwell_seconds: Optional[float] = None,
        required_dwell_seconds: Optional[float] = None,
        dwell_satisfied: Optional[bool] = None,
        active_enter_ts: Optional[float] = None,
        completed: Optional[bool] = None,
        distance_to_target_m: Optional[float] = None,
        drone_true_position: Optional[tuple] = None,
        checkpoint_center: Optional[tuple] = None,
        tick_ts: Optional[float] = None,
    ):
        completed_ids = [str(v).upper() for v in list(completed_checkpoint_ids or [])]
        self.latest_benchmark_progress = {
            "completed": sorted(set(completed_ids)),
            "current_target": (None if current_target_checkpoint is None else str(current_target_checkpoint).upper()),
            "in_radius": (None if in_radius is None else bool(in_radius)),
            "dwell_seconds": (None if dwell_seconds is None else float(dwell_seconds)),
            "required_dwell_seconds": (None if required_dwell_seconds is None else float(required_dwell_seconds)),
            "dwell_satisfied": (None if dwell_satisfied is None else bool(dwell_satisfied)),
            "active_enter_ts": (None if active_enter_ts is None else float(active_enter_ts)),
            "completed_flag": (None if completed is None else bool(completed)),
            "distance_to_target_m": (None if distance_to_target_m is None else float(distance_to_target_m)),
            "drone_true_position": (
                None
                if drone_true_position is None
                else tuple(float(v) for v in drone_true_position)
            ),
            "checkpoint_center": (
                None
                if checkpoint_center is None
                else tuple(float(v) for v in checkpoint_center)
            ),
            "tick_ts": (None if tick_ts is None else float(tick_ts)),
            "plan_checkpoint_sequence": list(self._benchmark_plan_checkpoint_sequence),
            "executed_gc_sequence": list(self._benchmark_executed_gc_sequence),
        }

    def update_ui_collision_probability(self, predicted_collision_probability: Optional[float]):
        if predicted_collision_probability is None:
            return
        current_p = float(predicted_collision_probability)
        self.latest_ui_collision_probability = current_p
        self.latest_ui_collision_timestamp = time.time()
        if self._should_trigger_auto_replan(current_p, source="ui_event"):
            self._set_runtime_replan_event(
                reason=(
                    f"predicted_collision_probability={current_p:.6f}>="
                    f"{self.predicted_collision_replan_threshold:.2f}, source=ui_event"
                )
            )

    def _set_runtime_replan_event(self, reason: str):
        reason_text = str(reason or "runtime_replan_event")
        if not self._runtime_replan_event.is_set():
            self._runtime_replan_reason = reason_text
        self._runtime_replan_event.set()

    def _clear_runtime_replan_event(self):
        self._runtime_replan_event.clear()
        self._runtime_replan_reason = ""

    def _consume_runtime_replan_event(self) -> Tuple[bool, str]:
        if not self._runtime_replan_event.is_set():
            return False, ""
        reason = str(self._runtime_replan_reason or "runtime_replan_event")
        self._clear_runtime_replan_event()
        return True, reason

    def _on_statement_executed_for_replan(self):
        if bool(getattr(self, "_interrupt_pending_until_new_plan", False)):
            self.next_statement_executed_after_interrupt = True
        if self.auto_replan_protection_remaining > 0:
            self.auto_replan_protection_remaining -= 1
            print_debug(
                "[REPLAN_DEBUG] "
                f"protection_window_active remaining_statements={self.auto_replan_protection_remaining}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )

    def _normalize_framework_mode(self, framework_mode: str) -> str:
        normalized = str(framework_mode or MODE_TYPEFLY_ONESHOT).strip().lower()
        return normalized if normalized in SUPPORTED_FRAMEWORK_MODES else MODE_TYPEFLY_ONESHOT

    def _is_agent_heartbeat_mode(self) -> bool:
        return self.framework_mode in {MODE_AGENT_HEARTBEAT_SOFT, MODE_AGENT_HEARTBEAT_HARDGATE}

    def _is_threshold_replan_mode(self) -> bool:
        return self.framework_mode == MODE_TYPEFLY_THRESHOLD_REPLAN

    def _is_agent_feedback_pipeline(self) -> bool:
        return str(getattr(self, "selected_pipeline_id", "")).strip().lower() == "agent"

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_vec3(value):
        try:
            if value is None:
                return None
            x, y, z = value
            return [float(x), float(y), float(z)]
        except Exception:
            return None

    @staticmethod
    def _safe_vec2(value):
        try:
            if value is None:
                return None
            x, y = value
            return [float(x), float(y)]
        except Exception:
            return None

    def _compute_closest_worker_distance(self, snapshot: dict) -> Optional[float]:
        drone = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est") or snapshot.get("drone_gt")
        if drone is None:
            return None
        try:
            dx, dy = float(drone[0]), float(drone[1])
        except Exception:
            return None
        min_dist = None
        for worker in list(snapshot.get("workers") or []):
            wxy = worker.get("ui_xy") or worker.get("est_xy_bias_corrected") or worker.get("gt_xy")
            if not wxy:
                continue
            try:
                d = float(math.hypot(dx - float(wxy[0]), dy - float(wxy[1])))
            except Exception:
                continue
            if min_dist is None or d < min_dist:
                min_dist = d
        return min_dist

    def _build_agent_decision_context(self, snapshot: dict, heartbeat_index: int, timestamp: float) -> dict:
        safety_context = snapshot.get("safety_context")
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        completed = [str(v).upper() for v in list(benchmark_progress.get("completed") or [])]
        active_ids = [str(v).upper() for v in list((snapshot.get("active_objective_set") or {}).get("active_checkpoint_ids") or [])]
        remaining = [cid for cid in active_ids if cid not in set(completed)]
        workers_summary = []
        for worker in list(snapshot.get("workers") or []):
            workers_summary.append(
                {
                    "id": str(worker.get("id", "")),
                    "xy": self._safe_vec2(worker.get("ui_xy") or worker.get("est_xy_bias_corrected") or worker.get("gt_xy")),
                }
            )
        active_run = getattr(self.task_run_logger, "_active", None)
        return {
            "heartbeat_index": int(heartbeat_index),
            "timestamp": float(timestamp),
            "pipeline_id": str(self.selected_pipeline_id),
            "task_id": (None if active_run is None else str(getattr(active_run, "task_id", ""))),
            "run_id": (None if active_run is None else str(getattr(active_run, "run_id", ""))),
            "uav_estimated_position": self._safe_vec3(snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est") or snapshot.get("drone_gt")),
            "uav_heading": self._safe_float(snapshot.get("drone_yaw_rad")),
            "worker_positions_summary": workers_summary,
            "remaining_checkpoints": list(remaining),
            "current_target_checkpoint": benchmark_progress.get("current_target"),
            "task_progress_summary": {
                "completed_count": int(len(completed)),
                "active_count": int(len(active_ids)),
                "completion_ratio": (None if not active_ids else float(len(completed) / max(1, len(active_ids)))),
                "completed_checkpoints": list(completed),
            },
            "predicted_collision_probability": (None if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))),
            "per_worker_collision_probabilities": ([] if safety_context is None else list(getattr(safety_context, "per_worker_collision_probabilities", []) or [])),
            "dominant_risky_worker": (None if safety_context is None else str(getattr(safety_context, "dominant_threat_id", "") or "")),
            "active_objective_set": dict(snapshot.get("active_objective_set") or {}),
        }

    def _build_agent_outcome_metrics(self, snapshot: dict, now_ts: float) -> dict:
        safety_context = snapshot.get("safety_context")
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        completed = [str(v).upper() for v in list(benchmark_progress.get("completed") or [])]
        active_ids = [str(v).upper() for v in list((snapshot.get("active_objective_set") or {}).get("active_checkpoint_ids") or [])]
        return {
            "timestamp": float(now_ts),
            "risk": (None if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))),
            "closest_worker_distance": self._compute_closest_worker_distance(snapshot),
            "near_miss_count": int(snapshot.get("near_miss_count", 0) or 0),
            "collision_count": int(snapshot.get("collision_count", 0) or 0),
            "completed_checkpoint_count": int(len(completed)),
            "elapsed_time": (None if self.mission_start_ts is None else float(now_ts - float(self.mission_start_ts))),
            "task_progress_ratio": (None if not active_ids else float(len(completed) / max(1, len(active_ids)))),
            "current_target_checkpoint": benchmark_progress.get("current_target"),
        }

    def _maybe_mature_agent_replan_records(self, current_heartbeat_index: int, snapshot: dict, now_ts: float):
        if not self._is_agent_feedback_pipeline():
            return
        remaining = []
        current_metrics = self._build_agent_outcome_metrics(snapshot, now_ts=now_ts)
        for record in list(self._agent_pending_replan_records or []):
            replan_idx = int(record.get("replan_heartbeat_index", -1))
            if replan_idx + 1 != int(current_heartbeat_index):
                remaining.append(record)
                continue
            start = dict(record.get("baseline_metrics") or {})
            delta = {
                "replan_heartbeat_index": int(replan_idx),
                "matured_at_heartbeat_index": int(current_heartbeat_index),
                "observation_window_heartbeats": 1,
                "observation_window_seconds": (None if start.get("timestamp") is None else float(now_ts - float(start.get("timestamp")))),
                "execution_progress_observed": bool((current_metrics.get("completed_checkpoint_count", 0) - start.get("completed_checkpoint_count", 0)) != 0),
                "risk_delta": (None if start.get("risk") is None or current_metrics.get("risk") is None else float(current_metrics["risk"] - start["risk"])),
                "closest_worker_distance_delta": (None if start.get("closest_worker_distance") is None or current_metrics.get("closest_worker_distance") is None else float(current_metrics["closest_worker_distance"] - start["closest_worker_distance"])),
                "near_miss_delta": int(current_metrics.get("near_miss_count", 0) - int(start.get("near_miss_count", 0) or 0)),
                "collision_delta": int(current_metrics.get("collision_count", 0) - int(start.get("collision_count", 0) or 0)),
                "completed_checkpoint_delta": int(current_metrics.get("completed_checkpoint_count", 0) - int(start.get("completed_checkpoint_count", 0) or 0)),
                "elapsed_time_delta": (None if start.get("elapsed_time") is None or current_metrics.get("elapsed_time") is None else float(current_metrics["elapsed_time"] - start["elapsed_time"])),
                "task_progress_delta": (None if start.get("task_progress_ratio") is None or current_metrics.get("task_progress_ratio") is None else float(current_metrics["task_progress_ratio"] - start["task_progress_ratio"])),
                "current_target_changed": bool(start.get("current_target_checkpoint") != current_metrics.get("current_target_checkpoint")),
            }
            matured_record = {
                "decision_context": dict(record.get("decision_context") or {}),
                "chosen_action": dict(record.get("chosen_action") or {}),
                "outcome_delta": delta,
            }
            with self._agent_eval_lock:
                self._agent_ready_for_eval_records.append(matured_record)
            if bool(self.archive_enabled):
                self.task_run_logger.append_planning_trace(
                    trace={
                        "planning_stage": "heartbeat",
                        "llm_call_purpose": "agent_replan_matured_ready_for_eval",
                        "plan_source": "agent_feedback_eval_queue",
                        "parsed_plan": {
                            "replan_record": matured_record,
                        },
                        "selected_baseline_id": self.selected_pipeline_id,
                        "scene_id": self.baseline_scene_id,
                    }
                )
        self._agent_pending_replan_records = remaining

    def _start_agent_eval_worker_if_needed(self):
        if not self._is_agent_feedback_pipeline():
            return
        with self._agent_eval_lock:
            if self._agent_eval_inflight:
                return
            if not self._agent_ready_for_eval_records:
                return
            generation = int(self._agent_eval_generation)
            self._agent_eval_inflight = True
            self._agent_eval_thread = threading.Thread(
                target=self._agent_eval_worker_loop,
                args=(generation,),
                daemon=True,
            )
            self._agent_eval_thread.start()

    def _agent_eval_worker_loop(self, generation: int):
        while True:
            with self._agent_eval_lock:
                if generation != int(self._agent_eval_generation):
                    self._agent_eval_inflight = False
                    self._agent_eval_thread = None
                    return
                if not self._agent_ready_for_eval_records:
                    self._agent_eval_inflight = False
                    self._agent_eval_thread = None
                    return
                record = self._agent_ready_for_eval_records.pop(0)
            eval_result = self.planner.evaluate_agent_replan_record(record)
            completed_ts = float(time.time())
            with self._agent_eval_lock:
                if generation != int(self._agent_eval_generation):
                    continue
                self._agent_eval_results_pending_commit.append(
                    {
                        "replan_record": record,
                        "evaluator_prompt": eval_result.get("prompt"),
                        "evaluator_raw_response": eval_result.get("raw_response"),
                        "evaluator_used_model_name": eval_result.get("used_model_name"),
                        "evaluator_parsed_json": eval_result.get("parsed"),
                        "evaluator_completed_ts": completed_ts,
                    }
                )

    def _commit_agent_eval_results(self, current_heartbeat_index: int):
        if not self._is_agent_feedback_pipeline():
            return
        with self._agent_eval_lock:
            pending = list(self._agent_eval_results_pending_commit)
            self._agent_eval_results_pending_commit = []
        if not pending:
            return
        for item in pending:
            replan_record = dict(item.get("replan_record") or {})
            outcome_delta = dict(replan_record.get("outcome_delta") or {})
            replan_heartbeat_index = int(outcome_delta.get("replan_heartbeat_index", -1))
            available_from_heartbeat_index = max(
                int(replan_heartbeat_index) + 2,
                int(current_heartbeat_index),
            )
            packet = {
                "decision_context": dict(replan_record.get("decision_context") or {}),
                "chosen_action": dict(replan_record.get("chosen_action") or {}),
                "outcome_delta": outcome_delta,
                "evaluator_feedback": dict(item.get("evaluator_parsed_json") or {}),
                "available_from_heartbeat_index": int(available_from_heartbeat_index),
            }
            self._agent_feedback_memory_packets.append(packet)
            if bool(self.archive_enabled):
                self.task_run_logger.append_planning_trace(
                    trace={
                        "planning_stage": "heartbeat",
                        "llm_call_purpose": "agent_replan_evaluator",
                        "plan_source": "agent_feedback_eval",
                        "prompt": str(item.get("evaluator_prompt") or ""),
                        "raw_response": str(item.get("evaluator_raw_response") or ""),
                        "parsed_plan": {
                            "replan_record": replan_record,
                            "evaluator_used_model_name": item.get("evaluator_used_model_name"),
                            "evaluator_parsed_json": item.get("evaluator_parsed_json"),
                            "matured_feedback_packet": packet,
                            "evaluator_completed_ts": item.get("evaluator_completed_ts"),
                        },
                        "selected_baseline_id": self.selected_pipeline_id,
                        "scene_id": self.baseline_scene_id,
                    }
                )

    def _build_injected_feedback_memory(self, heartbeat_index: int) -> list[dict]:
        packets = []
        for packet in list(self._agent_feedback_memory_packets or []):
            if int(packet.get("available_from_heartbeat_index", 10**9)) <= int(heartbeat_index):
                packets.append(packet)
        return packets

    def _pending_execution_statement_count(self) -> int:
        queue_obj = getattr(Statement, "execution_queue", None)
        if queue_obj is None:
            return 0
        try:
            return int(queue_obj.qsize())
        except Exception:
            return 0

    def _is_active_objective_completed(self) -> bool:
        active_ids = set(str(v).upper() for v in self.active_objective_set.get("active_checkpoint_ids", []))
        if not active_ids:
            return False
        completed = set(str(v).upper() for v in (self.latest_benchmark_progress.get("completed") or []))
        return all(cid in completed for cid in active_ids)

    def _should_skip_heartbeat_after_task_completion(self) -> bool:
        if not self._is_active_objective_completed():
            return False
        if self._pending_execution_statement_count() > 0:
            return False
        # When mission objectives are complete and there are no remaining execution statements,
        # heartbeat should stop calling LLM continuously.
        return True

    def _maybe_run_agent_heartbeat(self, force: bool = False) -> bool:
        if not self._is_agent_heartbeat_mode():
            return False
        if self._pending_heartbeat_replan_plan:
            return True
        if self._should_skip_heartbeat_after_task_completion():
            print_debug(
                "[AGENT-HEARTBEAT] skipped after task completion "
                f"pending_statements={self._pending_execution_statement_count()}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
            return False
        now = time.time()
        if (not force) and (now - float(self.last_heartbeat_ts) < float(self.heartbeat_interval_seconds)):
            return False
        self.last_heartbeat_ts = now
        if getattr(self, "_replan_attempts", 0) >= int(self.replan_limit):
            print_t(f"[REPLAN-COUNT] current={self._replan_attempts} limit={self.replan_limit}")
            return False
        snapshot = self.get_live_ui_snapshot() or {}
        heartbeat_index = int(self._agent_heartbeat_index) + 1
        now_ts = time.time()
        self._commit_agent_eval_results(current_heartbeat_index=heartbeat_index)
        self._maybe_mature_agent_replan_records(
            current_heartbeat_index=heartbeat_index,
            snapshot=snapshot,
            now_ts=now_ts,
        )
        feedback_memory_injected = self._build_injected_feedback_memory(heartbeat_index)
        response = self.planner.plan_agent_heartbeat(
            task_description=self.current_task_description,
            snapshot=snapshot,
            execution_history=self._build_execution_history_for_llm(),
            current_plan=self.current_plan,
            mission_original_plan=self._mission_original_plan,
            current_active_plan=self._current_active_plan,
            latest_full_replan_response=self._latest_full_replan_response,
            full_replan_count=int(getattr(self, "_replan_attempts", 0)),
            hard_gate=(self.framework_mode == MODE_AGENT_HEARTBEAT_HARDGATE),
            feedback_memory_packets=feedback_memory_injected,
        )
        self._agent_heartbeat_index = heartbeat_index
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        completed = [str(v).upper() for v in list(benchmark_progress.get("completed") or [])]
        active_ids = [str(v).upper() for v in list((snapshot.get("active_objective_set") or {}).get("active_checkpoint_ids") or [])]
        remaining = [cid for cid in active_ids if cid not in set(completed)]
        should_log_heartbeat_trace = not (self._is_agent_feedback_pipeline() and (not bool(self.archive_enabled)))
        if should_log_heartbeat_trace:
            self.task_run_logger.append_planning_trace(
                trace={
                    **self.planner.get_last_heartbeat_trace(),
                    "planning_stage": "heartbeat",
                    "plan_source": "heartbeat_decision",
                    "llm_call_purpose": "heartbeat",
                    "selected_baseline_id": self.selected_pipeline_id,
                    "scene_id": self.baseline_scene_id,
                    "current_target_checkpoint": benchmark_progress.get("current_target"),
                    "parsed_plan": (
                        {
                            "feedback_memory_injected": feedback_memory_injected,
                            "heartbeat_used_model_name": self.planner.get_last_heartbeat_trace().get("used_model_name"),
                        }
                        if bool(self.archive_enabled)
                        else None
                    ),
                    "true_completed_checkpoints": [str(v).upper() for v in list(completed or [])],
                    "true_remaining_checkpoints": [str(v).upper() for v in list(remaining or [])],
                    "completion_state_source": "benchmark_progress/dwell_tracker",
                }
            )
        raw_response = str(response.get("raw_response", "") or "").strip()
        if raw_response:
            self.append_message(f"[AGENT-HEARTBEAT-RAW] {raw_response}")
        response_type = str(response.get("response", "continue"))
        reason = str(response.get("reason", ""))
        if response_type == "full_replan_plan":
            plan_text = self._sanitize_minispec_plan(response.get("plan", ""))
            if plan_text:
                if self._is_agent_feedback_pipeline():
                    replan_record = {
                        "replan_heartbeat_index": int(heartbeat_index),
                        "decision_context": self._build_agent_decision_context(
                            snapshot=snapshot,
                            heartbeat_index=heartbeat_index,
                            timestamp=now_ts,
                        ),
                        "chosen_action": {
                            "raw_llm_response": raw_response,
                        },
                        "baseline_metrics": self._build_agent_outcome_metrics(snapshot, now_ts=now_ts),
                    }
                    self._agent_pending_replan_records.append(replan_record)
                    if bool(self.archive_enabled):
                        self.task_run_logger.append_planning_trace(
                            trace={
                                "planning_stage": "heartbeat",
                                "llm_call_purpose": "agent_replan_record",
                                "plan_source": "agent_replan_record_created",
                                "parsed_plan": {
                                    "replan_record": {
                                        "decision_context": replan_record.get("decision_context"),
                                        "chosen_action": replan_record.get("chosen_action"),
                                    }
                                },
                                "selected_baseline_id": self.selected_pipeline_id,
                                "scene_id": self.baseline_scene_id,
                            }
                        )
                self._pending_heartbeat_replan_plan = plan_text
                self._pending_heartbeat_reason = reason
                self._set_runtime_replan_event(
                    reason=f"agent_heartbeat_replan:{reason if reason else 'llm'}"
                )
                self._record_replan_response(
                    source="agent_heartbeat_full_replan_response",
                    reason=reason,
                    plan_text=plan_text,
                    raw_response=raw_response,
                )
                print_t(f"[AGENT-HEARTBEAT] response=replan plan={plan_text}")
                self.append_message(
                    f"[AGENT-HEARTBEAT-REPLAN] reason={reason if reason else 'n/a'}"
                )
                self.append_message(f"[AGENT-HEARTBEAT-REPLAN-PLAN] {plan_text}")
                self._start_agent_eval_worker_if_needed()
                return True
        self._start_agent_eval_worker_if_needed()
        print_t(f"[AGENT-HEARTBEAT] response=continue reason={reason}")
        return False

    def _should_trigger_auto_replan(self, predicted_p: float, source: str) -> bool:
        if not self._is_threshold_replan_mode():
            return False
        if int(getattr(self, "_replan_attempts", 0)) >= int(self.replan_limit):
            print_debug(
                "[REPLAN_DEBUG] "
                f"auto_replan_suppressed p={float(predicted_p):.6f} reason=max_replan_attempts_reached "
                f"current={int(getattr(self, '_replan_attempts', 0))} limit={int(self.replan_limit)} source={source}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
            return False
        predicted_p = float(predicted_p)
        threshold = float(self.predicted_collision_replan_threshold)
        rearm_threshold = float(self.predicted_collision_rearm_threshold)
        if self.auto_replan_protection_remaining > 0:
            if str(source) == "go_checkpoint_loop" and predicted_p >= threshold:
                print_debug(
                    "[REPLAN_DEBUG] "
                    f"protection_window_bypassed_for_gc p={predicted_p:.6f} "
                    f"remaining_statements={self.auto_replan_protection_remaining} source={source}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            else:
                print_debug(
                    "[REPLAN_DEBUG] "
                    f"auto_replan_suppressed p={predicted_p:.6f} reason=protection_window "
                    f"remaining_statements={self.auto_replan_protection_remaining} source={source}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
                return False

        if not self.auto_replan_armed:
            if predicted_p <= rearm_threshold:
                self.auto_replan_armed = True
                print_debug(
                    "[REPLAN_DEBUG] "
                    f"auto_replan_rearmed p={predicted_p:.6f} "
                    f"threshold={rearm_threshold:.2f}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            else:
                print_debug(
                    "[REPLAN_DEBUG] "
                    f"auto_replan_suppressed p={predicted_p:.6f} reason=disarmed source={source}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            return False

        trigger_replan = (predicted_p > threshold) if bool(self.predicted_collision_replan_strictly_greater) else (predicted_p >= threshold)
        if trigger_replan:
            self.auto_replan_armed = False
            print_debug(
                "[REPLAN_DEBUG] "
                f"auto_replan_triggered p={predicted_p:.6f} armed=True source={source} "
                f"trigger_threshold={threshold:.2f}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
            print_debug("[REPLAN_DEBUG] auto_replan_armed=False", env_var="TYPEFLY_VERBOSE_DEBUG")
            return True
        return False

    def stop_controller(self):
        self.controller_active = False

    def get_latest_frame(self, plot=False):
        image = self.shared_frame.get_image()
        if plot and image:
            self.vision.update()
            YoloClient.plot_results_oi(image, self.vision.object_list)
        return image
    
    def execute_minispec(self, minispec: str, silent: bool = False, allow_auto_interrupt: bool = True):
        statement_count = len([segment for segment in str(minispec or "").split(";") if str(segment).strip()])
        print_t(
            "[PLAN-QUEUE] "
            f"enqueued replan statements count={statement_count} program={str(minispec or '').strip()}"
        )
        self._benchmark_plan_checkpoint_sequence = self._extract_checkpoint_sequence_from_plan(minispec)
        print_debug(
            "[BENCHMARK-PLAN-PARSE] "
            f"gc_sequence={self._benchmark_plan_checkpoint_sequence}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        focus_checkpoint = self._extract_first_checkpoint_from_plan(minispec)
        if focus_checkpoint is not None:
            self.set_benchmark_progress_focus_checkpoint(focus_checkpoint)
        interpreter = MiniSpecInterpreter(
            None if silent else self.message_queue,
            should_abort=(self._should_abort_current_execution_for_replan if allow_auto_interrupt else None),
            on_statement_executed=self._on_statement_executed_for_replan,
        )
        interpreter.execute(minispec)
        self.execution_history = interpreter.execution_history
        ret_val = interpreter.ret_queue.get()
        if hasattr(ret_val, "value") and isinstance(ret_val.value, str) and ret_val.value.startswith("MiniSpec execution error"):
            raise RuntimeError(ret_val.value)
        return ret_val

    def _extract_first_checkpoint_from_plan(self, minispec: str) -> Optional[str]:
        text = str(minispec or "")
        m = re.search(r"(?:gc|go_checkpoint)\(\s*['\"]?\s*([A-Za-z]\d+)\s*['\"]?\s*\)", text)
        if not m:
            return None
        checkpoint_id = str(m.group(1)).upper()
        if checkpoint_id in BENCHMARK_CHECKPOINTS_BY_ID:
            return checkpoint_id
        return None

    def _should_abort_current_execution_for_replan(self) -> Tuple[bool, str]:
        if self._is_active_objective_completed():
            return False, ""
        event_triggered, event_reason = self._consume_runtime_replan_event()
        if event_triggered:
            print_t("[QUEUE] clearing remaining statements due to replan")
            self.interrupted_for_replan = True
            self.entered_awaiting_replan_response = True
            self._interrupt_pending_until_new_plan = True
            return True, event_reason
        if self._maybe_run_agent_heartbeat():
            print_t("[QUEUE] clearing remaining statements due to replan")
            self.interrupted_for_replan = True
            self.entered_awaiting_replan_response = True
            self._interrupt_pending_until_new_plan = True
            return True, f"agent_heartbeat_replan:{self._pending_heartbeat_reason or 'llm'}"
        if not self._is_threshold_replan_mode():
            return False, ""
        snapshot = self.get_live_ui_snapshot()
        if not isinstance(snapshot, dict):
            return False, ""
        safety_context = snapshot.get("safety_context")
        if safety_context is None:
            return False, ""
        callback_p = float(getattr(safety_context, "predicted_collision_probability", 0.0))
        current_p = callback_p
        ui_p = self.latest_ui_collision_probability
        ui_is_fresh = bool(ui_p is not None and (time.time() - float(self.latest_ui_collision_timestamp)) <= 1.5)
        if ui_is_fresh:
            current_p = max(float(current_p), float(ui_p))
        should_abort = self._should_trigger_auto_replan(current_p, source="interpreter_callback")
        if should_abort:
            dominant = str(getattr(safety_context, "dominant_threat_id", "unknown"))
            ui_p_text = "n/a" if ui_p is None else f"{float(ui_p):.6f}"
            print_t(
                "[TYPEFLY-INTERRUPT] "
                f"predicted_collision_probability={current_p:.6f} > {self.predicted_collision_replan_threshold:.2f}, "
                "aborting current execution"
            )
            self.interrupted_for_replan = True
            self.entered_awaiting_replan_response = True
            self._interrupt_pending_until_new_plan = True
            print_t(
                "[REPLAN_DEBUG] "
                f"source=collision_threshold "
                f"ui_pc={ui_p_text} "
                f"callback_pc={callback_p:.6f} "
                f"decision_pc={current_p:.6f} "
                f"threshold={self.predicted_collision_replan_threshold:.6f} "
                f"should_abort={should_abort} "
                f"dominant={dominant}"
            )
            return True, (
                f"predicted_collision_probability={current_p:.6f}>="
                f"{self.predicted_collision_replan_threshold:.2f}, dominant={dominant}"
            )
        return False, ""

    def _sanitize_minispec_plan(self, raw_plan: str) -> str:
        if raw_plan is None:
            return ""
        text = str(raw_plan).strip()
        if not text:
            return ""

        response_match = re.search(r"(?is)\bresponse\s*:\s*(.+)$", text)
        if response_match:
            text = response_match.group(1).strip()

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            if re.match(r"(?is)^(plan|analysis|thought|reasoning)\s*:", stripped):
                continue
            lines.append(stripped)
        text = " ".join(lines).strip()

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,;:?!&|<>=(){}[]'\"+-*/% \t")
        filtered = "".join(ch for ch in text if ch in allowed_chars).strip()
        if filtered:
            text = filtered

        command_pattern = re.compile(
            r"([A-Za-z_][A-Za-z0-9_-]*\s*\([^;{}]*\)|->[A-Za-z0-9_.'\"+\-*/%()]+|[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;{}]+|\d+\s*\{|[{}])\s*;?",
        )
        matches = [m.group(1).strip() for m in command_pattern.finditer(text)]
        if matches:
            rebuilt = []
            for token in matches:
                if token in {"{", "}"}:
                    rebuilt.append(token)
                elif token.endswith("{"):
                    rebuilt.append(token)
                else:
                    rebuilt.append(f"{token};")
            candidate = "".join(rebuilt).strip()
            if candidate:
                return candidate
        return text

    def _has_live_sim_user_position(self) -> bool:
        if not hasattr(self, "state_provider"):
            return False
        last_ts = getattr(self.state_provider, "_last_user_position_ts", 0.0)
        return bool(last_ts and (time.time() - float(last_ts) < 1.5))

    def _start_sim_user_position_publisher_if_needed(self):
        if self.robot_type != RobotType.PX4_SIM:
            return

        autostart = os.getenv("SIM_USER_POSITION_AUTOSTART", "1").strip().lower()
        if autostart in {"0", "false", "no", "off"}:
            return

        # If external source already publishes user position, do not start another publisher.
        deadline = time.time() + 0.6
        while time.time() < deadline:
            if self._has_live_sim_user_position():
                return
            time.sleep(0.1)

        if self._sim_user_publisher_proc is not None and self._sim_user_publisher_proc.poll() is None:
            return

        script_path = os.path.join(CURRENT_DIR, "sim_user_position_publisher.py")
        if not os.path.exists(script_path):
            print_t(f"[WARN] sim user publisher script not found: {script_path}")
            return

        topic = os.getenv("SIM_USER_POSITION_TOPIC", "/sim/user_position")
        x = os.getenv("SIM_USER_POSITION_PUB_X", "8.0")
        y = os.getenv("SIM_USER_POSITION_PUB_Y", "8.0")
        z = os.getenv("SIM_USER_POSITION_PUB_Z", "0.0")
        rate = os.getenv("SIM_USER_POSITION_PUB_RATE", "10.0")

        cmd = [sys.executable, script_path, "--topic", topic, "--x", x, "--y", y, "--z", z, "--rate", rate]
        try:
            self._sim_user_publisher_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._owns_sim_user_publisher = True
            print_t(f"[C] Started sim user position publisher on {topic}")
        except Exception as exc:
            print_t(f"[WARN] Failed to start sim user position publisher: {exc}")

    def _stop_sim_user_position_publisher(self):
        if not self._owns_sim_user_publisher:
            return
        proc = self._sim_user_publisher_proc
        if proc is None:
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

        self._sim_user_publisher_proc = None
        self._owns_sim_user_publisher = False

    def execute_task_description(self, task_description: str, framework_mode: str = MODE_TYPEFLY_ONESHOT):
        if self.controller_wait_takeoff:
            self.append_message("[Warning] Controller is waiting for takeoff...")
            return
        pipeline = self.get_selected_pipeline_config()
        if pipeline.id in {"baseline1", "baseline2"}:
            framework_mode = MODE_AGENT_HEARTBEAT_SOFT
        elif pipeline.id == "agent":
            framework_mode = MODE_AGENT_HEARTBEAT_SOFT
        elif pipeline.id == "baseline3":
            framework_mode = MODE_TYPEFLY_THRESHOLD_REPLAN
        selected_framework = self._normalize_framework_mode(framework_mode)
        self.framework_mode = selected_framework
        self.set_selected_pipeline(pipeline.id)
        self.near_miss_count = 0
        self.collision_count = 0
        self.min_uav_worker_distance_m = None
        self._near_miss_active_by_worker = {wid: False for wid in COLLISION_RISK_WORKER_IDS}
        self._collision_active_by_worker = {wid: False for wid in COLLISION_RISK_WORKER_IDS}
        self._latest_near_miss_events = []
        self.current_task_description = str(task_description or "")
        print_t(f"[MODE] selected={selected_framework}")
        self.execution_mode = "Planning"
        self.active_objective_set = self._resolve_active_objective_set(task_description)
        self._reset_benchmark_progress_tracking()
        self._task_id_counter += 1
        task_id = f"task_{self._task_id_counter:05d}"
        initial_snapshot = self.get_live_ui_snapshot()
        self.task_run_logger.start_run(
            task_id=task_id,
            task_text=task_description,
            scenario_name=self.get_active_scenario_name(),
            initial_snapshot=initial_snapshot,
            archive_enabled=bool(self.archive_enabled),
            run_context={
                "selected_baseline_id": pipeline.id,
                "selected_baseline_name": pipeline.name,
                "trigger_type": pipeline.trigger_type,
                "trigger_params": dict(pipeline.trigger_params),
                "prompt_variant": pipeline.prompt_variant,
                "example_variant": pipeline.example_variant,
                "state_fields": list(pipeline.state_fields),
                "use_output_example": bool(pipeline.use_output_example),
                "archive_enabled": bool(self.archive_enabled),
                "scene_id": self.baseline_scene_id,
                "baseline_scene_id": self.baseline_scene_id,
                "framework_mode": selected_framework,
            },
        )
        self.append_message('[TASK]: ' + task_description)
        ret_val = None
        monitor_stop = threading.Event()
        monitor_thread = None
        replan_attempts = 0
        max_replan_attempts = int(self.replan_limit)
        self._replan_attempts = 0
        self.auto_replan_armed = True
        self.auto_replan_protection_remaining = 0
        self._runtime_replan_event.clear()
        self._runtime_replan_reason = ""
        self.last_heartbeat_ts = 0.0
        self._pending_heartbeat_replan_plan = None
        self._pending_heartbeat_reason = ""
        self._replan_response_history = []
        self._mission_original_plan = None
        self._current_active_plan = None
        self._latest_full_replan_response = None
        self._agent_heartbeat_index = 0
        self._agent_pending_replan_records = []
        self._agent_feedback_memory_packets = []
        with self._agent_eval_lock:
            self._agent_eval_generation += 1
            self._agent_ready_for_eval_records = []
            self._agent_eval_results_pending_commit = []
            self._agent_eval_inflight = False
            self._agent_eval_thread = None
        self.mission_start_ts = time.time()
        self.mission_end_ts = None
        self.final_mission_summary = None
        self.interrupted_for_replan = False
        self.entered_awaiting_replan_response = False
        self.execution_resumed_from_new_plan = False
        self.next_statement_executed_after_interrupt = False
        self._interrupt_pending_until_new_plan = False
        def _run_monitor():
            while not monitor_stop.is_set():
                try:
                    self.task_run_logger.consume_runtime_snapshot(self.get_live_ui_snapshot())
                except Exception:
                    pass
                monitor_stop.wait(0.2)
        monitor_thread = threading.Thread(target=_run_monitor, daemon=True)
        monitor_thread.start()
        while True:
            location_info = self._format_planner_location_info()
            runtime_snapshot = self.get_live_ui_snapshot()

            scene_description = self.vision.get_obj_list() if self.enable_video else ''
            if hasattr(self.state_provider, "debug_log_latest_localization_snapshot"):
                self.state_provider.debug_log_latest_localization_snapshot(reason="pre-plan")
            safety_context = runtime_snapshot.get("safety_context") if isinstance(runtime_snapshot, dict) else None
            if safety_context is None:
                safety_context = self.state_provider.get_latest_safety_context() if hasattr(self.state_provider, "get_latest_safety_context") else None
            if safety_context is None:
                safety_context = self.safety_assessor.build_from_provider(self.state_provider)
            self._debug_log_safety_context(safety_context)
            
            try:
                baseline = self._build_baseline_control_plan(task_description=task_description, snapshot=initial_snapshot)
                if baseline is not None:
                    self.latest_baseline_decision = baseline
                    self.task_run_logger.update_baseline_info(baseline)

                llm_called = False
                final_plan_source = "llm"
                baseline_shortcut_triggered = False
                if self.planner_mode == "rule_baseline" and baseline is not None:
                    self.current_plan = baseline["plan"]
                    llm_called = False
                    final_plan_source = "baseline_rule"
                    baseline_shortcut_triggered = True
                else:
                    previous_plan = self.current_plan
                    self.execution_mode = "Planning"
                    planning_stage = ("initial" if replan_attempts == 0 else "replan")
                    if self._pending_heartbeat_replan_plan:
                        self.current_plan = self._pending_heartbeat_replan_plan
                        self._pending_heartbeat_replan_plan = None
                        self._pending_heartbeat_reason = ""
                        self._clear_runtime_replan_event()
                        if self._interrupt_pending_until_new_plan:
                            self.execution_resumed_from_new_plan = True
                        self._interrupt_pending_until_new_plan = False
                        self.entered_awaiting_replan_response = False
                        current_progress = dict(self.latest_benchmark_progress or {})
                        completed_now = [str(v).upper() for v in list(current_progress.get("completed") or [])]
                        active_now = [str(v).upper() for v in list(self.active_objective_set.get("active_checkpoint_ids") or [])]
                        remaining_now = [cid for cid in active_now if cid not in set(completed_now)]
                        self.task_run_logger.append_planning_trace(
                            trace={
                                "planning_stage": "replan",
                                "plan_source": "committed_replan",
                                "llm_call_purpose": "heartbeat_commit",
                                "parsed_plan": self.current_plan,
                                "selected_baseline_id": self.selected_pipeline_id,
                                "scene_id": self.baseline_scene_id,
                                "current_target_checkpoint": current_progress.get("current_target"),
                                "true_completed_checkpoints": completed_now,
                                "true_remaining_checkpoints": remaining_now,
                                "completion_state_source": "benchmark_progress/dwell_tracker",
                            }
                        )
                        llm_called = False
                        final_plan_source = "agent_heartbeat"
                    else:
                        self.current_plan = self.planner.plan(
                            task_description=task_description,
                            scene_description=scene_description,
                            location_info=location_info,
                            execution_history=self._build_execution_history_for_llm(),
                            safety_context=safety_context,
                            previous_plan=previous_plan,
                            planning_stage=planning_stage,
                        )
                        llm_called = True
                        final_plan_source = f"llm_{planning_stage}"
                        if str(planning_stage).strip().lower() == "replan" and self._interrupt_pending_until_new_plan:
                            self.execution_resumed_from_new_plan = True
                            self.entered_awaiting_replan_response = False
                            self._interrupt_pending_until_new_plan = False
                        self.task_run_logger.append_planning_trace(
                            trace={
                                **self.planner.get_last_plan_trace(),
                                "plan_source": final_plan_source,
                                "llm_call_purpose": planning_stage,
                                "parsed_plan": self.current_plan,
                                "selected_baseline_id": pipeline.id,
                                "scene_id": self.baseline_scene_id,
                            }
                        )
                self.current_plan = self._sanitize_minispec_plan(self.current_plan)
                if replan_attempts > 0 and llm_called:
                    self._record_replan_response(
                        source="typefly_llm_full_replan_response",
                        reason=f"planning_stage={planning_stage}",
                        plan_text=self.current_plan,
                    )
                if replan_attempts == 0 and self._mission_original_plan is None:
                    self._mission_original_plan = str(self.current_plan)
                if self._current_active_plan is None:
                    self._current_active_plan = str(self.current_plan)
                if replan_attempts > 0:
                    self._current_active_plan = str(self.current_plan)
                if replan_attempts == 0:
                    print_t(f"[PLAN-INITIAL] llm_response={self.current_plan}")
                else:
                    print_t(f"[FULL-REPLAN] llm_response={self.current_plan}")
                self.latest_safety_context = safety_context
                self.task_run_logger.update_plan_info(self.current_plan, generation_success=True)
                debug_info = {
                    "task_text": task_description,
                    "planner_mode": self.planner_mode,
                    "llm_called": llm_called,
                    "llm_function": "LLMPlanner.plan" if llm_called else "None",
                    "baseline_shortcut_triggered": baseline_shortcut_triggered,
                    "selected_target": None if baseline is None else baseline.get("target_task_point"),
                    "path_clear": None if baseline is None else baseline.get("path_clear"),
                    "blocking_entity": None if baseline is None else baseline.get("blocking_entity"),
                    "final_plan_source": final_plan_source,
                    "final_plan_text": self.current_plan,
                }
                if hasattr(self.task_run_logger, "update_planner_info"):
                    self.task_run_logger.update_planner_info(debug_info)
                print_debug(
                    "[TASK-PLANNER-FLOW] "
                    + ", ".join(f"{k}={v}" for k, v in debug_info.items())
                )

                self.append_message(f'[Plan]: \\\\')
                print_t(f"[PLAN-DISPLAY] {self.current_plan}")
                self.execution_time = time.time()
                self.execution_mode = "Executing"
                self.auto_replan_protection_remaining = int(AUTO_REPLAN_PROTECTION_STATEMENTS)
                self.auto_replan_armed = True
                print_t(
                    "[REPLAN_DEBUG] "
                    f"protection_window_active remaining_statements={self.auto_replan_protection_remaining} "
                    f"auto_replan_armed={self.auto_replan_armed}"
                )
                print_t(f"[PLAN-QUEUE] {self.current_plan}")
                allow_auto_interrupt = selected_framework != MODE_TYPEFLY_ONESHOT
                ret_val = self.execute_minispec(self.current_plan, allow_auto_interrupt=allow_auto_interrupt)
                execution_success = True
                task_completed = False
                if isinstance(ret_val, tuple) and len(ret_val) >= 2:
                    execution_success = bool(ret_val[0] is not False)
                if hasattr(ret_val, "replan") and bool(ret_val.replan):
                    replan_value = str(getattr(ret_val, "value", "") or "")
                    if selected_framework == MODE_TYPEFLY_ONESHOT:
                        print_debug(
                            f"[REPLAN_DEBUG] ret_val.replan ignored in oneshot mode value={replan_value}",
                            env_var="TYPEFLY_VERBOSE_DEBUG",
                        )
                    else:
                        if replan_attempts >= max_replan_attempts:
                            print_t(f"[REPLAN-COUNT] current={replan_attempts} limit={max_replan_attempts}")
                            self.append_message(
                                f"[LOG] Replan requested but limit reached ({replan_attempts}/{max_replan_attempts})."
                            )
                        else:
                            replan_attempts += 1
                            self._replan_attempts = replan_attempts
                            self.execution_mode = "AwaitingReplanResponse"
                            self.interrupted_for_replan = True
                            self.entered_awaiting_replan_response = True
                            self._interrupt_pending_until_new_plan = True
                            self.append_message(
                                "[TYPEFLY-INTERRUPT] statement requested replan, aborting current execution: "
                                f"{replan_value if replan_value else 'no detail'}"
                            )
                            print_t(
                                "[TYPEFLY-INTERRUPT] statement requested replan, aborting current execution: "
                                f"{replan_value if replan_value else 'no detail'}"
                            )
                            continue
                completed_set = set(str(v).upper() for v in (self.latest_benchmark_progress.get("completed") or []))
                planned_sequence = list(self._benchmark_plan_checkpoint_sequence)
                missing_from_plan = [cid for cid in planned_sequence if cid not in completed_set]
                active_ids = set(str(v).upper() for v in self.active_objective_set.get("active_checkpoint_ids", []))
                zone_completed = bool(active_ids) and all(cid in completed_set for cid in active_ids)
                plan_completed = all(cid in completed_set for cid in planned_sequence) if planned_sequence else True
                print_debug(
                    "[TASK-END-CHECK] "
                    f"plan_sequence={planned_sequence} completed={sorted(completed_set)} "
                    f"missing_from_plan={missing_from_plan} zone_completed={zone_completed} "
                    f"ret_replan={bool(getattr(ret_val, 'replan', False))}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
                if (not plan_completed) or (active_ids and (not zone_completed)):
                    self.append_message(
                        f"[LOG] Completion mismatch detected (auto-replan disabled), "
                        f"missing={missing_from_plan if missing_from_plan else '[]'}"
                    )
                remaining_active = sorted(cid for cid in active_ids if cid not in completed_set)
                task_completed = (len(remaining_active) == 0)
                termination_reason = (
                    "mission_completed_all_active_checkpoints"
                    if task_completed
                    else "queue_exhausted_with_unfinished_checkpoints"
                )
                queue_exhausted_with_unfinished = bool((not task_completed) and bool(remaining_active))
                self.task_run_logger.update_execution_info(
                    execution_success=execution_success,
                    task_completed=task_completed,
                    mission_success=task_completed,
                    termination_reason=termination_reason,
                    queue_exhausted_with_unfinished=queue_exhausted_with_unfinished,
                    ended_due_to_replan_interrupt=False,
                    true_completed_checkpoints=sorted(completed_set),
                    true_remaining_checkpoints=remaining_active,
                    current_target_checkpoint=self.latest_benchmark_progress.get("current_target"),
                    checkpoint_status_snapshot=dict(self.latest_benchmark_progress),
                    completion_state_source="benchmark_progress/dwell_tracker",
                    completion_time_mission_sec=((time.time() - float(self.mission_start_ts)) if (task_completed and self.mission_start_ts is not None) else None),
                    mission_start_ts=self.mission_start_ts,
                    mission_end_ts=(time.time() if task_completed else None),
                    final_summary_source="final_mission_summary",
                    final_runtime_snapshot_ts=None,
                    final_status_source="benchmark_progress/dwell_tracker",
                    interrupted_for_replan=bool(self.interrupted_for_replan),
                    entered_awaiting_replan_response=bool(self.entered_awaiting_replan_response),
                    execution_resumed_from_new_plan=bool(self.execution_resumed_from_new_plan),
                    next_statement_executed_after_interrupt=bool(self.next_statement_executed_after_interrupt),
                )
                # TypeFly post-check auto repair removed by design:
                # completion mismatch is logged only; no queue-finished auto replan.
            except Exception as e:
                self.execution_mode = "Yielding"
                error_message = f"[C] Error: {e}"
                print_t(error_message)
                self.append_message(error_message)
                self.task_run_logger.update_plan_info(self.current_plan or "", generation_success=bool(self.current_plan))
                self.task_run_logger.update_execution_info(
                    execution_success=False,
                    failure_reason=str(e),
                    task_completed=False,
                )
                self.task_run_logger.end_run(run_status="exception", failure_reason=str(e))
                monitor_stop.set()
                if monitor_thread is not None:
                    monitor_thread.join(timeout=1.0)
                self.append_message(f'\n[Task ended]')
                self.append_message('end')
                self.current_plan = None
                self.execution_history = None
                self.execution_mode = "Waiting"
                return

            break
        final_snapshot = self.get_live_ui_snapshot()
        completion_state = self._derive_completion_state_from_snapshot(final_snapshot)
        completed_set = set(completion_state.get("true_completed_checkpoints", []))
        remaining_active = list(completion_state.get("true_remaining_checkpoints", []))
        mission_success = bool(completion_state.get("mission_success"))
        run_status = "completed" if mission_success else "incomplete"
        termination_reason = (
            "mission_completed_all_active_checkpoints"
            if mission_success
            else "queue_exhausted_with_unfinished_checkpoints"
        )
        queue_exhausted_with_unfinished = bool((not mission_success) and bool(remaining_active))
        if not mission_success:
            self.append_message(
                f"[LOG] Mission incomplete: queue exhausted with unfinished checkpoints={remaining_active}"
            )
        self.execution_mode = ("Completed" if mission_success else "Incomplete")
        final_snapshot["execution_mode"] = self.execution_mode
        final_summary = self._build_final_mission_summary(
            final_snapshot,
            run_status=run_status,
            mission_success=mission_success,
            termination_reason=termination_reason,
        )
        final_snapshot["final_mission_summary"] = dict(final_summary)
        self.task_run_logger.consume_runtime_snapshot(final_snapshot)
        self.task_run_logger.update_execution_info(
            execution_success=True,
            task_completed=mission_success,
            mission_success=mission_success,
            termination_reason=termination_reason,
            queue_exhausted_with_unfinished=queue_exhausted_with_unfinished,
            ended_due_to_replan_interrupt=False,
            true_completed_checkpoints=sorted(completed_set),
            true_remaining_checkpoints=remaining_active,
            current_target_checkpoint=completion_state.get("current_target_checkpoint"),
            checkpoint_status_snapshot=dict(final_snapshot.get("benchmark_progress") or {}),
            completion_state_source="benchmark_progress/dwell_tracker",
            completion_time_mission_sec=final_summary.get("completion_time_mission_sec"),
            mission_start_ts=final_summary.get("mission_start_ts"),
            mission_end_ts=final_summary.get("mission_end_ts"),
            final_summary_source=final_summary.get("final_summary_source", "final_runtime_snapshot"),
            final_runtime_snapshot_ts=str(final_summary.get("final_runtime_snapshot_ts")),
            final_completion_snapshot_ts=str(final_summary.get("final_completion_snapshot_ts")),
            final_completion_snapshot_source=final_summary.get("final_completion_snapshot_source", "benchmark_progress/dwell_tracker"),
            final_execution_mode=final_summary.get("final_execution_mode"),
            final_run_status=final_summary.get("final_run_status"),
            final_mission_success=final_summary.get("final_mission_success"),
            final_termination_reason=final_summary.get("final_termination_reason"),
            final_true_completed_checkpoints=final_summary.get("final_true_completed_checkpoints"),
            final_true_remaining_checkpoints=final_summary.get("final_true_remaining_checkpoints"),
            final_status_source=final_summary.get("final_status_source", "benchmark_progress/dwell_tracker"),
            interrupted_for_replan=bool(final_summary.get("interrupted_for_replan")),
            entered_awaiting_replan_response=bool(final_summary.get("entered_awaiting_replan_response")),
            execution_resumed_from_new_plan=bool(final_summary.get("execution_resumed_from_new_plan")),
            next_statement_executed_after_interrupt=bool(final_summary.get("next_statement_executed_after_interrupt")),
        )
        self.task_run_logger.end_run(run_status=run_status)
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        self.append_message(f'\n[Task ended]')
        self.append_message('end')
        self.current_plan = None
        self.execution_history = None
        self.execution_mode = "Waiting"
        self.framework_mode = MODE_TYPEFLY_ONESHOT

    def get_active_scenario_name(self) -> str:
        return self.scenario_manager.selected_name()

    def set_archive_enabled(self, enabled: bool):
        self.archive_enabled = bool(enabled)

    def set_selected_pipeline(self, pipeline_id: str) -> str:
        self.selected_pipeline_id = normalize_pipeline_id(pipeline_id)
        config = get_pipeline_config(self.selected_pipeline_id)
        self.replan_limit = int(config.replan_cap)
        if config.trigger_type == "event_predicted_collision_probability":
            threshold = float(config.trigger_params.get("predicted_collision_threshold", PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD))
            self.predicted_collision_replan_threshold = threshold
            self.predicted_collision_rearm_threshold = max(0.0, threshold - 0.05)
            self.predicted_collision_replan_strictly_greater = bool(config.trigger_params.get("strictly_greater", True))
            self.heartbeat_interval_seconds = float(AGENT_HEARTBEAT_INTERVAL_SECONDS)
        else:
            self.predicted_collision_replan_threshold = float(PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD)
            self.predicted_collision_rearm_threshold = float(PREDICTED_COLLISION_PROBABILITY_REARM_THRESHOLD)
            self.predicted_collision_replan_strictly_greater = False
            self.heartbeat_interval_seconds = float(config.trigger_params.get("heartbeat_seconds", AGENT_HEARTBEAT_INTERVAL_SECONDS))
        self.planner.set_runtime_prompt_example_variant(
            prompt_variant=config.prompt_variant,
            example_variant=config.example_variant,
            use_output_example=bool(config.use_output_example),
        )
        return self.selected_pipeline_id

    def get_selected_pipeline_config(self):
        return get_pipeline_config(self.selected_pipeline_id)

    def get_baseline_scene(self) -> BaselineScene:
        return BASELINE_SCENES[self.baseline_scene_id]

    def set_baseline_scene(self, scene_id: str) -> str:
        self.baseline_scene_id = normalize_baseline_scene_id(scene_id)
        return self.baseline_scene_id

    def apply_baseline_scene(self):
        scene = self.get_baseline_scene()
        print_t(worker_mode_summary_log())
        provider = getattr(self, "state_provider", None)
        drone = getattr(self, "drone", None)
        repositioned = False
        if drone is not None and hasattr(drone, "reposition_for_scenario"):
            repositioned = bool(drone.reposition_for_scenario(scene))
        if provider is not None and hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason=f"baseline_scene:{scene.id}")
        if provider is not None and hasattr(provider, "set_user_position"):
            ux, uy, uz = scene.user_position
            provider.set_user_position(float(ux), float(uy), float(uz), source=f"baseline_scene:{scene.id}")
        self.baseline_scene_state = {
            "scene_id": scene.id,
            "drone_initial_pose": tuple(float(v) for v in scene.drone_initial_pose),
            "drone_initial_yaw_rad": float(scene.drone_initial_yaw_rad),
            "user_position": tuple(float(v) for v in scene.user_position),
            "repositioned": repositioned,
            "notes": scene.notes,
            "captured_at": time.time(),
        }
        self.user_heading_yaw_rad = float(scene.user_initial_yaw_rad)
        self._reset_manual_worker_poses_from_scene(scene)
        return self.baseline_scene_state

    def get_baseline_scene_state(self):
        return self.baseline_scene_state

    def get_user_heading_yaw(self) -> float:
        return float(self.user_heading_yaw_rad)

    def set_user_heading_yaw(self, yaw_rad: float):
        self.user_heading_yaw_rad = float(yaw_rad)
        return self.user_heading_yaw_rad

    def turn_user_heading(self, delta_deg: float):
        self.user_heading_yaw_rad = float(self.user_heading_yaw_rad + math.radians(float(delta_deg)))
        return self.user_heading_yaw_rad

    def set_manual_worker_selection(self, worker_id: str) -> str:
        candidate = str(worker_id or "").strip()
        if candidate not in {"worker_1", "worker_2", "worker_3"}:
            candidate = "worker_1"
        self.manual_worker_selection_id = candidate
        return self.manual_worker_selection_id

    def _reset_manual_worker_poses_from_scene(self, scene: BaselineScene):
        self.manual_worker_poses = {}
        self.manual_worker_localization_state = {}
        for obstacle in scene.obstacles:
            worker_id = str(obstacle.id)
            if worker_id not in {"worker_1", "worker_2", "worker_3"}:
                continue
            self.manual_worker_poses[worker_id] = {
                "x": float(obstacle.gt_x),
                "y": float(obstacle.gt_y),
                "z": 0.0,
                "yaw_rad": 0.0,
            }

    def move_selected_worker_relative(self, local_forward: float, local_right: float, step_m: float):
        if self.get_baseline_scene().id != "SCENE_MANUAL_WORKER_CONTROL":
            return None
        worker_id = str(self.manual_worker_selection_id or "worker_1")
        pose = self.manual_worker_poses.get(worker_id)
        if pose is None:
            return None
        step = max(0.0, float(step_m))
        yaw = float(pose.get("yaw_rad", 0.0))
        forward = float(local_forward) * step
        right = float(local_right) * step
        dx = math.cos(yaw) * forward + math.sin(yaw) * right
        dy = math.sin(yaw) * forward - math.cos(yaw) * right
        pose["x"] = float(pose["x"] + dx)
        pose["y"] = float(pose["y"] + dy)
        return {
            "worker_id": worker_id,
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw_deg": float(math.degrees(pose["yaw_rad"])),
        }

    def turn_selected_worker(self, delta_deg: float):
        if self.get_baseline_scene().id != "SCENE_MANUAL_WORKER_CONTROL":
            return None
        worker_id = str(self.manual_worker_selection_id or "worker_1")
        pose = self.manual_worker_poses.get(worker_id)
        if pose is None:
            return None
        pose["yaw_rad"] = float(pose.get("yaw_rad", 0.0) + math.radians(float(delta_deg)))
        return {
            "worker_id": worker_id,
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw_deg": float(math.degrees(pose["yaw_rad"])),
        }

    def set_active_scenario(self, scenario_name: str):
        scenario = self.scenario_manager.select(scenario_name)
        return scenario.name

    def apply_selected_scenario(self):
        report = self.scenario_manager.apply_to_runtime(self)
        self.latest_scenario_report = report
        self.initial_scenario_state = {
            "selected_mode": report.selected_mode,
            "target_drone_gt": report.target_drone_position_3d,
            "actual_drone_gt": report.actual_drone_gt_position_3d,
            "safety_score": report.measured_initial_safety_score,
            "current_collision_probability": report.measured_initial_collision_probability,
            "envelope_gap_m": report.measured_initial_envelope_gap_m,
            "uncertainty_scale_m": report.measured_initial_uncertainty_scale_m,
            "repositioned": report.repositioned,
            "calibration_iterations": report.calibration_iterations,
            "captured_at": time.time(),
        }
        # Force one immediate safety refresh after scenario apply.
        try:
            now = time.time()
            if hasattr(self.state_provider, "flush_due_packets"):
                self.state_provider.flush_due_packets(now=now)
        except Exception:
            pass
        return report

    def get_initial_scenario_state(self):
        return self.initial_scenario_state

    def get_scenario_projection(self):
        baseline_uncertainty = 0.85
        snapshot = self.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        if safety_context is not None:
            baseline_uncertainty = float(safety_context.uncertainty_scale_m)
        return self.scenario_manager.projected_assessment(baseline_uncertainty_scale_m=baseline_uncertainty)

    def get_scenario_runtime_status(self):
        report = self.latest_scenario_report
        snapshot = self.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        return {
            "selected_mode": None if report is None else report.selected_mode,
            "target_drone_gt": None if report is None else report.target_drone_position_3d,
            "actual_drone_gt": snapshot.get("drone_gt") if snapshot else None,
            "predicted_collision_probability": None if safety_context is None else float(safety_context.predicted_collision_probability),
            "safety_score": None if safety_context is None else float(safety_context.safety_score),
            "envelope_gap_m": None if safety_context is None else float(safety_context.envelope_gap_m),
            "uncertainty_scale_m": None if safety_context is None else float(safety_context.uncertainty_scale_m),
        }

    def move_user_world(self, dx: float, dy: float, dz: float = 0.0):
        provider = getattr(self, "state_provider", None)
        if provider is None or not hasattr(provider, "get_user_position") or not hasattr(provider, "set_user_position"):
            return None
        if hasattr(provider, "lock_user_position"):
            provider.lock_user_position(True, reason="ui_manual_user_move")
        ux, uy, uz = provider.get_user_position()
        nx = float(ux + dx)
        ny = float(uy + dy)
        nz = float(uz + dz)
        provider.set_user_position(nx, ny, nz, source="ui_manual_user_move")
        if hasattr(provider, "flush_due_packets"):
            provider.flush_due_packets(now=time.time())
        return (nx, ny, nz)

    def _debug_log_safety_context(self, safety_context):
        if safety_context is None:
            print_debug("[SAFETY] unavailable")
            return
        print_debug(
            "[SAFETY]\n"
            f"  distance_xy={safety_context.drone_to_user_distance_xy:.3f}\n"
            f"  gap={safety_context.envelope_gap_m:.3f}\n"
            f"  uncertainty={safety_context.uncertainty_scale_m:.3f}\n"
            f"  overlap={safety_context.envelopes_overlap}\n"
            f"  score={safety_context.safety_score:.3f}\n"
            f"  current_instantaneous_collision_probability={safety_context.current_collision_probability:.6f}\n"
            f"  predicted_collision_probability={safety_context.predicted_collision_probability:.6f}\n"
            f"  standoff={safety_context.preferred_standoff_m:.3f}\n"
            f"  dominant_threat={safety_context.dominant_threat_type}:{safety_context.dominant_threat_id}\n"
            f"  dominant_gap={safety_context.dominant_gap_m:.3f}\n"
            f"  dominant_uncertainty={safety_context.dominant_uncertainty_scale_m:.3f}\n"
            f"  reasons={safety_context.reason_tags}"
        )

    def _simulate_obstacle_returns(self, obstacle_states, now: float):
        _ = now
        states = list(obstacle_states or [])
        if self.get_baseline_scene().id != "SCENE_MANUAL_WORKER_CONTROL":
            return states
        updated_states = []
        for state in states:
            worker_id = str(state.id)
            pose = self.manual_worker_poses.get(worker_id)
            if pose is None:
                updated_states.append(state)
                continue
            packet = state.localization_packet.copy()
            gt_x = float(pose["x"])
            gt_y = float(pose["y"])
            packet.gt_position_3d[0] = gt_x
            packet.gt_position_3d[1] = gt_y
            # Simulate localization estimate dynamics (instead of snapping est to gt+bias),
            # so MANUAL_WORKER_CONTROL keeps a realistic est!=gt behavior.
            loc_state = self.manual_worker_localization_state.get(worker_id)
            bias_x = float(packet.b_xy[0])
            bias_y = float(packet.b_xy[1])
            sigma_x = float(max(np.sqrt(max(float(packet.P_xy[0][0]), 1e-8)), 0.01))
            sigma_y = float(max(np.sqrt(max(float(packet.P_xy[1][1]), 1e-8)), 0.01))
            noise_rng = np.random.default_rng((hash(worker_id) ^ int(time.time() * 10.0)) & 0xFFFFFFFF)
            measured_x = float(gt_x + bias_x + noise_rng.normal(0.0, sigma_x))
            measured_y = float(gt_y + bias_y + noise_rng.normal(0.0, sigma_y))
            est_x = measured_x
            est_y = measured_y
            self.manual_worker_localization_state[worker_id] = {
                "est_x": est_x,
                "est_y": est_y,
                "measured_x": measured_x,
                "measured_y": measured_y,
            }
            packet.estimated_position_3d[0] = est_x
            packet.estimated_position_3d[1] = est_y
            packet.localization_error_vector_3d[0] = float(packet.estimated_position_3d[0] - gt_x)
            packet.localization_error_vector_3d[1] = float(packet.estimated_position_3d[1] - gt_y)
            envelope = build_safety_envelope(packet)
            updated_states.append(
                replace(
                    state,
                    gt_xy=(gt_x, gt_y),
                    est_xy=(float(packet.estimated_position_3d[0]), float(packet.estimated_position_3d[1])),
                    matrix_xy=((float(packet.M_xy[0][0]), float(packet.M_xy[0][1])), (float(packet.M_xy[1][0]), float(packet.M_xy[1][1]))),
                    localization_packet=packet,
                    envelope=envelope,
                )
            )
        return updated_states

    def _build_collision_worker_packets_from_obstacles(self, obstacle_states):
        obstacle_map = {str(obs.id): obs for obs in (obstacle_states or [])}
        packets = []
        for worker_id in COLLISION_RISK_WORKER_IDS:
            obs = obstacle_map.get(worker_id)
            if obs is not None:
                packets.append((worker_id, obs.localization_packet))
        return packets

    def _build_uav_prediction_intent(self, drone_xy, drone_yaw_rad: float) -> dict:
        intent = {
            "mode": "velocity_fallback",
            "target_xy": None,
            "speed_mps": float(UAV_GC_NOMINAL_SPEED_MPS),
            "body_velocity_xy": None,
        }
        if drone_xy is None:
            return intent
        progress = dict(self.latest_benchmark_progress or {})
        current_target = progress.get("current_target")
        if current_target and str(current_target).upper() in BENCHMARK_CHECKPOINTS_BY_ID:
            cp = BENCHMARK_CHECKPOINTS_BY_ID[str(current_target).upper()]
            intent["mode"] = "gc_target"
            intent["target_xy"] = (float(cp.x), float(cp.y))
            intent["speed_mps"] = float(UAV_GC_NOMINAL_SPEED_MPS)
            return intent
        plan_text = str(self.current_plan or "").strip().lower()
        if plan_text.startswith("mf("):
            body = np.array([1.0, 0.0], dtype=float)
        elif plan_text.startswith("mb("):
            body = np.array([-1.0, 0.0], dtype=float)
        elif plan_text.startswith("ml("):
            body = np.array([0.0, 1.0], dtype=float)
        elif plan_text.startswith("mr("):
            body = np.array([0.0, -1.0], dtype=float)
        else:
            body = np.array([0.0, 0.0], dtype=float)
        if float(np.linalg.norm(body)) <= 1e-6:
            return intent
        c = math.cos(float(drone_yaw_rad))
        s = math.sin(float(drone_yaw_rad))
        rot = np.array([[c, -s], [s, c]], dtype=float)
        intent["mode"] = "body_action"
        intent["body_velocity_xy"] = tuple(((rot @ body) * float(UAV_BODY_NOMINAL_SPEED_MPS)).tolist())
        intent["speed_mps"] = float(UAV_BODY_NOMINAL_SPEED_MPS)
        return intent

    def _build_uav_prediction_velocity_hint(self, drone_xy, drone_yaw_rad: float) -> np.ndarray:
        intent = self._build_uav_prediction_intent(drone_xy=drone_xy, drone_yaw_rad=drone_yaw_rad)
        mode = str(intent.get("mode", "velocity_fallback"))
        if mode == "gc_target":
            target_xy = intent.get("target_xy")
            if target_xy is None:
                return np.zeros(2, dtype=float)
            delta = np.array([float(target_xy[0]) - float(drone_xy[0]), float(target_xy[1]) - float(drone_xy[1])], dtype=float)
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-6:
                return np.zeros(2, dtype=float)
            return (delta / norm) * float(intent.get("speed_mps", UAV_GC_NOMINAL_SPEED_MPS))
        if mode == "body_action":
            body_velocity_xy = intent.get("body_velocity_xy")
            if body_velocity_xy is None:
                return np.zeros(2, dtype=float)
            return np.asarray(body_velocity_xy, dtype=float).reshape(2)
        return np.zeros(2, dtype=float)

    def _build_dominant_threat_context(
        self,
        safety_state,
        obstacle_states,
        scene,
        now: float,
        candidate_targets_summary=None,
        candidate_path_summaries=None,
    ):
        if safety_state is None:
            return None

        worker_packets = self._build_collision_worker_packets_from_obstacles(obstacle_states)

        assessed_context = self.safety_assessor.build_from_packets(
            drone_packet=safety_state.drone_packet,
            worker_packets=worker_packets,
            now=now,
            safety_state=safety_state,
            uav_velocity_hint_xy=self._build_uav_prediction_velocity_hint(
                drone_xy=np.asarray(safety_state.drone_packet.estimated_position_3d[:2], dtype=float)
                - np.asarray(safety_state.drone_packet.b_xy[:2], dtype=float),
                drone_yaw_rad=self._get_drone_yaw_rad(),
            ),
            uav_prediction_intent=self._build_uav_prediction_intent(
                drone_xy=np.asarray(safety_state.drone_packet.estimated_position_3d[:2], dtype=float)
                - np.asarray(safety_state.drone_packet.b_xy[:2], dtype=float),
                drone_yaw_rad=self._get_drone_yaw_rad(),
            ),
        )

        task_points_summary = []
        for point in scene.task_points if scene is not None else []:
            task_points_summary.append({"id": point.id, "x": float(point.x), "y": float(point.y), "z": float(point.z)})
        obstacles_summary = []
        for obs in obstacle_states or []:
            obs_packet = obs.localization_packet
            obstacles_summary.append(
                {
                    "id": obs.id,
                    "est_x": float(obs_packet.estimated_position_3d[0]),
                    "est_y": float(obs_packet.estimated_position_3d[1]),
                    "major_axis_m": float(obs.envelope.major_axis_radius),
                    "minor_axis_m": float(obs.envelope.minor_axis_radius),
                    "orientation_deg": float(obs.envelope.orientation_deg),
                }
            )
        assessed_context.task_points_summary = task_points_summary
        assessed_context.obstacles_summary = obstacles_summary
        assessed_context.path_summary = None
        assessed_context.candidate_targets_summary = candidate_targets_summary or []
        assessed_context.candidate_path_summaries = candidate_path_summaries or []
        return assessed_context

    def _build_final_mission_summary(self, snapshot: dict, *, run_status: str, mission_success: bool, termination_reason: str) -> dict:
        mission_end_ts = time.time()
        self.mission_end_ts = mission_end_ts if bool(mission_success) else None
        completion_state = self._derive_completion_state_from_snapshot(snapshot)
        summary = {
            "run_status": str(run_status),
            "mission_success": bool(mission_success),
            "termination_reason": str(termination_reason or ""),
            "mission_start_ts": self.mission_start_ts,
            "mission_end_ts": self.mission_end_ts,
            "completion_time_mission_sec": (
                None
                if (not bool(mission_success) or self.mission_start_ts is None or self.mission_end_ts is None)
                else round(float(self.mission_end_ts) - float(self.mission_start_ts), 3)
            ),
            "collision_count": int((snapshot or {}).get("collision_count", 0) or 0),
            "near_miss_count": int((snapshot or {}).get("near_miss_count", 0) or 0),
            "replan_count": int(getattr(self, "_replan_attempts", 0)),
            "queue_exhausted_with_unfinished": bool((not bool(mission_success))),
            "final_summary_source": "final_runtime_snapshot",
            "final_runtime_snapshot_ts": float(mission_end_ts),
            "final_completion_snapshot_ts": float(mission_end_ts),
            "final_completion_snapshot_source": "benchmark_progress/dwell_tracker",
            "final_execution_mode": ("Completed" if bool(mission_success) else "Incomplete"),
            "final_run_status": str(run_status),
            "final_mission_success": bool(mission_success),
            "final_termination_reason": str(termination_reason or ""),
            "final_true_completed_checkpoints": list(completion_state.get("true_completed_checkpoints", [])),
            "final_true_remaining_checkpoints": list(completion_state.get("true_remaining_checkpoints", [])),
            "final_status_source": "benchmark_progress/dwell_tracker",
            "interrupted_for_replan": bool(self.interrupted_for_replan),
            "entered_awaiting_replan_response": bool(self.entered_awaiting_replan_response),
            "execution_resumed_from_new_plan": bool(self.execution_resumed_from_new_plan),
            "next_statement_executed_after_interrupt": bool(self.next_statement_executed_after_interrupt),
        }
        self.final_mission_summary = dict(summary)
        return summary

    def _derive_completion_state_from_snapshot(self, snapshot: dict) -> dict:
        progress = dict((snapshot or {}).get("benchmark_progress") or {})
        active_ids = [
            str(v).upper()
            for v in list(((snapshot or {}).get("active_objective_set") or {}).get("active_checkpoint_ids") or [])
        ]
        completed = [str(v).upper() for v in list(progress.get("completed") or [])]
        completed_in_scope = [cid for cid in completed if (not active_ids) or (cid in set(active_ids))]
        remaining = [cid for cid in active_ids if cid not in set(completed_in_scope)]
        return {
            "active_checkpoint_ids": list(active_ids),
            "true_completed_checkpoints": list(completed_in_scope),
            "true_remaining_checkpoints": list(remaining),
            "current_target_checkpoint": progress.get("current_target"),
            "mission_success": (len(remaining) == 0),
        }

    def get_live_ui_snapshot(self):
        provider = getattr(self, "state_provider", None)
        if provider is None:
            return {}

        def _as_position_tuple(value):
            if value is None:
                return None
            return tuple(float(v) for v in value)

        def _call_position(source, attr):
            fn = getattr(source, attr, None)
            if callable(fn):
                value = fn()
                if value is not None:
                    return _as_position_tuple(value)
            return None

        now = time.time()
        if hasattr(provider, "flush_due_packets"):
            provider.flush_due_packets(now=now)
        safety_state = provider.get_latest_gcs_safety_state(now=now) if hasattr(provider, "get_latest_gcs_safety_state") else None
        safety_context = None
        if safety_state is not None:
            safety_context = self.safety_assessor.build_from_safety_state(safety_state, now=now)
        elif hasattr(provider, "get_latest_safety_context"):
            safety_context = provider.get_latest_safety_context(now=now)
        else:
            safety_context = self.latest_safety_context

        drone_gt = (
            _as_position_tuple(getattr(getattr(provider, "_cache", None), "position", None))
            or _call_position(self.drone, "get_ground_truth_drone_position")
            or _call_position(provider, "get_ground_truth_drone_position")
            or _call_position(self.drone, "get_drone_position")
            or _call_position(provider, "get_drone_position")
            or (0.0, 0.0, 0.0)
        )
        drone_est = _call_position(self.drone, "get_estimated_drone_position")
        if drone_est is None:
            drone_packet = provider.get_latest_received_drone_packet() if hasattr(provider, "get_latest_received_drone_packet") else None
            drone_est = None if drone_packet is None else _as_position_tuple(drone_packet.estimated_position_3d)
        else:
            drone_packet = provider.get_latest_received_drone_packet() if hasattr(provider, "get_latest_received_drone_packet") else None

        baseline_state = self.baseline_scene_state or {}
        scene_start_ts = float(baseline_state.get("captured_at", now))
        elapsed_scene_s = max(0.0, now - scene_start_ts)
        obstacle_states_generated = compute_obstacle_envelope_states(self.get_baseline_scene(), now_s=elapsed_scene_s)
        obstacle_states = self._simulate_obstacle_returns(obstacle_states_generated, now=now)
        if safety_state is not None:
            worker_packets = self._build_collision_worker_packets_from_obstacles(obstacle_states)
            uav_hint = self._build_uav_prediction_velocity_hint(
                drone_xy=np.asarray(safety_state.drone_packet.estimated_position_3d[:2], dtype=float)
                - np.asarray(safety_state.drone_packet.b_xy[:2], dtype=float),
                drone_yaw_rad=self._get_drone_yaw_rad(),
            )
            safety_context = self.safety_assessor.build_from_packets(
                drone_packet=safety_state.drone_packet,
                worker_packets=worker_packets,
                now=now,
                safety_state=safety_state,
                uav_velocity_hint_xy=uav_hint,
                uav_prediction_intent=self._build_uav_prediction_intent(
                    drone_xy=np.asarray(safety_state.drone_packet.estimated_position_3d[:2], dtype=float)
                    - np.asarray(safety_state.drone_packet.b_xy[:2], dtype=float),
                    drone_yaw_rad=self._get_drone_yaw_rad(),
                ),
            )
        candidate_targets = []
        for point in self.get_baseline_scene().task_points:
            candidate_targets.append({"id": point.id, "x": float(point.x), "y": float(point.y), "z": float(point.z)})
        candidate_path_summaries = []
        for target in candidate_targets:
            path = self._compute_path_eval_for_target(
                self.get_baseline_scene(),
                drone_est or drone_gt,
                None,
                target_xy=(target["x"], target["y"]),
                obstacle_envelopes=obstacle_states,
                user_envelope=None,
            )
            candidate_path_summaries.append(
                {
                    "target_id": target["id"],
                    "path_clear": bool(path.path_clear),
                    "blocking_entity": str(path.blocking_entity),
                    "corridor_min_gap": float(path.corridor_min_gap),
                }
            )
        dominant_safety_context = self._build_dominant_threat_context(
            safety_state=safety_state,
            obstacle_states=obstacle_states,
            scene=self.get_baseline_scene(),
            now=now,
            candidate_targets_summary=candidate_targets,
            candidate_path_summaries=candidate_path_summaries,
        )
        if dominant_safety_context is not None:
            dominant_safety_context.path_summary = None
            safety_context = dominant_safety_context
        elif drone_packet is not None:
            # Fallback: when provider-level safety_state is unavailable, still
            # compute collision probability with UAV + canonical worker localization packets.
            worker_packets = self._build_collision_worker_packets_from_obstacles(obstacle_states)
            if worker_packets:
                safety_context = self.safety_assessor.build_from_packets(
                    drone_packet=drone_packet,
                    worker_packets=worker_packets,
                    now=now,
                    safety_state=None,
                    uav_velocity_hint_xy=self._build_uav_prediction_velocity_hint(
                        drone_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float)
                        - np.asarray(drone_packet.b_xy[:2], dtype=float),
                        drone_yaw_rad=self._get_drone_yaw_rad(),
                    ),
                    uav_prediction_intent=self._build_uav_prediction_intent(
                        drone_xy=np.asarray(drone_packet.estimated_position_3d[:2], dtype=float)
                        - np.asarray(drone_packet.b_xy[:2], dtype=float),
                        drone_yaw_rad=self._get_drone_yaw_rad(),
                    ),
                )
        drone_bias_xy = None if drone_packet is None else tuple(float(v) for v in drone_packet.b_xy[:2])
        drone_corrected = None if drone_est is None else (
            float(drone_est[0] - (0.0 if drone_bias_xy is None else drone_bias_xy[0])),
            float(drone_est[1] - (0.0 if drone_bias_xy is None else drone_bias_xy[1])),
            float(drone_est[2]),
        )
        self._update_benchmark_progress_from_snapshot({"drone_gt": drone_gt})

        snapshot = {
            "drone_gt": drone_gt,
            "drone_est": drone_est,
            "drone_est_bias_corrected": drone_corrected,
            "drone_est_raw": drone_est,
            "drone_bias_xy": drone_bias_xy,
            "drone_P_xy": (None if drone_packet is None else np.asarray(drone_packet.P_xy, dtype=float).copy()),
            "safety_state": safety_state,
            "safety_context": safety_context,
            "drone_yaw_rad": self._get_drone_yaw_rad(),
            "baseline_scene_id": self.baseline_scene_id,
            "baseline_scene": self.get_baseline_scene(),
            "baseline_scene_state": self.baseline_scene_state,
            "obstacle_envelope_states": obstacle_states,
            "workers": [
                {
                    "id": str(obs.id),
                    "gt_xy": tuple(float(v) for v in obs.gt_xy),
                    "est_xy_raw": (
                        float(obs.localization_packet.estimated_position_3d[0]),
                        float(obs.localization_packet.estimated_position_3d[1]),
                    ),
                    "bias_xy": tuple(float(v) for v in obs.localization_packet.b_xy[:2]),
                    "est_xy_bias_corrected": (
                        float(obs.localization_packet.estimated_position_3d[0] - obs.localization_packet.b_xy[0]),
                        float(obs.localization_packet.estimated_position_3d[1] - obs.localization_packet.b_xy[1]),
                    ),
                    "ui_xy": (
                        tuple(float(v) for v in obs.gt_xy)
                        if self.get_baseline_scene().id == "SCENE_MANUAL_WORKER_CONTROL"
                        else (
                            float(obs.localization_packet.estimated_position_3d[0] - obs.localization_packet.b_xy[0]),
                            float(obs.localization_packet.estimated_position_3d[1] - obs.localization_packet.b_xy[1]),
                        )
                    ),
                    "heading_yaw_rad": float(self.manual_worker_poses.get(str(obs.id), {}).get("yaw_rad", 0.0)),
                    "P_xy": np.asarray(obs.localization_packet.P_xy, dtype=float).copy(),
                }
                for obs in obstacle_states
            ],
            "candidate_targets": candidate_targets,
            "candidate_path_summaries": candidate_path_summaries,
            "baseline_decision": self.latest_baseline_decision,
            "framework_name": str(self.framework_mode),
            "selected_baseline_id": self.selected_pipeline_id,
            "selected_baseline_name": self.get_selected_pipeline_config().name,
            "archive_enabled": bool(self.archive_enabled),
            "mode_name": self.get_active_scenario_name(),
            "execution_mode": self.execution_mode,
            "active_objective_set": dict(self.active_objective_set),
            "benchmark_progress": dict(self.latest_benchmark_progress),
            "checkpoint_order": list(BENCHMARK_CHECKPOINT_ORDER),
            "benchmark_checkpoints": [
                {"id": cp.id, "zone_id": cp.zone_id, "x": float(cp.x), "y": float(cp.y), "radius_m": float(cp.radius_m)}
                for cp in BENCHMARK_CHECKPOINTS
            ],
            "benchmark_zones": [
                {"id": zone.id, "x_range": zone.x_range, "y_range": zone.y_range, "label_xy": zone.label_xy}
                for zone in BENCHMARK_ZONES
            ],
            "original_planned_path": None,
            "updated_path": None,
            "replan_count": int(getattr(self, "_replan_attempts", 0)),
        }
        self._update_proximity_metrics(snapshot)
        if self.final_mission_summary is not None:
            summary = dict(self.final_mission_summary)
            summary["collision_count"] = int(summary.get("collision_count", snapshot.get("collision_count", 0)) or 0)
            summary["near_miss_count"] = int(summary.get("near_miss_count", snapshot.get("near_miss_count", 0)) or 0)
            summary["replan_count"] = int(summary.get("replan_count", getattr(self, "_replan_attempts", 0)) or 0)
            snapshot["final_mission_summary"] = summary
        if safety_state is not None and safety_context is not None:
            consistency_from_gap = bool(float(safety_state.envelope_gap_m) < 0.0)
            print_debug(
                "[UI-SAFETY-SNAPSHOT] "
                f"gap={safety_context.envelope_gap_m:.6f} "
                f"overlap={safety_context.envelopes_overlap} "
                f"overlap_from_gap={consistency_from_gap} "
                f"drone_center={tuple(float(v) for v in safety_state.drone_center_xy)} "
                f"drone_radius={safety_state.drone_radius_along_user_direction:.6f} "
                f"score={safety_context.safety_score:.6f} "
                f"current_p={safety_context.predicted_collision_probability:.6f} "
                f"reason_tags={safety_context.reason_tags}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        self._debug_log_obstacle_envelopes(snapshot.get("obstacle_envelope_states"))
        self._debug_log_localization_pipeline_comparison(snapshot)
        self._debug_log_collision_probability_pipeline(snapshot)
        print_debug(
            "[UI-SNAPSHOT] "
            f"drone_gt={snapshot['drone_gt']} drone_est={snapshot['drone_est']}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return snapshot

    def _update_proximity_metrics(self, snapshot: dict):
        drone_gt = snapshot.get("drone_gt")
        workers = list(snapshot.get("workers") or [])
        events = []
        if drone_gt is None:
            snapshot["near_miss_count"] = int(self.near_miss_count)
            snapshot["collision_count"] = int(self.collision_count)
            snapshot["near_miss_events"] = events
            snapshot["min_uav_worker_distance_m"] = self.min_uav_worker_distance_m
            return
        collision_distance = float(UAV_RADIUS_M + WORKER_RADIUS_M)
        for worker in workers:
            wid = str(worker.get("id"))
            if wid not in self._near_miss_active_by_worker:
                self._near_miss_active_by_worker[wid] = False
                self._collision_active_by_worker[wid] = False
            worker_xy = worker.get("gt_xy")
            if worker_xy is None:
                continue
            d = float(math.hypot(float(drone_gt[0]) - float(worker_xy[0]), float(drone_gt[1]) - float(worker_xy[1])))
            if self.min_uav_worker_distance_m is None or d < self.min_uav_worker_distance_m:
                self.min_uav_worker_distance_m = d
            colliding = bool(d <= collision_distance)
            if colliding and not self._collision_active_by_worker.get(wid, False):
                self.collision_count += 1
                events.append({"worker_id": wid, "event": "collision_enter", "distance_m": d})
            self._collision_active_by_worker[wid] = colliding

            near_active = bool(self._near_miss_active_by_worker.get(wid, False))
            if (not colliding) and (d <= NEAR_MISS_ENTER_DISTANCE_M) and (not near_active):
                self.near_miss_count += 1
                self._near_miss_active_by_worker[wid] = True
                events.append({"worker_id": wid, "event": "near_miss_enter", "distance_m": d})
            elif near_active and (d > NEAR_MISS_EXIT_DISTANCE_M):
                self._near_miss_active_by_worker[wid] = False
                events.append({"worker_id": wid, "event": "near_miss_exit", "distance_m": d})
        self._latest_near_miss_events = list(events)
        snapshot["near_miss_count"] = int(self.near_miss_count)
        snapshot["collision_count"] = int(self.collision_count)
        snapshot["near_miss_events"] = events
        snapshot["min_uav_worker_distance_m"] = self.min_uav_worker_distance_m

    def _debug_log_collision_probability_pipeline(self, snapshot: dict):
        if not isinstance(snapshot, dict):
            return
        safety_context = snapshot.get("safety_context")
        workers = snapshot.get("workers") or []
        if safety_context is None:
            return
        per_worker = list(getattr(safety_context, "per_worker_collision_probabilities", []) or [])
        if not per_worker:
            return
        worker_gt_map = {str(w.get("id")): w.get("gt_xy") for w in workers}
        worker_est_map = {str(w.get("id")): w.get("est_xy_bias_corrected") for w in workers}
        drone_gt = snapshot.get("drone_gt")
        drone_est_bias = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est")
        lines = [
            "[COLLISION_DEBUG] pipeline_snapshot",
            f"  drone_true_xy={None if drone_gt is None else (float(drone_gt[0]), float(drone_gt[1]))}",
            f"  drone_bias_corrected_xy={None if drone_est_bias is None else (float(drone_est_bias[0]), float(drone_est_bias[1]))}",
        ]
        for item in per_worker:
            wid = str(item.get("id"))
            lines.append(
                "  "
                + f"worker={wid} "
                + f"worker_true_xy={worker_gt_map.get(wid)} "
                + f"worker_bias_corrected_xy={worker_est_map.get(wid)} "
                + f"mu={item.get('mu_xy')} "
                + f"sigma_rel={item.get('sigma_rel')} "
                + f"r_u={item.get('r_u')} r_h={item.get('r_h')} r_c={item.get('r_c')} "
                + f"p_exact={item.get('exact_series_probability')} "
                + f"p_mc={item.get('monte_carlo_probability')} "
                + f"p_worker={item.get('collision_probability')}"
            )
        lines.append(
            "  "
            + f"scene_predicted_collision_probability={float(getattr(safety_context, 'predicted_collision_probability', 0.0)):.6f}"
        )
        print_debug("\n".join(lines), env_var="TYPEFLY_VERBOSE_DEBUG")

    def _get_drone_yaw_rad(self) -> float:
        get_yaw = getattr(self.drone, "get_drone_yaw", None)
        if callable(get_yaw):
            try:
                return float(get_yaw())
            except Exception:
                pass
        if hasattr(self.drone, "rotation_accumulator"):
            try:
                return math.radians(float(self.drone.rotation_accumulator))
            except Exception:
                pass
        return 0.0

    def _extract_target_task_point(self, task_text: str) -> str:
        text = (task_text or "").upper()
        for token in ("A", "B", "C"):
            if f" {token}" in f" {text}" or f"{token} " in text or f"TO {token}" in text:
                return token
        return "A"

    def _compute_path_eval(self, scene: BaselineScene, drone_pos, user_pos, now_s: float = 0.0, obstacle_envelopes=None, user_envelope=None):
        target_id = "A"
        if isinstance(self.latest_baseline_decision, dict):
            target_id = str(self.latest_baseline_decision.get("target_task_point") or "A")
        point = get_task_point(scene, target_id) or scene.task_points[0]
        safety_context = None
        snapshot = getattr(self, "latest_safety_context", None)
        if snapshot is not None:
            safety_context = snapshot
        user_radius = 0.75
        if safety_context is not None:
            user_radius = max(0.45, float(safety_context.uncertainty_scale_m) * 0.5)
        obstacle_envelopes = obstacle_envelopes if obstacle_envelopes is not None else compute_obstacle_envelope_states(scene, now_s=now_s)
        return evaluate_path_clear(
            drone_xy=(float(drone_pos[0]), float(drone_pos[1])),
            target_xy=(float(point.x), float(point.y)),
            user_xy=(None if user_pos is None else (float(user_pos[0]), float(user_pos[1]))),
            user_radius_m=user_radius,
            user_envelope=user_envelope,
            obstacle_envelopes=obstacle_envelopes,
            corridor_half_width_m=0.35,
        )

    def _compute_path_eval_for_target(self, scene: BaselineScene, drone_pos, user_pos, target_xy, obstacle_envelopes=None, user_envelope=None):
        safety_context = getattr(self, "latest_safety_context", None)
        user_radius = 0.75
        if safety_context is not None:
            user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        return evaluate_path_clear(
            drone_xy=(float(drone_pos[0]), float(drone_pos[1])),
            target_xy=(float(target_xy[0]), float(target_xy[1])),
            user_xy=(None if user_pos is None else (float(user_pos[0]), float(user_pos[1]))),
            user_radius_m=user_radius,
            user_envelope=user_envelope,
            obstacle_envelopes=obstacle_envelopes or [],
            corridor_half_width_m=0.35,
        )

    def _build_baseline_control_plan(self, task_description: str, snapshot):
        scene = self.get_baseline_scene()
        target_id = self._extract_target_task_point(task_description)
        target = get_task_point(scene, target_id)
        if target is None:
            return None
        drone_pos = snapshot.get("drone_est") or snapshot.get("drone_gt")
        if drone_pos is None:
            return None
        safety_context = snapshot.get("safety_context")
        obstacle_envelopes = snapshot.get("obstacle_envelope_states")
        if obstacle_envelopes is None:
            obstacle_envelopes = compute_obstacle_envelope_states(scene, now_s=time.time())
        path_eval = self._compute_path_eval(
            scene,
            drone_pos,
            None,
            now_s=time.time(),
            obstacle_envelopes=obstacle_envelopes,
            user_envelope=None,
        )
        risk_high = bool(
            safety_context is not None
            and float(safety_context.predicted_collision_probability) >= PREDICTED_COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        direct_go_to = bool(path_eval.path_clear and not risk_high)
        if direct_go_to:
            plan = "mf(1.0);d(0.2);mf(0.8);d(0.2);"
            mode = "direct_staged"
        else:
            side = "left"
            if path_eval.blocking_entity.startswith("O"):
                obstacle = next((o for o in obstacle_envelopes if o.id == path_eval.blocking_entity), None)
                if obstacle is not None:
                    side = "left" if float(obstacle.est_xy[1]) <= float(drone_pos[1]) else "right"
            if side == "left":
                plan = "tu(35);mf(0.9);tc(35);mf(0.9);d(0.2);"
            else:
                plan = "tc(35);mf(0.9);tu(35);mf(0.9);d(0.2);"
            mode = "staged_detour"
        note = (
            f"target={target.id}; path_clear={path_eval.path_clear}; "
            f"blocked_by={path_eval.blocking_entity}; direct_go_to={direct_go_to}; chosen={mode}"
        )
        obstacle_summary = [
            {
                "id": obs.id,
                "gt_xy": [float(obs.gt_xy[0]), float(obs.gt_xy[1])],
                "est_xy": [float(obs.est_xy[0]), float(obs.est_xy[1])],
                "matrix_xy": obs.matrix_xy,
                "envelope_major_axis_m": float(obs.envelope.major_axis_radius),
                "envelope_minor_axis_m": float(obs.envelope.minor_axis_radius),
                "orientation_deg": float(obs.envelope.orientation_deg),
            }
            for obs in obstacle_envelopes
        ]
        task_points_summary = [
            {"id": pt.id, "x": float(pt.x), "y": float(pt.y), "z": float(pt.z)}
            for pt in scene.task_points
        ]
        return {
            "scene_id": scene.id,
            "target_task_point": target.id,
            "task_text": task_description,
            "obstacle_positions_summary": obstacle_summary,
            "task_point_positions_summary": task_points_summary,
            "path_clear": bool(path_eval.path_clear),
            "blocking_entity": path_eval.blocking_entity,
            "corridor_min_gap": float(path_eval.corridor_min_gap),
            "chosen_motion_mode": mode,
            "direct_go_to": direct_go_to,
            "staged_detour": not direct_go_to,
            "recovery_first": False,
            "unknown": False,
            "generated_control_plan": plan,
            "decision_note": note,
            "plan": plan,
        }

    def get_baseline_expectation_summary(self, safety_context=None):
        scene = self.get_baseline_scene()
        risk_high = bool(
            safety_context is not None
            and float(safety_context.predicted_collision_probability) >= PREDICTED_COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        expectations = build_scene_expectations(
            scene=scene,
            user_radius_m=user_radius,
            corridor_half_width_m=0.35,
            high_risk=risk_high,
            now_s=time.time(),
        )
        return [
            {
                "scene_id": item.scene_id,
                "target_task_point": item.target_task_point,
                "expected_path_clear": item.expected_path_clear,
                "expected_blocking_entity": item.expected_blocking_entity,
                "expected_motion_mode": item.expected_motion_mode,
            }
            for item in expectations
        ]

    def get_all_scene_expectation_summary(self, safety_context=None):
        risk_high = bool(
            safety_context is not None
            and float(safety_context.predicted_collision_probability) >= PREDICTED_COLLISION_PROBABILITY_HIGH_RISK_THRESHOLD
        )
        user_radius = max(0.45, float(getattr(safety_context, "uncertainty_scale_m", 1.0)) * 0.5)
        expectations = build_all_scene_expectations(
            user_radius_m=user_radius,
            corridor_half_width_m=0.35,
            high_risk=risk_high,
            now_s=time.time(),
        )
        return [
            {
                "scene_id": item.scene_id,
                "target_task_point": item.target_task_point,
                "expected_path_clear": item.expected_path_clear,
                "expected_blocking_entity": item.expected_blocking_entity,
                "expected_motion_mode": item.expected_motion_mode,
            }
            for item in expectations
        ]

    def _debug_log_obstacle_envelopes(self, obstacle_states):
        if not obstacle_states:
            return
        parts = []
        for obs in obstacle_states:
            packet = obs.localization_packet
            parts.append(
                f"{obs.id}:gt=({obs.gt_xy[0]:.2f},{obs.gt_xy[1]:.2f}) "
                f"est=({obs.est_xy[0]:.2f},{obs.est_xy[1]:.2f}) "
                f"matrix={obs.matrix_xy} "
                f"axes=({obs.envelope.major_axis_radius:.3f},{obs.envelope.minor_axis_radius:.3f}) "
                f"ori={obs.envelope.orientation_deg:.1f}"
            )
        print_debug("[BASELINE-OBS] " + " | ".join(parts), env_var="TYPEFLY_VERBOSE_DEBUG")

    def _build_envelope_audit_summary(self, snapshot):
        safety_state = snapshot.get("safety_state")
        scene = snapshot.get("baseline_scene")
        obstacle_states = snapshot.get("obstacle_envelope_states") or []
        result = {"entities": []}
        if safety_state is None:
            return result

        drone_matrix = getattr(safety_state.drone_packet, "M_xy", None)
        result["entities"].append(
            {
                "id": "drone",
                "gt": snapshot.get("drone_gt"),
                "est": snapshot.get("drone_est"),
                "matrix_xy": drone_matrix,
                "matrix_source": "provider_packet.M_xy",
                "called_function": "build_safety_envelope",
                "localization_pipeline_function": "IterativeLeastSquaresEstimator3D.estimate",
                "base_sigma": "packet covariance",
                "nominal_size_used": False,
                "bias_used": False,
                "extra_inflation": "none",
                "chi2": getattr(safety_state.drone_envelope, "chi2_val", None),
                "major_axis": float(safety_state.drone_envelope.major_axis_radius),
                "minor_axis": float(safety_state.drone_envelope.minor_axis_radius),
                "orientation_deg": float(safety_state.drone_envelope.orientation_deg),
            }
        )
        obstacles_by_id = {}
        if scene is not None:
            obstacles_by_id = {obs.id: obs for obs in scene.obstacles}
        for obs_state in obstacle_states:
            base = obstacles_by_id.get(obs_state.id)
            result["entities"].append(
                {
                    "id": obs_state.id,
                    "gt": [float(obs_state.gt_xy[0]), float(obs_state.gt_xy[1])],
                    "est": [float(obs_state.est_xy[0]), float(obs_state.est_xy[1])],
                    "matrix_xy": obs_state.matrix_xy,
                    "matrix_source": "obstacle_state_packet.M_xy (consumed by build_safety_envelope)",
                    "called_function": obs_state.envelope_builder_function,
                    "localization_pipeline_function": obs_state.localization_pipeline_function,
                    "base_sigma": "packet covariance (anchor-based localization output)",
                    "nominal_size_used": False,
                    "bias_used": None if base is None else "range-bias model via LocalizationErrorModel",
                    "extra_inflation": "none (single chi2 expansion)",
                    "chi2": float(obs_state.envelope.chi2_val),
                    "major_axis": float(obs_state.envelope.major_axis_radius),
                    "minor_axis": float(obs_state.envelope.minor_axis_radius),
                    "orientation_deg": float(obs_state.envelope.orientation_deg),
                }
            )

        drone_major = float(safety_state.drone_envelope.major_axis_radius)
        ratios = {}
        for obs_state in obstacle_states:
            ratios[f"{obs_state.id}_to_drone_major"] = float(obs_state.envelope.major_axis_radius / max(1e-6, drone_major))
        result["ratios"] = ratios
        return result

    def _debug_log_envelope_audit(self, audit):
        if not audit or not audit.get("entities"):
            return
        lines = []
        for entity in audit["entities"]:
            lines.append(
                f"{entity['id']}: gt={entity['gt']} est={entity['est']} "
                f"matrix={entity['matrix_xy']} source={entity['matrix_source']} "
                f"loc_fn={entity.get('localization_pipeline_function', 'unknown')} "
                f"env_fn={entity.get('called_function', 'build_safety_envelope(via GCS service)')} "
                f"base_sigma={entity['base_sigma']} nominal_size_used={entity['nominal_size_used']} "
                f"bias={entity['bias_used']} inflate={entity['extra_inflation']} "
                f"chi2={entity['chi2']} major={entity['major_axis']:.3f} "
                f"minor={entity['minor_axis']:.3f} ori={entity['orientation_deg']:.2f}"
            )
        ratio_text = ", ".join(f"{k}={v:.3f}" for k, v in sorted((audit.get("ratios") or {}).items()))
        print_debug("[ENVELOPE-AUDIT] " + " || ".join(lines), env_var="TYPEFLY_VERBOSE_DEBUG")
        if ratio_text:
            print_debug("[ENVELOPE-RATIOS] " + ratio_text, env_var="TYPEFLY_VERBOSE_DEBUG")

    def _fmt_array_debug(self, arr):
        values = np.asarray(arr, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"

    def _debug_log_localization_pipeline_comparison(self, snapshot):
        safety_state = snapshot.get("safety_state")
        if safety_state is None:
            return
        obstacle_states = snapshot.get("obstacle_envelope_states") or []

        for obs in obstacle_states:
            packet = obs.localization_packet
            obs_lines = [
                f"[PIPELINE-CHECK] {obs.id}",
                f"  GT position: ({float(packet.gt_position_3d[0]):.3f}, {float(packet.gt_position_3d[1]):.3f}, {float(packet.gt_position_3d[2]):.3f})",
                f"  simulated anchor measurements: {self._fmt_array_debug(packet.measured_ranges)}",
                f"  estimated position: ({float(packet.estimated_position_3d[0]):.3f}, {float(packet.estimated_position_3d[1]):.3f}, {float(packet.estimated_position_3d[2]):.3f})",
                f"  localization covariance-like M_xy: {np.asarray(packet.M_xy, dtype=float).tolist()}",
                f"  localization pipeline function: {obs.localization_pipeline_function}",
                f"  envelope builder function: {obs.envelope_builder_function}",
                (
                    "  final major/minor/orientation: "
                    f"{float(obs.envelope.major_axis_radius):.3f}/"
                    f"{float(obs.envelope.minor_axis_radius):.3f}/"
                    f"{float(obs.envelope.orientation_deg):.3f}"
                ),
            ]
            print_debug("\n".join(obs_lines), env_var="TYPEFLY_VERBOSE_DEBUG")

    def start_robot(self):
        print_t("[C] Connecting to robot...")

        # Start state provider first so PX4_SIM wrapper can immediately consume live state.
        self.start_uwb()
        self.drone.connect()
        print_t("[C] Starting robot...")

        # Start state provider before PX4_SIM takeoff so wrapper has live sim state.
        self.start_uwb()

        if self.robot_type != RobotType.PX4_SIM:
            self.drone.takeoff()
            self.drone.move_up(0.25)
        self.apply_selected_scenario()

        if self.enable_video:
            print_t("[C] Starting stream...")
            self.drone.start_stream()
        if self.robot_type != RobotType.PX4_SIM:
            print_t("[C] Starting virtual position loop...")
            self.start_virtual_position_loop()
        self.controller_wait_takeoff = False

    def stop_robot(self):
        print_t("[C] Drone is landing...")
        self.drone.land()
        if self.enable_video:
            self.drone.stop_stream()
        print_t("[C] Stopping UWB tracking...")
        self.stop_uwb()
        if self.robot_type != RobotType.PX4_SIM:
            print_t("[C] Stopping virtual position loop...")
            self.stop_virtual_position_loop()
        self.controller_wait_takeoff = True

    def capture_loop(self, asyncio_loop):
        print_t("[C] Start capture loop...")
        frame_reader = self.drone.get_frame_reader()
        
        if frame_reader is None:
            print_t("[WARN] frame_reader is None, skipping capture loop")
            return
        
        while self.controller_active:
            self.drone.keep_active()
            if frame_reader is None or not hasattr(frame_reader, 'frame'):
                time.sleep(0.1)
                continue
            
            self.latest_frame = frame_reader.frame
            frame = Frame(frame_reader.frame,
                          frame_reader.depth if hasattr(frame_reader, 'depth') else None)
            
            if self.yolo_client.is_local_service():
                self.yolo_client.detect_local(frame)
            else:
                asyncio_loop.call_soon_threadsafe(asyncio.create_task, self.yolo_client.detect(frame))
            
            time.sleep(0.10)
            
        for task in asyncio.all_tasks(asyncio_loop):
            task.cancel()
        self.drone.stop_stream()
        self.drone.land()
        asyncio_loop.stop()
        print_t("[C] Capture loop stopped")
