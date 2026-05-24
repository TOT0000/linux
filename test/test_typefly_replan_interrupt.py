from types import SimpleNamespace
import threading

import pytest

from controller.minispec_interpreter import MiniSpecInterpreter, MiniSpecReturnValue, Statement
from controller.abs.skill_item import SkillArg
from controller.skillset import LowLevelSkillItem, SkillSet


class _NoopLogger:
    def start_run(self, **kwargs):
        return None

    def update_plan_info(self, *args, **kwargs):
        return None

    def update_baseline_info(self, *args, **kwargs):
        return None

    def update_planner_info(self, *args, **kwargs):
        return None

    def update_execution_info(self, *args, **kwargs):
        return None

    def consume_runtime_snapshot(self, *args, **kwargs):
        return None

    def end_run(self, *args, **kwargs):
        return None


def _install_minimal_skillset(gc_replan: bool):
    low = SkillSet(level="low")
    counters = {"gc": 0, "d": 0}

    def _gc(_checkpoint):
        counters["gc"] += 1
        return "risk_abort", gc_replan

    def _d(_seconds):
        counters["d"] += 1
        return None, False

    low.add_skill(LowLevelSkillItem("go_checkpoint", _gc, args=[SkillArg("checkpoint_id", str)]))
    low.add_skill(LowLevelSkillItem("delay", _d, args=[SkillArg("seconds", float)]))

    Statement.low_level_skillset = low
    Statement.high_level_skillset = SkillSet(level="high", lower_level_skillset=low)
    return counters


def test_gc_replan_interrupt_clears_remaining_queue_and_skips_following_statements():
    counters = _install_minimal_skillset(gc_replan=True)
    interpreter = MiniSpecInterpreter(message_queue=None)

    interpreter.execute(["gc('A1');d(2.0);gc('A2');"])
    ret_val = interpreter.ret_queue.get(timeout=1.0)

    assert ret_val.replan is True
    assert counters["gc"] == 1
    assert counters["d"] == 0
    assert Statement.execution_queue.empty()


def test_interpreter_abort_callback_before_execute_has_ret_queue_ready():
    _install_minimal_skillset(gc_replan=False)
    interpreter = MiniSpecInterpreter(
        message_queue=None,
        should_abort=lambda: (True, "pre_execute_abort"),
    )

    ret_val = interpreter.ret_queue.get(timeout=1.0)
    assert ret_val.replan is True
    assert "pre_execute_abort" in str(ret_val.value)


def test_typefly_replan_uses_fresh_llm_response_and_discards_old_queue(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)

    displayed_messages = []
    queued_programs = []
    planner_calls = []

    class _Planner:
        def __init__(self):
            self._responses = ["gc('A1');d(2.0);", "ml(1.0);gc('A2');d(2.0);"]

        def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return self._responses[len(planner_calls) - 1]

    def _execute_minispec_stub(program_text, silent=False, allow_auto_interrupt=True):
        queued_programs.append(program_text)
        if len(queued_programs) == 1:
            return MiniSpecReturnValue("MiniSpec execution interrupted for replan: current_collision_probability=0.700000", True)
        return MiniSpecReturnValue("ok", False)

    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "typefly-threshold-replan"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": []}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)

    controller.append_message = displayed_messages.append
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": []}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": {"completed": []}}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = _execute_minispec_stub
    controller.get_active_scenario_name = lambda: "test"

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)

    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert len(planner_calls) == 2
    assert queued_programs == ["gc('A1');d(2.0);", "ml(1.0);gc('A2');d(2.0);"]
    assert queued_programs[1] != queued_programs[0]
    plan_markers = [msg for msg in displayed_messages if isinstance(msg, str) and msg.startswith("[Plan]:")]
    assert len(plan_markers) == 2


def test_ui_collision_event_triggers_runtime_replan_abort():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    controller._runtime_replan_event = threading.Event()
    controller._runtime_replan_reason = ""
    controller.latest_ui_collision_probability = None
    controller.latest_ui_collision_timestamp = 0.0
    controller.predicted_collision_replan_threshold = 0.7
    controller._is_active_objective_completed = lambda: False
    controller._maybe_run_agent_heartbeat = lambda: False
    controller._should_trigger_auto_replan = lambda p, source="": float(p) >= 0.7

    controller.update_ui_collision_probability(0.9)
    should_abort, reason = controller._should_abort_current_execution_for_replan()

    assert should_abort is True
    assert "source=ui_event" in str(reason)


def _build_minimal_controller_for_postcheck(plans, completed_after_exec, mode="typefly-threshold-replan", replan_limit=8):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    displayed_messages = []
    queued_programs = []
    planner_calls = []

    class _Planner:
        def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return plans[len(planner_calls) - 1]

    def _execute_minispec_stub(program_text, silent=False, allow_auto_interrupt=True):
        queued_programs.append(program_text)
        idx = len(queued_programs) - 1
        controller.latest_benchmark_progress = {"completed": list(completed_after_exec[idx])}
        return MiniSpecReturnValue("ok", False)

    controller.replan_limit = int(replan_limit)
    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = mode
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": ["A1", "A2"]}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)

    controller.append_message = displayed_messages.append
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": ["A1", "A2"]}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": dict(controller.latest_benchmark_progress)}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = _execute_minispec_stub
    controller.get_active_scenario_name = lambda: "test"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""

    return controller, planner_calls, queued_programs, displayed_messages


def test_threshold_mode_postqueue_unfinished_does_not_trigger_auto_replan(monkeypatch):
    controller, planner_calls, queued_programs, displayed_messages = _build_minimal_controller_for_postcheck(
        plans=["gc('A1');", "gc('A2');"],
        completed_after_exec=[["A1"], ["A1", "A2"]],
        mode="typefly-threshold-replan",
        replan_limit=8,
    )
    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert len(planner_calls) == 1
    assert queued_programs == ["gc('A1');"]
    assert not any("TYPEFLY-POSTCHECK-REPLAN" in str(msg) for msg in displayed_messages)


def test_threshold_mode_no_postqueue_retry_loop(monkeypatch):
    controller, planner_calls, queued_programs, _ = _build_minimal_controller_for_postcheck(
        plans=["gc('A1');", "gc('A1');", "gc('A2');"],
        completed_after_exec=[[], ["A1"], ["A1", "A2"]],
        mode="typefly-threshold-replan",
        replan_limit=8,
    )
    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")
    assert len(planner_calls) == 1
    assert len(queued_programs) == 1


def test_postqueue_auto_replan_not_applied_to_agent_modes(monkeypatch):
    controller, planner_calls, queued_programs, displayed_messages = _build_minimal_controller_for_postcheck(
        plans=["gc('A1');", "gc('A2');"],
        completed_after_exec=[["A1"], ["A1", "A2"]],
        mode="agent-heartbeat-soft",
        replan_limit=8,
    )
    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="agent-heartbeat-soft")

    assert len(planner_calls) == 1
    assert len(queued_programs) == 1
    assert not any("TYPEFLY-POSTCHECK-REPLAN" in str(msg) for msg in displayed_messages)


def test_queue_exhausted_with_unfinished_checkpoints_marks_incomplete(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    class _CaptureLogger(_NoopLogger):
        def __init__(self):
            self.execution_updates = []
            self.end_status = None

        def update_execution_info(self, *args, **kwargs):
            self.execution_updates.append(dict(kwargs))

        def end_run(self, *args, **kwargs):
            self.end_status = dict(kwargs)

    controller = LLMController.__new__(LLMController)
    logger = _CaptureLogger()

    class _Planner:
        def plan(self, **kwargs):
            return "gc('A1');"

        def get_last_plan_trace(self):
            return {}

    controller.replan_limit = 8
    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "typefly-threshold-replan"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": ["A1", "A2"]}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = logger
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)
    controller.append_message = lambda _msg: None
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": ["A1", "A2"]}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": dict(controller.latest_benchmark_progress)}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = lambda _program_text, **kwargs: (
        controller.latest_benchmark_progress.update({"completed": ["A1"], "current_target": "A2"}) or MiniSpecReturnValue("ok", False)
    )
    controller.get_active_scenario_name = lambda: "test"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""
    controller._runtime_replan_event = __import__("threading").Event()
    controller._runtime_replan_reason = ""

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert logger.end_status == {"run_status": "incomplete"}
    assert any(update.get("queue_exhausted_with_unfinished") is True for update in logger.execution_updates)
    assert any(update.get("mission_success") is False for update in logger.execution_updates)
    assert any(update.get("completion_time_sec") is None for update in logger.execution_updates)


def test_agent_heartbeat_replan_response_is_emitted_to_chat():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    displayed_messages = []

    controller.framework_mode = "agent-heartbeat-soft"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""
    controller.last_heartbeat_ts = 0.0
    controller.replan_limit = 8
    controller._replan_attempts = 0
    controller.current_task_description = "task"
    controller.execution_history = "history"
    controller.current_plan = "gc('A1');"
    controller.append_message = displayed_messages.append
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.get_live_ui_snapshot = lambda: {}
    controller.planner = SimpleNamespace(
        plan_agent_heartbeat=lambda **kwargs: {
            "response": "full_replan_plan",
            "reason": "geometry_risk",
            "plan": "ml(0.3);gc('A2');d(2.0);",
        }
    )

    triggered = controller._maybe_run_agent_heartbeat(force=True)

    assert triggered is True
    assert controller._pending_heartbeat_replan_plan == "ml(0.3);gc('A2');d(2.0);"
    assert any(str(msg).startswith("[AGENT-HEARTBEAT-REPLAN]") for msg in displayed_messages)
    assert any(str(msg).startswith("[AGENT-HEARTBEAT-REPLAN-PLAN]") for msg in displayed_messages)


def test_pending_heartbeat_plan_clears_runtime_event_before_execute(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    displayed_messages = []
    planner_calls = []
    executed_programs = []

    class _Planner:
        def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return "gc('SHOULD_NOT_BE_USED');"

    def _execute_minispec_stub(program_text, silent=False, allow_auto_interrupt=True):
        executed_programs.append(program_text)
        assert bool(controller._runtime_replan_event.is_set()) is False
        return MiniSpecReturnValue("ok", False)

    controller.replan_limit = 8
    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "agent-heartbeat-soft"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": []}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)
    controller.append_message = displayed_messages.append
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": []}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": {"completed": []}}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = _execute_minispec_stub
    controller.get_active_scenario_name = lambda: "test"
    controller._pending_heartbeat_replan_plan = "ml(0.3);gc('A2');d(2.0);"
    controller._pending_heartbeat_reason = "geometry_risk"
    controller._runtime_replan_event = __import__("threading").Event()
    controller._runtime_replan_reason = "agent_heartbeat_replan:geometry_risk"
    controller._runtime_replan_event.set()

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="agent-heartbeat-soft")

    assert executed_programs[0] == "ml(0.3);gc('A2');d(2.0);"
    assert len(planner_calls) == 0


def test_agent_heartbeat_raw_response_is_emitted_even_when_non_json_continue():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    displayed_messages = []

    controller.framework_mode = "agent-heartbeat-soft"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""
    controller.last_heartbeat_ts = 0.0
    controller.replan_limit = 8
    controller._replan_attempts = 0
    controller.current_task_description = "task"
    controller.execution_history = "history"
    controller.current_plan = "gc('A1');"
    controller.append_message = displayed_messages.append
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.get_live_ui_snapshot = lambda: {}
    controller.planner = SimpleNamespace(
        plan_agent_heartbeat=lambda **kwargs: {
            "response": "continue",
            "reason": "non_json_response:```json{\"response\":\"full_replan_plan\"}",
            "plan": "",
            "raw_response": "```json\n{\"response\":\"full_replan_plan\",\"reason\":\"anomaly\",\"plan\":\"gc('A2');\"}\n```",
            "parsed_ok": False,
        }
    )

    triggered = controller._maybe_run_agent_heartbeat(force=True)

    assert triggered is False
    assert controller._pending_heartbeat_replan_plan is None
    assert any(str(msg).startswith("[AGENT-HEARTBEAT-RAW]") for msg in displayed_messages)


def test_agent_heartbeat_prompt_receives_replan_response_history():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    planner_calls = []

    controller.framework_mode = "agent-heartbeat-soft"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""
    controller.last_heartbeat_ts = 0.0
    controller.replan_limit = 8
    controller._replan_attempts = 3
    controller.current_task_description = "task"
    controller.execution_history = "gc('C1');d(2.0);"
    controller.current_plan = "gc('C2');d(2.0);"
    controller._replan_response_history = [
        {
            "ts": 1.0,
            "source": "agent_heartbeat_full_replan_response",
            "reason": "anomaly",
            "plan": "ml(0.5);gc('C2');d(2.0);",
            "raw_response": "{\"response\":\"full_replan_plan\"}",
        }
    ]
    controller.append_message = lambda _msg: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.get_live_ui_snapshot = lambda: {}
    controller.planner = SimpleNamespace(
        plan_agent_heartbeat=lambda **kwargs: planner_calls.append(kwargs) or {
            "response": "continue",
            "reason": "ok",
            "plan": "",
            "raw_response": "{\"response\":\"continue\"}",
            "parsed_ok": True,
        }
    )

    triggered = controller._maybe_run_agent_heartbeat(force=True)

    assert triggered is False
    assert len(planner_calls) == 1
    execution_history_text = str(planner_calls[0]["execution_history"])
    assert "replan_response_history:" in execution_history_text
    assert "agent_heartbeat_full_replan_response" in execution_history_text
    assert "ml(0.5);gc('C2');d(2.0);" in execution_history_text
    assert int(planner_calls[0]["full_replan_count"]) == 3


def test_threshold_mode_does_not_interrupt_when_replan_limit_reached():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    controller.framework_mode = "typefly-threshold-replan"
    controller.replan_limit = 8
    controller._replan_attempts = 5
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0

    should_replan = controller._should_trigger_auto_replan(0.95, source="interpreter_callback")

    assert should_replan is False
    assert controller.auto_replan_armed is True


def test_agent_heartbeat_stops_requesting_replan_after_limit_reached():
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    planner_calls = []

    controller.framework_mode = "agent-heartbeat-soft"
    controller._pending_heartbeat_replan_plan = None
    controller._pending_heartbeat_reason = ""
    controller.last_heartbeat_ts = 0.0
    controller.replan_limit = 8
    controller._replan_attempts = 5
    controller.current_task_description = "task"
    controller.execution_history = "history"
    controller.current_plan = "gc('A1');"
    controller.append_message = lambda _msg: None
    controller.get_live_ui_snapshot = lambda: {}
    controller._is_active_objective_completed = lambda: False
    controller._pending_execution_statement_count = lambda: 1
    controller.planner = SimpleNamespace(
        plan_agent_heartbeat=lambda **kwargs: planner_calls.append(kwargs) or {
            "response": "full_replan_plan",
            "reason": "geometry_risk",
            "plan": "gc('A2');",
        }
    )

    triggered = controller._maybe_run_agent_heartbeat(force=True)

    assert triggered is False
    assert len(planner_calls) == 0


def test_current_active_plan_matches_mission_original_when_no_full_replan(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)

    class _Planner:
        def plan(self, **kwargs):
            return "gc('A1');d(2.0);gc('A2');d(2.0);"

    controller.replan_limit = 8
    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "typefly-threshold-replan"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": ["A1", "A2"]}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)
    controller.append_message = lambda _msg: None
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": ["A1", "A2"]}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": {"completed": ["A1", "A2"]}}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = lambda *args, **kwargs: MiniSpecReturnValue("ok", False)
    controller.get_active_scenario_name = lambda: "test"

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert controller._mission_original_plan == "gc('A1');d(2.0);gc('A2');d(2.0);"
    assert controller._current_active_plan == controller._mission_original_plan


def test_current_active_plan_updates_after_full_replan(monkeypatch):
    pytest.importorskip("PIL")
    from controller.llm_controller import LLMController

    controller = LLMController.__new__(LLMController)
    planner_calls = []

    class _Planner:
        def __init__(self):
            self._responses = ["gc('A1');d(2.0);", "ml(0.5);gc('A2');d(2.0);"]

        def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return self._responses[len(planner_calls) - 1]

    exec_calls = []

    def _execute_minispec(program_text, silent=False, allow_auto_interrupt=True):
        exec_calls.append(program_text)
        if len(exec_calls) == 1:
            return MiniSpecReturnValue("MiniSpec execution interrupted for replan: current_collision_probability=0.700000", True)
        controller.latest_benchmark_progress = {"completed": ["A1", "A2"]}
        return MiniSpecReturnValue("ok", False)

    controller.replan_limit = 8
    controller.controller_wait_takeoff = False
    controller.message_queue = None
    controller.execution_history = []
    controller.current_plan = None
    controller.framework_mode = "typefly-threshold-replan"
    controller.execution_mode = "Waiting"
    controller.active_objective_set = {"active_checkpoint_ids": ["A1", "A2"]}
    controller.latest_benchmark_progress = {"completed": []}
    controller._benchmark_plan_checkpoint_sequence = []
    controller._task_id_counter = 0
    controller.auto_replan_armed = True
    controller.auto_replan_protection_remaining = 0
    controller.planner_mode = "llm"
    controller.planner = _Planner()
    controller.task_run_logger = _NoopLogger()
    controller.vision = SimpleNamespace(get_obj_list=lambda: "")
    controller.enable_video = False
    controller.state_provider = SimpleNamespace(debug_log_latest_localization_snapshot=lambda **kwargs: None)
    controller.safety_assessor = SimpleNamespace(build_from_provider=lambda provider: None)
    controller.append_message = lambda _msg: None
    controller._resolve_active_objective_set = lambda task_text: {"active_checkpoint_ids": ["A1", "A2"]}
    controller._reset_benchmark_progress_tracking = lambda: None
    controller._format_planner_location_info = lambda: "loc"
    controller.get_live_ui_snapshot = lambda: {"safety_context": None, "benchmark_progress": dict(controller.latest_benchmark_progress)}
    controller._debug_log_safety_context = lambda safety: None
    controller._build_baseline_control_plan = lambda **kwargs: None
    controller._sanitize_minispec_plan = lambda raw_plan: str(raw_plan)
    controller.execute_minispec = _execute_minispec
    controller.get_active_scenario_name = lambda: "test"

    monkeypatch.setattr("controller.llm_controller.AUTO_REPLAN_PROTECTION_STATEMENTS", 0)
    controller.execute_task_description("run mission", framework_mode="typefly-threshold-replan")

    assert controller._mission_original_plan == "gc('A1');d(2.0);"
    assert controller._current_active_plan == "ml(0.5);gc('A2');d(2.0);"
    assert controller._latest_full_replan_response == "ml(0.5);gc('A2');d(2.0);"
