from __future__ import annotations

import math
import re
import time
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .benchmark_layout import BENCHMARK_CHECKPOINTS_BY_ID
from .utils import print_debug

FrameworkMode = Literal["typefly_baseline", "langgraph_agent"]
MissionStatus = Literal["running", "completed", "failed"]
RouteDecision = Literal["continue", "retry_plan", "reselect_subgoal", "end", "abort"]


class AgentState(TypedDict, total=False):
    # task layer
    user_task: str
    task_id: str
    framework_mode: FrameworkMode
    mission_status: MissionStatus

    # objective layer
    active_zone_ids: list[str]
    active_checkpoint_ids: list[str]
    completed_checkpoint_ids: list[str]
    remaining_checkpoint_ids: list[str]
    current_subgoal_type: str
    current_subgoal_id: str | None
    subgoal_queue: list[str]

    # observation layer
    latest_snapshot: dict[str, Any] | None
    uav_est_xy: tuple[float, float] | None
    worker_states: list[dict[str, Any]]
    predicted_collision_risk: float
    predicted_collision_risk: float
    per_worker_collision_risks: dict[str, float]
    dominant_risky_worker: str | None
    mission_collision_count: int

    # execution layer
    last_plan_text: str
    last_action_text: str
    last_action_result: dict[str, Any]
    execution_history: list[dict[str, Any]]
    replan_count: int
    agent_step_count: int

    # protection layer
    auto_replan_armed: bool
    replan_protection_remaining_steps: int
    max_agent_steps: int
    max_replan_attempts: int
    last_error: str | None

    # internal routing
    route_decision: RouteDecision
    last_progress_signature: str
    last_subgoal_distance_m: float | None
    subgoal_phase: str
    subgoal_reached: bool
    arrived_but_not_completed: bool
    arrived_wait_cycles: int
    is_current_subgoal_completed: bool
    is_current_subgoal_in_radius: bool
    current_subgoal_dwell_seconds: float
    required_dwell_seconds: float
    dwell_satisfied: bool
    waiting_on_checkpoint_completion: bool
    waiting_checkpoint_id: str | None
    last_progress_event: dict[str, Any] | None
    completion_monitor_status: str
    recovery_mode: bool
    recovery_reason: str | None
    recovery_entry_risk: float | None
    recovery_last_risk: float | None
    recovery_just_exited: bool

    # strategy memory layer
    current_strategy_summary: str
    last_failure_reason: str | None
    recent_failed_approach_pattern: str | None
    recent_recovery_hypothesis: str | None
    blocked_workers_by_subgoal: dict[str, list[str]]
    subgoal_attempt_history: dict[str, list[str]]
    latest_failure_analysis: dict[str, Any] | None
    pending_strategy_update: dict[str, Any] | None
    current_mode: str
    mode_reason: str | None
    last_decision_payload: dict[str, Any] | None
    recent_failure_reason: str | None
    subgoal_reprioritization_suggestion: list[str]
    last_wait_event: dict[str, Any] | None
    last_risk_event: dict[str, Any] | None
    latest_collision_risk: float
    latest_dominant_risky_worker: str | None
    latest_per_worker_collision_risks: dict[str, float]


class LangGraphOrchestrationRunner:
    def __init__(self, controller):
        self.controller = controller
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def run_task(self, task_description: str, task_id: str, active_objective_set: dict[str, Any]) -> AgentState:
        active_checkpoint_ids = [str(v).upper() for v in active_objective_set.get("active_checkpoint_ids", [])]
        active_zone_ids = [str(v) for v in active_objective_set.get("active_zone_ids", [])]
        decomposed = self.controller.planner.decompose_task_for_langgraph(
            task_description=task_description,
            active_checkpoint_ids=active_checkpoint_ids,
            active_zone_ids=active_zone_ids,
        )
        if not decomposed:
            decomposed = list(active_checkpoint_ids)
        initial_state: AgentState = {
            "user_task": task_description,
            "task_id": task_id,
            "framework_mode": "langgraph_agent",
            "mission_status": "running",
            "active_zone_ids": active_zone_ids,
            "active_checkpoint_ids": active_checkpoint_ids,
            "completed_checkpoint_ids": [],
            "remaining_checkpoint_ids": list(decomposed),
            "current_subgoal_type": "checkpoint",
            "current_subgoal_id": None,
            "subgoal_queue": list(decomposed),
            "latest_snapshot": None,
            "uav_est_xy": None,
            "worker_states": [],
            "predicted_collision_risk": 0.0,
            "predicted_collision_risk": 0.0,
            "per_worker_collision_risks": {},
            "dominant_risky_worker": None,
            "mission_collision_count": 0,
            "last_plan_text": "",
            "last_action_text": "",
            "last_action_result": {},
            "execution_history": [],
            "replan_count": 0,
            "agent_step_count": 0,
            "auto_replan_armed": bool(getattr(self.controller, "auto_replan_armed", True)),
            "replan_protection_remaining_steps": int(getattr(self.controller, "auto_replan_protection_remaining", 0)),
            "max_agent_steps": 64,
            "max_replan_attempts": 8,
            "last_error": None,
            "route_decision": "continue",
            "last_progress_signature": "",
            "last_subgoal_distance_m": None,
            "subgoal_phase": "APPROACH_SUBGOAL",
            "subgoal_reached": False,
            "arrived_but_not_completed": False,
            "arrived_wait_cycles": 0,
            "is_current_subgoal_completed": False,
            "is_current_subgoal_in_radius": False,
            "current_subgoal_dwell_seconds": 0.0,
            "required_dwell_seconds": 2.0,
            "dwell_satisfied": False,
            "waiting_on_checkpoint_completion": False,
            "waiting_checkpoint_id": None,
            "last_progress_event": None,
            "completion_monitor_status": "idle",
            "recovery_mode": False,
            "recovery_reason": None,
            "recovery_entry_risk": None,
            "recovery_last_risk": None,
            "recovery_just_exited": False,
            "current_strategy_summary": "Prioritize safe progress toward current checkpoint; adapt approach when blocked.",
            "last_failure_reason": None,
            "recent_failed_approach_pattern": None,
            "recent_recovery_hypothesis": None,
            "blocked_workers_by_subgoal": {},
            "subgoal_attempt_history": {},
            "latest_failure_analysis": None,
            "pending_strategy_update": None,
            "current_mode": "approach",
            "mode_reason": None,
            "last_decision_payload": None,
            "recent_failure_reason": None,
            "subgoal_reprioritization_suggestion": [],
            "last_wait_event": None,
            "last_risk_event": None,
            "latest_collision_risk": 0.0,
            "latest_dominant_risky_worker": None,
            "latest_per_worker_collision_risks": {},
        }
        self._emit_agent_message(
            f"[AGENT] subgoal queue: {' -> '.join(list(decomposed)) if decomposed else '(empty)'}"
        )
        config = {"configurable": {"thread_id": task_id}}
        return self.graph.invoke(initial_state, config=config)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_runtime_state", self._node_load_runtime_state)
        builder.add_node("refresh_progress", self._node_refresh_progress)
        builder.add_node("select_subgoal", self._node_select_subgoal)
        builder.add_node("plan_step", self._node_plan_step)
        builder.add_node("execute_step", self._node_execute_step)
        builder.add_node("evaluate_outcome", self._node_evaluate_outcome)
        builder.add_node("analyze_failure_or_progress", self._node_analyze_failure_or_progress)
        builder.add_node("update_strategy_memory", self._node_update_strategy_memory)

        builder.add_edge(START, "load_runtime_state")
        builder.add_edge("load_runtime_state", "refresh_progress")
        builder.add_edge("refresh_progress", "select_subgoal")
        builder.add_edge("select_subgoal", "plan_step")
        builder.add_edge("plan_step", "execute_step")
        builder.add_edge("execute_step", "evaluate_outcome")
        builder.add_conditional_edges(
            "evaluate_outcome",
            self._route_from_evaluation,
            {
                "continue": "analyze_failure_or_progress",
                "retry_plan": "analyze_failure_or_progress",
                "reselect_subgoal": "analyze_failure_or_progress",
                "end": END,
                "abort": END,
            },
        )
        builder.add_edge("analyze_failure_or_progress", "update_strategy_memory")
        builder.add_edge("update_strategy_memory", "load_runtime_state")
        return builder.compile(checkpointer=self.checkpointer)

    def _node_load_runtime_state(self, state: AgentState) -> AgentState:
        snapshot = self.controller.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        workers = list(snapshot.get("workers") or []) if isinstance(snapshot, dict) else []
        per_worker = {}
        if safety_context is not None:
            per_worker = {
                str(item.get("id")): float(item.get("collision_probability", 0.0))
                for item in (getattr(safety_context, "per_worker_collision_probabilities", []) or [])
            }
        uav_est = None
        if isinstance(snapshot, dict):
            uav_pos = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est")
            if uav_pos is not None:
                uav_est = (float(uav_pos[0]), float(uav_pos[1]))
        predicted_collision_risk = 0.0 if safety_context is None else float(getattr(safety_context, "current_collision_probability", 0.0))
        predicted_collision_risk = 0.0 if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))
        dominant_risky_worker = None if safety_context is None else str(getattr(safety_context, "dominant_threat_id", "unknown"))
        print_debug(
            "[AGENT-LOAD-RUNTIME] "
            f"step={int(state.get('agent_step_count', 0))} "
            f"route_decision={state.get('route_decision')} "
            f"predicted_collision_risk={predicted_collision_risk:.6f} "
            f"dominant_risky_worker={dominant_risky_worker} "
            f"per_worker={per_worker}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return {
            "latest_snapshot": snapshot,
            "uav_est_xy": uav_est,
            "worker_states": workers,
            "predicted_collision_risk": predicted_collision_risk,
            "latest_collision_risk": predicted_collision_risk,
            "predicted_collision_risk": predicted_collision_risk,
            "per_worker_collision_risks": per_worker,
            "latest_per_worker_collision_risks": per_worker,
            "dominant_risky_worker": dominant_risky_worker,
            "latest_dominant_risky_worker": dominant_risky_worker,
            "mission_collision_count": int(state.get("mission_collision_count", 0)),
        }

    def _node_refresh_progress(self, state: AgentState) -> AgentState:
        snapshot = state.get("latest_snapshot") or {}
        completed = set(str(v).upper() for v in state.get("completed_checkpoint_ids", []))
        progress = snapshot.get("benchmark_progress") if isinstance(snapshot, dict) else None
        if isinstance(progress, dict):
            completed.update(str(v).upper() for v in progress.get("completed", []))
        active_ids = [str(v).upper() for v in state.get("active_checkpoint_ids", [])]
        queue_order = [str(v).upper() for v in state.get("subgoal_queue", [])]
        if not queue_order:
            queue_order = [str(v).upper() for v in state.get("remaining_checkpoint_ids", [])]
        if not queue_order:
            queue_order = list(active_ids)
        remaining = [cid for cid in queue_order if cid in active_ids and cid not in completed]
        return {
            "completed_checkpoint_ids": sorted(completed),
            "remaining_checkpoint_ids": remaining,
            "subgoal_queue": list(remaining),
        }

    def _node_select_subgoal(self, state: AgentState) -> AgentState:
        remaining = list(state.get("remaining_checkpoint_ids", []))
        current_subgoal = state.get("current_subgoal_id")
        completed = set(state.get("completed_checkpoint_ids", []))
        subgoal_phase = str(state.get("subgoal_phase", "APPROACH_SUBGOAL"))
        subgoal_reached = bool(state.get("subgoal_reached", False))
        arrived_but_not_completed = bool(state.get("arrived_but_not_completed", False))
        arrived_wait_cycles = int(state.get("arrived_wait_cycles", 0))
        waiting_on_completion = bool(state.get("waiting_on_checkpoint_completion", False))
        waiting_checkpoint_id = state.get("waiting_checkpoint_id")
        monitor_status = str(state.get("completion_monitor_status", "idle"))
        if current_subgoal is None or current_subgoal in completed:
            current_subgoal = remaining[0] if remaining else None
            subgoal_phase = "APPROACH_SUBGOAL"
            subgoal_reached = False
            arrived_but_not_completed = False
            arrived_wait_cycles = 0
            waiting_on_completion = False
            waiting_checkpoint_id = None
            monitor_status = "idle"
        if state.get("route_decision") == "reselect_subgoal":
            current_subgoal = remaining[0] if remaining else None
            subgoal_phase = "APPROACH_SUBGOAL"
            subgoal_reached = False
            arrived_but_not_completed = False
            arrived_wait_cycles = 0
            waiting_on_completion = False
            waiting_checkpoint_id = None
            monitor_status = "idle"
        return {
            "current_subgoal_type": "checkpoint",
            "current_subgoal_id": current_subgoal,
            "subgoal_phase": subgoal_phase,
            "subgoal_reached": subgoal_reached,
            "arrived_but_not_completed": arrived_but_not_completed,
            "arrived_wait_cycles": arrived_wait_cycles,
            "waiting_on_checkpoint_completion": waiting_on_completion,
            "waiting_checkpoint_id": waiting_checkpoint_id,
            "completion_monitor_status": monitor_status,
        }

    def _node_plan_step(self, state: AgentState) -> AgentState:
        print_debug(
            "[AGENT-PLAN-INPUT] "
            f"step={int(state.get('agent_step_count', 0))} "
            f"route_decision={state.get('route_decision')} "
            f"subgoal={state.get('current_subgoal_id')} "
            f"predicted_collision_risk={float(state.get('predicted_collision_risk', 0.0)):.6f} "
            f"dominant_risky_worker={state.get('dominant_risky_worker')} "
            f"per_worker={state.get('per_worker_collision_risks')}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        subgoal = state.get("current_subgoal_id")
        current_mode = str(state.get("current_mode", "approach"))
        completed = set(str(v).upper() for v in state.get("completed_checkpoint_ids", []))
        remaining = [str(v).upper() for v in state.get("remaining_checkpoint_ids", []) if str(v).upper() not in completed]
        if subgoal in completed:
            subgoal = None
        if subgoal is None and remaining:
            subgoal = remaining[0]
        if subgoal is None:
            if hasattr(self.controller, "set_benchmark_progress_focus_checkpoint"):
                self.controller.set_benchmark_progress_focus_checkpoint(None)
            return {"last_plan_text": "", "last_action_text": "", "route_decision": "end"}
        if hasattr(self.controller, "set_benchmark_progress_focus_checkpoint"):
            self.controller.set_benchmark_progress_focus_checkpoint(str(subgoal).upper())
        subgoal_phase = str(state.get("subgoal_phase", "APPROACH_SUBGOAL"))
        # In agent mode, completion/verify handling is observation-driven and delegated to LLM.
        if False and subgoal_phase in {"COMPLETE_SUBGOAL", "VERIFY_COMPLETE"}:
            in_radius = bool(state.get("is_current_subgoal_in_radius", False))
            dwell_satisfied = bool(state.get("dwell_satisfied", False))
            waiting_checkpoint = state.get("waiting_checkpoint_id")
            waiting_active = bool(state.get("waiting_on_checkpoint_completion", False)) and waiting_checkpoint == subgoal
            if in_radius and not dwell_satisfied:
                plan = "wait_checkpoint_event();"
                if not waiting_active:
                    print_debug(
                        "[AGENT-WAIT] "
                        f"checkpoint={str(subgoal).upper()} "
                        f"dwell={float(state.get('current_subgoal_dwell_seconds', 0.0)):.2f}/"
                        f"{float(state.get('required_dwell_seconds', 2.0)):.2f}s",
                        env_var="TYPEFLY_VERBOSE_DEBUG",
                    )
            elif not in_radius:
                plan = f'go_checkpoint("{str(subgoal).upper()}");'
                self._emit_agent_message("[STEP] phase: left checkpoint area, re-approach subgoal")
            else:
                plan = "wait_checkpoint_event();"
                if not waiting_active:
                    print_debug(
                        f"[AGENT-WAIT] checkpoint={str(subgoal).upper()} dwell_satisfied",
                        env_var="TYPEFLY_VERBOSE_DEBUG",
                    )
            action_text = plan
            self._emit_agent_message(f"[STEP] current subgoal: {subgoal}")
            if plan != "wait_checkpoint_event();":
                self._emit_agent_message(f"[ACTION] {plan}")
            self._emit_agent_message("[MODE] approach | reason=complete_or_verify_subgoal_phase")
            return {
                "last_plan_text": plan,
                "last_action_text": action_text,
                "route_decision": "continue",
                "current_subgoal_id": subgoal,
                "current_subgoal_type": "checkpoint",
                "current_mode": "approach",
                "mode_reason": "complete_or_verify_subgoal_phase",
                "last_decision_payload": {
                    "mode": "approach",
                    "reason": "complete_or_verify_subgoal_phase",
                    "strategy_summary": str(state.get("current_strategy_summary", "")),
                    "action": plan,
                },
                "waiting_on_checkpoint_completion": bool(plan == "wait_checkpoint_event();"),
                "waiting_checkpoint_id": (str(subgoal).upper() if plan == "wait_checkpoint_event();" else None),
                "completion_monitor_status": ("waiting_event" if plan == "wait_checkpoint_event();" else "active"),
            }

        collision_risk = float(state.get("predicted_collision_risk", 0.0))
        last_action = str(state.get("last_action_text", "")).strip()
        recovery_mode = bool(current_mode == "recovery")
        decision = self.controller.planner.decide_langgraph_mode_and_action(
            task_description=str(state.get("user_task", "")),
            current_subgoal=str(subgoal),
            remaining_checkpoints=remaining,
            predicted_collision_risk=collision_risk,
            per_worker_collision_risks=dict(state.get("per_worker_collision_risks", {})),
            dominant_risky_worker=state.get("dominant_risky_worker"),
            worker_states_summary=list(state.get("worker_states", [])),
            last_action=last_action,
            last_result=str((state.get("last_action_result") or {}).get("message", "")),
            recent_history=list(state.get("execution_history", [])),
            recovery_mode=recovery_mode,
            recovery_reason=(None if state.get("recovery_reason") is None else str(state.get("recovery_reason"))),
            strategy_summary=str(state.get("current_strategy_summary", "")),
            last_failure_reason=(None if state.get("last_failure_reason") is None else str(state.get("last_failure_reason"))),
            failed_approach_pattern=(None if state.get("recent_failed_approach_pattern") is None else str(state.get("recent_failed_approach_pattern"))),
            recovery_hypothesis=(None if state.get("recent_recovery_hypothesis") is None else str(state.get("recent_recovery_hypothesis"))),
            blocked_workers_for_subgoal=list((state.get("blocked_workers_by_subgoal", {}) or {}).get(str(subgoal).upper(), [])),
            subgoal_attempts=list((state.get("subgoal_attempt_history", {}) or {}).get(str(subgoal).upper(), []))[-6:],
            current_mode=current_mode,
            last_wait_event=dict(state.get("last_wait_event") or {}),
            last_risk_event=dict(state.get("last_risk_event") or {}),
            current_subgoal_phase=subgoal_phase,
            reached_not_completed=bool(state.get("arrived_but_not_completed", False)),
            in_checkpoint_radius=bool(state.get("is_current_subgoal_in_radius", False)),
            dwell_seconds=float(state.get("current_subgoal_dwell_seconds", 0.0) or 0.0),
            dwell_required_seconds=float(state.get("required_dwell_seconds", 2.0) or 2.0),
            dwell_satisfied=bool(state.get("dwell_satisfied", False)),
        )
        plan = str(decision.get("action", "")).strip()
        mode = str(decision.get("mode", "approach")).strip().lower()
        reason = str(decision.get("reason", "")).strip()[:180]
        strategy_summary = str(decision.get("strategy_summary", state.get("current_strategy_summary", ""))).strip()
        suggested_subgoal = decision.get("next_subgoal")
        why_match = str(decision.get("why_action_matches_mode", "")).strip()[:220]
        decision_trace = dict(decision.get("trace") or {})
        trace_events = list(decision_trace.get("events") or [])
        if not plan:
            if mode == "skip_current_subgoal":
                plan = "d(0.5);"
            elif mode == "replan":
                plan = "lo('request_replan');"
            else:
                plan = "lo('empty_action_from_decision');"
            trace_events.append("planner_empty_action_fallback")
        action_target = self._extract_checkpoint_target(plan)
        if action_target is not None:
            allow_switch = bool(
                mode in {"skip_current_subgoal", "replan"}
                or (isinstance(suggested_subgoal, str) and str(suggested_subgoal).strip())
            )
            if (subgoal is not None) and action_target != str(subgoal).upper() and (not allow_switch):
                trace_events.append("target_mismatch_preserved")
            if action_target in completed or action_target not in remaining:
                trace_events.append("target_outside_remaining_preserved")
            if action_target in remaining:
                subgoal = action_target
        action_text = plan
        self._emit_agent_message(f"[STEP] current subgoal: {subgoal}")
        self._emit_agent_message(f"[MODE] {mode} | reason={reason or 'n/a'} | strategy={strategy_summary[:100]}")
        if why_match:
            self._emit_agent_message(f"[ANALYZE] {why_match}")
        for event in trace_events:
            self._emit_agent_message(f"[EVENT] {event}")
        self._emit_agent_message(f"[ACTION] {plan}")
        next_route: RouteDecision = "continue"
        if mode in {"skip_current_subgoal", "replan"}:
            next_route = "reselect_subgoal"
        updated_queue = [str(v).upper() for v in list(state.get("subgoal_queue", []))]
        if not updated_queue:
            updated_queue = list(remaining)
        if isinstance(suggested_subgoal, str) and suggested_subgoal.strip().upper() in remaining:
            next_route = "reselect_subgoal"
            suggested_key = suggested_subgoal.strip().upper()
            updated_queue = [suggested_key] + [cid for cid in updated_queue if cid != suggested_key]
        return {
            "last_plan_text": plan,
            "last_action_text": action_text,
            "route_decision": next_route,
            "current_subgoal_id": subgoal,
            "current_subgoal_type": "checkpoint",
            "current_mode": mode,
            "mode_reason": (reason or None),
            "current_strategy_summary": strategy_summary,
            "subgoal_reprioritization_suggestion": ([str(suggested_subgoal).upper()] if isinstance(suggested_subgoal, str) else []),
            "subgoal_queue": updated_queue,
            "last_decision_payload": {
                "mode": mode,
                "reason": reason,
                "strategy_summary": strategy_summary,
                "action": plan,
                "next_subgoal": suggested_subgoal,
                "why_action_matches_mode": why_match,
                "trace": {
                    "raw_llm_payload": decision_trace.get("raw_llm_payload"),
                    "parsed_payload": decision_trace.get("parsed_payload"),
                    "sanitized_payload": decision_trace.get("sanitized_payload"),
                    "executor_input": plan,
                    "events": trace_events,
                },
            },
            "recovery_mode": bool(mode == "recovery"),
            "recovery_reason": (reason or None) if mode == "recovery" else None,
            "recovery_just_exited": False,
        }

    def _node_execute_step(self, state: AgentState) -> AgentState:
        if str(state.get("route_decision", "")) == "end":
            return {
                "last_action_result": {"ok": True, "recoverable": False, "message": "terminal_noop"},
                "last_error": None,
            }
        plan = str(state.get("last_plan_text", "")).strip()
        if not plan:
            return {
                "last_action_result": {"ok": False, "recoverable": False, "message": "empty_plan"},
                "last_error": "empty_plan",
            }
        self._emit_agent_message(f"[EVENT] executor_input={plan}")
        if plan == "wait_checkpoint_event();":
            checkpoint_id = state.get("waiting_checkpoint_id") or state.get("current_subgoal_id")
            event = self.controller.wait_for_checkpoint_progress_event(
                str(checkpoint_id),
                timeout_seconds=8.0,
                risk_abort_threshold=None,
            )
            event_type = str(event.get("event_type", "unknown"))
            ok = event_type not in {"waiting_timeout", "risk_abort"}
            return {
                "last_action_result": {
                    "ok": ok,
                    "recoverable": True,
                    "message": f"checkpoint_event:{event_type}",
                    "event": event,
                },
                "last_error": None if ok else str(event.get("reason", event_type)),
            }
        try:
            ret = self.controller.execute_minispec(plan, silent=True, allow_auto_interrupt=True)
            ok = True
            recoverable = False
            if isinstance(ret, tuple) and len(ret) >= 2:
                ok = bool(ret[0] is not False)
            if hasattr(ret, "replan") and bool(ret.replan):
                ok = False
                recoverable = True
            return {
                "last_action_result": {
                    "ok": ok,
                    "recoverable": recoverable,
                    "message": str(getattr(ret, "value", ret)),
                },
                "last_error": None if ok else str(getattr(ret, "value", "step_failed")),
            }
        except Exception as exc:
            return {
                "last_action_result": {"ok": False, "recoverable": True, "message": str(exc)},
                "last_error": str(exc),
            }

    def _node_evaluate_outcome(self, state: AgentState) -> AgentState:
        step_count = int(state.get("agent_step_count", 0)) + 1
        replan_count = int(state.get("replan_count", 0))
        result = dict(state.get("last_action_result") or {})
        progress_event = result.get("event") if isinstance(result.get("event"), dict) else None
        last_wait_event = None if progress_event is None else dict(progress_event)
        last_risk_event = dict(state.get("last_risk_event") or {})
        if isinstance(progress_event, dict):
            risk_payload = progress_event.get("risk")
            if risk_payload is not None or str(progress_event.get("event_type", "")) in {"risk_abort", "waiting_timeout"}:
                last_risk_event = {
                    "event_type": str(progress_event.get("event_type", "")),
                    "reason": progress_event.get("reason"),
                    "risk": risk_payload,
                    "checkpoint_id": progress_event.get("checkpoint_id"),
                    "ts": progress_event.get("timestamp", time.time()),
                }
        history = list(state.get("execution_history", []))
        history.append(
            {
                "step": step_count,
                "subgoal": state.get("current_subgoal_id"),
                "plan": state.get("last_plan_text"),
                "ok": bool(result.get("ok", False)),
                "recoverable": bool(result.get("recoverable", False)),
                "message": result.get("message", ""),
                "ts": time.time(),
            }
        )
        if not bool(result.get("ok", False)):
            replan_count += 1
        if progress_event is not None:
            event_type = str(progress_event.get("event_type", "unknown"))
            checkpoint = str(progress_event.get("checkpoint_id") or state.get("current_subgoal_id") or "n/a").upper()
            dwell = float(progress_event.get("dwell_seconds", 0.0) or 0.0)
            required = float(progress_event.get("required_dwell_seconds", 0.0) or 0.0)
            if event_type in {"dwell_started", "entered_checkpoint_area"}:
                self._emit_agent_message(f"[EVENT] dwell in progress: {checkpoint} ({dwell:.1f} / {required:.1f} s)")
            elif event_type == "dwell_progress":
                print_debug(
                    f"[EVENT] dwell progress: {checkpoint} ({dwell:.1f} / {required:.1f} s)",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )
            elif event_type == "dwell_satisfied":
                self._emit_agent_message(f"[EVENT] dwell satisfied: {checkpoint}")
            elif event_type == "checkpoint_completed":
                self._emit_agent_message(f"[EVENT] checkpoint completed: {checkpoint}")
            elif event_type == "left_checkpoint_area":
                self._emit_agent_message(f"[EVENT] left checkpoint area before completion: {checkpoint}")
            elif event_type == "risk_abort":
                self._emit_agent_message(f"[EVENT] risk too high while waiting: {checkpoint} ({progress_event.get('reason')})")
            elif event_type == "waiting_timeout":
                self._emit_agent_message(f"[EVENT] waiting timeout: {checkpoint} ({progress_event.get('reason')})")
            else:
                self._emit_agent_message(f"[EVENT] {event_type}: {checkpoint}")
        else:
            self._emit_agent_message(
                f"[RESULT] {'ok' if bool(result.get('ok', False)) else 'failed'}: {str(result.get('message', ''))[:120]}"
            )

        max_steps = int(state.get("max_agent_steps", 64))
        max_replans = int(state.get("max_replan_attempts", 8))
        if step_count >= max_steps:
            self._emit_agent_message(
                f"[EVENT] soft_limit_reached:max_agent_steps({max_steps}) -> continue_with_agent_control"
            )
        if replan_count > max_replans:
            self._emit_agent_message(
                f"[EVENT] soft_limit_reached:max_replan_attempts({max_replans}) -> continue_with_agent_control"
            )

        remaining = list(state.get("remaining_checkpoint_ids", []))
        subgoal = state.get("current_subgoal_id")
        current_mode = str(state.get("current_mode", "approach"))
        recovery_mode = bool(current_mode == "recovery")
        recovery_reason = state.get("mode_reason") if recovery_mode else None
        recovery_entry_risk = state.get("recovery_entry_risk")
        recovery_last_risk = state.get("recovery_last_risk")
        recovery_just_exited = False
        latest_snapshot = self.controller.get_live_ui_snapshot()
        latest_safety = latest_snapshot.get("safety_context") if isinstance(latest_snapshot, dict) else None
        latest_collision_risk = 0.0 if latest_safety is None else float(getattr(latest_safety, "current_collision_probability", 0.0))
        recovery_last_risk = float(latest_collision_risk)
        if recovery_mode:
            action_text = str(state.get("last_action_text", "")).strip()
            print_debug(
                f"[RECOVERY] step summary: subgoal={state.get('current_subgoal_id')} "
                f"action={action_text} latest_risk={latest_collision_risk:.3f} "
                f"recovery_mode={recovery_mode}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        progress = latest_snapshot.get("benchmark_progress") if isinstance(latest_snapshot, dict) else None
        progress_current_target = None
        progress_active_enter_ts = None
        progress_distance_m = None
        progress_tick_ts = None
        completed_from_progress = set()
        if isinstance(progress, dict):
            progress_current_target = progress.get("current_target")
            progress_active_enter_ts = progress.get("active_enter_ts")
            progress_distance_m = progress.get("distance_to_target_m")
            progress_tick_ts = progress.get("tick_ts")
            completed_from_progress = set(str(v).upper() for v in progress.get("completed", []))
        completed = set(str(v).upper() for v in state.get("completed_checkpoint_ids", []))
        completed.update(completed_from_progress)
        remaining = [cid for cid in remaining if str(cid).upper() not in completed]
        arrived_but_not_completed = bool(state.get("arrived_but_not_completed", False))
        arrived_wait_cycles = int(state.get("arrived_wait_cycles", 0))
        in_radius = False
        dwell_seconds = 0.0
        required_dwell_seconds = float(state.get("required_dwell_seconds", 2.0))
        dwell_satisfied = False
        if isinstance(progress, dict):
            in_radius = bool(progress.get("in_radius", False))
            dwell_seconds = float(progress.get("dwell_seconds", 0.0) or 0.0)
            required_dwell_seconds = float(progress.get("required_dwell_seconds", required_dwell_seconds) or required_dwell_seconds)
            dwell_satisfied = bool(progress.get("dwell_satisfied", False))
        waiting_hint = (
            bool(state.get("waiting_on_checkpoint_completion", False))
            and state.get("waiting_checkpoint_id") is not None
            and str(state.get("waiting_checkpoint_id")).upper() == str(subgoal).upper()
        )
        subgoal_center = None
        subgoal_dist = None
        subgoal_in_radius_geom = False
        if subgoal is not None:
            cp = BENCHMARK_CHECKPOINTS_BY_ID.get(str(subgoal))
            drone_gt = latest_snapshot.get("drone_gt") if isinstance(latest_snapshot, dict) else None
            if cp is not None and drone_gt is not None:
                subgoal_center = (float(cp.x), float(cp.y), float(cp.radius_m))
                subgoal_dist = math.hypot(float(drone_gt[0]) - float(cp.x), float(drone_gt[1]) - float(cp.y))
                subgoal_in_radius_geom = bool(subgoal_dist <= float(cp.radius_m))
        target_aligned = (
            subgoal is not None
            and progress_current_target is not None
            and str(progress_current_target).upper() == str(subgoal).upper()
        )
        if waiting_hint and subgoal is not None:
            target_aligned = True
        effective_in_radius = bool(in_radius if target_aligned else subgoal_in_radius_geom)
        if (not target_aligned) and (not waiting_hint):
            dwell_seconds = 0.0
            dwell_satisfied = False
        if subgoal is not None:
            print_debug(
                "[AGENT-CHECKPOINT-ALIGN] "
                f"subgoal={str(subgoal).upper()} "
                f"action_target={self._extract_checkpoint_target(str(state.get('last_plan_text', '')))} "
                f"progress_target={None if progress_current_target is None else str(progress_current_target).upper()} "
                f"in_radius_progress={in_radius} "
                f"in_radius_geom={subgoal_in_radius_geom} "
                f"effective_in_radius={effective_in_radius} "
                f"dwell_seconds={dwell_seconds:.3f} "
                f"required_dwell={required_dwell_seconds:.3f} "
                f"dwell_satisfied={dwell_satisfied} "
                f"completed={sorted(completed)} "
                f"progress_active_enter_ts={progress_active_enter_ts} "
                f"progress_distance_m={progress_distance_m} "
                f"progress_tick_ts={progress_tick_ts} "
                f"drone_gt={latest_snapshot.get('drone_gt') if isinstance(latest_snapshot, dict) else None} "
                f"cp_center={subgoal_center} "
                f"dist_to_subgoal={subgoal_dist}",
                env_var="TYPEFLY_VERBOSE_DEBUG",
            )
        action_target = self._extract_checkpoint_target(str(state.get("last_plan_text", "")))
        result_msg = str(result.get("message", "")).lower()
        reached_area = (" reached:" in result_msg or " reached " in result_msg) and ("approached" not in result_msg)
        if action_target and reached_area:
            self._emit_agent_message(f"[RESULT] reached checkpoint area: {action_target} (not yet completed)")
        phase = str(state.get("subgoal_phase", "APPROACH_SUBGOAL"))
        reached_flag = bool(state.get("subgoal_reached", False))
        waiting_on_completion = bool(state.get("waiting_on_checkpoint_completion", False))
        waiting_checkpoint_id = state.get("waiting_checkpoint_id")
        completion_monitor_status = str(state.get("completion_monitor_status", "idle"))
        if action_target and reached_area and action_target == str(subgoal).upper():
            phase = "COMPLETE_SUBGOAL"
            reached_flag = True
            arrived_but_not_completed = True
            arrived_wait_cycles = 0
            waiting_on_completion = True
            waiting_checkpoint_id = str(subgoal).upper()
            completion_monitor_status = "waiting_event"
        if progress_event is not None:
            event_type = str(progress_event.get("event_type", ""))
            event_cp = str(progress_event.get("checkpoint_id") or (subgoal or "")).upper()
            event_in_radius = bool(progress_event.get("in_radius", False))
            event_dwell_satisfied = bool(progress_event.get("dwell_satisfied", False))
            if event_type in {"entered_checkpoint_area", "dwell_started", "dwell_progress"}:
                phase = "COMPLETE_SUBGOAL"
                reached_flag = True
                arrived_but_not_completed = True
                waiting_on_completion = True
                waiting_checkpoint_id = event_cp
                completion_monitor_status = "event_progress"
            elif event_type == "dwell_satisfied":
                phase = "VERIFY_COMPLETE"
                reached_flag = True
                arrived_but_not_completed = True
                waiting_on_completion = True
                waiting_checkpoint_id = event_cp
                completion_monitor_status = "event_dwell_satisfied"
            elif event_type == "checkpoint_completed":
                completed.add(event_cp)
                phase = "DONE"
                waiting_on_completion = False
                waiting_checkpoint_id = None
                completion_monitor_status = "event_completed"
            elif event_type == "left_checkpoint_area":
                phase = "APPROACH_SUBGOAL"
                reached_flag = False
                arrived_but_not_completed = False
                waiting_on_completion = False
                waiting_checkpoint_id = None
                completion_monitor_status = "event_left_area"
            elif event_type in {"risk_abort", "waiting_timeout"}:
                waiting_on_completion = False
                waiting_checkpoint_id = None
                completion_monitor_status = f"event_{event_type}"
                replan_count += 1 if event_type == "risk_abort" else 0
            if event_type in {"entered_checkpoint_area", "dwell_started", "dwell_progress", "dwell_satisfied"}:
                in_radius = event_in_radius
                dwell_satisfied = event_dwell_satisfied
                dwell_seconds = float(progress_event.get("dwell_seconds", dwell_seconds) or dwell_seconds)
                required_dwell_seconds = float(progress_event.get("required_dwell_seconds", required_dwell_seconds) or required_dwell_seconds)
        if subgoal is not None and str(subgoal).upper() in completed:
            self._emit_agent_message(f"[RESULT] checkpoint completed: {str(subgoal).upper()}")
            phase = "DONE"
            subgoal = None
            reached_flag = False
            arrived_but_not_completed = False
            arrived_wait_cycles = 0
            waiting_on_completion = False
            waiting_checkpoint_id = None
            completion_monitor_status = "completed"
        elif phase == "VERIFY_COMPLETE" and subgoal is not None and str(subgoal).upper() not in completed:
            if arrived_but_not_completed and effective_in_radius and not dwell_satisfied:
                phase = "COMPLETE_SUBGOAL"
                waiting_on_completion = True
                waiting_checkpoint_id = str(subgoal).upper()
                completion_monitor_status = "waiting_event"
                self._emit_agent_message(
                    f"[RESULT] dwell in progress: {str(subgoal).upper()} ({dwell_seconds:.2f} / {required_dwell_seconds:.2f} s)"
                )
            elif arrived_but_not_completed and (not effective_in_radius):
                phase = "APPROACH_SUBGOAL"
                reached_flag = False
                arrived_but_not_completed = False
                arrived_wait_cycles = 0
                waiting_on_completion = False
                waiting_checkpoint_id = None
                completion_monitor_status = "left_area"
                self._emit_agent_message(
                    f"[RESULT] left checkpoint area before dwell completion: {str(subgoal).upper()}"
                )
            elif arrived_but_not_completed and dwell_satisfied:
                phase = "VERIFY_COMPLETE"
                waiting_on_completion = True
                waiting_checkpoint_id = str(subgoal).upper()
                completion_monitor_status = "waiting_completion_sync"
                self._emit_agent_message(
                    f"[RESULT] dwell satisfied for {str(subgoal).upper()}, waiting completion state sync."
                )
            else:
                phase = "APPROACH_SUBGOAL"
                reached_flag = False
                arrived_but_not_completed = False
                arrived_wait_cycles = 0
                waiting_on_completion = False
                waiting_checkpoint_id = None
                completion_monitor_status = "approach_reset"
                self._emit_agent_message(
                    f"[RESULT] verify incomplete for {str(subgoal).upper()}, re-approach subgoal."
                )
        progress_signature = f"{subgoal}:{','.join(remaining)}"
        last_subgoal_distance = state.get("last_subgoal_distance_m")
        if subgoal in remaining:
            snapshot = state.get("latest_snapshot") or {}
            drone_xy = None
            if isinstance(snapshot, dict):
                drone_gt = snapshot.get("drone_gt")
                if drone_gt is not None:
                    drone_xy = (float(drone_gt[0]), float(drone_gt[1]))
            checkpoint = BENCHMARK_CHECKPOINTS_BY_ID.get(str(subgoal))
            if checkpoint is not None and drone_xy is not None:
                distance_m = math.hypot(drone_xy[0] - float(checkpoint.x), drone_xy[1] - float(checkpoint.y))
                last_subgoal_distance = float(distance_m)

        if not remaining:
            current_incomplete = bool(
                subgoal is not None
                and str(subgoal).upper() not in completed
                and (arrived_but_not_completed or waiting_on_completion or (not dwell_satisfied))
            )
            if current_incomplete:
                self._emit_agent_message(
                    f"[EVENT] final_subgoal_incomplete:{str(subgoal).upper()} -> continue_until_completion_event"
                )
                return {
                    "agent_step_count": step_count,
                    "replan_count": replan_count,
                    "execution_history": history,
                    "mission_status": "running",
                    "route_decision": "continue",
                    "remaining_checkpoint_ids": [str(subgoal).upper()],
                    "subgoal_queue": [str(subgoal).upper()],
                    "waiting_on_checkpoint_completion": bool(waiting_on_completion),
                    "waiting_checkpoint_id": waiting_checkpoint_id,
                    "completion_monitor_status": completion_monitor_status,
                    "arrived_but_not_completed": True,
                    "subgoal_phase": ("APPROACH_SUBGOAL" if subgoal is None else phase),
                }
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "completed",
                "route_decision": "end",
                "completed_checkpoint_ids": sorted(completed),
                "remaining_checkpoint_ids": remaining,
                "current_subgoal_id": subgoal,
                "subgoal_phase": ("APPROACH_SUBGOAL" if subgoal is None else phase),
                "subgoal_reached": reached_flag,
                "arrived_but_not_completed": arrived_but_not_completed,
                "arrived_wait_cycles": arrived_wait_cycles,
                "is_current_subgoal_completed": bool(subgoal is not None and str(subgoal).upper() in completed),
                "is_current_subgoal_in_radius": bool(effective_in_radius),
                "current_subgoal_dwell_seconds": float(dwell_seconds),
                "required_dwell_seconds": float(required_dwell_seconds),
                "dwell_satisfied": bool(dwell_satisfied),
                "waiting_on_checkpoint_completion": bool(waiting_on_completion),
                "waiting_checkpoint_id": waiting_checkpoint_id,
                "last_progress_event": (None if progress_event is None else dict(progress_event)),
                "completion_monitor_status": completion_monitor_status,
                "last_progress_signature": progress_signature,
                "last_subgoal_distance_m": last_subgoal_distance,
                "recovery_mode": bool(recovery_mode),
                "recovery_reason": (None if recovery_reason is None else str(recovery_reason)),
                "recovery_entry_risk": (None if recovery_entry_risk is None else float(recovery_entry_risk)),
                "recovery_last_risk": float(recovery_last_risk),
                "recovery_just_exited": bool(recovery_just_exited),
                "last_wait_event": last_wait_event,
                "last_risk_event": (last_risk_event if last_risk_event else None),
                "latest_collision_risk": float(latest_collision_risk),
                "latest_dominant_risky_worker": state.get("dominant_risky_worker"),
                "latest_per_worker_collision_risks": dict(state.get("per_worker_collision_risks", {})),
            }
        if not bool(result.get("ok", False)):
            return {
                "agent_step_count": step_count,
                "replan_count": replan_count,
                "execution_history": history,
                "mission_status": "running",
                "route_decision": "continue",
                "completed_checkpoint_ids": sorted(completed),
                "remaining_checkpoint_ids": remaining,
                "current_subgoal_id": subgoal,
                "subgoal_phase": ("APPROACH_SUBGOAL" if subgoal is None else phase),
                "subgoal_reached": reached_flag,
                "arrived_but_not_completed": arrived_but_not_completed,
                "arrived_wait_cycles": arrived_wait_cycles,
                "is_current_subgoal_completed": bool(subgoal is not None and str(subgoal).upper() in completed),
                "is_current_subgoal_in_radius": bool(effective_in_radius),
                "current_subgoal_dwell_seconds": float(dwell_seconds),
                "required_dwell_seconds": float(required_dwell_seconds),
                "dwell_satisfied": bool(dwell_satisfied),
                "waiting_on_checkpoint_completion": bool(waiting_on_completion),
                "waiting_checkpoint_id": waiting_checkpoint_id,
                "last_progress_event": (None if progress_event is None else dict(progress_event)),
                "completion_monitor_status": completion_monitor_status,
                "last_progress_signature": progress_signature,
                "last_subgoal_distance_m": last_subgoal_distance,
                "recovery_mode": bool(recovery_mode),
                "recovery_reason": (None if recovery_reason is None else str(recovery_reason)),
                "recovery_entry_risk": (None if recovery_entry_risk is None else float(recovery_entry_risk)),
                "recovery_last_risk": float(recovery_last_risk),
                "recovery_just_exited": bool(recovery_just_exited),
                "recent_failure_reason": str(result.get("message", ""))[:160],
                "last_wait_event": last_wait_event,
                "last_risk_event": (last_risk_event if last_risk_event else None),
                "latest_collision_risk": float(latest_collision_risk),
                "latest_dominant_risky_worker": state.get("dominant_risky_worker"),
                "latest_per_worker_collision_risks": dict(state.get("per_worker_collision_risks", {})),
            }
        return {
            "agent_step_count": step_count,
            "replan_count": replan_count,
            "execution_history": history,
            "mission_status": "running",
            "route_decision": "continue",
            "completed_checkpoint_ids": sorted(completed),
            "remaining_checkpoint_ids": remaining,
            "current_subgoal_id": subgoal,
            "subgoal_phase": ("APPROACH_SUBGOAL" if subgoal is None else phase),
            "subgoal_reached": reached_flag,
            "arrived_but_not_completed": arrived_but_not_completed,
            "arrived_wait_cycles": arrived_wait_cycles,
            "is_current_subgoal_completed": bool(subgoal is not None and str(subgoal).upper() in completed),
            "is_current_subgoal_in_radius": bool(effective_in_radius),
            "current_subgoal_dwell_seconds": float(dwell_seconds),
            "required_dwell_seconds": float(required_dwell_seconds),
            "dwell_satisfied": bool(dwell_satisfied),
            "waiting_on_checkpoint_completion": bool(waiting_on_completion),
            "waiting_checkpoint_id": waiting_checkpoint_id,
            "last_progress_event": (None if progress_event is None else dict(progress_event)),
            "completion_monitor_status": completion_monitor_status,
            "last_progress_signature": progress_signature,
            "last_subgoal_distance_m": last_subgoal_distance,
            "recovery_mode": bool(recovery_mode),
            "recovery_reason": (None if recovery_reason is None else str(recovery_reason)),
            "recovery_entry_risk": (None if recovery_entry_risk is None else float(recovery_entry_risk)),
            "recovery_last_risk": float(recovery_last_risk),
            "recovery_just_exited": bool(recovery_just_exited),
            "recent_failure_reason": None,
            "last_wait_event": last_wait_event,
            "last_risk_event": (last_risk_event if last_risk_event else None),
            "latest_collision_risk": float(latest_collision_risk),
            "latest_dominant_risky_worker": state.get("dominant_risky_worker"),
            "latest_per_worker_collision_risks": dict(state.get("per_worker_collision_risks", {})),
        }

    def _route_from_evaluation(self, state: AgentState) -> RouteDecision:
        route = str(state.get("route_decision", "continue"))
        print_debug(
            "[AGENT-ROUTE] "
            f"step={int(state.get('agent_step_count', 0))} "
            f"route_decision={route} "
            f"predicted_collision_risk={float(state.get('predicted_collision_risk', 0.0)):.6f}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return route  # type: ignore[return-value]

    def _node_analyze_failure_or_progress(self, state: AgentState) -> AgentState:
        result = dict(state.get("last_action_result") or {})
        last_plan = str(state.get("last_plan_text", "")).strip()
        subgoal = None if state.get("current_subgoal_id") is None else str(state.get("current_subgoal_id")).upper()
        route_decision = str(state.get("route_decision", "continue"))
        analysis = self.controller.planner.reflect_langgraph_strategy(
            task_description=str(state.get("user_task", "")),
            current_subgoal=subgoal,
            route_decision=route_decision,
            last_action=last_plan,
            last_result=result,
            predicted_collision_risk=float(state.get("predicted_collision_risk", 0.0)),
            per_worker_collision_risks=dict(state.get("per_worker_collision_risks", {})),
            dominant_risky_worker=(None if state.get("dominant_risky_worker") is None else str(state.get("dominant_risky_worker"))),
            worker_states_summary=list(state.get("worker_states", [])),
            remaining_checkpoints=list(state.get("remaining_checkpoint_ids", [])),
            subgoal_queue=list(state.get("subgoal_queue", [])),
            current_strategy_summary=str(state.get("current_strategy_summary", "")),
            last_failure_reason=(None if state.get("last_failure_reason") is None else str(state.get("last_failure_reason"))),
            recent_failed_approach_pattern=(None if state.get("recent_failed_approach_pattern") is None else str(state.get("recent_failed_approach_pattern"))),
            recent_recovery_hypothesis=(None if state.get("recent_recovery_hypothesis") is None else str(state.get("recent_recovery_hypothesis"))),
            blocked_workers_by_subgoal=dict(state.get("blocked_workers_by_subgoal", {})),
            subgoal_attempt_history=dict(state.get("subgoal_attempt_history", {})),
            recent_history=list(state.get("execution_history", []))[-6:],
        )
        should_emit = route_decision in {"retry_plan", "reselect_subgoal"} or (not bool(result.get("ok", True)))
        if should_emit and analysis.get("failure_reason"):
            self._emit_agent_message(
                f"[ANALYZE] subgoal={subgoal or 'none'} failure_reason={analysis.get('failure_reason')} "
                f"new_strategy={analysis.get('strategy_summary')}"
            )
        return {"pending_strategy_update": analysis}

    def _node_update_strategy_memory(self, state: AgentState) -> AgentState:
        pending = dict(state.get("pending_strategy_update") or {})
        route_decision = str(state.get("route_decision", "continue"))
        result_ok = bool((state.get("last_action_result") or {}).get("ok", True))
        allow_reprioritize = route_decision in {"retry_plan", "reselect_subgoal"} or (not result_ok)
        subgoal = None if state.get("current_subgoal_id") is None else str(state.get("current_subgoal_id")).upper()
        remaining = [str(v).upper() for v in list(state.get("remaining_checkpoint_ids", []))]
        queue = [str(v).upper() for v in list(state.get("subgoal_queue", []))]
        blocked_workers = dict(state.get("blocked_workers_by_subgoal", {}))
        attempt_history = dict(state.get("subgoal_attempt_history", {}))

        action_signature = self._action_signature(str(state.get("last_action_text", "")))
        if subgoal is not None and action_signature:
            history_items = [str(v) for v in list(attempt_history.get(subgoal, []))]
            if action_signature not in history_items:
                history_items.append(action_signature)
            attempt_history[subgoal] = history_items[-8:]

        dominant_worker = state.get("dominant_risky_worker")
        if subgoal is not None and route_decision in {"retry_plan", "reselect_subgoal"} and dominant_worker is not None:
            workers = [str(v) for v in list(blocked_workers.get(subgoal, []))]
            wid = str(dominant_worker)
            if wid and wid not in workers and wid != "None":
                workers.append(wid)
            blocked_workers[subgoal] = workers[-4:]

        failure_reason = pending.get("failure_reason")
        failed_pattern = pending.get("failed_approach_pattern") or (action_signature if route_decision in {"retry_plan", "reselect_subgoal"} else None)
        recovery_hypothesis = pending.get("recovery_hypothesis")
        strategy_summary = pending.get("strategy_summary") or state.get("current_strategy_summary", "")

        new_queue = list(queue) if queue else list(remaining)
        if allow_reprioritize:
            raw_reordered = pending.get("reprioritized_subgoals")
            if isinstance(raw_reordered, list):
                reordered = [str(v).upper() for v in raw_reordered if str(v).upper() in remaining]
                carry = [cid for cid in remaining if cid not in reordered]
                if reordered:
                    new_queue = reordered + carry

            target_subgoal = pending.get("next_subgoal")
            if isinstance(target_subgoal, str):
                target_subgoal = str(target_subgoal).upper()
                if target_subgoal in remaining:
                    new_queue = [target_subgoal] + [cid for cid in new_queue if cid != target_subgoal]

        next_route = str(pending.get("next_route_decision", route_decision)) if allow_reprioritize else route_decision
        if next_route not in {"continue", "retry_plan", "reselect_subgoal", "end", "abort"}:
            next_route = route_decision
        if next_route == "reselect_subgoal" and len(new_queue) > 1:
            next_subgoal = new_queue[0]
            if subgoal == next_subgoal:
                new_queue = new_queue[1:] + [next_subgoal]
            subgoal = None
        elif next_route == "retry_plan":
            next_route = "continue"

        return {
            "current_strategy_summary": str(strategy_summary),
            "last_failure_reason": (None if failure_reason is None else str(failure_reason)),
            "recent_failed_approach_pattern": (None if failed_pattern is None else str(failed_pattern)),
            "recent_recovery_hypothesis": (None if recovery_hypothesis is None else str(recovery_hypothesis)),
            "blocked_workers_by_subgoal": blocked_workers,
            "subgoal_attempt_history": attempt_history,
            "latest_failure_analysis": pending,
            "pending_strategy_update": None,
            "subgoal_queue": new_queue,
            "current_subgoal_id": subgoal,
            "route_decision": next_route,
        }

    def _emit_agent_message(self, message: str):
        if hasattr(self.controller, "append_message"):
            self.controller.append_message(message)

    def _extract_checkpoint_target(self, action_text: str) -> str | None:
        text = str(action_text or "")
        m = re.search(r"(?:go_checkpoint|gc)\(\s*['\"]?([A-Za-z]\d+)['\"]?\s*\)", text)
        if not m:
            return None
        return str(m.group(1)).strip().upper()

    def _action_signature(self, action_text: str) -> str:
        text = str(action_text or "").strip().lower()
        if not text:
            return ""
        if text.startswith("go_checkpoint(") or text.startswith("gc("):
            return "go_checkpoint"
        if "(" in text:
            return text.split("(", 1)[0].strip()
        return text[:24]
