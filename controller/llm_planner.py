import os, ast, re, json, math
from typing import Optional

from .safety_context import SafetyContext
from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, GPT3, GPT4, chat_log_path
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_debug, print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType
from .benchmark_layout import CHECKPOINT_DWELL_SECONDS, CHECKPOINT_RADIUS_M, UAV_RADIUS_M, WORKER_RADIUS_M
from .task_run_logger import resolve_archive_root_and_excel_path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLISION_PROBABILITY_REPLAN_THRESHOLD = 0.70

class LLMPlanner():
    def __init__(self, robot_type: RobotType):
        self.llm = LLMWrapper()
        self.model_name = GPT4
        heartbeat_model_env = str(os.getenv("TYPEFLY_HEARTBEAT_MODEL", "") or "").strip()
        evaluator_model_env = str(os.getenv("TYPEFLY_EVALUATOR_MODEL", "") or "").strip()
        self._heartbeat_model_explicit = bool(heartbeat_model_env)
        self._evaluator_model_explicit = bool(evaluator_model_env)
        self.heartbeat_model_name = (heartbeat_model_env if heartbeat_model_env else self.model_name)
        self.evaluator_model_name = (evaluator_model_env if evaluator_model_env else self.model_name)
        self.controller = None  # 後續由 controller.llm_controller 綁定

        type_folder_name = 'tello'
        if robot_type == RobotType.GEAR:
            type_folder_name = 'gear'

        # read prompt from txt
        self.prompt_plan_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan.txt")
        self.prompt_plan_initial_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan_initial.txt")
        self.prompt_plan_replan_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan_replan.txt")
        self.prompt_probe_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe.txt")
        self.guides_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/guides.txt")
        self.typefly_initial_examples_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/typefly_initial_examples.txt")
        self.typefly_replan_examples_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/typefly_replan_examples.txt")
        self.agent_decomposition_examples_path = None
        self.agent_mode_action_examples_path = None
        self.agent_reflection_examples_path = None
        self.agent_decomposition_examples = ""
        self.agent_mode_action_examples = ""
        self.agent_reflection_examples = ""
        self.agent_heartbeat_soft_prompt_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/agent_heartbeat_soft_prompt.txt")
        self.agent_heartbeat_hardgate_prompt_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/agent_heartbeat_hardgate_prompt.txt")
        self.agent_heartbeat_soft_examples_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/agent_heartbeat_soft_examples.txt")
        self.agent_heartbeat_hardgate_examples_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/agent_heartbeat_hardgate_examples.txt")
        self.agent_evaluator_prompt_path = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/agent_evaluator_prompt.txt")
        self.runtime_prompt_variant = "default"
        self.runtime_example_variant = "default"
        self.runtime_use_output_example = True
        self._last_plan_trace = {}
        self._last_heartbeat_trace = {}
        with open(self.prompt_plan_path, "r") as f:
            self.prompt_plan = f.read()

        with open(self.prompt_probe_path, "r") as f:
            self.prompt_probe = f.read()

        with open(self.guides_path, "r") as f:
            self.guides = f.read()

        with open(self.typefly_initial_examples_path, "r") as f:
            self.typefly_initial_examples = f.read()
        with open(self.typefly_replan_examples_path, "r") as f:
            self.typefly_replan_examples = f.read()
        with open(self.agent_heartbeat_soft_prompt_path, "r") as f:
            self.agent_heartbeat_soft_prompt = f.read()
        with open(self.agent_heartbeat_hardgate_prompt_path, "r") as f:
            self.agent_heartbeat_hardgate_prompt = f.read()
        with open(self.agent_heartbeat_soft_examples_path, "r") as f:
            self.agent_heartbeat_soft_examples = f.read()
        with open(self.agent_heartbeat_hardgate_examples_path, "r") as f:
            self.agent_heartbeat_hardgate_examples = f.read()
        if os.path.exists(self.agent_evaluator_prompt_path):
            with open(self.agent_evaluator_prompt_path, "r") as f:
                self.agent_evaluator_prompt = f.read()
        else:
            self.agent_evaluator_prompt = ""
        with open(self.prompt_plan_initial_path, "r") as f:
            self.prompt_plan_initial = f.read()
        with open(self.prompt_plan_replan_path, "r") as f:
            self.prompt_plan_replan = f.read()
        self.baseline_context_assets = self._build_baseline_context_assets(type_folder_name)
        self.prompt_variant_assets = self._build_prompt_variant_assets(type_folder_name)
        self.example_variant_assets = self._build_example_variant_assets(type_folder_name)
        self.prompt_langgraph_decomposition = (
            "{shared_opening_block}\n\n"
            "{shared_runtime_context_block}\n\n"
            "You are the Agent Task Decomposer. This stage does NOT generate a full MiniSpec program.\n"
            "It only outputs a checkpoint visiting order for unfinished checkpoints.\n"
            "Output JSON array only (checkpoint IDs in order), with no explanation.\n"
            "Ordering rules:\n"
            "- Sort unfinished checkpoints by safety + efficiency jointly.\n"
            "- Safety depends on UAV position, UAV heading, worker positions, UAV/worker/checkpoint radii, checkpoint geometry, and collision probabilities.\n"
            "- Efficiency depends on whether approach from current UAV position + heading is smooth and avoids unnecessary detours.\n"
            "- Do not use fixed lexical order like A1->A2->A3->A4.\n"
            "- If a checkpoint is temporarily unfavorable (e.g., behind a worker, too close to a worker, or inefficient from current heading), defer it.\n"
            "Task: {task_description}\n"
            "Allowed checkpoints: {allowed_checkpoints}\n"
            "Active zones: {active_zones}\n\n"
            "Agent decomposition examples:\n"
            "{agent_decomposition_examples}\n\n"
            "Output JSON array only."
        )
        self.prompt_langgraph_step = (
            "{shared_opening_block}\n\n"
            "{shared_runtime_context_block}\n\n"
            "You are the Agent Step Planner. You are NOT planning a full mission. You are deciding only the NEXT step.\n"
            "Output one action or a very short action segment, each statement ending with ';'.\n"
            "Allowed actions (abbreviation form only):\n"
            "gc('ID'); mf(value); mb(value); ml(value); mr(value); tc(value); tu(value); d(value); lo('text');\n"
            "Hard constraints:\n"
            "- Do not output multi-checkpoint full plans.\n"
            "- current_subgoal is the primary mission target for this step.\n"
            "- If the same action was already executed and no progress improved, avoid repeating it.\n"
            "- If the current situation appears risky, you may temporarily choose conservative avoidance or recovery action.\n"
            "- Avoidance is temporary, not a new mission objective.\n"
            "- Recovery mode is controlled by system state; follow it explicitly.\n"
            "- If recovery_mode is true, prioritize conservative recovery movement and re-observation over direct checkpoint approach.\n"
            "- If recovery_mode is false, prioritize safe mission progress toward current_subgoal.\n"
            "- If current_subgoal appears risky, do not blindly rush gc(current_subgoal).\n"
            "- Recovery should be step-by-step: after each recovery step, re-observe latest risk/state before deciding the next step.\n"
            "- During each re-observation, explicitly reassess current collision risk and geometry.\n"
            "- When you judge that immediate risk has improved to a safe level, return to checkpoint progress.\n"
            "- If risk still appears high after re-observation, choose another conservative recovery step and re-observe again.\n"
            "- Recovery goal is to leave danger and then resume current_subgoal, not endless wandering.\n"
            "- If stalled or no-progress, switch to a different conservative strategy.\n"
            "- Use strategy memory to avoid repeating known failed approach patterns for this subgoal.\n"
            "- Avoid long turn or move loops without objective progress.\n"
            "- Reaching checkpoint area is NOT the same as completing checkpoint.\n"
            "- Completion is determined only by official completion_state and progress.\n"
            "- If reached but not completed, prioritize finishing current_subgoal (for example hold, re-approach, or micro-adjust), not jumping to the next checkpoint.\n"
            "Task: {task_description}\n"
            "Current subgoal: {current_subgoal}\n"
            "Remaining checkpoints: {remaining_checkpoints}\n"
            "Predicted collision risk: {predicted_collision_risk}\n"
            ""
            "Dominant risky worker: {dominant_risky_worker}\n"
            "Per-worker collision risks: {per_worker_collision_risks}\n"
            "Recovery mode: {recovery_mode}\n"
            "Recovery reason: {recovery_reason}\n"
            "Strategy summary: {strategy_summary}\n"
            "Last failure reason: {last_failure_reason}\n"
            "Recent failed approach pattern: {failed_approach_pattern}\n"
            "Recent recovery hypothesis: {recovery_hypothesis}\n"
            "Blocked workers for current subgoal: {blocked_workers_for_subgoal}\n"
            "Attempt history for current subgoal: {subgoal_attempts}\n"
            "Worker states summary: {worker_states_summary}\n"
            "Last action: {last_action}\n"
            "Last result: {last_result}\n"
            "Recent execution history: {recent_history}\n"
            "Return action statements only, no explanation."
        )
        self.prompt_langgraph_mode_action = (
            "{shared_opening_block}\n\n"
            "{shared_runtime_context_block}\n\n"
            "You are the Agent Mode+Action Decision module.\n"
            "This stage does NOT create a full mission plan from zero. It decides only the next short action segment.\n"
            "Decide BOTH the mode and action for this step.\n"
            "Output JSON only with keys: mode, reason, strategy_summary, action, next_subgoal, why_action_matches_mode.\n"
            "mode must be one of: approach, recovery, replan, skip_current_subgoal.\n"
            "next_subgoal default is null.\n"
            "Rules:\n"
            "- Decide mode from latest runtime state; do not mechanically repeat previous mode.\n"
            "- action may be a short MiniSpec segment (1~4 statements), not only one skill call.\n"
            "- Danger judgment cannot rely on a single hard threshold.\n"
            "- Evaluate risk using geometry + probability jointly (collision probability + UAV heading + UAV-worker geometry + radii + immediate approach corridor).\n"
            "- If probability is low but geometry is too close, still treat as risky.\n"
            "- Match avoidance amplitude to risk severity: high risk -> larger conservative detour; low-but-suspicious risk -> small preventive detour.\n"
            "- gc() is convenient but path details are not fully controllable; if direct gc() looks unsafe, use heading-aware low-level shaping first.\n"
            "- mode-action alignment is strict:\n"
            "  - If mode = recovery: action MUST start with one or more recovery-style repositioning steps; do not start with gc(current_subgoal).\n"
            "  - If mode = approach: prioritize mission progress and may include short supporting moves before/after gc(current_subgoal).\n"
            "  - If mode = skip_current_subgoal: action should progress toward the newly chosen checkpoint, and next_subgoal MUST be provided and MUST differ from current_subgoal.\n"
            "  - If mode = replan: this means strategy-level change (route/order/tactic redesign), not just one-step local recovery.\n"
            "- Distinguish skip_current_subgoal vs replan:\n"
            "  - skip_current_subgoal: temporary checkpoint switch within remaining checkpoints; must nominate a different next_subgoal now.\n"
            "  - replan: requires broader strategy update (e.g., resequencing or route strategy change), beyond a temporary one-step detour.\n"
            "- Checkpoint ordering is dynamic, not fixed: never follow A1->A2->A3->A4 mechanically.\n"
            "- If current_subgoal is temporarily high-risk/inefficient, prefer skip_current_subgoal with a safer and shorter next_subgoal.\n"
            "- If action intends to complete a checkpoint in this step, include d(2.0).\n"
            "- Mode semantics:\n"
            "  - approach: efficient progress toward checkpoint objective under acceptable geometry.\n"
            "  - recovery: local risk-reduction maneuver before resuming approach.\n"
            "  - replan: broader strategy-level change (ordering/tactic redesign).\n"
            "  - skip_current_subgoal: temporary target switch to a safer or more efficient unfinished checkpoint.\n"
            "- Output format strictness:\n"
            "  - reason: one short concrete sentence.\n"
            "  - strategy_summary: one short concrete sentence.\n"
            "  - action: a short MiniSpec segment (1~4 statements), each statement ending with ';'.\n"
            "  - why_action_matches_mode: one short sentence explicitly proving action and mode are consistent.\n"
            "Executable action grammar (abbreviation form required):\n"
            "- mf(value); mb(value); ml(value); mr(value);\n"
            "- tc(value); tu(value); d(value); gc('A1'); lo('short_text');\n"
            "Task: {task_description}\n"
            "Current mode: {current_mode}\n"
            "Current subgoal: {current_subgoal}\n"
            "Remaining checkpoints: {remaining_checkpoints}\n"
            "Predicted collision risk: {predicted_collision_risk}\n"
            ""
            "Dominant risky worker: {dominant_risky_worker}\n"
            "Per-worker collision risks: {per_worker_collision_risks}\n"
            "Worker states summary: {worker_states_summary}\n"
            "Last action: {last_action}\n"
            "Last result: {last_result}\n"
            "Recent failure reason: {recent_failure_reason}\n"
            "Last wait/monitor event: {last_wait_event}\n"
            "Last risk event: {last_risk_event}\n"
            "Strategy summary: {strategy_summary}\n"
            "Recent failed approach pattern: {failed_approach_pattern}\n"
            "Recent recovery hypothesis: {recovery_hypothesis}\n"
            "Blocked workers for current subgoal: {blocked_workers_for_subgoal}\n"
            "Attempt history for current subgoal: {subgoal_attempts}\n"
            "Recent execution history: {recent_history}\n"
            "Agent mode+action examples:\n"
            "{agent_mode_action_examples}\n"
            "Output JSON only."
        )
        self.prompt_langgraph_reflection = (
            "{shared_opening_block}\n\n"
            "{shared_runtime_context_block}\n\n"
            "You are the Agent Strategy Reflector.\n"
            "This stage does NOT output control actions; it reflects on the previous step and updates strategy memory.\n"
            "Analyze last step outcome and decide strategy-level correction.\n"
            "Return JSON only with keys:\n"
            "- failure_reason: short string\n"
            "- failed_approach_pattern: short string for what should NOT be repeated now\n"
            "- recovery_hypothesis: short string describing safer next tactic\n"
            "- strategy_summary: short string for ongoing strategy memory\n"
            "- next_route_decision: one of [continue, retry_plan, reselect_subgoal]\n"
            "- next_subgoal: checkpoint id or null\n"
            "- reprioritized_subgoals: ordered checkpoint list or []\n"
            "Rules:\n"
            "- If last step failed or risk_abort happened, explain why and avoid repeating same approach.\n"
            "- You may reprioritize checkpoints when current subgoal is temporarily high-risk or blocked.\n"
            "- Reprioritization must be temporary and mission-oriented.\n"
            "- Prioritize safe + efficient ordering; do not keep a fixed lexical checkpoint order.\n"
            "- Consider collision probability, UAV heading, worker geometry, checkpoint geometry, and whether recent motion formed a safe and efficient approach corridor.\n"
            "- If current subgoal is temporarily unsafe, geometrically awkward, or heading-inefficient, suggest reprioritizing unfinished checkpoints.\n"
            "- Explicitly encode whether risk was high-conservative-detour case or low-risk-small-avoidance case.\n"
            "- Keep every field concise.\n"
            "Task: {task_description}\n"
            "Current subgoal: {current_subgoal}\n"
            "Route decision from evaluator: {route_decision}\n"
            "Last action: {last_action}\n"
            "Last result: {last_result}\n"
            "Predicted collision risk: {predicted_collision_risk}\n"
            "Dominant risky worker: {dominant_risky_worker}\n"
            "Per-worker collision risks: {per_worker_collision_risks}\n"
            "Worker states summary: {worker_states_summary}\n"
            "Remaining checkpoints: {remaining_checkpoints}\n"
            "Current queue: {subgoal_queue}\n"
            "Current strategy summary: {current_strategy_summary}\n"
            "Last failure reason memory: {last_failure_reason}\n"
            "Recent failed approach memory: {recent_failed_approach_pattern}\n"
            "Recent recovery hypothesis memory: {recent_recovery_hypothesis}\n"
            "Blocked workers by subgoal: {blocked_workers_by_subgoal}\n"
            "Subgoal attempt history: {subgoal_attempt_history}\n"
            "Recent execution history: {recent_history}\n"
            "Agent reflection examples:\n"
            "{agent_reflection_examples}\n"
            "Output JSON only."
        )
    def set_model(self, model_name):
        old_default = self.model_name
        self.model_name = model_name
        if not self._heartbeat_model_explicit and self.heartbeat_model_name == old_default:
            self.heartbeat_model_name = self.model_name
        if not self._evaluator_model_explicit and self.evaluator_model_name == old_default:
            self.evaluator_model_name = self.model_name

    def set_agent_model_names(
        self,
        heartbeat_model_name: Optional[str] = None,
        evaluator_model_name: Optional[str] = None,
    ):
        hb = str(heartbeat_model_name or "").strip()
        ev = str(evaluator_model_name or "").strip()
        if hb:
            self.heartbeat_model_name = hb
            self._heartbeat_model_explicit = True
        else:
            self.heartbeat_model_name = self.model_name
            self._heartbeat_model_explicit = False
        if ev:
            self.evaluator_model_name = ev
            self._evaluator_model_explicit = True
        else:
            self.evaluator_model_name = self.model_name
            self._evaluator_model_explicit = False

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def _build_prompt_variant_assets(self, type_folder_name: str) -> dict:
        base_dir = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}")
        variants = {
            "default": {
                "plan_initial_prompt": self.prompt_plan_initial,
                "plan_replan_prompt": self.prompt_plan_replan,
                "heartbeat_soft_prompt": self.agent_heartbeat_soft_prompt,
                "heartbeat_hardgate_prompt": self.agent_heartbeat_hardgate_prompt,
            }
        }
        for baseline_id in ("baseline1", "baseline2", "baseline3", "agent"):
            key = f"{baseline_id}_prompt"
            init_path = os.path.join(base_dir, f"{baseline_id}_prompt_plan_initial.txt")
            replan_path = os.path.join(base_dir, f"{baseline_id}_prompt_plan_replan.txt")
            hb_soft_path = os.path.join(base_dir, f"{baseline_id}_prompt_heartbeat_soft.txt")
            payload = dict(variants["default"])
            if baseline_id in {"baseline1", "baseline2", "agent"}:
                if not (os.path.exists(init_path) and os.path.exists(hb_soft_path)):
                    variants[key] = payload
                    continue
                payload["plan_initial_prompt"] = self._read_text(init_path)
                if baseline_id in {"baseline1", "baseline2"}:
                    # baseline1/2 do not keep a dedicated replan-plan prompt; reuse initial prompt.
                    payload["plan_replan_prompt"] = payload["plan_initial_prompt"]
                else:
                    payload["plan_replan_prompt"] = self._read_text(replan_path) if os.path.exists(replan_path) else payload["plan_initial_prompt"]
                payload["heartbeat_soft_prompt"] = self._read_text(hb_soft_path)
            elif baseline_id == "baseline3":
                if not (os.path.exists(init_path) and os.path.exists(replan_path)):
                    variants[key] = payload
                    continue
                payload["plan_initial_prompt"] = self._read_text(init_path)
                payload["plan_replan_prompt"] = self._read_text(replan_path)
                if os.path.exists(hb_soft_path):
                    payload["heartbeat_soft_prompt"] = self._read_text(hb_soft_path)
            variants[key] = payload
        return variants

    def _build_example_variant_assets(self, type_folder_name: str) -> dict:
        base_dir = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}")
        variants = {
            "default": {
                "initial_examples": self.typefly_initial_examples,
                "replan_examples": self.typefly_replan_examples,
                "heartbeat_soft_examples": self.agent_heartbeat_soft_examples,
                "heartbeat_hardgate_examples": self.agent_heartbeat_hardgate_examples,
            }
        }
        for baseline_id in ("baseline1", "baseline2", "baseline3", "agent"):
            key = f"{baseline_id}_example"
            init_ex_path = os.path.join(base_dir, f"{baseline_id}_example_initial.txt")
            replan_ex_path = os.path.join(base_dir, f"{baseline_id}_example_replan.txt")
            hb_soft_ex_path = os.path.join(base_dir, f"{baseline_id}_example_heartbeat_soft.txt")
            payload = dict(variants["default"])
            if baseline_id in {"baseline1", "baseline2", "agent"}:
                if not (os.path.exists(init_ex_path) and os.path.exists(hb_soft_ex_path)):
                    variants[key] = payload
                    continue
                payload["initial_examples"] = self._read_text(init_ex_path)
                if baseline_id in {"baseline1", "baseline2"}:
                    # baseline1/2 do not keep dedicated replan examples; reuse initial examples.
                    payload["replan_examples"] = payload["initial_examples"]
                else:
                    payload["replan_examples"] = self._read_text(replan_ex_path) if os.path.exists(replan_ex_path) else payload["initial_examples"]
                payload["heartbeat_soft_examples"] = self._read_text(hb_soft_ex_path)
            elif baseline_id == "baseline3":
                if not (os.path.exists(init_ex_path) and os.path.exists(replan_ex_path)):
                    variants[key] = payload
                    continue
                payload["initial_examples"] = self._read_text(init_ex_path)
                payload["replan_examples"] = self._read_text(replan_ex_path)
                if os.path.exists(hb_soft_ex_path):
                    payload["heartbeat_soft_examples"] = self._read_text(hb_soft_ex_path)
            variants[key] = payload
        return variants

    def _build_baseline_context_assets(self, type_folder_name: str) -> dict:
        base_dir = os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}")
        default = {
            "opening_block": "",
            "runtime_context_block": "",
            "guides": self.guides,
        }
        assets = {"default": default}
        for baseline_id in ("baseline1", "baseline2", "baseline3"):
            payload = dict(default)
            opening_path = os.path.join(base_dir, f"{baseline_id}_opening_block.txt")
            runtime_path = os.path.join(base_dir, f"{baseline_id}_runtime_context_block.txt")
            guides_path = os.path.join(base_dir, f"{baseline_id}_guides.txt")
            if os.path.exists(opening_path):
                payload["opening_block"] = self._read_text(opening_path)
            if os.path.exists(runtime_path):
                payload["runtime_context_block"] = self._read_text(runtime_path)
            if os.path.exists(guides_path):
                payload["guides"] = self._read_text(guides_path)
            assets[baseline_id] = payload
        return assets

    def _get_prompt_variant_payload(self) -> dict:
        return dict(self.prompt_variant_assets.get(self.runtime_prompt_variant) or self.prompt_variant_assets["default"])

    def _get_example_variant_payload(self) -> dict:
        return dict(self.example_variant_assets.get(self.runtime_example_variant) or self.example_variant_assets["default"])

    def set_runtime_prompt_example_variant(
        self,
        *,
        prompt_variant: str,
        example_variant: str,
        use_output_example: bool,
    ):
        self.runtime_prompt_variant = str(prompt_variant or "default")
        self.runtime_example_variant = str(example_variant or "default")
        self.runtime_use_output_example = bool(use_output_example)

    def get_last_plan_trace(self) -> dict:
        return dict(self._last_plan_trace or {})

    def get_last_heartbeat_trace(self) -> dict:
        return dict(self._last_heartbeat_trace or {})

    def init(self, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def _fmt_xyz(self, value) -> str:
        if value is None:
            return "(n/a)"
        try:
            x, y, z = value
            return f"({float(x):.2f}, {float(y):.2f}, {float(z):.2f})"
        except Exception:
            return "(n/a)"

    def _build_shared_opening_block(self) -> str:
        return (
            "You are an autonomous UAV mission planning and control agent operating in a structured benchmark scene that may be divided into multiple zones such as zone_A, zone_B, and zone_C. "
            "Each zone contains multiple checkpoints that define the search coverage required in that zone. "
            "The environment contains one UAV and three workers, and workers may be static or moving. "
            "Your job is to use only the provided skills to control the UAV and complete one or more zone-search tasks.\n\n"
            "A checkpoint is not completed merely because the UAV passes nearby. "
            f"A checkpoint is completed only when the UAV true position stays continuously inside that checkpoint radius for {float(CHECKPOINT_DWELL_SECONDS):.1f} seconds. "
            f"If the UAV leaves the checkpoint region before the full {float(CHECKPOINT_DWELL_SECONDS):.1f} seconds are accumulated, the dwell timer resets. "
            "A zone is completed only when all active checkpoints in that zone are completed. "
            "Completed checkpoints must never be redone.\n\n"
            "Your planning must always balance safety and efficiency. Safety means that the UAV must not collide with any worker. "
            "To help you reason about safety, you are given UAV position, UAV heading, worker positions, geometry sizes, and collision probability information. "
            "You must use geometry and collision probability jointly. Do not rely on probability alone. "
            "Even when collision probability is low, if the UAV is geometrically too close to a worker after considering both sizes, you must still treat the situation as risky.\n\n"
            "Efficiency means minimizing unnecessary detours and mission completion time while maintaining zero collisions. "
            "You can improve efficiency by choosing a smoother and shorter checkpoint order from the current UAV position and heading. "
            "You must not mechanically follow a fixed lexical order such as A1 -> A2 -> A3 -> A4. "
            "Instead, you should choose the order that is safer, smoother, and more efficient for the current geometry.\n\n"
            "The available movement skills include low-level body-frame actions such as forward, backward, left, right, and turning, as well as checkpoint navigation through gc(). "
            "Low-level movement and turning are not only for emergencies. "
            "You may also use them proactively to shape a safer and more controllable approach corridor toward a checkpoint. "
            "This matters because gc() is convenient but does not expose detailed path control. "
            "In practice, gc() usually aligns toward the target checkpoint and moves approximately straight. "
            "Therefore, if direct gc() appears risky because of worker geometry, you may first use heading-aware low-level motion and turning to create a safer approach, and then continue toward the checkpoint.\n\n"
            "When choosing an avoidance maneuver, always consider UAV heading explicitly. "
            "The body-frame motions mf, mb, ml, and mr are all defined relative to the current UAV heading. "
            "Therefore, the correct way to detour around a worker depends not only on positions, but also on current heading. "
            "If the risk appears mild but suspicious, a small preventive detour may be enough. "
            "If the risk appears severe or geometry is very close, you should prefer a larger and more conservative detour. "
            "The overall goal is to complete the mission with no collisions while keeping mission time as low as possible."
        )

    def _get_baseline_id_from_variant(self) -> str:
        variant = str(self.runtime_prompt_variant or "")
        if variant.startswith("baseline1_"):
            return "baseline1"
        if variant.startswith("baseline2_"):
            return "baseline2"
        if variant.startswith("baseline3_"):
            return "baseline3"
        if variant.startswith("agent_"):
            return "baseline2"
        return "default"

    def _build_opening_block(self) -> str:
        baseline_id = self._get_baseline_id_from_variant()
        payload = self.baseline_context_assets.get(baseline_id) or self.baseline_context_assets["default"]
        text = str(payload.get("opening_block") or "").strip()
        if text:
            return text
        return self._build_shared_opening_block()

    def _build_shared_runtime_context_block(
        self,
        safety_context: Optional[SafetyContext],
        *,
        snapshot: Optional[dict] = None,
    ) -> str:
        if snapshot is None:
            snapshot = {}
            if self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot"):
                try:
                    snapshot = self.controller.get_live_ui_snapshot() or {}
                except Exception:
                    snapshot = {}
        drone_pos = snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est") or snapshot.get("drone_gt")
        drone_yaw_deg = math.degrees(float(snapshot.get("drone_yaw_rad") or 0.0))
        workers = list(snapshot.get("workers") or [])
        workers_sorted = sorted(workers, key=lambda row: str(row.get("id", "")))
        worker_lines = []
        for idx in range(3):
            label = f"worker_{idx + 1}"
            if idx < len(workers_sorted):
                row = workers_sorted[idx]
                est_xy = row.get("est_xy_bias_corrected") or row.get("est_xy_raw")
                if est_xy is None:
                    worker_lines.append(f"- {label} bias-corrected estimated position: (n/a)")
                else:
                    worker_lines.append(f"- {label} bias-corrected estimated position: ({float(est_xy[0]):.2f}, {float(est_xy[1]):.2f}, 0.00)")
            else:
                worker_lines.append(f"- {label} bias-corrected estimated position: (n/a)")

        predicted_collision_probability = 0.0 if safety_context is None else float(safety_context.predicted_collision_probability)
        per_worker_probs = []
        if safety_context is not None:
            for row in (getattr(safety_context, "per_worker_collision_probabilities", []) or []):
                worker_id = str(row.get("id", "unknown"))
                p_val = float(row.get("collision_probability", 0.0))
                per_worker_probs.append((worker_id, p_val))
        dominant_worker = "n/a"
        if safety_context is not None:
            dominant_worker = str(getattr(safety_context, "dominant_threat_id", "n/a") or "n/a")

        objective = dict(snapshot.get("active_objective_set") or {})
        active_zone_ids = [str(v) for v in objective.get("active_zone_ids", [])]
        active_checkpoint_ids = [str(v) for v in objective.get("active_checkpoint_ids", [])]
        checkpoint_map = {
            str(row.get("id")): row
            for row in (snapshot.get("benchmark_checkpoints") or [])
            if row.get("id") is not None
        }
        checkpoint_lines = []
        for cid in active_checkpoint_ids:
            row = checkpoint_map.get(cid)
            if row is None:
                checkpoint_lines.append(f"- {cid}: (x=n/a, y=n/a)")
            else:
                checkpoint_lines.append(f"- {cid}: (x={float(row.get('x')):.2f}, y={float(row.get('y')):.2f})")
        if not checkpoint_lines:
            checkpoint_lines.append("- (n/a)")

        worker_radii_block = "\n".join(
            [f"- worker_{idx + 1}: {float(WORKER_RADIUS_M):.2f} m" for idx in range(3)]
        )
        per_worker_collision_probabilities_block = "\n".join(
            [f"- {wid}: {prob:.6f}" for wid, prob in per_worker_probs]
        ) if per_worker_probs else "- (n/a)"

        return (
            "Shared runtime context (identical skill availability for TypeFly mode and Agent mode):\n"
            "\n"
            "Skills (abbreviation required in outputs and examples):\n"
            "- gc = go_checkpoint\n"
            "- mf = move_forward\n"
            "- mb = move_backward\n"
            "- ml = move_left\n"
            "- mr = move_right\n"
            "- tc = turn_cw\n"
            "- tu = turn_ccw\n"
            "- d = delay\n"
            "- lo = log\n"
            "- TypeFly mode and Agent mode must use exactly the same available skills listed above.\n"
            "- Use only listed skills; do not invent new skills.\n"
            "- Runtime may accept full-name aliases for compatibility, but prompt/example/output style must use abbreviations.\n"
            "UAV state:\n"
            f"- UAV bias-corrected estimated position: {self._fmt_xyz(drone_pos)}\n"
            f"- UAV heading / yaw (deg): {drone_yaw_deg:.2f}\n"
            "Workers state:\n"
            + "\n".join(worker_lines)
            + "\n"
            "\n"
            "Mission structure:\n"
            f"- active zones: {active_zone_ids if active_zone_ids else ['(n/a)']}\n"
            f"- active checkpoints: {active_checkpoint_ids if active_checkpoint_ids else ['(n/a)']}\n"
            "- checkpoint coordinates:\n"
            + "\n".join(checkpoint_lines)
            + "\n"
            "Geometry information:\n"
            f"- UAV radius: {float(UAV_RADIUS_M):.2f} m\n"
            "- worker radii:\n"
            f"{worker_radii_block}\n"
            f"- checkpoint radius: {float(CHECKPOINT_RADIUS_M):.2f} m\n"
            "\n"
            "Risk context:\n"
            f"- predicted collision probability: {predicted_collision_probability:.6f}\n"
            "- per-worker collision probabilities:\n"
            f"{per_worker_collision_probabilities_block}\n"
            f"- dominant risky worker: {dominant_worker}\n"
        )

    def _build_runtime_context_block(
        self,
        safety_context: Optional[SafetyContext],
        *,
        snapshot: Optional[dict] = None,
    ) -> str:
        baseline_id = self._get_baseline_id_from_variant()
        payload = self.baseline_context_assets.get(baseline_id) or self.baseline_context_assets["default"]
        runtime_intro = str(payload.get("runtime_context_block") or "").strip()
        if baseline_id == "baseline1":
            context_block = self._build_shared_runtime_context_block_without_risk(safety_context, snapshot=snapshot)
        else:
            context_block = self._build_shared_runtime_context_block(safety_context, snapshot=snapshot)
        if runtime_intro:
            return f"{runtime_intro}\n\n{context_block}"
        return context_block

    def _build_shared_runtime_context_block_without_risk(
        self,
        safety_context: Optional[SafetyContext],
        *,
        snapshot: Optional[dict] = None,
    ) -> str:
        base_block = self._build_shared_runtime_context_block(safety_context, snapshot=snapshot)
        marker = "\nRisk context:\n"
        if marker in base_block:
            base_block = base_block.split(marker, 1)[0].rstrip()
        return base_block

    def _get_guides_text(self) -> str:
        baseline_id = self._get_baseline_id_from_variant()
        payload = self.baseline_context_assets.get(baseline_id) or self.baseline_context_assets["default"]
        text = str(payload.get("guides") or "").strip()
        return text if text else self.guides

    def _build_replan_history_block(
        self,
        task_description: str,
        previous_plan: Optional[str],
        execution_history,
        safety_context: Optional[SafetyContext],
        active_checkpoint_ids: list[str],
        benchmark_progress: Optional[dict] = None,
        trigger_threshold: Optional[float] = None,
    ) -> str:
        threshold = float(COLLISION_PROBABILITY_REPLAN_THRESHOLD if trigger_threshold is None else trigger_threshold)
        predicted_collision_probability = 0.0 if safety_context is None else float(safety_context.predicted_collision_probability)
        if predicted_collision_probability < threshold:
            return ""
        if previous_plan is None and execution_history is None:
            return ""

        progress = dict(benchmark_progress or {})
        completed = [str(v).upper() for v in list(progress.get("completed") or [])]
        remaining = [cid for cid in active_checkpoint_ids if cid not in completed]
        current_target = progress.get("current_target")
        if current_target is not None:
            current_target = str(current_target).upper()
        if not current_target:
            current_target = "(n/a)" if not remaining else remaining[0]
        mission_progress = {
            "current_target": current_target,
            "in_radius": progress.get("in_radius"),
            "dwell_seconds": progress.get("dwell_seconds"),
            "required_dwell_seconds": progress.get("required_dwell_seconds"),
            "dwell_satisfied": progress.get("dwell_satisfied"),
            "completed": completed,
        }
        dominant_worker = "n/a"
        if safety_context is not None:
            dominant_worker = str(getattr(safety_context, "dominant_threat_id", "n/a") or "n/a")

        previous_plan_text = str(previous_plan or "").strip()
        if not previous_plan_text:
            previous_plan_text = "(n/a)"

        return (
            "Replan runtime history (this call is replan, not a fresh task):\n"
            f"- original user task: {task_description}\n"
            f"- previous plan: {previous_plan_text}\n"
            f"- execution history: {execution_history if execution_history is not None else '(n/a)'}\n"
            f"- mission progress snapshot: {mission_progress}\n"
            "- completion_state_source: benchmark_progress/dwell_tracker\n"
            f"- completed checkpoints: {completed if completed else ['(none_detected)']}\n"
            f"- remaining checkpoints: {remaining if remaining else ['(none)']}\n"
            f"- current target checkpoint: {current_target}\n"
            "- replan trigger reason:\n"
            f"  - predicted collision probability > {threshold:.2f} "
            f"(current={predicted_collision_probability:.6f})\n"
            f"  - dominant risky worker = {dominant_worker}"
        )

    def plan(self, task_description: str, scene_description: Optional[str] = None, location_info: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None, safety_context: Optional[SafetyContext] = None, previous_plan: Optional[str] = None, planning_stage: str = "initial"):
    
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description
            
        
        # 自動處理 scene_description
        if scene_description is None:
            try:
                if self.vision_skill and getattr(self.vision_skill, 'enabled', True):
                    scene_description = self.vision_skill.get_obj_list()
                else:
                    scene_description = ''
            except Exception:
                scene_description = ''

        # 自動處理 location_info
        if location_info is None:
            try:
                if self.controller and hasattr(self.controller, '_format_planner_location_info'):
                    location_info = self.controller._format_planner_location_info()
            except Exception:
                location_info = None
            if location_info is None:
                drone_pos = (0.00, 0.00, 0.00)
                try:
                    if self.controller:
                        if hasattr(self.controller, 'state_provider'):
                            get_est_drone = getattr(self.controller.state_provider, 'get_estimated_drone_position', None)
                            if callable(get_est_drone):
                                value = get_est_drone()
                                if value is not None:
                                    drone_pos = value
                except Exception:
                    pass
                location_info = (
                    f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}"
                )

        full_scene = f"{scene_description}\n{location_info}".strip()
        snapshot = {}
        if self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot"):
            try:
                snapshot = self.controller.get_live_ui_snapshot() or {}
            except Exception:
                snapshot = {}
        shared_opening_block = self._build_opening_block()
        shared_runtime_context_block = self._build_runtime_context_block(
            safety_context,
            snapshot=snapshot,
        )
        objective = dict(snapshot.get("active_objective_set") or {})
        active_checkpoint_ids = [str(v) for v in objective.get("active_checkpoint_ids", [])]
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        true_completed_checkpoints = [str(v).upper() for v in list(benchmark_progress.get("completed") or [])]
        true_remaining_checkpoints = [str(cid).upper() for cid in active_checkpoint_ids if str(cid).upper() not in set(true_completed_checkpoints)]
        current_target_checkpoint = benchmark_progress.get("current_target") if isinstance(benchmark_progress, dict) else None
        if current_target_checkpoint is not None:
            current_target_checkpoint = str(current_target_checkpoint).upper()
        predicted_collision_probability = 0.0 if safety_context is None else float(safety_context.predicted_collision_probability)
        trigger_threshold = float(getattr(self.controller, "predicted_collision_replan_threshold", COLLISION_PROBABILITY_REPLAN_THRESHOLD))
        is_replan_call = str(planning_stage or "initial").strip().lower() == "replan"
        replan_history_block = self._build_replan_history_block(
            task_description=task_description,
            previous_plan=previous_plan,
            execution_history=execution_history,
            safety_context=safety_context,
            active_checkpoint_ids=active_checkpoint_ids,
            benchmark_progress=benchmark_progress,
            trigger_threshold=trigger_threshold,
        )
        prompt_template = (self.prompt_plan_replan if is_replan_call else self.prompt_plan_initial)
        prompt_variant_payload = self._get_prompt_variant_payload()
        example_variant_payload = self._get_example_variant_payload()
        prompt_template = (
            prompt_variant_payload["plan_replan_prompt"]
            if is_replan_call
            else prompt_variant_payload["plan_initial_prompt"]
        )
        execution_history_block = (execution_history if is_replan_call else None)
        mission_progress_block = (benchmark_progress if is_replan_call else None)
        prompt = prompt_template.format(
            system_skill_description_low=self.low_level_skillset,
            guides=self._get_guides_text(),
            typefly_initial_examples=example_variant_payload["initial_examples"],
            typefly_replan_examples=example_variant_payload["replan_examples"],
            BASELINE1_OPENING_BLOCK=shared_opening_block,
            BASELINE2_OPENING_BLOCK=shared_opening_block,
            BASELINE3_OPENING_BLOCK=shared_opening_block,
            BASELINE1_RUNTIME_CONTEXT_BLOCK=shared_runtime_context_block,
            BASELINE2_RUNTIME_CONTEXT_BLOCK=shared_runtime_context_block,
            BASELINE3_RUNTIME_CONTEXT_BLOCK=shared_runtime_context_block,
            BASELINE1_GUIDES=self._get_guides_text(),
            BASELINE2_GUIDES=self._get_guides_text(),
            BASELINE3_GUIDES=self._get_guides_text(),
            BASELINE1_INITIAL_EXAMPLES=example_variant_payload["initial_examples"],
            BASELINE2_INITIAL_EXAMPLES=example_variant_payload["initial_examples"],
            BASELINE3_INITIAL_EXAMPLES=example_variant_payload["initial_examples"],
            BASELINE1_REPLAN_EXAMPLES=example_variant_payload["replan_examples"],
            BASELINE2_REPLAN_EXAMPLES=example_variant_payload["replan_examples"],
            BASELINE3_REPLAN_EXAMPLES=example_variant_payload["replan_examples"],
            error_message=error_message,
            scene_description=full_scene,
            task_description=task_description,
            shared_opening_block=shared_opening_block,
            shared_runtime_context_block=shared_runtime_context_block,
            replan_history_block=replan_history_block,
            execution_history=execution_history_block,
            mission_progress=mission_progress_block,
            previous_plan=previous_plan if is_replan_call else None,
            completed_checkpoints=true_completed_checkpoints if is_replan_call else [],
            remaining_checkpoints=true_remaining_checkpoints if is_replan_call else [str(v).upper() for v in active_checkpoint_ids],
            current_target_checkpoint=current_target_checkpoint,
        )
        dump_prompt = str(os.getenv("TYPEFLY_DUMP_LLM_PROMPT", "1")).strip().lower() not in {"0", "false", "no"}
        if dump_prompt:
            default_archive_root, _ = resolve_archive_root_and_excel_path(None)
            dump_path = os.getenv("TYPEFLY_LAST_PROMPT_PATH", os.path.join(default_archive_root, "last_llm_prompt.txt"))
            try:
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                with open(dump_path, "w") as f:
                    f.write(prompt)
                print_debug(f"[P-PROMPT-DUMP] wrote final prompt to {dump_path}")
            except Exception as exc:
                print_debug(f"[P-PROMPT-DUMP] failed to write prompt: {exc}")
        print_t(f"[P] Planning request: {task_description}")
        print_debug(
            f"[P-PROMPT-PATHS] prompt_plan={(self.prompt_plan_replan_path if is_replan_call else self.prompt_plan_initial_path)} "
            f"guides={self.guides_path} "
            f"typefly_initial_examples={self.typefly_initial_examples_path} "
            f"typefly_replan_examples={self.typefly_replan_examples_path} "
            f"agent_decomposition_examples={self.agent_decomposition_examples_path} "
            f"agent_mode_action_examples={self.agent_mode_action_examples_path} "
            f"agent_reflection_examples={self.agent_reflection_examples_path}"
        )
        print_debug(f"[P-RUNTIME-CONTEXT]\n{shared_runtime_context_block}")
        if replan_history_block:
            print_debug(f"[P-REPLAN-HISTORY]\n{replan_history_block}")
        print_debug(f"[P] Full prompt debug log: {chat_log_path}")
        raw_response = self.llm.request(prompt, self.model_name, stream=False)
        self._last_plan_trace = {
            "prompt": prompt,
            "raw_response": raw_response,
            "planning_stage": str(planning_stage or "initial"),
            "plan_source": ("llm_replan" if str(planning_stage or "initial").strip().lower() == "replan" else "llm_initial"),
            "prompt_variant": self.runtime_prompt_variant,
            "example_variant": self.runtime_example_variant,
            "use_output_example": bool(self.runtime_use_output_example),
            "source": "plan",
            "true_completed_checkpoints": list(true_completed_checkpoints),
            "true_remaining_checkpoints": list(true_remaining_checkpoints),
            "current_target_checkpoint": current_target_checkpoint,
            "completion_state_source": "benchmark_progress/dwell_tracker",
        }
        return raw_response

    def plan_agent_heartbeat(
        self,
        task_description: str,
        snapshot: dict,
        execution_history,
        current_plan: str,
        mission_original_plan: Optional[str] = None,
        current_active_plan: Optional[str] = None,
        latest_full_replan_response: Optional[str] = None,
        full_replan_count: int = 0,
        hard_gate: bool = False,
        feedback_memory_packets: Optional[list] = None,
    ) -> dict:
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        collision_probability = 0.0 if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))
        dominant_worker = "n/a" if safety_context is None else str(getattr(safety_context, "dominant_threat_id", "n/a"))
        benchmark_progress = dict(snapshot.get("benchmark_progress") or {})
        completed = list(benchmark_progress.get("completed") or [])
        active = list((snapshot.get("active_objective_set") or {}).get("active_checkpoint_ids", []))
        workers = []
        for worker in list(snapshot.get("workers") or []):
            workers.append({
                "id": str(worker.get("id")),
                "xy": tuple(worker.get("ui_xy") or worker.get("gt_xy") or (None, None)),
            })
        hard_gate_rule = (
            "If predicted_collision_probability > 0.7, you MUST output response=full_replan_plan with a new complete MiniSpec plan."
            if hard_gate
            else "You may choose continue or full_replan_plan based on your judgment."
        )
        agent_heartbeat_examples = (
            self._get_example_variant_payload().get("heartbeat_hardgate_examples", self.agent_heartbeat_hardgate_examples)
            if hard_gate
            else self._get_example_variant_payload().get("heartbeat_soft_examples", self.agent_heartbeat_soft_examples)
        )
        mission_original_plan_text = str(mission_original_plan or current_plan or "none")
        current_active_plan_text = str(current_active_plan or mission_original_plan_text)
        latest_full_replan_text = str(latest_full_replan_response or "none")
        heartbeat_prompt_template = (
            self._get_prompt_variant_payload().get("heartbeat_hardgate_prompt", self.agent_heartbeat_hardgate_prompt)
            if hard_gate
            else self._get_prompt_variant_payload().get("heartbeat_soft_prompt", self.agent_heartbeat_soft_prompt)
        )
        prompt = heartbeat_prompt_template.format(
            shared_opening_block=self._build_opening_block(),
            shared_runtime_context_block=self._build_runtime_context_block(
                safety_context,
                snapshot=(snapshot if isinstance(snapshot, dict) else {}),
            ),
            BASELINE1_OPENING_BLOCK=self._build_opening_block(),
            BASELINE2_OPENING_BLOCK=self._build_opening_block(),
            BASELINE3_OPENING_BLOCK=self._build_opening_block(),
            BASELINE1_RUNTIME_CONTEXT_BLOCK=self._build_runtime_context_block(
                safety_context,
                snapshot=(snapshot if isinstance(snapshot, dict) else {}),
            ),
            BASELINE2_RUNTIME_CONTEXT_BLOCK=self._build_runtime_context_block(
                safety_context,
                snapshot=(snapshot if isinstance(snapshot, dict) else {}),
            ),
            BASELINE3_RUNTIME_CONTEXT_BLOCK=self._build_runtime_context_block(
                safety_context,
                snapshot=(snapshot if isinstance(snapshot, dict) else {}),
            ),
            task_description=task_description,
            mission_original_plan=mission_original_plan_text,
            current_active_plan=current_active_plan_text,
            latest_full_replan_response=latest_full_replan_text,
            full_replan_count=int(max(0, full_replan_count)),
            completed_checkpoints=completed,
            unfinished_checkpoints=[cid for cid in active if cid not in completed],
            uav_position=(snapshot.get('drone_est_bias_corrected') or snapshot.get('drone_est') or snapshot.get('drone_gt')),
            uav_heading=snapshot.get('drone_yaw_rad'),
            worker_positions=workers,
            predicted_collision_probability=f"{collision_probability:.6f}",
            dominant_worker=dominant_worker,
            current_executing_plan=current_plan,
            queue_progress=benchmark_progress,
            execution_history=execution_history,
            feedback_memory_packets=json.dumps(feedback_memory_packets or [], ensure_ascii=False),
        )
        feedback_block = (
            "Matured evaluator feedback memory packets (persist across heartbeats):\n"
            f"{json.dumps(feedback_memory_packets or [], ensure_ascii=False, indent=2)}\n\n"
            "Use these packets as episodic evidence. Do not assume any hidden intent beyond these records.\n\n"
        )
        examples_block = ""
        if self.runtime_use_output_example:
            examples_block = (
                "Agent heartbeat examples:\n"
                f"{agent_heartbeat_examples}\n\n"
            )
        prompt = (
            f"{prompt}\n\n"
            f"{feedback_block}"
            f"Hard gate policy note: {hard_gate_rule}\n\n"
            f"{examples_block}"
            "Return JSON only."
        )
        print_debug(
            "[AGENT-HEARTBEAT-PROMPT-CONTEXT] "
            f"predicted_collision_probability={collision_probability:.6f} "
            f"full_replan_count={int(max(0, full_replan_count))} "
            f"hard_gate={hard_gate}"
        )
        raw = str(self.llm.request(prompt, self.heartbeat_model_name, stream=False) or "").strip()
        parsed, parsed_ok = self._parse_heartbeat_response_json(raw)
        if not parsed_ok:
            parsed = {"response": "continue", "reason": f"non_json_response:{raw[:120]}", "plan": ""}
        response = str(parsed.get("response", "continue")).strip().lower()
        if response not in {"continue", "full_replan_plan"}:
            response = "continue"
        result = {
            "response": response,
            "reason": str(parsed.get("reason", "")).strip(),
            "plan": str(parsed.get("plan", "")).strip(),
            "raw_response": raw,
            "parsed_ok": bool(parsed_ok),
        }
        self._last_heartbeat_trace = {
            "prompt": prompt,
            "raw_response": raw,
            "parsed_response": result,
            "prompt_variant": self.runtime_prompt_variant,
            "example_variant": self.runtime_example_variant,
            "use_output_example": bool(self.runtime_use_output_example),
            "source": "agent_heartbeat",
            "used_model_name": self.heartbeat_model_name,
        }
        return result

    def evaluate_agent_replan_record(self, replan_record: dict) -> dict:
        prompt_template = str(self.agent_evaluator_prompt or "").strip()
        if not prompt_template:
            prompt_template = (
                "You are an evaluator LLM. Return JSON with keys necessity_assessment, outcome_assessment, "
                "evidence_summary, timing_suggestion, content_suggestion, next_time_hint, confidence."
            )
        prompt = (
            f"{prompt_template}\n\n"
            f"decision_context:\n{json.dumps(replan_record.get('decision_context', {}), ensure_ascii=False, indent=2)}\n\n"
            f"chosen_action:\n{json.dumps(replan_record.get('chosen_action', {}), ensure_ascii=False, indent=2)}\n\n"
            f"outcome_delta:\n{json.dumps(replan_record.get('outcome_delta', {}), ensure_ascii=False, indent=2)}\n"
        )
        raw = str(self.llm.request(prompt, self.evaluator_model_name, stream=False) or "").strip()
        parsed, parsed_ok = self._parse_agent_evaluator_json(raw)
        if not parsed_ok:
            parsed = {
                "necessity_assessment": "uncertain",
                "outcome_assessment": "uncertain",
                "evidence_summary": "parser_failed_or_non_json_response",
                "timing_suggestion": "insufficient_evidence",
                "content_suggestion": "insufficient_evidence",
                "next_time_hint": "keep conservative until clearer evidence appears",
                "confidence": "low",
            }
        return {
            "prompt": prompt,
            "raw_response": raw,
            "parsed": parsed,
            "parsed_ok": bool(parsed_ok),
            "used_model_name": self.evaluator_model_name,
        }

    @staticmethod
    def _parse_agent_evaluator_json(raw: str) -> tuple[dict, bool]:
        parsed, ok = LLMPlanner._parse_heartbeat_response_json(raw)
        if not ok or not isinstance(parsed, dict):
            return {}, False
        necessity = str(parsed.get("necessity_assessment", "uncertain")).strip().lower()
        outcome = str(parsed.get("outcome_assessment", "uncertain")).strip().lower()
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        if necessity not in {"likely_necessary", "possibly_unnecessary", "uncertain"}:
            necessity = "uncertain"
        if outcome not in {"improved_safety", "little_change", "worsened", "uncertain"}:
            outcome = "uncertain"
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        normalized = {
            "necessity_assessment": necessity,
            "outcome_assessment": outcome,
            "evidence_summary": str(parsed.get("evidence_summary", "")).strip(),
            "timing_suggestion": str(parsed.get("timing_suggestion", "")).strip(),
            "content_suggestion": str(parsed.get("content_suggestion", "")).strip(),
            "next_time_hint": str(parsed.get("next_time_hint", "")).strip(),
            "confidence": confidence,
        }
        return normalized, True

    @staticmethod
    def _parse_heartbeat_response_json(raw: str) -> tuple[dict, bool]:
        text = str(raw or "").strip()
        if not text:
            return {}, False
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_match:
            text = str(fenced_match.group(1) or "").strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass
        return {}, False
    
    def probe(self, question: str) -> MiniSpecValueType:
        location_info = None
        try:
            if self.controller and hasattr(self.controller, '_format_planner_location_info'):
                location_info = self.controller._format_planner_location_info()
        except Exception:
            location_info = None

        if location_info is None:
            drone_pos = (0.00, 0.00, 0.00)

            try:
                if self.controller and hasattr(self.controller, 'state_provider'):
                    get_est_drone = getattr(self.controller.state_provider, 'get_estimated_drone_position', None)
                    if callable(get_est_drone):
                        value = get_est_drone()
                        if value is not None:
                            drone_pos = value
            except Exception:
                pass

            location_info = (
                f"Drone estimated position: x={drone_pos[0]:.2f}, y={drone_pos[1]:.2f}, z={drone_pos[2]:.2f}"
            )

        # 是否啟用影像辨識
        try:
            if self.vision_skill and getattr(self.vision_skill, 'enabled', True):
                scene_description = self.vision_skill.get_obj_list()
            else:
                scene_description = ''
        except Exception:
            scene_description = ''

        full_scene = f"{scene_description}\n{location_info}".strip()

        prompt = self.prompt_probe.format(scene_description=full_scene, question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt, self.model_name)), False

    def decompose_task_for_langgraph(
        self,
        task_description: str,
        active_checkpoint_ids: list[str],
        active_zone_ids: list[str],
    ) -> list[str]:
        snapshot = self.controller.get_live_ui_snapshot() if (self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot")) else {}
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        shared_runtime_context_block = self._build_shared_runtime_context_block(
            safety_context,
            snapshot=(snapshot if isinstance(snapshot, dict) else {}),
        )
        prompt = self.prompt_langgraph_decomposition.format(
            task_description=str(task_description or ""),
            allowed_checkpoints=[str(v).upper() for v in list(active_checkpoint_ids or [])],
            active_zones=[str(v) for v in list(active_zone_ids or [])],
            shared_opening_block=self._build_shared_opening_block(),
            shared_runtime_context_block=shared_runtime_context_block,
            agent_decomposition_examples=self.agent_decomposition_examples,
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        parsed: list[str] = []
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, list):
                parsed = [str(v).upper() for v in obj]
        except Exception:
            tokens = re.findall(r"[A-Za-z]\d+", raw)
            parsed = [str(v).upper() for v in tokens]
        allowed = [str(v).upper() for v in list(active_checkpoint_ids or [])]
        filtered = [cid for cid in parsed if cid in allowed]
        if filtered:
            return filtered
        return allowed

    def plan_langgraph_step_action(
        self,
        task_description: str,
        current_subgoal: str | None,
        remaining_checkpoints: list[str],
        predicted_collision_risk: float,
                per_worker_collision_risks: dict[str, float],
        dominant_risky_worker: str | None,
        worker_states_summary: list[dict],
        last_action: str,
        last_result: str,
        recent_history: list[dict],
        recovery_mode: bool = False,
        recovery_reason: str | None = None,
        strategy_summary: str = "",
        last_failure_reason: str | None = None,
        failed_approach_pattern: str | None = None,
        recovery_hypothesis: str | None = None,
        blocked_workers_for_subgoal: list[str] | None = None,
        subgoal_attempts: list[str] | None = None,
    ) -> str:
        snapshot = self.controller.get_live_ui_snapshot() if (self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot")) else {}
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        shared_runtime_context_block = self._build_shared_runtime_context_block(
            safety_context,
            snapshot=(snapshot if isinstance(snapshot, dict) else {}),
        )
        prompt = self.prompt_langgraph_step.format(
            task_description=str(task_description or ""),
            current_subgoal=str(current_subgoal or "None"),
            remaining_checkpoints=[str(v) for v in list(remaining_checkpoints or [])],
            predicted_collision_risk=f"{float(predicted_collision_risk):.6f}",
                        per_worker_collision_risks=str(dict(per_worker_collision_risks or {})),
            dominant_risky_worker=str(dominant_risky_worker or "n/a"),
            recovery_mode=str(bool(recovery_mode)),
            recovery_reason=str(recovery_reason or "none"),
            strategy_summary=str(strategy_summary or ""),
            last_failure_reason=str(last_failure_reason or "none"),
            failed_approach_pattern=str(failed_approach_pattern or "none"),
            recovery_hypothesis=str(recovery_hypothesis or "none"),
            blocked_workers_for_subgoal=str(list(blocked_workers_for_subgoal or [])),
            subgoal_attempts=str(list(subgoal_attempts or [])),
            worker_states_summary=str(list(worker_states_summary or [])[:3]),
            last_action=str(last_action or ""),
            last_result=str(last_result or ""),
            recent_history=str(list(recent_history or [])[-4:]),
            shared_opening_block=self._build_shared_opening_block(),
            shared_runtime_context_block=shared_runtime_context_block,
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        action = self._sanitize_langgraph_action(raw)
        if action:
            return action
        if current_subgoal:
            return f"gc('{str(current_subgoal).upper()}');"
        return "d(1.0);"

    def reflect_langgraph_strategy(
        self,
        task_description: str,
        current_subgoal: str | None,
        route_decision: str,
        last_action: str,
        last_result: dict,
        predicted_collision_risk: float,
        per_worker_collision_risks: dict[str, float],
        dominant_risky_worker: str | None,
        worker_states_summary: list[dict],
        remaining_checkpoints: list[str],
        subgoal_queue: list[str],
        current_strategy_summary: str,
        last_failure_reason: str | None,
        recent_failed_approach_pattern: str | None,
        recent_recovery_hypothesis: str | None,
        blocked_workers_by_subgoal: dict[str, list[str]],
        subgoal_attempt_history: dict[str, list[str]],
        recent_history: list[dict],
    ) -> dict:
        snapshot = self.controller.get_live_ui_snapshot() if (self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot")) else {}
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        shared_runtime_context_block = self._build_shared_runtime_context_block(
            safety_context,
            snapshot=(snapshot if isinstance(snapshot, dict) else {}),
        )
        prompt = self.prompt_langgraph_reflection.format(
            task_description=str(task_description or ""),
            current_subgoal=str(current_subgoal or "None"),
            route_decision=str(route_decision or "continue"),
            last_action=str(last_action or ""),
            last_result=str(last_result or {}),
            predicted_collision_risk=f"{float(predicted_collision_risk):.6f}",
            dominant_risky_worker=str(dominant_risky_worker or "n/a"),
            per_worker_collision_risks=str(dict(per_worker_collision_risks or {})),
            worker_states_summary=str(list(worker_states_summary or [])[:3]),
            remaining_checkpoints=str([str(v).upper() for v in list(remaining_checkpoints or [])]),
            subgoal_queue=str([str(v).upper() for v in list(subgoal_queue or [])]),
            current_strategy_summary=str(current_strategy_summary or ""),
            last_failure_reason=str(last_failure_reason or "none"),
            recent_failed_approach_pattern=str(recent_failed_approach_pattern or "none"),
            recent_recovery_hypothesis=str(recent_recovery_hypothesis or "none"),
            blocked_workers_by_subgoal=str(dict(blocked_workers_by_subgoal or {})),
            subgoal_attempt_history=str(dict(subgoal_attempt_history or {})),
            recent_history=str(list(recent_history or [])[-6:]),
            shared_opening_block=self._build_shared_opening_block(),
            shared_runtime_context_block=shared_runtime_context_block,
            agent_reflection_examples=self.agent_reflection_examples,
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        parsed = self._safe_json_object(raw)

        fallback_failure = ""
        if not bool((last_result or {}).get("ok", True)):
            fallback_failure = str((last_result or {}).get("message", "step_failed"))[:120]
        next_route = str(parsed.get("next_route_decision", route_decision or "continue"))
        if next_route not in {"continue", "retry_plan", "reselect_subgoal"}:
            next_route = str(route_decision or "continue")
        return {
            "failure_reason": str(parsed.get("failure_reason", fallback_failure)).strip()[:200],
            "failed_approach_pattern": str(parsed.get("failed_approach_pattern", "")).strip()[:120],
            "recovery_hypothesis": str(parsed.get("recovery_hypothesis", "")).strip()[:160],
            "strategy_summary": str(parsed.get("strategy_summary", current_strategy_summary or "")).strip()[:240],
            "next_route_decision": next_route,
            "next_subgoal": parsed.get("next_subgoal"),
            "reprioritized_subgoals": parsed.get("reprioritized_subgoals", []),
        }

    def decide_langgraph_mode_and_action(
        self,
        task_description: str,
        current_subgoal: str | None,
        remaining_checkpoints: list[str],
        predicted_collision_risk: float,
                per_worker_collision_risks: dict[str, float],
        dominant_risky_worker: str | None,
        worker_states_summary: list[dict],
        last_action: str,
        last_result: str,
        recent_history: list[dict],
        recovery_mode: bool = False,
        recovery_reason: str | None = None,
        strategy_summary: str = "",
        last_failure_reason: str | None = None,
        failed_approach_pattern: str | None = None,
        recovery_hypothesis: str | None = None,
        blocked_workers_for_subgoal: list[str] | None = None,
        subgoal_attempts: list[str] | None = None,
        current_mode: str = "approach",
        last_wait_event: dict | None = None,
        last_risk_event: dict | None = None,
        current_subgoal_phase: str = "APPROACH_SUBGOAL",
        reached_not_completed: bool = False,
        in_checkpoint_radius: bool = False,
        dwell_seconds: float = 0.0,
        dwell_required_seconds: float = 2.0,
        dwell_satisfied: bool = False,
    ) -> dict:
        snapshot = self.controller.get_live_ui_snapshot() if (self.controller is not None and hasattr(self.controller, "get_live_ui_snapshot")) else {}
        safety_context = snapshot.get("safety_context") if isinstance(snapshot, dict) else None
        shared_runtime_context_block = self._build_shared_runtime_context_block(
            safety_context,
            snapshot=(snapshot if isinstance(snapshot, dict) else {}),
        )
        prompt = self.prompt_langgraph_mode_action.format(
            task_description=str(task_description or ""),
            current_mode=str(current_mode or "approach"),
            current_subgoal=str(current_subgoal or "None"),
            current_subgoal_phase=str(current_subgoal_phase or "APPROACH_SUBGOAL"),
            reached_not_completed=bool(reached_not_completed),
            in_checkpoint_radius=bool(in_checkpoint_radius),
            dwell_seconds=f"{float(dwell_seconds):.3f}",
            dwell_required_seconds=f"{float(dwell_required_seconds):.3f}",
            dwell_satisfied=bool(dwell_satisfied),
            remaining_checkpoints=[str(v) for v in list(remaining_checkpoints or [])],
            predicted_collision_risk=f"{float(predicted_collision_risk):.6f}",
                        per_worker_collision_risks=str(dict(per_worker_collision_risks or {})),
            dominant_risky_worker=str(dominant_risky_worker or "n/a"),
            worker_states_summary=str(list(worker_states_summary or [])[:3]),
            last_action=str(last_action or ""),
            last_result=str(last_result or ""),
            recent_failure_reason=str(last_failure_reason or "none"),
            last_wait_event=str(dict(last_wait_event or {})),
            last_risk_event=str(dict(last_risk_event or {})),
            strategy_summary=str(strategy_summary or ""),
            failed_approach_pattern=str(failed_approach_pattern or "none"),
            recovery_hypothesis=str(recovery_hypothesis or "none"),
            blocked_workers_for_subgoal=str(list(blocked_workers_for_subgoal or [])),
            subgoal_attempts=str(list(subgoal_attempts or [])),
            recent_history=str(list(recent_history or [])[-4:]),
            shared_opening_block=self._build_shared_opening_block(),
            shared_runtime_context_block=shared_runtime_context_block,
            agent_mode_action_examples=self.agent_mode_action_examples,
        )
        raw = str(self.llm.request(prompt, self.model_name, stream=False) or "").strip()
        parsed = self._safe_json_object(raw)
        events: list[str] = []
        if not parsed:
            events.append("parser_empty_object")
        elif str(parsed.get("mode", "")).strip().lower() not in {"approach", "recovery", "replan", "skip_current_subgoal"}:
            events.append("parser_normalized_mode_to_approach")
        mode = str(parsed.get("mode", "approach")).strip().lower()
        if mode not in {"approach", "recovery", "replan", "skip_current_subgoal"}:
            mode = "approach"
        raw_action = str(parsed.get("action", ""))
        action = self._sanitize_langgraph_action(raw_action)
        sanitized_payload = {
            "mode": mode,
            "reason": str(parsed.get("reason", "")).strip()[:180],
            "strategy_summary": str(parsed.get("strategy_summary", strategy_summary or "")).strip()[:240],
            "action": action,
            "next_subgoal": parsed.get("next_subgoal"),
            "why_action_matches_mode": str(parsed.get("why_action_matches_mode", "")).strip()[:220],
        }
        if raw_action.strip() and (action != raw_action.strip()):
            events.append("sanitize_changed_action_format")
        if not action:
            events.append("sanitize_invalid_action")
            retry_prompt = (
                prompt
                + "\n\nYour previous action was invalid/unparseable for MiniSpec."
                + " Return JSON only. Keep same intent, but rewrite action as a short valid MiniSpec segment (1~4 statements)."
            )
            retry_raw = str(self.llm.request(retry_prompt, self.model_name, stream=False) or "").strip()
            retry_parsed = self._safe_json_object(retry_raw)
            retry_action_raw = str(retry_parsed.get("action", ""))
            retry_action = self._sanitize_langgraph_action(retry_action_raw)
            events.append("fallback_reprompt")
            if retry_action:
                action = retry_action
                if retry_action_raw.strip() and retry_action != retry_action_raw.strip():
                    events.append("sanitize_changed_action_format_on_reprompt")
                sanitized_payload["action"] = action
                sanitized_payload["reason"] = str(retry_parsed.get("reason", sanitized_payload["reason"])).strip()[:180]
                sanitized_payload["strategy_summary"] = str(
                    retry_parsed.get("strategy_summary", sanitized_payload["strategy_summary"])
                ).strip()[:240]
                sanitized_payload["next_subgoal"] = retry_parsed.get("next_subgoal", sanitized_payload["next_subgoal"])
                sanitized_payload["why_action_matches_mode"] = str(
                    retry_parsed.get("why_action_matches_mode", sanitized_payload["why_action_matches_mode"])
                ).strip()[:220]
                parsed = retry_parsed
                raw = retry_raw
                raw_action = retry_action_raw
            else:
                if mode == "recovery":
                    action = "tc(15);"
                    events.append("fallback_proxy_recovery_action")
                else:
                    action = "d(0.4);"
                    events.append("fallback_proxy_safe_delay")
                sanitized_payload["action"] = action
                sanitized_payload["why_action_matches_mode"] = (
                    sanitized_payload.get("why_action_matches_mode")
                    or "Fallback action preserves safety while awaiting cleaner structured decision output."
                )
        next_subgoal = parsed.get("next_subgoal")
        return {
            "mode": mode,
            "reason": str(parsed.get("reason", "")).strip()[:180],
            "strategy_summary": str(parsed.get("strategy_summary", strategy_summary or "")).strip()[:240],
            "action": action,
            "next_subgoal": next_subgoal,
            "why_action_matches_mode": str(parsed.get("why_action_matches_mode", "")).strip()[:220],
            "trace": {
                "raw_llm_payload": raw,
                "parsed_payload": {
                    "mode": str(parsed.get("mode", "")),
                    "reason": str(parsed.get("reason", "")),
                    "strategy_summary": str(parsed.get("strategy_summary", "")),
                    "action": raw_action,
                    "next_subgoal": parsed.get("next_subgoal"),
                    "why_action_matches_mode": str(parsed.get("why_action_matches_mode", "")),
                },
                "sanitized_payload": sanitized_payload,
                "events": events,
            },
        }

    def _safe_json_object(self, raw_text: str) -> dict:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            try:
                obj = ast.literal_eval(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {}
        return {}

    def _sanitize_langgraph_action(self, raw_text: str) -> str:
        text = str(raw_text or "").strip().replace("\n", " ")
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        commands = []
        alias = {
            "gc": "go_checkpoint",
            "mf": "move_forward",
            "mb": "move_backward",
            "ml": "move_left",
            "mr": "move_right",
            "tc": "turn_cw",
            "tu": "turn_ccw",
            "d": "delay",
            "lo": "log",
            "move_back": "move_backward",
            "turn_right": "turn_cw",
            "turn_left": "turn_ccw",
            "hold": "delay",
            "wait": "delay",
            "reobserve": "delay",
        }
        abbr_out = {
            "go_checkpoint": "gc",
            "move_forward": "mf",
            "move_backward": "mb",
            "move_left": "ml",
            "move_right": "mr",
            "turn_cw": "tc",
            "turn_ccw": "tu",
            "delay": "d",
            "log": "lo",
        }
        parts = [p.strip() for p in text.split(";") if p.strip()]
        for part in parts:
            if len(commands) >= 4:
                break
            m = re.fullmatch(r'(?:go_checkpoint|gc)\(\s*[\'"]?([A-Za-z]\d+)[\'"]?\s*\)', part)
            if m:
                commands.append(f"gc('{m.group(1).upper()}');")
                continue
            m = re.fullmatch(r'(?:log|lo)\(\s*[\'"]([^\'"]{1,60})[\'"]\s*\)', part)
            if m:
                safe = re.sub(r"[^a-zA-Z0-9 _-]", "", m.group(1))[:60]
                commands.append(f"lo('{safe}');")
                continue
            m = re.fullmatch(r'([a-zA-Z_]+)\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)', part)
            if not m:
                continue
            name = alias.get(m.group(1), m.group(1))
            if name not in {"move_forward", "move_backward", "move_left", "move_right", "turn_cw", "turn_ccw", "delay"}:
                continue
            value = float(m.group(2))
            if name.startswith("turn_"):
                commands.append(f"{abbr_out[name]}({int(value)});")
            else:
                commands.append(f"{abbr_out[name]}({value:.1f});")
        if not commands:
            lower = text.lower()
            number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", lower)
            value = 0.6 if number_match is None else float(number_match.group(1))
            if "move left" in lower:
                commands.append(f"ml({value:.1f});")
            elif "move right" in lower:
                commands.append(f"mr({value:.1f});")
            elif "move back" in lower or "backward" in lower:
                commands.append(f"mb({value:.1f});")
            elif "move forward" in lower:
                commands.append(f"mf({value:.1f});")
            elif "turn left" in lower or "ccw" in lower:
                commands.append(f"tu({int(max(5.0, value))});")
            elif "turn right" in lower or "cw" in lower:
                commands.append(f"tc({int(max(5.0, value))});")
            elif "hold" in lower or "wait" in lower or "reobserve" in lower:
                commands.append("d(0.5);")
        return "".join(commands)
