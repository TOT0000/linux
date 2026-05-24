from pathlib import Path

PLANNER_SOURCE = Path('controller/llm_planner.py').read_text(encoding='utf-8')
SOFT_EXAMPLES = Path('controller/assets/tello/agent_heartbeat_soft_examples.txt').read_text(encoding='utf-8')
HARD_EXAMPLES = Path('controller/assets/tello/agent_heartbeat_hardgate_examples.txt').read_text(encoding='utf-8')
SOFT_PROMPT = Path('controller/assets/tello/agent_heartbeat_soft_prompt.txt').read_text(encoding='utf-8')
HARD_PROMPT = Path('controller/assets/tello/agent_heartbeat_hardgate_prompt.txt').read_text(encoding='utf-8')
CONTROLLER_SOURCE = Path('controller/llm_controller.py').read_text(encoding='utf-8')


def test_soft_heartbeat_prompt_injects_soft_examples():
    assert 'agent_heartbeat_soft_prompt_path' in PLANNER_SOURCE
    assert 'agent_heartbeat_soft_examples_path' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_soft_examples' in PLANNER_SOURCE
    assert 'if hard_gate' in PLANNER_SOURCE
    assert 'heartbeat_soft_examples' in PLANNER_SOURCE


def test_hardgate_heartbeat_prompt_injects_hardgate_examples():
    assert 'agent_heartbeat_hardgate_prompt_path' in PLANNER_SOURCE
    assert 'agent_heartbeat_hardgate_examples_path' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_hardgate_examples' in PLANNER_SOURCE
    assert 'self.agent_heartbeat_hardgate_examples' in PLANNER_SOURCE


def test_hardgate_prompt_has_extra_hard_gate_rule():
    assert 'If predicted_collision_probability > 0.7, you MUST output response=full_replan_plan with a new complete MiniSpec plan.' in PLANNER_SOURCE
    assert 'You may choose continue or full_replan_plan based on your judgment.' in PLANNER_SOURCE


def test_heartbeat_prompt_contains_budget_and_timing_policy():
    assert 'The system calls you once every 5 seconds during execution.' in SOFT_PROMPT
    assert 'The system calls you once every 5 seconds during execution.' in HARD_PROMPT
    assert 'Full replans already used: {full_replan_count} / 8' in SOFT_PROMPT
    assert 'Full replans already used: {full_replan_count} / 8' in HARD_PROMPT


def test_heartbeat_prompt_includes_plan_context_fields():
    assert 'Mission original plan: {mission_original_plan}' in SOFT_PROMPT
    assert 'Current active plan: {current_active_plan}' in SOFT_PROMPT
    assert 'Latest full replan response (if any): {latest_full_replan_response}' in SOFT_PROMPT


def test_heartbeat_prompt_output_format_unchanged():
    assert 'Return strict JSON with keys: response, reason, plan.' in SOFT_PROMPT
    assert 'response must be one of: continue, full_replan_plan.' in SOFT_PROMPT
    assert 'If response = continue, set plan to an empty string.' in SOFT_PROMPT


def test_examples_files_are_real_and_nonempty_and_referenced():
    assert 'Example S1 (Soft heartbeat: continue current plan under low future risk)' in SOFT_EXAMPLES
    assert 'Example H1 (HardGate heartbeat: below threshold, continue)' in HARD_EXAMPLES
    assert 'Example H5 (HardGate heartbeat: below threshold and budget is limited, so avoid unnecessary replan)' in HARD_EXAMPLES
    assert 'Mission original plan:' in SOFT_EXAMPLES
    assert 'Current active plan:' in SOFT_EXAMPLES
    assert 'Latest full replan response (if any):' in SOFT_EXAMPLES
    assert 'Agent heartbeat examples:' in PLANNER_SOURCE
    assert '{agent_heartbeat_examples}' in PLANNER_SOURCE
    assert 'historical_max_collision_probability' not in PLANNER_SOURCE
    assert 'anomaly-aware replan' not in SOFT_EXAMPLES


def test_heartbeat_parser_has_fenced_and_embedded_json_fallback_paths():
    assert '_parse_heartbeat_response_json' in PLANNER_SOURCE
    assert "```(?:json)?\\s*(.*?)\\s*```" in PLANNER_SOURCE
    assert 'start = text.find("{")' in PLANNER_SOURCE
    assert 'end = text.rfind("}")' in PLANNER_SOURCE


def test_heartbeat_skips_llm_after_task_completed_and_queue_empty():
    assert 'def _pending_execution_statement_count(self) -> int:' in CONTROLLER_SOURCE
    assert 'def _is_active_objective_completed(self) -> bool:' in CONTROLLER_SOURCE
    assert 'def _should_skip_heartbeat_after_task_completion(self) -> bool:' in CONTROLLER_SOURCE
    assert 'if self._should_skip_heartbeat_after_task_completion():' in CONTROLLER_SOURCE
