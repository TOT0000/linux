from pathlib import Path


def test_predicted_collision_probability_computation_exists():
    src = Path('controller/gcs_safety_assessment.py').read_text(encoding='utf-8')
    assert '_compute_predicted_collision_probability' in src
    assert 'PREDICTION_HORIZON_SECONDS' in src
    assert 'PREDICTION_DT_SECONDS' in src


def test_typefly_threshold_replan_uses_predicted_prob_03():
    src = Path('controller/llm_controller.py').read_text(encoding='utf-8')
    assert 'PREDICTED_COLLISION_PROBABILITY_REPLAN_THRESHOLD = TYPEFLY_REPLAN_THRESHOLD' in src
    assert 'predicted_collision_probability' in src


def test_prompts_use_predicted_and_no_historical_max():
    planner_src = Path('controller/llm_planner.py').read_text(encoding='utf-8')
    assert 'predicted_collision_probability' in planner_src
    assert 'historical_max_collision_risk' not in planner_src
    assert 'agent_heartbeat_soft_prompt_path' in planner_src
    assert 'agent_heartbeat_hardgate_prompt_path' in planner_src
    assert 'self.agent_decomposition_examples_path = None' in planner_src


def test_ui_uses_predicted_collision_probability_for_status_and_charts():
    ui_src = Path('serving/webui/typefly.py').read_text(encoding='utf-8')
    assert "predicted_collision_probability" in ui_src
    assert '3s Predicted Collision Probability' in ui_src
    assert 'removed_historical_max_collision_probability' not in ui_src
