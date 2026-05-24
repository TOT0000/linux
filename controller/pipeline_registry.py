from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PipelineConfig:
    id: str
    name: str
    description: str
    base_mode: str
    trigger_type: str
    trigger_params: Dict[str, float]
    prompt_variant: str
    example_variant: str
    state_fields: List[str]
    use_output_example: bool
    replan_cap: int
    archive_enabled_default: Optional[bool] = None


PIPELINE_REGISTRY: Dict[str, PipelineConfig] = {
    "baseline1": PipelineConfig(
        id="baseline1",
        name="Periodic-Minimal",
        description="Heartbeat periodic trigger with minimal state payload.",
        base_mode="agent-heartbeat-soft",
        trigger_type="periodic",
        trigger_params={"heartbeat_seconds": 5.0},
        prompt_variant="baseline1_prompt",
        example_variant="baseline1_example",
        state_fields=[
            "uav_pose_heading",
            "worker_positions",
            "remaining_checkpoints",
            "task_progress",
        ],
        use_output_example=True,
        replan_cap=8,
        archive_enabled_default=True,
    ),
    "baseline2": PipelineConfig(
        id="baseline2",
        name="Periodic-InfoAware",
        description="Heartbeat periodic trigger with risk-aware state payload.",
        base_mode="agent-heartbeat-soft",
        trigger_type="periodic",
        trigger_params={"heartbeat_seconds": 5.0},
        prompt_variant="baseline2_prompt",
        example_variant="baseline2_example",
        state_fields=[
            "uav_pose_heading",
            "worker_positions",
            "remaining_checkpoints",
            "task_progress",
            "predicted_collision_probability",
            "risk_summary",
        ],
        use_output_example=True,
        replan_cap=8,
        archive_enabled_default=True,
    ),
    "agent": PipelineConfig(
        id="agent",
        name="Agent-Feedback-Eval",
        description="Periodic heartbeat with evaluator feedback memory",
        base_mode="agent-heartbeat-soft",
        trigger_type="periodic",
        trigger_params={"heartbeat_seconds": 5.0},
        prompt_variant="agent_prompt",
        example_variant="agent_example",
        state_fields=[
            "uav_pose_heading",
            "worker_positions",
            "remaining_checkpoints",
            "task_progress",
            "predicted_collision_probability",
            "risk_summary",
        ],
        use_output_example=True,
        replan_cap=8,
        archive_enabled_default=True,
    ),
    "baseline3": PipelineConfig(
        id="baseline3",
        name="Event-PredRisk-0.5",
        description="Event trigger when predicted collision risk crosses threshold.",
        base_mode="typefly-threshold-replan",
        trigger_type="event_predicted_collision_probability",
        trigger_params={"predicted_collision_threshold": 0.5, "strictly_greater": True},
        prompt_variant="baseline3_prompt",
        example_variant="baseline3_example",
        state_fields=[
            "uav_pose_heading",
            "worker_positions",
            "remaining_checkpoints",
            "task_progress",
            "predicted_collision_probability",
            "risk_summary",
        ],
        use_output_example=True,
        replan_cap=8,
        archive_enabled_default=True,
    ),
}


def normalize_pipeline_id(pipeline_id: Optional[str]) -> str:
    candidate = str(pipeline_id or "baseline1").strip().lower()
    if candidate not in PIPELINE_REGISTRY:
        return "baseline1"
    return candidate


def get_pipeline_config(pipeline_id: Optional[str]) -> PipelineConfig:
    return PIPELINE_REGISTRY[normalize_pipeline_id(pipeline_id)]


def list_pipeline_options() -> List[str]:
    return list(PIPELINE_REGISTRY.keys())
