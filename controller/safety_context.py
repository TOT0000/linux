from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SafetyContext:
    safety_score: float
    preferred_standoff_m: float
    reason_tags: List[str]
    envelope_gap_m: float
    uncertainty_scale_m: float
    drone_to_user_distance_xy: float
    envelopes_overlap: bool
    dominant_threat_type: str = "user"
    dominant_threat_id: str = "user"
    dominant_gap_m: float = 0.0
    dominant_uncertainty_scale_m: float = 1.0
    task_points_summary: Optional[List[Dict[str, Any]]] = None
    obstacles_summary: Optional[List[Dict[str, Any]]] = None
    path_summary: Optional[Dict[str, Any]] = None
    candidate_targets_summary: Optional[List[Dict[str, Any]]] = None
    candidate_path_summaries: Optional[List[Dict[str, Any]]] = None
    current_collision_probability: float = 0.0
    predicted_collision_probability: float = 0.0
    historical_max_collision_probability: float = 0.0  # deprecated internal compatibility field
    per_worker_collision_probabilities: Optional[List[Dict[str, Any]]] = None
    collision_debug_info: Optional[Dict[str, Any]] = None

    def to_prompt_block(self) -> str:
        per_worker_probs = self.per_worker_collision_probabilities or []
        per_worker_prob_block = (
            "\n".join(
                f"- {row.get('id')}: P_c={float(row.get('collision_probability')):.6f}"
                for row in per_worker_probs
            )
            if per_worker_probs
            else "- (n/a)"
        )
        return (
            f"current_instantaneous_collision_probability: {self.current_collision_probability:.6f}\n"
            f"predicted_collision_probability: {self.predicted_collision_probability:.6f}\n"
            f"safety_score: {self.safety_score:.3f}\n"
            f"reason_tags: {self.reason_tags}\n"
            f"dominant_threat_type: {self.dominant_threat_type}\n"
            f"dominant_threat_id: {self.dominant_threat_id}\n"
            f"dominant_gap_m: {self.dominant_gap_m:.2f}\n"
            f"dominant_uncertainty_scale_m: {self.dominant_uncertainty_scale_m:.2f}\n"
            f"drone_to_user_distance_xy: {self.drone_to_user_distance_xy:.2f}\n"
            f"envelope_gap_m(centerline_ray_gap): {self.envelope_gap_m:.2f}\n"
            f"uncertainty_scale_m: {self.uncertainty_scale_m:.2f}\n"
            f"envelopes_overlap(centerline): {self.envelopes_overlap}\n"
            f"PerWorkerCollisionProbabilities:\n{per_worker_prob_block}"
        )
