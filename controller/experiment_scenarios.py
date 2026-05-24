from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    drone_position_3d: Tuple[float, float, float]
    user_position_3d: Tuple[float, float, float]
    drone_yaw_rad: float = 0.0
    notes: str = ""


# PX4 local frame assumes NED (+Z is down). Negative z means higher altitude.
# The positions are fixed and deterministic for repeatable tests.
SCENARIOS: Dict[str, ExperimentScenario] = {
    "SAFE": ExperimentScenario(
        name="SAFE",
        drone_position_3d=(5.5, 6.0, -1.6),
        user_position_3d=(10.8, 6.1, 0.0),
        drone_yaw_rad=0.0,
        notes="Center-region geometry with long standoff and favorable localization condition.",
    ),
    "CAUTION": ExperimentScenario(
        name="CAUTION",
        drone_position_3d=(3.2, 8.8, -1.5),
        user_position_3d=(6.9, 10.3, 0.0),
        drone_yaw_rad=0.4,
        notes="Upper-left zone with moderate standoff and moderate risk.",
    ),
    "WARNING": ExperimentScenario(
        name="WARNING",
        drone_position_3d=(9.8, 2.2, -1.4),
        user_position_3d=(11.0, 3.1, 0.0),
        drone_yaw_rad=0.6,
        notes="Edge-adjacent zone with tighter geometry for warning-level risk.",
    ),
    "DANGER": ExperimentScenario(
        name="DANGER",
        drone_position_3d=(0.9, 0.9, -1.3),
        user_position_3d=(1.9, 1.6, 0.0),
        drone_yaw_rad=0.55,
        notes="Corner-region geometry with higher risk while avoiding immediate overlap.",
    ),
}


def normalize_scenario_name(name: str | None) -> str:
    candidate = (name or "SAFE").strip().upper()
    if candidate not in SCENARIOS:
        return "SAFE"
    return candidate
