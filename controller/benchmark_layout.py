from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


WORKSPACE_SIZE_M = 12.0
CHECKPOINT_RADIUS_M = 0.5
CHECKPOINT_DWELL_SECONDS = 2.0
UAV_RADIUS_M = 0.22
WORKER_RADIUS_M = 0.30
WORKER_DEFAULT_SPEED_MPS = 0.4
FULL_REPLAN_LIMIT = 8
TYPEFLY_REPLAN_THRESHOLD = 0.7
AGENT_HEARTBEAT_INTERVAL_SECONDS = 5.0
PREDICTION_HORIZON_SECONDS = 3.0
PREDICTION_DT_SECONDS = 0.5
UAV_GC_NOMINAL_SPEED_MPS = 0.8
UAV_BODY_NOMINAL_SPEED_MPS = 0.6


@dataclass(frozen=True)
class ZoneDef:
    id: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    label_xy: Tuple[float, float]


@dataclass(frozen=True)
class CheckpointDef:
    id: str
    zone_id: str
    x: float
    y: float
    radius_m: float = CHECKPOINT_RADIUS_M


BENCHMARK_ZONES: Tuple[ZoneDef, ...] = (
    ZoneDef("zone_A", (0.0, 6.0), (6.0, 12.0), (2.2, 10.8)),
    ZoneDef("zone_B", (6.0, 12.0), (6.0, 12.0), (8.2, 10.8)),
    ZoneDef("zone_C", (0.0, 12.0), (0.0, 6.0), (5.2, 5.2)),
)

BENCHMARK_CHECKPOINTS: Tuple[CheckpointDef, ...] = (
    CheckpointDef("A1", "zone_A", 1.4, 10.6),
    CheckpointDef("A2", "zone_A", 4.6, 10.6),
    CheckpointDef("A3", "zone_A", 1.4, 7.8),
    CheckpointDef("A4", "zone_A", 4.6, 7.8),
    CheckpointDef("B1", "zone_B", 7.4, 10.6),
    CheckpointDef("B2", "zone_B", 10.6, 10.6),
    CheckpointDef("B3", "zone_B", 7.4, 7.8),
    CheckpointDef("B4", "zone_B", 10.6, 7.8),
    CheckpointDef("C1", "zone_C", 1.6, 4.5),
    CheckpointDef("C2", "zone_C", 4.2, 3.9),
    CheckpointDef("C3", "zone_C", 6.2, 4.7),
    CheckpointDef("C4", "zone_C", 8.4, 3.5),
    CheckpointDef("C5", "zone_C", 10.3, 4.6),
    CheckpointDef("C6", "zone_C", 6.0, 1.7),
)

BENCHMARK_CHECKPOINT_ORDER: Tuple[str, ...] = tuple(cp.id for cp in BENCHMARK_CHECKPOINTS)
BENCHMARK_CHECKPOINTS_BY_ID: Dict[str, CheckpointDef] = {cp.id: cp for cp in BENCHMARK_CHECKPOINTS}


def build_checkpoint_order_polyline() -> List[Tuple[float, float]]:
    return [(BENCHMARK_CHECKPOINTS_BY_ID[cid].x, BENCHMARK_CHECKPOINTS_BY_ID[cid].y) for cid in BENCHMARK_CHECKPOINT_ORDER]
