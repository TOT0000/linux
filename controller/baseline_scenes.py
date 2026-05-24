from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import os
import numpy as np

from .anchor_provider import AnchorGeometryProvider
from .localization_error_model import LocalizationErrorModel
from .localization_estimator import IterativeLeastSquaresEstimator3D
from .state_packet import LocalizedStatePacket
from .safety_envelope import build_safety_envelope, SafetyEnvelope2D
from .benchmark_layout import (
    WORKER_DEFAULT_SPEED_MPS,
    CHECKPOINT_RADIUS_M,
    WORKER_RADIUS_M,
    BENCHMARK_CHECKPOINTS,
)


@dataclass(frozen=True)
class TaskPoint:
    id: str
    x: float
    y: float
    z: float = -1.5


@dataclass(frozen=True)
class StaticObstacle:
    id: str
    gt_x: float
    gt_y: float
    cov_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    est_bias_x_m: float = 0.0
    est_bias_y_m: float = 0.0


@dataclass(frozen=True)
class ObstacleEnvelopeState:
    id: str
    gt_xy: Tuple[float, float]
    est_xy: Tuple[float, float]
    matrix_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    localization_packet: LocalizedStatePacket
    localization_pipeline_function: str
    envelope_builder_function: str
    envelope: SafetyEnvelope2D


@dataclass(frozen=True)
class BaselineScene:
    id: str
    drone_initial_pose: Tuple[float, float, float]
    drone_initial_yaw_rad: float
    user_position: Tuple[float, float, float]
    user_initial_yaw_rad: float
    task_points: Tuple[TaskPoint, ...]
    obstacles: Tuple[StaticObstacle, ...]
    notes: str = ""


@dataclass(frozen=True)
class PathClearResult:
    path_clear: bool
    blocking_entity: str
    corridor_min_gap: float
    blocking_entities: Tuple[str, ...]


@dataclass(frozen=True)
class BaselineExpectation:
    scene_id: str
    target_task_point: str
    expected_path_clear: bool
    expected_blocking_entity: str
    expected_motion_mode: str


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = (abx * abx) + (aby * aby)
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _rotate_to_obstacle_local(x: float, y: float, center_x: float, center_y: float, orientation_deg: float) -> Tuple[float, float]:
    dx = float(x - center_x)
    dy = float(y - center_y)
    theta = math.radians(float(orientation_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    return ((dx * c) + (dy * s), (-dx * s) + (dy * c))


def _signed_gap_segment_to_obstacle_envelope(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    obstacle_state: ObstacleEnvelopeState,
    corridor_half_width_m: float,
    samples: int = 81,
) -> float:
    env = obstacle_state.envelope
    return _signed_gap_segment_to_ellipse_envelope(
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        center_x=float(env.center_xy[0]),
        center_y=float(env.center_xy[1]),
        major_axis_radius=float(env.major_axis_radius),
        minor_axis_radius=float(env.minor_axis_radius),
        orientation_deg=float(env.orientation_deg),
        corridor_half_width_m=float(corridor_half_width_m),
        samples=samples,
    )


def _signed_gap_segment_to_ellipse_envelope(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    center_x: float,
    center_y: float,
    major_axis_radius: float,
    minor_axis_radius: float,
    orientation_deg: float,
    corridor_half_width_m: float,
    samples: int = 81,
) -> float:
    inflated_major = max(1e-4, float(major_axis_radius) + float(corridor_half_width_m))
    inflated_minor = max(1e-4, float(minor_axis_radius) + float(corridor_half_width_m))

    min_f = float("inf")
    for i in range(max(3, samples)):
        t = float(i / (samples - 1))
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        lx, ly = _rotate_to_obstacle_local(
            px,
            py,
            center_x=float(center_x),
            center_y=float(center_y),
            orientation_deg=float(orientation_deg),
        )
        f = ((lx / inflated_major) ** 2) + ((ly / inflated_minor) ** 2)
        if f < min_f:
            min_f = f
    return (math.sqrt(min_f) - 1.0) * min(inflated_major, inflated_minor)


_BASELINE_ANCHOR_PROVIDER = AnchorGeometryProvider()
_BASELINE_LOCALIZATION_ERROR_MODEL = LocalizationErrorModel()
_BASELINE_LOCALIZATION_ESTIMATOR = IterativeLeastSquaresEstimator3D()
_SCENARIO_WORKER_MODE_SUMMARY = "[SCENARIO] zoneA=patrol zoneB=bottleneck zoneC=cross_traffic speed=0.4"


def worker_mode_summary_log() -> str:
    return _SCENARIO_WORKER_MODE_SUMMARY


def _build_localized_packet_from_anchor_pipeline(
    *,
    entity_type: str,
    entity_key: str,
    gt_position_3d: np.ndarray,
    now_s: float,
    sequence_number: int,
    rng: np.random.Generator,
) -> LocalizedStatePacket:
    anchors = _BASELINE_ANCHOR_PROVIDER.get_anchor_positions()
    true_ranges = np.linalg.norm(gt_position_3d[None, :] - anchors, axis=1)
    range_result = _BASELINE_LOCALIZATION_ERROR_MODEL.perturb_ranges(
        true_ranges,
        rng,
        entity_key=entity_key,
        timestamp=now_s,
    )
    estimate = _BASELINE_LOCALIZATION_ESTIMATOR.estimate(
        anchors=anchors,
        measured_ranges=range_result.measured_ranges,
        sigma_values=range_result.sigma_values,
        bias_values=range_result.bias_values,
        initial_guess=_BASELINE_ANCHOR_PROVIDER.get_workspace_center(),
        true_ranges=true_ranges,
    )
    est_position = np.asarray(estimate.est_position_3d, dtype=float).copy()
    return LocalizedStatePacket(
        entity_type=str(entity_type),
        sequence_number=int(sequence_number),
        state_generation_timestamp=float(now_s),
        gt_position_3d=gt_position_3d.copy(),
        estimated_position_3d=est_position,
        localization_error_vector_3d=(est_position - gt_position_3d).copy(),
        range_residuals=np.asarray(estimate.range_residuals, dtype=float).copy(),
        range_residual_rms_m=float(estimate.range_residual_rms_m),
        normalized_range_residual_rms=float(estimate.normalized_range_residual_rms),
        gt_user_position_3d=np.zeros(3, dtype=float),
        est_user_position_3d=None,
        anchor_positions_3d=anchors.copy(),
        true_ranges=range_result.true_ranges.copy(),
        measured_ranges=range_result.measured_ranges.copy(),
        bias_values=range_result.bias_values.copy(),
        sigma_values=range_result.sigma_values.copy(),
        random_noise_values=range_result.random_noise_values.copy(),
        jacobian_h_3d=np.asarray(estimate.jacobian_h_3d, dtype=float).copy(),
        P_3d=np.asarray(estimate.P_3d, dtype=float).copy(),
        b_3d=np.asarray(estimate.b_3d, dtype=float).copy(),
        M_3d=np.asarray(estimate.M_3d, dtype=float).copy(),
        P_xy=np.asarray(estimate.P_xy, dtype=float).copy(),
        b_xy=np.asarray(estimate.b_xy, dtype=float).copy(),
        M_xy=np.asarray(estimate.M_xy, dtype=float).copy(),
        confidence_alpha=0.95,
        est_position_timestamp=float(now_s),
    )


def compute_obstacle_envelope_states(scene: BaselineScene, now_s: float) -> List[ObstacleEnvelopeState]:
    states: List[ObstacleEnvelopeState] = []
    for idx, obstacle in enumerate(scene.obstacles):
        if scene.id == "SCENE_MANUAL_WORKER_CONTROL":
            gt_x, gt_y = (float(obstacle.gt_x), float(obstacle.gt_y))
        elif scene.id == "SCENE_FIXED_W13_MANUAL_W2":
            if str(obstacle.id) == "worker_1":
                gt_x, gt_y = (3.0, 4.0)
            elif str(obstacle.id) == "worker_3":
                gt_x, gt_y = (7.0, 3.7)
            else:
                # Keep worker_2 exactly at the manual-control scene default/static location.
                gt_x, gt_y = (float(obstacle.gt_x), float(obstacle.gt_y))
        elif scene.id == "SCENE3":
            gt_x, gt_y = (float(obstacle.gt_x), float(obstacle.gt_y))
        else:
            gt_x, gt_y = _scripted_worker_gt_xy(obstacle.id, now_s, fallback_xy=(obstacle.gt_x, obstacle.gt_y))
        gt = np.asarray([float(gt_x), float(gt_y), 0.0], dtype=float)
        # deterministic seed per worker and 10Hz simulation tick.
        seed = (hash(obstacle.id) & 0xFFFFFFFF) ^ int(max(0.0, now_s) * 10.0)
        rng = np.random.default_rng(seed)
        packet = _build_localized_packet_from_anchor_pipeline(
            entity_type="obstacle",
            entity_key=f"obstacle:{obstacle.id}",
            gt_position_3d=gt,
            now_s=now_s,
            sequence_number=idx + 1,
            rng=rng,
        )
        envelope = build_safety_envelope(packet)
        est_x = float(packet.estimated_position_3d[0])
        est_y = float(packet.estimated_position_3d[1])

        states.append(
            ObstacleEnvelopeState(
                id=obstacle.id,
                gt_xy=(float(gt_x), float(gt_y)),
                est_xy=(est_x, est_y),
                matrix_xy=((float(packet.M_xy[0][0]), float(packet.M_xy[0][1])), (float(packet.M_xy[1][0]), float(packet.M_xy[1][1]))),
                localization_packet=packet,
                localization_pipeline_function="IterativeLeastSquaresEstimator3D.estimate",
                envelope_builder_function="build_safety_envelope",
                envelope=envelope,
            )
        )
    return states


def _scripted_worker_gt_xy(worker_id: str, now_s: float, fallback_xy: Tuple[float, float]) -> Tuple[float, float]:
    t = max(0.0, float(now_s))
    if worker_id == "worker_3" and _debug_force_close_worker_enabled():
        # Debug-only mode: force worker_3 near UAV start region to validate
        # collision-probability pipeline end-to-end.
        return (1.35, 1.0)
    if worker_id == "worker_1":
        # zone_A patrol loop that repeatedly traverses link corridors between checkpoints.
        waypoints = [(2.2, 10.0), (3.2, 10.8), (4.4, 9.2), (3.4, 8.0), (2.0, 8.8), (1.8, 9.8)]
        pt = _sample_polyline_loop(waypoints, speed_mps=WORKER_DEFAULT_SPEED_MPS, t=t, smooth_turn=True)
        return _avoid_checkpoint_overlap(pt)
    if worker_id == "worker_2":
        # zone_B bottleneck shuttle that repeatedly blocks inter-cluster channel.
        waypoints = [(6.7, 9.0), (8.5, 9.3), (10.9, 9.2), (8.5, 8.7), (6.7, 8.8)]
        pt = _sample_polyline_pingpong(waypoints, speed_mps=WORKER_DEFAULT_SPEED_MPS, t=t, smooth_turn=True)
        return _avoid_checkpoint_overlap(pt)
    if worker_id == "worker_3":
        # zone_C cross-traffic weaving route that cuts through common shortest paths.
        waypoints = [(0.8, 2.3), (3.2, 5.2), (6.1, 2.4), (8.9, 5.0), (11.4, 2.8), (8.7, 0.9), (5.4, 3.1), (2.3, 0.9)]
        pt = _sample_polyline_loop(waypoints, speed_mps=WORKER_DEFAULT_SPEED_MPS, t=t, smooth_turn=True)
        return _avoid_checkpoint_overlap(pt)
    return (float(fallback_xy[0]), float(fallback_xy[1]))


def _sample_polyline_loop(waypoints: List[Tuple[float, float]], speed_mps: float, t: float, smooth_turn: bool = False) -> Tuple[float, float]:
    if len(waypoints) < 2:
        return waypoints[0]
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
    total_len = 0.0
    for i in range(len(waypoints)):
        p0 = waypoints[i]
        p1 = waypoints[(i + 1) % len(waypoints)]
        seg_len = math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1]))
        if seg_len > 1e-9:
            segments.append((p0, p1, seg_len))
            total_len += seg_len
    if total_len <= 1e-9:
        return waypoints[0]
    dist = (float(speed_mps) * float(t)) % total_len
    for p0, p1, seg_len in segments:
        if dist <= seg_len:
            ratio = dist / seg_len
            if smooth_turn:
                ratio = 0.5 - (0.5 * math.cos(math.pi * max(0.0, min(1.0, ratio))))
            return (p0[0] + (p1[0] - p0[0]) * ratio, p0[1] + (p1[1] - p0[1]) * ratio)
        dist -= seg_len
    return segments[-1][1]


def _sample_polyline_pingpong(waypoints: List[Tuple[float, float]], speed_mps: float, t: float, smooth_turn: bool = False) -> Tuple[float, float]:
    if len(waypoints) < 2:
        return waypoints[0]
    forward = list(waypoints)
    backward = list(reversed(waypoints[1:-1]))
    full_path = forward + backward
    return _sample_polyline_loop(full_path, speed_mps=speed_mps, t=t, smooth_turn=smooth_turn)


def _avoid_checkpoint_overlap(point_xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y = float(point_xy[0]), float(point_xy[1])
    # Keep worker circle fully outside checkpoint circles.
    hard_clearance = float(CHECKPOINT_RADIUS_M + WORKER_RADIUS_M + 0.02)
    influence_radius = hard_clearance + 0.55
    for _ in range(3):
        push_x = 0.0
        push_y = 0.0
        for checkpoint in BENCHMARK_CHECKPOINTS:
            dx = x - float(checkpoint.x)
            dy = y - float(checkpoint.y)
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                dx, dy, dist = 1.0, 0.0, 1.0
            if dist < influence_radius:
                # Smooth soft repulsion in influence band.
                softness = max(0.0, influence_radius - dist) / influence_radius
                gain = (softness * softness) * 0.30
                push_x += (dx / dist) * gain
                push_y += (dy / dist) * gain
            if dist < hard_clearance:
                # Hard projection to avoid any overlap.
                scale = (hard_clearance - dist) / dist
                push_x += dx * scale
                push_y += dy * scale
        x += push_x
        y += push_y
    x = min(11.9, max(0.1, x))
    y = min(11.9, max(0.1, y))
    return (x, y)


def _debug_force_close_worker_enabled() -> bool:
    value = str(os.getenv("DEBUG_FORCE_CLOSE_WORKER", "false")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def evaluate_path_clear(
    drone_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    user_xy: Optional[Tuple[float, float]],
    user_radius_m: float,
    user_envelope: Optional[SafetyEnvelope2D],
    obstacle_envelopes: List[ObstacleEnvelopeState],
    corridor_half_width_m: float = 0.35,
) -> PathClearResult:
    ax, ay = float(drone_xy[0]), float(drone_xy[1])
    bx, by = float(target_xy[0]), float(target_xy[1])

    blockers: List[Tuple[str, float]] = []

    if user_envelope is not None:
        blockers.append(
            (
                "user",
                _signed_gap_segment_to_ellipse_envelope(
                    ax=ax,
                    ay=ay,
                    bx=bx,
                    by=by,
                    center_x=float(user_envelope.center_xy[0]),
                    center_y=float(user_envelope.center_xy[1]),
                    major_axis_radius=float(user_envelope.major_axis_radius),
                    minor_axis_radius=float(user_envelope.minor_axis_radius),
                    orientation_deg=float(user_envelope.orientation_deg),
                    corridor_half_width_m=float(corridor_half_width_m),
                ),
            )
        )
    elif user_xy is not None:
        center_distance = _distance_point_to_segment(float(user_xy[0]), float(user_xy[1]), ax, ay, bx, by)
        blockers.append(("user", center_distance - (float(user_radius_m) + float(corridor_half_width_m))))

    for obstacle_state in obstacle_envelopes:
        blockers.append(
            (
                obstacle_state.id,
                _signed_gap_segment_to_obstacle_envelope(
                    ax,
                    ay,
                    bx,
                    by,
                    obstacle_state=obstacle_state,
                    corridor_half_width_m=float(corridor_half_width_m),
                ),
            )
        )

    if not blockers:
        return PathClearResult(True, "none", float("inf"), tuple())

    blockers_sorted = sorted(blockers, key=lambda item: item[1])
    min_entity, min_gap = blockers_sorted[0]
    overlapping = tuple(entity for entity, gap in blockers_sorted if gap < 0.0)
    return PathClearResult(min_gap >= 0.0, "none" if min_gap >= 0.0 else min_entity, float(min_gap), overlapping)


def build_scene_expectations(
    scene: BaselineScene,
    user_radius_m: float,
    corridor_half_width_m: float,
    high_risk: bool,
    now_s: float = 0.0,
) -> List[BaselineExpectation]:
    drone_xy = (float(scene.drone_initial_pose[0]), float(scene.drone_initial_pose[1]))
    user_xy = (float(scene.user_position[0]), float(scene.user_position[1]))
    obstacle_states = compute_obstacle_envelope_states(scene, now_s=now_s)

    rows: List[BaselineExpectation] = []
    for point in scene.task_points:
        result = evaluate_path_clear(
            drone_xy=drone_xy,
            target_xy=(float(point.x), float(point.y)),
            user_xy=user_xy,
            user_radius_m=float(user_radius_m),
            user_envelope=None,
            obstacle_envelopes=obstacle_states,
            corridor_half_width_m=float(corridor_half_width_m),
        )
        direct = bool(result.path_clear and not high_risk)
        rows.append(
            BaselineExpectation(
                scene_id=scene.id,
                target_task_point=point.id,
                expected_path_clear=bool(result.path_clear),
                expected_blocking_entity=result.blocking_entity,
                expected_motion_mode="direct_go_to" if direct else "staged_detour",
            )
        )
    return rows


def build_all_scene_expectations(
    user_radius_m: float,
    corridor_half_width_m: float,
    high_risk: bool,
    now_s: float = 0.0,
) -> List[BaselineExpectation]:
    rows: List[BaselineExpectation] = []
    for scene in BASELINE_SCENES.values():
        rows.extend(
            build_scene_expectations(
                scene=scene,
                user_radius_m=user_radius_m,
                corridor_half_width_m=corridor_half_width_m,
                high_risk=high_risk,
                now_s=now_s,
            )
        )
    return rows


BASELINE_SCENES: Dict[str, BaselineScene] = {
    "SCENE_BENCHMARK_DEMO": BaselineScene(
        id="SCENE_BENCHMARK_DEMO",
        drone_initial_pose=(1.0, 1.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 1.5, 10.5, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 8.5, 7.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 2.0, 3.2, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="Deterministic benchmark demo scene with fixed workers and fixed checkpoint order.",
    ),
    "SCENE_MANUAL_WORKER_CONTROL": BaselineScene(
        id="SCENE_MANUAL_WORKER_CONTROL",
        drone_initial_pose=(1.0, 1.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 1.5, 10.5, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 8.5, 7.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 2.0, 3.2, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="Manual worker-control scene: worker scripted motion is disabled and UI controls drive worker poses.",
    ),
    "SCENE_FIXED_W13_MANUAL_W2": BaselineScene(
        id="SCENE_FIXED_W13_MANUAL_W2",
        drone_initial_pose=(1.0, 1.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 3.0, 4.0, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 8.5, 7.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 7.0, 3.7, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="worker_1 fixed at (3,4), worker_3 fixed at (7,3.7); worker_2 stays at manual-control default location.",
    ),
    "SCENE1": BaselineScene(
        id="SCENE1",
        drone_initial_pose=(4.0, 6.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 1.5, 10.5, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 8.5, 7.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 2.0, 3.2, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="scene1 based on SCENE_BENCHMARK_DEMO with UAV true start changed to (4,6).",
    ),
    "SCENE2": BaselineScene(
        id="SCENE2",
        drone_initial_pose=(6.0, 6.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 1.5, 10.5, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 8.5, 7.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 2.0, 3.2, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="scene2 based on SCENE_BENCHMARK_DEMO with UAV true start changed to (6,6).",
    ),
    "SCENE3": BaselineScene(
        id="SCENE3",
        drone_initial_pose=(1.0, 1.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.8, 10.2, 0.0),
        user_initial_yaw_rad=-2.0,
        task_points=(
            TaskPoint("A", 1.4, 10.6, -1.5),
            TaskPoint("B", 7.4, 10.6, -1.5),
            TaskPoint("C", 1.7, 4.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("worker_1", 3.5, 3.0, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_2", 9.5, 3.8, cov_xy=((0.010, 0.000), (0.000, 0.008))),
            StaticObstacle("worker_3", 7.0, 2.2, cov_xy=((0.010, 0.000), (0.000, 0.008))),
        ),
        notes="scene3 based on SCENE_FIXED_W13_MANUAL_W2 with worker true positions overridden.",
    ),
    "SCENE_1_CLEAR_PATH": BaselineScene(
        id="SCENE_1_CLEAR_PATH",
        drone_initial_pose=(1.0, 1.5, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.7, 10.6, 0.0),
        user_initial_yaw_rad=-2.4,
        task_points=(
            TaskPoint("A", 10.0, 1.6, -1.5),
            TaskPoint("B", 9.7, 4.9, -1.5),
            TaskPoint("C", 9.4, 7.9, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 3.2, 9.8, cov_xy=((0.010, 0.001), (0.001, 0.008)), est_bias_x_m=0.03, est_bias_y_m=-0.02),
            StaticObstacle("O2", 5.8, 10.0, cov_xy=((0.011, -0.001), (-0.001, 0.009)), est_bias_x_m=-0.02, est_bias_y_m=0.03),
            StaticObstacle("O3", 8.0, 9.3, cov_xy=((0.009, 0.0015), (0.0015, 0.008)), est_bias_x_m=0.02, est_bias_y_m=0.02),
        ),
        notes="Bottom corridor to A is intentionally clear from user and obstacle envelopes.",
    ),
    "SCENE_2_OBSTACLE_BLOCKS": BaselineScene(
        id="SCENE_2_OBSTACLE_BLOCKS",
        drone_initial_pose=(1.3, 2.5, -1.5),
        drone_initial_yaw_rad=0.06,
        user_position=(10.1, 9.3, 0.0),
        user_initial_yaw_rad=-2.2,
        task_points=(
            TaskPoint("A", 9.7, 2.5, -1.5),
            TaskPoint("B", 8.9, 5.7, -1.5),
            TaskPoint("C", 8.5, 8.3, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 5.1, 2.5, cov_xy=((0.015, 0.001), (0.001, 0.012)), est_bias_x_m=0.02, est_bias_y_m=0.01),
            StaticObstacle("O2", 6.5, 5.9, cov_xy=((0.014, 0.002), (0.002, 0.012)), est_bias_x_m=-0.03, est_bias_y_m=0.01),
            StaticObstacle("O3", 3.1, 7.6, cov_xy=((0.012, -0.001), (-0.001, 0.010)), est_bias_x_m=0.01, est_bias_y_m=-0.02),
        ),
        notes="Obstacle O1 is positioned to cut the direct A corridor but still leaves side detour room.",
    ),
    "SCENE_3_USER_NEAR_CORRIDOR": BaselineScene(
        id="SCENE_3_USER_NEAR_CORRIDOR",
        drone_initial_pose=(1.4, 3.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(5.1, 3.05, 0.0),
        user_initial_yaw_rad=0.1,
        task_points=(
            TaskPoint("A", 9.3, 3.1, -1.5),
            TaskPoint("B", 8.9, 6.2, -1.5),
            TaskPoint("C", 8.6, 8.9, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 2.3, 8.7, cov_xy=((0.011, 0.000), (0.000, 0.009)), est_bias_x_m=-0.02, est_bias_y_m=0.02),
            StaticObstacle("O2", 6.8, 8.2, cov_xy=((0.014, 0.001), (0.001, 0.011)), est_bias_x_m=0.02, est_bias_y_m=-0.01),
            StaticObstacle("O3", 10.2, 5.9, cov_xy=((0.012, -0.001), (-0.001, 0.010)), est_bias_x_m=0.02, est_bias_y_m=0.02),
        ),
        notes="User envelope is intentionally near the A corridor to make user the primary blocker.",
    ),
    "SCENE_4_CORNER_CONSTRAINED": BaselineScene(
        id="SCENE_4_CORNER_CONSTRAINED",
        drone_initial_pose=(0.8, 11.1, -1.5),
        drone_initial_yaw_rad=-0.75,
        user_position=(1.9, 9.8, 0.0),
        user_initial_yaw_rad=-1.1,
        task_points=(
            TaskPoint("A", 1.5, 8.8, -1.5),
            TaskPoint("B", 2.4, 7.6, -1.5),
            TaskPoint("C", 3.6, 6.5, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 1.3, 10.1, cov_xy=((0.015, 0.001), (0.001, 0.012)), est_bias_x_m=0.02, est_bias_y_m=-0.01),
            StaticObstacle("O2", 2.3, 8.9, cov_xy=((0.018, 0.002), (0.002, 0.014)), est_bias_x_m=-0.02, est_bias_y_m=0.03),
            StaticObstacle("O3", 3.2, 9.7, cov_xy=((0.016, -0.002), (-0.002, 0.013)), est_bias_x_m=0.02, est_bias_y_m=0.02),
        ),
        notes="Upper-left corner compression + clustered obstacles create clear constrained geometry.",
    ),
}


def normalize_baseline_scene_id(scene_id: Optional[str]) -> str:
    candidate = (scene_id or "SCENE_BENCHMARK_DEMO").strip().upper()
    if candidate not in BASELINE_SCENES:
        return "SCENE_BENCHMARK_DEMO"
    return candidate


def get_task_point(scene: BaselineScene, task_point_id: str) -> Optional[TaskPoint]:
    target = (task_point_id or "").strip().upper()
    for point in scene.task_points:
        if point.id.upper() == target:
            return point
    return None
