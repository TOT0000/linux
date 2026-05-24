import math

import pytest


pytest.importorskip("PIL")
from controller import llm_controller as llm_mod
from controller.llm_controller import LLMController
from controller.baseline_scenes import BENCHMARK_CHECKPOINTS_BY_ID


class _DroneKinematics:
    def __init__(self, x: float, y: float, yaw_rad: float):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw_rad)
        self.actions = []

    def move_forward(self, dist: float):
        d = float(dist)
        self.actions.append(("move_forward", d))
        self.x += math.cos(self.yaw) * d
        self.y += math.sin(self.yaw) * d

    def move_left(self, dist: float):
        d = float(dist)
        self.actions.append(("move_left", d))
        self.x += math.cos(self.yaw + math.pi / 2.0) * d
        self.y += math.sin(self.yaw + math.pi / 2.0) * d

    def move_right(self, dist: float):
        d = float(dist)
        self.actions.append(("move_right", d))
        self.x += math.cos(self.yaw - math.pi / 2.0) * d
        self.y += math.sin(self.yaw - math.pi / 2.0) * d

    def turn_ccw(self, deg: int):
        self.actions.append(("turn_ccw", int(deg)))
        self.yaw += math.radians(float(deg))

    def turn_cw(self, deg: int):
        self.actions.append(("turn_cw", int(deg)))
        self.yaw -= math.radians(float(deg))


class _FakePx4Base:
    pass


class _FakePx4Drone(_FakePx4Base, _DroneKinematics):
    pass


def _build_controller(drone: _DroneKinematics) -> LLMController:
    controller = LLMController.__new__(LLMController)
    controller.drone = drone
    controller._benchmark_executed_gc_sequence = []
    controller.set_benchmark_progress_focus_checkpoint = lambda _cp: None
    controller._maybe_run_agent_heartbeat = lambda: False
    controller._should_trigger_auto_replan = lambda _p, source=None: False

    def _snapshot():
        return {
            "drone_gt": (drone.x, drone.y, 0.0),
            "drone_est": (drone.x, drone.y, 0.0),
            "drone_est_bias_corrected": (drone.x, drone.y, 0.0),
            "drone_yaw_rad": drone.yaw,
            "benchmark_progress": {"completed": [], "current_target": None, "in_radius": False},
            "safety_context": None,
        }

    controller.get_live_ui_snapshot = _snapshot
    return controller


def test_gc_aligns_heading_before_forward_in_px4_sim(monkeypatch):
    monkeypatch.setattr(llm_mod, "Px4SimRobotWrapper", _FakePx4Base)

    cp = BENCHMARK_CHECKPOINTS_BY_ID["A1"]
    drone = _FakePx4Drone(float(cp.x) - 1.0, float(cp.y), math.pi / 2.0)
    controller = _build_controller(drone)

    summary, should_replan = controller.skill_go_checkpoint("A1")

    assert should_replan is False
    assert "reached" in summary

    first_forward_idx = next(i for i, (name, _) in enumerate(drone.actions) if name == "move_forward")
    assert first_forward_idx > 0
    assert all(name in {"turn_ccw", "turn_cw"} for name, _ in drone.actions[:first_forward_idx])
    assert not any(name in {"move_left", "move_right"} for name, _ in drone.actions)
