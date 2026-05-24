import threading
from collections import deque

from controller.benchmark_layout import BENCHMARK_CHECKPOINTS_BY_ID
from controller.llm_controller import LLMController


def _build_minimal_controller(active_ids):
    c = LLMController.__new__(LLMController)
    c.active_objective_set = {"active_checkpoint_ids": list(active_ids)}
    c.latest_benchmark_progress = {"completed": [], "current_target": None}
    c._benchmark_completed = set()
    c._benchmark_active_enter_ts = None
    c._benchmark_active_enter_ts_by_checkpoint = {}
    c._benchmark_last_update_ts = None
    c._benchmark_last_distance_m = None
    c._benchmark_prev_target = None
    c._benchmark_prev_in_radius = False
    c._benchmark_prev_dwell_satisfied = False
    c._benchmark_prev_dwell_bucket = 0
    c._benchmark_prev_completed_ids = set()
    c._benchmark_focus_checkpoint_id = None
    c._benchmark_plan_checkpoint_sequence = []
    c._benchmark_executed_gc_sequence = []
    c._progress_event_cv = threading.Condition()
    c._progress_event_seq = 0
    c._progress_event_queue = deque(maxlen=256)
    c._progress_event_cursor_by_checkpoint = {}
    return c


def _tick_at_checkpoint(controller, monkeypatch, t, checkpoint_id):
    cp = BENCHMARK_CHECKPOINTS_BY_ID[checkpoint_id]
    monkeypatch.setattr("controller.llm_controller.time.time", lambda: t)
    controller._update_benchmark_progress_from_snapshot({"drone_gt": (cp.x, cp.y, 0.0)})


def test_fixed_order_completion(monkeypatch):
    seq = ["B1", "B2", "B3", "B4"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    t = 10.0
    for cp in seq:
        _tick_at_checkpoint(c, monkeypatch, t, cp)
        _tick_at_checkpoint(c, monkeypatch, t + 2.1, cp)
        t += 3.0
    completed = set(c.latest_benchmark_progress["completed"])
    assert completed == set(seq)
    assert c.latest_benchmark_progress["completed_flag"] is True


def test_non_fixed_order_completion_case1(monkeypatch):
    seq = ["B3", "B1", "B4", "B2"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    t = 20.0
    for cp in seq:
        _tick_at_checkpoint(c, monkeypatch, t, cp)
        assert c.latest_benchmark_progress["current_target"] == cp
        _tick_at_checkpoint(c, monkeypatch, t + 2.1, cp)
        t += 3.0
    assert set(c.latest_benchmark_progress["completed"]) == set(seq)


def test_non_fixed_order_completion_case2(monkeypatch):
    seq = ["B2", "B4", "B1", "B3"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    t = 30.0
    for cp in seq:
        _tick_at_checkpoint(c, monkeypatch, t, cp)
        _tick_at_checkpoint(c, monkeypatch, t + 2.1, cp)
        t += 3.0
    assert set(c.latest_benchmark_progress["completed"]) == set(seq)
    assert c.latest_benchmark_progress["completed_flag"] is True


def test_dwell_not_enough_should_not_complete(monkeypatch):
    seq = ["B3", "B1"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    _tick_at_checkpoint(c, monkeypatch, 100.0, "B3")
    _tick_at_checkpoint(c, monkeypatch, 101.0, "B3")
    # leave radius -> reset
    monkeypatch.setattr("controller.llm_controller.time.time", lambda: 101.5)
    c._update_benchmark_progress_from_snapshot({"drone_gt": (0.0, 0.0, 0.0)})
    assert "B3" not in set(c.latest_benchmark_progress["completed"])


def test_zone_completion_set_based(monkeypatch):
    seq = ["B3", "B1", "B4", "B2"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    t = 200.0
    for cp in seq:
        _tick_at_checkpoint(c, monkeypatch, t, cp)
        _tick_at_checkpoint(c, monkeypatch, t + 2.1, cp)
        t += 3.0
    assert c.latest_benchmark_progress["completed_flag"] is True


def test_executor_and_progress_sequence_alignment():
    seq = ["B3", "B1", "B4", "B2"]
    c = _build_minimal_controller(seq)
    c._benchmark_plan_checkpoint_sequence = list(seq)
    c._benchmark_executed_gc_sequence = list(seq)
    c.update_benchmark_progress(
        completed_checkpoint_ids=[],
        current_target_checkpoint="B3",
        in_radius=False,
        dwell_seconds=0.0,
        required_dwell_seconds=2.0,
        dwell_satisfied=False,
    )
    assert c.latest_benchmark_progress["plan_checkpoint_sequence"] == seq
    assert c.latest_benchmark_progress["executed_gc_sequence"] == seq
