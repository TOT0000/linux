import queue
import time
import sys, os
import asyncio
import io, time
import math
import glob
from collections import deque
import gradio as gr
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非互動後端避免開啟GUI視窗
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Arc
from PIL import Image
from threading import Thread
from flask import Flask, Response, request

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

from controller.llm_controller import LLMController
from controller.utils import print_debug, print_t
from controller.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType
from controller.llm_controller import (
    MODE_TYPEFLY_ONESHOT,
    MODE_TYPEFLY_THRESHOLD_REPLAN,
    MODE_AGENT_HEARTBEAT_SOFT,
    MODE_AGENT_HEARTBEAT_HARDGATE,
)
from controller.experiment_scenarios import SCENARIOS, normalize_scenario_name
from controller.baseline_scenes import BASELINE_SCENES, normalize_baseline_scene_id
from controller.pipeline_registry import PIPELINE_REGISTRY, normalize_pipeline_id
from controller.anchor_provider import AnchorGeometryProvider
from controller.benchmark_layout import (
    WORKSPACE_SIZE_M,
    CHECKPOINT_DWELL_SECONDS,
    CHECKPOINT_RADIUS_M,
    UAV_RADIUS_M,
    WORKER_RADIUS_M,
    BENCHMARK_CHECKPOINT_ORDER,
    BENCHMARK_CHECKPOINTS_BY_ID,
)
from gradio import Timer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TypeFly:
    def __init__(self, robot_type, use_http=False, enable_video=False, backend="uwb", initial_scenario="SAFE"):
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self.message_queue = queue.Queue()
        self.uwb_queue = queue.Queue(maxsize=500)
        self.virtual_queue = queue.Queue(maxsize=500)

        controller_robot_type = RobotType.PX4_SIM if backend == "sim" else robot_type
        self.llm_controller = LLMController(controller_robot_type, self.virtual_queue, use_http, self.message_queue, enable_video=enable_video)
        self.llm_controller.register_position_callback(self.receive_position)
        self.llm_controller.set_selected_pipeline(normalize_pipeline_id(os.getenv("TYPEFLY_BASELINE_ID", "baseline1")))
        self.llm_controller.set_archive_enabled(True)
        self.active_scenario = self.llm_controller.set_active_scenario(initial_scenario)
        self.active_baseline_scene = self.llm_controller.set_baseline_scene(
            normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_BENCHMARK_DEMO"))
        )
        
        self.system_stop = False
        self.ui_css = """
            .gradio-container, .gr-markdown, .gr-image, .gr-image img, .prose {
                opacity: 1 !important;
                filter: none !important;
                transition: none !important;
                animation: none !important;
            }
            .user-move-panel {
                max-width: 340px;
                margin: 0 auto;
            }
            .user-move-step {
                margin-bottom: 6px !important;
            }
            .scenario-panel {
                max-width: 320px;
            }
            .user-move-row {
                justify-content: center;
                gap: 8px;
                margin: 1px 0 !important;
            }
            .user-move-btn button {
                width: 96px !important;
                min-width: 96px !important;
                padding: 6px 8px !important;
            }
            """
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False
        self.robot_type = controller_robot_type
        self.selected_framework_mode = MODE_TYPEFLY_ONESHOT
        self.selected_baseline_id = normalize_pipeline_id(os.getenv("TYPEFLY_BASELINE_ID", "baseline1"))
        self.archive_enabled = True
        self.selected_worker_move_step = 0.5
        self.selected_worker_turn_step = 15.0

        # 狀態資料
        self.anchor_count = 0
        self.anchor_input_history = ""
        self.position_history = {
            "drone_gt": deque(maxlen=100),
            "drone_est": deque(maxlen=100),
        }
        self.worker_collision_history = {
            "worker_1": deque(maxlen=100),
            "worker_2": deque(maxlen=100),
            "worker_3": deque(maxlen=100),
        }
        self.worker_collision_active = {
            "worker_1": False,
            "worker_2": False,
            "worker_3": False,
        }
        self.mission_collision_count = 0
        self.plot_style = {
            "drone": {"main": "#0B57D0", "light": "#8AB4F8"},
            "user": {"main": "#C5221F", "light": "#F28B82"},
        }
        self.anchor_provider = AnchorGeometryProvider()
        self.benchmark_progress = {
            "order": list(BENCHMARK_CHECKPOINT_ORDER),
            "completed": set(),
            "active_enter_ts": None,
            "active_progress": 0.0,
            "current_target": BENCHMARK_CHECKPOINT_ORDER[0] if BENCHMARK_CHECKPOINT_ORDER else None,
            "executed_gc_sequence": [],
        }
        self.objective_state = {
            # Active objective set is explicitly tracked for future framework linkage.
            "active_checkpoint_ids": set(BENCHMARK_CHECKPOINT_ORDER),
            "active_zone_ids": {"zone_A", "zone_B", "zone_C"},
        }
        self.mission_clock = {
            "started_at": None,
            "completed_at": None,
            "is_running": False,
            "objective_completed": False,
        }

        # 浮動提示 internal state
        self._temp_message = ""
        self._temp_message_expire = 0.0
        '''
        default_sentences = [
            "Find something I can eat.",
            "What you can see?",
            "Follow that ball for 20 seconds",
            "Find a chair for me.",
            "Go to the chair without book."
        ]
        '''

        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())

            # 浮動提示（頂端）
            self.message_markdown = gr.Markdown(value="", visible=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=260, elem_classes="scenario-panel"):
                    self.framework_mode_selector = gr.Dropdown(
                        choices=[
                            MODE_TYPEFLY_ONESHOT,
                            MODE_TYPEFLY_THRESHOLD_REPLAN,
                            MODE_AGENT_HEARTBEAT_SOFT,
                            MODE_AGENT_HEARTBEAT_HARDGATE,
                        ],
                        value=MODE_TYPEFLY_ONESHOT,
                        label="Execution Mode",
                    )
                    self.baseline_selector = gr.Dropdown(
                        choices=[(cfg.name, cfg.id) for cfg in PIPELINE_REGISTRY.values()],
                        value=self.selected_baseline_id,
                        label="Baseline Pipeline",
                    )
                    self.postrun_summary = gr.Markdown(value="### Post-run archive\nNo finished run awaiting decision.")
                    with gr.Row():
                        self.save_run_btn = gr.Button("Save this run", variant="primary")
                        self.discard_run_btn = gr.Button("Discard this run")
                    baseline_scene_choices = [
                        sid for sid in (
                            "SCENE_BENCHMARK_DEMO",
                            "SCENE_MANUAL_WORKER_CONTROL",
                            "SCENE_FIXED_W13_MANUAL_W2",
                            "SCENE1",
                            "SCENE2",
                            "SCENE3",
                        ) if sid in BASELINE_SCENES
                    ]
                    self.baseline_scene_selector = gr.Dropdown(
                        choices=baseline_scene_choices,
                        value=normalize_baseline_scene_id(os.getenv("TYPEFLY_BASELINE_SCENE", "SCENE_BENCHMARK_DEMO")),
                        label="Baseline Scene",
                    )
                    self.baseline_scene_apply_btn = gr.Button("Apply Baseline Scene")
                    self.reset_system_btn = gr.Button("Reset System (Clear All Records)")
                with gr.Column(scale=1, min_width=320, elem_classes="user-move-panel"):
                    self.worker_selector = gr.Dropdown(
                        choices=["worker_1", "worker_2", "worker_3"],
                        value="worker_1",
                        label="Controlled Worker",
                    )
                    self.user_move_step = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Worker Move Step (m)",
                        elem_classes="user-move-step",
                    )
                    self.user_turn_step = gr.Slider(
                        minimum=5,
                        maximum=90,
                        value=15,
                        step=5,
                        label="Worker Turn Step (deg)",
                        elem_classes="user-move-step",
                    )
                    with gr.Row(elem_classes="user-move-row"):
                        gr.Markdown("")
                        self.user_move_forward_btn = gr.Button("Forward", elem_classes="user-move-btn")
                        gr.Markdown("")
                    with gr.Row(elem_classes="user-move-row"):
                        self.user_move_left_btn = gr.Button("Left", elem_classes="user-move-btn")
                        gr.Markdown("")
                        self.user_move_right_btn = gr.Button("Right", elem_classes="user-move-btn")
                    with gr.Row(elem_classes="user-move-row"):
                        gr.Markdown("")
                        self.user_move_backward_btn = gr.Button("Backward", elem_classes="user-move-btn")
                        gr.Markdown("")
                    with gr.Row(elem_classes="user-move-row"):
                        self.user_turn_ccw_btn = gr.Button("Turn Counter Clockwise", elem_classes="user-move-btn")
                        self.user_turn_cw_btn = gr.Button("Turn Clockwise", elem_classes="user-move-btn")
            self.scenario_status = gr.Markdown(value="")

            self.baseline_scene_apply_btn.click(
                fn=self.apply_baseline_scene,
                inputs=[self.baseline_scene_selector],
                outputs=[self.scenario_status],
            )
            self.reset_system_btn.click(
                fn=self.reset_system_state,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.framework_mode_selector.change(
                fn=self.set_framework_mode,
                inputs=[self.framework_mode_selector],
                outputs=[self.scenario_status],
            )
            self.baseline_selector.change(
                fn=self.set_selected_baseline,
                inputs=[self.baseline_selector],
                outputs=[self.scenario_status],
            )
            self.save_run_btn.click(
                fn=self.save_last_run,
                inputs=[],
                outputs=[self.scenario_status, self.postrun_summary],
            )
            self.discard_run_btn.click(
                fn=self.discard_last_run,
                inputs=[],
                outputs=[self.scenario_status, self.postrun_summary],
            )
            self.worker_selector.change(
                fn=self.select_controlled_worker,
                inputs=[self.worker_selector],
                outputs=[self.scenario_status],
            )
            self.user_move_step.change(
                fn=self.set_worker_move_step,
                inputs=[self.user_move_step],
                outputs=[],
            )
            self.user_turn_step.change(
                fn=self.set_worker_turn_step,
                inputs=[self.user_turn_step],
                outputs=[],
            )

            self.user_move_forward_btn.click(
                fn=self.move_worker_forward,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.user_move_backward_btn.click(
                fn=self.move_worker_backward,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.user_move_left_btn.click(
                fn=self.move_worker_left,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.user_move_right_btn.click(
                fn=self.move_worker_right,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.user_turn_cw_btn.click(
                fn=self.turn_worker_cw,
                inputs=[],
                outputs=[self.scenario_status],
            )
            self.user_turn_ccw_btn.click(
                fn=self.turn_worker_ccw,
                inputs=[],
                outputs=[self.scenario_status],
            )

            # floating message refresher
            self.message_timer = Timer(value=0.5)
            self.message_timer.tick(
                fn=self._refresh_temp_message,
                inputs=[],
                outputs=[self.message_markdown]
            )

            with gr.Row():
                with gr.Column(scale=2, min_width=320):
                    self.anchor_3d_plot = gr.Image(
                        value=self.create_blank_plot("Anchor 3D Layout", "X (m)", "Y (m)", xlim=(0, 12), ylim=(0, 12), figsize=(5, 4)),
                        label="Anchor 3D Panel",
                        height=360,
                    )
                    self.toggle_error_ellipse = gr.Checkbox(label="Show variance error ellipse", value=False)
                    self.toggle_raw_estimate = gr.Checkbox(label="Debug: show raw estimate", value=False)
                with gr.Column(scale=4, min_width=520):
                    self.global_xy_plot = gr.Image(
                        value=self.create_blank_plot(
                            "Benchmark Workspace XY",
                            "X (m)",
                            "Y (m)",
                            xlim=(0, 12),
                            ylim=(0, 12),
                            figsize=(10, 8),
                        ),
                        label="Main XY Workspace",
                        height=640,
                    )
                with gr.Column(scale=2, min_width=300):
                    self.status_markdown = gr.Markdown(value="### Status\nWaiting for live data...")
                    self.entity_markdown = gr.Markdown(value="### Entity positions\nWaiting for live data...")

            with gr.Row():
                self.xy_plot = gr.Image(value=self.create_blank_plot("Local XY", "X (m)", "Y (m)", xlim=(0, 12), ylim=(0, 12), figsize=(5, 4)), label="Local XY", height=320)
                self.x_plot = gr.Image(value=self.create_sequence_plot("worker_1 3s Predicted Collision Probability", "Sample", "P(predicted collision)", xlim=(0, 1), ylim=(0, 1)), label="worker_1 P(predicted collision)", height=320)
                self.y_plot = gr.Image(value=self.create_sequence_plot("worker_2 3s Predicted Collision Probability", "Sample", "P(predicted collision)", xlim=(0, 1), ylim=(0, 1)), label="worker_2 P(predicted collision)", height=320)
                self.z_plot = gr.Image(value=self.create_sequence_plot("worker_3 3s Predicted Collision Probability", "Sample", "P(predicted collision)", xlim=(0, 1), ylim=(0, 1)), label="worker_3 P(predicted collision)", height=320)

            self.counter = gr.State(0)
            self.timer = Timer(value=0.08)
            self.timer.tick(
                fn=self.update_and_step,
                inputs=[self.counter, self.toggle_error_ellipse, self.toggle_raw_estimate],
                outputs=[
                    self.anchor_3d_plot,
                    self.global_xy_plot,
                    self.xy_plot,
                    self.x_plot,
                    self.y_plot,
                    self.z_plot,
                    self.counter,
                    self.status_markdown,
                    self.entity_markdown,
                    self.postrun_summary,
                ]
            )

            self.chat = gr.ChatInterface(self.process_message, fill_height=False)

    def show_temporary_message(self, text, duration=3):
        self._temp_message = text
        self._temp_message_expire = time.time() + duration

    def _refresh_temp_message(self):
        if hasattr(self, "_temp_message") and time.time() < self._temp_message_expire:
            return gr.update(value=f"**{self._temp_message}**", visible=True)
        else:
            return gr.update(value="", visible=False)
            
    def reset_anchors(self):
        # 同步到 UWBWrapper 透過 controller
        if hasattr(self, "llm_controller") and hasattr(self.llm_controller, "uwb"):
            self.llm_controller.uwb.reset_anchors()

        self.anchor_input_history = ""
        self.show_temporary_message("Reset anchor count and positions.")
        # 清空 input，恢復 submit 按鈕可用，anchor line 按鈕不可用
        return "", "", "", gr.update(interactive=True), gr.update(interactive=True)


    def receive_position(self, *args):
        if len(args) == 4:
            x, y, z, source = args
        elif len(args) == 3:
            x, y, z = args
            source = "unknown"
        else:
            return

        timestamp = time.time()
        tag = "[DronePos]" if source == "drone" else "[Pos]"

        # 初始化紀錄字典
        if not hasattr(self, '_last_position_map'):
            self._last_position_map = {}

        current_pos = (x, y, z)
        last_pos = self._last_position_map.get(source)

        if current_pos != last_pos:
            # 有改變才放入 queue
            try:
                if source == "user":
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                elif source == "drone":
                    self.virtual_queue.put_nowait((timestamp, x, y, z))
            except queue.Full:
                if source == "user":
                    self.uwb_queue.get()
                    self.uwb_queue.put_nowait((timestamp, x, y, z))
                elif source == "drone":
                    self.virtual_queue.get()
                    self.virtual_queue.put_nowait((timestamp, x, y, z))

            # 每種來源各自印出，每 5 秒一次
            if not hasattr(self, '_last_print_position_map'):
                self._last_print_position_map = {}

            last_print_time = self._last_print_position_map.get(source, 0)
            if source == "drone" and timestamp - last_print_time > 5:
                print_debug(f"{tag} x={x:.2f}, y={y:.2f}, z={z:.2f}")
                self._last_print_position_map[source] = timestamp

            # 更新最後位置
            self._last_position_map[source] = current_pos



    def set_anchor_count(self, anchor_count_input):
        try:
            n = int(anchor_count_input)
            if n <= 0:
                self.show_temporary_message("Anchor count must be a positive integer.")
                return anchor_count_input, gr.update(), gr.update(value=self.anchor_input_history), gr.update(interactive=True), gr.update(interactive=True)
            self.llm_controller = getattr(self, "llm_controller", None)
            # call underlying wrapper
            self.llm_controller.uwb.set_anchor_count(n)
            self.anchor_count = n
            self.anchor_input_history = ""
            self.show_temporary_message(f"Anchor count set to {n}. Please enter anchor 1's position.")
            return anchor_count_input, gr.update(value=""), gr.update(value=""), gr.update(interactive=False), gr.update(interactive=True)
        except Exception as e:
            self.show_temporary_message(f"Failed to set anchor count: {e}")
            return anchor_count_input, gr.update(), gr.update(value=self.anchor_input_history), gr.update(interactive=True), gr.update(interactive=True)

    def input_anchor_line(self, line_str):
        if self.llm_controller.uwb.anchor_count <= 0:
            self.show_temporary_message("Please submit a valid anchor count first.")
            return "", self.anchor_input_history, gr.update(interactive=True)

        line_str = line_str.strip()
        if not line_str:
            self.show_temporary_message("Anchor position input cannot be empty.")
            return "", self.anchor_input_history, gr.update(interactive=True)

        parts = line_str.split(',')
        if len(parts) != 4:
            self.show_temporary_message("Invalid format. Please use: i,x,y,z")
            return "", self.anchor_input_history, gr.update(interactive=True)

        try:
            user_i = int(parts[0])
            if user_i < 1 or user_i > self.llm_controller.uwb.anchor_count:
                self.show_temporary_message(f"Anchor index {user_i} out of range (1 to {self.llm_controller.uwb.anchor_count})")
                return "", self.anchor_input_history, gr.update(interactive=True)
            self.llm_controller.uwb.input_anchor_line(line_str)
        except Exception:
            self.show_temporary_message("Invalid format. i must be int, x,y,z must be float")
            return "", self.anchor_input_history, gr.update(interactive=True)

        idx = user_i - 1
        anchor = self.llm_controller.uwb.anchors[idx]
        if anchor is not None:
            x, y, z = anchor
            self.anchor_input_history += f"Anchor {user_i} set to ({x:.2f}, {y:.2f}, {z:.2f})\n"

        if all(a is not None for a in self.llm_controller.uwb.anchors):
            self.show_temporary_message("All anchors set. Starting positioning.")
            # 禁用按鈕，防止再新增或更新
            return "", self.anchor_input_history, gr.update(interactive=False)
        else:
            next_idx = self.llm_controller.uwb.anchors.index(None) + 1
            self.show_temporary_message(f"Please enter anchor {next_idx}'s position.")
            return "", self.anchor_input_history, gr.update(interactive=True)


    def checkbox_llama3(self):
        self.use_llama3 = not self.use_llama3
        if self.use_llama3:
            print_t("Switch to llama3")
            self.llm_controller.planner.set_model(LLAMA3)
        else:
            print_t("Switch to gpt4")
            self.llm_controller.planner.set_model(GPT4)

    def apply_scenario(self, scenario_name):
        normalized, report, runtime = self._apply_mode_and_collect(scenario_name)
        return (
            f"Scenario `{normalized}` applied. "
            f"Live 3s predicted collision probability: {self._fmt_float(runtime.get('predicted_collision_probability'))}"
            f""
        )

    def apply_baseline_scene(self, scene_id):
        normalized = self.llm_controller.set_baseline_scene(scene_id)
        state = self.llm_controller.apply_baseline_scene()
        user_yaw_deg = math.degrees(self.llm_controller.get_user_heading_yaw())
        return f"Baseline scene `{normalized}` applied. drone_init={self._fmt_vec(state.get('drone_initial_pose'))} user={self._fmt_vec(state.get('user_position'))} user_yaw={user_yaw_deg:.1f}deg"

    @staticmethod
    def _drain_queue(q):
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _reset_runtime_records(self):
        self.position_history = {
            "drone_gt": deque(maxlen=100),
            "drone_est": deque(maxlen=100),
        }
        self.worker_collision_history = {
            "worker_1": deque(maxlen=100),
            "worker_2": deque(maxlen=100),
            "worker_3": deque(maxlen=100),
        }
        self.worker_collision_active = {
            "worker_1": False,
            "worker_2": False,
            "worker_3": False,
        }
        self.mission_collision_count = 0
        self.mission_clock = {
            "started_at": None,
            "completed_at": None,
            "is_running": False,
            "objective_completed": False,
        }
        active_ids = set(self.objective_state.get("active_checkpoint_ids", set()))
        self.benchmark_progress = {
            "order": list(BENCHMARK_CHECKPOINT_ORDER),
            "completed": set(),
            "active_enter_ts": None,
            "active_progress": 0.0,
            "current_target": next(
                (cid for cid in BENCHMARK_CHECKPOINT_ORDER if cid in active_ids),
                None,
            ),
            "executed_gc_sequence": [],
        }
        if hasattr(self, "_last_position_map"):
            self._last_position_map.clear()
        if hasattr(self, "_last_print_position_map"):
            self._last_print_position_map.clear()
        self._temp_message = ""
        self._temp_message_expire = 0.0
        self._drain_queue(self.uwb_queue)
        self._drain_queue(self.virtual_queue)
        self._drain_queue(self.message_queue)

    def _reset_persisted_logs(self):
        logger = getattr(self.llm_controller, "task_run_logger", None)
        if logger is None:
            return
        try:
            with logger._lock:
                logger._active = None
                if hasattr(logger, "_pending_completed"):
                    logger._pending_completed = None
        except Exception:
            pass
        for file_path in [
            getattr(logger, "excel_path", None),
            getattr(logger, "debug_jsonl_path", None),
            getattr(logger, "runtime_trace_jsonl_path", None),
            getattr(logger, "planning_trace_jsonl_path", None),
        ]:
            if not file_path:
                continue
            for matched in glob.glob(file_path):
                try:
                    os.remove(matched)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print_t(f"[WARN] Failed to remove log file `{matched}`: {e}")

    def reset_system_state(self):
        try:
            self.llm_controller.current_plan = None
            self.llm_controller.execution_history = None
            self.llm_controller.current_task_description = ""
            self.llm_controller.execution_mode = "Waiting"
            self.llm_controller.framework_mode = MODE_TYPEFLY_ONESHOT
            self.llm_controller._pending_heartbeat_replan_plan = None
            self.llm_controller._pending_heartbeat_reason = ""
            self.llm_controller._reset_benchmark_progress_tracking()
            self.llm_controller.apply_baseline_scene()
            self._reset_runtime_records()
            return "System reset complete: drone/workers repositioned and runtime progress cleared (archive logs retained)."
        except Exception as e:
            return f"System reset failed: {e}"

    def set_framework_mode(self, framework_mode: str):
        normalized = self.llm_controller._normalize_framework_mode(framework_mode)
        self.selected_framework_mode = normalized
        print_t(f"[MODE] selected={normalized}")
        return f"Execution mode switched to `{normalized}`."

    def set_selected_baseline(self, baseline_id: str):
        normalized = self.llm_controller.set_selected_pipeline(baseline_id)
        self.selected_baseline_id = normalized
        cfg = PIPELINE_REGISTRY[normalized]
        return f"Baseline switched to `{cfg.id}` ({cfg.name})."

    def set_archive_enabled(self, enabled: bool):
        self.archive_enabled = bool(enabled)
        self.llm_controller.set_archive_enabled(self.archive_enabled)
        return f"Archive logging {'enabled' if self.archive_enabled else 'disabled'} for next run."

    def save_last_run(self):
        logger = getattr(self.llm_controller, "task_run_logger", None)
        if logger is None:
            return "Save failed: logger unavailable.", self.render_postrun_summary()
        saved = bool(getattr(logger, "save_pending_run", lambda: False)())
        if not saved:
            return "No finished run to save.", self.render_postrun_summary()
        return "Run saved to formal archive.", self.render_postrun_summary()

    def discard_last_run(self):
        logger = getattr(self.llm_controller, "task_run_logger", None)
        if logger is None:
            return "Discard failed: logger unavailable.", self.render_postrun_summary()
        discarded = bool(getattr(logger, "discard_pending_run", lambda: False)())
        if not discarded:
            return "No finished run to discard.", self.render_postrun_summary()
        return "Discarded finished run (not written to formal archive).", self.render_postrun_summary()

    def _apply_mode_and_collect(self, scenario_name):
        normalized = normalize_scenario_name(scenario_name)
        self.llm_controller.set_active_scenario(normalized)
        report = self.llm_controller.apply_selected_scenario()
        runtime = self.llm_controller.get_scenario_runtime_status()
        self.active_scenario = normalized
        return normalized, report, runtime

    def _move_user(self, local_forward: float, local_right: float, step_m: float):
        step = float(step_m)
        yaw = float(self.llm_controller.get_user_heading_yaw())
        dx_world = ((local_forward * math.cos(yaw)) + (local_right * math.sin(yaw))) * step
        dy_world = ((local_forward * math.sin(yaw)) - (local_right * math.cos(yaw))) * step
        updated = self.llm_controller.move_user_world(dx=dx_world, dy=dy_world, dz=0.0)
        if updated is None:
            return "User move failed: no simulation user-position provider."
        runtime = self.llm_controller.get_scenario_runtime_status()
        return (
            f"User moved to {self._fmt_vec(updated)} | "
            f"live predicted_collision_probability={self._fmt_float(runtime.get('predicted_collision_probability'))} "
            f""
        )

    def move_user_forward(self, step_m: float):
        return self._move_user(local_forward=1.0, local_right=0.0, step_m=step_m)

    def move_user_backward(self, step_m: float):
        return self._move_user(local_forward=-1.0, local_right=0.0, step_m=step_m)

    def move_user_left(self, step_m: float):
        return self._move_user(local_forward=0.0, local_right=-1.0, step_m=step_m)

    def move_user_right(self, step_m: float):
        return self._move_user(local_forward=0.0, local_right=1.0, step_m=step_m)

    def turn_user_cw(self, deg_step: float):
        yaw = self.llm_controller.turn_user_heading(-float(deg_step))
        return f"User heading turned CW by {deg_step:.1f}°. new_yaw={math.degrees(yaw):.1f}°"

    def turn_user_ccw(self, deg_step: float):
        yaw = self.llm_controller.turn_user_heading(float(deg_step))
        return f"User heading turned CCW by {deg_step:.1f}°. new_yaw={math.degrees(yaw):.1f}°"

    def select_controlled_worker(self, worker_id: str):
        selected = self.llm_controller.set_manual_worker_selection(worker_id)
        return f"Controlled worker set to {selected}"

    def set_worker_move_step(self, step_m: float):
        self.selected_worker_move_step = float(step_m)

    def set_worker_turn_step(self, deg_step: float):
        self.selected_worker_turn_step = float(deg_step)

    def _move_worker(self, local_forward: float, local_right: float, step_m: float | None = None):
        step = float(self.selected_worker_move_step if step_m is None else step_m)
        state = self.llm_controller.move_selected_worker_relative(local_forward=local_forward, local_right=local_right, step_m=step)
        if state is None:
            return "Manual worker control is only available in SCENE_MANUAL_WORKER_CONTROL."
        return (
            f"{state['worker_id']} moved to ({state['x']:.2f}, {state['y']:.2f}), "
            f"heading={state['yaw_deg']:.1f}°"
        )

    def move_worker_forward(self, step_m: float | None = None):
        return self._move_worker(local_forward=1.0, local_right=0.0, step_m=step_m)

    def move_worker_backward(self, step_m: float | None = None):
        return self._move_worker(local_forward=-1.0, local_right=0.0, step_m=step_m)

    def move_worker_left(self, step_m: float | None = None):
        return self._move_worker(local_forward=0.0, local_right=-1.0, step_m=step_m)

    def move_worker_right(self, step_m: float | None = None):
        return self._move_worker(local_forward=0.0, local_right=1.0, step_m=step_m)

    def turn_worker_cw(self, deg_step: float | None = None):
        turn_step = float(self.selected_worker_turn_step if deg_step is None else deg_step)
        state = self.llm_controller.turn_selected_worker(-turn_step)
        if state is None:
            return "Manual worker control is only available in SCENE_MANUAL_WORKER_CONTROL."
        return f"{state['worker_id']} heading={state['yaw_deg']:.1f}°"

    def turn_worker_ccw(self, deg_step: float | None = None):
        turn_step = float(self.selected_worker_turn_step if deg_step is None else deg_step)
        state = self.llm_controller.turn_selected_worker(turn_step)
        if state is None:
            return "Manual worker control is only available in SCENE_MANUAL_WORKER_CONTROL."
        return f"{state['worker_id']} heading={state['yaw_deg']:.1f}°"

    def process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            return "[WARNING] Empty command!"
        else:
            self.mission_clock["started_at"] = time.time()
            self.mission_clock["completed_at"] = None
            self.mission_clock["is_running"] = True
            self.mission_clock["objective_completed"] = False
            self.benchmark_progress["completed"] = set()
            self.benchmark_progress["active_enter_ts"] = None
            self.benchmark_progress["active_progress"] = 0.0
            active_ids = set(self.objective_state.get("active_checkpoint_ids", set()))
            self.benchmark_progress["current_target"] = next(
                (cid for cid in self.benchmark_progress["order"] if cid in active_ids),
                None,
            )
            self.worker_collision_active = {k: False for k in self.worker_collision_active.keys()}
            self.mission_collision_count = 0
            framework_mode = str(getattr(self, "selected_framework_mode", MODE_TYPEFLY_ONESHOT))
            self.llm_controller.set_selected_pipeline(self.selected_baseline_id)
            task_thread = Thread(
                target=self.llm_controller.execute_task_description,
                args=(message, framework_mode),
            )
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple):
                    history.append((None, msg))
                elif isinstance(msg, str):
                    if msg == 'end':
                        if self.mission_clock["is_running"]:
                            self.mission_clock["completed_at"] = time.time()
                            self.mission_clock["is_running"] = False
                        return "Command Complete! Please click 'Save this run' or 'Discard this run'."
                    if msg.startswith('[LOG]'):
                        complete_response += '\n'
                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield complete_response

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        try:
            self.llm_controller.apply_baseline_scene()
        except Exception:
            pass

        if self.llm_controller.enable_video:
            llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
            llmc_thread.start()
        else:
            llmc_thread = None

        if self.llm_controller.enable_video:
            app = Flask(__name__)

            @app.route('/drone-pov/')
            def video_feed():
                return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

            @app.route('/shutdown', methods=['POST'])
            def shutdown():
                func = request.environ.get('werkzeug.server.shutdown')
                if func:
                    func()
                    return 'Server shutting down...'
                else:
                    return 'Unable to shut down server', 500

            PORT = int(os.environ.get("FLASK_PORT", 50000))
            flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': PORT, 'debug': False, 'use_reloader': False})
            flask_thread.start()
        else:
            flask_thread = None

        self.chat.queue()
        self.ui.launch(server_port=50001, prevent_thread_lock=True, css=self.ui_css)

        while not self.system_stop:
            time.sleep(1)

        print_t("[C] Shutting down system...")
        self.llm_controller.stop_robot()

        if self.llm_controller.enable_video and flask_thread:
            try:
                import requests
                requests.post("http://localhost:50000/shutdown")
            except Exception as e:
                print_t(f"[WARN] Failed to shutdown Flask server: {e}")

        if llmc_thread:
            llmc_thread.join()
        asyncio_thread.join()

        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))

    def create_blank_plot(self, title="Empty Plot", xlabel="X", ylabel="Y", xlim=(0, 1), ylim=(0, 1), figsize=(5, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([round(xlim[0] + i * 0.5, 2) for i in range(int((xlim[1]-xlim[0])/0.5)+1)])
        ax.set_yticks([round(ylim[0] + i * 0.5, 2) for i in range(int((ylim[1]-ylim[0])/0.5)+1)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def create_sequence_plot(self, title, xlabel, ylabel, xlim=(0, 1), ylim=(0, 5)):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([i * 0.2 for i in range(6)])
        ax.set_yticks([i * 0.5 for i in range(11)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def update_and_step(self, counter, show_error_ellipse=False, show_raw_estimate=False):
        snapshot = self.llm_controller.get_live_ui_snapshot()
        safety_context = snapshot.get("safety_context") if snapshot else None
        ui_pc = None if safety_context is None else float(getattr(safety_context, "predicted_collision_probability", 0.0))
        if hasattr(self.llm_controller, "update_ui_collision_probability"):
            self.llm_controller.update_ui_collision_probability(ui_pc)
        self._sync_objective_state(snapshot)
        self._append_history(snapshot)
        self._append_worker_collision_history(snapshot)
        self._update_mission_collision_count(snapshot)
        self._update_checkpoint_progress(snapshot)
        anchor_plot = self.render_anchor_3d_plot()
        global_xy, xy, x, y, z = self.update_position_plot(snapshot, show_error_ellipse=show_error_ellipse, show_raw_estimate=show_raw_estimate)
        status_md = self.render_status_markdown(snapshot)
        entity_md = self.render_entity_markdown(snapshot)
        postrun_md = self.render_postrun_summary()
        counter += 1
        print_debug(
            "[UI-CALLBACK] "
            "outputs=[anchor_3d,global_xy_plot,xy_plot,x_plot,y_plot,z_plot,counter,status,entity,postrun] "
            f"drone_gt={None if not snapshot else snapshot.get('drone_gt')} "
            f"drone_est={None if not snapshot else snapshot.get('drone_est')} "
            f"counter={counter}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )
        return anchor_plot, global_xy, xy, x, y, z, counter, status_md, entity_md, postrun_md

    def render_postrun_summary(self):
        logger = getattr(self.llm_controller, "task_run_logger", None)
        if logger is None:
            return "### Post-run archive\nLogger unavailable."
        summary = getattr(logger, "get_pending_run_summary", lambda: {})() or {}
        if not summary:
            return "### Post-run archive\nNo finished run awaiting decision."
        lines = [
            "### Post-run archive (pending decision)",
            f"- run_id: {summary.get('run_id', 'n/a')}",
            f"- baseline: {summary.get('selected_baseline_id', 'n/a')} ({summary.get('selected_baseline_name', 'n/a')})",
            f"- scene_id: {summary.get('scene_id', 'n/a')}",
            f"- status: {summary.get('run_status', 'n/a')}",
            f"- mission_success: {summary.get('mission_success', 'n/a')}",
            f"- termination_reason: {summary.get('termination_reason', 'n/a')}",
            f"- completion_time_mission_sec: {summary.get('completion_time_mission_sec', 'n/a')}",
            f"- replan_count: {summary.get('replan_count', 0)}",
            f"- collision_count: {summary.get('collision_count', 0)}",
            f"- near_miss_count: {summary.get('near_miss_count', 0)}",
            f"- completion_ratio: {float(summary.get('completion_ratio', 0.0)):.2f}",
            f"- runtime_trace_count: {summary.get('runtime_trace_count', 0)}",
            f"- planning_trace_count: {summary.get('planning_trace_count', 0)}",
            "- action required: click **Save this run** or **Discard this run**.",
        ]
        return "\\n".join(lines)

    def _fmt_vec(self, value):
        if value is None:
            return "(n/a)"
        return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"

    def _fmt_float(self, value, suffix=""):
        if value is None:
            return "n/a"
        return f"{value:.3f}{suffix}"

    def _fmt_prob(self, value):
        if value is None:
            return "n/a"
        value = float(value)
        if abs(value) < 1e-4 and value != 0.0:
            return f"{value:.3e}"
        return f"{value:.6f}"

    def _extract_ui_positions(self, snapshot):
        if not snapshot:
            return {
                "drone_gt": None,
                "drone_est": None,
            }
        positions = {
            "drone_gt": snapshot.get("drone_gt"),
            # default visualization uses bias-corrected estimate.
            "drone_est": snapshot.get("drone_est_bias_corrected") or snapshot.get("drone_est"),
        }
        return positions

    def _sync_objective_state(self, snapshot):
        if not snapshot:
            return
        objective = snapshot.get("active_objective_set")
        if not isinstance(objective, dict):
            return
        zone_ids = objective.get("active_zone_ids")
        cp_ids = objective.get("active_checkpoint_ids")
        if zone_ids:
            self.objective_state["active_zone_ids"] = set(str(v) for v in zone_ids)
        if cp_ids:
            self.objective_state["active_checkpoint_ids"] = set(str(v) for v in cp_ids)

    def _update_checkpoint_progress(self, snapshot):
        if not isinstance(snapshot, dict):
            return
        progress = snapshot.get("benchmark_progress")
        if not isinstance(progress, dict):
            return
        completed = set(str(v).upper() for v in (progress.get("completed") or []))
        dwell_seconds = float(progress.get("dwell_seconds", 0.0) or 0.0)
        required_dwell_seconds = float(progress.get("required_dwell_seconds", CHECKPOINT_DWELL_SECONDS) or CHECKPOINT_DWELL_SECONDS)
        if required_dwell_seconds <= 1e-6:
            active_progress = 0.0
        else:
            active_progress = min(1.0, max(0.0, dwell_seconds / required_dwell_seconds))
        self.benchmark_progress["completed"] = completed
        self.benchmark_progress["current_target"] = progress.get("current_target")
        self.benchmark_progress["active_enter_ts"] = progress.get("active_enter_ts")
        self.benchmark_progress["active_progress"] = active_progress
        self.benchmark_progress["executed_gc_sequence"] = [
            str(v).upper() for v in (progress.get("executed_gc_sequence") or [])
        ]
        active_ids = set(self.objective_state.get("active_checkpoint_ids", set()))
        now = time.time()
        mission_completed = bool(active_ids) and all(cid in completed for cid in active_ids)
        self.mission_clock["objective_completed"] = mission_completed
        if mission_completed and self.mission_clock.get("started_at") is not None and self.mission_clock.get("completed_at") is None:
            self.mission_clock["completed_at"] = now
            self.mission_clock["is_running"] = False

    def render_anchor_3d_plot(self):
        fig = plt.figure(figsize=(5.2, 4.2))
        ax = fig.add_subplot(111, projection='3d')
        anchors = self.anchor_provider.get_anchor_positions()
        ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], c="#1A73E8", s=42, depthshade=False)
        for idx, (x, y, z) in enumerate(anchors, start=1):
            ax.text(float(x) + 0.1, float(y) + 0.1, float(z) + 0.05, f"A{idx}", fontsize=8)
        square = np.array([[0, 0], [12, 0], [12, 12], [0, 12], [0, 0]], dtype=float)
        ax.plot(square[:, 0], square[:, 1], zs=0.0, color="#7A7A7A", linewidth=1.2, linestyle="--")
        for z in (2.5, 5.5):
            ax.plot(square[:, 0], square[:, 1], zs=z, color="#9AA0A6", linewidth=1.0, linestyle=":")
            ax.text(12.2, 12.2, z, f"z={z:.1f}m", fontsize=8, color="#5F6368")
        ax.set_xlim(0, WORKSPACE_SIZE_M)
        ax.set_ylim(0, WORKSPACE_SIZE_M)
        ax.set_zlim(0, 7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Anchor Layout (3D)")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def _append_history(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        if not snapshot:
            return
        for key, value in positions.items():
            if value is not None:
                self.position_history[key].append(tuple(float(v) for v in value))
                print_debug(
                    f"[UI-HISTORY] key={key} appended={self.position_history[key][-1]}",
                    env_var="TYPEFLY_VERBOSE_DEBUG",
                )

    def _append_worker_collision_history(self, snapshot):
        safety_context = snapshot.get("safety_context") if snapshot else None
        per_worker = {}
        if safety_context is not None:
            per_worker = {
                str(row.get("id")): float(row.get("predicted_collision_probability", 0.0))
                for row in (getattr(safety_context, "per_worker_collision_probabilities", []) or [])
            }
        for worker_id in ("worker_1", "worker_2", "worker_3"):
            self.worker_collision_history[worker_id].append(float(per_worker.get(worker_id, 0.0)))

    def _update_mission_collision_count(self, snapshot):
        if not snapshot:
            return
        drone_gt = snapshot.get("drone_gt")
        workers = snapshot.get("workers") or []
        if drone_gt is None:
            return
        collision_radius = float(UAV_RADIUS_M + WORKER_RADIUS_M)
        worker_map = {str(item.get("id")): item for item in workers}
        for worker_id in ("worker_1", "worker_2", "worker_3"):
            worker = worker_map.get(worker_id)
            worker_gt = None if worker is None else worker.get("gt_xy")
            currently_colliding = False
            if worker_gt is not None:
                distance_xy = math.hypot(float(drone_gt[0]) - float(worker_gt[0]), float(drone_gt[1]) - float(worker_gt[1]))
                currently_colliding = bool(distance_xy <= collision_radius)
            if currently_colliding and not self.worker_collision_active.get(worker_id, False):
                self.mission_collision_count += 1
            self.worker_collision_active[worker_id] = currently_colliding

    def render_status_markdown(self, snapshot):
        safety_context = snapshot.get("safety_context") if snapshot else None
        if safety_context is None:
            return "### Status\nWaiting for safety state..."
        final_summary = dict((snapshot or {}).get("final_mission_summary") or {})
        active_ids = set(self.objective_state.get("active_checkpoint_ids", set()))
        completed_set = set(self.benchmark_progress["completed"])
        completed_active = len(completed_set.intersection(active_ids))
        total = len(active_ids)
        target = self.benchmark_progress.get("current_target") or "n/a"
        zone_map = {"zone_A": ["A1", "A2", "A3", "A4"], "zone_B": ["B1", "B2", "B3", "B4"], "zone_C": ["C1", "C2", "C3", "C4", "C5", "C6"]}
        zone_parts = []
        for zid, ids in zone_map.items():
            active_zone_ids = [cid for cid in ids if cid in active_ids]
            if not active_zone_ids:
                continue
            done_zone = len([cid for cid in active_zone_ids if cid in completed_set])
            zone_parts.append(f"{zid[-1]}: {done_zone}/{len(active_zone_ids)}")

        now_ts = time.time()
        started_at = self.mission_clock.get("started_at")
        completed_at = self.mission_clock.get("completed_at")
        elapsed_text = "n/a"
        if started_at is not None:
            end_for_elapsed = now_ts if self.mission_clock.get("is_running") else (completed_at or now_ts)
            elapsed_text = f"{max(0.0, end_for_elapsed - float(started_at)):.2f} s"
        completion_text = "n/a" if completed_at is None or started_at is None else f"{max(0.0, float(completed_at - started_at)):.2f} s"
        if final_summary:
            ctm = final_summary.get("completion_time_mission_sec")
            completion_text = "n/a" if ctm is None else f"{float(ctm):.2f} s"
        collision_count = int(final_summary.get("collision_count", snapshot.get("collision_count", 0)) or 0)
        near_miss_count = int(final_summary.get("near_miss_count", snapshot.get("near_miss_count", 0)) or 0)
        run_status = str(final_summary.get("run_status", "running"))
        mission_success = bool(final_summary.get("mission_success", self.mission_clock.get("objective_completed", False)))
        termination_reason = str(final_summary.get("termination_reason", ""))
        final_execution_mode = str(final_summary.get("final_execution_mode", snapshot.get("execution_mode", "Waiting")))

        lines = [
            "### Status",
            f"- current framework: {snapshot.get('framework_name', 'n/a')}",
            f"- current mode: {final_execution_mode}",
            f"- selected baseline: {snapshot.get('selected_baseline_id', 'n/a')} ({snapshot.get('selected_baseline_name', 'n/a')})",
            f"- current scene: {snapshot.get('baseline_scene_id', 'n/a')}",
            f"- archive policy: post-run Save/Discard decision",
            f"- active zones: {', '.join(sorted(z.replace('zone_', '') for z in self.objective_state.get('active_zone_ids', set())))}",
            f"- active checkpoints: {len(active_ids)}",
            f"- predicted_collision_probability: {self._fmt_prob(getattr(safety_context, 'predicted_collision_probability', 0.0))}",
            f"- dominant risky worker: {getattr(safety_context, 'dominant_threat_id', 'n/a')}",
            f"- current target checkpoint: {target}",
            f"- checkpoint progress: {completed_active}/{total}",
            f"- zone progress: {', '.join(zone_parts) if zone_parts else 'n/a'}",
            f"- mission collision count: {collision_count}",
            f"- near-miss count: {near_miss_count}",
            f"- replan count: {int(snapshot.get('replan_count', 0) or 0)}",
            f"- run status: {run_status}",
            f"- mission completed: {mission_success}",
            f"- termination reason: {termination_reason if termination_reason else 'n/a'}",
            f"- mission elapsed time: {elapsed_text}",
            f"- mission completion time: {completion_text}",
        ]

        return "\n".join(lines)

    def render_entity_markdown(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        workers = snapshot.get("workers") or []
        worker_map = {str(item.get("id")): item for item in workers}

        def _fmt_xy(pos):
            if pos is None:
                return "(n/a)"
            return f"({float(pos[0]):.2f}, {float(pos[1]):.2f})"

        lines = [
            "### Entity positions",
            f"- UAV true: {_fmt_xy(positions.get('drone_gt'))}",
            f"- UAV est: {_fmt_xy(positions.get('drone_est'))}",
        ]
        for worker_id in ("worker_1", "worker_2", "worker_3"):
            worker = worker_map.get(worker_id)
            lines.append(f"- {worker_id} true: {_fmt_xy(None if worker is None else worker.get('gt_xy'))}")
            lines.append(f"- {worker_id} est: {_fmt_xy(None if worker is None else worker.get('est_xy_bias_corrected'))}")
        return "\n".join(lines)

    def _estimate_heading_from_history(self, primary_key: str, fallback_key: str = None):
        history = list(self.position_history.get(primary_key, []))
        if len(history) < 2 and fallback_key:
            history = list(self.position_history.get(fallback_key, []))
        if len(history) >= 2:
            p0 = history[-2]
            p1 = history[-1]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return float(math.atan2(dy, dx)), "trajectory_history"
        return 0.0, "fallback_zero"

    def _axis_limits_from_snapshot(self, snapshot):
        positions = self._extract_ui_positions(snapshot)
        xs, ys = [], []
        for key in ("drone_gt", "drone_est"):
            value = positions.get(key)
            if value is not None:
                xs.append(float(value[0]))
                ys.append(float(value[1]))
            history = self.position_history.get(key)
            if history:
                xs.extend(float(point[0]) for point in history)
                ys.extend(float(point[1]) for point in history)
        safety_state = snapshot.get("safety_state") if snapshot else None
        if safety_state is not None:
            envelope = safety_state.drone_envelope
            xs.extend([
                float(envelope.center_xy[0] - envelope.major_axis_radius),
                float(envelope.center_xy[0] + envelope.major_axis_radius),
            ])
            ys.extend([
                float(envelope.center_xy[1] - envelope.major_axis_radius),
                float(envelope.center_xy[1] + envelope.major_axis_radius),
            ])
        if not xs or not ys:
            return (0.0, 5.0), (0.0, 5.0)
        pad = 0.5
        return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


    def _render_xy_view(self, snapshot, xlim, ylim, title, figsize=(5, 4), show_legend=True, show_error_ellipse=False, show_raw_estimate=False):
        positions = self._extract_ui_positions(snapshot)
        fig_xy, ax_xy = plt.subplots(figsize=figsize)
        ax_xy.add_patch(plt.Rectangle((0, 0), WORKSPACE_SIZE_M, WORKSPACE_SIZE_M, fill=False, edgecolor="#263238", linewidth=1.8))
        ax_xy.plot([6, 6], [6, 12], color="#5F6368", linewidth=1.2)
        ax_xy.plot([0, 12], [6, 6], color="#5F6368", linewidth=1.2)
        ax_xy.text(2.2, 10.8, "zone_A", fontsize=9, color="#37474F")
        ax_xy.text(8.2, 10.8, "zone_B", fontsize=9, color="#37474F")
        ax_xy.text(5.2, 5.2, "zone_C", fontsize=9, color="#37474F")

        current_target = self.benchmark_progress.get("current_target")
        active_progress = float(self.benchmark_progress.get("active_progress", 0.0))
        for cid in BENCHMARK_CHECKPOINT_ORDER:
            cp = BENCHMARK_CHECKPOINTS_BY_ID[cid]
            if cid in self.benchmark_progress["completed"]:
                color = "#2E7D32"
            elif cid == current_target and active_progress > 0:
                color = "#FB8C00"
            else:
                color = "#9E9E9E"
            ax_xy.add_patch(Circle((cp.x, cp.y), CHECKPOINT_RADIUS_M, fill=False, edgecolor=color, linewidth=1.5))
            if cid == current_target and active_progress > 0:
                ax_xy.add_patch(Arc((cp.x, cp.y), width=2.0 * (CHECKPOINT_RADIUS_M + 0.08), height=2.0 * (CHECKPOINT_RADIUS_M + 0.08), theta1=90, theta2=90 - (360.0 * active_progress), edgecolor="#FF9800", linewidth=2.0))
            ax_xy.scatter([cp.x], [cp.y], c=color, s=12)
            ax_xy.text(cp.x + 0.08, cp.y + 0.08, cid, fontsize=8, color="#37474F")

        drone_gt = positions.get("drone_gt")
        drone_est = positions.get("drone_est")
        gt_history = list(self.position_history.get("drone_gt", []))
        est_history = list(self.position_history.get("drone_est", []))
        if len(gt_history) >= 2:
            ax_xy.plot(
                [p[0] for p in gt_history],
                [p[1] for p in gt_history],
                color="#0B57D0",
                linewidth=1.3,
                alpha=0.45,
                label="UAV trajectory",
            )
        if show_raw_estimate and len(est_history) >= 2:
            ax_xy.plot(
                [p[0] for p in est_history],
                [p[1] for p in est_history],
                color="#8AB4F8",
                linewidth=1.0,
                alpha=0.35,
                linestyle="--",
                label="UAV est trajectory",
            )
        if drone_gt is not None:
            ax_xy.add_patch(Circle((drone_gt[0], drone_gt[1]), UAV_RADIUS_M, fill=False, edgecolor="#0B57D0", linewidth=2.0, label="UAV true"))
        if drone_est is not None:
            ax_xy.add_patch(Circle((drone_est[0], drone_est[1]), UAV_RADIUS_M, fill=False, edgecolor="#8AB4F8", linewidth=1.6, linestyle="--", label="UAV bias-corrected"))
        if drone_gt is not None and drone_est is not None:
            ax_xy.plot([drone_gt[0], drone_est[0]], [drone_gt[1], drone_est[1]], color="#0B57D0", linewidth=0.8, alpha=0.8)

        workers = snapshot.get("workers") or []
        for worker in workers:
            gt_xy = worker.get("gt_xy")
            est_xy = worker.get("est_xy_bias_corrected")
            ui_xy = worker.get("ui_xy") or est_xy or gt_xy
            wid = worker.get("id")
            if gt_xy is not None:
                ax_xy.add_patch(Circle((gt_xy[0], gt_xy[1]), WORKER_RADIUS_M, fill=False, edgecolor="#7B1FA2", linewidth=1.8))
            if ui_xy is not None:
                ax_xy.add_patch(Circle((ui_xy[0], ui_xy[1]), WORKER_RADIUS_M, fill=False, edgecolor="#CE93D8", linewidth=1.3, linestyle="--"))
                ax_xy.text(ui_xy[0] + 0.08, ui_xy[1] + 0.08, str(wid), fontsize=8, color="#4A148C")
                heading = float(worker.get("heading_yaw_rad", 0.0))
                arrow_len = 0.45
                wx, wy = float(ui_xy[0]), float(ui_xy[1])
                wdx = arrow_len * float(math.cos(heading))
                wdy = arrow_len * float(math.sin(heading))
                ax_xy.arrow(wx, wy, wdx, wdy, head_width=0.12, head_length=0.14, color="#6A1B9A", linewidth=1.2, length_includes_head=True, zorder=4)
            if gt_xy is not None and ui_xy is not None:
                ax_xy.plot([gt_xy[0], ui_xy[0]], [gt_xy[1], ui_xy[1]], color="#8E24AA", linewidth=0.7, alpha=0.8)
            if show_raw_estimate and worker.get("est_xy_raw") is not None:
                raw = worker["est_xy_raw"]
                ax_xy.scatter([raw[0]], [raw[1]], marker="x", c="#6A1B9A", s=22)

        if show_raw_estimate and snapshot.get("drone_est_raw") is not None:
            raw = snapshot["drone_est_raw"]
            ax_xy.scatter([raw[0]], [raw[1]], marker="x", c="#1E88E5", s=36, label="UAV raw est")

        if show_error_ellipse and snapshot.get("drone_P_xy") is not None and drone_est is not None:
            p = np.asarray(snapshot["drone_P_xy"], dtype=float)
            eigvals, eigvecs = np.linalg.eigh(p)
            eigvals = np.maximum(eigvals, 1e-8)
            angle = math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1]))
            ax_xy.add_patch(Ellipse((drone_est[0], drone_est[1]), width=2 * math.sqrt(eigvals[1]), height=2 * math.sqrt(eigvals[0]), angle=angle, edgecolor="#42A5F5", facecolor="none", linestyle=":", linewidth=1.4, label="UAV variance ellipse"))
        if show_error_ellipse:
            for worker in workers:
                p = worker.get("P_xy")
                est_xy = worker.get("est_xy_bias_corrected")
                if p is None or est_xy is None:
                    continue
                p = np.asarray(p, dtype=float)
                eigvals, eigvecs = np.linalg.eigh(p)
                eigvals = np.maximum(eigvals, 1e-8)
                angle = math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1]))
                ax_xy.add_patch(Ellipse((est_xy[0], est_xy[1]), width=2 * math.sqrt(eigvals[1]), height=2 * math.sqrt(eigvals[0]), angle=angle, edgecolor="#B39DDB", facecolor="none", linestyle=":", linewidth=1.1))

        original_path = snapshot.get("original_planned_path") or []
        if len(original_path) >= 2:
            ax_xy.plot([p[0] for p in original_path], [p[1] for p in original_path], color="#9E9E9E", linestyle="--", linewidth=1.4, label="Original planned path")
        updated_path = snapshot.get("updated_path") or []
        if len(updated_path) >= 2:
            ax_xy.plot([p[0] for p in updated_path], [p[1] for p in updated_path], color="#1565C0", linestyle="-", linewidth=1.7, label="Current path")

        drone_for_heading = positions.get("drone_gt") or positions.get("drone_est")
        yaw_rad = float(snapshot.get("drone_yaw_rad") or 0.0) if snapshot else 0.0
        if drone_for_heading is not None:
            hx = float(drone_for_heading[0])
            hy = float(drone_for_heading[1])
            arrow_len = 0.55
            dx = arrow_len * float(math.cos(yaw_rad))
            dy = arrow_len * float(math.sin(yaw_rad))
            ax_xy.arrow(hx, hy, dx, dy, head_width=0.16, head_length=0.18, color="#0B57D0", linewidth=1.6, length_includes_head=True, zorder=5)
            ax_xy.text(hx + dx + 0.05, hy + dy + 0.05, "Heading", fontsize=8, color="#0B57D0")

        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.set_title(title)
        ax_xy.grid(True, linestyle='--', linewidth=0.5)
        if show_legend:
            handles, labels = ax_xy.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            ax_xy.legend(dedup.values(), dedup.keys(), fontsize=8)

        buf_xy = io.BytesIO()
        fig_xy.savefig(buf_xy, format='png')
        buf_xy.seek(0)
        plt.close(fig_xy)
        return Image.open(buf_xy)

    def update_position_plot(self, snapshot, show_error_ellipse=False, show_raw_estimate=False):
        positions = self._extract_ui_positions(snapshot)
        dynamic_xlim, dynamic_ylim = self._axis_limits_from_snapshot(snapshot)
        print_debug(
            "[UI-PLOT] "
            f"drone_gt={positions['drone_gt']} "
            f"drone_est={positions['drone_est']}",
            env_var="TYPEFLY_VERBOSE_DEBUG",
        )

        global_xy = self._render_xy_view(
            snapshot=snapshot,
            xlim=(0.0, 12.0),
            ylim=(0.0, 12.0),
            title="Global XY Map (Fixed 0-12m Workspace)",
            figsize=(10, 8),
            show_legend=True,
            show_error_ellipse=show_error_ellipse,
            show_raw_estimate=show_raw_estimate,
        )
        local_xy = self._render_xy_view(
            snapshot=snapshot,
            xlim=dynamic_xlim,
            ylim=dynamic_ylim,
            title="Drone Localization & Safety Envelope (XY view)",
            figsize=(5.8, 4.4),
            show_legend=False,
            show_error_ellipse=show_error_ellipse,
            show_raw_estimate=show_raw_estimate,
        )

        imgs = []
        worker_specs = [
            ("worker_1", "#7B1FA2"),
            ("worker_2", "#00897B"),
            ("worker_3", "#EF6C00"),
        ]
        for worker_id, color in worker_specs:
            fig, ax = plt.subplots(figsize=(5, 4))
            history = list(self.worker_collision_history[worker_id])
            if history:
                ax.plot(
                    list(range(len(history))),
                    history,
                    color=color,
                    linestyle="-",
                    marker='o',
                    markersize=3,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=f"{worker_id} P(predicted collision)",
                )
            else:
                ax.plot([], [], color=color, label=f"{worker_id} P(predicted collision)")
            max_len = max(len(history), 1)
            ax.set_xlim(0, max(max_len - 1, 1))
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{worker_id} 3s Predicted Collision Probability")
            ax.set_xlabel("Sample")
            ax.set_ylabel("P(predicted collision)")
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            imgs.append(Image.open(buf))

        return global_xy, local_xy, imgs[0], imgs[1], imgs[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--px4_sim', action='store_true')
    parser.add_argument('--scenario', type=str, default=os.getenv("TYPEFLY_SCENARIO", "SAFE"))

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    backend = "uwb"
    if args.px4_sim:
        robot_type = RobotType.PX4_SIM
        backend = "sim"
    elif args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR

    typefly = TypeFly(
        robot_type,
        use_http=args.use_http,
        enable_video=args.image,
        backend=backend,
        initial_scenario=normalize_scenario_name(args.scenario),
    )
    typefly.run()
