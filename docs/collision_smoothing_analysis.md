# Collision Probability Smoothing Analysis Notes

This note records where smoothing is applied in the collision-probability pipeline:

- Main smoothing in `controller/collision_probability_core.py` via `_smooth_position` (EMA-like update) for UAV/worker bias-corrected XY means.
- Optional upstream smoothing in `controller/llm_controller.py` (`_simulate_obstacle_returns`) for manual worker control scene only.
- No smoothing over covariance matrices or final collision probability output.
