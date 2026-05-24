from pathlib import Path


def test_predicted_path_uses_bias_corrected_start_positions():
    src = Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert "_bias_corrected_xy" in src
    assert "uav_corrected_xy = self._bias_corrected_xy(" in src
    assert "worker_start_xy_map[str(worker_key)] = self._bias_corrected_xy(" in src
    assert "uav_start_xy=uav_corrected_xy" in src


def test_velocity_history_is_pushed_with_corrected_samples():
    src = Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert "self._last_uav_samples.append((float(now), np.asarray(uav_corrected_xy, dtype=float).reshape(2)))" in src
    assert "corrected_xy = self._bias_corrected_xy(" in src
    assert "samples.append((float(now), corrected_xy))" in src


def test_uav_prediction_has_gc_body_and_fallback_branches():
    src = Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert 'if mode == "gc_target":' in src
    assert 'if mode == "body_action":' in src
    assert 'return start + fallback * float(tau), "velocity_fallback"' in src


def test_covariance_diffusion_depends_on_tau():
    src = Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert "process_var" in src
    assert "sigma_rel = sigma_rel_now + (process_var * np.eye(2, dtype=float))" in src


def test_max_tau_dominant_and_per_worker_outputs_remain_present():
    src = Path("controller/gcs_safety_assessment.py").read_text(encoding="utf-8")
    assert "max_tau" in src
    assert "dominant_worker" in src
    assert "per_worker_max" in src
    assert '"max_risk_tau_seconds": float(max_tau)' in src
    assert '"dominant_predicted_worker": str(dominant_predicted_worker)' in src


def test_llm_controller_passes_intent_and_corrected_drone_xy():
    src = Path("controller/llm_controller.py").read_text(encoding="utf-8")
    assert "uav_prediction_intent=self._build_uav_prediction_intent(" in src
    assert "- np.asarray(safety_state.drone_packet.b_xy[:2], dtype=float)" in src
    assert "- np.asarray(drone_packet.b_xy[:2], dtype=float)" in src
