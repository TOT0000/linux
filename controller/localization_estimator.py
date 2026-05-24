from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LocalizationEstimate:
    est_position_3d: np.ndarray
    jacobian_h_3d: np.ndarray
    estimated_ranges: np.ndarray
    range_residuals: np.ndarray
    range_residual_rms_m: float
    normalized_range_residual_rms: float
    P_3d: np.ndarray
    b_3d: np.ndarray
    M_3d: np.ndarray
    P_xy: np.ndarray
    b_xy: np.ndarray
    M_xy: np.ndarray
    iterations: int
    converged: bool


class IterativeLeastSquaresEstimator3D:
    def __init__(self, max_iterations: int = 15, tolerance: float = 1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    @staticmethod
    def _compute_predicted_ranges(position: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        deltas = position[None, :] - anchors
        distances = np.linalg.norm(deltas, axis=1)
        return np.maximum(distances, 1e-6)

    @staticmethod
    def _compute_jacobian(position: np.ndarray, anchors: np.ndarray, predicted_ranges: np.ndarray) -> np.ndarray:
        deltas = position[None, :] - anchors
        safe_ranges = np.maximum(predicted_ranges[:, None], 1e-6)
        return deltas / safe_ranges

    def estimate(
        self,
        anchors: np.ndarray,
        measured_ranges: np.ndarray,
        sigma_values: np.ndarray,
        bias_values: np.ndarray,
        initial_guess: np.ndarray,
        true_ranges: Optional[np.ndarray] = None,
    ) -> LocalizationEstimate:
        anchors = np.asarray(anchors, dtype=float)
        measured_ranges = np.asarray(measured_ranges, dtype=float)
        sigma_values = np.asarray(sigma_values, dtype=float)
        bias_values = np.asarray(bias_values, dtype=float)
        p_hat = np.asarray(initial_guess, dtype=float).reshape(3).copy()

        converged = False
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            predicted_ranges = self._compute_predicted_ranges(p_hat, anchors)
            H = self._compute_jacobian(p_hat, anchors, predicted_ranges)
            residual = measured_ranges - predicted_ranges
            try:
                delta_p, *_ = np.linalg.lstsq(H, residual, rcond=None)
            except np.linalg.LinAlgError:
                delta_p = np.zeros(3, dtype=float)
            p_hat = p_hat + delta_p
            if float(np.linalg.norm(delta_p)) < self.tolerance:
                converged = True
                break

        predicted_ranges = self._compute_predicted_ranges(p_hat, anchors)
        H = self._compute_jacobian(p_hat, anchors, predicted_ranges)
        residual = measured_ranges - predicted_ranges
        residual_rms_m = float(np.sqrt(np.mean(np.square(residual)))) if residual.size > 0 else 0.0
        normalized_residual = residual / np.maximum(sigma_values, 1e-9)
        normalized_residual_rms = float(np.sqrt(np.mean(np.square(normalized_residual)))) if normalized_residual.size > 0 else 0.0

        variances = np.maximum(np.square(sigma_values), 1e-9)
        R_inv = np.diag(1.0 / variances)
        try:
            information_matrix = H.T @ R_inv @ H
            P_3d = np.linalg.inv(information_matrix)
        except np.linalg.LinAlgError:
            information_matrix = H.T @ R_inv @ H
            P_3d = np.linalg.pinv(information_matrix)

        if true_ranges is None:
            true_ranges = predicted_ranges
        true_ranges = np.asarray(true_ranges, dtype=float)

        try:
            b_3d = np.linalg.inv(H.T @ H) @ H.T @ bias_values
        except np.linalg.LinAlgError:
            b_3d = np.linalg.pinv(H.T @ H) @ H.T @ bias_values

        M_3d = P_3d + np.outer(b_3d, b_3d)
        P_xy = P_3d[0:2, 0:2].copy()
        b_xy = b_3d[0:2].copy()
        M_xy = P_xy + np.outer(b_xy, b_xy)

        return LocalizationEstimate(
            est_position_3d=p_hat,
            jacobian_h_3d=H,
            estimated_ranges=predicted_ranges,
            range_residuals=residual,
            range_residual_rms_m=residual_rms_m,
            normalized_range_residual_rms=normalized_residual_rms,
            P_3d=P_3d,
            b_3d=b_3d,
            M_3d=M_3d,
            P_xy=P_xy,
            b_xy=b_xy,
            M_xy=M_xy,
            iterations=iterations,
            converged=converged,
        )
