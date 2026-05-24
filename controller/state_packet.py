from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LocalizedStatePacket:
    entity_type: str
    sequence_number: int
    state_generation_timestamp: float
    gt_position_3d: np.ndarray
    estimated_position_3d: np.ndarray
    localization_error_vector_3d: np.ndarray
    range_residuals: np.ndarray
    range_residual_rms_m: float
    normalized_range_residual_rms: float
    gt_user_position_3d: np.ndarray
    est_user_position_3d: Optional[np.ndarray]
    anchor_positions_3d: np.ndarray
    true_ranges: np.ndarray
    measured_ranges: np.ndarray
    bias_values: np.ndarray
    sigma_values: np.ndarray
    random_noise_values: np.ndarray
    jacobian_h_3d: np.ndarray
    P_3d: np.ndarray
    b_3d: np.ndarray
    M_3d: np.ndarray
    P_xy: np.ndarray
    b_xy: np.ndarray
    M_xy: np.ndarray
    confidence_alpha: float
    est_position_timestamp: float

    def copy(self) -> "LocalizedStatePacket":
        return LocalizedStatePacket(
            entity_type=str(self.entity_type),
            sequence_number=int(self.sequence_number),
            state_generation_timestamp=float(self.state_generation_timestamp),
            gt_position_3d=self.gt_position_3d.copy(),
            estimated_position_3d=self.estimated_position_3d.copy(),
            localization_error_vector_3d=self.localization_error_vector_3d.copy(),
            range_residuals=self.range_residuals.copy(),
            range_residual_rms_m=float(self.range_residual_rms_m),
            normalized_range_residual_rms=float(self.normalized_range_residual_rms),
            gt_user_position_3d=self.gt_user_position_3d.copy(),
            est_user_position_3d=None if self.est_user_position_3d is None else self.est_user_position_3d.copy(),
            anchor_positions_3d=self.anchor_positions_3d.copy(),
            true_ranges=self.true_ranges.copy(),
            measured_ranges=self.measured_ranges.copy(),
            bias_values=self.bias_values.copy(),
            sigma_values=self.sigma_values.copy(),
            random_noise_values=self.random_noise_values.copy(),
            jacobian_h_3d=self.jacobian_h_3d.copy(),
            P_3d=self.P_3d.copy(),
            b_3d=self.b_3d.copy(),
            M_3d=self.M_3d.copy(),
            P_xy=self.P_xy.copy(),
            b_xy=self.b_xy.copy(),
            M_xy=self.M_xy.copy(),
            confidence_alpha=float(self.confidence_alpha),
            est_position_timestamp=float(self.est_position_timestamp),
        )
