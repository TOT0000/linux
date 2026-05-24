from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state_packet import LocalizedStatePacket


CHI2_2_095 = 5.991


@dataclass
class SafetyEnvelope2D:
    entity_type: str
    center_xy: np.ndarray
    major_axis_radius: float
    minor_axis_radius: float
    orientation_rad: float
    orientation_deg: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    chi2_val: float
    matrix_xy: np.ndarray
    packet_generation_timestamp: float
    sequence_number: int

    def ray_radius(self, direction_xy: np.ndarray) -> float:
        """
        Center-to-boundary distance along a ray direction for the ellipse
        defined by (x-c)^T M_xy^{-1} (x-c) = chi2_val.

        For unit direction u, the ray radius is:
            rho(u) = sqrt(chi2_val / (u^T M_xy^{-1} u))
        """
        direction = np.asarray(direction_xy, dtype=float).reshape(2)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return 0.0
        u = direction / norm
        inv_matrix = np.linalg.pinv(self.matrix_xy)
        quad = float(u.T @ inv_matrix @ u)
        quad = max(quad, 1e-12)
        return float(np.sqrt(self.chi2_val / quad))

    def directional_radius(self, unit_vec_xy: np.ndarray) -> float:
        # Backward-compatible alias; now returns the centerline ray radius
        # instead of support/projection radius semantics.
        return self.ray_radius(unit_vec_xy)


def build_safety_envelope(packet: LocalizedStatePacket, chi2_val: float = CHI2_2_095) -> SafetyEnvelope2D:
    matrix_xy = np.asarray(packet.M_xy, dtype=float)
    center_xy = np.asarray(packet.estimated_position_3d[:2], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix_xy)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    major_axis_radius = float(np.sqrt(chi2_val * eigenvalues[0]))
    minor_axis_radius = float(np.sqrt(chi2_val * eigenvalues[1]))
    orientation_rad = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    orientation_deg = float(np.degrees(orientation_rad))

    return SafetyEnvelope2D(
        entity_type=packet.entity_type,
        center_xy=center_xy,
        major_axis_radius=major_axis_radius,
        minor_axis_radius=minor_axis_radius,
        orientation_rad=orientation_rad,
        orientation_deg=orientation_deg,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chi2_val=float(chi2_val),
        matrix_xy=matrix_xy,
        packet_generation_timestamp=float(packet.state_generation_timestamp),
        sequence_number=int(packet.sequence_number),
    )
