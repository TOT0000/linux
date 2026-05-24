from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .safety_envelope import SafetyEnvelope2D, build_safety_envelope
from .state_packet import LocalizedStatePacket


@dataclass
class GcsSafetyState:
    drone_packet: LocalizedStatePacket
    user_packet: LocalizedStatePacket
    drone_envelope: SafetyEnvelope2D
    user_envelope: SafetyEnvelope2D
    drone_center_xy: np.ndarray
    user_center_xy: np.ndarray
    delta_xy: np.ndarray
    drone_to_user_distance_xy: float
    drone_radius_along_user_direction: float
    user_radius_along_drone_direction: float
    envelope_gap_m: float
    envelopes_overlap: bool
    latest_generation_timestamp: float


class GcsSafetyStateService:
    @staticmethod
    def build(
        drone_packet: LocalizedStatePacket,
        user_packet: LocalizedStatePacket,
    ) -> GcsSafetyState:
        drone_envelope = build_safety_envelope(drone_packet)
        user_envelope = build_safety_envelope(user_packet)

        drone_center_xy = drone_envelope.center_xy
        user_center_xy = user_envelope.center_xy
        delta_xy = drone_center_xy - user_center_xy
        distance_xy = float(np.linalg.norm(delta_xy))

        if distance_xy < 1e-9:
            unit_vec = np.array([1.0, 0.0], dtype=float)
        else:
            unit_vec = delta_xy / distance_xy

        # Centerline ray radii: shoot from each center along the line-of-centers.
        drone_radius = drone_envelope.ray_radius(unit_vec)
        user_radius = user_envelope.ray_radius(-unit_vec)
        envelope_gap_m = float(distance_xy - (drone_radius + user_radius))

        return GcsSafetyState(
            drone_packet=drone_packet.copy(),
            user_packet=user_packet.copy(),
            drone_envelope=drone_envelope,
            user_envelope=user_envelope,
            drone_center_xy=drone_center_xy.copy(),
            user_center_xy=user_center_xy.copy(),
            delta_xy=delta_xy.copy(),
            drone_to_user_distance_xy=distance_xy,
            drone_radius_along_user_direction=drone_radius,
            user_radius_along_drone_direction=user_radius,
            envelope_gap_m=envelope_gap_m,
            # NOTE: this is centerline (line-of-centers) overlap, not full 2D ellipse intersection truth.
            envelopes_overlap=bool(envelope_gap_m < 0.0),
            latest_generation_timestamp=float(
                max(drone_packet.state_generation_timestamp, user_packet.state_generation_timestamp)
            ),
        )

    @staticmethod
    def build_from_provider(state_provider, now: Optional[float] = None) -> Optional[GcsSafetyState]:
        if hasattr(state_provider, "flush_due_packets"):
            state_provider.flush_due_packets(now=now)
        drone_packet = state_provider.get_latest_received_drone_packet()
        user_packet = state_provider.get_latest_received_user_packet()
        if drone_packet is None or user_packet is None:
            return None
        return GcsSafetyStateService.build(drone_packet, user_packet)
