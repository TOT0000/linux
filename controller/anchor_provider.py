from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class AnchorProfile:
    name: str
    workspace_bounds: Dict[str, Tuple[float, float]]
    anchors: Tuple[Vector3, ...]

    @property
    def workspace_center(self) -> np.ndarray:
        return np.array(
            [
                (self.workspace_bounds["x"][0] + self.workspace_bounds["x"][1]) / 2.0,
                (self.workspace_bounds["y"][0] + self.workspace_bounds["y"][1]) / 2.0,
                (self.workspace_bounds["z"][0] + self.workspace_bounds["z"][1]) / 2.0,
            ],
            dtype=float,
        )

    def anchor_array(self) -> np.ndarray:
        return np.asarray(self.anchors, dtype=float)


class AnchorGeometryProvider:
    DEFAULT_PROFILE = "cuboid_8anchors_outdoor_main"

    def __init__(self, profile_name: str = DEFAULT_PROFILE):
        self._profiles = {
            self.DEFAULT_PROFILE: AnchorProfile(
                name=self.DEFAULT_PROFILE,
                workspace_bounds={
                    "x": (0.0, 12.0),
                    "y": (0.0, 12.0),
                    "z": (1.0, 5.0),
                },
                anchors=(
                    (0.0, 0.0, 2.5),
                    (12.0, 0.0, 2.5),
                    (12.0, 12.0, 2.5),
                    (0.0, 12.0, 2.5),
                    (0.0, 0.0, 5.5),
                    (12.0, 0.0, 5.5),
                    (12.0, 12.0, 5.5),
                    (0.0, 12.0, 5.5),
                ),
            )
        }
        if profile_name not in self._profiles:
            raise ValueError(f"Unknown anchor profile: {profile_name}")
        self._profile_name = profile_name

    def get_profile(self) -> AnchorProfile:
        return self._profiles[self._profile_name]

    def get_profile_names(self) -> List[str]:
        return list(self._profiles.keys())

    def get_anchor_positions(self) -> np.ndarray:
        return self.get_profile().anchor_array()

    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.get_profile().workspace_bounds)

    def get_workspace_center(self) -> np.ndarray:
        return self.get_profile().workspace_center.copy()
