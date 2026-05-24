from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


LookupTable = Sequence[Tuple[float, float]]


@dataclass(frozen=True)
class RangeMeasurementResult:
    true_ranges: np.ndarray
    measured_ranges: np.ndarray
    bias_values: np.ndarray
    sigma_values: np.ndarray
    random_noise_values: np.ndarray


class LocalizationErrorModel:
    sigma_r_const: float = 0.04

    mu_bias_table: LookupTable = (
        (1.0, 0.40),
        (3.0, 0.44),
        (5.0, 0.50),
        (8.0, 0.60),
        (10.0, 0.67),
        (12.0, 0.70),
        (15.0, 0.72),
        (18.0, 0.76),
        (20.0, 0.80),
    )

    def __init__(self):
        pass

    @staticmethod
    def _interp_table(distance: float, table: LookupTable) -> float:
        xs = np.asarray([row[0] for row in table], dtype=float)
        ys = np.asarray([row[1] for row in table], dtype=float)
        d = float(np.clip(distance, xs[0], xs[-1]))
        return float(np.interp(d, xs, ys))

    def sigma_r(self, distance: float) -> float:
        _ = distance
        return float(self.sigma_r_const)

    def mu_bias(self, distance: float) -> float:
        return self._interp_table(distance, self.mu_bias_table)

    def perturb_ranges(
        self,
        true_ranges: Iterable[float],
        rng: np.random.Generator,
        *,
        entity_key: str = "default",
        timestamp: float | None = None,
    ) -> RangeMeasurementResult:
        _ = entity_key
        _ = timestamp
        true_ranges = np.asarray(list(true_ranges), dtype=float)
        sigma_values = np.asarray([self.sigma_r(d) for d in true_ranges], dtype=float)
        bias_values = np.asarray([self.mu_bias(d) for d in true_ranges], dtype=float)
        random_noise_values = rng.normal(loc=0.0, scale=sigma_values)
        measured_ranges = true_ranges + bias_values + random_noise_values
        return RangeMeasurementResult(
            true_ranges=true_ranges,
            measured_ranges=measured_ranges,
            bias_values=bias_values,
            sigma_values=sigma_values,
            random_noise_values=random_noise_values,
        )
