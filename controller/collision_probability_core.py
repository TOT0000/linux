from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import os
import numpy as np


@dataclass(frozen=True)
class CollisionEntity2D:
    """2D Gaussian state container for collision-probability evaluation.

    Attributes:
        entity_id: Stable identifier used in per-entity probability output.
        mean_xy: Estimated 2D position (x, y).
        cov_xy: 2x2 covariance matrix in position domain.
        bias_xy: 2D position-domain bias vector.
        radius_m: Circular safety/collision radius for this entity.
    """

    entity_id: str
    mean_xy: np.ndarray
    cov_xy: np.ndarray
    bias_xy: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class CollisionProbabilityResult:
    entity_id: str
    probability: float
    soft_probability: float
    approximate_probability: float
    hard_approx_probability: float
    mu_xy: np.ndarray
    sigma_rel: np.ndarray
    lambdas: np.ndarray
    transformed_b: np.ndarray
    terms_used: int
    converged: bool
    exact_series_probability: float
    monte_carlo_probability: Optional[float] = None


@dataclass(frozen=True)
class SceneCollisionSummary:
    per_entity: Tuple[CollisionProbabilityResult, ...]
    current_probability: float
    historical_max_probability: float
    dominant_entity_id: str
    sanity_case_probabilities: Optional[Dict[str, float]] = None


def _eigh_symmetric_psd(matrix: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, eps)
    return values, vectors


def _sqrtm_and_invsqrtm_psd(matrix: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    values, vectors = _eigh_symmetric_psd(matrix, eps=eps)
    sqrt_values = np.sqrt(values)
    invsqrt_values = 1.0 / sqrt_values
    sqrt_matrix = vectors @ np.diag(sqrt_values) @ vectors.T
    invsqrt_matrix = vectors @ np.diag(invsqrt_values) @ vectors.T
    return sqrt_matrix, invsqrt_matrix


def quadratic_form_cdf_exact_series(
    mu_xy: np.ndarray,
    sigma_xy: np.ndarray,
    A: np.ndarray,
    q: float = 1.0,
    max_terms: int = 256,
    tolerance: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray, int, bool]:
    """Exact-series CDF for quadratic form v = z^T A z with z~N(mu, Sigma).

    This follows the series form requested by the task definition:
        F_v(q) = sum_k (-1)^k c_k q^(n/2+k) / Gamma(n/2+k+1)
    with c_k generated from d_k recursion.
    """

    mu_xy = np.asarray(mu_xy, dtype=float).reshape(2)
    sigma_xy = np.asarray(sigma_xy, dtype=float).reshape(2, 2)
    A = np.asarray(A, dtype=float).reshape(2, 2)
    A = 0.5 * (A + A.T)
    sigma_xy = 0.5 * (sigma_xy + sigma_xy.T)

    n = int(mu_xy.shape[0])
    q = float(max(q, 0.0))
    if q == 0.0:
        return 0.0, np.ones(n, dtype=float), np.zeros(n, dtype=float), 1, True

    sigma_sqrt, sigma_invsqrt = _sqrtm_and_invsqrtm_psd(sigma_xy, eps=1e-10)
    M = sigma_sqrt @ A @ sigma_sqrt
    lambda_vals, U = _eigh_symmetric_psd(M, eps=1e-12)

    shifted_mean = sigma_invsqrt @ mu_xy
    transformed_b = U.T @ shifted_mean

    # c0
    exp_term = math.exp(-0.5 * float(np.sum(np.square(transformed_b))))
    prod_term = float(np.prod(np.power(2.0 * lambda_vals, -0.5)))
    c = [exp_term * prod_term]

    base_vals = np.maximum(2.0 * lambda_vals, 1e-300)
    log_base_vals = np.log(base_vals)
    negative_logs = -log_base_vals[log_base_vals < 0.0]
    dynamic_k_limit = int(max_terms)
    if negative_logs.size > 0:
        max_neg = float(np.max(negative_logs))
        if max_neg > 0.0:
            dynamic_k_limit = min(dynamic_k_limit, max(6, int(700.0 / max_neg)))

    # Precompute d_k lazily as needed by recursion.
    def compute_d(k: int) -> float:
        exponent = -float(k) * log_base_vals
        inv_term = np.exp(np.clip(exponent, -700.0, 700.0))
        value = 0.5 * float(np.sum((1.0 - (float(k) * np.square(transformed_b))) * inv_term))
        return value

    total = 0.0
    converged = False
    terms_used = 0
    half_n = 0.5 * float(n)

    for k in range(dynamic_k_limit):
        if k > 0:
            ck = 0.0
            for i in range(k):
                d = compute_d(k - i)
                ck += d * c[i]
            ck = ck / float(k)
            if not math.isfinite(ck):
                break
            c.append(ck)
        ck = c[k]

        if ck == 0.0:
            term = 0.0
        else:
            log_term_abs = math.log(abs(float(ck))) - math.lgamma(half_n + float(k) + 1.0)
            if q > 0.0:
                log_term_abs += (half_n + float(k)) * math.log(q)
            if log_term_abs > 700.0:
                break
            if log_term_abs < -745.0:
                term = 0.0
            else:
                term_mag = math.exp(log_term_abs)
                term = ((-1.0) ** k) * (1.0 if ck >= 0 else -1.0) * term_mag
        total += term
        terms_used = k + 1
        if abs(term) < tolerance:
            converged = True
            break

    cdf = float(np.clip(total, 0.0, 1.0))
    return cdf, lambda_vals, transformed_b, terms_used, converged


def _stable_cholesky_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(2, 2)
    matrix = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, eps)
    stabilized = vecs @ np.diag(vals) @ vecs.T
    stabilized = 0.5 * (stabilized + stabilized.T)
    jitter = eps
    for _ in range(6):
        try:
            return np.linalg.cholesky(stabilized + jitter * np.eye(2, dtype=float))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(stabilized + 1e-4 * np.eye(2, dtype=float))


def approximate_collision_probability_gauss_hermite(
    mu_xy: np.ndarray,
    sigma_xy: np.ndarray,
    r_c: float,
    *,
    quadrature_order: int = 20,
    tau_v: float = 0.10,
) -> float:
    mu_xy = np.asarray(mu_xy, dtype=float).reshape(2)
    sigma_xy = np.asarray(sigma_xy, dtype=float).reshape(2, 2)
    r_c = float(max(r_c, 1e-6))
    L = _stable_cholesky_psd(sigma_xy, eps=1e-10)
    nodes, weights = np.polynomial.hermite.hermgauss(int(max(2, quadrature_order)))
    acc = 0.0
    sqrt2 = math.sqrt(2.0)
    tau_v = float(max(1e-6, tau_v))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            xi = np.array([nodes[i], nodes[j]], dtype=float)
            sample = mu_xy + sqrt2 * (L @ xi)
            v = float(np.dot(sample, sample) / (r_c * r_c))
            sigmoid_input = (1.0 - v) / tau_v
            sigmoid_input = max(min(sigmoid_input, 60.0), -60.0)
            soft_value = 1.0 / (1.0 + math.exp(-sigmoid_input))
            acc += float(weights[i] * weights[j]) * float(soft_value)
    estimate = (1.0 / math.pi) * float(acc)
    return float(np.clip(estimate, 0.0, 1.0))


def hard_collision_probability_gauss_hermite(
    mu_xy: np.ndarray,
    sigma_xy: np.ndarray,
    r_c: float,
    *,
    quadrature_order: int = 20,
) -> float:
    mu_xy = np.asarray(mu_xy, dtype=float).reshape(2)
    sigma_xy = np.asarray(sigma_xy, dtype=float).reshape(2, 2)
    r_c = float(max(r_c, 1e-6))
    L = _stable_cholesky_psd(sigma_xy, eps=1e-10)
    nodes, weights = np.polynomial.hermite.hermgauss(int(max(2, quadrature_order)))
    acc = 0.0
    sqrt2 = math.sqrt(2.0)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            xi = np.array([nodes[i], nodes[j]], dtype=float)
            sample = mu_xy + sqrt2 * (L @ xi)
            inside = float(np.linalg.norm(sample) <= r_c)
            acc += float(weights[i] * weights[j]) * inside
    estimate = (1.0 / math.pi) * float(acc)
    return float(np.clip(estimate, 0.0, 1.0))


class CollisionProbabilityCore:
    """Stateful collision-probability engine with historical-max tracking."""

    def __init__(self):
        self._historical_max_probability = 0.0
        self._debug_mc_cache: Dict[Tuple[str, Tuple[float, ...], Tuple[float, ...], float], float] = {}
        self._sanity_cache: Optional[Dict[str, float]] = None
        self._gh_order = int(os.getenv("COLLISION_GAUSS_HERMITE_ORDER", "20"))
        self._soft_tau_v = float(os.getenv("COLLISION_SOFT_TAU_V", "0.10"))

    def reset_history(self):
        self._historical_max_probability = 0.0

    def get_historical_max_probability(self) -> float:
        return float(self._historical_max_probability)

    def evaluate_scene(
        self,
        uav: CollisionEntity2D,
        workers: List[CollisionEntity2D],
        *,
        max_terms: int = 256,
        tolerance: float = 1e-12,
    ) -> SceneCollisionSummary:
        results: List[CollisionProbabilityResult] = []
        debug_mc_enabled = _env_flag("DEBUG_COLLISION_MONTE_CARLO", default=False)

        # Bias-corrected UAV mean: p_u_hat - b_u
        uav_mean = np.asarray(uav.mean_xy, dtype=float).reshape(2) - np.asarray(uav.bias_xy, dtype=float).reshape(2)
        uav_cov = np.asarray(uav.cov_xy, dtype=float).reshape(2, 2)

        for worker in workers:
            # Bias-corrected worker mean: p_k_hat - b_k
            worker_mean = np.asarray(worker.mean_xy, dtype=float).reshape(2) - np.asarray(worker.bias_xy, dtype=float).reshape(2)
            worker_cov = np.asarray(worker.cov_xy, dtype=float).reshape(2, 2)

            # Relative Gaussian
            mu_k = worker_mean - uav_mean
            sigma_rel = worker_cov + uav_cov

            # Circle collision region quadratic form
            r_c = max(1e-6, float(uav.radius_m) + float(worker.radius_m))
            A = (1.0 / (r_c * r_c)) * np.eye(2, dtype=float)

            p_soft = approximate_collision_probability_gauss_hermite(
                mu_xy=mu_k,
                sigma_xy=sigma_rel,
                r_c=float(r_c),
                quadrature_order=self._gh_order,
                tau_v=self._soft_tau_v,
            )
            p_hard_approx = hard_collision_probability_gauss_hermite(
                mu_xy=mu_k,
                sigma_xy=sigma_rel,
                r_c=float(r_c),
                quadrature_order=self._gh_order,
            )
            p_exact, lambdas, transformed_b, terms_used, converged = quadratic_form_cdf_exact_series(
                mu_xy=mu_k,
                sigma_xy=sigma_rel,
                A=A,
                q=1.0,
                max_terms=max_terms,
                tolerance=tolerance,
            )
            p_ck = float(p_hard_approx)

            p_mc = None
            if debug_mc_enabled and worker.entity_id == "worker_3":
                p_mc = self._estimate_collision_probability_monte_carlo(
                    entity_id=str(worker.entity_id),
                    mu_k=mu_k,
                    sigma_rel=sigma_rel,
                    r_c=float(r_c),
                    samples=20000,
                )

            results.append(
                CollisionProbabilityResult(
                    entity_id=str(worker.entity_id),
                    probability=float(p_ck),
                    soft_probability=float(p_soft),
                    approximate_probability=float(p_hard_approx),
                    hard_approx_probability=float(p_hard_approx),
                    mu_xy=mu_k,
                    sigma_rel=sigma_rel,
                    lambdas=lambdas,
                    transformed_b=transformed_b,
                    terms_used=int(terms_used),
                    converged=bool(converged),
                    exact_series_probability=float(p_exact),
                    monte_carlo_probability=(None if p_mc is None else float(p_mc)),
                )
            )

        if not results:
            current = 0.0
            dominant_id = "none"
        else:
            dominant = max(results, key=lambda item: item.probability)
            current = float(dominant.probability)
            dominant_id = str(dominant.entity_id)

        self._historical_max_probability = max(float(self._historical_max_probability), current)
        sanity = self._get_sanity_case_probabilities(max_terms=max_terms, tolerance=tolerance)
        return SceneCollisionSummary(
            per_entity=tuple(results),
            current_probability=current,
            historical_max_probability=float(self._historical_max_probability),
            dominant_entity_id=dominant_id,
            sanity_case_probabilities=sanity,
        )

    def _estimate_collision_probability_monte_carlo(
        self,
        *,
        entity_id: str,
        mu_k: np.ndarray,
        sigma_rel: np.ndarray,
        r_c: float,
        samples: int,
    ) -> float:
        mu_key = tuple(float(round(v, 4)) for v in np.asarray(mu_k, dtype=float).reshape(-1))
        sigma_key = tuple(float(round(v, 5)) for v in np.asarray(sigma_rel, dtype=float).reshape(-1))
        key = (str(entity_id), mu_key, sigma_key, float(round(r_c, 4)))
        if key in self._debug_mc_cache:
            return float(self._debug_mc_cache[key])
        rng = np.random.default_rng(20260403)
        draws = rng.multivariate_normal(mean=np.asarray(mu_k, dtype=float), cov=np.asarray(sigma_rel, dtype=float), size=int(samples))
        norms = np.linalg.norm(draws, axis=1)
        estimate = float(np.mean(norms <= float(r_c)))
        self._debug_mc_cache[key] = estimate
        return estimate

    def _get_sanity_case_probabilities(self, max_terms: int, tolerance: float) -> Dict[str, float]:
        if self._sanity_cache is not None:
            return dict(self._sanity_cache)
        try:
            case1_mu = np.array([0.0, 0.0], dtype=float)
            case2_mu = np.array([3.0, 0.0], dtype=float)
            case_sigma = np.array([[1e-4, 0.0], [0.0, 1e-4]], dtype=float)
            rc = 0.52
            A = (1.0 / (rc * rc)) * np.eye(2, dtype=float)
            case1_prob, _, _, _, _ = quadratic_form_cdf_exact_series(case1_mu, case_sigma, A, q=1.0, max_terms=max_terms, tolerance=tolerance)
            case2_prob, _, _, _, _ = quadratic_form_cdf_exact_series(case2_mu, case_sigma, A, q=1.0, max_terms=max_terms, tolerance=tolerance)
            self._sanity_cache = {"case1_exact": float(case1_prob), "case2_exact": float(case2_prob)}
        except Exception:
            self._sanity_cache = {"case1_exact": float("nan"), "case2_exact": float("nan")}
        return dict(self._sanity_cache)



def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "true" if default else "false")).strip().lower()
    return value in {"1", "true", "yes", "on"}
