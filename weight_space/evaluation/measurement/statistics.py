"""Statistical utilities for rigorous confidence intervals and uncertainty estimation.

Implements:
1. Wilson score confidence interval for binomial proportions (with continuity correction or standard Wilson).
2. Non-parametric bootstrap confidence interval for general metrics.
"""

import math
from typing import List, Tuple
import numpy as np


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates the Wilson score confidence interval for a binomial proportion.

    Unlike the standard Wald normal approximation (p +/- z*sqrt(p(1-p)/n)), the Wilson score
    interval remains mathematically valid near boundary conditions (p = 0 or p = 1) and small sample sizes.

    Args:
        successes: Number of positive outcomes (e.g., correct behavioral choices or flagged persona shifts).
        total: Total number of trials (sample size n).
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (center_estimate, lower_bound, upper_bound) in percentage [0.0, 100.0].
    """
    if total <= 0:
        return 0.0, 0.0, 0.0

    p = successes / total
    # Two-tailed standard normal critical value
    # For 95% confidence, z ~ 1.95996
    if confidence == 0.95:
        z = 1.959963984540054
    elif confidence == 0.99:
        z = 2.5758293035489004
    elif confidence == 0.90:
        z = 1.6448536269514722
    else:
        # Generic inverse error function approximation
        z = math.sqrt(2.0) * math.erf(confidence)

    denominator = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denominator
    half_width = (z / denominator) * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4.0 * total**2))

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)

    return round(p * 100.0, 2), round(lower * 100.0, 2), round(upper * 100.0, 2)


def compute_wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Convenience wrapper returning (lower_bound, upper_bound) in percent [0.0, 100.0]."""
    _, lower, upper = wilson_score_interval(successes, total, confidence)
    return lower, upper


def bootstrap_confidence_interval(
    values: List[float], n_bootstraps: int = 1000, confidence: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Computes a non-parametric percentile bootstrap confidence interval.

    Args:
        values: 1D array-like of sample observations.
        n_bootstraps: Number of bootstrap resamples (default 1000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)

    # Vectorized bootstrap resampling
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(arr[indices], axis=1)

    alpha_tail = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, alpha_tail * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha_tail) * 100.0))
    point_mean = float(np.mean(arr))

    return round(point_mean, 2), round(lower, 2), round(upper, 2)
