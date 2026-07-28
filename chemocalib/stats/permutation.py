"""
Permutation test & bootstrap confidence intervals
==================================================
Statistical validation for MB-PLS models and
chemometric constraint mappings.

References
----------
- Westerhuis et al. (1998) J. Chemometrics
- Szymanska et al. (2012) Metabolomics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


def permutation_test(
    model_factory: Callable,
    blocks: List[np.ndarray],
    y: np.ndarray,
    n_permutations: int = 1000,
    metric: str = "r2",
    seed: int = 42,
) -> Dict:
    """Permutation test for MB-PLS model significance.

    Tests the null hypothesis that the observed model performance
    is no better than a model fit on permuted response labels.

    Parameters
    ----------
    model_factory : callable
        Function that returns a fresh unfitted model instance,
        e.g. ``lambda: MultiBlockPLS(n_components=3)``.
    blocks : list of np.ndarray
        Data blocks [X1, X2, ..., Xk].
    y : np.ndarray, shape (n_samples,)
        Response variable.
    n_permutations : int
        Number of permutations (default 1000).
    metric : str
        Performance metric: "r2" or "spearman_r".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Keys: ``metric``, ``observed``, ``p_value``, ``null_distribution``,
        ``ci95_null``, ``permutations``.
    """
    rng = np.random.RandomState(seed)

    # Observed performance
    model = model_factory()
    model.fit(blocks, y)
    y_hat = _predict_from_model(model, blocks)
    observed = _compute_metric(y, y_hat, metric)

    # Null distribution
    null_values = np.zeros(n_permutations)
    for p in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        model_p = model_factory()
        model_p.fit(blocks, y_perm)
        y_hat_p = _predict_from_model(model_p, blocks)
        null_values[p] = _compute_metric(y_perm, y_hat_p, metric)

    # P-value: fraction of null >= observed
    p_value = float(np.mean(null_values >= observed))
    if p_value == 0:
        p_value = 1.0 / (n_permutations + 1)

    return {
        "metric": metric,
        "observed": float(observed),
        "p_value": p_value,
        "null_distribution": null_values.tolist(),
        "ci95_null": [float(np.percentile(null_values, 2.5)),
                       float(np.percentile(null_values, 97.5))],
        "n_permutations": n_permutations,
        "significant_at_005": p_value < 0.05,
    }


def bootstrap_ci(
    model_factory: Callable,
    blocks: List[np.ndarray],
    y: np.ndarray,
    n_bootstrap: int = 500,
    metric: str = "r2",
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence intervals for model metrics.

    Uses case resampling (with replacement) to estimate
    the empirical distribution of a performance metric.

    Parameters
    ----------
    model_factory : callable
        Returns a fresh unfitted model instance.
    blocks : list of np.ndarray
    y : np.ndarray
    n_bootstrap : int
        Number of bootstrap replicates.
    metric : str
        "r2" or "spearman_r".
    alpha : float
        Significance level for CI (default 0.05 → 95% CI).
    seed : int

    Returns
    -------
    result : dict
        Keys: ``metric``, ``point_estimate``, ``ci_lower``, ``ci_upper``,
        ``bias``, ``std_err``, ``distribution``.
    """
    rng = np.random.RandomState(seed)
    n_samples = blocks[0].shape[0]

    estimates = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        blocks_b = [X[idx] for X in blocks]
        y_b = y[idx]
        model = model_factory()
        model.fit(blocks_b, y_b)
        y_hat = _predict_from_model(model, blocks)
        estimates[b] = _compute_metric(y, y_hat, metric)

    point = float(np.mean(estimates))
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    bias = float(np.mean(estimates) - estimates[0])
    std_err = float(np.std(estimates, ddof=1))

    return {
        "metric": metric,
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "bias": bias,
        "std_err": std_err,
        "n_bootstrap": n_bootstrap,
        "distribution": estimates.tolist(),
    }


def bootstrap_vip_ci(
    model_factory: Callable,
    blocks: List[np.ndarray],
    y: np.ndarray,
    block_idx: int = 0,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Bootstrap confidence intervals for VIP scores.

    Parameters
    ----------
    model_factory : callable
    blocks : list of np.ndarray
    y : np.ndarray
    block_idx : int
        Which block's VIP scores to bootstrap.
    n_bootstrap : int
    alpha : float
    seed : int

    Returns
    -------
    result : dict
        Keys: ``vip_mean``, ``vip_ci_lower``, ``vip_ci_upper``,
        ``reliable_mask`` (True where CI excludes zero or 1.0).
    """
    rng = np.random.RandomState(seed)
    n_samples = blocks[0].shape[0]
    n_features = blocks[block_idx].shape[1]

    # Point estimate
    model = model_factory()
    model.fit(blocks, y)
    vip_point = model.vip_scores[block_idx].copy()

    # Bootstrap distribution
    vip_dist = np.zeros((n_bootstrap, n_features))
    for b in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        blocks_b = [X[idx] for X in blocks]
        y_b = y[idx]
        try:
            model_b = model_factory()
            model_b.fit(blocks_b, y_b)
            vip_dist[b] = model_b.vip_scores[block_idx]
        except Exception:
            vip_dist[b] = np.nan

    vip_mean = np.nanmean(vip_dist, axis=0)
    vip_lower = np.nanpercentile(vip_dist, 100 * alpha / 2, axis=0)
    vip_upper = np.nanpercentile(vip_dist, 100 * (1 - alpha / 2), axis=0)

    reliable = vip_lower > 0.8  # VIP > 0.8 typically considered "interesting"

    return {
        "vip_mean": vip_mean.tolist(),
        "vip_ci_lower": vip_lower.tolist(),
        "vip_ci_upper": vip_upper.tolist(),
        "reliable_indices": np.where(reliable)[0].tolist(),
        "n_reliable": int(np.sum(reliable)),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def _predict_from_model(model, blocks: List[np.ndarray]) -> np.ndarray:
    """Extract predictions from an MB-PLS-like model."""
    # Try common prediction interfaces
    if hasattr(model, "predict"):
        return model.predict(blocks)
    # Fallback: use super_scores to reconstruct Y via y_loadings
    if hasattr(model, "super_scores") and hasattr(model, "y_loadings"):
        if model.super_scores is not None and model.y_loadings is not None:
            return model.super_scores @ model.y_loadings.ravel()
    # Last resort: transform + simple regression
    if hasattr(model, "transform"):
        T = model.transform(blocks)
        return T[:, 0]  # Use first component as proxy
    raise ValueError("Cannot extract predictions from model")


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute a regression metric."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if metric == "r2":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))
    elif metric == "spearman_r":
        from scipy.stats import spearmanr
        r, _ = spearmanr(y_true, y_pred)
        return float(r)
    else:
        raise ValueError(f"Unknown metric: {metric}")
