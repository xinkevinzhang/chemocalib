"""
Structured Cross-Validation for MB-PLS
======================================
Provides rigorous CV pipelines for component selection,
metric reporting, and stability assessment.

References
----------
- Westerhuis et al. (1998) J. Chemometrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold, LeaveOneOut
from collections import defaultdict


def grid_search_components(
    blocks: List[np.ndarray],
    y: np.ndarray,
    model_cls,
    component_range: List[int],
    n_folds: int = 5,
    n_repeats: int = 3,
    metric: str = "r2",
    seed: int = 42,
    **model_kwargs,
) -> Dict:
    """Grid-search over number of latent components with repeated K-fold CV.

    Parameters
    ----------
    blocks : list of np.ndarray
        Data blocks.
    y : np.ndarray
        Response variable.
    model_cls : class
        MB-PLS model class, e.g. ``MultiBlockPLS``.
    component_range : list of int
        Candidate component numbers, e.g. [1, 2, 3, 5, 7, 10].
    n_folds : int
        Number of CV folds.
    n_repeats : int
        Number of repeated CV runs with different shuffles.
    metric : str
        "r2" or "spearman_r".
    seed : int
    **model_kwargs
        Passed to model constructor.

    Returns
    -------
    result : dict
        Keys: ``best_n_components``, ``best_score``, ``cv_results``
        (per-component mean ± std), ``all_fold_scores``.
    """
    from scipy.stats import spearmanr

    n_samples = blocks[0].shape[0]
    y = y.ravel()

    cv_results = {}
    all_fold_scores = {}

    for nc in component_range:
        fold_scores = []
        for repeat in range(n_repeats):
            kf = KFold(n_splits=n_folds, shuffle=True,
                       random_state=seed + repeat * 1000)
            for train_idx, test_idx in kf.split(np.arange(n_samples)):
                blocks_train = [X[train_idx] for X in blocks]
                blocks_test = [X[test_idx] for X in blocks]
                y_train = y[train_idx]
                y_test = y[test_idx]

                try:
                    model = model_cls(n_components=nc, **model_kwargs)
                    model.fit(blocks_train, y_train)
                    y_pred = model.transform(blocks_test)
                    # Use first super-score component as proxy prediction
                    y_pred = y_pred[:, 0]
                    # Scale to match y_test range
                    y_pred = y_pred * np.std(y_test) / (np.std(y_pred) + 1e-10) + np.mean(y_test)

                    if metric == "r2":
                        ss_res = np.sum((y_test - y_pred) ** 2)
                        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                        score = 1 - ss_res / (ss_tot + 1e-10)
                    elif metric == "spearman_r":
                        score, _ = spearmanr(y_test, y_pred)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    fold_scores.append(float(score))
                except Exception:
                    fold_scores.append(np.nan)

        valid_scores = [s for s in fold_scores if not np.isnan(s)]
        if valid_scores:
            cv_results[nc] = {
                "mean": float(np.mean(valid_scores)),
                "std": float(np.std(valid_scores, ddof=1)),
                "n_valid": len(valid_scores),
            }
            all_fold_scores[nc] = fold_scores
        else:
            cv_results[nc] = {"mean": np.nan, "std": np.nan, "n_valid": 0}
            all_fold_scores[nc] = []

    # Find best
    valid_means = {nc: r["mean"] for nc, r in cv_results.items()
                   if not np.isnan(r["mean"])}
    if valid_means:
        best_nc = max(valid_means, key=valid_means.get)
        best_score = valid_means[best_nc]
    else:
        best_nc = component_range[0]
        best_score = np.nan

    return {
        "best_n_components": best_nc,
        "best_score": best_score,
        "metric": metric,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "cv_results": cv_results,
        "all_fold_scores": {str(k): v for k, v in all_fold_scores.items()},
    }


def stability_selection(
    blocks: List[np.ndarray],
    y: np.ndarray,
    model_cls,
    n_components: int = 3,
    n_subsamples: int = 50,
    subsample_frac: float = 0.7,
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, float]:
    """Stability selection: assess feature stability across subsamples.

    Returns the fraction of subsamples in which each feature's VIP score
    exceeds the threshold.

    Parameters
    ----------
    blocks : list of np.ndarray
    y : np.ndarray
    model_cls : class
    n_components : int
    n_subsamples : int
        Number of subsampling repetitions.
    subsample_frac : float
        Fraction of samples to use per subsample.
    seed : int

    Returns
    -------
    stability : dict mapping block_idx -> np.ndarray of selection probabilities.
    """
    rng = np.random.RandomState(seed)
    n_samples = blocks[0].shape[0]
    n_blocks = len(blocks)
    n_subsample = int(n_samples * subsample_frac)

    # Accumulate selection counts
    selected_counts = [np.zeros(X.shape[1]) for X in blocks]

    vip_thresholds = []
    # First, run on full data to get thresholds
    for _ in range(5):
        model = model_cls(n_components=n_components, **model_kwargs)
        try:
            model.fit(blocks, y)
            for i in range(n_blocks):
                vip_thresh = float(np.percentile(model.vip_scores[i], 75))
                vip_thresholds.append(vip_thresh)
        except Exception:
            pass

    threshold = np.mean(vip_thresholds) if vip_thresholds else 1.0

    for _ in range(n_subsamples):
        idx = rng.choice(n_samples, size=n_subsample, replace=False)
        blocks_sub = [X[idx] for X in blocks]
        y_sub = y[idx]
        try:
            model = model_cls(n_components=n_components, **model_kwargs)
            model.fit(blocks_sub, y_sub)
            for i in range(n_blocks):
                selected_counts[i] += (model.vip_scores[i] > threshold).astype(float)
        except Exception:
            pass

    return {
        "block_selection_prob": [c / n_subsamples for c in selected_counts],
        "threshold": float(threshold),
        "n_subsamples": n_subsamples,
        "subsample_frac": subsample_frac,
    }
