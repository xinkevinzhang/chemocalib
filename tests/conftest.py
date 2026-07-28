"""Shared fixtures for ChemoCalib test suite."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Deterministic random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_blocks(rng):
    """Three-block synthetic multi-omics data: 100 samples, 30/20/15 features."""
    n_samples = 100
    latent1 = rng.normal(0, 1, (n_samples, 1))
    latent2 = rng.normal(0, 1, (n_samples, 1))
    latent = np.hstack([latent1, latent2])

    W1 = rng.normal(0, 1, (2, 30))
    W2 = rng.normal(0, 1, (2, 20))
    W3 = rng.normal(0, 1, (2, 15))

    X1 = latent @ W1 + rng.normal(0, 0.5, (n_samples, 30))
    X2 = latent @ W2 + rng.normal(0, 0.5, (n_samples, 20))
    X3 = latent @ W3 + rng.normal(0, 0.5, (n_samples, 15))

    y = latent1.ravel() + 0.3 * latent2.ravel() + rng.normal(0, 0.3, n_samples)
    return [X1, X2, X3], y


@pytest.fixture(scope="session")
def synthetic_blocks_named():
    """Synthetic blocks with feature names."""
    np.random.seed(42)
    n_samples = 100
    latent = np.random.randn(n_samples, 2)
    X1 = latent @ np.random.randn(2, 10) + np.random.randn(n_samples, 10) * 0.5
    X2 = latent @ np.random.randn(2, 8) + np.random.randn(n_samples, 8) * 0.5
    X3 = latent @ np.random.randn(2, 6) + np.random.randn(n_samples, 6) * 0.5
    y = latent[:, 0] + 0.3 * latent[:, 1] + np.random.randn(n_samples) * 0.3
    names1 = [f"Met_{i}" for i in range(10)]
    names2 = [f"Gene_{i}" for i in range(8)]
    names3 = [f"Prot_{i}" for i in range(6)]
    return [X1, X2, X3], [names1, names2, names3], y


@pytest.fixture
def small_blocks(rng):
    """Small three-block data for fast tests."""
    X1 = rng.normal(0, 1, (50, 10))
    X2 = rng.normal(0, 1, (50, 8))
    X3 = rng.normal(0, 1, (50, 6))
    y = rng.normal(0, 1, 50)
    return [X1, X2, X3], y


@pytest.fixture
def mbpls_fitted(synthetic_blocks):
    """A pre-fitted MultiBlockPLS model."""
    from chemocalib.models.mbpls import MultiBlockPLS

    blocks, y = synthetic_blocks
    model = MultiBlockPLS(n_components=3, scale=True)
    model.fit(blocks, y)
    return model, blocks, y


@pytest.fixture
def diablo_fitted(synthetic_blocks):
    """A pre-fitted MultiBlockAligner (DIABLO)."""
    from chemocalib.models.diablo_like import MultiBlockAligner

    blocks, y = synthetic_blocks
    model = MultiBlockAligner(n_components=2, lambda_reg=0.1)
    model.fit(blocks, y)
    return model, blocks, y
