"""
Statistical inference module for ChemoCalib.

Provides:
  - Permutation tests for MB-PLS significance
  - Bootstrap confidence intervals for latent-variable-backed constraints
  - Multiple testing corrections
"""

from .permutation import permutation_test, bootstrap_ci

__all__ = ["permutation_test", "bootstrap_ci"]
