"""Tests for statistics and cross-validation modules."""

import numpy as np


class TestPermutation:
    """Permutation test functionality."""

    def test_permutation_import(self):
        from chemocalib.stats.permutation import permutation_test

        assert callable(permutation_test)

    def test_permutation_basic(self, small_blocks):
        from chemocalib.stats.permutation import permutation_test
        from chemocalib.models.mbpls import MultiBlockPLS

        blocks, y = small_blocks
        factory = lambda: MultiBlockPLS(n_components=2)
        result = permutation_test(
            model_factory=factory, blocks=blocks, y=y, n_permutations=30, seed=42
        )
        assert "p_value" in result
        assert "observed" in result
        assert 0 <= result["p_value"] <= 1

    def test_permutation_null(self, small_blocks):
        from chemocalib.stats.permutation import permutation_test
        from chemocalib.models.mbpls import MultiBlockPLS

        blocks, y = small_blocks
        factory = lambda: MultiBlockPLS(n_components=2)
        result = permutation_test(
            model_factory=factory, blocks=blocks, y=y, n_permutations=20, seed=99
        )
        assert "null_distribution" in result
        assert len(result["null_distribution"]) == 20

    def test_bootstrap_ci(self, small_blocks):
        from chemocalib.stats.permutation import bootstrap_ci
        from chemocalib.models.mbpls import MultiBlockPLS

        blocks, y = small_blocks
        factory = lambda: MultiBlockPLS(n_components=1)
        result = bootstrap_ci(
            model_factory=factory,
            blocks=blocks,
            y=y,
            n_bootstrap=20,
            alpha=0.05,
            seed=42,
        )
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["ci_upper"]


class TestCrossValidation:
    """Cross-validation and stability selection."""

    def test_grid_search_import(self):
        from chemocalib.cross_validation import grid_search_components

        assert callable(grid_search_components)

    def test_grid_search(self, small_blocks):
        from chemocalib.cross_validation import grid_search_components
        from chemocalib.models.mbpls import MultiBlockPLS

        blocks, y = small_blocks
        result = grid_search_components(
            blocks=blocks,
            y=y,
            model_cls=MultiBlockPLS,
            component_range=[1, 2, 3],
            n_folds=3,
            seed=42,
        )
        assert "best_n_components" in result
        assert "cv_results" in result
        assert "best_score" in result
        assert 1 <= result["best_n_components"] <= 3

    def test_stability_selection(self, small_blocks):
        from chemocalib.cross_validation import stability_selection
        from chemocalib.models.mbpls import MultiBlockPLS

        blocks, y = small_blocks
        result = stability_selection(
            blocks=blocks,
            y=y,
            model_cls=MultiBlockPLS,
            n_components=2,
            n_subsamples=20,
            subsample_frac=0.7,
            seed=42,
        )
        assert isinstance(result, dict)
        assert "block_selection_prob" in result
        assert isinstance(result["block_selection_prob"], list)
        assert len(result["block_selection_prob"]) == len(blocks)
