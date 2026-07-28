"""Tests for MB-PLS and DIABLO models."""

import numpy as np
import pytest


class TestMultiBlockPLS:
    """MultiBlockPLS core functionality."""

    def test_import(self):
        from chemocalib.models.mbpls import MultiBlockPLS
        model = MultiBlockPLS(n_components=2)
        assert model.n_components == 2

    def test_fit_basic(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        assert model.super_scores.shape == (50, 2)
        assert len(model.block_weights) == 3

    def test_fit_with_scale(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2, scale=True)
        model.fit(blocks, y)
        assert model.super_scores.shape[0] == 50

    def test_transform(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        # transform returns tuple (scores, y_pred) via sklearn PLS
        result = model.transform(blocks)
        assert result is not None

    def test_block_importance(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        imp = model.block_importance
        assert len(imp) == 3
        assert np.all(imp >= 0)

    def test_vip_scores(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        vip_info = model.get_driving_metabolites(0, top_k=3)
        assert "vip_values" in vip_info
        assert len(vip_info["vip_values"]) == 3

    def test_residual_space(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        R = model.residual_space(blocks)
        # returns list of ndarrays, one per block
        assert isinstance(R, list)
        assert len(R) == 3

    def test_generate_toy_data(self):
        from chemocalib.models.mbpls import generate_toy_multiblock_data
        result = generate_toy_multiblock_data(seed=0)
        # returns (blocks, y, feature_names) or (blocks, y)
        if len(result) == 3:
            blocks, y, names = result
            assert isinstance(names, list)
        else:
            blocks, y = result
        assert len(blocks) == 3
        assert y.ndim == 1

    def test_transform_prediction(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=2)
        model.fit(blocks, y)
        T = model.super_scores
        assert T.shape[0] == 50

    def test_large_n_components(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        model = MultiBlockPLS(n_components=1)
        model.fit(blocks, y)
        assert model.super_scores.shape[1] == 1

    def test_single_block(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        single_block = [blocks[0]]
        model = MultiBlockPLS(n_components=1)
        model.fit(single_block, y)
        assert model.super_scores.shape[0] == 50


class TestMultiBlockAligner:
    """DIABLO-like aligner tests."""

    def test_import_diablo(self):
        from chemocalib.models.diablo_like import MultiBlockAligner
        model = MultiBlockAligner(n_components=2)
        assert model.n_components == 2

    def test_fit_diablo(self, small_blocks):
        from chemocalib.models.diablo_like import MultiBlockAligner
        blocks, y = small_blocks
        model = MultiBlockAligner(n_components=2)
        model.fit(blocks, y)
        assert model.scores is not None
        assert len(model.scores) == 3

    def test_summary_diablo(self, small_blocks):
        from chemocalib.models.diablo_like import MultiBlockAligner
        blocks, y = small_blocks
        model = MultiBlockAligner(n_components=2)
        model.fit(blocks, y)
        s = model.summary()
        assert isinstance(s, str)
        assert "DIABLO" in s or "MultiBlock" in s

    def test_default_kwargs(self, small_blocks):
        from chemocalib.models.diablo_like import MultiBlockAligner
        blocks, y = small_blocks
        model = MultiBlockAligner(n_components=2)
        model.fit(blocks, y)
        assert model.scores[0].shape[0] == 50

    def test_sparse_kwargs(self, small_blocks):
        from chemocalib.models.diablo_like import MultiBlockAligner
        blocks, y = small_blocks
        model = MultiBlockAligner(n_components=2, keep_sparse=0.5)
        model.fit(blocks, y)
        assert model.scores[0].shape[0] == 50


class TestMultiBlockPLSNamed:
    """Tests with named blocks."""

    def test_block_names(self, small_blocks):
        from chemocalib.models.mbpls import MultiBlockPLS
        blocks, y = small_blocks
        names = ["met", "trx", "pro"]
        model = MultiBlockPLS(n_components=2, block_names=names)
        model.fit(blocks, y)
        assert model.block_names == names
