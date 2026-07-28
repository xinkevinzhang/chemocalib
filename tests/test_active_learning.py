"""Tests for active learning and DoE."""

import numpy as np
import pandas as pd


class TestUncertaintySampler:
    """Uncertainty sampling tests."""

    def test_import(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="hybrid")
        assert sampler.strategy == "hybrid"

    def test_select_samples_hybrid(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="hybrid")
        residuals = [np.random.randn(50, 10), np.random.randn(50, 8)]
        indices, scores = sampler.select_samples(residuals, n_select=5)
        assert len(indices) <= 5
        assert len(indices) == len(scores)

    def test_select_samples_residual(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="residual")
        residuals = [np.random.randn(30, 5), np.random.randn(30, 5)]
        indices, scores = sampler.select_samples(residuals, n_select=3)
        assert len(indices) <= 3

    def test_select_samples_entropy(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="entropy")
        residuals = [np.random.randn(40, 8)]
        indices, scores = sampler.select_samples(residuals, n_select=4)
        assert len(indices) <= 4

    def test_select_samples_diversity(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="diversity")
        residuals = [np.random.randn(30, 6)]
        indices, scores = sampler.select_samples(residuals, n_select=5)
        assert len(indices) <= 5

    def test_select_double_knockout(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler(strategy="hybrid")
        pairs = [(f"gene_{i}", f"gene_{j}") for i in range(5) for j in range(i + 1, 5)]
        residuals = [np.random.randn(50, 10)]
        candidates = sampler.select_double_knockout_candidates(
            all_gene_pairs=pairs,
            pair_features=np.random.randn(len(pairs), 5),
            residuals=residuals,
            n_select=3,
            n_pool=8,
        )
        assert isinstance(candidates, pd.DataFrame)

    def test_pool_exceeds_population(self):
        from chemocalib.active_learning.uncertainty import UncertaintySampler

        sampler = UncertaintySampler()
        residuals = [np.random.randn(10, 3)]
        indices, scores = sampler.select_samples(residuals, n_select=5)
        assert len(indices) <= 5


class TestExperimentDesigner:
    """Design of Experiments tests."""

    def test_import(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=3)
        assert designer.n_factors == 3

    def test_full_factorial(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=3, design_type="factorial")
        design = designer.generate_design()
        assert design.ndim == 2
        assert design.shape[1] == 3

    def test_lhs(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=4)
        design = designer.latin_hypercube(n_samples=10)
        assert design.shape == (10, 4)
        assert np.all(design >= 0) and np.all(design <= 1)

    def test_ccd(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=2, design_type="ccd")
        design = designer.generate_design(alpha=1.4)
        assert design.ndim == 2
        assert design.shape[1] == 2

    def test_bbd(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=3, design_type="factorial")
        design = designer.generate_design()
        assert design.ndim == 2
        assert design.shape[1] == 3

    def test_to_dataframe(self):
        from chemocalib.active_learning.doe import ExperimentDesigner

        designer = ExperimentDesigner(n_factors=3, design_type="factorial")
        design = designer.generate_design()
        df = designer.to_dataframe(design, factor_names=["A", "B", "C"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 3
