"""Tests for validation (benchmark) and data loader modules."""

import numpy as np
import pytest


class TestEFluxBaseline:
    """E-Flux baseline tests."""

    def test_import(self):
        from chemocalib.validation.benchmark import EFluxBaseline
        eflux = EFluxBaseline()
        assert eflux.pseudocount == 0.01

    def test_compute_bounds(self):
        from chemocalib.validation.benchmark import EFluxBaseline
        eflux = EFluxBaseline()
        expr = np.array([1.0, 2.0, 0.5])
        bounds = eflux.compute_bounds(expr)
        assert len(bounds) == 3

    def test_predict_growth(self):
        from chemocalib.validation.benchmark import EFluxBaseline
        eflux = EFluxBaseline()
        expr = np.array([1.0, 1.5, 0.8])
        wt_expr = np.array([1.0, 1.0, 1.0])
        g = eflux.predict_growth_from_expression(expr, 1.0, wt_expr)
        assert g > 0.0
        assert g < 5.0


class TestMADEApproximation:
    """MADE tests."""

    def test_classify(self):
        from chemocalib.validation.benchmark import MADEApproximation
        made = MADEApproximation(fold_change_threshold=1.0)
        fc = np.array([2.0, -1.5, 0.3, -0.2])
        cls = made.classify_reactions(fc)
        assert len(cls["up"]) == 1
        assert len(cls["down"]) == 1
        assert len(cls["unchanged"]) == 2

    def test_compute_bounds(self):
        from chemocalib.validation.benchmark import MADEApproximation
        made = MADEApproximation()
        fc = np.array([2.0, -2.0, 0.0])
        base = np.ones((3, 2)) * 100
        adjusted = made.compute_bounds(fc, base)
        assert adjusted.shape == (3, 2)
        assert adjusted[0, 1] > base[0, 1]   # up-regulated
        assert adjusted[1, 1] < base[1, 1]   # down-regulated

    def test_predict_growth(self):
        from chemocalib.validation.benchmark import MADEApproximation
        made = MADEApproximation()
        fc = np.array([2.5, 2.0, -1.5])
        g = made.predict_growth(fc, wt_growth=1.0)
        assert g > 0.0


class TestGECKOApproximation:
    """GECKO tests."""

    def test_compute_bounds(self):
        from chemocalib.validation.benchmark import GECKOApproximation
        gecko = GECKOApproximation()
        protein = np.array([0.01, 0.02, 0.005])
        mw = np.array([50000.0, 30000.0, 60000.0])
        kcat = np.array([100.0, 200.0, 50.0])
        vmax = gecko.compute_kcat_constrained_bounds(protein, mw, kcat)
        assert len(vmax) == 3
        assert all(v > 0 for v in vmax)

    def test_predict_growth(self):
        from chemocalib.validation.benchmark import GECKOApproximation
        gecko = GECKOApproximation()
        proteome = np.array([0.01, 0.03, 0.02, 0.04])
        g = gecko.predict_growth(proteome, wt_growth=1.0)
        assert g > 0.0


class TestBenchmarkRunner:
    """Benchmark integration tests."""

    def test_run_all_methods(self, small_blocks):
        from chemocalib.validation.benchmark import BenchmarkRunner
        blocks, y = small_blocks
        runner = BenchmarkRunner(n_repeats=3, seed=42)
        summary = runner.run(blocks, y)
        assert len(summary) == 4  # all methods
        for method in ["chemocalib", "eflux", "made", "gecko"]:
            assert method in summary
            assert "rmse_mean" in summary[method]

    def test_comparison_table(self, small_blocks):
        from chemocalib.validation.benchmark import BenchmarkRunner
        blocks, y = small_blocks
        runner = BenchmarkRunner(n_repeats=3, seed=42)
        runner.run(blocks, y)
        table = runner.comparison_table()
        assert table.shape[0] == 4
        assert "Method" in table.columns

    def test_statistical_tests(self, small_blocks):
        from chemocalib.validation.benchmark import BenchmarkRunner
        blocks, y = small_blocks
        runner = BenchmarkRunner(n_repeats=5, seed=42)
        runner.run(blocks, y)
        tests_df = runner.statistical_tests()
        assert tests_df.shape[0] >= 1
        assert "p_value" in tests_df.columns


class TestDataLoader:
    """Data loader tests."""

    def test_generate_realistic_data(self):
        from chemocalib.data.loader import generate_realistic_e_coli_data
        blocks, growth, meta = generate_realistic_e_coli_data(
            n_conditions=6, seed=0)
        assert len(blocks) == 3
        assert len(growth) == 6
        assert "carbon_sources" in meta
        assert len(meta["carbon_sources"]) == 6

    def test_max_conditions(self):
        # n_conditions is capped at the number of hardcoded carbon sources
        from chemocalib.data.loader import generate_realistic_e_coli_data
        blocks, growth, _ = generate_realistic_e_coli_data(
            n_conditions=6, seed=1)
        assert len(growth) == 6

    def test_reproducible(self):
        from chemocalib.data.loader import generate_realistic_e_coli_data
        _, g1, _ = generate_realistic_e_coli_data(n_conditions=5, seed=42)
        _, g2, _ = generate_realistic_e_coli_data(n_conditions=5, seed=42)
        assert np.allclose(g1, g2)
