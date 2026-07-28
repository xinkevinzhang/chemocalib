"""Tests for virtual experiment modules (knockout + surrogate)."""

import numpy as np


class TestDoubleKnockoutDesigner:
    """Double knockout experiment design tests."""

    def test_import(self):
        from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

        designer = DoubleKnockoutDesigner(gene_pool=["g1", "g2", "g3"])
        assert len(designer.gene_pool) == 3

    def test_generate_pairs(self):
        from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

        genes = [f"gene_{i}" for i in range(10)]
        designer = DoubleKnockoutDesigner(gene_pool=genes)
        pairs = designer.generate_pairs(n_pairs=20)
        assert len(pairs) == 20
        for a, b in pairs:
            assert a in genes and b in genes
            assert a != b

    def test_generate_pairs_over_max(self):
        from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

        genes = ["A", "B", "C"]
        designer = DoubleKnockoutDesigner(gene_pool=genes)
        pairs = designer.generate_pairs(n_pairs=100)
        max_pairs = 3 * 2 // 2
        assert len(pairs) == max_pairs

    def test_generate_single_knockouts(self):
        from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

        genes = [f"g{i}" for i in range(5)]
        designer = DoubleKnockoutDesigner(gene_pool=genes)
        singles = designer.generate_single_knockouts()
        assert len(singles) == 5

    def test_results_attribute(self):
        from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

        genes = [f"g{i}" for i in range(6)]
        designer = DoubleKnockoutDesigner(gene_pool=genes)
        designer.generate_pairs(n_pairs=10)
        # After generating pairs, pairs is set but results is None
        assert designer.pairs is not None
        assert len(designer.pairs) == 10


class TestSurrogateModel:
    """Surrogate model tests."""

    def test_import(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        sm = SurrogateModel()
        assert sm is not None

    def test_fit(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        sm = SurrogateModel()
        sm.fit(X, y)
        # Check it was fitted (beta is set)
        assert sm._fitted or sm.beta is not None

    def test_predict(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        sm = SurrogateModel()
        sm.fit(X, y)
        X_test = np.random.randn(10, 5)
        y_pred = sm.predict(X_test)
        assert len(y_pred) == 10

    def test_evaluate(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        sm = SurrogateModel()
        sm.fit(X[:70], y[:70])
        metrics = sm.evaluate(X[70:], y[70:])
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_predict_with_uncertainty(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(80, 5)
        y = np.random.randn(80)
        sm = SurrogateModel()
        sm.fit(X, y)
        X_test = np.random.randn(10, 5)
        pred_mean, lower, upper = sm.predict_with_uncertainty(X_test, n_bootstrap=30)
        assert len(pred_mean) == 10
        assert len(lower) == 10
        assert len(upper) == 10
        assert np.all(lower <= upper)

    def test_summary(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        sm = SurrogateModel()
        sm.fit(X, y)
        s = sm.summary()
        assert isinstance(s, str)
        assert "SurrogateModel" in s

    def test_fit_returns_self(self):
        from chemocalib.virtual_experiment.surrogate import SurrogateModel

        np.random.seed(42)
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        sm = SurrogateModel()
        result = sm.fit(X, y)
        assert result is sm
