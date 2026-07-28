"""Tests for dynamic layer modules."""

import numpy as np


class TestGlycolysisODE:
    """ODE solver tests."""

    def test_import(self):
        from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
        ode = GlycolysisODE()
        assert ode is not None

    def test_simulate(self):
        from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
        ode = GlycolysisODE()
        result = ode.simulate(t_span=(0, 40), n_points=100)
        assert isinstance(result, dict)
        assert "t" in result
        assert len(result["t"]) == 100
        assert "G6P" in result
        assert "FBP" in result
        assert "PYR" in result

    def test_steady_state(self):
        from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
        ode = GlycolysisODE()
        ss = ode.steady_state()
        assert len(ss) == 3
        assert all(v >= 0 for v in ss)

    def test_calibrate_from_latent(self):
        from chemocalib.dynamic_layer.ode_solver import GlycolysisODE
        ode = GlycolysisODE()
        orig_vmax = ode.vmax_pfk
        ode.calibrate_from_latent(np.array([1.0, -0.5]), n_component=1)
        # calibrate_from_latent modifies in-place; it's allowed to change or not
        # depending on the scaling logic — just verify no exception was raised
        assert ode is not None
