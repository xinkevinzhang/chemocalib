"""Tests for GEM (Genome-Scale Metabolic Model) modules."""

import numpy as np
import pytest


class TestLatentToConstraint:
    """Latent-to-constraint mapping."""

    def test_import(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint()
        assert mapper.scaling_mode == "soft"

    def test_build_feature_reaction_map(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint()
        feature_names = [f"met_{i}" for i in range(5)]
        reaction_ids = ["EX_A", "EX_B", "EX_C", "EX_D", "EX_E"]
        d = mapper.build_feature_reaction_map(feature_names, reaction_ids)
        assert len(d) == 5

    def test_feature_map_return(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint()
        result = mapper.build_feature_reaction_map(["a", "b"], ["r1", "r2"])
        from typing import Dict
        assert isinstance(result, Dict)

    def test_latent_to_bounds_soft(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint(scaling_mode="soft")
        mapper.build_feature_reaction_map(
            [f"m{i}" for i in range(5)],
            [f"R{i}" for i in range(5)],
        )
        latent = np.array([1.0, -0.5, 2.0, 0.1, -1.5])
        bounds = mapper.latent_to_bounds(latent, n_components=3)
        for rxn_id, (lb, ub) in bounds.items():
            assert lb <= ub, f"Invalid bounds for {rxn_id}: lb={lb} > ub={ub}"

    def test_latent_to_bounds_hard(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint(scaling_mode="hard")
        mapper.build_feature_reaction_map(
            [f"m{i}" for i in range(5)],
            [f"R{i}" for i in range(5)],
        )
        latent = np.array([1.0, -0.5, 2.0, 0.1, -1.5])
        bounds = mapper.latent_to_bounds(latent, n_components=3)
        for rxn_id, (lb, ub) in bounds.items():
            assert lb <= ub

    def test_latent_to_bounds_adaptive(self):
        from chemocalib.gem.constraints import LatentToConstraint
        mapper = LatentToConstraint(scaling_mode="adaptive")
        mapper.build_feature_reaction_map(
            [f"m{i}" for i in range(5)],
            [f"R{i}" for i in range(5)],
        )
        latent = np.array([1.0, -0.5, 2.0, 0.1, -1.5])
        bounds = mapper.latent_to_bounds(latent, n_components=3)
        for rxn_id, (lb, ub) in bounds.items():
            assert lb <= ub

    def test_zero_latent(self):
        from chemocalib.gem.constraints import LatentToConstraint
        for mode in ["soft", "hard", "adaptive"]:
            mapper = LatentToConstraint(scaling_mode=mode)
            mapper.build_feature_reaction_map(
                ["m1", "m2"], ["r1", "r2"]
            )
            bounds = mapper.latent_to_bounds(np.zeros(2))
            for rxn_id, (lb, ub) in bounds.items():
                assert lb <= ub


# FBA tests: require COBRApy with working GLPK solver
try:
    from chemocalib.gem.fba import FBASimulator
    sim_check = FBASimulator(model_name="textbook")
    sim_check.load_model()
    sim_check.wild_type_fba()
    FBA_AVAILABLE = True
except Exception:
    FBA_AVAILABLE = False


@pytest.mark.skipif(not FBA_AVAILABLE, reason="COBRApy FBA not available")
class TestFBASimulator:
    """FBA (Flux Balance Analysis) tests."""

    def test_wild_type_fba_feasible(self):
        from chemocalib.gem.fba import FBASimulator
        sim = FBASimulator(model_name="textbook")
        sim.load_model()
        result = sim.wild_type_fba()
        assert result["status"] == "optimal"
        assert result["objective_value"] > 0

    def test_get_exchange_reactions(self):
        from chemocalib.gem.fba import FBASimulator
        sim = FBASimulator(model_name="textbook")
        sim.load_model()
        exchanges = sim.get_exchange_reactions()
        assert len(exchanges) > 0
        assert all(r.startswith("EX_") for r in exchanges)

    def test_get_all_genes(self):
        from chemocalib.gem.fba import FBASimulator
        sim = FBASimulator(model_name="textbook")
        sim.load_model()
        genes = sim.get_all_genes()
        assert len(genes) > 1

    def test_fba_with_constraints(self):
        from chemocalib.gem.fba import FBASimulator
        from chemocalib.gem.constraints import LatentToConstraint
        sim = FBASimulator(model_name="textbook")
        sim.load_model()
        exchange_ids = sim.get_exchange_reactions()
        clean_ids = [r for r in exchange_ids]
        mapper = LatentToConstraint(scaling_mode="soft")
        names = [f"feature_{i}" for i in range(len(clean_ids))]
        mapper.build_feature_reaction_map(names, clean_ids)
        latent = np.random.randn(5) * 0.5
        bounds = mapper.latent_to_bounds(latent, n_components=3)
        result = sim.fba_with_chemometric_constraints(bounds)
        assert isinstance(result, dict)
        assert "objective_value" in result
        assert result["status"] == "optimal"


class TestFVAAnalyzer:
    """Flux Variability Analysis tests."""

    def test_fva_import(self):
        from chemocalib.gem.fva import FVAAnalyzer
        assert FVAAnalyzer is not None
