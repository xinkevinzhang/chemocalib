"""
Flux Variability Analysis (FVA) integration
============================================
Quantifies the shrinkage of the feasible flux space
when chemometric constraints are applied.

This is a key scientific metric: how much does the MB-PLS-guided
constraint set reduce the solution space compared to baseline FBA?

References
----------
- Mahadevan & Schilling (2003) Metab. Eng.
- Gudmundsson & Thiele (2010) Bioinformatics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class FVAAnalyzer:
    """Analyze flux variability before/after chemometric constraints.

    Parameters
    ----------
    simulator : FBASimulator
        COBRApy-based FBA simulator instance.
    reactions_of_interest : list of str, optional
        Subset of reactions for detailed FVA (default: all exchange reactions).
    """

    def __init__(self, simulator, reactions_of_interest=None):
        self.sim = simulator
        self.model = simulator.model
        self.reactions_of_interest = reactions_of_interest

        if self.reactions_of_interest is None and self.model is not None:
            self.reactions_of_interest = [
                r.id for r in self.model.reactions
                if r.id.startswith("EX_") or r.id.startswith("R")
            ]

    def run_fva(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fraction_of_optimum: float = 0.9,
        loopless: bool = False,
    ) -> Dict:
        """Run FVA with and without chemometric constraints.

        Parameters
        ----------
        bounds : dict, optional
            Chemometric constraints (reaction_id → (lb, ub)).
            If None, runs baseline FVA only.
        fraction_of_optimum : float
            Fraction of optimal objective for FVA constraint.

        Returns
        -------
        report : dict
            Keys: ``reaction_flux_ranges``, ``space_volume_proxy``,
            ``fva_baseline`` (if bounds provided), ``fva_constrained``.
        """
        rxns = self.reactions_of_interest or [
            r.id for r in self.model.reactions[:20]
        ]

        # Baseline FVA
        fva_baseline = self._fva_single(
            rxns, constraints=None, fraction=fraction_of_optimum)

        result = {"fva_baseline": fva_baseline}

        if bounds is not None:
            fva_constrained = self._fva_single(
                rxns, constraints=bounds, fraction=fraction_of_optimum)
            result["fva_constrained"] = fva_constrained

            # Compute flux space shrinkage
            result["space_shrinkage"] = self._compute_schrinkage(
                fva_baseline, fva_constrained)

        return result

    def _fva_single(
        self,
        reactions: List[str],
        constraints: Optional[Dict] = None,
        fraction: float = 0.9,
    ) -> Dict:
        """Run a single FVA analysis."""
        from cobra.flux_analysis import flux_variability_analysis

        with self.model as m:
            # Apply chemometric constraints if provided
            if constraints:
                for rxn_id, (lb, ub) in constraints.items():
                    if rxn_id in m.reactions:
                        rxn = m.reactions.get_by_id(rxn_id)
                        rxn.lower_bound = max(rxn.lower_bound, lb)
                        rxn.upper_bound = min(rxn.upper_bound, ub)

            # Fix biomass at fraction of optimum
            sol = m.optimize()
            if sol.status == "optimal":
                biomass_rxn = m.reactions.get_by_id("BIOMASS")
                biomass_rxn.lower_bound = sol.objective_value * fraction

            try:
                fva_result = flux_variability_analysis(
                    m, reaction_list=[m.reactions.get_by_id(r) for r in reactions
                                      if r in m.reactions],
                    fraction_of_optimum=fraction,
                )
            except Exception:
                fva_result = None

        return self._parse_fva(fva_result, reactions)

    def _parse_fva(self, fva_df, reactions: List[str]) -> Dict:
        """Parse COBRApy FVA DataFrame to structured dict."""
        if fva_df is None or fva_df.empty:
            return {}

        parsed = {}
        for rxn in reactions:
            if rxn in fva_df.index:
                row = fva_df.loc[rxn]
                parsed[rxn] = {
                    "minimum": float(row["minimum"]),
                    "maximum": float(row["maximum"]),
                    "range": float(row["maximum"] - row["minimum"]),
                }
        return parsed

    def _compute_schrinkage(self, baseline: Dict, constrained: Dict) -> Dict:
        """Compute flux space shrinkage metrics."""
        common_rxns = set(baseline.keys()) & set(constrained.keys())
        if not common_rxns:
            return {"n_common_reactions": 0}

        range_ratios = []
        for rxn in common_rxns:
            base_range = baseline[rxn]["range"]
            const_range = constrained[rxn]["range"]
            if base_range > 1e-10:
                range_ratios.append(const_range / base_range)

        if not range_ratios:
            return {"n_common_reactions": len(common_rxns)}

        return {
            "n_common_reactions": len(common_rxns),
            "mean_range_ratio": float(np.mean(range_ratios)),
            "median_range_ratio": float(np.median(range_ratios)),
            "min_range_ratio": float(np.min(range_ratios)),
            "max_range_ratio": float(np.max(range_ratios)),
            "n_fully_constrained": int(np.sum(np.array(range_ratios) < 0.01)),
            "space_shrinkage_percent": float(
                (1 - np.mean(range_ratios)) * 100),
        }

    def gene_essentiality_validation(
        self,
        gene_list: List[str],
        known_essential: List[str],
        known_nonessential: List[str],
        growth_threshold: float = 0.01,
    ) -> Dict:
        """Validate gene essentiality predictions against known labels.

        Parameters
        ----------
        gene_list : list of str
            All genes to test.
        known_essential : list of str
            Experimentally verified essential genes.
        known_nonessential : list of str
            Experimentally verified non-essential genes.
        growth_threshold : float
            Growth rate below which a gene is predicted essential.

        Returns
        -------
        metrics : dict
            TPR, FPR, accuracy, AUC, confusion matrix.
        """
        essential_pred = []
        true_labels = []
        growth_rates = []

        for gene in gene_list:
            if gene in self.model.genes:
                with self.model as m:
                    g = m.genes.get_by_id(gene)
                    original_bounds = (g.reactions[0].lower_bound,
                                     g.reactions[0].upper_bound) if g.reactions else (-1000, 1000)
                    # Simulate knockout
                    for rxn in g.reactions:
                        rxn.lower_bound = 0
                        rxn.upper_bound = 0
                    try:
                        sol = m.optimize()
                        gr = sol.objective_value if sol.status == "optimal" else 0.0
                    except Exception:
                        gr = 0.0
                    growth_rates.append(float(gr))

                is_essential = float(gr < growth_threshold)
                essential_pred.append(is_essential)

                if gene in known_essential:
                    true_labels.append(1)
                elif gene in known_nonessential:
                    true_labels.append(0)
                else:
                    continue

        from sklearn.metrics import (
            roc_auc_score, confusion_matrix, accuracy_score)

        y_true = np.array(true_labels)
        y_pred = np.array(essential_pred[:len(true_labels)])

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = accuracy_score(y_true, y_pred)

        return {
            "TPR": float(tpr),
            "FPR": float(fpr),
            "accuracy": float(acc),
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "n_tested": int(len(true_labels)),
            "n_genes_total": int(len(gene_list)),
        }
