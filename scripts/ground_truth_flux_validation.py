#!/usr/bin/env python
"""
Ground-truth flux validation against 13C-MFA reference data.
=============================================================
Compares ChemoCalib predicted fluxes against published 13C-MFA
flux measurements from:

  1. Ishii et al. (2007) Keio fluxome: 8 carbon source conditions
  2. Holm et al. (2010): 3 conditions (aerobic/anaerobic glucose, aerobic acetate)
  3. S. cerevisiae branching ratios from literature consensus

Metrics: RMSE, Spearman rank correlation, per-pathway error
Baselines: E-Flux, MADE, GECKO

Author: Zhang Xin, Department of Chemistry, Capital Normal University
Email: xinzhang@cnu.edu.cn
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")

from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.gem.constraints import LatentToConstraint
from chemocalib.gem.fba import FBASimulator
from chemocalib.data.loader import generate_realistic_e_coli_data
from chemocalib.data.fluxome import (
    load_ecoli_combined_fluxome,
    load_yeast_branching_ratios,
    REACTION_NAMES, N_REACTIONS, subset_fluxes,
)


# ──────────────────────────────────────────────────────────────────────
# Flux prediction interface
# ──────────────────────────────────────────────────────────────────────

def predict_fluxes_chemocalib(
    blocks, growth, model_name="textbook",
    n_components=3, seed=42
) -> np.ndarray:
    """Predict fluxes using the ChemoCalib MB-PLS + FBA pipeline.

    Returns:
        predicted_fluxes : np.ndarray (n_conditions, n_reactions)
    """
    mbpls = MultiBlockPLS(n_components=min(n_components, min(b.shape[1] for b in blocks)))
    mbpls.fit(blocks, growth)

    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    sim.wild_type_fba()

    exchanges = sim.get_exchange_reactions()
    clean_ids = [r.replace("EX_", "") for r in exchanges]

    mapper = LatentToConstraint(scaling_mode="soft")
    names = [f"Met_{i}" for i in range(min(len(clean_ids), 31))]
    mapper.build_feature_reaction_map(names, clean_ids[:len(names)])

    super_scores = mbpls.super_scores
    n_cond = super_scores.shape[0]
    n_rxns = len(sim.model.reactions)
    n_total = min(n_rxns, 31)

    flux_pred = np.zeros((n_cond, n_total))
    for i in range(n_cond):
        bounds = mapper.latent_to_bounds(super_scores[i], n_components=3)
        result = sim.fba_with_chemometric_constraints(bounds)
        if result["status"] == "optimal":
            for j, rxn in enumerate(list(sim.model.reactions)[:n_total]):
                flux_pred[i, j] = result["key_fluxes"].get(rxn.id, 0.0)

    return flux_pred


def predict_fluxes_eflux(blocks, growth, model_name="textbook") -> np.ndarray:
    """E-Flux baseline."""
    expression = blocks[1] if len(blocks) > 1 else blocks[0]
    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    sim.wild_type_fba()

    n_rxns = min(len(sim.model.reactions), 31)
    wt_expr = expression.mean(axis=0) + 1e-6

    flux_pred = np.zeros((expression.shape[0], n_rxns))
    for i in range(expression.shape[0]):
        fc = np.clip(expression[i] / wt_expr, 0.01, 100)
        eflux_bounds = {}
        for j, rxn in enumerate(sim.model.reactions):
            if j < len(fc):
                if fc[j] > 1:
                    eflux_bounds[rxn.id] = (-1000, 1000 * min(fc[j], 10))
                else:
                    eflux_bounds[rxn.id] = (-1000 * min(fc[j], 10), 1000)
        result = sim.fba_with_chemometric_constraints(eflux_bounds)
        if result["status"] == "optimal":
            for j, rxn in enumerate(list(sim.model.reactions)[:n_rxns]):
                flux_pred[i, j] = result["key_fluxes"].get(rxn.id, 0.0)

    return flux_pred


def predict_fluxes_made(blocks, growth, model_name="textbook") -> np.ndarray:
    """MADE baseline -- expression-weighted FBA with soft constraints."""
    expression = blocks[1] if len(blocks) > 1 else blocks[0]
    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    sim.wild_type_fba()

    n_rxns = min(len(sim.model.reactions), 31)
    wt_expr = expression.mean(axis=0) + 1e-6

    flux_pred = np.zeros((expression.shape[0], n_rxns))
    for i in range(expression.shape[0]):
        fc = np.clip(expression[i] / wt_expr, 0.01, 5)
        bounds = {}
        for j, rxn in enumerate(sim.model.reactions):
            if j < len(fc):
                ub = 1000 * fc[j]
                bounds[rxn.id] = (-ub, ub)
        result = sim.fba_with_chemometric_constraints(bounds)
        if result["status"] == "optimal":
            for j, rxn in enumerate(list(sim.model.reactions)[:n_rxns]):
                flux_pred[i, j] = result["key_fluxes"].get(rxn.id, 0.0)

    return flux_pred


def predict_fluxes_gecko(blocks, growth, model_name="textbook") -> np.ndarray:
    """GECKO-style baseline -- proteomics-weighted FBA constraints."""
    proteome = blocks[2] if len(blocks) > 2 else blocks[0]
    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    sim.wild_type_fba()

    n_rxns = min(len(sim.model.reactions), 31)
    wt_prot = proteome.mean(axis=0) + 1e-6

    flux_pred = np.zeros((proteome.shape[0], n_rxns))
    for i in range(proteome.shape[0]):
        fc = np.clip(proteome[i] / wt_prot, 0.01, 5)
        bounds = {}
        for j, rxn in enumerate(sim.model.reactions):
            if j < len(fc):
                ub = 1000 * fc[j]
                bounds[rxn.id] = (-ub, ub)
        result = sim.fba_with_chemometric_constraints(bounds)
        if result["status"] == "optimal":
            for j, rxn in enumerate(list(sim.model.reactions)[:n_rxns]):
                flux_pred[i, j] = result["key_fluxes"].get(rxn.id, 0.0)

    return flux_pred


# ──────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_flux_metrics(
    predicted: np.ndarray,
    measured: np.ndarray,
    eps: float = 1e-8,
) -> Dict:
    """Compute RMSE and Spearman rank correlation for flux predictions.

    Parameters
    ----------
    predicted : (n_cond, n_rxn)
    measured  : (n_cond, n_rxn)

    Returns
    -------
    metrics : dict with RMSE, Spearman r, per-pathway breakdown
    """
    # Flatten for overall metrics
    pred_flat = predicted.ravel()
    meas_flat = measured.ravel()

    # Remove near-zero entries from both
    mask = (pred_flat > eps) | (meas_flat > eps)
    pred_f = pred_flat[mask]
    meas_f = meas_flat[mask]

    rmse = np.sqrt(np.mean((pred_f - meas_f) ** 2))
    mae = np.mean(np.abs(pred_f - meas_f))

    # Normalized RMSE (by measured range)
    m_range = np.max(meas_f) - np.min(meas_f)
    nrmse = rmse / (m_range + eps)

    # Spearman rank correlation
    sp_rho, sp_p = stats.spearmanr(pred_f, meas_f)

    # Pearson correlation
    pr_r, pr_p = stats.pearsonr(pred_f, meas_f)

    # Per-reaction Spearman
    per_rxn_rho = []
    for j in range(measured.shape[1]):
        pj = predicted[:, j][predicted[:, j] > eps]
        mj = measured[:, j][measured[:, j] > eps]
        if len(pj) >= 3 and len(set(pj)) > 1:
            rho_j, _ = stats.spearmanr(pj[:len(mj)], mj[:len(pj)])
            per_rxn_rho.append(rho_j)

    mean_rxn_rho = np.mean(per_rxn_rho) if per_rxn_rho else np.nan

    # Per-pathway breakdown (cap at n_rxns)
    n_rxn = measured.shape[1]
    glycolysis_idx = [i for i in range(0, 9) if i < n_rxn]        # PGI .. PYK
    ppp_idx = [i for i in range(9, 17) if i < n_rxn]              # G6PDH .. TALA
    tca_idx = [i for i in range(17, 25) if i < n_rxn]             # CS .. MDH
    anaplerotic_idx = [i for i in range(25, 31) if i < n_rxn]     # PPC .. ADH

    pathways = {}
    if glycolysis_idx: pathways["Glycolysis"] = glycolysis_idx
    if ppp_idx: pathways["PPP"] = ppp_idx
    if tca_idx: pathways["TCA"] = tca_idx
    if anaplerotic_idx: pathways["Anaplerotic/Ferm."] = anaplerotic_idx

    pathway_metrics = {}
    for pname, idxs in pathways.items():
        pm = predicted[:, idxs].ravel()
        mm = measured[:, idxs].ravel()
        mask_p = (pm > eps) | (mm > eps)
        if mask_p.sum() >= 3:
            rho_p, _ = stats.spearmanr(pm[mask_p], mm[mask_p])
            rmse_p = np.sqrt(np.mean((pm[mask_p] - mm[mask_p]) ** 2))
            pathway_metrics[pname] = {"rho": rho_p, "rmse": rmse_p}

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "spearman_rho": sp_rho,
        "spearman_p": sp_p,
        "pearson_r": pr_r,
        "pearson_p": pr_p,
        "mean_per_rxn_spearman": mean_rxn_rho,
        "n_rxns_evaluated": len(per_rxn_rho),
        "pathway": pathway_metrics,
    }


# ──────────────────────────────────────────────────────────────────────
# Yeast branching ratio validation
# ──────────────────────────────────────────────────────────────────────

def compute_yeast_branching_validation(
    blocks, growth, model_name="textbook", seed=42
) -> Dict:
    """Validate predicted flux distributions against yeast branching ratio consensus.

    Since the yeast stress dataset and 13C-MFA branching data are not from
    the same experiment, this is a distribution-level validation rather
    than a per-condition match.

    Parameters
    ----------
    blocks : list of np.ndarray
        Multi-omics blocks from yeast data generator.
    growth : np.ndarray
        Growth rates.
    model_name : str
        COBRApy model name.
    seed : int

    Returns
    -------
    report : dict
    """
    yeast_ref = load_yeast_branching_ratios()

    # Predict fluxes using ChemoCalib
    flux_pred = predict_fluxes_chemocalib(blocks, growth, model_name, seed=seed)
    flux_eflux = predict_fluxes_eflux(blocks, growth, model_name)

    # Compute predicted branching ratios from flux distributions
    # Map branch points to reaction groups
    branch_reactions = {
        "G6P => PPP vs Glycolysis": ("G6PDH", "PFK"),
        "PEP => Pyruvate vs OAA (anaplerotic)": ("PYK", "PPC"),
    }

    branch_results = []
    for bp_name, (rxn_a, rxn_b) in branch_reactions.items():
        try:
            ia = REACTION_NAMES.index(rxn_a)
            ib = REACTION_NAMES.index(rxn_b)
        except ValueError:
            continue

        # Predicted ratio: flux_A / (flux_A + flux_B)
        for method_name, flux_mat in [("ChemoCalib", flux_pred), ("E-Flux", flux_eflux)]:
            if ia < flux_mat.shape[1] and ib < flux_mat.shape[1]:
                fa = flux_mat[:, ia]
                fb = flux_mat[:, ib]
                ratio_pred = np.mean(fa / (fa + fb + 1e-8))
                branch_results.append({
                    "branch_point": bp_name,
                    "rxn_a": rxn_a,
                    "rxn_b": rxn_b,
                    "method": method_name,
                    "predicted_ratio": ratio_pred,
                })

    # Consensus values (mean from literature)
    consensus = {
        "G6P => PPP vs Glycolysis": 0.18,       # 18% to PPP
        "PEP => Pyruvate vs OAA (anaplerotic)": 0.25,  # 25% to OAA
    }

    for r in branch_results:
        r["consensus_ratio"] = consensus.get(r["branch_point"], np.nan)
        r["abs_error"] = abs(r["predicted_ratio"] - r["consensus_ratio"])

    df = pd.DataFrame(branch_results)

    # Summary
    if len(df) > 0:
        chemo_errors = df[df["method"] == "ChemoCalib"]["abs_error"]
        eflux_errors = df[df["method"] == "E-Flux"]["abs_error"]
        mean_chemo_error = chemo_errors.mean() if len(chemo_errors) > 0 else np.nan
        mean_eflux_error = eflux_errors.mean() if len(eflux_errors) > 0 else np.nan
    else:
        mean_chemo_error = np.nan
        mean_eflux_error = np.nan

    return {
        "branch_results": df,
        "mean_abs_error_chemocalib": mean_chemo_error,
        "mean_abs_error_eflux": mean_eflux_error,
        "reference": yeast_ref["description"],
        "ref_ratios": yeast_ref["ratios"],
        "ref_ratios_std": yeast_ref["ratios_std"],
        "note": (
            "Distribution-level validation: yeast stress omics dataset "
            "and 13C-MFA branching ratios are from different experiments, "
            "not paired per-condition. Only central carbon branch-point "
            "agreement is assessed."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────

def run_ecoli_ground_truth_benchmark(
    model_name="textbook",
    n_conditions=8,
    seed=42,
) -> pd.DataFrame:
    """Run ChemoCalib vs baselines vs 13C-MFA (E. coli combined dataset).

    Compares flux predictions against the Ishii (2007) + Holm (2010)
    11-condition 13C-MFA reference dataset.

    Parameters
    ----------
    model_name : str
    n_conditions : int
    seed : int

    Returns
    -------
    df_results : pd.DataFrame
    """
    # Generate realistic omics data (carbon source conditions)
    blocks, growth, meta = generate_realistic_e_coli_data(
        n_conditions=n_conditions, seed=seed
    )

    # Load 13C-MFA reference
    ref = load_ecoli_combined_fluxome()
    ref_flux = ref["flux_matrix"][:n_conditions, :]  # match n_cond

    print(f"\n{'='*60}")
    print(f"Ground-truth Flux Validation: E. coli (n={n_conditions})")
    print(f"Reference: {ref['references'][0]}")
    print(f"{'='*60}")

    methods = {
        "ChemoCalib": predict_fluxes_chemocalib,
        "E-Flux": predict_fluxes_eflux,
        "MADE": predict_fluxes_made,
        "GECKO": predict_fluxes_gecko,
    }

    results = []
    for method_name, pred_fn in methods.items():
        print(f"\n--- {method_name} ---")
        pred_flux = pred_fn(blocks, growth, model_name=model_name)

        # Align dimensions: take min of predicted and measured columns
        n_common = min(pred_flux.shape[1], ref_flux.shape[1])
        metrics = compute_flux_metrics(
            pred_flux[:, :n_common],
            ref_flux[:n_conditions, :n_common],
        )

        print(f"  RMSE={metrics['rmse']:.2f}, NRMSE={metrics['nrmse']:.3f}")
        print(f"  Spearman rho={metrics['spearman_rho']:.3f} (p={metrics['spearman_p']:.2e})")
        print(f"  Pearson r={metrics['pearson_r']:.3f}")
        print(f"  Per-rxn mean Spearman={metrics['mean_per_rxn_spearman']:.3f}")

        for pname, pm in metrics["pathway"].items():
            print(f"    {pname}: rho={pm['rho']:.3f}, RMSE={pm['rmse']:.1f}")

        results.append({
            "Method": method_name,
            "RMSE": round(metrics["rmse"], 2),
            "NRMSE": round(metrics["nrmse"], 4),
            "MAE": round(metrics["mae"], 2),
            "Spearman_rho": round(metrics["spearman_rho"], 3),
            "Spearman_p": f"{metrics['spearman_p']:.2e}",
            "Pearson_r": round(metrics["pearson_r"], 3),
            "Mean_per_rxn_Spearman": round(metrics["mean_per_rxn_spearman"], 3),
            "N_rxns_evaluated": metrics["n_rxns_evaluated"],
        })

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("Summary: E. coli Ground-truth Flux Validation")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    return df


def run_yeast_branching_validation(model_name="textbook", seed=42) -> Dict:
    """Run yeast branching ratio validation."""
    from chemocalib.data.loader import generate_realistic_yeast_stress_data

    blocks, growth, meta = generate_realistic_yeast_stress_data(
        n_conditions=8, seed=seed
    )

    print(f"\n{'='*60}")
    print("Yeast Branching Ratio Validation")
    print(f"{'='*60}")

    report = compute_yeast_branching_validation(blocks, growth, model_name, seed)

    df_br = report["branch_results"]
    if len(df_br) > 0:
        print(df_br.to_string(index=False))
        print(f"\nChemoCalib mean abs error: {report['mean_abs_error_chemocalib']:.3f}")
        print(f"E-Flux mean abs error:      {report['mean_abs_error_eflux']:.3f}")
    else:
        print("  (No branching ratio matches could be computed)")

    print(f"\nNote: {report['note']}")
    return report


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main(args):
    # E. coli ground truth
    ecoli_df = run_ecoli_ground_truth_benchmark(
        model_name=args.model,
        n_conditions=args.conditions,
        seed=args.seed,
    )

    # Yeast branching
    yeast_report = run_yeast_branching_validation(
        model_name=args.model,
        seed=args.seed + 1,
    )

    # Save
    os.makedirs(args.output, exist_ok=True)
    ecoli_df.to_csv(os.path.join(args.output, "ground_truth_ecoli.csv"), index=False)
    print(f"\nResults saved to: {args.output}/")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ground-truth flux validation against 13C-MFA reference"
    )
    parser.add_argument("--model", default="textbook", help="COBRApy model")
    parser.add_argument("--conditions", type=int, default=8, help="E. coli conditions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()
    sys.exit(main(args))
