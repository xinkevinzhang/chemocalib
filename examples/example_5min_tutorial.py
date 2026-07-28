#!/usr/bin/env python
"""ChemoCalib 5-Minute Tutorial
====================================================================
Demonstrates the complete pipeline on synthetic multi-omics data:
  MB-PLS decomposition -> GPR-VIP feature selection -> constrained FBA.

Estimated runtime: ~2 minutes (no solver required for synthetic data).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add package root to path if running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.data.fluxome import generate_synthetic_flux_data
from chemocalib.data.loader import load_example_expression

print("=" * 70)
print("ChemoCalib 5-Minute Tutorial")
print("=" * 70)

# -----------------------------------------------------------------------
# Step 1: Generate synthetic multi-omics data
# -----------------------------------------------------------------------
print("\n[1/5] Generating synthetic multi-omics data...")

# 11 conditions (Ishii 8 + Holm 3), 500 genes, 30 model reactions
n_conditions = 11
n_genes = 500
n_reactions = 30
rng = np.random.RandomState(42)

# Transcriptome: conditions x genes
transcriptome = rng.lognormal(mean=0.0, sigma=0.6, size=(n_conditions, n_genes))
transcriptome = pd.DataFrame(
    transcriptome,
    index=[f"cond_{i+1}" for i in range(n_conditions)],
    columns=[f"gene_{j+1}" for j in range(n_genes)],
)

# Metabolome: conditions x metabolites (optional second block)
metabolome = rng.lognormal(mean=-0.5, sigma=0.4, size=(n_conditions, 80))
metabolome = pd.DataFrame(
    metabolome,
    index=transcriptome.index,
    columns=[f"met_{k+1}" for k in range(80)],
)

# GPR rules: reaction -> list of gene IDs
gpr_rules = {}
for r in range(n_reactions):
    n_genes_in_rule = rng.randint(1, 4)
    gene_ids = [
        f"gene_{j+1}" for j in rng.choice(n_genes, n_genes_in_rule, replace=False)
    ]
    gpr_rules[f"R{r+1:03d}"] = gene_ids

# True fluxes (ground truth from 13C-MFA)
# Simulate per-condition fluxes around realistic central-carbon values
base_fluxes = np.array([100.0, 80.0, 15.0, 12.0, 3.0] * 6)  # 5 pathways x 6 rxns = 30
base_fluxes = base_fluxes + rng.randn(n_reactions) * 5.0
true_flux_df = pd.DataFrame(index=transcriptome.index, columns=gpr_rules.keys())
for i in range(n_conditions):
    true_flux_df.iloc[i] = base_fluxes + rng.randn(n_reactions) * 3.0

print(f"  Transcriptome: {transcriptome.shape}")
print(f"  Metabolome:    {metabolome.shape}")
print(f"  GPR rules:     {len(gpr_rules)} reactions")
print(f"  True fluxes:   {true_flux_df.shape} ({n_conditions} conditions)")

# -----------------------------------------------------------------------
# Step 2: MB-PLS Decomposition (Core Algorithm)
# -----------------------------------------------------------------------
print("\n[2/5] Running Multi-Block PLS decomposition...")

X_blocks = [transcriptome.values, metabolome.values]
Y = np.ones((n_conditions, 1))  # placeholder Y for unsupervised MB-PLS
K = 3  # number of latent components

mbpls = MultiBlockPLS(n_components=K, scale=True)
mbpls.fit(X_blocks, Y)

print(f"  Components extracted: K = {K}")
print(f"  Block 0 (transcriptome) weights shape: {mbpls.W_[0].shape}")
print(f"  Block 1 (metabolome) weights shape:    {mbpls.W_[1].shape}")
print(f"  Cumulative variance explained: {mbpls.cumulative_variance_[0].sum():.1%}")

# -----------------------------------------------------------------------
# Step 3: GPR-VIP Feature Selection
# -----------------------------------------------------------------------
print("\n[3/5] Computing GPR-VIP scores...")

from chemocalib.gem.gpr_vip import compute_gpr_vip

# Compute VIP for transcriptome block
vip_scores = compute_gpr_vip(mbpls, block_idx=0, component_weights=None)
vip_df = pd.Series(vip_scores, index=transcriptome.columns)

# Aggregate VIP by GPR rule
reaction_vip = {}
for rxn_id, gene_list in gpr_rules.items():
    gene_vips = [vip_df.get(g, 0.0) for g in gene_list]
    reaction_vip[rxn_id] = np.sqrt(
        np.sum(np.array(gene_vips) ** 2)
    )  # Euclidean aggregation

reaction_vip = pd.Series(reaction_vip).sort_values(ascending=False)

print(f"  Top 5 reactions by GPR-VIP:")
for rxn, score in reaction_vip.head(5).items():
    print(f"    {rxn}: VIP = {score:.3f} (genes: {', '.join(gpr_rules[rxn][:3])}...)")

# Select top VIP reactions
vip_threshold = 1.0
selected_reactions = reaction_vip[reaction_vip > vip_threshold].index.tolist()
print(
    f"  Selected {len(selected_reactions)}/{len(reaction_vip)} reactions (VIP > {vip_threshold})"
)

# -----------------------------------------------------------------------
# Step 4: Constrained FBA (simulated)
# -----------------------------------------------------------------------
print("\n[4/5] Running constrained FBA with GPR-VIP bounds...")

from chemocalib.gem.constrained_fba import apply_constraints, predict_flux_distribution

# Build reaction bounds from VIP scores
reaction_bounds = {}
for rxn_id in gpr_rules:
    score = reaction_vip.get(rxn_id, 0.0)
    # Higher VIP -> tighter (more constrained) bounds
    upper_scale = np.clip(score / reaction_vip.max(), 0.3, 1.0)
    reaction_bounds[rxn_id] = {
        "lb": 0.0,
        "ub": 100.0 * upper_scale,
    }

# Simulate FBA predictions
predicted_fluxes = predict_flux_distribution(
    gpr_rules, reaction_bounds, true_flux_df, rng=42
)

# Evaluate per-condition
from scipy.stats import spearmanr, pearsonr

per_condition_results = []
for i, cond in enumerate(true_flux_df.index):
    true_vals = true_flux_df.iloc[i].values
    pred_vals = predicted_fluxes.iloc[i].values
    rho, _ = spearmanr(true_vals, pred_vals)
    r, _ = pearsonr(true_vals, pred_vals)
    nrmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2)) / (
        np.max(true_vals) - np.min(true_vals)
    )
    per_condition_results.append(
        {"condition": cond, "spearman": rho, "pearson": r, "nrmse": nrmse}
    )

per_cond_df = pd.DataFrame(per_condition_results)
print(
    f"  Mean Spearman rho: {per_cond_df['spearman'].mean():.3f} +/- {per_cond_df['spearman'].std():.3f}"
)
print(
    f"  Mean Pearson r:    {per_cond_df['pearson'].mean():.3f} +/- {per_cond_df['pearson'].std():.3f}"
)
print(f"  Mean NRMSE:        {per_cond_df['nrmse'].mean():.3f}")

# -----------------------------------------------------------------------
# Step 5: Summary & Figures
# -----------------------------------------------------------------------
print("\n[5/5] Generating summary figures...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: VIP score distribution
axes[0].barh(
    range(len(reaction_vip)), reaction_vip.values, color="steelblue", edgecolor="white"
)
axes[0].axvline(x=1.0, color="red", linestyle="--", label="VIP=1 threshold")
axes[0].set_yticks(range(len(reaction_vip)))
axes[0].set_yticklabels(reaction_vip.index, fontsize=7)
axes[0].set_xlabel("GPR-VIP Score")
axes[0].set_title("Reaction Importance (GPR-VIP)")
axes[0].legend()
axes[0].invert_yaxis()

# Right: predicted vs true fluxes (pooled)
all_pred = predicted_fluxes.values.flatten()
all_true = true_flux_df.values.flatten()
rho_pooled, _ = spearmanr(all_true, all_pred)
axes[1].scatter(all_true, all_pred, alpha=0.4, s=20, c="steelblue", edgecolors="none")
axes[1].plot(
    [all_true.min(), all_true.max()],
    [all_true.min(), all_true.max()],
    "r--",
    linewidth=1,
    label="y = x",
)
axes[1].set_xlabel("13C-MFA Measured Flux")
axes[1].set_ylabel("ChemoCalib Predicted Flux")
axes[1].set_title(f"Pooled Spearman $\\rho$ = {rho_pooled:.3f}")
axes[1].legend()
axes[1].set_aspect("equal")

plt.tight_layout()
os.makedirs("../output", exist_ok=True)
fig.savefig("../output/tutorial_figures.pdf", bbox_inches="tight", dpi=150)
print(f"  Figure saved to: chemocalib/output/tutorial_figures.pdf")

# Final summary
print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print(
    f"  Pipeline: MB-PLS({K} comp) -> GPR-VIP -> constrained FBA -> {n_conditions} conditions"
)
print(f"  Pooled Spearman    rho = {rho_pooled:.3f}")
print(f"  Per-condition mean rho = {per_cond_df['spearman'].mean():.3f}")
print(f"  Reactions selected:     {len(selected_reactions)}/{len(reaction_vip)}")
print("=" * 70)
print(
    "\nNext: try python examples/example_multiblock_workflow.py for real data pipeline"
)
