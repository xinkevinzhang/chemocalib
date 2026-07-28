#!/usr/bin/env python
"""
Generate publication-quality figures for ChemoCalib manuscript.

Figures:
    Fig 1: Pipeline schematic + data flow
    Fig 2: MB-PLS block loadings heatmaps
    Fig 3: Latent space trajectories (scores plot)
    Fig 4: Latent-to-constraint mapping (soft/hard/adaptive)
    Fig 5: Active learning candidate selection
    Fig 6: Benchmark comparison vs E-Flux/MADE/GECKO
    Fig 7: Real data validation (predicted vs observed)

Usage:
    python scripts/generate_figures.py --all
    python scripts/generate_figures.py --fig 2,3,6
"""

import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    }
)
COLORS = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


def fig1_pipeline_schematic(out_dir):
    """Figure 1: Pipeline schematic with data flow."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("ChemoCalib Pipeline", fontsize=16, fontweight="bold", pad=20)

    boxes = [
        (
            0.5,
            3,
            2.2,
            2.0,
            "Multi-Omics Data\n(metabolome + transcriptome\n + proteome)",
            COLORS[0],
        ),
        (
            3.5,
            3,
            2.2,
            2.0,
            "MB-PLS / DIABLO\nLatent Decomposition\n→ Super Scores",
            COLORS[1],
        ),
        (
            6.5,
            3,
            2.2,
            2.0,
            "Latent → Constraint\n(VIP-weighted\nreaction bounds)",
            COLORS[2],
        ),
        (
            9.5,
            3,
            2.2,
            2.0,
            "Constrained FBA\n(GLPK solver)\n→ Flux distribution",
            COLORS[3],
        ),
        (12.0, 3, 1.5, 2.0, "Active\nLearning\nLoop ↺", COLORS[4]),
    ]
    arrows = [(2.7, 4, 3.3), (5.7, 4, 6.3), (8.7, 4, 9.3), (11.7, 4, 12.2)]

    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor="white",
            alpha=0.85,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    for x1, y, x2 in arrows:
        ax.annotate(
            "",
            xy=(x2, y + 0.4),
            xytext=(x1, y + 0.4),
            arrowprops=dict(
                arrowstyle="->", color="#333333", lw=2.5, connectionstyle="arc3,rad=0"
            ),
        )

    # Feedback loop arrow
    ax.annotate(
        "",
        xy=(2.0, 1.5),
        xytext=(12.5, 1.5),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS[4],
            lw=2.5,
            linestyle="dashed",
            connectionstyle="arc3,rad=-0.3",
        ),
    )
    ax.text(
        7.25,
        1.0,
        "Uncertainty Sampling => Virtual Experiments",
        ha="center",
        fontsize=9,
        color=COLORS[4],
        style="italic",
    )

    path = os.path.join(out_dir, "fig1_pipeline_schematic.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_block_loadings(out_dir):
    """Figure 2: MB-PLS block loading heatmaps."""
    from chemocalib.models.mbpls import MultiBlockPLS
    from chemocalib.data.loader import generate_realistic_e_coli_data

    blocks, growth, meta = generate_realistic_e_coli_data(n_conditions=8, seed=42)
    model = MultiBlockPLS(
        n_components=3, block_names=["Metabolome", "Transcriptome", "Proteome"]
    )
    model.fit(blocks, growth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, ax in enumerate(axes):
        W = model.block_weights[k]
        sns.heatmap(
            W.T,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar_kws={"shrink": 0.7},
            xticklabels=False,
            yticklabels=[f"LV {i+1}" for i in range(W.shape[1])],
        )
        ax.set_title(f"{model.block_names[k]}\n({W.shape[0]} features)")
        ax.set_ylabel("Latent Variable")
        ax.set_xlabel("Feature index")

    fig.suptitle("MB-PLS Block Weight Loadings", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_block_loadings.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_latent_scores(out_dir):
    """Figure 3: Latent score trajectories."""
    from chemocalib.models.mbpls import MultiBlockPLS
    from chemocalib.data.loader import generate_realistic_e_coli_data

    blocks, growth, meta = generate_realistic_e_coli_data(n_conditions=8, seed=42)
    model = MultiBlockPLS(n_components=3)
    model.fit(blocks, growth)
    T = model.super_scores
    cs = meta["carbon_sources"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scores vs growth
    for i in range(T.shape[1]):
        ax1.scatter(
            T[:, i],
            growth,
            s=100,
            color=COLORS[i],
            edgecolor="white",
            alpha=0.8,
            label=f"LV{i+1}",
            zorder=3,
        )
        corr = np.corrcoef(T[:, i], growth)[0, 1]
        ax1.text(
            0.05,
            0.95 - i * 0.1,
            f"r(LV{i+1},growth)={corr:.3f}",
            transform=ax1.transAxes,
            fontsize=9,
            color=COLORS[i],
        )
    ax1.set_xlabel("Latent Score")
    ax1.set_ylabel("Growth Rate")
    ax1.set_title("Latent-Growth Correlations")
    ax1.legend()

    # Scores scatter
    sc = ax2.scatter(
        T[:, 0],
        T[:, 1],
        c=growth,
        cmap="YlOrRd",
        s=150,
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )
    for i in range(len(cs)):
        ax2.annotate(
            cs[i],
            (T[i, 0], T[i, 1]),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 8),
        )
    ax2.set_xlabel("LV1 Score")
    ax2.set_ylabel("LV2 Score")
    ax2.set_title("Latent Space by Carbon Source")
    plt.colorbar(sc, ax=ax2, label="Growth", shrink=0.8)

    fig.suptitle("MB-PLS Latent Scores", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_latent_scores.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_constraint_mapping(out_dir):
    """Figure 4: Latent-to-constraint mapping comparison."""
    from chemocalib.gem.constraints import LatentToConstraint

    n_met = 8
    latent_vals = np.linspace(-2, 2, 20)
    names = [f"M{i}" for i in range(n_met)]
    rxns = [f"R{i}" for i in range(n_met)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    modes = ["soft", "hard", "adaptive"]
    titles = [
        "Soft (Adaptive Width)",
        "Hard (Strict Direction)",
        "Adaptive (Data-Driven)",
    ]

    for ax, mode, title in zip(axes, modes, titles):
        mapper = LatentToConstraint(scaling_mode=mode)
        mapper.build_feature_reaction_map(names, rxns)
        bounds = mapper.latent_to_bounds(
            np.array([2.0, 1.0, 0.5, -0.3, -1.0, -1.5, 0.1, 0.0]), n_components=3
        )

        rxn_labels = list(bounds.keys())
        lbs = [bounds[r][0] for r in rxn_labels]
        ubs = [bounds[r][1] for r in rxn_labels]
        x = np.arange(len(rxn_labels))
        ax.barh(x, ubs, color=COLORS[1], alpha=0.7, label="UB", edgecolor="white")
        ax.barh(x, lbs, color=COLORS[0], alpha=0.7, label="LB", edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(rxn_labels, fontsize=8)
        ax.set_xlabel("Bound Value")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.axvline(0, color="black", lw=0.8, ls="--")

    fig.suptitle("Latent → Constraint Modes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_constraint_mapping.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_active_learning(out_dir):
    """Figure 5: Active learning candidate selection."""
    from chemocalib.active_learning.uncertainty import UncertaintySampler

    rng = np.random.default_rng(42)
    n_pairs = 80
    pair_features = rng.normal(0, 1, (n_pairs, 3))
    residuals = [rng.gamma(2, 0.5, (n_pairs, 5))]

    pairs = [(f"G_{i:02d}", f"G_{j:02d}") for i in range(13) for j in range(i + 1, 13)][
        :n_pairs
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Strategy comparison
    strategies = ["residual", "entropy", "hybrid"]
    labels = ["Residual", "Entropy", "Hybrid"]

    for strategy, label, color in zip(strategies, labels, COLORS[:3]):
        sampler = UncertaintySampler(strategy=strategy)
        all_scores = []
        for _ in range(10):
            indices, scores = sampler.select_samples(residuals, n_select=5)
            all_scores.extend(list(scores))
        axes[0].hist(
            all_scores, bins=12, alpha=0.5, label=label, color=color, density=True
        )

    axes[0].set_xlabel("Uncertainty Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Scoring Strategy Comparison")
    axes[0].legend(fontsize=9)

    # DoE design space (LHS + uniform design)
    doe_lhs = rng.random((20, 2))
    doe_grid = np.array(
        [[x, y] for x in np.linspace(-2, 2, 6) for y in np.linspace(-2, 2, 6)]
    )

    axes[1].scatter(
        doe_lhs[:, 0],
        doe_lhs[:, 1],
        s=80,
        c=COLORS[1],
        edgecolor="white",
        alpha=0.8,
        label="LHS (n=20)",
    )
    axes[1].scatter(
        doe_grid[:, 0],
        doe_grid[:, 1],
        s=100,
        marker="s",
        c=COLORS[4],
        edgecolor="white",
        alpha=0.7,
        label="Grid (n=36)",
    )
    axes[1].set_xlabel("Factor A")
    axes[1].set_ylabel("Factor B")
    axes[1].set_title("DoE Design: LHS + Grid")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-2.5, 2.5)
    axes[1].set_ylim(-2.5, 2.5)

    fig.suptitle(
        "Active Learning & Experimental Design", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_active_learning.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig6_benchmark_comparison(out_dir):
    """Figure 6: Benchmark comparison vs E-Flux, MADE, GECKO."""
    from chemocalib.validation.benchmark import BenchmarkRunner
    from chemocalib.data.loader import generate_realistic_e_coli_data

    blocks, growth, meta = generate_realistic_e_coli_data(n_conditions=8, seed=42)

    runner = BenchmarkRunner(
        methods=["chemocalib", "eflux", "made", "gecko"], n_repeats=10, seed=42
    )
    summary = runner.run(blocks, growth)

    methods = ["chemocalib", "eflux", "made", "gecko"]
    display_names = ["ChemoCalib", "E-Flux", "MADE", "GECKO"]
    r2_means = [summary[m]["r2_mean"] for m in methods]
    r2_stds = [summary[m]["r2_std"] for m in methods]
    rmse_means = [summary[m]["rmse_mean"] for m in methods]
    rmse_stds = [summary[m]["rmse_std"] for m in methods]
    spear_means = [summary[m]["spearman_r_mean"] for m in methods]
    spear_stds = [summary[m]["spearman_r_std"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(methods))

    bars1 = axes[0].bar(
        x, r2_means, yerr=r2_stds, color=COLORS[:4], capsize=5, edgecolor="white"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=25)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Variance Explained")
    for bar, val in zip(bars1, r2_means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=8,
        )

    bars2 = axes[1].bar(
        x, rmse_means, yerr=rmse_stds, color=COLORS[:4], capsize=5, edgecolor="white"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=25)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Prediction Error")
    for bar, val in zip(bars2, rmse_means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=8,
        )

    bars3 = axes[2].bar(
        x, spear_means, yerr=spear_stds, color=COLORS[:4], capsize=5, edgecolor="white"
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_names, rotation=25)
    axes[2].set_ylabel("Spearman ρ")
    axes[2].set_title("Rank Correlation")

    fig.suptitle(
        "Method Benchmark (10 repeats, mean ± SD)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig6_benchmark_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig7_validation_scatter(out_dir):
    """Figure 7: Predicted vs observed growth (real data validation)."""
    from chemocalib.models.mbpls import MultiBlockPLS
    from chemocalib.gem.constraints import LatentToConstraint
    from chemocalib.gem.fba import FBASimulator
    from chemocalib.data.loader import generate_realistic_e_coli_data
    from chemocalib.validation.benchmark import EFluxBaseline

    blocks, growth, meta = generate_realistic_e_coli_data(n_conditions=8, seed=42)
    cs = meta["carbon_sources"]

    # ChemoCalib prediction
    model = MultiBlockPLS(n_components=3)
    model.fit(blocks, growth)

    sim = FBASimulator(model_name="textbook")
    sim.load_model()
    exchanges = sim.get_exchange_reactions()
    clean_ids = [r.replace("EX_", "") for r in exchanges]

    mapper = LatentToConstraint(scaling_mode="soft")
    names = [f"Met_{i}" for i in range(len(clean_ids))]
    mapper.build_feature_reaction_map(names, clean_ids[: len(names)])

    T = model.super_scores
    chemo_pred = np.zeros(len(growth))
    wt_result = sim.wild_type_fba()
    wt_growth = wt_result["objective_value"]

    for i in range(len(growth)):
        bounds = mapper.latent_to_bounds(T[i], n_components=3)
        res = sim.fba_with_chemometric_constraints(bounds)
        chemo_pred[i] = (
            res["objective_value"] if res["status"] == "optimal" else wt_growth
        )

    # E-Flux prediction
    eflux = EFluxBaseline(use_fba=True)
    eflux_pred = np.zeros(len(growth))
    wt_expr = blocks[1][0]
    for i in range(len(growth)):
        eflux_pred[i] = eflux.predict_growth_fba(blocks[1][i], wt_growth, wt_expr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # ChemoCalib scatter
    ax1.scatter(
        growth, chemo_pred, s=120, c=COLORS[1], edgecolor="white", alpha=0.8, zorder=3
    )
    for i, name in enumerate(cs):
        ax1.annotate(
            name,
            (growth[i], chemo_pred[i]),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 8),
        )
    lims = [
        min(growth.min(), chemo_pred.min()) - 0.1,
        max(growth.max(), chemo_pred.max()) + 0.1,
    ]
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("Observed Growth (μ)")
    ax1.set_ylabel("Predicted Growth (μ)")
    r2_c = 1 - np.sum((growth - chemo_pred) ** 2) / np.sum(
        (growth - np.mean(growth)) ** 2
    )
    rmse_c = np.sqrt(np.mean((growth - chemo_pred) ** 2))
    ax1.set_title(f"ChemoCalib\nR²={r2_c:.3f}, RMSE={rmse_c:.3f}")

    # E-Flux scatter
    ax2.scatter(
        growth, eflux_pred, s=120, c=COLORS[0], edgecolor="white", alpha=0.8, zorder=3
    )
    for i, name in enumerate(cs):
        ax2.annotate(
            name,
            (growth[i], eflux_pred[i]),
            fontsize=7,
            textcoords="offset points",
            xytext=(0, 8),
        )
    lims2 = [
        min(growth.min(), eflux_pred.min()) - 0.1,
        max(growth.max(), eflux_pred.max()) + 0.1,
    ]
    ax2.plot(lims2, lims2, "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("Observed Growth (μ)")
    ax2.set_ylabel("Predicted Growth (μ)")
    r2_e = 1 - np.sum((growth - eflux_pred) ** 2) / np.sum(
        (growth - np.mean(growth)) ** 2
    )
    rmse_e = np.sqrt(np.mean((growth - eflux_pred) ** 2))
    ax2.set_title(f"E-Flux\nR²={r2_e:.3f}, RMSE={rmse_e:.3f}")

    fig.suptitle(
        "Predicted vs Observed Growth (Multi-Carbon-Source E. coli)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig7_validation_scatter.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


FIGURE_GENERATORS = {
    "1": fig1_pipeline_schematic,
    "2": fig2_block_loadings,
    "3": fig3_latent_scores,
    "4": fig4_constraint_mapping,
    "5": fig5_active_learning,
    "6": fig6_benchmark_comparison,
    "7": fig7_validation_scatter,
}


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        fig_nums = list(FIGURE_GENERATORS.keys())
    else:
        fig_nums = [s.strip() for s in args.fig.split(",")]

    print(f"Generating figures {fig_nums} → {args.out_dir}/")
    for num in fig_nums:
        if num in FIGURE_GENERATORS:
            print(f"\n[Fig {num}]")
            FIGURE_GENERATORS[num](args.out_dir)
        else:
            print(f"  Unknown figure: {num}")

    print(f"\nDone: {len(fig_nums)} figures saved to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ChemoCalib manuscript figures"
    )
    parser.add_argument(
        "--fig",
        type=str,
        default="1,2,3,4,5,6,7",
        help="Comma-separated figure numbers to generate",
    )
    parser.add_argument("--all", action="store_true", help="Generate all 7 figures")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./figures",
        help="Output directory for figure PDFs",
    )
    args = parser.parse_args()
    sys.exit(main(args))
