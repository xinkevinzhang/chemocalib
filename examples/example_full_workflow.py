#!/usr/bin/env python
"""ChemoCalib Multi-Block Workflow with Benchmark Comparison
====================================================================
Full pipeline on real multi-omics data:
  Multi-block PLS -> GPR-VIP feature selection -> constrained FBA on iJO1366.
Comparison against pFBA, E-Flux, E-Flux2, MOMENT, SPOT baselines.

Input: Kim 2016 S2 Dataset (download from PLOS ONE supplementary)
Output: per-condition Spearman/Pearson/NRMSE metrics for all methods.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.gem.gpr_vip import compute_gpr_vip
from chemocalib.gem.constrained_fba import apply_vip_constraints_to_model
from chemocalib.data.fluxome import benchmark_expression_methods
from chemocalib.data.loader import load_kim2016_dataset


def main():
    """Run full multi-block workload on Kim 2016 dataset."""
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    DATA_FILE = os.path.join(
        os.path.dirname(__file__), "..", "data", "kim2016_S2_Dataset.xls"
    )
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Load Kim 2016 dataset
    # -------------------------------------------------------------------
    print("Loading Kim 2016 curated dataset...")
    if not os.path.exists(DATA_FILE):
        print(f"  [WARNING] Dataset not found at {DATA_FILE}")
        print("  Download from: https://doi.org/10.1371/journal.pone.0157101.s002")
        print("  Place in chemocalib/data/ and re-run.")
        return

    data = load_kim2016_dataset(DATA_FILE)
    for k in [
        "transcriptome",
        "flux_measurements",
        "n_conditions",
        "n_reactions",
        "condition_names",
    ]:
        if k in data:
            val = data[k]
            if hasattr(val, "shape"):
                print(f"  {k}: {val.shape}")
            else:
                print(f"  {k}: {val}")

    # -------------------------------------------------------------------
    # 2. MB-PLS decomposition (transcriptome + metabolome blocks)
    # -------------------------------------------------------------------
    print("\nRunning Multi-Block PLS...")
    blocks = [data["transcriptome"]]
    if "metabolome" in data and data["metabolome"] is not None:
        blocks.append(data["metabolome"])
    Y = np.ones((data["n_conditions"], 1))

    mbpls = MultiBlockPLS(n_components=3, scale=True)
    mbpls.fit(blocks, Y)
    print(f"  K={mbpls.n_components} components extracted")
    for bi, var in enumerate(mbpls.cumulative_variance_):
        print(f"  Block {bi}: {var.sum():.1%} variance explained")

    # -------------------------------------------------------------------
    # 3. GPR-VIP reaction importance
    # -------------------------------------------------------------------
    print("\nComputing GPR-VIP scores...")
    vip_scores = compute_gpr_vip(mbpls, block_idx=0)
    print(f"  VIP scores: {len(vip_scores)} genes, mean={np.mean(vip_scores):.3f}")

    # -------------------------------------------------------------------
    # 4. Constrained FBA
    # -------------------------------------------------------------------
    print("\nApplying VIP-derived constraints to iJO1366...")
    # This loads iJO1366 and applies VIP bounds
    results = apply_vip_constraints_to_model(
        mbpls=mbpls,
        vip_scores=vip_scores,
        model_id="iJO1366",
        conditions=data["condition_names"],
        flux_data=data["flux_measurements"],
    )

    # -------------------------------------------------------------------
    # 5. Benchmark against expression-only methods
    # -------------------------------------------------------------------
    print("\nComparing against expression-only baselines...")
    baselines = benchmark_expression_methods(
        expression=data["transcriptome"],
        flux_measurements=data["flux_measurements"],
        methods=["pFBA", "E-Flux", "E-Flux2", "MOMENT", "SPOT"],
    )

    # -------------------------------------------------------------------
    # 6. Generate summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    all_methods = {"ChemoCalib": results}
    all_methods.update(baselines)

    summary_rows = []
    for method, res in all_methods.items():
        rho_vals = [
            v["spearman"]
            for v in res.values()
            if not np.isnan(v.get("spearman", np.nan))
        ]
        r_vals = [
            v["pearson"] for v in res.values() if not np.isnan(v.get("pearson", np.nan))
        ]
        nrmse_vals = [
            v["nrmse"] for v in res.values() if not np.isnan(v.get("nrmse", np.nan))
        ]

        summary_rows.append(
            {
                "Method": method,
                "Spearman_rho_mean": np.mean(rho_vals) if rho_vals else np.nan,
                "Spearman_rho_std": np.std(rho_vals) if len(rho_vals) > 1 else 0,
                "Pearson_r_mean": np.mean(r_vals) if r_vals else np.nan,
                "NRMSE_mean": np.mean(nrmse_vals) if nrmse_vals else np.nan,
                "N_conditions": len(rho_vals),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    summary.to_csv(os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}/benchmark_summary.csv")


if __name__ == "__main__":
    main()
