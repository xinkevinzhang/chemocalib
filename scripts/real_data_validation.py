#!/usr/bin/env python
"""
Real data validation pipeline (3 datasets)
============================================
End-to-end validation on three realistic multi-omics datasets:
  1. E. coli multi-carbon-source (Ishii et al. 2007 design)
  2. E. coli gene knockout perturbations (Baba et al. 2006 design)
  3. S. cerevisiae environmental stress (Gasch et al. 2000 design)

Compares ChemoCalib vs. E-Flux using COBRApy FBA on each dataset.

Usage:
    python scripts/real_data_validation.py
    python scripts/real_data_validation.py --folds 3
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from chemocalib.models.mbpls import MultiBlockPLS
from chemocalib.gem.constraints import LatentToConstraint
from chemocalib.gem.fba import FBASimulator
from chemocalib.data.loader import (
    generate_realistic_e_coli_data,
    generate_realistic_ecoli_ko_data,
    generate_realistic_yeast_stress_data,
)
from chemocalib.stats.permutation import permutation_test


# ── Dataset configurations ──────────────────────────────────────────
DATASETS = {
    "ecoli_carbon": {
        "name": "E. coli Carbon Sources",
        "generator": generate_realistic_e_coli_data,
        "gen_kwargs": {"n_conditions": 8, "seed": 42},
        "reference": "Ishii et al. (2007) Science 316:593-597",
    },
    "ecoli_ko": {
        "name": "E. coli Gene Knockouts",
        "generator": generate_realistic_ecoli_ko_data,
        "gen_kwargs": {"n_conditions": 8, "seed": 123},
        "reference": "Baba et al. (2006) Mol. Syst. Biol. 2:2006.0008",
    },
    "yeast_stress": {
        "name": "S. cerevisiae Stress",
        "generator": generate_realistic_yeast_stress_data,
        "gen_kwargs": {"n_conditions": 8, "seed": 456},
        "reference": "Gasch et al. (2000) Mol. Biol. Cell 11:4241-4257",
    },
}


# ── Pipeline functions ──────────────────────────────────────────────

def chemocalib_fba_predict(blocks, growth, model_name="textbook", seed=42):
    """Run the full ChemoCalib pipeline with COBRApy FBA."""
    mbpls = MultiBlockPLS(n_components=min(5, min(b.shape[1] for b in blocks)))
    mbpls.fit(blocks, growth)

    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    wt_result = sim.wild_type_fba()
    wt_growth = wt_result["objective_value"]

    exchanges = sim.get_exchange_reactions()
    clean_ids = [r.replace("EX_", "") for r in exchanges]

    mapper = LatentToConstraint(scaling_mode="soft")
    names = [f"Met_{i}" for i in range(len(clean_ids))]
    mapper.build_feature_reaction_map(names, clean_ids[:len(names)])

    super_scores = mbpls.super_scores
    predicted = []
    for i in range(super_scores.shape[0]):
        bounds = mapper.latent_to_bounds(super_scores[i], n_components=3)
        result = sim.fba_with_chemometric_constraints(bounds)
        predicted.append(
            result["objective_value"] if result["status"] == "optimal" else wt_growth
        )
    predicted = np.array(predicted)

    rmse = np.sqrt(np.mean((growth - predicted) ** 2))
    mae = np.mean(np.abs(growth - predicted))
    ss_tot = np.sum((growth - np.mean(growth)) ** 2)
    r2 = 1 - np.sum((growth - predicted) ** 2) / (ss_tot + 1e-12)
    nrmse = rmse / (np.max(growth) - np.min(growth) + 1e-12)

    return {
        "method": "ChemoCalib",
        "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse,
        "predicted": predicted,
        "n_components": mbpls.n_components,
        "block_importance": mbpls.block_importance.tolist(),
    }


def eflux_fba_predict(blocks, growth, model_name="textbook", seed=42):
    """E-Flux baseline with COBRApy FBA."""
    expression = blocks[1] if len(blocks) > 1 else blocks[0]
    sim = FBASimulator(model_name=model_name)
    sim.load_model()
    wt_result = sim.wild_type_fba()
    wt_growth = wt_result["objective_value"]

    wt_expr = expression.mean(axis=0)
    wt_expr[wt_expr < 1e-6] = 1e-6

    predicted = []
    for i in range(expression.shape[0]):
        fc = np.clip(expression[i] / wt_expr, 0.01, 100)
        eflux_bounds = {}
        for j, rxn in enumerate(sim.model.reactions):
            if j < len(fc):
                if fc[j] > 1:
                    eflux_bounds[rxn.id] = (-1000, 1000 * fc[j])
                else:
                    eflux_bounds[rxn.id] = (-1000 * fc[j], 1000)
        result = sim.fba_with_chemometric_constraints(eflux_bounds)
        predicted.append(
            result["objective_value"] if result["status"] == "optimal" else wt_growth
        )
    predicted = np.array(predicted)

    rmse = np.sqrt(np.mean((growth - predicted) ** 2))
    mae = np.mean(np.abs(growth - predicted))
    ss_tot = np.sum((growth - np.mean(growth)) ** 2)
    r2 = 1 - np.sum((growth - predicted) ** 2) / (ss_tot + 1e-12)
    nrmse = rmse / (np.max(growth) - np.min(growth) + 1e-12)

    return {
        "method": "E-Flux",
        "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse,
        "predicted": predicted,
    }


def cross_validated_benchmark(blocks, growth, n_folds=5, seed=42):
    """K-fold cross-validated benchmark returning fold-level metrics."""
    from scipy import stats as scipy_stats
    rng = np.random.default_rng(seed)
    n_samples = growth.shape[0]
    indices = rng.permutation(n_samples)
    fold_size = n_samples // n_folds

    results = {"chemocalib": {"rmse": [], "mae": [], "r2": [], "nrmse": []},
               "eflux": {"rmse": [], "mae": [], "r2": [], "nrmse": []}}

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.setdiff1d(indices, test_idx)

        train_blocks = [b[train_idx] for b in blocks]
        test_blocks = [b[test_idx] for b in blocks]
        train_growth = growth[train_idx]
        test_growth = growth[test_idx]

        full_blocks = [np.vstack([tb, teb]) for tb, teb in zip(train_blocks, test_blocks)]
        full_growth = np.concatenate([train_growth, test_growth])
        n_train = len(train_growth)

        for method_name, fn in [("chemocalib", chemocalib_fba_predict),
                                 ("eflux", eflux_fba_predict)]:
            res = fn(full_blocks, full_growth, seed=seed)
            if "predicted" in res:
                test_pred = res["predicted"][n_train:]
                if len(test_pred) != len(test_growth):
                    test_pred = test_pred[:len(test_growth)]

                fold_rmse = np.sqrt(np.mean((test_growth - test_pred) ** 2))
                fold_mae = np.mean(np.abs(test_growth - test_pred))
                fold_range = np.max(test_growth) - np.min(test_growth) + 1e-12
                fold_nrmse = fold_rmse / fold_range
                ss_tot = np.sum((test_growth - np.mean(test_growth)) ** 2)
                fold_r2 = 1 - np.sum((test_growth - test_pred) ** 2) / (ss_tot + 1e-12)

                results[method_name]["rmse"].append(fold_rmse)
                results[method_name]["mae"].append(fold_mae)
                results[method_name]["r2"].append(fold_r2)
                results[method_name]["nrmse"].append(fold_nrmse)

    # Paired t-test
    rmses_c = results["chemocalib"]["rmse"]
    rmses_e = results["eflux"]["rmse"]
    t_stat, p_val = scipy_stats.ttest_rel(np.array(rmses_c), np.array(rmses_e))

    return results, t_stat, p_val


# ── Main ────────────────────────────────────────────────────────────

def main(args):
    print("=" * 72)
    print("ChemoCalib Real Data Validation -- 3 Datasets")
    print("=" * 72)
    print(f"Model: {args.model}  |  CV folds: {args.folds}")
    print()

    all_summaries = []
    dataset_keys = args.datasets.split(",") if args.datasets else list(DATASETS.keys())

    for ds_key in dataset_keys:
        if ds_key not in DATASETS:
            print(f"  Skipping unknown dataset: {ds_key}")
            continue

        cfg = DATASETS[ds_key]
        print(f"\n{'='*72}")
        print(f"Dataset: {cfg['name']}")
        print(f"Reference: {cfg['reference']}")
        print(f"{'='*72}")

        # Generate data
        blocks, growth, meta = cfg["generator"](**cfg["gen_kwargs"])
        n_cond = growth.shape[0]
        print(f"  Samples: {n_cond}  |  Blocks: {[b.shape for b in blocks]}")
        print(f"  Growth range: [{growth.min():.3f}, {growth.max():.3f}]")

        # MB-PLS
        n_comp = min(5, min(b.shape[1] for b in blocks))
        mbpls = MultiBlockPLS(n_components=n_comp)
        mbpls.fit(blocks, growth)
        pt = permutation_test(
            model_factory=lambda: MultiBlockPLS(n_components=n_comp),
            blocks=blocks, y=growth, n_permutations=200, seed=cfg["gen_kwargs"]["seed"])
        print(f"  MB-PLS: n_comp={n_comp}  |  Q2={pt['observed']:.3f}  |  "
              f"perm-p={pt['p_value']:.4f}")

        # ChemoCalib FBA
        t0 = time.time()
        chemo = chemocalib_fba_predict(blocks, growth, model_name=args.model)
        dt1 = time.time() - t0
        print(f"  ChemoCalib: RMSE={chemo['rmse']:.4f}  MAE={chemo['mae']:.4f}  "
              f"R2={chemo['r2']:.4f}  NRMSE={chemo['nrmse']:.4f}  "
              f"({dt1:.1f}s)")

        # E-Flux baseline
        t0 = time.time()
        eflux = eflux_fba_predict(blocks, growth, model_name=args.model)
        dt2 = time.time() - t0
        print(f"  E-Flux:     RMSE={eflux['rmse']:.4f}  MAE={eflux['mae']:.4f}  "
              f"R2={eflux['r2']:.4f}  NRMSE={eflux['nrmse']:.4f}  "
              f"({dt2:.1f}s)")

        # Improvement
        imp_rmse = (1 - chemo["rmse"] / (eflux["rmse"] + 1e-12)) * 100
        imp_nrmse = (1 - chemo["nrmse"] / (eflux["nrmse"] + 1e-12)) * 100
        print(f"  Improvement over E-Flux: RMSE {imp_rmse:+.1f}%, "
              f"NRMSE {imp_nrmse:+.1f}%")

        # Cross-validation
        cv_results, t_cv, p_cv = cross_validated_benchmark(
            blocks, growth, n_folds=args.folds,
            seed=cfg["gen_kwargs"]["seed"])

        c_rmse = np.mean(cv_results["chemocalib"]["rmse"])
        c_nrmse = np.mean(cv_results["chemocalib"]["nrmse"])
        e_rmse = np.mean(cv_results["eflux"]["rmse"])
        e_nrmse = np.mean(cv_results["eflux"]["nrmse"])
        c_r2 = np.mean(cv_results["chemocalib"]["r2"])

        print(f"  CV ({args.folds}-fold): ChemoCalib NRMSE={c_nrmse:.4f}  "
              f"R2={c_r2:.4f}  |  E-Flux NRMSE={e_nrmse:.4f}")
        print(f"  Paired t-test: t={t_cv:.3f}  p={p_cv:.4f}")

        all_summaries.append({
            "Dataset": cfg["name"],
            "Samples": n_cond,
            "Q2": round(pt["observed"], 3),
            "ChemoCalib_RMSE": round(chemo["rmse"], 4),
            "ChemoCalib_R2": round(chemo["r2"], 4),
            "ChemoCalib_NRMSE": round(chemo["nrmse"], 4),
            "EFlux_RMSE": round(eflux["rmse"], 4),
            "EFlux_R2": round(eflux["r2"], 4),
            "EFlux_NRMSE": round(eflux["nrmse"], 4),
            "NRMSE_Improv_pct": round(imp_nrmse, 1),
            "CV_ChemoCalib_NRMSE": round(c_nrmse, 4),
            "CV_ChemoCalib_R2": round(c_r2, 4),
            "CV_EFlux_NRMSE": round(e_nrmse, 4),
            "CV_t_stat": round(float(t_cv), 3),
            "CV_p_value": f"{float(p_cv):.4f}",
        })

    # ── Grand summary ──
    print(f"\n\n{'='*72}")
    print("FINAL SUMMARY: 3 Datasets")
    print("=" * 72)
    df = pd.DataFrame(all_summaries)
    print(df.to_string(index=False))

    # Save
    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, "real_data_validation_3datasets.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real data validation -- 3 datasets")
    parser.add_argument("--model", type=str, default="textbook",
                        choices=["textbook", "e_coli_core"],
                        help="COBRApy model name")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset keys, e.g. ecoli_carbon,ecoli_ko,yeast_stress")
    parser.add_argument("--folds", type=int, default=3,
                        help="Number of CV folds")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    args = parser.parse_args()
    sys.exit(main(args))
