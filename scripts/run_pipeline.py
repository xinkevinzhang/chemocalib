#!/usr/bin/env python
"""
====================================================================
ChemoCalib 完整闭环运行脚本
====================================================================
可在轻薄本上直接运行: python scripts/run_pipeline.py

运行:
  python scripts/run_pipeline.py                  # 默认参数 (textbook 模型, 150 对敲除)
  python scripts/run_pipeline.py --model iMM904   # 使用酵母模型 (需联网)
  python scripts/run_pipeline.py --skip-ode       # 跳过 ODE
  python scripts/run_pipeline.py --n-pairs 300    # 更多双敲除

输出在 ./output/ 目录
====================================================================
"""

import os
import sys
import argparse
import time

# 确保项目在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description="ChemoCalib Stage 2 Pipeline — 多块PLS→多约束GEM→主动选样闭环"
    )
    parser.add_argument("--n-samples", type=int, default=100, help="合成数据样本数 (默认100)")
    parser.add_argument("--n-components", type=int, default=5, help="潜变量分量数 (默认5)")
    parser.add_argument("--n-pairs", type=int, default=150, help="双敲除对数 (轻薄本建议100-200)")
    parser.add_argument("--n-select", type=int, default=10, help="主动学习选取数 (默认10)")
    parser.add_argument("--model", type=str, default="textbook", help="GEM模型: textbook 或 iMM904")
    parser.add_argument("--mode", type=str, default="soft", help="约束模式: soft/hard/adaptive")
    parser.add_argument("--skip-ode", action="store_true", help="跳过 ODE 动态层")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")
    args = parser.parse_args()

    print("=" * 70)
    print("  ChemoCalib Stage 2 Pipeline")
    print("  多块 PLS → 多约束 GEM → 虚拟双敲除 → 主动选样 → ODE")
    print("=" * 70)
    print(f"  配置: 样本={args.n_samples}, 潜变量={args.n_components}")
    print(f"        GEM={args.model}, 双敲除对数={args.n_pairs}")
    print(f"        约束模式={args.mode}, 主动选取={args.n_select}")
    print(f"        ODE={'跳过' if args.skip_ode else '运行'}")
    print("=" * 70)

    overall_start = time.time()

    # ============================================================
    # Step 1: 多块 PLS
    # ============================================================
    print("\n" + "=" * 50)
    print("[Step 1/5] 训练多块 PLS 模型...")
    print("=" * 50)

    from chemocalib.models.mbpls import MultiBlockPLS, generate_toy_multiblock_data

    blocks, y, feature_names = generate_toy_multiblock_data(
        n_samples=args.n_samples,
        n_metabolites=50,
        n_transcripts=200,
        n_proteins=80,
    )
    mbpls = MultiBlockPLS(
        n_components=args.n_components,
        block_names=["代谢组", "转录组", "蛋白组"],
    )
    mbpls.fit(blocks, y)
    print(mbpls.summary())

    # 驱动代谢物
    drivers = mbpls.get_driving_metabolites(block_idx=0, top_k=10)
    print(f"\n  VIP Top-5 驱动代谢物: {list(drivers['indices'][:5])}")

    # ============================================================
    # Step 2: 潜变量 → GEM 约束
    # ============================================================
    print("\n" + "=" * 50)
    print("[Step 2/5] 潜变量 → GEM 约束映射...")
    print("=" * 50)

    from chemocalib.gem.constraints import LatentToConstraint
    from chemocalib.gem.fba import FBASimulator

    sim = FBASimulator(model_name=args.model)
    sim.load_model()
    wt_result = sim.wild_type_fba()
    print(f"  野生型 FBA 生物量: {wt_result['objective_value']:.4f}")

    gem_metabolites = sim.get_exchange_reactions()
    gem_met_ids = [m.replace("EX_", "") for m in gem_metabolites]

    mapper = LatentToConstraint(scaling_mode=args.mode)
    mapper.build_feature_reaction_map(
        feature_names[0],
        gem_met_ids[:len(feature_names[0])],
        vip_scores=mbpls.vip_scores[0],
    )

    latent_mean = mbpls.super_scores.mean(axis=0)
    bounds = mapper.latent_to_bounds(latent_mean, n_components=3)
    fba_result = sim.fba_with_chemometric_constraints(bounds)
    print(f"  约束后生物量: {fba_result['objective_value']:.4f}")
    print(f"  已施加约束: {fba_result['constraints_applied']}/{fba_result['constraints_total']}")

    # ============================================================
    # Step 3: 虚拟双敲除
    # ============================================================
    print("\n" + "=" * 50)
    print("[Step 3/5] 虚拟双敲除实验...")
    print("=" * 50)

    from chemocalib.virtual_experiment.knockout import DoubleKnockoutDesigner

    genes = sim.get_all_genes()
    designer = DoubleKnockoutDesigner(gene_pool=genes, design_strategy="exhaustive")
    designer.generate_pairs(n_pairs=args.n_pairs)
    dko_results = designer.run_virtual_experiments(sim, verbose=True)
    analysis = designer.analyze_results()
    print(f"  完成 {analysis['n_total']} 组双敲除")
    print(f"  致死: {analysis['n_lethal']} | 非致死: {analysis.get('n_nonlethal', '?')}")
    gs = analysis.get("growth_stats", {})
    print(f"  平均生长率: {gs.get('mean', 0):.4f} ± {gs.get('std', 0):.4f}")

    # ============================================================
    # Step 4: 主动学习
    # ============================================================
    print("\n" + "=" * 50)
    print("[Step 4/5] 主动学习 —— 选择最优双敲除候选...")
    print("=" * 50)

    from chemocalib.active_learning.uncertainty import UncertaintySampler

    residuals = mbpls.residual_space(blocks)
    sampler = UncertaintySampler(strategy="hybrid")

    gene_pairs = list(zip(dko_results["gene_a"], dko_results["gene_b"]))
    candidates = sampler.select_double_knockout_candidates(
        all_gene_pairs=gene_pairs,
        pair_features=dko_results[["growth_ratio"]].values,
        residuals=residuals,
        n_select=args.n_select,
        n_pool=min(200, len(gene_pairs)),
    )

    print(f"\n  {'='*40}")
    print(f"  【主动学习推荐】Top {args.n_select} 给合作者真做:")
    print(f"  {'='*40}")
    for _, row in candidates.iterrows():
        print(f"  #{int(row['rank']):2d}  敲除 {row['gene_a']} + {row['gene_b']}  (不确定性: {row['uncertainty']:.4f})")

    # ============================================================
    # Step 5: ODE 动态层
    # ============================================================
    if not args.skip_ode:
        print("\n" + "=" * 50)
        print("[Step 5/5] 糖酵解 ODE 动态模拟...")
        print("=" * 50)

        from chemocalib.dynamic_layer.ode_solver import GlycolysisODE

        ode_model = GlycolysisODE()
        ode_model.calibrate_from_latent(latent_mean, n_component=0)
        ode_result = ode_model.simulate(t_span=(0, 50), n_points=200)

        ss = ode_model.steady_state()
        if ss is not None:
            print(f"  解析稳态 G6P: {ss[0]:.4f}, FBP: {ss[1]:.4f}, PYR: {ss[2]:.4f}")
        print(f"  ODE 终点 G6P: {ode_result['G6P'][-1]:.4f}")
        kcat = ode_model.extract_kcat_proxies()
        print(f"  kcat 代理: {kcat}")

    # ============================================================
    # 保存结果
    # ============================================================
    os.makedirs(args.output_dir, exist_ok=True)
    dko_results.to_csv(os.path.join(args.output_dir, "double_knockout_results.csv"), index=False)
    candidates.to_csv(os.path.join(args.output_dir, "active_learning_candidates.csv"), index=False)

    elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"  Pipeline 完成! 总耗时: {elapsed:.1f}s")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    print(f"  研究身份: 我用化学计量学把多组学定量校准进代谢约束的那一层")
    print("=" * 70)


if __name__ == "__main__":
    main()
